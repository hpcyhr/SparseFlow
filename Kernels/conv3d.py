"""
SparseFlow Kernels/conv3d.py - v3 unified Triton sparse Conv3d.

Two-stage Triton kernel following the canonical SparseFlow method.

Changes from v2.0:
  [A.1]  Legacy compat kwargs removed.
  [B.4]  bias_ptr placeholder is fp32 device tensor.
  [C.3]  Output tensor pre-filled with bias so TILE_ZERO tiles are correct.
  [C.7]  Single popcount + single GPU->CPU sync in need_stats path.
  [B.3]  Zero-tile early-return runs regardless of need_stats.

Layouts:
  - Input  x permuted NCDHW -> NDHWC
  - Weight permuted [C_out, C_in, KD, KH, KW] -> [C_out, KD*KH*KW*C_in]
  - Output written to NCDHW (PyTorch standard)

Supported sparse-path configurations:
  - groups == 1, dilation == 1
  - kernel_size in {1, 3} (cube)
  - stride in {1, 2}
  - padding consistent with kernel_size and stride

Maturity: main_path.

Change-log
----------
  - v3: migrated from Triton prescan kernel to shared PyTorch three-stage
    prescan in Kernels/_prescan_common.py. Triton _prescan_conv3d_kernel removed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as Fn
import triton
import triton.language as tl
from triton import autotune, Config

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Utils.config import PRESCAN_ACTIVITY_EPS, SPARSE_DENSE_RATIO_THRESHOLD
from Kernels._prescan_common import build_rf_prescan_metadata
from Utils.sparse_helpers import (
    TILE_ZERO,
    TILE_SPARSE,
    TILE_DENSEISH,
    choose_group_size,
    popcount_buf,
)

FALLBACK_RATIO = SPARSE_DENSE_RATIO_THRESHOLD


def _select_3d_tile_sizes(d_out: int, h_out: int, w_out: int):
    voxels = d_out * h_out * w_out
    if voxels >= 4096:
        return 2, 4, 8   # BLOCK_M = 64
    if voxels >= 1024:
        return 2, 4, 4   # BLOCK_M = 32
    return 2, 2, 4       # BLOCK_M = 16


def _is_supported_sparse_pattern(kernel_size, stride, padding, dilation, groups):
    if groups != 1 or dilation != 1:
        return False
    if kernel_size == 1 and stride in (1, 2) and padding == 0:
        return True
    if kernel_size == 3 and stride in (1, 2) and padding == 1:
        return True
    return False


# ===========================================================================
# Stage 2: compute kernel
# ===========================================================================

_CONV3D_CONFIGS = [
    Config({"BLOCK_N_OUT": 32, "DENSE_K": 32}, num_warps=4, num_stages=1),
    Config({"BLOCK_N_OUT": 64, "DENSE_K": 32}, num_warps=4, num_stages=1),
    Config({"BLOCK_N_OUT": 64, "DENSE_K": 64}, num_warps=8, num_stages=1),
]


@autotune(
    configs=_CONV3D_CONFIGS,
    key=["C_IN", "C_OUT", "D_OUT", "H_OUT", "W_OUT", "BLOCK_M", "KD", "STRIDE"],
)
@triton.jit
def _sparse_conv3d_kernel(
    x_ndhwc_ptr, w_kc_ptr, bias_ptr, y_ptr,
    ag_mask_ptr, tile_class_ptr, active_tile_ids_ptr,
    N_val,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    D_IN: tl.constexpr, H_IN: tl.constexpr, W_IN: tl.constexpr,
    D_OUT: tl.constexpr, H_OUT: tl.constexpr, W_OUT: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PADDING: tl.constexpr,
    GD: tl.constexpr, GH: tl.constexpr, GW: tl.constexpr,
    BD: tl.constexpr, BH: tl.constexpr, BW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, NUM_GROUPS: tl.constexpr,
    HAS_BIAS: tl.constexpr, USE_TILE_IDS: tl.constexpr,
    BLOCK_N_OUT: tl.constexpr, DENSE_K: tl.constexpr,
):
    pid_tile = tl.program_id(0)
    pid_n_out = tl.program_id(1)

    if USE_TILE_IDS:
        tile_id = tl.load(active_tile_ids_ptr + pid_tile)
    else:
        tile_id = pid_tile

    tiles_per_n = GD * GH * GW
    n_idx = tile_id // tiles_per_n
    rem = tile_id % tiles_per_n
    if n_idx >= N_val:
        return

    off1 = tl.arange(0, 1)
    tc_t = tl.load(tile_class_ptr + tile_id + off1)
    tile_cls = tl.sum(tc_t)

    # ZERO tile: y was pre-filled with bias on host, nothing to do.
    if tile_cls == TILE_ZERO:
        return

    gd = rem // (GH * GW)
    gh = (rem // GW) % GH
    gw = rem % GW
    d_base = gd * BD
    h_base = gh * BH
    w_base = gw * BW

    m_local = tl.arange(0, BLOCK_M)
    d_local = m_local // (BH * BW)
    h_local = (m_local // BW) % BH
    w_local = m_local % BW

    out_d = d_base + d_local
    out_h = h_base + h_local
    out_w = w_base + w_local
    m_mask = (out_d < D_OUT) & (out_h < H_OUT) & (out_w < W_OUT)

    offs_n_out = pid_n_out * BLOCK_N_OUT + tl.arange(0, BLOCK_N_OUT)
    n_out_mask = offs_n_out < C_OUT

    acc = tl.zeros([BLOCK_M, BLOCK_N_OUT], dtype=tl.float32)

    HWC = H_IN * W_IN * C_IN
    WC = W_IN * C_IN
    KCC = KD * KH * KW * C_IN
    n_offset = n_idx * D_IN * HWC

    if tile_cls == TILE_DENSEISH:
        for kd in tl.static_range(KD):
            for kh in tl.static_range(KH):
                for kw in tl.static_range(KW):
                    in_d = out_d * STRIDE + kd - PADDING
                    in_h = out_h * STRIDE + kh - PADDING
                    in_w = out_w * STRIDE + kw - PADDING
                    dhw_ok = (m_mask
                              & (in_d >= 0) & (in_d < D_IN)
                              & (in_h >= 0) & (in_h < H_IN)
                              & (in_w >= 0) & (in_w < W_IN))
                    safe_d = tl.minimum(tl.maximum(in_d, 0), D_IN - 1)
                    safe_h = tl.minimum(tl.maximum(in_h, 0), H_IN - 1)
                    safe_w = tl.minimum(tl.maximum(in_w, 0), W_IN - 1)

                    x_pix = n_offset + safe_d * HWC + safe_h * WC + safe_w * C_IN
                    w_k_base = (kd * KH * KW + kh * KW + kw) * C_IN

                    for cin_base in range(0, C_IN, DENSE_K):
                        offs_k = cin_base + tl.arange(0, DENSE_K)
                        k_mask = offs_k < C_IN
                        x_addrs = x_pix[:, None] + offs_k[None, :]
                        x_t = tl.load(x_ndhwc_ptr + x_addrs,
                                      mask=dhw_ok[:, None] & k_mask[None, :], other=0.0).to(tl.float16)
                        w_addrs = offs_n_out[None, :] * KCC + w_k_base + offs_k[:, None]
                        w_t = tl.load(w_kc_ptr + w_addrs,
                                      mask=k_mask[:, None] & n_out_mask[None, :], other=0.0).to(tl.float16)
                        acc += tl.dot(x_t, w_t)
    else:
        ag_t = tl.load(ag_mask_ptr + tile_id + off1)
        ag = tl.sum(ag_t)
        for kd in tl.static_range(KD):
            for kh in tl.static_range(KH):
                for kw in tl.static_range(KW):
                    in_d = out_d * STRIDE + kd - PADDING
                    in_h = out_h * STRIDE + kh - PADDING
                    in_w = out_w * STRIDE + kw - PADDING
                    dhw_ok = (m_mask
                              & (in_d >= 0) & (in_d < D_IN)
                              & (in_h >= 0) & (in_h < H_IN)
                              & (in_w >= 0) & (in_w < W_IN))
                    safe_d = tl.minimum(tl.maximum(in_d, 0), D_IN - 1)
                    safe_h = tl.minimum(tl.maximum(in_h, 0), H_IN - 1)
                    safe_w = tl.minimum(tl.maximum(in_w, 0), W_IN - 1)

                    x_pix = n_offset + safe_d * HWC + safe_h * WC + safe_w * C_IN
                    w_k_base = (kd * KH * KW + kh * KW + kw) * C_IN

                    for g in range(NUM_GROUPS):
                        g_active = (ag >> g) & 1
                        if g_active != 0:
                            cs = g * GROUP_SIZE_C
                            offs_k = cs + tl.arange(0, GROUP_SIZE_C)
                            k_mask = offs_k < C_IN
                            x_addrs = x_pix[:, None] + offs_k[None, :]
                            x_t = tl.load(x_ndhwc_ptr + x_addrs,
                                          mask=dhw_ok[:, None] & k_mask[None, :], other=0.0).to(tl.float16)
                            w_addrs = offs_n_out[None, :] * KCC + w_k_base + offs_k[:, None]
                            w_t = tl.load(w_kc_ptr + w_addrs,
                                          mask=k_mask[:, None] & n_out_mask[None, :], other=0.0).to(tl.float16)
                            acc += tl.dot(x_t, w_t)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n_out, mask=n_out_mask, other=0.0)[None, :]

    DHW_OUT = D_OUT * H_OUT * W_OUT
    pix_offset = out_d * (H_OUT * W_OUT) + out_h * W_OUT + out_w
    out_addrs = (
        n_idx * C_OUT * DHW_OUT
        + offs_n_out[None, :] * DHW_OUT
        + pix_offset[:, None]
    )
    out_mask = m_mask[:, None] & n_out_mask[None, :]
    tl.store(y_ptr + out_addrs, acc, mask=out_mask)


def _build_active_tile_ids(tile_class_buf, total_tiles):
    nz = torch.nonzero(tile_class_buf[:total_tiles] != TILE_ZERO, as_tuple=False)
    if nz.numel() == 0:
        return None, 0
    ids = nz.view(-1).to(torch.int32)
    return ids, int(ids.numel())


# ===========================================================================
# Public entry
# ===========================================================================

def sparse_conv3d_forward(
    x, weight, bias,
    kernel_size=None, stride=1, padding=0, dilation=1, groups=1,
    threshold=PRESCAN_ACTIVITY_EPS,
    ag_mask_buf=None, tile_class_buf=None,
    return_ms=False, fallback_ratio=FALLBACK_RATIO,
    return_avg_active_ratio=False, return_tile_stats=False,
    return_backend_meta=False,
    active_tile_ids_buf=None,
    launch_all_tiles=False,
):
    # ---- normalize args ----
    if isinstance(kernel_size, (tuple, list)):
        kernel_size = int(kernel_size[0])
    weight_ks = int(weight.shape[2])
    if kernel_size is None or int(kernel_size) != weight_ks:
        kernel_size = weight_ks
    if isinstance(stride, (tuple, list)):
        stride = int(stride[0])
    if isinstance(padding, (tuple, list)):
        padding = int(padding[0])
    if isinstance(dilation, (tuple, list)):
        dilation = int(dilation[0])
    stride = int(stride)
    padding = int(padding)
    dilation = int(dilation)
    groups = int(groups)
    KS = int(kernel_size)
    KD = KH = KW = KS

    N = int(x.shape[0])
    C_IN = int(x.shape[1])
    D_IN = int(x.shape[2])
    H_IN = int(x.shape[3])
    W_IN = int(x.shape[4])
    C_OUT = int(weight.shape[0])
    device = x.device
    HAS_BIAS = bias is not None

    D_OUT = (D_IN + 2 * padding - dilation * (KD - 1) - 1) // stride + 1
    H_OUT = (H_IN + 2 * padding - dilation * (KH - 1) - 1) // stride + 1
    W_OUT = (W_IN + 2 * padding - dilation * (KW - 1) - 1) // stride + 1

    need_stats = return_tile_stats or return_avg_active_ratio

    def _finalize(y, ms, avg=None, ts=None, bm=None):
        ret = (y, ms)
        if return_avg_active_ratio:
            ret = ret + (avg,)
        if return_tile_stats:
            ret = ret + (ts,)
        if return_backend_meta:
            ret = ret + (bm,)
        return ret

    def _dense_fallback(reason, avg_ratio_val=None, tile_stats_val=None):
        ms = 0.0
        if return_ms:
            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            se.record()
        y = Fn.conv3d(
            x.float(), weight.float(),
            bias.float() if bias is not None else None,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
        ).float()
        if return_ms:
            ee.record()
            torch.cuda.synchronize(device)
            ms = se.elapsed_time(ee)
        backend_meta = {
            "backend": "dense_fallback", "reason": reason,
            "kernel_type": f"3d_k{KS}_s{stride}",
            "total_tiles": -1, "launch_count": -1, "launch_mode": "dense",
        }
        if avg_ratio_val is None and return_avg_active_ratio:
            avg_ratio_val = 1.0
        return _finalize(y, ms, avg_ratio_val, tile_stats_val, backend_meta)

    if not _is_supported_sparse_pattern(KS, stride, padding, dilation, groups):
        return _dense_fallback("unsupported_sparse_pattern")
    if not x.is_cuda:
        return _dense_fallback("not_cuda")
    if D_OUT <= 0 or H_OUT <= 0 or W_OUT <= 0:
        return _dense_fallback("nonpositive_output_shape")

    BD, BH, BW = _select_3d_tile_sizes(D_OUT, H_OUT, W_OUT)
    BLOCK_M = BD * BH * BW
    GROUP_SIZE_C = choose_group_size(C_IN)
    NUM_GROUPS = (C_IN + GROUP_SIZE_C - 1) // GROUP_SIZE_C
    if NUM_GROUPS > 32:
        return _dense_fallback(f"num_groups_exceeds_uint32({NUM_GROUPS})")

    GD = (D_OUT + BD - 1) // BD
    GH = (H_OUT + BH - 1) // BH
    GW = (W_OUT + BW - 1) // BW
    tiles_per_n = GD * GH * GW
    N_TILES = N * tiles_per_n

    # ---- prepare buffers ----
    x_ndhwc = x.permute(0, 2, 3, 4, 1).contiguous().to(torch.float16)

    if ag_mask_buf is None or ag_mask_buf.numel() < N_TILES:
        ag_mask_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
    if tile_class_buf is None or tile_class_buf.numel() < N_TILES:
        tile_class_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)

    sparse_ms = 0.0
    if return_ms:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()

    # ---- Stage 1: shared PyTorch three-stage prescan ----
    tile_class, ag_mask = build_rf_prescan_metadata(
        x_channels_last=x_ndhwc,
        spatial_dims=(D_OUT, H_OUT, W_OUT),
        kernel_dims=(KD, KH, KW),
        stride=stride,
        padding=padding,
        block_dims=(BD, BH, BW),
        group_size_c=GROUP_SIZE_C,
        num_groups=NUM_GROUPS,
        threshold=float(threshold),
    )
    tile_class_buf[:N_TILES].copy_(tile_class)
    ag_mask_buf[:N_TILES].copy_(ag_mask)

    avg_active_ratio = None
    tile_stats = None

    if need_stats:
        tc = tile_class_buf[:N_TILES]
        zt_t = (tc == TILE_ZERO).sum()
        sp_t = (tc == TILE_SPARSE).sum()
        dt_t = (tc == TILE_DENSEISH).sum()
        pc_sum_t = popcount_buf(ag_mask_buf, N_TILES).sum()
        stats_host = torch.stack([zt_t.long(), sp_t.long(), dt_t.long(), pc_sum_t.long()]).cpu()
        zt, sp_ct, dt, pc_sum = [int(v) for v in stats_host.tolist()]
        avg_active_ratio = float(pc_sum) / max(float(N_TILES * NUM_GROUPS), 1.0)

        if return_tile_stats:
            tile_stats = {
                "zero_tiles": zt, "sparse_tiles": sp_ct, "denseish_tiles": dt,
                "total_tiles": int(N_TILES),
                "prescan_mode": "pytorch_three_stage_rf_common_v3",
                "block_d": int(BD), "block_h": int(BH), "block_w": int(BW),
                "block_m": int(BLOCK_M),
                "group_size_c": int(GROUP_SIZE_C),
                "num_groups": int(NUM_GROUPS),
                "active_tile_ratio": float(N_TILES - zt) / max(float(N_TILES), 1.0),
                "avg_active_group_ratio": float(avg_active_ratio),
            }

        if avg_active_ratio > float(fallback_ratio):
            if return_ms:
                end_evt.record()
                torch.cuda.synchronize(device)
                sparse_ms = start_evt.elapsed_time(end_evt)
            return _dense_fallback(
                "post_metadata_dense_fallback",
                avg_ratio_val=avg_active_ratio,
                tile_stats_val=tile_stats,
            )

    # [B.3] active-tile compaction runs regardless of need_stats
    active_ids = None
    active_count = N_TILES
    use_tile_ids = False

    if not launch_all_tiles:
        ids, n_active = _build_active_tile_ids(tile_class_buf, N_TILES)
        if n_active == 0:
            y = torch.zeros((N, C_OUT, D_OUT, H_OUT, W_OUT), device=device, dtype=torch.float32)
            if HAS_BIAS:
                y += bias.float().view(1, C_OUT, 1, 1, 1)
            if return_ms:
                end_evt.record()
                torch.cuda.synchronize(device)
                sparse_ms = start_evt.elapsed_time(end_evt)
            backend_meta = {
                "backend": "all_zero_after_metadata",
                "reason": "no_active_tiles",
                "kernel_type": f"3d_k{KS}_s{stride}",
                "total_tiles": int(N_TILES), "launch_count": 0,
                "launch_mode": "active_only", "active_tiles": 0,
            }
            return _finalize(y, sparse_ms, avg_active_ratio, tile_stats, backend_meta)

        if active_tile_ids_buf is None or active_tile_ids_buf.numel() < n_active:
            active_tile_ids_buf = torch.empty(n_active, dtype=torch.int32, device=device)
        active_tile_ids_buf[:n_active].copy_(ids)
        active_ids = active_tile_ids_buf
        active_count = n_active
        use_tile_ids = True

    w_kc = (
        weight.permute(0, 2, 3, 4, 1)
        .contiguous()
        .to(torch.float16)
        .view(C_OUT, KD * KH * KW * C_IN)
    )
    bias_arg = (
        bias.to(torch.float32) if bias is not None
        else torch.empty(1, dtype=torch.float32, device=device)
    )

    # ---- Stage 2: compute ----
    # [C.3] Pre-fill bias so TILE_ZERO tiles (early-return in kernel) are correct.
    y = torch.empty((N, C_OUT, D_OUT, H_OUT, W_OUT), device=device, dtype=torch.float32)
    if HAS_BIAS:
        y.copy_(bias.float().view(1, C_OUT, 1, 1, 1).expand(N, C_OUT, D_OUT, H_OUT, W_OUT))
    else:
        y.zero_()

    if active_ids is None:
        active_ids = torch.empty(0, dtype=torch.int32, device=device)

    grid = lambda META: (active_count, triton.cdiv(C_OUT, META["BLOCK_N_OUT"]))
    _sparse_conv3d_kernel[grid](
        x_ndhwc, w_kc, bias_arg, y,
        ag_mask_buf, tile_class_buf, active_ids, N,
        C_IN=C_IN, C_OUT=C_OUT,
        D_IN=D_IN, H_IN=H_IN, W_IN=W_IN,
        D_OUT=D_OUT, H_OUT=H_OUT, W_OUT=W_OUT,
        KD=KD, KH=KH, KW=KW,
        STRIDE=stride, PADDING=padding,
        GD=GD, GH=GH, GW=GW,
        BD=BD, BH=BH, BW=BW,
        BLOCK_M=BLOCK_M,
        GROUP_SIZE_C=GROUP_SIZE_C, NUM_GROUPS=NUM_GROUPS,
        HAS_BIAS=HAS_BIAS, USE_TILE_IDS=use_tile_ids,
    )

    if return_ms:
        end_evt.record()
        torch.cuda.synchronize(device)
        sparse_ms = start_evt.elapsed_time(end_evt)

    backend_meta = {
        "backend": "sparse_active_tiles" if use_tile_ids else "sparse_all_tiles",
        "reason": "ok",
        "kernel_type": f"3d_k{KS}_s{stride}",
        "total_tiles": int(N_TILES),
        "launch_count": int(active_count),
        "launch_mode": "active_only" if use_tile_ids else "all_tiles",
    }
    if avg_active_ratio is not None:
        backend_meta["avg_active_group_ratio"] = float(avg_active_ratio)

    return _finalize(y, sparse_ms, avg_active_ratio, tile_stats, backend_meta)
