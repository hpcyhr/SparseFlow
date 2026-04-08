"""
SparseFlow Kernels/conv1d.py - Unified Sparse Conv1d kernel framework.

This file aligns Conv1d with the Conv2d framework semantics:
1) output-space tiling + channel grouping
2) metadata build (tile_class + active-group bitmask)
3) optional sync-gated stats / dense fallback
4) launch mode switch (active_only vs all_tiles)
5) active-group sparse execution
6) structured outputs (avg_active_ratio / tile_stats / backend_meta)
"""

import torch
import triton
import triton.language as tl
from triton import autotune, Config

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Utils.config import PRESCAN_ACTIVITY_EPS, SPARSE_DENSE_RATIO_THRESHOLD
from Utils.sparse_helpers import (
    TILE_ZERO,
    TILE_SPARSE,
    TILE_DENSEISH,
    choose_group_size,
    popcount_buf,
)

FALLBACK_RATIO = SPARSE_DENSE_RATIO_THRESHOLD


def _select_tile_size_1d(l_out: int) -> int:
    if l_out >= 256:
        return 32
    if l_out >= 64:
        return 16
    return max(1, int(l_out))


def _check_dense_fallback(
    ag_mask_buf: torch.Tensor,
    total_tiles: int,
    num_groups: int,
    fallback_ratio: float = FALLBACK_RATIO,
) -> bool:
    """
    NOTE: calls .item() and syncs GPU->CPU. Only call this in need_stats paths.
    """
    if num_groups == 0:
        return False
    pc = popcount_buf(ag_mask_buf, total_tiles)
    avg_active = pc.float().mean().item()
    return avg_active > float(fallback_ratio) * float(num_groups)


def _build_active_tile_ids(tile_class_buf: torch.Tensor, total_tiles: int):
    """
    NOTE: calls torch.nonzero() and syncs GPU->CPU. Only call in active_only mode.
    """
    tc = tile_class_buf[:total_tiles]
    active = torch.nonzero(tc != TILE_ZERO, as_tuple=False).flatten()
    if active.numel() == 0:
        return active.to(dtype=torch.int32), 0
    return active.to(dtype=torch.int32).contiguous(), int(active.numel())


def _ensure_metadata_buffers(
    ag_mask_buf: torch.Tensor,
    tile_class_buf: torch.Tensor,
    total_tiles: int,
    device: torch.device,
):
    if ag_mask_buf is None or ag_mask_buf.numel() < total_tiles:
        ag_mask_buf = torch.empty(total_tiles, dtype=torch.int32, device=device)
    if tile_class_buf is None or tile_class_buf.numel() < total_tiles:
        tile_class_buf = torch.empty(total_tiles, dtype=torch.int32, device=device)
    return ag_mask_buf, tile_class_buf


@triton.jit
def _prescan_conv1d_kernel(
    x_ptr,  # [N, C_IN, L_IN]
    ag_mask_ptr,  # [TOTAL_TILES]
    tile_class_ptr,  # [TOTAL_TILES]
    C_IN: tl.constexpr,
    L_IN: tl.constexpr,
    L_OUT: tl.constexpr,
    KS: tl.constexpr,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    BL: tl.constexpr,
    N_TILES_L: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    ALL_ONES: tl.constexpr,
    THRESHOLD: tl.constexpr,
    RF_BLOCK: tl.constexpr,
):
    """
    Single-stage exact RF prescan:
    - build per-tile group-activity bitmask
    - classify tile into ZERO / SPARSE / DENSEISH
    """
    pid = tl.program_id(0)
    n_idx = pid // N_TILES_L
    tile_idx = pid % N_TILES_L

    off1 = tl.arange(0, 1)
    ag_mask = tl.zeros([1], dtype=tl.int32)
    any_nonzero = tl.zeros([1], dtype=tl.int32)

    rf_start = tile_idx * BL * STRIDE - PADDING
    RF_LEN: tl.constexpr = (BL - 1) * STRIDE + KS

    for g in range(NUM_GROUPS):
        g_start = g * GROUP_SIZE_C
        group_has_nonzero = tl.zeros([1], dtype=tl.int32)

        for ci in range(GROUP_SIZE_C):
            c = g_start + ci
            if c < C_IN:
                base = n_idx * C_IN * L_IN + c * L_IN
                group_max = tl.zeros([1], dtype=tl.float32)
                for rb in range(0, RF_LEN, RF_BLOCK):
                    offs = rb + tl.arange(0, RF_BLOCK)
                    l_in = rf_start + offs
                    m = (offs < RF_LEN) & (l_in >= 0) & (l_in < L_IN)
                    vals = tl.load(x_ptr + base + l_in, mask=m, other=0.0)
                    group_max = tl.maximum(group_max, tl.max(tl.abs(vals), axis=0))
                if tl.sum(group_max > THRESHOLD) > 0:
                    group_has_nonzero = tl.full([1], 1, dtype=tl.int32)

        if tl.sum(group_has_nonzero) != 0:
            ag_mask = ag_mask + group_has_nonzero * (1 << g)
            any_nonzero = tl.full([1], 1, dtype=tl.int32)

    tl.store(ag_mask_ptr + pid + off1, ag_mask)

    if tl.sum(any_nonzero) == 0:
        tl.store(tile_class_ptr + pid + off1, tl.zeros([1], dtype=tl.int32))
    else:
        if tl.sum(ag_mask == ALL_ONES) > 0:
            tl.store(tile_class_ptr + pid + off1, tl.full([1], TILE_DENSEISH, dtype=tl.int32))
        else:
            tl.store(tile_class_ptr + pid + off1, tl.full([1], TILE_SPARSE, dtype=tl.int32))


def _build_conv1d_metadata(
    x_f16: torch.Tensor,
    ag_mask_buf: torch.Tensor,
    tile_class_buf: torch.Tensor,
    *,
    c_in: int,
    l_in: int,
    l_out: int,
    ks: int,
    stride: int,
    padding: int,
    bl: int,
    n_tiles_l: int,
    group_size_c: int,
    num_groups: int,
    all_ones: int,
    threshold: float,
    prescan_stats: dict = None,
):
    total_tiles = int(x_f16.shape[0]) * int(n_tiles_l)
    _prescan_conv1d_kernel[(total_tiles,)](
        x_f16,
        ag_mask_buf,
        tile_class_buf,
        C_IN=c_in,
        L_IN=l_in,
        L_OUT=l_out,
        KS=ks,
        STRIDE=stride,
        PADDING=padding,
        BL=bl,
        N_TILES_L=n_tiles_l,
        GROUP_SIZE_C=group_size_c,
        NUM_GROUPS=num_groups,
        ALL_ONES=all_ones,
        THRESHOLD=threshold,
        RF_BLOCK=64,
    )
    if prescan_stats is not None:
        prescan_stats.update(
            {
                "prescan_mode": "single_stage_rf_conv1d_v4",
                "tile_l": int(bl),
                "n_tiles_l": int(n_tiles_l),
                "group_size_c": int(group_size_c),
                "num_groups": int(num_groups),
            }
        )


_CFG_CONV1D = [
    Config({"BLOCK_N": 32}, num_warps=4, num_stages=2),
    Config({"BLOCK_N": 64}, num_warps=4, num_stages=2),
    Config({"BLOCK_N": 64}, num_warps=8, num_stages=3),
]


@autotune(configs=_CFG_CONV1D, key=["C_IN", "C_OUT", "L_OUT", "N_TILES_L"])
@triton.jit
def _sparse_conv1d_compute_kernel(
    x_ptr,  # [N, C_IN, L_IN] fp16
    w_ptr,  # [C_OUT, C_IN, KS] fp16 contiguous
    ag_mask_ptr,  # [TOTAL_TILES] int32
    tile_ids_ptr,  # [active_tiles] int32 (or placeholder)
    y_ptr,  # [N, C_OUT, L_OUT] fp32 (bias baseline prefilled)
    N_batch,
    C_IN,
    C_OUT,
    L_IN,
    L_OUT,
    N_TILES_L,
    KS: tl.constexpr,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    ALL_ONES_MASK: tl.constexpr,
    DENSE_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_TILE_IDS: tl.constexpr,
):
    pid_tile = tl.program_id(0)
    tile_id = tl.load(tile_ids_ptr + pid_tile) if USE_TILE_IDS else pid_tile
    pid_cout = tl.program_id(1)

    if tile_id >= N_batch * N_TILES_L:
        return

    n_idx = tile_id // N_TILES_L
    t_idx = tile_id % N_TILES_L

    offs_n = pid_cout * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < C_OUT

    offs_m = tl.arange(0, BLOCK_M)
    l_out = t_idx * BLOCK_M + offs_m
    m_mask = l_out < L_OUT

    ag_mask = tl.load(ag_mask_ptr + tile_id)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    if ag_mask != 0:
        if ag_mask == ALL_ONES_MASK:
            for k in range(KS):
                l_in = l_out * STRIDE - PADDING + k
                li_ok = m_mask & (l_in >= 0) & (l_in < L_IN)
                safe_l = tl.minimum(tl.maximum(l_in, 0), L_IN - 1)

                for cin_base in range(0, NUM_GROUPS * GROUP_SIZE_C, DENSE_K):
                    offs_k = cin_base + tl.arange(0, DENSE_K)
                    k_mask = offs_k < C_IN

                    x_addr = n_idx * C_IN * L_IN + offs_k[None, :] * L_IN + safe_l[:, None]
                    x_t = tl.load(
                        x_ptr + x_addr,
                        mask=li_ok[:, None] & k_mask[None, :],
                        other=0.0,
                    ).to(tl.float16)

                    w_addr = offs_n[None, :] * (C_IN * KS) + offs_k[:, None] * KS + k
                    w_t = tl.load(
                        w_ptr + w_addr,
                        mask=k_mask[:, None] & n_mask[None, :],
                        other=0.0,
                    ).to(tl.float16)
                    acc += tl.dot(x_t, w_t)
        else:
            for k in range(KS):
                l_in = l_out * STRIDE - PADDING + k
                li_ok = m_mask & (l_in >= 0) & (l_in < L_IN)
                safe_l = tl.minimum(tl.maximum(l_in, 0), L_IN - 1)

                for g in range(NUM_GROUPS):
                    if ((ag_mask >> g) & 1) != 0:
                        offs_k = g * GROUP_SIZE_C + tl.arange(0, GROUP_SIZE_C)
                        k_mask = offs_k < C_IN

                        x_addr = n_idx * C_IN * L_IN + offs_k[None, :] * L_IN + safe_l[:, None]
                        x_t = tl.load(
                            x_ptr + x_addr,
                            mask=li_ok[:, None] & k_mask[None, :],
                            other=0.0,
                        ).to(tl.float16)

                        w_addr = offs_n[None, :] * (C_IN * KS) + offs_k[:, None] * KS + k
                        w_t = tl.load(
                            w_ptr + w_addr,
                            mask=k_mask[:, None] & n_mask[None, :],
                            other=0.0,
                        ).to(tl.float16)
                        acc += tl.dot(x_t, w_t)

    out_addr = (n_idx * C_OUT + offs_n[None, :]) * L_OUT + l_out[:, None]
    out_old = tl.load(y_ptr + out_addr, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    tl.store(
        y_ptr + out_addr,
        out_old + acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


def sparse_conv1d_forward(
    x, weight, bias,
    kernel_size=None, stride=1, padding=0, dilation=1, groups=1,
    threshold=PRESCAN_ACTIVITY_EPS,
    ag_mask_buf=None, tile_class_buf=None,
    return_ms=False, fallback_ratio=FALLBACK_RATIO,
    return_avg_active_ratio=False, return_tile_stats=False,
    return_backend_meta=False,
    active_tile_ids_buf=None,
    launch_all_tiles=False,
    # Legacy compat (kept for unified call surface)
    block_size=None, counts_buf=None, tile_cin_buf=None,
    group_flags_buf=None, ag_count_buf=None, ag_list_buf=None,
    tile_alive_buf=None,
):
    import torch.nn.functional as Fn

    N, C_IN, L_IN = x.shape
    C_OUT = int(weight.shape[0])
    device = x.device

    if isinstance(kernel_size, (tuple, list)):
        kernel_size = int(kernel_size[0])
    weight_kernel_size = int(weight.shape[2])
    if kernel_size is None:
        kernel_size = weight_kernel_size
    if int(kernel_size) != weight_kernel_size:
        kernel_size = weight_kernel_size
    if isinstance(stride, (tuple, list)):
        stride = int(stride[0])
    if isinstance(padding, (tuple, list)):
        padding = int(padding[0])
    if isinstance(dilation, (tuple, list)):
        dilation = int(dilation[0])

    need_stats = return_tile_stats or return_avg_active_ratio

    def _finalize_return(y, ms, avg_active_ratio_val=None, tile_stats_val=None, backend_meta_val=None):
        ret = (y, ms)
        if return_avg_active_ratio:
            ret = ret + (avg_active_ratio_val,)
        if return_tile_stats:
            ret = ret + (tile_stats_val,)
        if return_backend_meta:
            ret = ret + (backend_meta_val,)
        return ret

    def _dense_fallback(reason="dense_fallback", avg_active_ratio_val=1.0, tile_stats_val=None, backend_meta_extra=None):
        dense_ms = 0.0
        if return_ms:
            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            se.record()

        y = Fn.conv1d(
            x.float(),
            weight.float(),
            bias.float() if bias is not None else None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        ).float()

        if return_ms:
            ee.record()
            torch.cuda.synchronize(device)
            dense_ms = se.elapsed_time(ee)

        bm = {"backend": "dense_fallback", "reason": reason}
        if backend_meta_extra:
            bm.update(backend_meta_extra)
        return _finalize_return(y, dense_ms, avg_active_ratio_val, tile_stats_val, bm)

    L_OUT = (L_IN + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    def _zero_tiles_output(reason, tile_stats_val=None, backend_meta_extra=None):
        y = torch.zeros(N, C_OUT, L_OUT, dtype=torch.float32, device=device)
        if bias is not None:
            y = y + bias.detach().float().view(1, -1, 1)
        bm = {"backend": "zero_tiles_only", "reason": reason}
        if backend_meta_extra:
            bm.update(backend_meta_extra)
        return _finalize_return(y, 0.0, 0.0, tile_stats_val, bm)

    # ------------------------------------------------------------------
    # 1) validate / support checks
    # ------------------------------------------------------------------
    if groups != 1 or dilation != 1:
        return _dense_fallback(reason="unsupported_groups_or_dilation")
    if L_OUT <= 0:
        return _dense_fallback(reason="invalid_output_shape")
    if x.ndim != 3 or weight.ndim != 3:
        return _dense_fallback(reason="invalid_tensor_rank")
    if x.device.type != "cuda" or weight.device.type != "cuda":
        return _dense_fallback(reason="not_cuda")

    # ------------------------------------------------------------------
    # 2) tile/group config
    # ------------------------------------------------------------------
    GROUP_SIZE_C = choose_group_size(C_IN)
    NUM_GROUPS = triton.cdiv(C_IN, GROUP_SIZE_C)
    ALL_ONES_MASK = (1 << NUM_GROUPS) - 1
    DENSE_K = min(max(GROUP_SIZE_C * 2, 16), 64)

    BL = _select_tile_size_1d(L_OUT)
    N_TILES_L = triton.cdiv(L_OUT, BL)
    N_TILES = N * N_TILES_L

    # ------------------------------------------------------------------
    # 3) layout prep + metadata allocation
    # ------------------------------------------------------------------
    x_f16 = x if (x.dtype == torch.float16 and x.is_contiguous()) else x.half().contiguous()
    w_f16 = weight if (weight.dtype == torch.float16 and weight.is_contiguous()) else weight.half().contiguous()

    ag_mask_buf, tile_class_buf = _ensure_metadata_buffers(ag_mask_buf, tile_class_buf, N_TILES, device)

    # ------------------------------------------------------------------
    # 4) metadata build
    # ------------------------------------------------------------------
    prescan_stats = {} if return_tile_stats else None
    try:
        _build_conv1d_metadata(
            x_f16=x_f16,
            ag_mask_buf=ag_mask_buf,
            tile_class_buf=tile_class_buf,
            c_in=C_IN,
            l_in=L_IN,
            l_out=L_OUT,
            ks=kernel_size,
            stride=stride,
            padding=padding,
            bl=BL,
            n_tiles_l=N_TILES_L,
            group_size_c=GROUP_SIZE_C,
            num_groups=NUM_GROUPS,
            all_ones=ALL_ONES_MASK,
            threshold=float(threshold),
            prescan_stats=prescan_stats,
        )
    except Exception:
        return _dense_fallback(reason="prescan_failed")

    # ------------------------------------------------------------------
    # 5) optional stats + optional fallback (sync-gated)
    # ------------------------------------------------------------------
    avg_active_ratio = None
    tile_stats_base = None
    active_tiles_for_meta = None

    if need_stats:
        tc = tile_class_buf[:N_TILES]
        zc = int((tc == TILE_ZERO).sum().item())
        sc = int((tc == TILE_SPARSE).sum().item())
        dc = int((tc == TILE_DENSEISH).sum().item())
        total_nonzero = sc + dc
        denseish_ratio = float(dc) / max(float(total_nonzero), 1.0)
        active_tiles_for_meta = int(total_nonzero)

        if NUM_GROUPS > 0:
            pc = popcount_buf(ag_mask_buf, N_TILES)
            avg_active_ratio = float(pc.sum().item()) / max(float(N_TILES * NUM_GROUPS), 1.0)
        else:
            avg_active_ratio = 1.0

        if return_tile_stats:
            tile_stats_base = {
                "zero_tiles": zc,
                "sparse_tiles": sc,
                "denseish_tiles": dc,
                "total_tiles": N_TILES,
                "prescan_mode": "single_stage_rf_conv1d_v4",
                "active_tiles": total_nonzero,
                "active_tile_ratio": float(total_nonzero) / max(float(N_TILES), 1.0),
                "denseish_ratio_nonzero": denseish_ratio,
                "avg_active_group_ratio": avg_active_ratio,
            }
            if prescan_stats:
                tile_stats_base.update(prescan_stats)

        if _check_dense_fallback(ag_mask_buf, N_TILES, NUM_GROUPS, fallback_ratio=fallback_ratio):
            return _dense_fallback(
                reason="post_metadata_dense_fallback",
                avg_active_ratio_val=avg_active_ratio,
                tile_stats_val=tile_stats_base,
                backend_meta_extra={
                    "active_tiles": total_nonzero,
                    "total_tiles": N_TILES,
                    "denseish_ratio_nonzero": denseish_ratio,
                },
            )

    # ------------------------------------------------------------------
    # 6) launch mode selection
    # ------------------------------------------------------------------
    if launch_all_tiles:
        launch_count = N_TILES
        use_tile_ids = False
        tile_ids_ptr = ag_mask_buf  # placeholder, ignored by kernel in all_tiles mode
    else:
        active_tile_ids, active_tile_count = _build_active_tile_ids(tile_class_buf, N_TILES)
        if active_tile_count == 0:
            return _zero_tiles_output(
                reason="all_tiles_zero_after_metadata",
                tile_stats_val=tile_stats_base,
                backend_meta_extra={"active_tiles": 0, "total_tiles": N_TILES},
            )
        if active_tile_ids_buf is not None and active_tile_ids_buf.numel() >= active_tile_count:
            active_tile_ids_buf[:active_tile_count].copy_(active_tile_ids)
            tile_ids_ptr = active_tile_ids_buf
        else:
            tile_ids_ptr = active_tile_ids
        launch_count = active_tile_count
        use_tile_ids = True
        if active_tiles_for_meta is None:
            active_tiles_for_meta = int(active_tile_count)

    # ------------------------------------------------------------------
    # 7) sparse execution (active-only compute over metadata-selected groups)
    # ------------------------------------------------------------------
    y = torch.zeros(N, C_OUT, L_OUT, dtype=torch.float32, device=device)
    if bias is not None:
        y = y + bias.detach().float().view(1, -1, 1)

    if return_ms:
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()

    def _grid(meta):
        return (launch_count, triton.cdiv(C_OUT, meta["BLOCK_N"]))

    _sparse_conv1d_compute_kernel[_grid](
        x_f16,
        w_f16,
        ag_mask_buf,
        tile_ids_ptr,
        y,
        N_batch=N,
        C_IN=C_IN,
        C_OUT=C_OUT,
        L_IN=L_IN,
        L_OUT=L_OUT,
        N_TILES_L=N_TILES_L,
        KS=kernel_size,
        STRIDE=stride,
        PADDING=padding,
        GROUP_SIZE_C=GROUP_SIZE_C,
        NUM_GROUPS=NUM_GROUPS,
        ALL_ONES_MASK=ALL_ONES_MASK,
        DENSE_K=DENSE_K,
        BLOCK_M=BL,
        USE_TILE_IDS=use_tile_ids,
    )

    sparse_ms = 0.0
    if return_ms:
        end_ev.record()
        torch.cuda.synchronize(device)
        sparse_ms = start_ev.elapsed_time(end_ev)

    # ------------------------------------------------------------------
    # 8) structured outputs
    # ------------------------------------------------------------------
    backend_meta = {
        "backend": "sparse_triton",
        "reason": "conv1d_unified_v1",
        "total_tiles": N_TILES,
        "launch_count": launch_count,
        "launch_mode": "all_tiles" if launch_all_tiles else "active_only",
    }
    if active_tiles_for_meta is not None:
        backend_meta["active_tiles"] = int(active_tiles_for_meta)
    if avg_active_ratio is not None:
        backend_meta["avg_active_group_ratio"] = float(avg_active_ratio)

    return _finalize_return(y, sparse_ms, avg_active_ratio, tile_stats_base, backend_meta)
