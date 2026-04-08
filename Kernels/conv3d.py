"""
SparseFlow Kernels/conv3d.py - Sparse Conv3d kernel.

Maturity: main_path (active-tile sparse execution).

Strategy:
1) Prescan output tiles and build active channel-group bitmask.
2) Skip zero tiles entirely.
3) For active tiles, compute only active channel groups.
4) Fallback to dense when active-group ratio is too high.
"""

import torch
import triton
import triton.language as tl

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Utils.sparse_helpers import (
    TILE_ZERO,
    TILE_SPARSE,
    TILE_DENSEISH,
    choose_group_size,
    popcount_buf,
)

FALLBACK_RATIO = 0.85


def _select_3d_tile_sizes(d_out: int, h_out: int, w_out: int):
    voxels = d_out * h_out * w_out
    if voxels >= 4096:
        return 2, 4, 8
    if voxels >= 1024:
        return 2, 4, 4
    return 2, 2, 4


@triton.jit
def _prescan_conv3d_kernel(
    x_ptr,  # [N, C_IN, D_IN, H_IN, W_IN]
    ag_mask_ptr,  # [TOTAL_TILES]
    tile_class_ptr,  # [TOTAL_TILES]
    C_IN: tl.constexpr,
    D_IN: tl.constexpr,
    H_IN: tl.constexpr,
    W_IN: tl.constexpr,
    D_OUT: tl.constexpr,
    H_OUT: tl.constexpr,
    W_OUT: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    BD: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    N_TILES_D: tl.constexpr,
    N_TILES_H: tl.constexpr,
    N_TILES_W: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    ALL_ONES: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles_per_n = N_TILES_D * N_TILES_H * N_TILES_W
    n_idx = pid // tiles_per_n
    rem = pid % tiles_per_n
    td = rem // (N_TILES_H * N_TILES_W)
    rem2 = rem % (N_TILES_H * N_TILES_W)
    th = rem2 // N_TILES_W
    tw = rem2 % N_TILES_W

    d_out_start = td * BD
    h_out_start = th * BH
    w_out_start = tw * BW

    off1 = tl.arange(0, 1)
    ag_mask = tl.zeros([1], dtype=tl.int32)
    any_nonzero = tl.zeros([1], dtype=tl.int32)
    HW = H_IN * W_IN
    DHW = D_IN * HW

    for g in range(NUM_GROUPS):
        g_start = g * GROUP_SIZE_C
        group_has_nonzero = tl.zeros([1], dtype=tl.int32)

        for bd in range(BD):
            d_out = d_out_start + bd
            if d_out < D_OUT:
                for bh in range(BH):
                    h_out = h_out_start + bh
                    if h_out < H_OUT:
                        for bw in range(BW):
                            w_out = w_out_start + bw
                            if w_out < W_OUT:
                                for kd in range(KD):
                                    d_in = d_out * STRIDE - PADDING + kd
                                    if (d_in >= 0) and (d_in < D_IN):
                                        for kh in range(KH):
                                            h_in = h_out * STRIDE - PADDING + kh
                                            if (h_in >= 0) and (h_in < H_IN):
                                                for kw in range(KW):
                                                    w_in = w_out * STRIDE - PADDING + kw
                                                    if (w_in >= 0) and (w_in < W_IN):
                                                        for ci in range(GROUP_SIZE_C):
                                                            c = g_start + ci
                                                            if c < C_IN:
                                                                addr = (
                                                                    n_idx * C_IN * DHW
                                                                    + c * DHW
                                                                    + d_in * HW
                                                                    + h_in * W_IN
                                                                    + w_in
                                                                )
                                                                val = tl.load(x_ptr + addr)
                                                                if tl.abs(val) > THRESHOLD:
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


def _build_active_tile_ids(tile_class_buf: torch.Tensor, total_tiles: int):
    tc = tile_class_buf[:total_tiles]
    active = torch.nonzero(tc != TILE_ZERO, as_tuple=False).flatten()
    if active.numel() == 0:
        return active.to(dtype=torch.int32), 0
    return active.to(dtype=torch.int32).contiguous(), int(active.numel())


def _decode_active_groups(mask: int, num_groups: int, group_size_c: int, c_in: int):
    channels = []
    for g in range(num_groups):
        if ((mask >> g) & 1) != 0:
            cs = g * group_size_c
            ce = min(cs + group_size_c, c_in)
            channels.extend(range(cs, ce))
    return channels


def sparse_conv3d_forward(
    x: torch.Tensor,  # [N, C_IN, D, H, W]
    weight: torch.Tensor,  # [C_OUT, C_IN, KD, KH, KW]
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 1,
    threshold: float = 1e-6,
    return_ms: bool = False,
    return_tile_stats: bool = False,
    return_backend_meta: bool = False,
    fallback_ratio: float = FALLBACK_RATIO,
):
    """
    Sparse 3D convolution with active-tile sparse execution.
    """
    import torch.nn.functional as Fn

    def _finalize_return(y, ms, stats=None, backend_meta=None):
        ret = (y, ms)
        if return_tile_stats:
            ret = ret + (stats,)
        if return_backend_meta:
            ret = ret + (backend_meta or {},)
        return ret

    if isinstance(stride, (tuple, list)):
        stride = stride[0]
    if isinstance(padding, (tuple, list)):
        padding = padding[0]

    N_batch, C_IN, D_IN, H_IN, W_IN = x.shape
    C_OUT = weight.shape[0]
    KD, KH, KW = int(weight.shape[2]), int(weight.shape[3]), int(weight.shape[4])
    device = x.device

    D_OUT = (D_IN + 2 * padding - KD) // stride + 1
    H_OUT = (H_IN + 2 * padding - KH) // stride + 1
    W_OUT = (W_IN + 2 * padding - KW) // stride + 1

    if D_OUT <= 0 or H_OUT <= 0 or W_OUT <= 0:
        y = Fn.conv3d(
            x.float(),
            weight.float(),
            bias.float() if bias is not None else None,
            stride=stride,
            padding=padding,
        ).float()
        stats = {
            "backend": "dense_fallback",
            "reason": "invalid_output_shape",
            "fallback": True,
            "total_tiles": 0,
            "prescan_version": "conv3d_v2_active_tile_sparse",
        }
        return _finalize_return(y, 0.0, stats, {"backend": "dense_fallback", "reason": "invalid_output_shape"})

    GROUP_SIZE_C = choose_group_size(C_IN)
    NUM_GROUPS = triton.cdiv(C_IN, GROUP_SIZE_C)
    ALL_ONES = (1 << NUM_GROUPS) - 1

    BD, BH, BW = _select_3d_tile_sizes(D_OUT, H_OUT, W_OUT)
    N_TILES_D = triton.cdiv(D_OUT, BD)
    N_TILES_H = triton.cdiv(H_OUT, BH)
    N_TILES_W = triton.cdiv(W_OUT, BW)
    TOTAL_TILES = N_batch * N_TILES_D * N_TILES_H * N_TILES_W

    x_f16 = x if (x.dtype == torch.float16 and x.is_contiguous()) else x.half().contiguous()
    ag_mask_buf = torch.empty(TOTAL_TILES, dtype=torch.int32, device=device)
    tile_class_buf = torch.empty(TOTAL_TILES, dtype=torch.int32, device=device)

    try:
        _prescan_conv3d_kernel[(TOTAL_TILES,)](
            x_f16,
            ag_mask_buf,
            tile_class_buf,
            C_IN=C_IN,
            D_IN=D_IN,
            H_IN=H_IN,
            W_IN=W_IN,
            D_OUT=D_OUT,
            H_OUT=H_OUT,
            W_OUT=W_OUT,
            KD=KD,
            KH=KH,
            KW=KW,
            STRIDE=stride,
            PADDING=padding,
            BD=BD,
            BH=BH,
            BW=BW,
            N_TILES_D=N_TILES_D,
            N_TILES_H=N_TILES_H,
            N_TILES_W=N_TILES_W,
            GROUP_SIZE_C=GROUP_SIZE_C,
            NUM_GROUPS=NUM_GROUPS,
            ALL_ONES=ALL_ONES,
            THRESHOLD=threshold,
        )
    except Exception:
        y = Fn.conv3d(
            x.float(),
            weight.float(),
            bias.float() if bias is not None else None,
            stride=stride,
            padding=padding,
        ).float()
        stats = {
            "backend": "dense_fallback",
            "reason": "prescan_failed",
            "fallback": True,
            "total_tiles": TOTAL_TILES,
            "prescan_version": "conv3d_v2_active_tile_sparse",
        }
        return _finalize_return(y, 0.0, stats, {"backend": "dense_fallback", "reason": "prescan_failed"})

    avg_active_group_ratio = 1.0
    if NUM_GROUPS > 0:
        pc = popcount_buf(ag_mask_buf, TOTAL_TILES)
        avg_active_group_ratio = float(pc.sum().item()) / max(float(TOTAL_TILES * NUM_GROUPS), 1.0)
    if avg_active_group_ratio > fallback_ratio:
        y = Fn.conv3d(
            x.float(),
            weight.float(),
            bias.float() if bias is not None else None,
            stride=stride,
            padding=padding,
        ).float()
        tc = tile_class_buf[:TOTAL_TILES]
        stats = {
            "backend": "dense_fallback",
            "reason": "post_metadata_dense_fallback",
            "fallback": True,
            "total_tiles": TOTAL_TILES,
            "zero_tiles": int((tc == TILE_ZERO).sum().item()),
            "sparse_tiles": int((tc == TILE_SPARSE).sum().item()),
            "denseish_tiles": int((tc == TILE_DENSEISH).sum().item()),
            "avg_active_group_ratio": avg_active_group_ratio,
            "prescan_version": "conv3d_v2_active_tile_sparse",
        }
        return _finalize_return(y, 0.0, stats, {"backend": "dense_fallback", "reason": "post_metadata_dense_fallback"})

    active_tile_ids, active_tile_count = _build_active_tile_ids(tile_class_buf, TOTAL_TILES)
    y = torch.zeros(N_batch, C_OUT, D_OUT, H_OUT, W_OUT, dtype=torch.float32, device=device)

    if active_tile_count > 0:
        x_f32 = x.float()
        w_f32 = weight.float()
        # pad order for 3d: (W_l, W_r, H_l, H_r, D_l, D_r)
        x_pad = Fn.pad(x_f32, (padding, padding, padding, padding, padding, padding))
        x_unfold = x_pad.unfold(2, KD, stride).unfold(3, KH, stride).unfold(4, KW, stride)
        # [N, C_IN, D_OUT, H_OUT, W_OUT, KD, KH, KW]

        start_ev = None
        end_ev = None
        if return_ms:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()

        mask_channel_cache = {}
        tiles_per_n = N_TILES_D * N_TILES_H * N_TILES_W

        for tile_id in active_tile_ids.tolist():
            tile_id = int(tile_id)
            n_idx = tile_id // tiles_per_n
            rem = tile_id % tiles_per_n
            td = rem // (N_TILES_H * N_TILES_W)
            rem2 = rem % (N_TILES_H * N_TILES_W)
            th = rem2 // N_TILES_W
            tw = rem2 % N_TILES_W

            d0 = td * BD
            h0 = th * BH
            w0 = tw * BW
            d1 = min(d0 + BD, D_OUT)
            h1 = min(h0 + BH, H_OUT)
            w1 = min(w0 + BW, W_OUT)
            if d1 <= d0 or h1 <= h0 or w1 <= w0:
                continue

            ag_mask = int(ag_mask_buf[tile_id].item())
            if ag_mask == 0:
                continue

            if ag_mask == ALL_ONES:
                x_tile = x_unfold[n_idx, :, d0:d1, h0:h1, w0:w1, :, :, :]
                x_mat = x_tile.permute(1, 2, 3, 0, 4, 5, 6).reshape((d1 - d0) * (h1 - h0) * (w1 - w0), C_IN * KD * KH * KW)
                w_mat = w_f32.reshape(C_OUT, C_IN * KD * KH * KW)
            else:
                c_idx = mask_channel_cache.get(ag_mask, None)
                if c_idx is None:
                    channels = _decode_active_groups(ag_mask, NUM_GROUPS, GROUP_SIZE_C, C_IN)
                    if not channels:
                        continue
                    c_idx = torch.tensor(channels, dtype=torch.long, device=device)
                    mask_channel_cache[ag_mask] = c_idx

                x_tile = x_unfold[n_idx, c_idx, d0:d1, h0:h1, w0:w1, :, :, :]
                c_sel = c_idx.numel()
                x_mat = x_tile.permute(1, 2, 3, 0, 4, 5, 6).reshape((d1 - d0) * (h1 - h0) * (w1 - w0), c_sel * KD * KH * KW)
                w_mat = w_f32[:, c_idx, :, :, :].reshape(C_OUT, c_sel * KD * KH * KW)

            y_tile = torch.matmul(x_mat, w_mat.t())  # [tile_voxels, C_OUT]
            y_tile = y_tile.reshape(d1 - d0, h1 - h0, w1 - w0, C_OUT).permute(3, 0, 1, 2).contiguous()
            y[n_idx, :, d0:d1, h0:h1, w0:w1] = y_tile

        ms = 0.0
        if return_ms:
            end_ev.record()
            torch.cuda.synchronize(device)
            ms = start_ev.elapsed_time(end_ev)
    else:
        ms = 0.0

    if bias is not None:
        y = y + bias.float().view(1, -1, 1, 1, 1)

    stats = None
    if return_tile_stats:
        tc = tile_class_buf[:TOTAL_TILES]
        stats = {
            "backend": "sparse_active_tiles" if active_tile_count > 0 else "zero_tiles_only",
            "reason": "ok" if active_tile_count > 0 else "all_tiles_zero_after_metadata",
            "fallback": False,
            "total_tiles": TOTAL_TILES,
            "active_tiles": int(active_tile_count),
            "zero_tiles": int((tc == TILE_ZERO).sum().item()),
            "sparse_tiles": int((tc == TILE_SPARSE).sum().item()),
            "denseish_tiles": int((tc == TILE_DENSEISH).sum().item()),
            "avg_active_group_ratio": avg_active_group_ratio,
            "prescan_version": "conv3d_v2_active_tile_sparse",
        }

    backend_meta = {
        "backend": "sparse_active_tiles" if active_tile_count > 0 else "zero_tiles_only",
        "reason": "ok" if active_tile_count > 0 else "all_tiles_zero_after_metadata",
        "active_tiles": int(active_tile_count),
        "total_tiles": TOTAL_TILES,
    }
    return _finalize_return(y, ms, stats, backend_meta)
