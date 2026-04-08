"""
SparseFlow Kernels/conv1d.py - Sparse Conv1d kernel.

Maturity: main_path (active-tile sparse execution).

Strategy:
1) Prescan per output tile to build active channel-group bitmask.
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


# ---------------------------------------------------------------------------
# Prescan: per (N, tile_l) -> grouped bitmask over C_IN
# ---------------------------------------------------------------------------


@triton.jit
def _prescan_conv1d_kernel(
    x_ptr,  # [N, C_IN, L_IN]
    ag_mask_ptr,  # [N * N_TILES_L]
    tile_class_ptr,  # [N * N_TILES_L]
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
):
    pid = tl.program_id(0)
    n_idx = pid // N_TILES_L
    tile_idx = pid % N_TILES_L

    l_out_start = tile_idx * BL
    off1 = tl.arange(0, 1)
    ag_mask = tl.zeros([1], dtype=tl.int32)
    any_nonzero = tl.zeros([1], dtype=tl.int32)

    for g in range(NUM_GROUPS):
        g_start = g * GROUP_SIZE_C
        group_has_nonzero = tl.zeros([1], dtype=tl.int32)

        for bl in range(BL):
            l_out = l_out_start + bl
            if l_out < L_OUT:
                for k in range(KS):
                    l_in = l_out * STRIDE - PADDING + k
                    if (l_in >= 0) and (l_in < L_IN):
                        for ci in range(GROUP_SIZE_C):
                            c = g_start + ci
                            if c < C_IN:
                                addr = n_idx * C_IN * L_IN + c * L_IN + l_in
                                val = tl.load(x_ptr + addr)
                                if tl.abs(val) > THRESHOLD:
                                    group_has_nonzero = tl.full([1], 1, dtype=tl.int32)

        if tl.sum(group_has_nonzero) != 0:
            ag_mask = ag_mask + group_has_nonzero * (1 << g)
            any_nonzero = tl.full([1], 1, dtype=tl.int32)

    out_idx = n_idx * N_TILES_L + tile_idx
    tl.store(ag_mask_ptr + out_idx + off1, ag_mask)

    if tl.sum(any_nonzero) == 0:
        tl.store(tile_class_ptr + out_idx + off1, tl.zeros([1], dtype=tl.int32))
    else:
        if tl.sum(ag_mask == ALL_ONES) > 0:
            tl.store(tile_class_ptr + out_idx + off1, tl.full([1], TILE_DENSEISH, dtype=tl.int32))
        else:
            tl.store(tile_class_ptr + out_idx + off1, tl.full([1], TILE_SPARSE, dtype=tl.int32))


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


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def sparse_conv1d_forward(
    x: torch.Tensor,  # [N, C_IN, L]
    weight: torch.Tensor,  # [C_OUT, C_IN, KS]
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    threshold: float = 1e-6,
    return_ms: bool = False,
    return_tile_stats: bool = False,
    return_backend_meta: bool = False,
    fallback_ratio: float = FALLBACK_RATIO,
):
    """
    Sparse 1D convolution with active-tile sparse execution.

    Compute path:
    - prescan metadata (group bitmask per tile)
    - active tile compaction
    - active-group-only GEMM per active tile
    """
    import torch.nn.functional as Fn

    def _finalize_return(y, ms, stats=None, backend_meta=None):
        ret = (y, ms)
        if return_tile_stats:
            ret = ret + (stats,)
        if return_backend_meta:
            ret = ret + (backend_meta or {},)
        return ret

    if isinstance(stride, tuple):
        stride = stride[0]
    if isinstance(padding, tuple):
        padding = padding[0]

    N_batch, C_IN, L_IN = x.shape
    C_OUT = weight.shape[0]
    KS = weight.shape[2]
    device = x.device

    L_OUT = (L_IN + 2 * padding - KS) // stride + 1
    if L_OUT <= 0:
        y = Fn.conv1d(
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
            "prescan_version": "conv1d_v2_active_tile_sparse",
        }
        return _finalize_return(y, 0.0, stats, {"backend": "dense_fallback", "reason": "invalid_output_shape"})

    GROUP_SIZE_C = choose_group_size(C_IN)
    NUM_GROUPS = triton.cdiv(C_IN, GROUP_SIZE_C)
    ALL_ONES = (1 << NUM_GROUPS) - 1

    BL = min(32, max(1, L_OUT))
    N_TILES_L = triton.cdiv(L_OUT, BL)
    TOTAL_TILES = N_batch * N_TILES_L

    x_f16 = x if (x.dtype == torch.float16 and x.is_contiguous()) else x.half().contiguous()
    ag_mask_buf = torch.empty(TOTAL_TILES, dtype=torch.int32, device=device)
    tile_class_buf = torch.empty(TOTAL_TILES, dtype=torch.int32, device=device)

    try:
        _prescan_conv1d_kernel[(TOTAL_TILES,)](
            x_f16,
            ag_mask_buf,
            tile_class_buf,
            C_IN=C_IN,
            L_IN=L_IN,
            L_OUT=L_OUT,
            KS=KS,
            STRIDE=stride,
            PADDING=padding,
            BL=BL,
            N_TILES_L=N_TILES_L,
            GROUP_SIZE_C=GROUP_SIZE_C,
            NUM_GROUPS=NUM_GROUPS,
            ALL_ONES=ALL_ONES,
            THRESHOLD=threshold,
        )
    except Exception:
        y = Fn.conv1d(
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
            "prescan_version": "conv1d_v2_active_tile_sparse",
        }
        return _finalize_return(y, 0.0, stats, {"backend": "dense_fallback", "reason": "prescan_failed"})

    # Dense fallback when active-group ratio is high.
    avg_active_group_ratio = 1.0
    if NUM_GROUPS > 0:
        pc = popcount_buf(ag_mask_buf, TOTAL_TILES)
        avg_active_group_ratio = float(pc.sum().item()) / max(float(TOTAL_TILES * NUM_GROUPS), 1.0)
    if avg_active_group_ratio > fallback_ratio:
        y = Fn.conv1d(
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
            "prescan_version": "conv1d_v2_active_tile_sparse",
        }
        return _finalize_return(y, 0.0, stats, {"backend": "dense_fallback", "reason": "post_metadata_dense_fallback"})

    active_tile_ids, active_tile_count = _build_active_tile_ids(tile_class_buf, TOTAL_TILES)

    # Compute output for active tiles only.
    y = torch.zeros(N_batch, C_OUT, L_OUT, dtype=torch.float32, device=device)
    if active_tile_count > 0:
        x_f32 = x.float()
        w_f32 = weight.float()
        x_pad = Fn.pad(x_f32, (padding, padding))
        x_unfold = x_pad.unfold(2, KS, stride)  # [N, C_IN, L_OUT, KS]

        start_ev = None
        end_ev = None
        if return_ms:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()

        mask_channel_cache = {}
        for tile_id in active_tile_ids.tolist():
            tile_id = int(tile_id)
            n_idx = tile_id // N_TILES_L
            tile_idx = tile_id % N_TILES_L
            l0 = tile_idx * BL
            l1 = min(l0 + BL, L_OUT)
            if l1 <= l0:
                continue

            ag_mask = int(ag_mask_buf[tile_id].item())
            if ag_mask == 0:
                continue

            if ag_mask == ALL_ONES:
                x_tile = x_unfold[n_idx, :, l0:l1, :]  # [C_IN, L_tile, KS]
                x_mat = x_tile.permute(1, 0, 2).reshape(l1 - l0, C_IN * KS)
                w_mat = w_f32.reshape(C_OUT, C_IN * KS)
            else:
                c_idx = mask_channel_cache.get(ag_mask, None)
                if c_idx is None:
                    channels = _decode_active_groups(ag_mask, NUM_GROUPS, GROUP_SIZE_C, C_IN)
                    if not channels:
                        continue
                    c_idx = torch.tensor(channels, dtype=torch.long, device=device)
                    mask_channel_cache[ag_mask] = c_idx
                x_tile = x_unfold[n_idx, c_idx, l0:l1, :]
                x_mat = x_tile.permute(1, 0, 2).reshape(l1 - l0, c_idx.numel() * KS)
                w_mat = w_f32[:, c_idx, :].reshape(C_OUT, c_idx.numel() * KS)

            y_tile = torch.matmul(x_mat, w_mat.t())  # [L_tile, C_OUT]
            y[n_idx, :, l0:l1] = y_tile.transpose(0, 1)

        ms = 0.0
        if return_ms:
            end_ev.record()
            torch.cuda.synchronize(device)
            ms = start_ev.elapsed_time(end_ev)
    else:
        ms = 0.0

    if bias is not None:
        y = y + bias.float().view(1, -1, 1)

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
            "prescan_version": "conv1d_v2_active_tile_sparse",
        }

    backend_meta = {
        "backend": "sparse_active_tiles" if active_tile_count > 0 else "zero_tiles_only",
        "reason": "ok" if active_tile_count > 0 else "all_tiles_zero_after_metadata",
        "active_tiles": int(active_tile_count),
        "total_tiles": TOTAL_TILES,
    }
    return _finalize_return(y, ms, stats, backend_meta)
