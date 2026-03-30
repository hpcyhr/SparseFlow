"""
SparseFlow Kernels/conv1d.py 鈥?Sparse Conv1d Triton Kernel v1.0

1D convolution [N, C_IN, L] 鈫?[N, C_OUT, L_OUT] with grouped-bitmask
sparsity exploitation on the input channel dimension.

Follows the same prescan + sparse-compute pattern as conv2d.py.
Useful for temporal spike processing in 1D SNN layers.

Supported: kernel_size={1,3,5,7}, stride=1, groups=1, dilation=1.
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
    TILE_ZERO, TILE_SPARSE, TILE_DENSEISH,
    choose_group_size, popcount_buf,
)

FALLBACK_RATIO = 0.85


# ---------------------------------------------------------------------------
# Prescan: per (N, tile_l) 鈫?grouped bitmask over C_IN
# ---------------------------------------------------------------------------

@triton.jit
def _prescan_conv1d_kernel(
    x_ptr,              # [N, C_IN, L_IN]
    ag_mask_ptr,        # [N * N_TILES_L]
    tile_class_ptr,     # [N * N_TILES_L]
    N_batch: tl.constexpr,
    C_IN: tl.constexpr,
    L_IN: tl.constexpr,
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


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def sparse_conv1d_forward(
    x: torch.Tensor,        # [N, C_IN, L]
    weight: torch.Tensor,   # [C_OUT, C_IN, KS]
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    threshold: float = 1e-6,
    return_ms: bool = False,
    return_tile_stats: bool = False,
    fallback_ratio: float = FALLBACK_RATIO,
):
    """
    Sparse 1D convolution.

    First version: prescan 鈫?classify tiles 鈫?F.conv1d on full input
    (dense fallback with tile stats for profiling).
    Sparse compute kernel to follow in v2 once tile distribution is validated.
    """
    import torch.nn.functional as Fn

    N_batch, C_IN, L_IN = x.shape
    C_OUT = weight.shape[0]
    KS = weight.shape[2]
    device = x.device

    if isinstance(stride, tuple):
        stride = stride[0]
    if isinstance(padding, tuple):
        padding = padding[0]

    L_OUT = (L_IN + 2 * padding - KS) // stride + 1

    if L_OUT <= 0:
        y = Fn.conv1d(
            x.float(),
            weight.float(),
            bias.float() if bias is not None else None,
            stride=stride,
            padding=padding,
        ).float()
        ret = (y, 0.0)
        if return_tile_stats:
            ret = ret + (None,)
        return ret

    GROUP_SIZE_C = choose_group_size(C_IN)
    NUM_GROUPS = triton.cdiv(C_IN, GROUP_SIZE_C)
    ALL_ONES = (1 << NUM_GROUPS) - 1

    BL = min(32, L_OUT)
    N_TILES_L = triton.cdiv(L_OUT, BL)
    TOTAL_TILES = N_batch * N_TILES_L

    x_f16 = x.half().contiguous()
    ag_mask_buf = torch.empty(TOTAL_TILES, dtype=torch.int32, device=device)
    tile_class_buf = torch.empty(TOTAL_TILES, dtype=torch.int32, device=device)

    # Prescan
    prescan_failed = False
    try:
        _prescan_conv1d_kernel[(TOTAL_TILES,)](
            x_f16, ag_mask_buf, tile_class_buf,
            N_batch=N_batch, C_IN=C_IN, L_IN=L_IN,
            KS=KS, STRIDE=stride, PADDING=padding,
            BL=BL, N_TILES_L=N_TILES_L,
            GROUP_SIZE_C=GROUP_SIZE_C, NUM_GROUPS=NUM_GROUPS,
            ALL_ONES=ALL_ONES, THRESHOLD=threshold,
        )
    except Exception:
        prescan_failed = True

    # v1: dense compute with tile stats reporting
    if return_ms:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    y = Fn.conv1d(
        x.float(),
        weight.float(),
        bias.float() if bias is not None else None,
        stride=stride,
        padding=padding,
    ).float()

    ms = 0.0
    if return_ms:
        end.record()
        torch.cuda.synchronize(device)
        ms = start.elapsed_time(end)

    stats = None
    if return_tile_stats:
        if prescan_failed:
            stats = {
                'total_tiles': TOTAL_TILES,
                'fallback': True,
                'reason': 'prescan_failed',
                'prescan_version': 'conv1d_v1_prescan_only',
            }
        else:
            tc = tile_class_buf[:TOTAL_TILES]
            stats = {
                'total_tiles': TOTAL_TILES,
                'zero_tiles': int((tc == TILE_ZERO).sum().item()),
                'sparse_tiles': int((tc == TILE_SPARSE).sum().item()),
                'denseish_tiles': int((tc == TILE_DENSEISH).sum().item()),
                'prescan_version': 'conv1d_v1_prescan_only',
                'fallback': False,
            }

    ret = (y, ms)
    if return_tile_stats:
        ret = ret + (stats,)
    return ret
