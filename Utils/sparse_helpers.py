"""
SparseFlow Utils/sparse_helpers.py — Shared prescan & metadata helpers

Minimal shared code factored out from the row-geometry prescan pattern
used by Linear, Matmul, and BMM kernels.  Conv2d has its own spatial
prescan and does NOT use this module.

This file must remain small and stable — it is a leaf dependency.
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Tile classification constants (same as conv2d.py / linear.py for compat)
# ---------------------------------------------------------------------------
TILE_ZERO = 0
TILE_SPARSE = 1
TILE_DENSEISH = 2


# ---------------------------------------------------------------------------
# Group-size selection (shared logic, mirrors conv2d choose_group_size)
# ---------------------------------------------------------------------------

def choose_group_size(k_dim: int) -> int:
    """
    Select GROUP_SIZE for the reduction dimension.
    Matches conv2d.py convention: 16 for k<=128, else 32, capped so
    NUM_GROUPS <= 32 (fits in a uint32 bitmask).
    """
    if k_dim <= 128:
        gs = 16
    else:
        gs = 32
    num_groups = (k_dim + gs - 1) // gs
    while num_groups > 32:
        gs *= 2
        num_groups = (k_dim + gs - 1) // gs
    return gs


# ---------------------------------------------------------------------------
# Row-geometry tile sizing
# ---------------------------------------------------------------------------

def select_row_tile_sizes(M: int, N: int):
    """
    Select (BLOCK_M, BLOCK_N) for a flat [M, N] output.
    Used by matmul / bmm kernels.
    """
    if M >= 128:
        bm = 64
    elif M >= 32:
        bm = 32
    else:
        bm = 16

    if N >= 256:
        bn = 64
    elif N >= 64:
        bn = 32
    else:
        bn = 16
    return bm, bn


# ---------------------------------------------------------------------------
# Vectorised bitmask popcount (reusable)
# ---------------------------------------------------------------------------

def popcount_buf(ag_mask_buf: torch.Tensor, count: int) -> torch.Tensor:
    """SWAR popcount on int32 bitmask buffer, returns int32 tensor."""
    v = ag_mask_buf[:count].int()
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F
    v = v + (v >> 8)
    v = v + (v >> 16)
    return (v & 0x3F).to(torch.int32)


# ---------------------------------------------------------------------------
# Row-geometry prescan kernel (1D reduction dim)
# ---------------------------------------------------------------------------

@triton.jit
def _prescan_rows_kernel(
    x_ptr,            # [M, K] input, row-major, fp16
    ag_mask_ptr,      # [N_TILES_M] output bitmask
    tile_class_ptr,   # [N_TILES_M] output class
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    ALL_ONES: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
    """
    Per-tile-row prescan: for each BLOCK_M-row tile, build a uint32
    bitmask indicating which K-groups have any nonzero element.
    """
    tile_id = tl.program_id(0)
    row_start = tile_id * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < M

    ag_mask = tl.zeros([], dtype=tl.int32)
    any_nonzero = tl.zeros([], dtype=tl.int32)

    for g in range(NUM_GROUPS):
        col_start = g * GROUP_SIZE_C
        cols = col_start + tl.arange(0, GROUP_SIZE_C)
        col_mask = cols < K

        addrs = rows[:, None] * K + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]
        vals = tl.load(x_ptr + addrs, mask=mask, other=0.0)

        has_nonzero = tl.sum(tl.abs(vals) > THRESHOLD) > 0
        if has_nonzero:
            ag_mask = ag_mask | (1 << g)
            any_nonzero = 1

    tl.store(ag_mask_ptr + tile_id, ag_mask)

    # classify
    cls = TILE_ZERO  # 0
    if any_nonzero != 0:
        if ag_mask == ALL_ONES:
            cls = TILE_DENSEISH  # 2
        else:
            cls = TILE_SPARSE    # 1
    tl.store(tile_class_ptr + tile_id, cls)


def build_row_metadata(
    x: torch.Tensor,     # [M, K] fp16, contiguous
    M: int, K: int,
    BLOCK_M: int,
    GROUP_SIZE_C: int,
    threshold: float,
    ag_mask_buf: torch.Tensor = None,
    tile_class_buf: torch.Tensor = None,
):
    """
    Run row-geometry prescan. Returns (ag_mask_buf, tile_class_buf, N_TILES).
    """
    device = x.device
    NUM_GROUPS = triton.cdiv(K, GROUP_SIZE_C)
    ALL_ONES = (1 << NUM_GROUPS) - 1
    N_TILES = triton.cdiv(M, BLOCK_M)

    if ag_mask_buf is None or ag_mask_buf.numel() < N_TILES:
        ag_mask_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
    if tile_class_buf is None or tile_class_buf.numel() < N_TILES:
        tile_class_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)

    _prescan_rows_kernel[(N_TILES,)](
        x, ag_mask_buf, tile_class_buf,
        M=M, K=K, BLOCK_M=BLOCK_M,
        GROUP_SIZE_C=GROUP_SIZE_C,
        NUM_GROUPS=NUM_GROUPS,
        ALL_ONES=ALL_ONES,
        THRESHOLD=threshold,
    )
    return ag_mask_buf, tile_class_buf, N_TILES