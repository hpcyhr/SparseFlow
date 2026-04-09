"""
SparseFlow Utils/sparse_helpers.py — Shared prescan metadata primitives.

Leaf-level module that provides:
  - Tile classification constants (single source of truth for the repo).
  - choose_group_size() : reduction-dim group sizing.
  - popcount_buf()      : vectorised SWAR popcount on an int32 bitmask.

Round 6 cleanup: deleted the DEPRECATED row-geometry single-stage prescan
(`select_row_tile_sizes`, `_prescan_rows_kernel`, `build_row_metadata`).
Kernels/matmul.py was the last remaining caller and has been migrated to
the three-stage prescan pipeline exported by Kernels/linear.py.

This file must remain small and stable — it is a leaf dependency.
"""

import torch


# ---------------------------------------------------------------------------
# Tile classification constants (single source of truth for the repo)
# ---------------------------------------------------------------------------
# Values 0..2 are the final classes produced by the three-stage prescan and
# consumed by the compute kernels. Values 3..4 are intermediate classes used
# internally by the three-stage prescan pipeline in Kernels/conv2d.py and
# Kernels/linear.py; they never appear in the final tile_class buffer at
# compute time.
TILE_ZERO           = 0
TILE_SPARSE         = 1
TILE_DENSEISH       = 2
TILE_UNCERTAIN      = 3
TILE_ZERO_CANDIDATE = 4


# ---------------------------------------------------------------------------
# Group-size selection
# ---------------------------------------------------------------------------

def choose_group_size(k_dim: int) -> int:
    """Select GROUP_SIZE for the reduction dimension.

    Convention: 16 for k<=128, else 32, then capped upward so NUM_GROUPS
    never exceeds 32 (keeps the active-group bitmask in a single uint32).
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
# Vectorised bitmask popcount
# ---------------------------------------------------------------------------

def popcount_buf(ag_mask_buf: torch.Tensor, count: int) -> torch.Tensor:
    """SWAR popcount on a prefix of an int32 bitmask buffer.

    Args:
        ag_mask_buf: int32 tensor holding one bitmask per tile.
        count:       number of entries to process from the front of the buffer.

    Returns:
        int32 tensor of shape [count] with the popcount of each entry.
    """
    v = ag_mask_buf[:count].int()
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F
    v = v + (v >> 8)
    v = v + (v >> 16)
    return (v & 0x3F).to(torch.int32)