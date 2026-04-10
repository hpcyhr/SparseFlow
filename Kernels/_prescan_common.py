"""
SparseFlow internal shared receptive-field prescan helpers.

This module centralizes the PyTorch vectorized three-stage prescan used by
Conv1d / Conv2d / Conv3d sparse kernels. It is intentionally internal-only.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from Utils.sparse_helpers import (
    TILE_DENSEISH,
    TILE_SPARSE,
    TILE_UNCERTAIN,
    TILE_ZERO,
    TILE_ZERO_CANDIDATE,
)


def _normalize_tuple(value, ndim: int) -> Tuple[int, ...]:
    if isinstance(value, int):
        return (int(value),) * ndim
    if len(value) != ndim:
        raise ValueError(f"Expected {ndim} values, got {value}")
    return tuple(int(v) for v in value)


def _pack_group_mask(group_active: torch.Tensor) -> torch.Tensor:
    num_groups = int(group_active.shape[-1])
    bit_weights = (1 << torch.arange(num_groups, device=group_active.device, dtype=torch.int32))
    return (group_active.to(torch.int32) * bit_weights).sum(dim=-1).to(torch.int32)


def _extract_rf_windows(
    x_channels_last: torch.Tensor,
    kernel_dims: Tuple[int, ...],
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    spatial_ndim = len(kernel_dims)
    x_channels_first = x_channels_last.movedim(-1, 1).contiguous()

    if any(padding):
        pad_args = []
        for pad in reversed(padding):
            pad_args.extend([pad, pad])
        x_channels_first = F.pad(x_channels_first, tuple(pad_args))

    unfolded = x_channels_first
    for dim, (kernel_size, step) in enumerate(zip(kernel_dims, stride), start=2):
        unfolded = unfolded.unfold(dim, kernel_size, step)

    out_dims = tuple(int(unfolded.shape[dim]) for dim in range(2, 2 + spatial_ndim))
    permute_order = (
        [0]
        + list(range(2, 2 + spatial_ndim))
        + list(range(2 + spatial_ndim, 2 + 2 * spatial_ndim))
        + [1]
    )
    patches = unfolded.permute(permute_order).contiguous()
    return patches, out_dims


def _tile_rf_windows(
    patches: torch.Tensor,
    spatial_dims: Tuple[int, ...],
    block_dims: Tuple[int, ...],
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    spatial_ndim = len(spatial_dims)
    kernel_dims = tuple(int(patches.shape[1 + spatial_ndim + i]) for i in range(spatial_ndim))
    padded_spatial = tuple(
        int(math.ceil(float(spatial) / float(block)) * block)
        for spatial, block in zip(spatial_dims, block_dims)
    )
    grid_dims = tuple(padded // block for padded, block in zip(padded_spatial, block_dims))

    padded_shape = (int(patches.shape[0]),) + padded_spatial + kernel_dims + (int(patches.shape[-1]),)
    padded = patches.new_zeros(padded_shape)
    valid_index = (
        (slice(None),)
        + tuple(slice(0, spatial) for spatial in spatial_dims)
        + tuple(slice(None) for _ in kernel_dims)
        + (slice(None),)
    )
    padded[valid_index] = patches

    reshape_dims = [int(patches.shape[0])]
    for padded_spatial_dim, block_dim in zip(padded_spatial, block_dims):
        reshape_dims.extend([padded_spatial_dim // block_dim, block_dim])
    reshape_dims.extend(kernel_dims)
    reshape_dims.append(int(patches.shape[-1]))
    tiled = padded.reshape(*reshape_dims)

    permute_order = (
        [0]
        + [1 + 2 * i for i in range(spatial_ndim)]
        + [2 + 2 * i for i in range(spatial_ndim)]
        + list(range(1 + 2 * spatial_ndim, 1 + 3 * spatial_ndim))
        + [1 + 3 * spatial_ndim]
    )
    tiled = tiled.permute(permute_order).contiguous()

    rf_elems = int(math.prod(block_dims) * math.prod(kernel_dims))
    tile_shape = (int(patches.shape[0]),) + grid_dims
    tile_flat = tiled.reshape(*tile_shape, rf_elems, int(patches.shape[-1]))
    return tile_flat, grid_dims


def _build_rf_prescan_metadata_impl(
    x_channels_last: torch.Tensor,
    spatial_dims: Tuple[int, ...],
    kernel_dims: Tuple[int, ...],
    stride,
    padding,
    block_dims: Tuple[int, ...],
    group_size_c: int,
    num_groups: int,
    threshold: float,
    return_debug_stats: bool = False,
):
    spatial_ndim = len(spatial_dims)
    if x_channels_last.ndim != spatial_ndim + 2:
        raise ValueError(
            f"Expected channels-last tensor with {spatial_ndim + 2} dims, got {tuple(x_channels_last.shape)}"
        )

    stride_t = _normalize_tuple(stride, spatial_ndim)
    padding_t = _normalize_tuple(padding, spatial_ndim)

    patches, out_dims = _extract_rf_windows(x_channels_last, kernel_dims, stride_t, padding_t)
    if tuple(int(v) for v in out_dims) != tuple(int(v) for v in spatial_dims):
        raise ValueError(
            f"Output spatial mismatch: expected {tuple(spatial_dims)}, got {tuple(out_dims)}"
        )

    tile_flat, grid_dims = _tile_rf_windows(patches, tuple(int(v) for v in spatial_dims), block_dims)
    c_in = int(tile_flat.shape[-1])
    n_tiles = int(tile_flat.numel() // (tile_flat.shape[-2] * tile_flat.shape[-1]))
    tile_view = tile_flat.reshape(n_tiles, int(tile_flat.shape[-2]), c_in)

    rep_idx = torch.arange(num_groups, device=tile_flat.device, dtype=torch.long) * int(group_size_c)
    rep_vals = tile_view.index_select(-1, rep_idx)
    rep_group_active = (rep_vals.abs() > float(threshold)).any(dim=1)
    rough_mask = _pack_group_mask(rep_group_active)
    all_ones_mask = (1 << num_groups) - 1

    stage1_zero_candidate = rough_mask == 0
    stage1_denseish = rough_mask == all_ones_mask
    stage1_uncertain = ~(stage1_zero_candidate | stage1_denseish)

    any_nonzero = (
        tile_view.abs() > float(threshold)
    ).reshape(n_tiles, -1).any(dim=1)
    exact_zero = stage1_zero_candidate & (~any_nonzero)
    stage3_needed = stage1_uncertain | (stage1_zero_candidate & any_nonzero)

    padded_c = num_groups * int(group_size_c)
    if padded_c != c_in:
        tile_view_padded = tile_view.new_zeros((n_tiles, int(tile_view.shape[1]), padded_c))
        tile_view_padded[..., :c_in] = tile_view
    else:
        tile_view_padded = tile_view
    group_view = tile_view_padded.reshape(n_tiles, int(tile_view.shape[1]), num_groups, int(group_size_c))
    exact_group_active = (group_view.abs() > float(threshold)).any(dim=-1).any(dim=1)
    exact_mask = _pack_group_mask(exact_group_active)

    ag_mask = torch.where(stage3_needed, exact_mask, rough_mask.clone())
    ag_mask = torch.where(exact_zero, torch.zeros_like(ag_mask), ag_mask).to(torch.int32)

    tile_class = torch.full_like(ag_mask, fill_value=TILE_SPARSE, dtype=torch.int32)
    tile_class = torch.where(exact_zero, torch.full_like(tile_class, TILE_ZERO), tile_class)
    tile_class = torch.where(stage1_denseish, torch.full_like(tile_class, TILE_DENSEISH), tile_class)
    tile_class = torch.where(ag_mask == 0, torch.full_like(tile_class, TILE_ZERO), tile_class)
    tile_class = torch.where(
        (ag_mask == all_ones_mask) & (~exact_zero),
        torch.full_like(tile_class, TILE_DENSEISH),
        tile_class,
    )
    tile_class = torch.where(
        (tile_class != TILE_ZERO) & (tile_class != TILE_DENSEISH),
        torch.full_like(tile_class, TILE_SPARSE),
        tile_class,
    )

    debug_stats: Optional[Dict[str, float]] = None
    if return_debug_stats:
        stage1_active_count = int(rep_group_active.to(torch.int32).sum().item())
        debug_stats = {
            "stage1_zero_candidate": int(stage1_zero_candidate.to(torch.int32).sum().item()),
            "stage1_denseish": int(stage1_denseish.to(torch.int32).sum().item()),
            "stage1_uncertain": int(stage1_uncertain.to(torch.int32).sum().item()),
            "stage1_avg_active_group_ratio_lower_bound": float(stage1_active_count) / max(float(n_tiles * num_groups), 1.0),
            "final_zero": int((tile_class == TILE_ZERO).to(torch.int32).sum().item()),
            "final_sparse": int((tile_class == TILE_SPARSE).to(torch.int32).sum().item()),
            "final_denseish": int((tile_class == TILE_DENSEISH).to(torch.int32).sum().item()),
            "stage2_zero_refine_tiles": int(stage1_zero_candidate.to(torch.int32).sum().item()),
            "stage2_uncertain_tiles": int(stage3_needed.to(torch.int32).sum().item()),
        }

    return tile_class.contiguous(), ag_mask.contiguous(), debug_stats


def build_rf_prescan_metadata(
    x_channels_last: torch.Tensor,
    spatial_dims: Tuple[int, ...],
    kernel_dims: Tuple[int, ...],
    stride,
    padding,
    block_dims: Tuple[int, ...],
    group_size_c: int,
    num_groups: int,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return final (tile_class, ag_mask) metadata on device without host sync."""
    tile_class, ag_mask, _ = _build_rf_prescan_metadata_impl(
        x_channels_last=x_channels_last,
        spatial_dims=spatial_dims,
        kernel_dims=kernel_dims,
        stride=stride,
        padding=padding,
        block_dims=block_dims,
        group_size_c=group_size_c,
        num_groups=num_groups,
        threshold=threshold,
        return_debug_stats=False,
    )
    return tile_class, ag_mask
