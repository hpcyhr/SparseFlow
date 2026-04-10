"""
SparseFlow internal shared receptive-field prescan helpers.

This module centralizes the PyTorch vectorized prescan used by Conv1d /
Conv2d / Conv3d sparse kernels. The implementation intentionally uses
group-wise channel reduction plus max_pool*d instead of unfold-based receptive
field materialization.

Change-log:
v3: added in-place output buffer support (tile_class_out / ag_mask_out)
to eliminate device-to-device memcpy after prescan.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from Utils.sparse_helpers import TILE_DENSEISH, TILE_SPARSE, TILE_ZERO


def _tuple(value, ndim: int) -> Tuple[int, ...]:
    if isinstance(value, int):
        return (int(value),) * ndim
    if len(value) != ndim:
        raise ValueError(f"Expected {ndim} values, got {value}")
    return tuple(int(v) for v in value)


def _groupwise_amax_last_dim(
    x_channels_last: torch.Tensor,
    group_size_c: int,
    num_groups: int,
) -> torch.Tensor:
    c_in = int(x_channels_last.shape[-1])
    c_padded = int(num_groups) * int(group_size_c)
    if c_in < c_padded:
        x_channels_last = F.pad(x_channels_last, (0, c_padded - c_in))
    elif c_in > c_padded:
        raise ValueError(f"C_in={c_in} exceeds num_groups*group_size_c={c_padded}")

    grouped_shape = tuple(x_channels_last.shape[:-1]) + (int(num_groups), int(group_size_c))
    return x_channels_last.abs().reshape(grouped_shape).amax(dim=-1)


def _pack_group_active(group_active_last: torch.Tensor) -> torch.Tensor:
    num_groups = int(group_active_last.shape[-1])
    bit_weights = torch.bitwise_left_shift(
        torch.ones(num_groups, device=group_active_last.device, dtype=torch.int32),
        torch.arange(num_groups, device=group_active_last.device, dtype=torch.int32),
    )
    return (group_active_last.to(torch.int32) * bit_weights).sum(dim=-1).to(torch.int32).reshape(-1)


def _classify_tiles(
    group_active_last: torch.Tensor,
    return_debug_stats: bool,
    tile_class_out: Optional[torch.Tensor] = None,
    ag_mask_out: Optional[torch.Tensor] = None,
):
    # group_active_last shape: [N, *tile_grid, num_groups]
    ag_mask = _pack_group_active(group_active_last)
    any_active = group_active_last.any(dim=-1).reshape(-1)
    all_active = group_active_last.all(dim=-1).reshape(-1)

    tile_class = torch.full_like(ag_mask, TILE_SPARSE, dtype=torch.int32)
    tile_class = torch.where(~any_active, torch.full_like(tile_class, TILE_ZERO), tile_class)
    tile_class = torch.where(all_active, torch.full_like(tile_class, TILE_DENSEISH), tile_class)
    n_tiles = int(ag_mask.numel())

    debug_stats: Optional[Dict[str, float]] = None
    if return_debug_stats:
        zero_tiles = int((tile_class == TILE_ZERO).sum().item())
        denseish_tiles = int((tile_class == TILE_DENSEISH).sum().item())
        sparse_tiles = int((tile_class == TILE_SPARSE).sum().item())
        debug_stats = {
            "stage1_zero_candidate": zero_tiles,
            "stage1_denseish": denseish_tiles,
            "stage1_uncertain": sparse_tiles,
            "stage1_avg_active_group_ratio_lower_bound": float(group_active_last.float().mean().item()),
            "final_zero": zero_tiles,
            "final_sparse": sparse_tiles,
            "final_denseish": denseish_tiles,
            "stage2_zero_refine_tiles": 0,
            "stage2_uncertain_tiles": 0,
            "total_tiles": n_tiles,
        }

    if tile_class_out is not None:
        if int(tile_class_out.numel()) < n_tiles:
            raise ValueError(f"tile_class_out is too small: {int(tile_class_out.numel())} < {n_tiles}")
        tile_class_slice = tile_class_out[:n_tiles]
        tile_class_slice.copy_(tile_class.to(torch.int32).reshape(-1))
        tile_class = tile_class_slice
    else:
        tile_class = tile_class.contiguous()

    if ag_mask_out is not None:
        if int(ag_mask_out.numel()) < n_tiles:
            raise ValueError(f"ag_mask_out is too small: {int(ag_mask_out.numel())} < {n_tiles}")
        ag_mask_slice = ag_mask_out[:n_tiles]
        ag_mask_slice.copy_(ag_mask.to(torch.int32).reshape(-1))
        ag_mask = ag_mask_slice
    else:
        ag_mask = ag_mask.contiguous()

    return tile_class, ag_mask, debug_stats


def _can_skip_rf_pool(kernel_dims: Tuple[int, ...], stride: Tuple[int, ...], padding: Tuple[int, ...]) -> bool:
    return all(k == 1 for k in kernel_dims) and all(s == 1 for s in stride) and all(p == 0 for p in padding)


def _prescan_1d_impl(
    x_channels_last: torch.Tensor,
    spatial_dims: Tuple[int, ...],
    kernel_dims: Tuple[int, ...],
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
    block_dims: Tuple[int, ...],
    group_size_c: int,
    num_groups: int,
    threshold: float,
    return_debug_stats: bool,
    tile_class_out: Optional[torch.Tensor] = None,
    ag_mask_out: Optional[torch.Tensor] = None,
):
    group_max = _groupwise_amax_last_dim(x_channels_last, group_size_c, num_groups)
    group_max_ncl = group_max.permute(0, 2, 1).contiguous()
    if _can_skip_rf_pool(kernel_dims, stride, padding):
        rf_max = group_max_ncl
    else:
        rf_max = F.max_pool1d(
            group_max_ncl,
            kernel_size=kernel_dims[0],
            stride=stride[0],
            padding=padding[0],
        )
    if int(rf_max.shape[-1]) != int(spatial_dims[0]):
        raise ValueError(f"Output spatial mismatch: expected {spatial_dims}, got {(int(rf_max.shape[-1]),)}")
    tile_max = F.max_pool1d(
        rf_max,
        kernel_size=block_dims[0],
        stride=block_dims[0],
        ceil_mode=True,
    )
    group_active_last = (tile_max > float(threshold)).permute(0, 2, 1)
    return _classify_tiles(group_active_last, return_debug_stats, tile_class_out, ag_mask_out)


def _prescan_2d_impl(
    x_channels_last: torch.Tensor,
    spatial_dims: Tuple[int, ...],
    kernel_dims: Tuple[int, ...],
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
    block_dims: Tuple[int, ...],
    group_size_c: int,
    num_groups: int,
    threshold: float,
    return_debug_stats: bool,
    tile_class_out: Optional[torch.Tensor] = None,
    ag_mask_out: Optional[torch.Tensor] = None,
):
    group_max = _groupwise_amax_last_dim(x_channels_last, group_size_c, num_groups)
    group_max_nchw = group_max.permute(0, 3, 1, 2).contiguous()
    if _can_skip_rf_pool(kernel_dims, stride, padding):
        rf_max = group_max_nchw
    else:
        rf_max = F.max_pool2d(
            group_max_nchw,
            kernel_size=kernel_dims,
            stride=stride,
            padding=padding,
        )
    if tuple(int(v) for v in rf_max.shape[-2:]) != tuple(int(v) for v in spatial_dims):
        raise ValueError(
            f"Output spatial mismatch: expected {tuple(spatial_dims)}, got {tuple(int(v) for v in rf_max.shape[-2:])}"
        )
    tile_max = F.max_pool2d(
        rf_max,
        kernel_size=block_dims,
        stride=block_dims,
        ceil_mode=True,
    )
    group_active_last = (tile_max > float(threshold)).permute(0, 2, 3, 1)
    return _classify_tiles(group_active_last, return_debug_stats, tile_class_out, ag_mask_out)


def _prescan_3d_impl(
    x_channels_last: torch.Tensor,
    spatial_dims: Tuple[int, ...],
    kernel_dims: Tuple[int, ...],
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
    block_dims: Tuple[int, ...],
    group_size_c: int,
    num_groups: int,
    threshold: float,
    return_debug_stats: bool,
    tile_class_out: Optional[torch.Tensor] = None,
    ag_mask_out: Optional[torch.Tensor] = None,
):
    group_max = _groupwise_amax_last_dim(x_channels_last, group_size_c, num_groups)
    group_max_ncdhw = group_max.permute(0, 4, 1, 2, 3).contiguous()
    if _can_skip_rf_pool(kernel_dims, stride, padding):
        rf_max = group_max_ncdhw
    else:
        rf_max = F.max_pool3d(
            group_max_ncdhw,
            kernel_size=kernel_dims,
            stride=stride,
            padding=padding,
        )
    if tuple(int(v) for v in rf_max.shape[-3:]) != tuple(int(v) for v in spatial_dims):
        raise ValueError(
            f"Output spatial mismatch: expected {tuple(spatial_dims)}, got {tuple(int(v) for v in rf_max.shape[-3:])}"
        )
    tile_max = F.max_pool3d(
        rf_max,
        kernel_size=block_dims,
        stride=block_dims,
        ceil_mode=True,
    )
    group_active_last = (tile_max > float(threshold)).permute(0, 2, 3, 4, 1)
    return _classify_tiles(group_active_last, return_debug_stats, tile_class_out, ag_mask_out)


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
    tile_class_out: Optional[torch.Tensor] = None,
    ag_mask_out: Optional[torch.Tensor] = None,
):
    spatial_ndim = len(spatial_dims)
    if x_channels_last.ndim != spatial_ndim + 2:
        raise ValueError(
            f"Expected channels-last tensor with {spatial_ndim + 2} dims, got {tuple(x_channels_last.shape)}"
        )

    spatial_dims = _tuple(spatial_dims, spatial_ndim)
    kernel_dims = _tuple(kernel_dims, spatial_ndim)
    stride = _tuple(stride, spatial_ndim)
    padding = _tuple(padding, spatial_ndim)
    block_dims = _tuple(block_dims, spatial_ndim)

    if spatial_ndim == 1:
        return _prescan_1d_impl(
            x_channels_last, spatial_dims, kernel_dims, stride, padding,
            block_dims, group_size_c, num_groups, threshold, return_debug_stats,
            tile_class_out, ag_mask_out,
        )
    if spatial_ndim == 2:
        return _prescan_2d_impl(
            x_channels_last, spatial_dims, kernel_dims, stride, padding,
            block_dims, group_size_c, num_groups, threshold, return_debug_stats,
            tile_class_out, ag_mask_out,
        )
    if spatial_ndim == 3:
        return _prescan_3d_impl(
            x_channels_last, spatial_dims, kernel_dims, stride, padding,
            block_dims, group_size_c, num_groups, threshold, return_debug_stats,
            tile_class_out, ag_mask_out,
        )
    raise ValueError(f"Unsupported spatial_ndim={spatial_ndim}")


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
    tile_class_out: Optional[torch.Tensor] = None,
    ag_mask_out: Optional[torch.Tensor] = None,
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
        tile_class_out=tile_class_out,
        ag_mask_out=ag_mask_out,
    )
    return tile_class, ag_mask
