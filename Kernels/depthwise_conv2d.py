"""
SparseFlow Kernels/depthwise_conv2d.py

Depthwise Conv2d is treated as the special case of grouped Conv2d where:
  groups == C_in == C_out

This keeps depthwise aligned with the same SparseFlow method used by:
  conv1d / conv2d / conv3d / linear / grouped_conv2d
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from Utils.config import PRESCAN_ACTIVITY_EPS, SPARSE_DENSE_RATIO_THRESHOLD
from Kernels.grouped_conv2d import sparse_grouped_conv2d_forward

FALLBACK_RATIO = SPARSE_DENSE_RATIO_THRESHOLD


def sparse_depthwise_conv2d_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride=1,
    padding=1,
    dilation=1,
    threshold: float = PRESCAN_ACTIVITY_EPS,
    w_cl_groups: Optional[Sequence[torch.Tensor]] = None,
    ag_mask_bufs: Optional[Sequence[torch.Tensor]] = None,
    tile_class_bufs: Optional[Sequence[torch.Tensor]] = None,
    active_tile_ids_bufs: Optional[Sequence[torch.Tensor]] = None,
    return_ms: bool = False,
    fallback_ratio: float = FALLBACK_RATIO,
    return_avg_active_ratio: bool = False,
    return_tile_stats: bool = False,
    return_backend_meta: bool = False,
    launch_all_tiles: bool = False,
):
    if x.ndim != 4 or weight.ndim != 4:
        raise ValueError("depthwise_conv2d expects x=[N,C,H,W], weight=[C,1,KH,KW]")

    channels = int(x.shape[1])
    if int(weight.shape[0]) != channels or int(weight.shape[1]) != 1:
        raise ValueError("depthwise_conv2d requires weight shape [C,1,KH,KW] with C == x.shape[1]")

    kernel_size = (int(weight.shape[2]), int(weight.shape[3]))
    use_sparse_path = any(
        v is not None for v in (w_cl_groups, ag_mask_bufs, tile_class_bufs, active_tile_ids_bufs)
    )

    def _finalize_return(y, ms, avg_active_ratio_val=None, tile_stats_val=None, backend_meta_val=None):
        ret = (y, ms)
        if return_avg_active_ratio:
            ret = ret + (avg_active_ratio_val,)
        if return_tile_stats:
            ret = ret + (tile_stats_val,)
        if return_backend_meta:
            ret = ret + (backend_meta_val,)
        return ret

    if not use_sparse_path:
        dense_ms = 0.0
        if return_ms:
            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            se.record()

        y = F.conv2d(
            x,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )

        if return_ms:
            ee.record()
            torch.cuda.synchronize(x.device)
            dense_ms = se.elapsed_time(ee)

        backend_meta = {
            "backend": "dense_fallback",
            "reason": "depthwise_direct_reference_path",
        }
        avg_ratio = 1.0 if return_avg_active_ratio else None
        return _finalize_return(y, dense_ms, avg_ratio, None, backend_meta)

    return sparse_grouped_conv2d_forward(
        x=x,
        weight=weight,
        bias=bias,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=channels,
        threshold=threshold,
        w_cl_groups=w_cl_groups,
        ag_mask_bufs=ag_mask_bufs,
        tile_class_bufs=tile_class_bufs,
        active_tile_ids_bufs=active_tile_ids_bufs,
        return_ms=return_ms,
        fallback_ratio=fallback_ratio,
        return_avg_active_ratio=return_avg_active_ratio,
        return_tile_stats=return_tile_stats,
        return_backend_meta=return_backend_meta,
        launch_all_tiles=launch_all_tiles,
    )
