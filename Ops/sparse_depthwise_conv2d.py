"""
SparseFlow Ops/sparse_depthwise_conv2d.py

Depthwise Conv2d is implemented as the special case of SparseGroupedConv2d
with groups == in_channels == out_channels.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch.nn as nn

from Utils.config import PRESCAN_ACTIVITY_EPS, SPARSE_DENSE_RATIO_THRESHOLD
from Ops.sparse_grouped_conv2d import SparseGroupedConv2d


class SparseDepthwiseConv2d(SparseGroupedConv2d):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        dilation: int = 1,
        threshold: float = PRESCAN_ACTIVITY_EPS,
        fallback_ratio: float = SPARSE_DENSE_RATIO_THRESHOLD,
        launch_all_tiles: bool = False,
        return_ms: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=in_channels,
            dilation=dilation,
            threshold=threshold,
            fallback_ratio=fallback_ratio,
            launch_all_tiles=launch_all_tiles,
            return_ms=return_ms,
        )
        self._sparse_backend_family = "sparse_depthwise_conv2d"
        self._sparse_diag_path = "depthwise_conv2d_sparse"
        self.backend_family = self._sparse_backend_family
        self.diag_path = self._sparse_diag_path

    @classmethod
    def from_dense(
        cls,
        conv: nn.Conv2d,
        threshold: float = PRESCAN_ACTIVITY_EPS,
        return_ms: bool = False,
        **kwargs,
    ):
        assert conv.groups == conv.in_channels == conv.out_channels, (
            "from_dense requires depthwise conv with "
            "groups == in_channels == out_channels"
        )
        sparse = cls(
            in_channels=conv.in_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            bias=conv.bias is not None,
            threshold=threshold,
            return_ms=return_ms,
            **kwargs,
        )
        sparse = sparse.to(conv.weight.device)
        sparse.weight.data.copy_(conv.weight.data)
        if conv.bias is not None and sparse.bias is not None:
            sparse.bias.data.copy_(conv.bias.data)
        return sparse

    def extra_repr(self):
        return (
            f"{self.in_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, "
            f"depthwise=True, threshold={self.threshold}, "
            f"fallback_ratio={self.fallback_ratio}, launch_all_tiles={self.launch_all_tiles}"
        )
