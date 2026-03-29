"""
SparseFlow Ops/sparse_depthwise_conv2d.py — SparseDepthwiseConv2d Module

Wraps Kernels/depthwise_conv2d.py. Handles groups=C_in depthwise convolution
with per-channel-per-tile zero-skip.

Follows SparseConv2d conventions: from_dense(), 5D input support, fallback.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseDepthwiseConv2d(nn.Module):
    """
    Sparse depthwise Conv2d (groups=C_in).

    Skips computation for (channel, spatial-tile) pairs where input is zero.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        threshold: float = 1e-6,
        return_ms: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels  # depthwise: C_out == C_in
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.groups = in_channels
        self.threshold = threshold
        self.return_ms = return_ms

        # Weight: [C, 1, KH, KW]
        self.weight = nn.Parameter(torch.empty(in_channels, 1, self.kernel_size[0], self.kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        self._triton_available = False
        try:
            import triton  # noqa: F401
            self._triton_available = True
        except Exception:
            pass

        self._last_sparse_ms = 0.0
        self._last_diag: Dict[str, Any] = {}
        self.collect_diag = False

    @classmethod
    def from_dense(cls, conv: nn.Conv2d, threshold: float = 1e-6, return_ms: bool = False, **kwargs):
        """Create from an existing nn.Conv2d with groups=C_in."""
        assert conv.groups == conv.in_channels, \
            f"from_dense requires depthwise conv (groups={conv.groups} != in_channels={conv.in_channels})"

        sparse = cls(
            in_channels=conv.in_channels,
            kernel_size=conv.kernel_size[0],
            stride=conv.stride[0],
            padding=conv.padding[0],
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle 5D [T, N, C, H, W] input
        if x.ndim == 5:
            T, N, C, H, W = x.shape
            x_4d = x.reshape(T * N, C, H, W)
            y_4d = self._forward_4d(x_4d)
            return y_4d.reshape(T, N, self.out_channels, y_4d.shape[2], y_4d.shape[3])
        return self._forward_4d(x)

    def _forward_4d(self, x: torch.Tensor) -> torch.Tensor:
        if not self._triton_available or not x.is_cuda:
            return self._fallback(x)

        from Kernels.depthwise_conv2d import sparse_depthwise_conv2d_forward

        result = sparse_depthwise_conv2d_forward(
            x=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride[0],
            padding=self.padding[0],
            threshold=self.threshold,
            return_ms=self.return_ms,
            return_tile_stats=self.collect_diag,
        )

        y = result[0]
        self._last_sparse_ms = result[1]

        if self.collect_diag and len(result) > 2:
            self._last_diag = result[2] if result[2] is not None else {}

        return y

    def _fallback(self, x):
        return F.conv2d(x, self.weight, self.bias,
                        self.stride, self.padding, groups=self.groups).float()

    def extra_repr(self):
        return (
            f"{self.in_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, "
            f"depthwise=True, threshold={self.threshold}"
        )