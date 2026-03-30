"""
SparseFlow Ops/sparse_conv1d.py 鈥?SparseConv1d nn.Module Wrapper

Wraps Kernels/conv1d.py. Manages buffers, provides from_dense(),
5D (T, N, C, L) input support, and F.conv1d fallback.

v1: prescan + dense compute; sparse compute kernel in v2.
"""

import sys
from pathlib import Path
from typing import Dict, Any

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseConv1d(nn.Module):
    """
    Sparse 1D convolution with grouped-bitmask prescan on input channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        groups: int = 1,
        threshold: float = 1e-6,
        return_ms: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride,) if isinstance(stride, int) else stride
        self.padding = (padding,) if isinstance(padding, int) else padding
        self.groups = groups
        self.threshold = threshold
        self.return_ms = return_ms

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, self.kernel_size[0])
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
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
    def from_dense(cls, conv: nn.Conv1d, threshold: float = 1e-6, return_ms: bool = False, **kwargs):
        sparse = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size[0],
            stride=conv.stride[0],
            padding=conv.padding[0],
            bias=conv.bias is not None,
            groups=conv.groups,
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
        # Handle 4D [T, N, C, L] input
        if x.ndim == 4:
            T, N, C, L = x.shape
            x_3d = x.reshape(T * N, C, L)
            y_3d = self._forward_3d(x_3d)
            L_out = y_3d.shape[2]
            return y_3d.reshape(T, N, self.out_channels, L_out)
        return self._forward_3d(x)

    def _forward_3d(self, x: torch.Tensor) -> torch.Tensor:
        # groups != 1 鈫?dense fallback (sparse kernel supports groups=1 only)
        if self.groups != 1 or not self._triton_available or not x.is_cuda:
            return self._fallback(x)

        from Kernels.conv1d import sparse_conv1d_forward

        result = sparse_conv1d_forward(
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
        return F.conv1d(
            x.float(),
            self.weight.float(),
            self.bias.float() if self.bias is not None else None,
            self.stride,
            self.padding,
            groups=self.groups,
        ).float()

    def extra_repr(self):
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, groups={self.groups}, "
            f"threshold={self.threshold}"
        )
