"""
SparseConv2d — v10.4 延迟决策版

核心改动：
  - block_size 默认 None（不预设，由 kernel 动态决策）。
  - _triton_forward 直接传递 self.block_size 给 sparse_conv2d_forward。
  - 当 block_size=None 时，kernel 根据 (H, W, N) 选择最优 BLOCK_M/BLOCK_N。
  - 当 block_size 被手动指定时，kernel 仍然动态覆盖（大图强制升级）。
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseConv2d(nn.Module):
    """
    稀疏加速版 Conv2d。

    block_size 语义：
      None  → kernel 根据 (H, W, N) 完全动态决策 BH/BW/BLOCK_M
      int   → 传递给 kernel，但 kernel 可能覆盖（大图自动升级）
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 block_size=None, threshold=1e-6):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.block_size = block_size    # None = 延迟决策
        self.threshold = threshold

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self._last_sparse_ms = 0.0

        self._triton_available = False
        try:
            import triton
            self._triton_available = True
        except ImportError:
            pass

    @classmethod
    def from_dense(cls, conv: nn.Conv2d, block_size=None,
                   threshold: float = 1e-6) -> "SparseConv2d":
        """从现有 nn.Conv2d 创建 SparseConv2d，复制权重。"""
        sparse_conv = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            block_size=block_size,
            threshold=threshold,
        )
        sparse_conv.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            sparse_conv.bias.data.copy_(conv.bias.data)
        sparse_conv = sparse_conv.to(conv.weight.device)
        return sparse_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reshaped = False
        if x.dim() == 5:
            T, B, C, H, W = x.shape
            x = x.reshape(T * B, C, H, W)
            reshaped = True

        use_triton = (
            self._triton_available
            and x.is_cuda
            and self.stride == (1, 1)
            and self.dilation == (1, 1)
            and self.groups == 1
            and self.kernel_size in [(3, 3), (1, 1)]
        )

        if use_triton:
            y = self._triton_forward(x)
        else:
            y = self._fallback_forward(x)

        if reshaped:
            _, C_out, H_out, W_out = y.shape
            y = y.reshape(T, B, C_out, H_out, W_out)

        return y

    def _triton_forward(self, x: torch.Tensor) -> torch.Tensor:
        from Kernels.conv2d import sparse_conv2d_forward

        k = self.kernel_size[0]
        y, sparse_ms = sparse_conv2d_forward(
            x=x.contiguous(),
            weight=self.weight.contiguous(),
            bias=self.bias,
            block_size=self.block_size,   # None 或 int，kernel 内部处理
            kernel_size=k,
            threshold=self.threshold,
        )
        self._last_sparse_ms = sparse_ms
        return y

    def _fallback_forward(self, x: torch.Tensor) -> torch.Tensor:
        self._last_sparse_ms = 0.0
        return F.conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )

    def extra_repr(self) -> str:
        bs = self.block_size if self.block_size is not None else "auto"
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, block_size={bs}, sparse=True"
        )