"""
SparseLinear — 稀疏 Linear 的 nn.Module 封装

可直接替换 torch.nn.Linear，接口完全兼容。
当输入具有高稀疏性时（如 LIF 输出展平后），自动跳过全零行获得加速。
"""

import sys
from pathlib import Path

# 项目根目录（SparseFlow/）
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseLinear(nn.Module):
    """
    稀疏加速版 Linear。

    工作模式：
      1. 输入按行 prescan，标记全零行
      2. 仅对非零行执行 Triton sparse matmul
      3. Triton 不可用时 fallback 到 F.linear
    """

    def __init__(self, in_features, out_features, bias=True,
                 threshold=1e-6):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
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
    def from_dense(cls, linear: nn.Linear, threshold: float = 1e-6) -> "SparseLinear":
        """从现有 nn.Linear 创建 SparseLinear，复制权重。"""
        sparse = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            threshold=threshold,
        )
        sparse.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            sparse.bias.data.copy_(linear.bias.data)
        sparse = sparse.to(linear.weight.device)
        return sparse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        输入支持：
          - [N, C_in]          — 标准 2D
          - [T, N, C_in]       — spikingjelly 多时间步 3D
          - [T, N, C, H, W]    — 5D（自动 flatten）
        """
        orig_shape = x.shape
        reshaped = False

        if x.dim() == 5:
            T, B, C, H, W = x.shape
            x = x.reshape(T * B, C * H * W)
            reshaped = True
        elif x.dim() == 3:
            T, B, Cin = x.shape
            x = x.reshape(T * B, Cin)
            reshaped = True

        use_triton = (
            self._triton_available
            and x.is_cuda
            and x.dim() == 2
        )

        if use_triton:
            y = self._triton_forward(x)
        else:
            y = self._fallback_forward(x)

        if reshaped:
            if len(orig_shape) == 5:
                y = y.reshape(T, B, self.out_features)
            elif len(orig_shape) == 3:
                y = y.reshape(T, B, self.out_features)

        return y

    def _triton_forward(self, x: torch.Tensor) -> torch.Tensor:
        from Kernels.linear import sparse_linear_forward
        y, sparse_ms = sparse_linear_forward(
            x=x.contiguous(),
            weight=self.weight.contiguous(),
            bias=self.bias,
            threshold=self.threshold,
        )
        self._last_sparse_ms = sparse_ms
        return y

    def _fallback_forward(self, x: torch.Tensor) -> torch.Tensor:
        self._last_sparse_ms = 0.0
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, sparse=True"
        )