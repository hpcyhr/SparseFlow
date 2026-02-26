"""
SparseConv2d — 稀疏 Conv2d 的 nn.Module 封装

可直接替换 torch.nn.Conv2d，接口完全兼容。
当输入具有高稀疏性时（如 LIF 输出），自动跳过全零 block 获得加速。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseConv2d(nn.Module):
    """
    稀疏加速版 Conv2d，支持 3×3 和 1×1 卷积。

    工作模式：
      1. 输入经 prescan 识别非零 block
      2. 仅对非零 block 执行 Triton sparse kernel
      3. 如果 Triton 不可用或输入不在 CUDA 上，fallback 到 F.conv2d

    Attributes:
        block_size: prescan 的 block 大小
        threshold: 零判断阈值
        _last_sparse_ms: 上一次 forward 的 Stage-2 耗时（供 profiler 读取）
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 block_size=16, threshold=1e-6):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.block_size = block_size
        self.threshold = threshold

        # 权重和偏置
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # profiler 用
        self._last_sparse_ms = 0.0

        # 检测 Triton 是否可用
        self._triton_available = False
        try:
            import triton
            self._triton_available = True
        except ImportError:
            pass

    @classmethod
    def from_dense(cls, conv: nn.Conv2d, block_size: int = 16,
                   threshold: float = 1e-6) -> "SparseConv2d":
        """
        从现有的 nn.Conv2d 创建 SparseConv2d，复制权重。

        Args:
            conv: 原始 Conv2d 模块
            block_size: prescan block 大小
            threshold: 零判断阈值

        Returns:
            SparseConv2d 实例，权重已复制
        """
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

        # 复制权重
        sparse_conv.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            sparse_conv.bias.data.copy_(conv.bias.data)

        return sparse_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        输入支持两种 shape：
          - [N, C, H, W]       — 标准 4D
          - [T, N, C, H, W]    — spikingjelly 多时间步格式，自动展平

        自动选择执行路径：
          1. CUDA + Triton 可用 + stride=1 + groups=1 → Triton sparse kernel
          2. 其他情况 → fallback 到 F.conv2d
        """
        # 处理 5D 输入 (T, N, C, H, W)
        reshaped = False
        if x.dim() == 5:
            T, B, C, H, W = x.shape
            x = x.reshape(T * B, C, H, W)
            reshaped = True

        # 判断是否可以走 Triton 路径
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

        # 恢复 5D
        if reshaped:
            _, C_out, H_out, W_out = y.shape
            y = y.reshape(T, B, C_out, H_out, W_out)

        return y

    def _triton_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Triton 稀疏卷积路径"""
        from sparseflow.kernels.conv2d import sparse_conv2d_forward

        k = self.kernel_size[0]  # 3 or 1
        y, sparse_ms = sparse_conv2d_forward(
            x=x.contiguous(),
            weight=self.weight.contiguous(),
            bias=self.bias,
            block_size=self.block_size,
            kernel_size=k,
            threshold=self.threshold,
        )
        self._last_sparse_ms = sparse_ms
        return y

    def _fallback_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback 到 PyTorch F.conv2d"""
        self._last_sparse_ms = 0.0
        return F.conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, block_size={self.block_size}, "
            f"sparse=True"
        )