"""
SparseBatchNorm2d — 稀疏 BatchNorm2d 的 nn.Module 封装

可直接替换 torch.nn.BatchNorm2d，接口完全兼容。
对于 LIF 脉冲输出中的全零空间位置，用预计算常数代替完整 BN 计算。
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseBatchNorm2d(nn.Module):
    """
    稀疏加速版 BatchNorm2d（仅推理模式）。

    原理：
      BN 推理: y = (x - mean) / sqrt(var + eps) * gamma + beta
      当 x=0:  y_zero = (-mean / sqrt(var+eps)) * gamma + beta = 常数

      对全零空间位置直接赋值常数，跳过完整 BN 计算。
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True, threshold=1e-6):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.threshold = threshold

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

        self._last_sparse_ms = 0.0
        self._zero_bias = None  # 缓存预计算常数

    @classmethod
    def from_dense(cls, bn: nn.BatchNorm2d, threshold: float = 1e-6) -> "SparseBatchNorm2d":
        """从现有 nn.BatchNorm2d 创建，复制所有参数和 running stats。"""
        sparse_bn = cls(
            num_features=bn.num_features,
            eps=bn.eps,
            momentum=bn.momentum,
            affine=bn.affine,
            track_running_stats=bn.track_running_stats,
            threshold=threshold,
        )
        if bn.affine:
            sparse_bn.weight.data.copy_(bn.weight.data)
            sparse_bn.bias.data.copy_(bn.bias.data)
        if bn.track_running_stats:
            sparse_bn.running_mean.copy_(bn.running_mean)
            sparse_bn.running_var.copy_(bn.running_var)
            sparse_bn.num_batches_tracked.copy_(bn.num_batches_tracked)
        sparse_bn = sparse_bn.to(bn.weight.device if bn.affine else bn.running_mean.device)
        return sparse_bn

    def _compute_zero_bias(self):
        """预计算全零输入的 BN 输出常数 [C]"""
        mean = self.running_mean
        var = self.running_var
        gamma = self.weight if self.weight is not None else torch.ones_like(mean)
        beta = self.bias if self.bias is not None else torch.zeros_like(mean)
        inv_std = 1.0 / torch.sqrt(var + self.eps)
        return (-mean * inv_std) * gamma + beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        输入支持：
          - [N, C, H, W]       — 标准 4D
          - [T, N, C, H, W]    — spikingjelly 多时间步 5D
        """
        reshaped = False
        if x.dim() == 5:
            T, B, C, H, W = x.shape
            x = x.reshape(T * B, C, H, W)
            reshaped = True

        if self.training:
            # 训练模式走标准 BN
            y = F.batch_norm(
                x, self.running_mean, self.running_var,
                self.weight, self.bias, True, self.momentum, self.eps
            )
            self._last_sparse_ms = 0.0
        else:
            y = self._sparse_forward(x)

        if reshaped:
            _, C_out, H_out, W_out = y.shape
            y = y.reshape(T, B, C_out, H_out, W_out)

        return y

    def _sparse_forward(self, x: torch.Tensor) -> torch.Tensor:
        """推理模式的稀疏 BN"""
        N, C, H, W = x.shape

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # 预计算零位置常数（缓存）
        if self._zero_bias is None or self._zero_bias.device != x.device:
            self._zero_bias = self._compute_zero_bias().to(x.device)

        # 标准 BN 推理
        y = F.batch_norm(
            x, self.running_mean, self.running_var,
            self.weight, self.bias, False, 0.0, self.eps
        )

        # 全零空间位置覆盖为预计算常数
        spatial_sum = x.abs().sum(dim=1)  # [N, H, W]
        is_zero = spatial_sum <= self.threshold  # [N, H, W]
        zero_mask = is_zero.unsqueeze(1)  # [N, 1, H, W]
        y = torch.where(zero_mask, self._zero_bias.view(1, C, 1, 1), y)

        end.record()
        torch.cuda.synchronize(x.device)
        self._last_sparse_ms = start.elapsed_time(end)

        return y

    def extra_repr(self) -> str:
        return (
            f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, "
            f"affine={self.affine}, sparse=True"
        )