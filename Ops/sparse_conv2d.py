"""
SparseConv2d — v14.1 Per-Tile + 零同步 + 预分配 Buffer

改动：
  1. _ensure_buffers() 同时预分配 counts_buf 和 tile_cin_buf
     tile_cin_buf 容量 = N_TILES * C_IN（最坏情况上界）
  2. 缓存 Channel-Last 权重 (self._w_cl)
  3. 全部 buffer 传递给 sparse_conv2d_forward，零 Host-Device 同步
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
    Per-Tile 稀疏加速版 Conv2d。

    block_size: None → kernel 动态决策
    return_ms: True → Stage-2 CUDA event 计时
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 block_size=None, threshold=1e-6,
                 return_ms=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size if isinstance(kernel_size, tuple)
                            else (kernel_size, kernel_size))
        self.stride = (stride if isinstance(stride, tuple)
                       else (stride, stride))
        self.padding = (padding if isinstance(padding, tuple)
                        else (padding, padding))
        self.dilation = (dilation if isinstance(dilation, tuple)
                         else (dilation, dilation))
        self.groups = groups
        self.block_size = block_size
        self.threshold = threshold
        self.return_ms = return_ms

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups,
                        *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self._last_sparse_ms = 0.0

        # 缓存
        self._w_cl = None           # Channel-Last 权重 fp16
        self._counts_buf = None     # prescan count 缓冲区
        self._tile_cin_buf = None   # prescan write 缓冲区（最大容量）

        self._triton_available = False
        try:
            import triton
            self._triton_available = True
        except ImportError:
            pass

    def _get_w_cl(self):
        """获取 Channel-Last 格式权重，首次转换并缓存。"""
        if self._w_cl is not None:
            return self._w_cl

        k = self.kernel_size[0]
        w = self.weight.data
        if k == 3:
            # [C_OUT, C_IN, 3, 3] → [C_OUT, 3, 3, C_IN]
            self._w_cl = w.half().permute(0, 2, 3, 1).contiguous()
        else:
            # [C_OUT, C_IN, 1, 1] → [C_OUT, C_IN]
            self._w_cl = w.half().reshape(
                self.out_channels, self.in_channels).contiguous()
        return self._w_cl

    def _invalidate_w_cl(self):
        """权重更新后清除缓存。"""
        self._w_cl = None

    def _ensure_buffers(self, x):
        """
        确保 counts_buf 和 tile_cin_buf 已分配且足够大。

        counts_buf:   [N_TILES] int32
        tile_cin_buf: [N_TILES * C_IN] int32  (最坏情况上界)

        这两个 buffer 只分配一次，后续复用（自动增长，不收缩）。
        """
        import triton
        from Kernels.conv2d import _select_block_sizes

        N, C_IN, H, W = x.shape
        C_OUT = self.out_channels
        k = self.kernel_size[0]

        BH, BW, _, _, _ = _select_block_sizes(H, W, C_IN, C_OUT, k, N)
        GH = triton.cdiv(H, BH)
        GW = triton.cdiv(W, BW)
        N_TILES = N * GH * GW

        # counts_buf: [N_TILES]
        if (self._counts_buf is None
                or self._counts_buf.numel() < N_TILES
                or self._counts_buf.device != x.device):
            self._counts_buf = torch.empty(
                N_TILES, dtype=torch.int32, device=x.device)

        # tile_cin_buf: [N_TILES * C_IN] — 最坏情况每个 tile 所有通道都活跃
        max_cin_entries = N_TILES * C_IN
        if (self._tile_cin_buf is None
                or self._tile_cin_buf.numel() < max_cin_entries
                or self._tile_cin_buf.device != x.device):
            self._tile_cin_buf = torch.empty(
                max_cin_entries, dtype=torch.int32, device=x.device)

        return self._counts_buf, self._tile_cin_buf

    @classmethod
    def from_dense(cls, conv: nn.Conv2d, block_size=None,
                   threshold: float = 1e-6,
                   return_ms: bool = False) -> "SparseConv2d":
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
            return_ms=return_ms,
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

        w_cl = self._get_w_cl()
        counts_buf, tile_cin_buf = self._ensure_buffers(x)
        k = self.kernel_size[0]

        y, sparse_ms = sparse_conv2d_forward(
            x=x.contiguous(),
            weight=self.weight,
            bias=self.bias,
            block_size=self.block_size,
            kernel_size=k,
            threshold=self.threshold,
            w_cl=w_cl,
            counts_buf=counts_buf,
            tile_cin_buf=tile_cin_buf,
            return_ms=self.return_ms,
        )
        self._last_sparse_ms = sparse_ms
        return y

    def _fallback_forward(self, x: torch.Tensor) -> torch.Tensor:
        self._last_sparse_ms = 0.0
        return F.conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self) -> str:
        bs = self.block_size if self.block_size is not None else "auto"
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, block_size={bs}, "
            f"return_ms={self.return_ms}, sparse=True")