"""
SparseConv2d — v12.1 Per-Tile CSR Compaction + Buffer Caching

Changes from v10.4:
  1. Stage-1 now builds a per-tile CSR structure (tile_ptr + tile_cin)
     instead of a global flags array + global active channel set.
  2. self._counts_buf: cached int32 buffer for prescan tile_counts.
     Allocated once, reused across forward calls. Grows if input
     produces more tiles, never shrinks.
  3. return_ms (default False): only creates CUDA events and
     synchronizes when True.
  4. _ensure_counts_buf(x): computes tile config and returns a
     reusable counts buffer + tile parameters.
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
    稀疏加速版 Conv2d with per-tile channel compaction.

    block_size 语义：
      None  → kernel 根据 (H, W, C_IN, N) 完全动态决策 BH/BW/BLOCK_M
      int   → 传递给 kernel，但大图 (H*W >= 3136) 会被覆盖

    return_ms: bool, default False.
      True → Stage-2 kernel timing via CUDA events.
      False → no sync, no timing overhead.
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
                        *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self._last_sparse_ms = 0.0

        # Cached buffer for prescan tile_counts (改动 4)
        self._counts_buf = None

        self._triton_available = False
        try:
            import triton
            self._triton_available = True
        except ImportError:
            pass

    def _ensure_counts_buf(self, x):
        """
        Compute tile config from input shape. Return a reusable counts
        buffer sized for prescan_count_kernel output.

        The buffer grows if needed, never shrinks. Device-aware.

        Returns:
            counts_buf: torch.Tensor [>= N_TILES], dtype=int32
            BH, BW, GH, GW: tile config (for info / debugging)
        """
        import triton
        from Kernels.conv2d import _select_block_sizes

        N, C_IN, H, W = x.shape
        C_OUT = self.out_channels
        k = self.kernel_size[0]

        pixels = H * W
        if self.block_size is None or pixels >= 3136:
            BH, BW, _, _, _ = _select_block_sizes(
                H, W, C_IN, C_OUT, k, N)
        else:
            bs = max(self.block_size, 4)
            BH, BW = bs, bs

        GH = triton.cdiv(H, BH)
        GW = triton.cdiv(W, BW)
        N_TILES = N * GH * GW

        if (self._counts_buf is None
                or self._counts_buf.numel() < N_TILES
                or self._counts_buf.device != x.device):
            self._counts_buf = torch.empty(
                N_TILES, dtype=torch.int32, device=x.device)

        return self._counts_buf, BH, BW, GH, GW

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

        # Get cached counts buffer
        counts_buf, _, _, _, _ = self._ensure_counts_buf(x)

        k = self.kernel_size[0]
        y, sparse_ms = sparse_conv2d_forward(
            x=x.contiguous(),
            weight=self.weight.contiguous(),
            bias=self.bias,
            block_size=self.block_size,
            kernel_size=k,
            threshold=self.threshold,
            counts_buf=counts_buf,
            return_ms=self.return_ms,
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
            f"padding={self.padding}, block_size={bs}, "
            f"return_ms={self.return_ms}, sparse=True"
        )