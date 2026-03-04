"""
SparseConv2d — v11.0 Channel-First Prescan + Flags Caching

Changes from v10.4:
  1. self._flags: cached prescan flags buffer, allocated once, reused across
     forward calls. Grows if input produces more blocks, never shrinks.
  2. _ensure_flags(x): computes tile config from input shape and returns
     (flags, BH, BW, BLOCK_M, BLOCK_K, GH, GW). Reuses / expands the
     cached self._flags tensor as needed. When merge_time_steps > 0 and
     5D input is detected, sizes the buffer for the compressed B (not T*B).
  3. _triton_forward passes flags + return_ms + merge_time_steps to
     sparse_conv2d_forward.
  4. return_ms attribute (default False) controls whether Stage-2 timing
     is performed. Set to True for benchmarking.
  5. merge_time_steps attribute (default 0) controls time-union prescan.
     Set to T (number of time steps) to enable OR-compression.
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

    return_ms: bool, default False.
      If True, Stage-2 kernel timing is measured via CUDA events.
      If False, no sync, no timing overhead.

    merge_time_steps: int, default 0.
      If > 0, prescan runs on OR-compressed time dimension.
      Set to T when input is shaped as [T*B, C, H, W] from SNN.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 block_size=None, threshold=1e-6,
                 return_ms=False, merge_time_steps=0):
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
        self.return_ms = return_ms
        self.merge_time_steps = merge_time_steps

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self._last_sparse_ms = 0.0

        # ── Cached flags buffer (改动 4) ──
        self._flags = None

        self._triton_available = False
        try:
            import triton
            self._triton_available = True
        except ImportError:
            pass

    def _ensure_flags(self, x):
        """
        Compute tile config from input shape and return a reusable flags buffer.

        The flags tensor is allocated once and reused across forward calls.
        If the current buffer is too small, it is re-allocated. Never shrinks.

        When merge_time_steps > 0, the prescan runs on B samples (not T*B),
        so flags are sized accordingly.

        Returns:
            flags: torch.Tensor [>= total_flags], dtype=int32
            BH, BW, BLOCK_M, BLOCK_K: tile config scalars
            GH, GW: grid dimensions
        """
        import triton
        from Kernels.conv2d import _select_block_sizes

        N, C_IN, H, W = x.shape
        C_OUT = self.out_channels
        k = self.kernel_size[0]

        # Compute tile config (same logic as sparse_conv2d_forward)
        pixels = H * W
        if self.block_size is None or pixels >= 3136:
            BH, BW, BLOCK_M, BLOCK_N, BLOCK_K = _select_block_sizes(
                H, W, C_IN, C_OUT, k, N)
        else:
            bs = max(self.block_size, 4)
            BH, BW = bs, bs
            BLOCK_M = BH * BW
            BLOCK_N = 32 if C_OUT >= 32 else 16
            BLOCK_K = 16

        GH = triton.cdiv(H, BH)
        GW = triton.cdiv(W, BW)

        # Prescan N may be smaller when time-union is enabled
        T = self.merge_time_steps
        if T > 0 and N > T:
            prescan_N = N // T  # B
        else:
            prescan_N = N

        total_flags = prescan_N * C_IN * GH * GW

        # Reuse / expand cached buffer
        if (self._flags is None
                or self._flags.numel() < total_flags
                or self._flags.device != x.device):
            self._flags = torch.empty(total_flags, dtype=torch.int32,
                                      device=x.device)

        return self._flags, BH, BW, BLOCK_M, BLOCK_K, GH, GW

    @classmethod
    def from_dense(cls, conv: nn.Conv2d, block_size=None,
                   threshold: float = 1e-6,
                   return_ms: bool = False,
                   merge_time_steps: int = 0) -> "SparseConv2d":
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
            merge_time_steps=merge_time_steps,
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

        # Get cached flags buffer
        flags, BH, BW, BLOCK_M, BLOCK_K, GH, GW = self._ensure_flags(x)

        k = self.kernel_size[0]
        y, sparse_ms = sparse_conv2d_forward(
            x=x.contiguous(),
            weight=self.weight.contiguous(),
            bias=self.bias,
            block_size=self.block_size,
            kernel_size=k,
            threshold=self.threshold,
            flags=flags,
            return_ms=self.return_ms,
            merge_time_steps=self.merge_time_steps,
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
            f"return_ms={self.return_ms}, "
            f"merge_time={self.merge_time_steps}, sparse=True"
        )