"""
SparseLinear  — Tile-level Dynamic-K sparse Linear nn.Module

Mirrors SparseCond architecture:
  - from_dense() class method for one-line conversion
  - 5D/3D/2D input handling
  - Cached transposed weight (W_T) for coalesced access
  - Pre-allocated buffers (counts_buf, tile_cin_buf)
  - Triton fallback to F.linear
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseLinear(nn.Module):
    """
    Tile-level Dynamic-K sparse Linear. Drop-in replacement for nn.Linear.

    Groups batch rows into tiles, prescans which C_IN channels are active
    per tile, then only multiplies active channels (Dynamic-K while loop).
    """

    def __init__(self, in_features, out_features, bias=True,
                 threshold=1e-6, return_ms=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.return_ms = return_ms

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

        # Cached buffers
        self._w_t = None           # [C_IN, C_OUT] transposed weight fp16
        self._w_t_version = -1     # track weight updates
        self._counts_buf = None
        self._tile_cin_buf = None

    def _get_w_t(self):
        """Get or recompute transposed weight cache."""
        ver = self.weight._version
        if self._w_t is None or self._w_t_version != ver:
            self._w_t = self.weight.data.half().t().contiguous()
            self._w_t_version = ver
        return self._w_t

    def _ensure_buffers(self, x):
        """Pre-allocate prescan buffers based on input shape."""
        N = x.shape[0]
        from Kernels.linear import _select_linear_block_m
        import triton
        BLOCK_M = _select_linear_block_m(N)
        N_TILES = triton.cdiv(N, BLOCK_M)
        C_IN = self.in_features

        if (self._counts_buf is None
                or self._counts_buf.numel() < N_TILES
                or self._counts_buf.device != x.device):
            self._counts_buf = torch.empty(N_TILES, dtype=torch.int32, device=x.device)

        max_cin = N_TILES * C_IN
        if (self._tile_cin_buf is None
                or self._tile_cin_buf.numel() < max_cin
                or self._tile_cin_buf.device != x.device):
            self._tile_cin_buf = torch.empty(max_cin, dtype=torch.int32, device=x.device)

        return self._counts_buf, self._tile_cin_buf

    @classmethod
    def from_dense(cls, linear: nn.Linear, threshold: float = 1e-6,
                   return_ms: bool = False) -> "SparseLinear":
        """Create SparseLinear from existing nn.Linear, copying weights."""
        sparse = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            threshold=threshold,
            return_ms=return_ms,
        )
        sparse.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            sparse.bias.data.copy_(linear.bias.data)
        return sparse.to(linear.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        use_triton = (self._triton_available and x.is_cuda and x.dim() == 2)

        if use_triton:
            y = self._triton_forward(x)
        else:
            y = self._fallback_forward(x)

        if reshaped:
            if len(orig_shape) == 5:
                T, B = orig_shape[0], orig_shape[1]
                y = y.reshape(T, B, self.out_features)
            elif len(orig_shape) == 3:
                T, B = orig_shape[0], orig_shape[1]
                y = y.reshape(T, B, self.out_features)

        return y

    def _triton_forward(self, x: torch.Tensor) -> torch.Tensor:
        from Kernels.linear import sparse_linear_forward

        w_t = self._get_w_t()
        counts_buf, tile_cin_buf = self._ensure_buffers(x)

        y, sparse_ms = sparse_linear_forward(
            x=x.contiguous(),
            weight=self.weight,
            bias=self.bias,
            threshold=self.threshold,
            w_t=w_t,
            counts_buf=counts_buf,
            tile_cin_buf=tile_cin_buf,
            return_ms=self.return_ms,
        )
        self._last_sparse_ms = sparse_ms
        return y

    def _fallback_forward(self, x: torch.Tensor) -> torch.Tensor:
        self._last_sparse_ms = 0.0
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (f"{self.in_features}, {self.out_features}, "
                f"bias={self.bias is not None}, sparse=True")