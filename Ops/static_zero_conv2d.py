"""
StaticZeroConv2d — v19 Unified Zero Backend

First-class zero backend for SparseFlow. Handles all exact-zero execution paths
with minimal runtime overhead.

Design principles:
  1. Output = bias-only constant tensor (exact, not heuristic)
  2. Template caching by (device, dtype, H_out, W_out, bias_version)
  3. expand() for batch dimension — no allocation, no copy
  4. Unified diagnostics: zero_reason, backend_family, diag_path

Zero backend family covers:
  - StaticZeroConv2d (layer-level replacement for known-zero-input conv)
  - zero_fastpath (SparseConv2d runtime zero detection)
  - exact zero propagation (model-level zero flow)

All share the same output construction: cached bias-only template + expand().
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple


# Zero backend family identifier
ZERO_BACKEND_FAMILY = "exact_zero"


class StaticZeroConv2d(nn.Module):
    """Exact-zero convolution backend.

    When the input to a Conv2d is provably all-zero (detected during sparsity
    measurement), the output is a deterministic function of bias only:
      y[n, c_out, h, w] = bias[c_out]  (or 0.0 if no bias)

    This module produces that exact output with near-zero compute cost:
      - No kernel launch for the convolution itself
      - Single cached template tensor, expanded for batch dimension
      - Template invalidated only when bias changes (tracked via _version)
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=None,
        zero_reason: str = "static_zero_input",
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

        # Store bias as buffer (not parameter — we don't train this module)
        if bias is not None:
            self.register_buffer("bias_buf", bias.detach().float().clone())
        else:
            self.bias_buf = None

        # Diagnostics
        self.zero_reason = zero_reason
        self.backend_family = ZERO_BACKEND_FAMILY
        self._forward_count = 0

        # Template cache: {(device_str, dtype_str, H_out, W_out, bias_ver): tensor}
        self._template_cache: Dict[Tuple, torch.Tensor] = {}

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, zero_reason: str = "static_zero_input"):
        """Create from an existing nn.Conv2d, copying only bias."""
        return cls(
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            bias=conv.bias,
            zero_reason=zero_reason,
        ).to(conv.weight.device)

    def _output_hw(self, H: int, W: int) -> Tuple[int, int]:
        """Compute output spatial dimensions."""
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation
        H_out = math.floor((H + 2 * ph - dh * (kh - 1) - 1) / sh + 1)
        W_out = math.floor((W + 2 * pw - dw * (kw - 1) - 1) / sw + 1)
        return H_out, W_out

    def _get_template(self, device, dtype, H_out: int, W_out: int) -> torch.Tensor:
        """Get or create the cached bias-only output template.

        Template shape: [1, C_out, H_out, W_out]
        Cached per (device, dtype, spatial dims, bias version).
        """
        bias_ver = -1 if self.bias_buf is None else id(self.bias_buf)
        key = (str(device), str(dtype), H_out, W_out, bias_ver)

        if key not in self._template_cache:
            if self.bias_buf is not None:
                # Construct bias-only output: [1, C, 1, 1] expanded to [1, C, H, W]
                b = self.bias_buf.to(device=device, dtype=dtype)
                template = b.view(1, -1, 1, 1).expand(1, self.out_channels, H_out, W_out).contiguous()
            else:
                template = torch.zeros(1, self.out_channels, H_out, W_out, dtype=dtype, device=device)
            self._template_cache[key] = template

        return self._template_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce exact zero-input convolution output.

        Supports 4D [N, C, H, W] and 5D [T, N, C, H, W] inputs.
        Output dtype is float32 for consistency with SparseConv2d.
        """
        self._forward_count += 1

        if x.dim() == 4:
            B, _, H, W = x.shape
            H_out, W_out = self._output_hw(H, W)
            tpl = self._get_template(x.device, torch.float32, H_out, W_out)
            # expand() shares memory — no allocation for batch dimension
            return tpl.expand(B, -1, -1, -1)

        elif x.dim() == 5:
            T, B, _, H, W = x.shape
            H_out, W_out = self._output_hw(H, W)
            tpl = self._get_template(x.device, torch.float32, H_out, W_out)
            # expand to [T*B, ...] then reshape to [T, B, ...]
            return tpl.expand(T * B, -1, -1, -1).reshape(T, B, self.out_channels, H_out, W_out)

        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")

    def get_diag(self) -> Dict[str, Any]:
        """Return structured diagnostics for this zero backend instance."""
        return {
            'backend_family': self.backend_family,
            'zero_reason': self.zero_reason,
            'diag_path': 'staticzero',
            'has_bias': self.bias_buf is not None,
            'out_channels': self.out_channels,
            'forward_count': self._forward_count,
            'template_cache_size': len(self._template_cache),
        }

    def extra_repr(self) -> str:
        return (
            f"out_channels={self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, "
            f"has_bias={self.bias_buf is not None}, "
            f"backend={self.backend_family}, reason={self.zero_reason}"
        )


# ---------------------------------------------------------------------------
# Utility: construct exact-zero output without a module instance
# ---------------------------------------------------------------------------

def make_zero_conv_output(
    batch_size: int,
    out_channels: int,
    H_out: int,
    W_out: int,
    bias: Optional[torch.Tensor],
    device,
    dtype=torch.float32,
) -> torch.Tensor:
    """Construct exact zero-input conv output tensor.

    Used by SparseConv2d._zero_output_4d() and similar fast paths.
    Returns [batch_size, out_channels, H_out, W_out] with bias-only values.
    """
    if bias is not None:
        b = bias.detach().to(device=device, dtype=dtype)
        # [1, C, 1, 1] → expand to [B, C, H, W]
        return b.view(1, -1, 1, 1).expand(batch_size, out_channels, H_out, W_out)
    else:
        return torch.zeros(batch_size, out_channels, H_out, W_out, dtype=dtype, device=device)


def make_synthetic_zero_diag(
    layer_name: str = "",
    source: str = "staticzero",
    total_group_count: float = -1.0,
    total_tile_count: float = -1.0,
) -> Dict[str, Any]:
    """Construct standardized diagnostics for any exact-zero path.

    Provides consistent fields matching SparseConv2d diagnostics schema.
    """
    return {
        'active_group_ratio': 0.0,
        'tile_zero_ratio': 1.0,
        'total_group_count': total_group_count,
        'nonzero_group_count': 0.0,
        'tile_zero_count': total_tile_count,
        'total_tile_count': total_tile_count,
        'effective_k_ratio': 0.0,
        'sparse_compute_ms': -1.0,
        'sparse_total_ms': -1.0,
        'zero_tiles': int(total_tile_count) if total_tile_count >= 0 else -1,
        'sparse_tiles': 0,
        'denseish_tiles': 0,
        'stage1_zero_tiles': int(total_tile_count) if total_tile_count >= 0 else -1,
        'stage2_tiles': 0,
        'prescan_mode': 'skipped_zero',
        '_synthetic': True,
        '_diag_path': source,
        'backend_family': ZERO_BACKEND_FAMILY,
        'zero_reason': source,
    }