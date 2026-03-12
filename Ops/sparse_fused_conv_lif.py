"""
FusedSparseConvLIF — Fused Sparse Conv2d + LIF Neuron nn.Module

Replaces the pattern: Conv2d → [optional BN] → LIFNode
with a single module that:
  1. Runs sparse conv (reusing conv2d.py prescan infrastructure)
  2. Applies LIF dynamics in-register (no intermediate VRAM write)
  3. Outputs spike tensor and updates membrane potential

The replaced LIFNode should be swapped with nn.Identity by the replacer.

Handles 5D (T, B, C, H, W) multi-step input by iterating over T,
since LIF state (V) depends on previous time step.
"""

import sys
from pathlib import Path
import math

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedSparseConvLIF(nn.Module):
    """
    Fused sparse Conv2d + LIF neuron.

    LIF parameters:
      - tau: membrane time constant (decay = exp(-dt/tau) ≈ 1 - 1/tau)
      - v_threshold: firing threshold
      - v_reset: soft reset (V_next = V_temp * (1 - spike))

    Conv parameters: inherited from the original Conv2d.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 block_size=None, threshold=1e-6,
                 tau=2.0, v_threshold=1.0,
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

        # Conv weights
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups,
                        *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # LIF parameters
        self.tau = tau
        self.v_threshold = v_threshold
        self.decay = math.exp(-1.0 / tau)  # exp(-dt/tau) with dt=1

        # Membrane potential state (managed per forward call)
        self.v = None  # [B, C_OUT, H_out, W_out], set during forward

        self._last_sparse_ms = 0.0
        self._triton_available = False
        try:
            import triton
            self._triton_available = True
        except ImportError:
            pass

        # Cached buffers (same as SparseConv2d)
        self._w_cl = None
        self._w_cl_version = -1
        self._counts_buf = None
        self._tile_cin_buf = None

    def _get_w_cl(self):
        ver = self.weight._version
        if self._w_cl is None or self._w_cl_version != ver:
            k = self.kernel_size[0]
            if k == 3:
                self._w_cl = self.weight.data.half().permute(0, 2, 3, 1).contiguous()
            else:
                self._w_cl = self.weight.data.half().reshape(
                    self.out_channels, self.in_channels).contiguous()
            self._w_cl_version = ver
        return self._w_cl

    def _ensure_buffers(self, x_single):
        """Pre-allocate buffers based on single-step input shape [B, C, H, W]."""
        from Kernels.conv2d import _select_tile_sizes
        import triton

        B, C_IN, H, W = x_single.shape
        BH, BW = _select_tile_sizes(H, W)
        GH = triton.cdiv(H, BH)
        GW = triton.cdiv(W, BW)
        N_TILES = B * GH * GW

        if (self._counts_buf is None
                or self._counts_buf.numel() < N_TILES
                or self._counts_buf.device != x_single.device):
            self._counts_buf = torch.empty(N_TILES, dtype=torch.int32,
                                           device=x_single.device)

        max_cin = N_TILES * C_IN
        if (self._tile_cin_buf is None
                or self._tile_cin_buf.numel() < max_cin
                or self._tile_cin_buf.device != x_single.device):
            self._tile_cin_buf = torch.empty(max_cin, dtype=torch.int32,
                                             device=x_single.device)

        return self._counts_buf, self._tile_cin_buf

    def reset(self):
        """Reset membrane potential. Called by sj_func.reset_net()."""
        self.v = None

    @classmethod
    def from_conv_and_lif(cls, conv: nn.Conv2d, lif_node,
                          block_size=None, threshold=1e-6,
                          return_ms=False) -> "FusedSparseConvLIF":
        """
        Create FusedSparseConvLIF from an existing Conv2d and LIFNode.

        Extracts tau and v_threshold from the LIFNode.
        """
        # Extract LIF parameters
        tau = getattr(lif_node, 'tau', 2.0)
        v_th = getattr(lif_node, 'v_threshold', 1.0)

        fused = cls(
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
            tau=tau,
            v_threshold=v_th,
            return_ms=return_ms,
        )

        fused.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            fused.bias.data.copy_(conv.bias.data)

        return fused.to(conv.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Input:
          - [T, B, C_IN, H, W]  (multi-step) → loop over T, update V
          - [B, C_IN, H, W]     (single-step) → single fused call

        Output: spike tensor, same shape as input.
        """
        if x.dim() == 5:
            return self._forward_multistep(x)
        elif x.dim() == 4:
            return self._forward_singlestep(x)
        else:
            raise ValueError(f"FusedSparseConvLIF expects 4D or 5D input, got {x.dim()}D")

    def _forward_multistep(self, x: torch.Tensor) -> torch.Tensor:
        T, B, C_IN, H, W = x.shape
        spikes_list = []

        for t in range(T):
            spike_t = self._forward_singlestep(x[t])  # [B, C_OUT, H_out, W_out]
            spikes_list.append(spike_t.unsqueeze(0))

        return torch.cat(spikes_list, dim=0)  # [T, B, C_OUT, H_out, W_out]

    def _forward_singlestep(self, x: torch.Tensor) -> torch.Tensor:
        """Single time-step forward with LIF state update."""
        B, C_IN, H, W = x.shape

        # Compute output spatial dims (stride=1 for supported cases)
        H_out, W_out = H, W

        # Initialize V if needed
        if self.v is None:
            self.v = torch.zeros(B, self.out_channels, H_out, W_out,
                                 dtype=torch.float32, device=x.device)

        # Check if V shape matches (batch size may change)
        if self.v.shape[0] != B:
            self.v = torch.zeros(B, self.out_channels, H_out, W_out,
                                 dtype=torch.float32, device=x.device)

        use_triton = (
            self._triton_available
            and x.is_cuda
            and self.stride == (1, 1)
            and self.dilation == (1, 1)
            and self.groups == 1
            and self.kernel_size in [(3, 3), (1, 1)]
        )

        if use_triton:
            spike, v_next = self._triton_forward(x)
        else:
            spike, v_next = self._fallback_forward(x)

        self.v = v_next
        return spike

    def _triton_forward(self, x):
        from Kernels.fused_conv_lif import fused_sparse_conv_lif_forward

        w_cl = self._get_w_cl()
        counts_buf, tile_cin_buf = self._ensure_buffers(x)
        k = self.kernel_size[0]

        spike, v_next, sparse_ms = fused_sparse_conv_lif_forward(
            x=x.contiguous(),
            weight=self.weight,
            bias=self.bias,
            v_prev=self.v,
            kernel_size=k,
            decay=self.decay,
            v_threshold=self.v_threshold,
            threshold=self.threshold,
            w_cl=w_cl,
            counts_buf=counts_buf,
            tile_cin_buf=tile_cin_buf,
            return_ms=self.return_ms,
        )
        self._last_sparse_ms = sparse_ms
        return spike, v_next

    def _fallback_forward(self, x):
        """Dense fallback: Conv2d → LIF dynamics in Python."""
        self._last_sparse_ms = 0.0

        # Standard conv
        y = F.conv2d(x, self.weight, self.bias,
                     self.stride, self.padding, self.dilation, self.groups)

        # LIF dynamics
        v_temp = self.v * self.decay + y.float()
        spike = (v_temp > self.v_threshold).float()
        v_next = v_temp * (1.0 - spike)

        return spike, v_next

    def extra_repr(self) -> str:
        bs = self.block_size if self.block_size is not None else "auto"
        return (f"{self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, tau={self.tau}, "
                f"v_th={self.v_threshold}, fused=True")