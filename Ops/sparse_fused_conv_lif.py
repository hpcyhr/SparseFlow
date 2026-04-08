"""
FusedSparseConvLIF — v18 bitmask-based stateful Conv+LIF fused module.

Changes from v16:
  - _ag_count_buf / _ag_list_buf replaced with single _ag_mask_buf
  - Uses choose_group_size() for adaptive GROUP_SIZE
  - Forwards to bitmask-based fused kernel
"""

import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import base as sj_base
from Utils.config import PRESCAN_ACTIVITY_EPS, SPARSE_DENSE_RATIO_THRESHOLD


class FusedSparseConvLIF(sj_base.MemoryModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        detach_reset: bool = False,
        decay_input: bool = True,
        backend: str = 'torch',
        threshold: float = PRESCAN_ACTIVITY_EPS,
        block_size=None,
        return_ms: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups

        self.tau = float(tau)
        self.v_threshold = float(v_threshold)
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.decay_input = bool(decay_input)
        self.backend = backend
        self.threshold = float(threshold)
        self.block_size = block_size
        self.return_ms = bool(return_ms)

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self._triton_available = False
        try:
            import triton  # noqa: F401
            self._triton_available = True
        except Exception:
            pass

        self._last_sparse_ms = 0.0
        self._w_cl = None

        # Bitmask buffer (replaces ag_count_buf + ag_list_buf)
        self._ag_mask_buf = None

        self._warmup_steps = 8
        self._calib_interval = 32
        self._ema_decay = 0.9
        self._dense_threshold = SPARSE_DENSE_RATIO_THRESHOLD
        self._ema_active_ratio = None
        self._step_count = 0
        self._runtime_mode = 'sparse'

        self.register_memory('v', None)

    @classmethod
    def from_conv_and_lif(cls, conv: nn.Conv2d, lif_node, block_size=None,
                          threshold: float = PRESCAN_ACTIVITY_EPS, return_ms: bool = False):
        mod = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=(conv.bias is not None),
            tau=float(getattr(lif_node, 'tau', 2.0)),
            v_threshold=float(getattr(lif_node, 'v_threshold', 1.0)),
            v_reset=getattr(lif_node, 'v_reset', 0.0),
            detach_reset=bool(getattr(lif_node, 'detach_reset', False)),
            decay_input=bool(getattr(lif_node, 'decay_input', True)),
            backend=str(getattr(lif_node, 'backend', 'torch')),
            threshold=threshold,
            block_size=block_size,
            return_ms=return_ms,
        )
        mod.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            mod.bias.data.copy_(conv.bias.data)
        return mod.to(conv.weight.device)

    def _conv_output_hw(self, h: int, w: int):
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation
        out_h = ((h + 2 * ph - dh * (kh - 1) - 1) // sh) + 1
        out_w = ((w + 2 * pw - dw * (kw - 1) - 1) // sw) + 1
        return out_h, out_w

    def _get_w_cl(self):
        ver = self.weight._version
        if self._w_cl is None or getattr(self, '_w_cl_version', -1) != ver:
            k = self.kernel_size[0]
            if k == 3:
                self._w_cl = self.weight.data.half().permute(0, 2, 3, 1).contiguous()
            else:
                self._w_cl = self.weight.data.half().reshape(self.out_channels, self.in_channels).contiguous()
            self._w_cl_version = ver
        return self._w_cl

    def _ensure_buffers(self, x_single):
        """Allocate bitmask buffer (one int32 per tile)."""
        from Kernels.conv2d import _select_tile_sizes
        import triton
        B, C_IN, H, W = x_single.shape
        BH, BW = _select_tile_sizes(H, W)
        GH = triton.cdiv(H, BH)
        GW = triton.cdiv(W, BW)
        n_tiles = B * GH * GW

        if (self._ag_mask_buf is None
                or self._ag_mask_buf.numel() < n_tiles
                or self._ag_mask_buf.device != x_single.device):
            self._ag_mask_buf = torch.empty(n_tiles, dtype=torch.int32, device=x_single.device)

        return self._ag_mask_buf

    def _supports_triton_fused(self, x):
        return (
            self._triton_available and x.is_cuda
            and self.kernel_size == (3, 3)
            and self.stride == (1, 1)
            and self.padding == (1, 1)
            and self.dilation == (1, 1)
            and self.groups == 1
        )

    def forward(self, x):
        if x.dim() == 5:
            return self._forward_multistep(x)
        if x.dim() == 4:
            return self._forward_singlestep(x)
        raise ValueError(f'Expected 4D or 5D input, got {x.dim()}D')

    def _forward_multistep(self, x):
        T, B, C, H, W = x.shape
        out_h, out_w = self._conv_output_hw(H, W)
        out = torch.empty(T, B, self.out_channels, out_h, out_w, dtype=torch.float32, device=x.device)
        for t in range(T):
            out[t] = self._forward_singlestep(x[t])
        return out

    def _forward_singlestep(self, x):
        B, C, H, W = x.shape
        out_h, out_w = self._conv_output_hw(H, W)
        if self.v is None or self.v.shape != (B, self.out_channels, out_h, out_w) or self.v.device != x.device:
            self.v = torch.zeros(B, self.out_channels, out_h, out_w, dtype=torch.float32, device=x.device)

        if not self._supports_triton_fused(x):
            spike, v_next = self._fallback_forward(x)
            self.v = v_next
            return spike

        need_ratio = (self._step_count < self._warmup_steps) or (self._step_count % self._calib_interval == 0)
        spike, v_next, avg_active_ratio = self._triton_forward(x, need_ratio=need_ratio)

        if avg_active_ratio is not None:
            if self._ema_active_ratio is None:
                self._ema_active_ratio = avg_active_ratio
            else:
                self._ema_active_ratio = self._ema_decay * self._ema_active_ratio + (1.0 - self._ema_decay) * avg_active_ratio
            self._runtime_mode = 'dense' if self._ema_active_ratio > self._dense_threshold else 'sparse'

        if self._runtime_mode == 'dense':
            spike, v_next = self._fallback_forward(x)

        self.v = v_next
        self._step_count += 1
        return spike

    def _triton_forward(self, x, need_ratio: bool = False):
        from Kernels.fused_conv_lif import sparse_fused_conv_lif_forward
        w_cl = self._get_w_cl()
        ag_mask_buf = self._ensure_buffers(x)

        if need_ratio:
            spike, v_next, ms, avg_active_ratio = sparse_fused_conv_lif_forward(
                x=x.contiguous(),
                v_prev=self.v,
                weight=self.weight,
                bias=self.bias,
                tau=self.tau,
                v_threshold=self.v_threshold,
                v_reset=self.v_reset,
                decay_input=self.decay_input,
                kernel_size=self.kernel_size[0],
                threshold=self.threshold,
                w_cl=w_cl,
                ag_mask_buf=ag_mask_buf,
                return_ms=self.return_ms,
                return_avg_active_ratio=True,
            )
            self._last_sparse_ms = ms
            return spike, v_next, avg_active_ratio

        spike, v_next, ms = sparse_fused_conv_lif_forward(
            x=x.contiguous(),
            v_prev=self.v,
            weight=self.weight,
            bias=self.bias,
            tau=self.tau,
            v_threshold=self.v_threshold,
            v_reset=self.v_reset,
            decay_input=self.decay_input,
            kernel_size=self.kernel_size[0],
            threshold=self.threshold,
            w_cl=w_cl,
            ag_mask_buf=ag_mask_buf,
            return_ms=self.return_ms,
            return_avg_active_ratio=False,
        )
        self._last_sparse_ms = ms
        return spike, v_next, None

    def _fallback_forward(self, x):
        self._last_sparse_ms = 0.0
        conv = F.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        ).float()
        v_prev = self.v
        v_reset = 0.0 if self.v_reset is None else float(self.v_reset)
        if self.decay_input:
            v_tmp = v_prev + (conv - (v_prev - v_reset)) / self.tau
        else:
            v_tmp = v_prev - (v_prev - v_reset) / self.tau + conv
        spike = (v_tmp >= self.v_threshold).to(v_tmp.dtype)
        if self.v_reset is None:
            v_next = v_tmp - spike * self.v_threshold
        else:
            v_next = torch.where(spike.bool(), torch.full_like(v_tmp, float(self.v_reset)), v_tmp)
        return spike, v_next

    def extra_repr(self):
        bs = self.block_size if self.block_size is not None else 'auto'
        return (
            f'{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, '
            f'stride={self.stride}, padding={self.padding}, tau={self.tau}, '
            f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, block_size={bs}, '
            f'fused=True, metadata=bitmask'
        )
