"""
Ops/sparse_conv1d.py

SparseConv1d wrapper.

Current maturity: prototype / stats path.
This operator currently runs prescan + dense compute fallback in kernel v1.
"""

import sys
import warnings
from pathlib import Path
from typing import Any, Dict

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseConv1d(nn.Module):
    _warned_prescan_only = False

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        groups: int = 1,
        threshold: float = 1e-6,
        return_ms: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride,) if isinstance(stride, int) else stride
        self.padding = (padding,) if isinstance(padding, int) else padding
        self.groups = int(groups)
        self.threshold = float(threshold)
        self.return_ms = bool(return_ms)

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, self.kernel_size[0]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        self._triton_available = False
        try:
            import triton  # noqa: F401

            self._triton_available = True
        except Exception:
            pass

        # Unified observability contract
        self.collect_diag = False
        self.profile_runtime = False
        self._inference_mode = False
        self._last_sparse_ms = 0.0
        self._last_diag: Dict[str, Any] = {}
        self.backend_family = "sparse_kernel"
        self.diag_path = "conv1d_v1_prescan_only"
        self.fallback_reason = ""
        self.meta_source = "measured"
        self.diag_source = "measured"
        self.support_status = "supported"
        self.score_family = "conv"

    @classmethod
    def from_dense(cls, conv: nn.Conv1d, threshold: float = 1e-6, return_ms: bool = False, **kwargs):
        if not cls._warned_prescan_only:
            warnings.warn(
                "[SparseFlow] SparseConv1d is currently prescan-only v1 (dense compute fallback).",
                UserWarning,
            )
            cls._warned_prescan_only = True
        sparse = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size[0],
            stride=conv.stride[0],
            padding=conv.padding[0],
            bias=conv.bias is not None,
            groups=conv.groups,
            threshold=threshold,
            return_ms=return_ms,
            **kwargs,
        )
        sparse = sparse.to(conv.weight.device)
        sparse.weight.data.copy_(conv.weight.data)
        if conv.bias is not None and sparse.bias is not None:
            sparse.bias.data.copy_(conv.bias.data)
        return sparse

    def set_inference_mode(self, enabled: bool):
        self._inference_mode = bool(enabled)
        if enabled:
            self.collect_diag = False
            self.profile_runtime = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            t, n, c, l = x.shape
            x_3d = x.reshape(t * n, c, l)
            y_3d = self._forward_3d(x_3d)
            l_out = y_3d.shape[2]
            return y_3d.reshape(t, n, self.out_channels, l_out)
        return self._forward_3d(x)

    def _forward_3d(self, x: torch.Tensor) -> torch.Tensor:
        self._last_diag = {}
        self.fallback_reason = ""
        if self.groups != 1 or not self._triton_available or not x.is_cuda:
            self.fallback_reason = "no_triton_or_not_cuda_or_groups"
            self.diag_source = "missing"
            return self._fallback(x)

        from Kernels.conv1d import sparse_conv1d_forward

        result = sparse_conv1d_forward(
            x=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride[0],
            padding=self.padding[0],
            threshold=self.threshold,
            return_ms=self.return_ms,
            return_tile_stats=self.collect_diag,
        )
        y = result[0]
        self._last_sparse_ms = float(result[1])
        self.fallback_reason = "prescan_only_dense_compute_v1"
        self.diag_source = "measured" if self.collect_diag else "missing"
        if self.collect_diag and len(result) > 2:
            self._last_diag = result[2] if isinstance(result[2], dict) else {}
            self._last_diag.setdefault("backend_family", self.backend_family)
            self._last_diag.setdefault("diag_path", self.diag_path)
            self._last_diag.setdefault("fallback_reason", self.fallback_reason)
        return y

    def _fallback(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(
            x.float(),
            self.weight.float(),
            self.bias.float() if self.bias is not None else None,
            self.stride,
            self.padding,
            groups=self.groups,
        ).float()

    def extra_repr(self):
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, groups={self.groups}, "
            f"threshold={self.threshold}, diag_path={self.diag_path}"
        )

