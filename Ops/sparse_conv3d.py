"""
Ops/sparse_conv3d.py — SparseConv3d wrapper.

Maturity: main_path. Prescan + active-tile sparse compute via
Kernels/conv3d.sparse_conv3d_forward.

Round 5 cleanup (no semantic changes):
  - Hoisted `from Kernels.conv3d import sparse_conv3d_forward` out of the
    forward hot path into a module-level try/except, mirroring the pattern
    used by Ops/sparse_conv2d.py v26.
  - Removed the stale `_warned_v2` migration warning and the `warnings`
    module import.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Module-level kernel availability probe (resolved once at import time).
# ---------------------------------------------------------------------------
try:
    import triton  # noqa: F401
    from Kernels.conv3d import sparse_conv3d_forward
    _TRITON_AVAILABLE = True
except ImportError:
    sparse_conv3d_forward = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


class SparseConv3d(nn.Module):
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
        fallback_ratio: float = 0.85,
        launch_all_tiles: bool = False,
        return_ms: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        ks = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        st = stride if isinstance(stride, int) else stride[0]
        pa = padding if isinstance(padding, int) else padding[0]
        self.kernel_size = (ks, ks, ks)
        self.stride = (st, st, st)
        self.padding = (pa, pa, pa)
        self.groups = int(groups)
        self.threshold = float(threshold)
        self.fallback_ratio = float(fallback_ratio)
        self.launch_all_tiles = bool(launch_all_tiles)
        self.return_ms = bool(return_ms)

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, ks, ks, ks)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        self._triton_available = _TRITON_AVAILABLE

        # Unified observability contract
        self.collect_diag = False
        self.profile_runtime = False
        self._inference_mode = False
        self._last_sparse_ms = 0.0
        self._last_diag: Dict[str, Any] = {}
        self.backend_family = "sparse_kernel"
        self.diag_path = "conv3d_active_tile_sparse"
        self.fallback_reason = ""
        self.meta_source = "measured"
        self.diag_source = "measured"
        self.support_status = "supported"
        self.score_family = "conv"

    @classmethod
    def from_dense(cls, conv: nn.Conv3d, threshold: float = 1e-6,
                   return_ms: bool = False, **kwargs):
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
        if x.ndim == 6:
            t, n, c, d, h, w = x.shape
            x_5d = x.reshape(t * n, c, d, h, w)
            y_5d = self._forward_5d(x_5d)
            return y_5d.reshape(t, n, self.out_channels, *y_5d.shape[2:])
        return self._forward_5d(x)

    def _forward_5d(self, x: torch.Tensor) -> torch.Tensor:
        self._last_diag = {}
        self.fallback_reason = ""

        if self.groups != 1 or not self._triton_available or not x.is_cuda:
            self.diag_path = "dense_fallback"
            self.fallback_reason = "no_triton_or_not_cuda_or_groups"
            self.diag_source = "missing"
            return self._fallback(x)

        result = sparse_conv3d_forward(
            x=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride[0],
            padding=self.padding[0],
            threshold=self.threshold,
            fallback_ratio=self.fallback_ratio,
            launch_all_tiles=self.launch_all_tiles,
            return_ms=self.return_ms,
            return_tile_stats=self.collect_diag,
            return_backend_meta=True,
        )
        y = result[0]
        self._last_sparse_ms = float(result[1])

        stats: Dict[str, Any] = {}
        idx = 2
        if self.collect_diag and len(result) > idx and isinstance(result[idx], dict):
            stats = result[idx]
            idx += 1
        backend_meta = result[idx] if (len(result) > idx and isinstance(result[idx], dict)) else {}
        backend = str(backend_meta.get("backend", stats.get("backend", "sparse_active_tiles")))
        reason = str(backend_meta.get("reason", stats.get("reason", "")))

        if backend == "dense_fallback":
            self.diag_path = "dense_fallback"
            self.fallback_reason = reason or "dense_fallback"
        elif backend == "zero_tiles_only":
            self.diag_path = "zero_tiles_only"
            self.fallback_reason = ""
        else:
            self.diag_path = "sparse_kernel"
            self.fallback_reason = ""

        self.diag_source = "measured" if self.collect_diag else "missing"
        if self.collect_diag:
            self._last_diag = {
                "sparse_path_executed": backend != "dense_fallback",
                "backend_family": self.backend_family,
                "diag_path": self.diag_path,
                "fallback_reason": self.fallback_reason,
            }
            self._last_diag.update(stats)
        return y

    def _fallback(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv3d(
            x.float(),
            self.weight.float(),
            self.bias.float() if self.bias is not None else None,
            self.stride,
            self.padding,
            groups=self.groups,
        ).float()

    def extra_repr(self):
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, groups={self.groups}, "
            f"threshold={self.threshold}, fallback_ratio={self.fallback_ratio}, "
            f"launch_all_tiles={self.launch_all_tiles}, diag_path={self.diag_path}"
        )