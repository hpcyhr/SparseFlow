"""
SparseFlow Ops/sparse_grouped_conv2d.py

SparseGroupedConv2d module wrapper — a contract-aligned grouped Conv2d with
  - from_dense()
  - 4D / 5D input support
  - metadata-first sparse execution
  - return_ms / fallback_ratio / launch_all_tiles
  - structured diagnostics via backend_meta + tile_stats

Also the base class for SparseDepthwiseConv2d (groups == in_channels ==
out_channels); subclass rebinds backend_family / diag_path.

Round 5 cleanup (no semantic changes):
  - Hoisted per-call imports out of `_ensure_group_buffers` and `_forward_4d`
    into a module-level try/except, matching Ops/sparse_conv2d.py v26. The
    `triton`, `_select_tile_sizes`, and `sparse_grouped_conv2d_forward`
    symbols are now resolved once at import time.
  - Removed the `get_diag()` method — it had zero external callers
    (verified by full-repo grep). Diagnostics are read directly via
    `self._last_diag` / `self.backend_family` / `self.diag_path` by
    Core/replacer's observability hooks.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Utils.config import PRESCAN_ACTIVITY_EPS, SPARSE_DENSE_RATIO_THRESHOLD


# ---------------------------------------------------------------------------
# Module-level kernel availability probe (resolved once at import time).
# ---------------------------------------------------------------------------
try:
    import triton  # noqa: F401
    from Kernels.conv2d import _select_tile_sizes
    from Kernels.grouped_conv2d import sparse_grouped_conv2d_forward
    _TRITON_AVAILABLE = True
except ImportError:
    triton = None  # type: ignore[assignment]
    _select_tile_sizes = None  # type: ignore[assignment]
    sparse_grouped_conv2d_forward = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


class SparseGroupedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        groups: int = 2,
        dilation: int = 1,
        threshold: float = PRESCAN_ACTIVITY_EPS,
        fallback_ratio: float = SPARSE_DENSE_RATIO_THRESHOLD,
        launch_all_tiles: bool = False,
        return_ms: bool = False,
    ):
        super().__init__()
        ks = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        st = stride if isinstance(stride, tuple) else (stride, stride)
        pa = padding if isinstance(padding, tuple) else (padding, padding)
        di = dilation if isinstance(dilation, tuple) else (dilation, dilation)

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (int(ks[0]), int(ks[1]))
        self.stride = (int(st[0]), int(st[1]))
        self.padding = (int(pa[0]), int(pa[1]))
        self.dilation = (int(di[0]), int(di[1]))
        self.groups = int(groups)
        self.threshold = float(threshold)
        self.fallback_ratio = float(fallback_ratio)
        self.launch_all_tiles = bool(launch_all_tiles)
        self.return_ms = bool(return_ms)

        self.weight = nn.Parameter(
            torch.empty(self.out_channels, self.in_channels // max(self.groups, 1), *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        self._triton_available = _TRITON_AVAILABLE

        # Per-group buffer caches (lazily allocated on first forward).
        self._w_cl_groups: Optional[List[torch.Tensor]] = None
        self._w_cl_version: int = -1
        self._ag_mask_bufs: Optional[List[torch.Tensor]] = None
        self._tile_class_bufs: Optional[List[torch.Tensor]] = None
        self._active_tile_ids_bufs: Optional[List[torch.Tensor]] = None

        # Unified observability contract
        self.collect_diag = False
        self.profile_runtime = False
        self._inference_mode = False
        self._last_sparse_ms = 0.0
        self._last_dense_ms = 0.0
        self._last_diag: Dict[str, Any] = {}
        self._sparse_backend_family = "sparse_grouped_conv2d"
        self._sparse_diag_path = "grouped_conv2d_sparse"
        self.backend_family = self._sparse_backend_family
        self.diag_path = self._sparse_diag_path
        self.fallback_reason = ""
        self.meta_source = "measured"
        self.diag_source = "measured"
        self.support_status = "supported"
        self.score_family = "conv"

    @classmethod
    def from_dense(cls, conv: nn.Conv2d, threshold: float = PRESCAN_ACTIVITY_EPS,
                   return_ms: bool = False, **kwargs):
        sparse = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
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

    # ------------------------------------------------------------------
    # Per-group weight layout cache
    # ------------------------------------------------------------------
    def _group_weight_layouts(self) -> List[torch.Tensor]:
        version = self.weight._version
        if self._w_cl_groups is not None and self._w_cl_version == version:
            return self._w_cl_groups

        cout_per_group = self.out_channels // max(self.groups, 1)
        layouts: List[torch.Tensor] = []
        for group_idx in range(self.groups):
            cout_start = group_idx * cout_per_group
            cout_end = cout_start + cout_per_group
            w_g = self.weight.data[cout_start:cout_end]
            if self.kernel_size == (3, 3):
                layouts.append(w_g.half().permute(0, 2, 3, 1).contiguous())
            else:
                layouts.append(w_g.half().reshape(cout_per_group, -1).contiguous())

        self._w_cl_groups = layouts
        self._w_cl_version = version
        return layouts

    def _ensure_group_buffers(self, x4d: torch.Tensor):
        _, _, h_in, w_in = x4d.shape
        bh, bw = _select_tile_sizes(h_in, w_in)
        n_tiles = x4d.shape[0] * triton.cdiv(h_in, bh) * triton.cdiv(w_in, bw)

        def _alloc_list(existing: Optional[List[torch.Tensor]]) -> List[torch.Tensor]:
            out: List[torch.Tensor] = []
            for idx in range(self.groups):
                buf = None if existing is None or idx >= len(existing) else existing[idx]
                if buf is None or buf.numel() < n_tiles or buf.device != x4d.device:
                    buf = torch.empty(n_tiles, dtype=torch.int32, device=x4d.device)
                out.append(buf)
            return out

        self._ag_mask_bufs = _alloc_list(self._ag_mask_bufs)
        self._tile_class_bufs = _alloc_list(self._tile_class_bufs)
        self._active_tile_ids_bufs = _alloc_list(self._active_tile_ids_bufs)
        return self._ag_mask_bufs, self._tile_class_bufs, self._active_tile_ids_bufs

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            t, n, c, h, w = x.shape
            x_4d = x.reshape(t * n, c, h, w)
            y_4d = self._forward_4d(x_4d)
            return y_4d.reshape(t, n, self.out_channels, y_4d.shape[2], y_4d.shape[3])
        return self._forward_4d(x)

    def _forward_4d(self, x: torch.Tensor) -> torch.Tensor:
        self._last_diag = {}
        self.fallback_reason = ""

        if self.groups <= 1 or not self._triton_available or not x.is_cuda:
            self.diag_path = "dense_fallback"
            self.diag_source = "missing"
            self.fallback_reason = "no_triton_or_not_cuda_or_groups<=1"
            return self._fallback(x)

        w_cl_groups = self._group_weight_layouts()
        ag_mask_bufs, tile_class_bufs, active_tile_ids_bufs = self._ensure_group_buffers(x)

        try:
            result = sparse_grouped_conv2d_forward(
                x=x,
                weight=self.weight,
                bias=self.bias,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                threshold=self.threshold,
                w_cl_groups=w_cl_groups,
                ag_mask_bufs=ag_mask_bufs,
                tile_class_bufs=tile_class_bufs,
                active_tile_ids_bufs=active_tile_ids_bufs,
                return_ms=self.return_ms,
                fallback_ratio=self.fallback_ratio,
                return_avg_active_ratio=self.collect_diag,
                return_tile_stats=self.collect_diag,
                return_backend_meta=True,
                launch_all_tiles=self.launch_all_tiles,
            )
        except Exception as err:
            self.backend_family = "dense_torch"
            self.diag_path = "dense_fallback"
            self.diag_source = "missing"
            self.fallback_reason = "triton_runtime_error"
            if self.collect_diag:
                self._last_diag = {
                    "sparse_path_executed": False,
                    "backend": "dense_fallback",
                    "backend_family": self.backend_family,
                    "diag_path": self.diag_path,
                    "fallback_reason": self.fallback_reason,
                    "backend_reason": "triton_runtime_error",
                    "error_type": type(err).__name__,
                    "error_msg": str(err),
                }
            return self._fallback(x)

        y = result[0]
        self._last_sparse_ms = float(result[1])

        idx = 2
        avg_active_ratio = None
        if self.collect_diag and len(result) > idx:
            avg_active_ratio = result[idx]
            idx += 1

        tile_stats: Dict[str, Any] = {}
        if self.collect_diag and len(result) > idx and isinstance(result[idx], dict):
            tile_stats = result[idx]
            idx += 1

        backend_meta = result[idx] if (len(result) > idx and isinstance(result[idx], dict)) else {}
        backend = str(backend_meta.get("backend", "sparse_triton"))
        reason = str(backend_meta.get("reason", ""))

        if backend == "dense_fallback":
            self._last_sparse_ms = 0.0
            self._last_dense_ms = float(result[1])
            self.backend_family = "dense_torch"
            self.diag_path = "dense_fallback"
            self.fallback_reason = reason or "dense_fallback"
        elif backend == "zero_tiles_only":
            self._last_dense_ms = 0.0
            self.backend_family = "exact_zero"
            self.diag_path = "zero_tiles_only"
            self.fallback_reason = ""
        else:
            self._last_dense_ms = 0.0
            self.backend_family = self._sparse_backend_family
            self.diag_path = self._sparse_diag_path
            self.fallback_reason = ""

        self.diag_source = "measured" if self.collect_diag else "missing"
        if self.collect_diag:
            self._last_diag = {
                "sparse_path_executed": backend != "dense_fallback",
                "backend": backend,
                "backend_family": self.backend_family,
                "diag_path": self.diag_path,
                "fallback_reason": self.fallback_reason,
                "metadata_kind": "groupwise_conv2d_v1",
                "kernel_type": f"{self.kernel_size[0]}x{self.kernel_size[1]}/s{self.stride[0]}",
                "groups": int(self.groups),
                "avg_active_ratio": float(avg_active_ratio) if avg_active_ratio is not None else -1.0,
            }
            if tile_stats:
                self._last_diag.update(tile_stats)
            if backend_meta:
                self._last_diag.update(
                    {
                        "launch_tile_count": float(backend_meta.get("launch_count", -1)),
                        "active_tiles": float(backend_meta.get("active_tiles", -1)),
                        "backend_reason": backend_meta.get("reason", ""),
                    }
                )
            group_size_c = self._last_diag.get("group_size_c", -1)
            num_groups = self._last_diag.get("num_groups", -1)
            total_tiles = self._last_diag.get("total_tiles", -1)
            avg_ratio = self._last_diag.get("avg_active_group_ratio", -1.0)
            if (
                isinstance(group_size_c, (int, float))
                and isinstance(num_groups, (int, float))
                and isinstance(total_tiles, (int, float))
                and avg_ratio is not None
                and float(num_groups) > 0
                and float(total_tiles) > 0
            ):
                total_group_count = float(total_tiles) * float(num_groups)
                self._last_diag["group_size"] = float(group_size_c)
                self._last_diag["num_groups"] = float(num_groups)
                self._last_diag["total_group_count"] = total_group_count
                self._last_diag["nonzero_group_count"] = float(avg_ratio) * total_group_count
                self._last_diag["active_group_ratio"] = float(avg_ratio)
                self._last_diag["tile_zero_count"] = float(self._last_diag.get("zero_tiles", -1))
                self._last_diag["total_tile_count"] = float(total_tiles)
                if total_tiles > 0:
                    self._last_diag["tile_zero_ratio"] = (
                        float(self._last_diag.get("zero_tiles", 0)) / float(total_tiles)
                    )

        return y

    def _fallback(self, x: torch.Tensor) -> torch.Tensor:
        self._last_sparse_ms = 0.0
        self._last_dense_ms = 0.0
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, groups={self.groups}, "
            f"threshold={self.threshold}, fallback_ratio={self.fallback_ratio}, "
            f"launch_all_tiles={self.launch_all_tiles}, diag_path={self.diag_path}"
        )
