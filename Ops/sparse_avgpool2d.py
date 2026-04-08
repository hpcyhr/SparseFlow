"""
SparseFlow Ops/sparse_avgpool2d.py

SparseAvgPool2d module wrapper.

This keeps AvgPool2d on the same SparseFlow runtime contract used by the
other sparse operators:
  - metadata-first sparse execution
  - 4D / 5D input support
  - return_ms / launch_all_tiles
  - structured diagnostics via backend_meta + tile_stats
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.config import PRESCAN_ACTIVITY_EPS, SPARSE_DENSE_RATIO_THRESHOLD


def _pair(v) -> Tuple[int, int]:
    if isinstance(v, (tuple, list)):
        if len(v) == 1:
            return int(v[0]), int(v[0])
        return int(v[0]), int(v[1])
    return int(v), int(v)


def k_pair_str(v: Tuple[int, int]) -> str:
    return f"{int(v[0])}x{int(v[1])}"


class SparseAvgPool2d(nn.Module):
    def __init__(
        self,
        kernel_size=2,
        stride=None,
        padding=0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        threshold: float = PRESCAN_ACTIVITY_EPS,
        fallback_ratio: float = SPARSE_DENSE_RATIO_THRESHOLD,
        launch_all_tiles: bool = False,
        return_ms: bool = False,
    ):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.ceil_mode = bool(ceil_mode)
        self.count_include_pad = bool(count_include_pad)
        self.divisor_override = divisor_override
        self.threshold = float(threshold)
        self.fallback_ratio = float(fallback_ratio)
        self.launch_all_tiles = bool(launch_all_tiles)
        self.return_ms = bool(return_ms)

        self._triton_available = False
        try:
            import triton  # noqa: F401

            self._triton_available = True
        except Exception:
            pass

        self._ag_mask_buf: Optional[torch.Tensor] = None
        self._tile_class_buf: Optional[torch.Tensor] = None
        self._active_tile_ids_buf: Optional[torch.Tensor] = None

        self.collect_diag = False
        self.profile_runtime = False
        self._inference_mode = False
        self._last_sparse_ms = 0.0
        self._last_dense_ms = 0.0
        self._last_diag: Dict[str, Any] = {}
        self.backend_family = "sparse_avgpool2d"
        self.diag_path = "avgpool2d_sparse"
        self.fallback_reason = ""
        self.meta_source = "measured"
        self.diag_source = "measured"
        self.support_status = "supported"
        self.score_family = "pool"

    @classmethod
    def from_dense(
        cls,
        pool: nn.AvgPool2d,
        threshold: float = PRESCAN_ACTIVITY_EPS,
        return_ms: bool = False,
        **kwargs,
    ):
        stride = pool.stride if getattr(pool, "stride", None) is not None else pool.kernel_size
        return cls(
            kernel_size=pool.kernel_size,
            stride=stride,
            padding=pool.padding,
            ceil_mode=pool.ceil_mode,
            count_include_pad=pool.count_include_pad,
            divisor_override=pool.divisor_override,
            threshold=threshold,
            return_ms=return_ms,
            **kwargs,
        )

    def set_launch_all_tiles(self, enabled: bool):
        self.launch_all_tiles = bool(enabled)

    def set_inference_mode(self, enabled: bool):
        self._inference_mode = bool(enabled)
        if enabled:
            self.collect_diag = False
            self.profile_runtime = False

    def _ensure_buffers(self, x4d: torch.Tensor):
        from Kernels.avgpool2d import _select_pool_tile_sizes
        import triton

        _, c, h_in, w_in = x4d.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        h_out = (h_in + 2 * ph - kh) // sh + 1
        w_out = (w_in + 2 * pw - kw) // sw + 1
        bh, bw = _select_pool_tile_sizes(h_out, w_out)
        total_tiles = x4d.shape[0] * c * triton.cdiv(h_out, bh) * triton.cdiv(w_out, bw)

        if self._ag_mask_buf is None or self._ag_mask_buf.numel() < total_tiles or self._ag_mask_buf.device != x4d.device:
            self._ag_mask_buf = torch.empty(total_tiles, dtype=torch.int32, device=x4d.device)
        if self._tile_class_buf is None or self._tile_class_buf.numel() < total_tiles or self._tile_class_buf.device != x4d.device:
            self._tile_class_buf = torch.empty(total_tiles, dtype=torch.int32, device=x4d.device)
        if self._active_tile_ids_buf is None or self._active_tile_ids_buf.numel() < total_tiles or self._active_tile_ids_buf.device != x4d.device:
            self._active_tile_ids_buf = torch.empty(total_tiles, dtype=torch.int32, device=x4d.device)
        return self._ag_mask_buf, self._tile_class_buf, self._active_tile_ids_buf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            t, n, c, h, w = x.shape
            x_4d = x.reshape(t * n, c, h, w)
            y_4d = self._forward_4d(x_4d)
            return y_4d.reshape(t, n, c, y_4d.shape[2], y_4d.shape[3])
        return self._forward_4d(x)

    def _forward_4d(self, x: torch.Tensor) -> torch.Tensor:
        self._last_diag = {}
        self.fallback_reason = ""

        if not self._triton_available or not x.is_cuda:
            self.backend_family = "dense_torch"
            self.diag_path = "dense_fallback"
            self.diag_source = "missing"
            self.fallback_reason = "no_triton_or_not_cuda"
            return self._fallback(x)

        from Kernels.avgpool2d import sparse_avgpool2d_forward

        ag_mask_buf, tile_class_buf, active_tile_ids_buf = self._ensure_buffers(x)

        try:
            result = sparse_avgpool2d_forward(
                x=x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
                divisor_override=self.divisor_override,
                threshold=self.threshold,
                ag_mask_buf=ag_mask_buf,
                tile_class_buf=tile_class_buf,
                return_ms=self.return_ms,
                fallback_ratio=self.fallback_ratio,
                return_avg_active_ratio=self.collect_diag,
                return_tile_stats=self.collect_diag,
                return_backend_meta=True,
                active_tile_ids_buf=active_tile_ids_buf,
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

        tile_stats = {}
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
            self.backend_family = "sparse_avgpool2d"
            self.diag_path = "avgpool2d_sparse"
            self.fallback_reason = ""

        self.diag_source = "measured" if self.collect_diag else "missing"
        if self.collect_diag:
            self._last_diag = {
                "sparse_path_executed": backend != "dense_fallback",
                "backend": backend,
                "backend_family": self.backend_family,
                "diag_path": self.diag_path,
                "fallback_reason": self.fallback_reason,
                "metadata_kind": "pool2d_zero_skip_v1",
                "kernel_type": f"avgpool{k_pair_str(self.kernel_size)}/s{k_pair_str(self.stride)}",
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
            total_tiles = self._last_diag.get("total_tiles", -1)
            if total_tiles and float(total_tiles) > 0:
                zero_tiles = float(self._last_diag.get("zero_tiles", 0))
                self._last_diag["group_size"] = 1.0
                self._last_diag["num_groups"] = 1.0
                self._last_diag["total_group_count"] = float(total_tiles)
                self._last_diag["nonzero_group_count"] = float(total_tiles) - zero_tiles
                self._last_diag["active_group_ratio"] = float(avg_active_ratio if avg_active_ratio is not None else 0.0)
                self._last_diag["tile_zero_count"] = zero_tiles
                self._last_diag["total_tile_count"] = float(total_tiles)
                self._last_diag["tile_zero_ratio"] = zero_tiles / float(total_tiles)

        return y

    def _fallback(self, x: torch.Tensor) -> torch.Tensor:
        self._last_sparse_ms = 0.0
        self._last_dense_ms = 0.0
        return F.avg_pool2d(
            x.float(),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=self.divisor_override,
        ).float()

    def get_diag(self) -> Dict[str, Any]:
        diag = dict(self._last_diag or {})
        diag.setdefault("backend_family", self.backend_family)
        diag.setdefault("diag_path", self.diag_path)
        diag.setdefault("fallback_reason", self.fallback_reason)
        diag.setdefault("meta_source", self.meta_source)
        diag.setdefault("diag_source", self.diag_source)
        diag.setdefault("support_status", self.support_status)
        diag.setdefault("score_family", self.score_family)
        diag.setdefault("sparse_total_ms", float(self._last_sparse_ms))
        diag.setdefault("dense_fallback_ms", float(self._last_dense_ms))
        return diag

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
            f"ceil_mode={self.ceil_mode}, count_include_pad={self.count_include_pad}, "
            f"divisor_override={self.divisor_override}, threshold={self.threshold}, "
            f"launch_all_tiles={self.launch_all_tiles}, diag_path={self.diag_path}"
        )
