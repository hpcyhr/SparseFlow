"""
SparseFlow Ops/sparse_bmm.py - SparseBMM nn.Module wrapper.

Batched sparse matmul C[b] = A[b] @ B[b] with standardized observability
fields aligned with other SparseFlow wrappers.
"""

from __future__ import annotations

import time
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from Utils.config import PRESCAN_ACTIVITY_EPS, SPARSE_DENSE_RATIO_THRESHOLD

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class SparseBMM(nn.Module):
    """Sparse batched matmul where A is expected sparse."""

    def __init__(
        self,
        threshold: float = PRESCAN_ACTIVITY_EPS,
        fallback_ratio: float = SPARSE_DENSE_RATIO_THRESHOLD,
        return_ms: bool = False,
        profile_runtime: bool = False,
    ):
        super().__init__()
        self.threshold = float(threshold)
        self.fallback_ratio = float(fallback_ratio)
        self.return_ms = bool(return_ms)
        self.profile_runtime = bool(profile_runtime)

        self._triton_available = False
        try:
            import triton  # noqa: F401
            self._triton_available = True
        except Exception:
            self._triton_available = False

        self._ag_mask_buf = None
        self._tile_class_buf = None
        self.collect_diag = False
        self._last_diag: Dict[str, Any] = {}
        self._last_sparse_ms = 0.0
        self._last_dense_ms = 0.0

        # Standardized observability contract
        self.backend_family = "sparse_bmm"
        self.diag_path = "runtime"
        self.fallback_reason = ""
        self.meta_source = "measured"
        self.diag_source = "measured"
        self.support_status = "supported"
        self.score_family = "bmm"

    def _stamp(self, device: torch.device) -> float:
        if self.profile_runtime and device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def _elapsed_ms(self, t0: float, device: torch.device) -> float:
        if self.profile_runtime and device.type == "cuda":
            torch.cuda.synchronize(device)
        return (time.perf_counter() - t0) * 1000.0

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.ndim != 3 or b.ndim != 3:
            raise ValueError(f"SparseBMM expects 3D tensors, got {a.shape} and {b.shape}")

        if self.collect_diag:
            self._last_diag = {
                "sparse_path_executed": False,
                "backend_family": self.backend_family,
                "diag_path": self.diag_path,
                "fallback_reason": self.fallback_reason,
                "meta_source": self.meta_source,
                "diag_source": self.diag_source,
                "support_status": self.support_status,
                "score_family": self.score_family,
            }
        t_total = self._stamp(a.device)
        self._last_sparse_ms = 0.0
        self._last_dense_ms = 0.0

        if not self._triton_available or not a.is_cuda:
            y = torch.bmm(a.float(), b.float())
            self.backend_family = "dense_torch"
            self.diag_path = "dense_fallback"
            self.fallback_reason = "triton_unavailable_or_cpu"
            if self.collect_diag:
                self._last_diag.update({
                    "backend": "dense_fallback",
                    "backend_family": self.backend_family,
                    "diag_path": self.diag_path,
                    "fallback_reason": self.fallback_reason,
                    "dense_fallback_ms": 0.0,
                    "runtime_total_ms": self._elapsed_ms(t_total, a.device),
                })
            return y

        from Kernels.bmm import sparse_bmm_forward

        want_stats = self.collect_diag
        result = sparse_bmm_forward(
            a=a,
            b=b,
            threshold=self.threshold,
            ag_mask_buf=self._ag_mask_buf,
            tile_class_buf=self._tile_class_buf,
            return_ms=self.return_ms,
            return_avg_active_ratio=want_stats,
            return_tile_stats=want_stats,
            fallback_ratio=self.fallback_ratio,
        )

        y = result[0]
        ms = float(result[1])
        self._last_sparse_ms = ms
        idx = 2
        avg_ratio = None
        tile_stats = None
        if want_stats and idx < len(result):
            avg_ratio = result[idx]
            idx += 1
        if want_stats and idx < len(result):
            tile_stats = result[idx]

        fallback = bool(tile_stats.get("fallback", False)) if isinstance(tile_stats, dict) else False
        if fallback:
            self.backend_family = "dense_torch"
            self.diag_path = "dense_fallback"
            self.fallback_reason = str(tile_stats.get("reason", "dense_fallback"))
            self._last_dense_ms = ms
            self._last_sparse_ms = 0.0
        else:
            self.backend_family = "sparse_bmm"
            self.diag_path = "sparse_kernel"
            self.fallback_reason = ""
            self._last_dense_ms = 0.0

        if self.collect_diag:
            self._last_diag = {
                "sparse_path_executed": not fallback,
                "backend": "dense_fallback" if fallback else "sparse_triton",
                "backend_family": self.backend_family,
                "diag_path": self.diag_path,
                "fallback_reason": self.fallback_reason,
                "meta_source": self.meta_source,
                "diag_source": self.diag_source,
                "support_status": self.support_status,
                "score_family": self.score_family,
                "avg_active_ratio": float(avg_ratio) if avg_ratio is not None else -1.0,
                "sparse_total_ms": float(self._last_sparse_ms),
                "dense_fallback_ms": float(self._last_dense_ms),
                "runtime_total_ms": self._elapsed_ms(t_total, a.device),
            }
            if isinstance(tile_stats, dict):
                self._last_diag["tile_stats"] = tile_stats
        return y

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

    def extra_repr(self):
        return (
            f"threshold={self.threshold}, fallback_ratio={self.fallback_ratio}, "
            f"return_ms={self.return_ms}, profile_runtime={self.profile_runtime}, "
            f"backend_family={self.backend_family}"
        )
