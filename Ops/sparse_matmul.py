"""
SparseFlow Ops/sparse_matmul.py - SparseMatmul nn.Module wrapper.

This wrapper follows the same observability contract used by SparseConv2d and
SparseLinear:
  - collect_diag / profile_runtime
  - _last_diag / _last_sparse_ms
  - backend_family / diag_path / fallback_reason
"""

from __future__ import annotations

import time
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class SparseMatmul(nn.Module):
    """Sparse matmul operator C = A @ B where A is expected sparse."""

    def __init__(
        self,
        threshold: float = 1e-6,
        fallback_ratio: float = 0.85,
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
        self.backend_family = "sparse_matmul"
        self.diag_path = "runtime"
        self.fallback_reason = ""
        self.meta_source = "measured"
        self.diag_source = "measured"
        self.support_status = "supported"
        self.score_family = "unknown"

    def _stamp(self, device: torch.device) -> float:
        if self.profile_runtime and device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def _elapsed_ms(self, t0: float, device: torch.device) -> float:
        if self.profile_runtime and device.type == "cuda":
            torch.cuda.synchronize(device)
        return (time.perf_counter() - t0) * 1000.0

    def _record_dense_diag(self, *, reason: str, runtime_total_ms: float = -1.0):
        self.backend_family = "dense_torch"
        self.diag_path = "dense_fallback"
        self.fallback_reason = reason
        if self.collect_diag:
            self._last_diag = {
                "sparse_path_executed": False,
                "backend": "dense_fallback",
                "backend_family": self.backend_family,
                "diag_path": self.diag_path,
                "fallback_reason": self.fallback_reason,
                "meta_source": self.meta_source,
                "diag_source": self.diag_source,
                "support_status": self.support_status,
                "score_family": self.score_family,
                "sparse_total_ms": 0.0,
                "dense_fallback_ms": float(self._last_dense_ms),
                "runtime_total_ms": float(runtime_total_ms),
            }

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
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

        # Handle higher-rank input by flattening leading dimensions.
        if a.ndim > 2:
            lead = a.shape[:-2]
            m_dim, k_dim = a.shape[-2], a.shape[-1]
            n_dim = b.shape[-1]
            a_flat = a.reshape(-1, k_dim)
            if b.ndim > 2:
                b_flat = b.reshape(-1, k_dim, n_dim)
                batch = a_flat.shape[0] // m_dim
                a_3d = a_flat.reshape(batch, m_dim, k_dim)
                out = self._bmm_forward(a_3d, b_flat)
                y = out.reshape(*lead, m_dim, n_dim)
            else:
                out = self._matmul_forward(a_flat, b)
                y = out.reshape(*lead, m_dim, n_dim)
        else:
            y = self._matmul_forward(a, b)

        if self.collect_diag:
            self._last_diag["runtime_total_ms"] = self._elapsed_ms(t_total, a.device)
        return y

    def _matmul_forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not self._triton_available or not a.is_cuda:
            y = torch.mm(a.float(), b.float())
            self._last_dense_ms = 0.0
            self._record_dense_diag(reason="triton_unavailable_or_cpu")
            return y

        from Kernels.matmul import sparse_matmul_forward

        want_stats = self.collect_diag
        result = sparse_matmul_forward(
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
            self.backend_family = "sparse_matmul"
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
            }
            if isinstance(tile_stats, dict):
                self._last_diag["tile_stats"] = tile_stats
                for key in ("total_tiles", "zero_tiles", "sparse_tiles", "denseish_tiles"):
                    if key in tile_stats:
                        self._last_diag[key] = tile_stats[key]
        return y

    def _bmm_forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not self._triton_available or not a.is_cuda:
            y = torch.bmm(a.float(), b.float())
            self._last_dense_ms = 0.0
            self._record_dense_diag(reason="triton_unavailable_or_cpu")
            return y

        from Kernels.bmm import sparse_bmm_forward

        want_stats = self.collect_diag
        result = sparse_bmm_forward(
            a=a,
            b=b,
            threshold=self.threshold,
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
