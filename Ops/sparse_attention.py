"""
SparseFlow Ops/sparse_attention.py - SparseAttention nn.Module.

This module accelerates attention matmul stages:
  - qk: Q @ K^T
  - av: Attn @ V

It keeps the wrapper contract consistent with other SparseFlow Ops:
  collect_diag, profile_runtime, _last_diag, _last_sparse_ms,
  backend_family, diag_path, fallback_reason.
"""

from __future__ import annotations

import math
import time
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class SparseAttention(nn.Module):
    """Sparse attention matmul wrapper for spike-based transformer blocks."""

    def __init__(
        self,
        num_heads: int = 8,
        head_dim: int = 64,
        threshold: float = 1e-6,
        return_ms: bool = False,
        profile_runtime: bool = False,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.scale = 1.0 / math.sqrt(max(self.head_dim, 1))
        self.threshold = float(threshold)
        self.return_ms = bool(return_ms)
        self.profile_runtime = bool(profile_runtime)

        self._triton_available = False
        try:
            import triton  # noqa: F401
            self._triton_available = True
        except Exception:
            self._triton_available = False

        self.collect_diag = False
        self._last_diag: Dict[str, Any] = {}
        self._last_sparse_ms = 0.0
        self._last_dense_ms = 0.0

        # Standardized observability contract
        self.backend_family = "sparse_attention"
        self.diag_path = "runtime"
        self.fallback_reason = ""
        self.meta_source = "measured"
        self.diag_source = "measured"
        self.support_status = "supported"
        self.score_family = "attn_matmul"

    def _stamp(self, device: torch.device) -> float:
        if self.profile_runtime and device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def _elapsed_ms(self, t0: float, device: torch.device) -> float:
        if self.profile_runtime and device.type == "cuda":
            torch.cuda.synchronize(device)
        return (time.perf_counter() - t0) * 1000.0

    def qk(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Sparse Q @ K^T -> attention logits."""
        if self.collect_diag:
            self._last_diag = {
                "sparse_path_executed": False,
                "op_stage": "attn_matmul_qk",
                "score_family": "attn_matmul",
            }

        t0 = self._stamp(q.device)
        if not self._triton_available or not q.is_cuda:
            y = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
            self.backend_family = "dense_torch"
            self.diag_path = "dense_fallback"
            self.fallback_reason = "triton_unavailable_or_cpu"
            self._last_dense_ms = 0.0
            self._last_sparse_ms = 0.0
            if self.collect_diag:
                self._last_diag.update({
                    "backend": "dense_fallback",
                    "backend_family": self.backend_family,
                    "diag_path": self.diag_path,
                    "fallback_reason": self.fallback_reason,
                    "runtime_total_ms": self._elapsed_ms(t0, q.device),
                })
            return y

        from Kernels.attention import sparse_qk_forward

        result = sparse_qk_forward(
            q=q,
            k=k,
            scale=self.scale,
            threshold=self.threshold,
            return_ms=self.return_ms,
            return_tile_stats=self.collect_diag,
        )
        y = result[0]
        ms = float(result[1])
        self._last_sparse_ms = ms
        self._last_dense_ms = 0.0
        self.backend_family = "sparse_attention"
        self.diag_path = "attn_matmul_qk"
        self.fallback_reason = ""
        if self.collect_diag:
            self._last_diag = {
                "sparse_path_executed": True,
                "op_stage": "attn_matmul_qk",
                "backend": "sparse_triton",
                "backend_family": self.backend_family,
                "diag_path": self.diag_path,
                "fallback_reason": self.fallback_reason,
                "meta_source": self.meta_source,
                "diag_source": self.diag_source,
                "support_status": self.support_status,
                "score_family": "attn_matmul",
                "sparse_total_ms": self._last_sparse_ms,
                "runtime_total_ms": self._elapsed_ms(t0, q.device),
            }
            if len(result) > 2:
                self._last_diag["tile_stats"] = result[2]
        return y

    def av(self, attn: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Sparse Attn @ V -> attention output."""
        if self.collect_diag:
            self._last_diag = {
                "sparse_path_executed": False,
                "op_stage": "attn_matmul_av",
                "score_family": "attn_matmul",
            }

        t0 = self._stamp(attn.device)
        if not self._triton_available or not attn.is_cuda:
            y = torch.matmul(attn.float(), v.float())
            self.backend_family = "dense_torch"
            self.diag_path = "dense_fallback"
            self.fallback_reason = "triton_unavailable_or_cpu"
            self._last_dense_ms = 0.0
            self._last_sparse_ms = 0.0
            if self.collect_diag:
                self._last_diag.update({
                    "backend": "dense_fallback",
                    "backend_family": self.backend_family,
                    "diag_path": self.diag_path,
                    "fallback_reason": self.fallback_reason,
                    "runtime_total_ms": self._elapsed_ms(t0, attn.device),
                })
            return y

        from Kernels.attention import sparse_attn_v_forward

        result = sparse_attn_v_forward(
            attn=attn,
            v=v,
            threshold=self.threshold,
            return_ms=self.return_ms,
            return_tile_stats=self.collect_diag,
        )
        y = result[0]
        ms = float(result[1])
        self._last_sparse_ms = ms
        self._last_dense_ms = 0.0
        self.backend_family = "sparse_attention"
        self.diag_path = "attn_matmul_av"
        self.fallback_reason = ""
        if self.collect_diag:
            self._last_diag = {
                "sparse_path_executed": True,
                "op_stage": "attn_matmul_av",
                "backend": "sparse_triton",
                "backend_family": self.backend_family,
                "diag_path": self.diag_path,
                "fallback_reason": self.fallback_reason,
                "meta_source": self.meta_source,
                "diag_source": self.diag_source,
                "support_status": self.support_status,
                "score_family": "attn_matmul",
                "sparse_total_ms": self._last_sparse_ms,
                "runtime_total_ms": self._elapsed_ms(t0, attn.device),
            }
            if len(result) > 2:
                self._last_diag["tile_stats"] = result[2]
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
            f"num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"threshold={self.threshold}, return_ms={self.return_ms}, "
            f"profile_runtime={self.profile_runtime}, score_family=attn_matmul"
        )
