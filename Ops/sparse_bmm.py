"""
SparseFlow Ops/sparse_bmm.py — SparseBMM nn.Module Wrapper

Batched sparse matmul module: C[b] = A[b] @ B[b] where A is sparse.
Functional (no learnable parameters). Manages metadata buffers.

Primary use case: Spikeformer attention (Q×K^T, attn×V).
"""

import sys
from pathlib import Path
from typing import Dict, Any

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn


class SparseBMM(nn.Module):
    """
    Sparse batched matmul C[b] = A[b] @ B[b] where A is expected sparse.
    """

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
            pass

        self._ag_mask_buf = None
        self._tile_class_buf = None
        self._last_sparse_ms = 0.0
        self._last_diag: Dict[str, Any] = {}
        self.collect_diag = False
        self._inference_mode = False

        # Unified observability fields
        self.backend_family = "sparse_kernel"
        self.diag_path = "bmm"
        self.fallback_reason = ""
        self.meta_source = "measured"
        self.diag_source = "measured"
        self.support_status = "supported"
        self.score_family = "bmm"

    def set_inference_mode(self, enabled: bool):
        self._inference_mode = bool(enabled)
        if enabled:
            self.collect_diag = False
            self.profile_runtime = False

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a: [B, M, K] — sparse left operand
            b: [B, K, N] — right operand

        Returns:
            C: [B, M, N]
        """
        assert a.ndim == 3 and b.ndim == 3
        self._last_diag = {}
        self.fallback_reason = ""

        if not self._triton_available or not a.is_cuda:
            self.fallback_reason = "no_triton_or_not_cuda"
            self.diag_source = "missing"
            return torch.bmm(a.float(), b.float())

        from Kernels.bmm import sparse_bmm_forward

        want_stats = self.collect_diag

        result = sparse_bmm_forward(
            a=a, b=b,
            threshold=self.threshold,
            ag_mask_buf=self._ag_mask_buf,
            tile_class_buf=self._tile_class_buf,
            return_ms=self.return_ms,
            return_avg_active_ratio=want_stats,
            return_tile_stats=want_stats,
            fallback_ratio=self.fallback_ratio,
        )

        y, ms = result[0], result[1]
        self._last_sparse_ms = ms

        idx = 2
        if want_stats and len(result) > idx:
            ratio = result[idx]
            self._last_diag["avg_active_ratio"] = ratio
            self._last_diag["active_group_ratio"] = float(ratio)
            self.diag_source = "measured"
            idx += 1
        if want_stats and len(result) > idx:
            ts = result[idx]
            self._last_diag["tile_stats"] = ts
            self.diag_source = "measured"
            if isinstance(ts, dict):
                total = ts.get("total_tiles", -1)
                zero = ts.get("zero_tiles", -1)
                if total > 0:
                    self._last_diag["total_tile_count"] = int(total)
                    if zero >= 0:
                        self._last_diag["tile_zero_count"] = int(zero)
                        self._last_diag["tile_zero_ratio"] = float(zero) / float(total)

        return y

    def extra_repr(self):
        return (
            f"threshold={self.threshold}, fallback_ratio={self.fallback_ratio}, "
            f"return_ms={self.return_ms}"
        )
