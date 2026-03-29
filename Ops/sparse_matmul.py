"""
SparseFlow Ops/sparse_matmul.py — SparseMatmul nn.Module Wrapper

Functional sparse matmul module. Unlike SparseLinear (which wraps a
fixed weight matrix), SparseMatmul is a *functional* module that takes
two arbitrary tensors as input (A, B).  It manages metadata buffers
and provides the same diagnostic/profiling interface as other Ops.

Typical usage in SNN: matmul of spike-sparse activations with weight
matrices that aren't nn.Linear (e.g., custom projection layers).
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn


class SparseMatmul(nn.Module):
    """
    Sparse matmul operator C = A @ B where A is expected sparse.

    This is a functional module — no learnable parameters.
    Manages prescan metadata buffers and provides diagnostic hooks.
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

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute C = A @ B with sparse acceleration on A.

        Args:
            a: [M, K] or [*, M, K] — sparse left operand
            b: [K, N] or [*, K, N] — right operand

        Returns:
            C: [M, N] or [*, M, N]
        """
        # Handle batched input
        if a.ndim > 2:
            # Flatten leading dims, compute, reshape back
            lead = a.shape[:-2]
            M, K = a.shape[-2], a.shape[-1]
            N = b.shape[-1]
            a_flat = a.reshape(-1, K)
            if b.ndim > 2:
                b_flat = b.reshape(-1, K, N)
                # Use BMM path
                batch = a_flat.shape[0] // M
                a_3d = a_flat.reshape(batch, M, K)
                result = self._bmm_forward(a_3d, b_flat)
                return result.reshape(*lead, M, N)
            else:
                result = self._matmul_forward(a_flat, b)
                return result.reshape(*lead, M, N)

        return self._matmul_forward(a, b)

    def _matmul_forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not self._triton_available or not a.is_cuda:
            return torch.mm(a.float(), b.float())

        from Kernels.matmul import sparse_matmul_forward

        want_stats = self.collect_diag

        result = sparse_matmul_forward(
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
            ratio = result[idx]; idx += 1
            self._last_diag['avg_active_ratio'] = ratio
        if want_stats and len(result) > idx:
            self._last_diag['tile_stats'] = result[idx]

        return y

    def _bmm_forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not self._triton_available or not a.is_cuda:
            return torch.bmm(a.float(), b.float())

        from Kernels.bmm import sparse_bmm_forward

        result = sparse_bmm_forward(
            a=a, b=b,
            threshold=self.threshold,
            return_ms=self.return_ms,
            return_tile_stats=self.collect_diag,
            fallback_ratio=self.fallback_ratio,
        )
        y, ms = result[0], result[1]
        self._last_sparse_ms = ms
        return y

    def extra_repr(self):
        return (
            f"threshold={self.threshold}, fallback_ratio={self.fallback_ratio}, "
            f"return_ms={self.return_ms}, profile_runtime={self.profile_runtime}"
        )