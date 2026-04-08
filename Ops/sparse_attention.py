"""
SparseFlow Ops/sparse_attention.py — SparseAttention nn.Module

Wraps Kernels/attention.py for Spikeformer-class models.
Provides sparse Q×K^T and attn×V as a single module with
configurable num_heads and head_dim.

This module does NOT implement softmax or the full MHA block —
it only accelerates the two matmul steps.  The surrounding
softmax / residual / projection logic stays in the model.

Usage:
    attn_module = SparseAttention(num_heads=8, head_dim=64)
    attn_logits = attn_module.qk(q, k)   # sparse Q × K^T
    attn_out    = attn_module.av(attn, v) # sparse attn × V
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import math

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn


class SparseAttention(nn.Module):
    """
    Sparse attention matmul module for Spikeformer.
    No learnable parameters — purely functional.
    """

    def __init__(
        self,
        num_heads: int = 8,
        head_dim: int = 64,
        threshold: float = 1e-6,
        return_ms: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.threshold = threshold
        self.return_ms = return_ms

        self._triton_available = False
        try:
            import triton  # noqa: F401
            self._triton_available = True
        except Exception:
            pass

        self._last_sparse_ms = 0.0
        self._last_diag: Dict[str, Any] = {}
        self.collect_diag = False
        self.profile_runtime = False
        self._inference_mode = False
        self.backend_family = "sparse_kernel"
        self.diag_path = "attention"
        self.fallback_reason = ""
        self.meta_source = "measured"
        self.diag_source = "measured"
        self.support_status = "supported"
        self.score_family = "attn_matmul"

    def set_inference_mode(self, enabled: bool):
        self._inference_mode = bool(enabled)
        if enabled:
            self.collect_diag = False
            self.profile_runtime = False

    def qk(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Sparse Q × K^T → attention logits.

        Args:
            q: [B, num_heads, seq_len, head_dim]
            k: [B, num_heads, seq_len, head_dim]

        Returns:
            attn_logits: [B, num_heads, seq_len, seq_len]
        """
        self.fallback_reason = ""
        if not self._triton_available or not q.is_cuda:
            self.fallback_reason = "no_triton_or_not_cuda"
            self.diag_source = "missing"
            return self._dense_qk(q, k)

        from Kernels.attention import sparse_qk_forward

        result = sparse_qk_forward(
            q=q, k=k,
            scale=self.scale,
            threshold=self.threshold,
            return_ms=self.return_ms,
            return_tile_stats=self.collect_diag,
        )
        attn_logits = result[0]
        self._last_sparse_ms = result[1]
        if self.collect_diag and len(result) > 2:
            self._last_diag['qk_stats'] = result[2]
            self.diag_source = "measured"
            self._last_diag.setdefault("backend_family", self.backend_family)
            self._last_diag.setdefault("diag_path", self.diag_path)
            self._last_diag.setdefault("fallback_reason", self.fallback_reason)
        return attn_logits

    def av(self, attn: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Sparse attn × V → attention output.

        Args:
            attn: [B, num_heads, seq_len, seq_len]
            v:    [B, num_heads, seq_len, head_dim]

        Returns:
            output: [B, num_heads, seq_len, head_dim]
        """
        if not self._triton_available or not attn.is_cuda:
            self.fallback_reason = "no_triton_or_not_cuda"
            self.diag_source = "missing"
            return self._dense_av(attn, v)

        from Kernels.attention import sparse_attn_v_forward

        result = sparse_attn_v_forward(
            attn=attn, v=v,
            threshold=self.threshold,
            return_ms=self.return_ms,
            return_tile_stats=self.collect_diag,
        )
        output = result[0]
        self._last_sparse_ms += result[1]
        if self.collect_diag and len(result) > 2:
            self._last_diag['av_stats'] = result[2]
            self.diag_source = "measured"
            self._last_diag.setdefault("backend_family", self.backend_family)
            self._last_diag.setdefault("diag_path", self.diag_path)
            self._last_diag.setdefault("fallback_reason", self.fallback_reason)
        return output

    def _dense_qk(self, q, k):
        return torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale

    def _dense_av(self, attn, v):
        return torch.matmul(attn.float(), v.float())

    def extra_repr(self):
        return (
            f"num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"scale={self.scale:.4f}, threshold={self.threshold}"
        )
