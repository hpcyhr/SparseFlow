"""
Ops/sparse_attention_block.py

Sparse wrapper for attention blocks used by external transformer models:
- attention_qkav   (Spikformer SSA style)
- attention_linear (SDT-v1 style linear attention)
- attention_qkmix  (QKFormer mixed token/channel linear attention)

Design:
- Keep original q/k/v/proj + BN + spike modules untouched (weights/state preserved).
- Replace internal matmul/attention event ops with sparse kernels:
  - Ops.sparse_attention.SparseAttention (qk + av)
  - Ops.sparse_matmul.SparseMatmul (general batched matmul)
"""

import sys
from pathlib import Path
import math
from typing import Dict, Any

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn

from Ops.sparse_attention import SparseAttention
from Ops.sparse_matmul import SparseMatmul


def _safe_bn(module: nn.Module, name: str) -> nn.Module:
    if hasattr(module, name):
        return getattr(module, name)
    return nn.Identity()


def _safe_attr(module: nn.Module, name: str, default):
    return getattr(module, name, default)


class SparseAttentionBlock(nn.Module):
    """
    Generic sparse attention block wrapper.

    Expected dense module structure:
      q, k, v, proj            : nn.Linear
      q_bn, k_bn, v_bn, proj_bn: nn.BatchNorm1d (or compatible)
      q_lif, k_lif, v_lif, attn_lif, proj_lif : spike modules
      dim, num_heads, head_dim : metadata (best effort)
    """

    def __init__(self, dense_attn: nn.Module, variant: str, threshold: float = 1e-6):
        super().__init__()
        self.variant = variant
        self.threshold = float(threshold)

        # Reuse original submodules directly to preserve weights/states.
        self.q = dense_attn.q
        self.k = dense_attn.k
        self.v = dense_attn.v
        self.proj = dense_attn.proj

        self.q_bn = _safe_bn(dense_attn, "q_bn")
        self.k_bn = _safe_bn(dense_attn, "k_bn")
        self.v_bn = _safe_bn(dense_attn, "v_bn")
        self.proj_bn = _safe_bn(dense_attn, "proj_bn")

        self.q_lif = _safe_attr(dense_attn, "q_lif", nn.Identity())
        self.k_lif = _safe_attr(dense_attn, "k_lif", nn.Identity())
        self.v_lif = _safe_attr(dense_attn, "v_lif", nn.Identity())
        self.attn_lif = _safe_attr(dense_attn, "attn_lif", nn.Identity())
        self.proj_lif = _safe_attr(dense_attn, "proj_lif", nn.Identity())

        self.dim = int(_safe_attr(dense_attn, "dim", self.q.out_features))
        self.num_heads = int(_safe_attr(dense_attn, "num_heads", 1))
        self.head_dim = int(
            _safe_attr(
                dense_attn,
                "head_dim",
                max(1, self.dim // max(1, self.num_heads)),
            )
        )
        self.scale = float(_safe_attr(dense_attn, "scale", 1.0 / math.sqrt(max(1, self.head_dim))))

        # Sparse event operators
        self._sparse_attn = SparseAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            threshold=self.threshold,
            return_ms=False,
        )

        # For linear / qkmix variants
        self._mm_1 = SparseMatmul(threshold=self.threshold, fallback_ratio=0.85, return_ms=False, profile_runtime=False)
        self._mm_2 = SparseMatmul(threshold=self.threshold, fallback_ratio=0.85, return_ms=False, profile_runtime=False)
        self._mm_3 = SparseMatmul(threshold=self.threshold, fallback_ratio=0.85, return_ms=False, profile_runtime=False)
        self._mm_4 = SparseMatmul(threshold=self.threshold, fallback_ratio=0.85, return_ms=False, profile_runtime=False)

        # Bench diagnostic compatibility
        self.collect_diag = False
        self._last_sparse_ms = 0.0
        self._last_diag: Dict[str, Any] = {}

    @classmethod
    def from_dense(cls, dense_attn: nn.Module, variant: str, threshold: float = 1e-6) -> "SparseAttentionBlock":
        return cls(dense_attn=dense_attn, variant=variant, threshold=threshold)

    @staticmethod
    def _normalize_diag(diag: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(diag or {})
        avg = out.get("avg_active_ratio", -1.0)
        if avg >= 0 and out.get("active_group_ratio", -1.0) < 0:
            out["active_group_ratio"] = float(avg)

        tile_stats = out.get("tile_stats", None)
        if isinstance(tile_stats, dict):
            total = tile_stats.get("total_tiles", -1)
            zero = tile_stats.get("zero_tiles", -1)
            sparse_tiles = tile_stats.get("sparse_tiles", -1)
            denseish_tiles = tile_stats.get("denseish_tiles", -1)
            if total > 0:
                if zero >= 0 and out.get("tile_zero_ratio", -1.0) < 0:
                    out["tile_zero_ratio"] = float(zero) / float(total)
                    out["tile_zero_count"] = int(zero)
                    out["total_tile_count"] = int(total)
                if sparse_tiles >= 0:
                    out["sparse_tiles"] = int(sparse_tiles)
                if denseish_tiles >= 0:
                    out["denseish_tiles"] = int(denseish_tiles)
                out["total_tiles"] = int(total)
        return out

    def _linear_bn_lif(
        self,
        x: torch.Tensor,
        linear: nn.Linear,
        bn: nn.Module,
        lif: nn.Module,
    ) -> torch.Tensor:
        # x: [T, B, N, C]
        t, b, n, c = x.shape
        y = linear(x.flatten(0, 1))  # [TB, N, C]
        y = bn(y.transpose(-1, -2)).transpose(-1, -2).reshape(t, b, n, c).contiguous()
        y = lif(y)
        return y

    def _set_diag_switch(self):
        self._sparse_attn.collect_diag = self.collect_diag
        self._mm_1.collect_diag = self.collect_diag
        self._mm_2.collect_diag = self.collect_diag
        self._mm_3.collect_diag = self.collect_diag
        self._mm_4.collect_diag = self.collect_diag

    def _final_project(self, y: torch.Tensor) -> torch.Tensor:
        # y: [T, B, N, C]
        t, b, n, c = y.shape
        y = self.proj(y.flatten(0, 1))
        y = self.proj_bn(y.transpose(-1, -2)).transpose(-1, -2).reshape(t, b, n, c).contiguous()
        y = self.proj_lif(y)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, N, C]
        t, b, n, c = x.shape
        n_float = float(max(1, n))
        self._set_diag_switch()
        self._last_sparse_ms = 0.0
        self._last_diag = {}

        q = self._linear_bn_lif(x, self.q, self.q_bn, self.q_lif)
        k = self._linear_bn_lif(x, self.k, self.k_bn, self.k_lif)
        v = self._linear_bn_lif(x, self.v, self.v_bn, self.v_lif)

        q = q.reshape(t, b, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4).contiguous()
        k = k.reshape(t, b, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4).contiguous()
        v = v.reshape(t, b, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4).contiguous()

        if self.variant == "attention_qkav":
            # Classic QK^T + AV path
            attn = self._sparse_attn.qk(q, k)   # includes scale internally
            y = self._sparse_attn.av(attn, v)
            self._last_sparse_ms = float(getattr(self._sparse_attn, "_last_sparse_ms", 0.0))
            diag = dict(getattr(self._sparse_attn, "_last_diag", {}) or {})
            qk_stats = diag.get("qk_stats", None)
            if isinstance(qk_stats, dict):
                diag = self._normalize_diag(qk_stats)
            self._last_diag = diag

        elif self.variant == "attention_linear":
            # SDT-v1 linear attention: Q (K^T V)
            kv = self._mm_1(k.transpose(-2, -1).contiguous(), v) / n_float
            y = self._mm_2(q, kv)
            self._last_sparse_ms = float(getattr(self._mm_1, "_last_sparse_ms", 0.0)) + float(
                getattr(self._mm_2, "_last_sparse_ms", 0.0)
            )
            self._last_diag = self._normalize_diag(getattr(self._mm_1, "_last_diag", {}))

        elif self.variant == "attention_qkmix":
            # QKFormer mixed branch:
            # token:   Q (K^T V)
            # channel: V (Q^T K)
            kv = self._mm_1(k.transpose(-2, -1).contiguous(), v) / n_float
            token_out = self._mm_2(q, kv)

            qk = self._mm_3(q.transpose(-2, -1).contiguous(), k) / n_float
            channel_out = self._mm_4(v, qk)

            y = token_out + channel_out
            self._last_sparse_ms = (
                float(getattr(self._mm_1, "_last_sparse_ms", 0.0))
                + float(getattr(self._mm_2, "_last_sparse_ms", 0.0))
                + float(getattr(self._mm_3, "_last_sparse_ms", 0.0))
                + float(getattr(self._mm_4, "_last_sparse_ms", 0.0))
            )
            self._last_diag = self._normalize_diag(getattr(self._mm_1, "_last_diag", {}))

        else:
            raise ValueError(f"Unsupported attention variant: {self.variant}")

        y = y.transpose(2, 3).reshape(t, b, n, c).contiguous()
        y = self.attn_lif(y)
        y = self._final_project(y)
        return y

    def extra_repr(self) -> str:
        return (
            f"variant={self.variant}, dim={self.dim}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, threshold={self.threshold}"
        )
