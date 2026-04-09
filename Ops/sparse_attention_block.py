"""
Ops/sparse_attention_block.py

Sparse wrapper for transformer-style attention blocks.

Supported variants:
- attention_qkav
- attention_linear
- attention_qkmix
"""

import sys
from pathlib import Path
from typing import Any, Dict

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn

from Ops.sparse_attention import SparseAttention
from Ops.sparse_matmul import SparseMatmul


def _safe_bn(module: nn.Module, name: str) -> nn.Module:
    return getattr(module, name) if hasattr(module, name) else nn.Identity()


def _safe_attr(module: nn.Module, name: str, default):
    return getattr(module, name, default)


class SparseAttentionBlock(nn.Module):
    """
    Generic sparse attention block wrapper.

    Observability contract:
    - collect_diag
    - profile_runtime
    - _last_diag
    - _last_sparse_ms
    - backend_family
    - diag_path
    - fallback_reason
    """

    def __init__(
        self,
        dense_attn: nn.Module,
        variant: str,
        threshold: float = 1e-6,
        return_ms: bool = False,
        profile_runtime: bool = False,
    ):
        super().__init__()
        v = str(variant)
        if v == "attention_proj_linear":
            v = "attention_linear"
        if v == "attention_matmul":
            v = "attention_qkav"
        self.variant = v
        self.threshold = float(threshold)
        self.return_ms = bool(return_ms)

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
            _safe_attr(dense_attn, "head_dim", max(1, self.dim // max(1, self.num_heads)))
        )

        self._sparse_attn = SparseAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            threshold=self.threshold,
            return_ms=self.return_ms,
        )

        # Build SparseMatmul modules only for variants that actually use them.
        self._mm_1 = None
        self._mm_2 = None
        self._mm_3 = None
        self._mm_4 = None
        if self.variant in ("attention_linear", "attention_qkmix"):
            self._mm_1 = SparseMatmul(
                threshold=self.threshold,
                fallback_ratio=0.85,
                return_ms=self.return_ms,
                profile_runtime=profile_runtime,
            )
            self._mm_2 = SparseMatmul(
                threshold=self.threshold,
                fallback_ratio=0.85,
                return_ms=self.return_ms,
                profile_runtime=profile_runtime,
            )
        if self.variant == "attention_qkmix":
            self._mm_3 = SparseMatmul(
                threshold=self.threshold,
                fallback_ratio=0.85,
                return_ms=self.return_ms,
                profile_runtime=profile_runtime,
            )
            self._mm_4 = SparseMatmul(
                threshold=self.threshold,
                fallback_ratio=0.85,
                return_ms=self.return_ms,
                profile_runtime=profile_runtime,
            )

        # Unified observability fields.
        self.collect_diag = False
        self.profile_runtime = bool(profile_runtime)
        self._inference_mode = False
        self._last_sparse_ms = 0.0
        self._last_diag: Dict[str, Any] = {}
        self.backend_family = "sparse_kernel"
        self.diag_path = "attention_block"
        self.fallback_reason = ""
        self.meta_source = "measured"
        self.diag_source = "measured"
        self.support_status = "supported"
        self.score_family = (
            "attn_linear" if self.variant == "attention_linear" else "attn_matmul"
        )

    @classmethod
    def from_dense(
        cls,
        dense_attn: nn.Module,
        variant: str,
        threshold: float = 1e-6,
        return_ms: bool = False,
        profile_runtime: bool = False,
    ) -> "SparseAttentionBlock":
        return cls(
            dense_attn=dense_attn,
            variant=variant,
            threshold=threshold,
            return_ms=return_ms,
            profile_runtime=profile_runtime,
        )

    @staticmethod
    def _normalize_diag(diag: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize diagnostic keys to the shared schema used by dispatch/logger.

        Accepts either:
        - direct tile stats dict
        - wrapper dict with nested "tile_stats"
        """
        raw = dict(diag or {})
        if "tile_stats" in raw and isinstance(raw["tile_stats"], dict):
            raw = dict(raw["tile_stats"])

        out = dict(raw)
        avg = out.get("avg_active_ratio", -1.0)
        if avg >= 0 and out.get("active_group_ratio", -1.0) < 0:
            out["active_group_ratio"] = float(avg)

        total = out.get("total_tiles", out.get("total_tile_count", -1))
        zero = out.get("zero_tiles", out.get("tile_zero_count", -1))
        sparse_tiles = out.get("sparse_tiles", -1)
        denseish_tiles = out.get("denseish_tiles", -1)

        if total > 0:
            out["total_tile_count"] = int(total)
            out["total_tiles"] = int(total)
            if zero >= 0:
                out["tile_zero_count"] = int(zero)
                if out.get("tile_zero_ratio", -1.0) < 0:
                    out["tile_zero_ratio"] = float(zero) / float(total)
            if sparse_tiles >= 0:
                out["sparse_tiles"] = int(sparse_tiles)
            if denseish_tiles >= 0:
                out["denseish_tiles"] = int(denseish_tiles)

        return out

    def set_inference_mode(self, enabled: bool):
        self._inference_mode = bool(enabled)
        if enabled:
            self.collect_diag = False
            self.profile_runtime = False
        self._sync_inner_runtime_flags()

    def _sync_inner_runtime_flags(self):
        self._sparse_attn.collect_diag = self.collect_diag
        if hasattr(self._sparse_attn, "return_ms"):
            self._sparse_attn.return_ms = self.return_ms
        if hasattr(self._sparse_attn, "profile_runtime"):
            self._sparse_attn.profile_runtime = self.profile_runtime
        if hasattr(self._sparse_attn, "set_inference_mode"):
            self._sparse_attn.set_inference_mode(self._inference_mode)
        for mm in (self._mm_1, self._mm_2, self._mm_3, self._mm_4):
            if mm is not None:
                mm.collect_diag = self.collect_diag
                if hasattr(mm, "return_ms"):
                    mm.return_ms = self.return_ms
                if hasattr(mm, "profile_runtime"):
                    mm.profile_runtime = self.profile_runtime
                if hasattr(mm, "set_inference_mode"):
                    mm.set_inference_mode(self._inference_mode)

    def _linear_bn_lif(self, x: torch.Tensor, linear: nn.Linear, bn: nn.Module, lif: nn.Module) -> torch.Tensor:
        t, b, n, c = x.shape
        y = linear(x.flatten(0, 1))
        y = bn(y.transpose(-1, -2)).transpose(-1, -2).reshape(t, b, n, c).contiguous()
        y = lif(y)
        return y

    def _final_project(self, y: torch.Tensor) -> torch.Tensor:
        t, b, n, c = y.shape
        y = self.proj(y.flatten(0, 1))
        y = self.proj_bn(y.transpose(-1, -2)).transpose(-1, -2).reshape(t, b, n, c).contiguous()
        y = self.proj_lif(y)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, b, n, c = x.shape
        n_float = float(max(1, n))
        self._sync_inner_runtime_flags()
        self._last_sparse_ms = 0.0
        self._last_diag = {}
        self.diag_path = self.variant
        self.fallback_reason = ""

        q = self._linear_bn_lif(x, self.q, self.q_bn, self.q_lif)
        k = self._linear_bn_lif(x, self.k, self.k_bn, self.k_lif)
        v = self._linear_bn_lif(x, self.v, self.v_bn, self.v_lif)

        q = q.reshape(t, b, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4).contiguous()
        k = k.reshape(t, b, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4).contiguous()
        v = v.reshape(t, b, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4).contiguous()

        if self.variant == "attention_qkav":
            # Kernels.attention expects [B, H, N, D]; fold time into batch.
            q_tb = q.reshape(t * b, self.num_heads, n, self.head_dim)
            k_tb = k.reshape(t * b, self.num_heads, n, self.head_dim)
            v_tb = v.reshape(t * b, self.num_heads, n, self.head_dim)
            attn = self._sparse_attn.qk(q_tb, k_tb)
            y = self._sparse_attn.av(attn, v_tb).reshape(
                t, b, self.num_heads, n, self.head_dim
            )
            self._last_sparse_ms = float(getattr(self._sparse_attn, "_last_sparse_ms", 0.0))
            diag_all = dict(getattr(self._sparse_attn, "_last_diag", {}) or {})
            qk_stats = diag_all.get("qk_stats", {})
            self._last_diag = self._normalize_diag(qk_stats)
            self.diag_source = "measured" if self._last_diag else "missing"

        elif self.variant == "attention_linear":
            if self._mm_1 is None or self._mm_2 is None:
                raise RuntimeError("attention_linear requires _mm_1/_mm_2")
            kv = self._mm_1(k.transpose(-2, -1).contiguous(), v) / n_float
            y = self._mm_2(q, kv)
            self._last_sparse_ms = float(getattr(self._mm_1, "_last_sparse_ms", 0.0)) + float(
                getattr(self._mm_2, "_last_sparse_ms", 0.0)
            )
            self._last_diag = self._normalize_diag(getattr(self._mm_1, "_last_diag", {}))
            self.diag_source = "measured" if self._last_diag else "missing"

        elif self.variant == "attention_qkmix":
            if None in (self._mm_1, self._mm_2, self._mm_3, self._mm_4):
                raise RuntimeError("attention_qkmix requires _mm_1/_mm_2/_mm_3/_mm_4")
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
            self.diag_source = "measured" if self._last_diag else "missing"

        else:
            raise ValueError(f"Unsupported attention variant: {self.variant}")

        y = y.transpose(2, 3).reshape(t, b, n, c).contiguous()
        y = self.attn_lif(y)
        y = self._final_project(y)
        if self.collect_diag:
            self._last_diag.setdefault("backend_family", self.backend_family)
            self._last_diag.setdefault("diag_path", self.diag_path)
            self._last_diag.setdefault("fallback_reason", self.fallback_reason)
        return y

    def extra_repr(self) -> str:
        return (
            f"variant={self.variant}, dim={self.dim}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, threshold={self.threshold}, "
            f"return_ms={self.return_ms}, profile_runtime={self.profile_runtime}, "
            f"backend_family={self.backend_family}"
        )
