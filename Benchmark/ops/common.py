from __future__ import annotations

from typing import Dict

import torch

try:
    from benchmark.utils.metrics import correctness_metrics
except ModuleNotFoundError:
    from utils.metrics import correctness_metrics  # type: ignore


def dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.float32:
        return "float32"
    return str(dtype)


def make_case_base(
    operator: str,
    variant: str,
    scale: str,
    sparsity_regime: str,
    sparsity_level: float,
    sparsity_mode: str,
    seed: int,
    shape_meta: Dict,
    dtype: torch.dtype,
) -> Dict:
    return {
        "operator": operator,
        "variant": variant,
        "scale": scale,
        "sparsity_regime": sparsity_regime,
        "sparsity_level": float(sparsity_level),
        "sparsity_mode": sparsity_mode,
        "seed": int(seed),
        "dtype": dtype_name(dtype),
        "shape": shape_meta,
    }


def finalize_ok_case(
    case: Dict,
    dense_latency_ms: float,
    sparse_latency_ms: float,
    dense_out: torch.Tensor,
    sparse_out: torch.Tensor,
) -> Dict:
    metrics = correctness_metrics(dense_out, sparse_out)
    speedup = float(dense_latency_ms / sparse_latency_ms) if sparse_latency_ms > 0 else 0.0
    case.update(
        {
            "status": "ok",
            "dense_latency_ms": float(dense_latency_ms),
            "sparse_latency_ms": float(sparse_latency_ms),
            "speedup": speedup,
            "cosine_similarity": float(metrics["cosine_similarity"]),
            "max_abs_error": float(metrics["max_abs_error"]),
        }
    )
    return case


def finalize_failed_case(case: Dict, error: Exception) -> Dict:
    case.update(
        {
            "status": "failed",
            "dense_latency_ms": 0.0,
            "sparse_latency_ms": 0.0,
            "speedup": 0.0,
            "cosine_similarity": 0.0,
            "max_abs_error": 0.0,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    )
    return case
