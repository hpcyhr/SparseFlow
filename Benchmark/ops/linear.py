from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from benchmark.ops.common import finalize_failed_case, finalize_ok_case, make_case_base
    from benchmark.utils.sparsity import make_sparse_input
    from benchmark.utils.timer import BenchmarkTimer
except ModuleNotFoundError:
    from ops.common import finalize_failed_case, finalize_ok_case, make_case_base  # type: ignore
    from utils.sparsity import make_sparse_input  # type: ignore
    from utils.timer import BenchmarkTimer  # type: ignore


LINEAR_SCALES = {
    "small": {"batch": 512, "in_features": 512, "out_features": 512},
    "medium": {"batch": 1024, "in_features": 1024, "out_features": 1024},
    "large": {"batch": 2048, "in_features": 2048, "out_features": 2048},
}


def run_linear_suite(
    timer: BenchmarkTimer,
    device: torch.device,
    dtype: torch.dtype,
    scale: str,
    sparsity_regime: str,
    sparsity_level: float,
    sparsity_mode: str,
    warmup: int,
    iters: int,
    seed: int,
    structured_tile: Sequence[int],
) -> List[Dict]:
    if scale not in LINEAR_SCALES:
        raise ValueError(f"Unknown Linear scale: {scale}")
    cfg = LINEAR_SCALES[scale]
    batch = int(cfg["batch"])
    in_features = int(cfg["in_features"])
    out_features = int(cfg["out_features"])

    case = make_case_base(
        operator="linear",
        variant=f"{in_features}x{out_features}",
        scale=scale,
        sparsity_regime=sparsity_regime,
        sparsity_level=sparsity_level,
        sparsity_mode=sparsity_mode,
        seed=seed,
        shape_meta={
            "input": [batch, in_features],
            "weight": [out_features, in_features],
        },
        dtype=dtype,
    )

    try:
        result = _run_one_linear_case(
            case=case,
            timer=timer,
            device=device,
            dtype=dtype,
            batch=batch,
            in_features=in_features,
            out_features=out_features,
            warmup=warmup,
            iters=iters,
            sparsity_level=sparsity_level,
            sparsity_mode=sparsity_mode,
            seed=seed,
            structured_tile=structured_tile,
        )
    except Exception as exc:
        result = finalize_failed_case(case, exc)
    return [result]


def _run_one_linear_case(
    case: Dict,
    timer: BenchmarkTimer,
    device: torch.device,
    dtype: torch.dtype,
    batch: int,
    in_features: int,
    out_features: int,
    warmup: int,
    iters: int,
    sparsity_level: float,
    sparsity_mode: str,
    seed: int,
    structured_tile: Sequence[int],
) -> Dict:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    x = make_sparse_input(
        shape=(batch, in_features),
        device=device,
        dtype=dtype,
        sparsity_level=sparsity_level,
        sparsity_mode=sparsity_mode,
        structured_tile=structured_tile,
        seed=seed + 1,
    )

    dense_linear = nn.Linear(in_features, out_features, bias=True).to(device=device, dtype=dtype).eval()
    sparse_linear = _build_sparse_linear_module(dense_linear).to(device=device).eval()

    with torch.no_grad():
        dense_ref = dense_linear(x)
        sparse_ref = sparse_linear(x)

    def dense_fn() -> torch.Tensor:
        with torch.no_grad():
            return dense_linear(x)

    def sparse_fn() -> torch.Tensor:
        with torch.no_grad():
            return sparse_linear(x)

    dense_t = timer.run(dense_fn, warmup=warmup, iters=iters)
    sparse_t = timer.run(sparse_fn, warmup=warmup, iters=iters)
    return finalize_ok_case(
        case=case,
        dense_latency_ms=float(dense_t["avg_ms"]),
        sparse_latency_ms=float(sparse_t["avg_ms"]),
        dense_out=dense_ref,
        sparse_out=sparse_ref,
    )


def _build_sparse_linear_module(dense_linear: nn.Linear) -> nn.Module:
    try:
        from Ops.sparse_linear import SparseLinear
    except Exception as exc:
        raise RuntimeError(
            "TODO hook: SparseFlow SparseLinear is not available in Ops/sparse_linear.py"
        ) from exc

    sparse_linear = SparseLinear.from_dense(dense_linear, return_ms=False)
    if hasattr(sparse_linear, "set_inference_mode"):
        sparse_linear.set_inference_mode(True)
    if hasattr(sparse_linear, "collect_diag"):
        sparse_linear.collect_diag = False
    return sparse_linear


def dense_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return F.linear(x, weight=weight, bias=bias)


def sparse_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    threshold: float = 1e-6,
) -> torch.Tensor:
    """SparseFlow compatibility hook: sparse_linear(x, weight)."""
    try:
        from Ops.sparse_linear import SparseLinear
    except Exception as exc:
        raise RuntimeError(
            "TODO hook: SparseFlow SparseLinear is not available in Ops/sparse_linear.py"
        ) from exc

    out_features, in_features = weight.shape
    dense = nn.Linear(in_features, out_features, bias=bias is not None).to(
        device=x.device, dtype=weight.dtype
    )
    with torch.no_grad():
        dense.weight.copy_(weight)
        if bias is not None and dense.bias is not None:
            dense.bias.copy_(bias)
    sparse = SparseLinear.from_dense(dense, threshold=threshold, return_ms=False).to(x.device).eval()
    if hasattr(sparse, "set_inference_mode"):
        sparse.set_inference_mode(True)
    return sparse(x)
