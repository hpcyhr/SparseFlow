from __future__ import annotations

from typing import Dict, List, Sequence

import torch

try:
    from benchmark.ops.common import finalize_failed_case, finalize_ok_case, make_case_base
    from benchmark.utils.sparsity import make_sparse_input
    from benchmark.utils.timer import BenchmarkTimer
except ModuleNotFoundError:
    from ops.common import finalize_failed_case, finalize_ok_case, make_case_base  # type: ignore
    from utils.sparsity import make_sparse_input  # type: ignore
    from utils.timer import BenchmarkTimer  # type: ignore


MATMUL_SCALES = {
    "small": {"m": 512, "k": 512, "n": 512},
    "medium": {"m": 1024, "k": 1024, "n": 1024},
    "large": {"m": 2048, "k": 2048, "n": 2048},
}

BMM_SCALES = {
    "small": {"batch": 8, "m": 128, "k": 128, "n": 128},
    "medium": {"batch": 16, "m": 256, "k": 256, "n": 256},
    "large": {"batch": 32, "m": 512, "k": 512, "n": 512},
}


def run_matmul_suite(
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
    if scale not in MATMUL_SCALES:
        raise ValueError(f"Unknown Matmul scale: {scale}")
    cfg = MATMUL_SCALES[scale]
    m = int(cfg["m"])
    k = int(cfg["k"])
    n = int(cfg["n"])

    case = make_case_base(
        operator="matmul",
        variant=f"{m}x{k}x{n}",
        scale=scale,
        sparsity_regime=sparsity_regime,
        sparsity_level=sparsity_level,
        sparsity_mode=sparsity_mode,
        seed=seed,
        shape_meta={"a": [m, k], "b": [k, n]},
        dtype=dtype,
    )

    try:
        result = _run_one_matmul_case(
            case=case,
            timer=timer,
            device=device,
            dtype=dtype,
            m=m,
            k=k,
            n=n,
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


def run_bmm_suite(
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
    if scale not in BMM_SCALES:
        raise ValueError(f"Unknown BMM scale: {scale}")
    cfg = BMM_SCALES[scale]
    batch = int(cfg["batch"])
    m = int(cfg["m"])
    k = int(cfg["k"])
    n = int(cfg["n"])

    case = make_case_base(
        operator="bmm",
        variant=f"{batch}x{m}x{k}x{n}",
        scale=scale,
        sparsity_regime=sparsity_regime,
        sparsity_level=sparsity_level,
        sparsity_mode=sparsity_mode,
        seed=seed,
        shape_meta={"a": [batch, m, k], "b": [batch, k, n]},
        dtype=dtype,
    )

    try:
        result = _run_one_bmm_case(
            case=case,
            timer=timer,
            device=device,
            dtype=dtype,
            batch=batch,
            m=m,
            k=k,
            n=n,
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


def _run_one_matmul_case(
    case: Dict,
    timer: BenchmarkTimer,
    device: torch.device,
    dtype: torch.dtype,
    m: int,
    k: int,
    n: int,
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

    a = make_sparse_input(
        shape=(m, k),
        device=device,
        dtype=dtype,
        sparsity_level=sparsity_level,
        sparsity_mode=sparsity_mode,
        structured_tile=structured_tile,
        seed=seed + 1,
    )
    b = torch.randn((k, n), device=device, dtype=dtype)

    sparse_matmul = _build_sparse_matmul_module().to(device=device).eval()

    with torch.no_grad():
        dense_ref = torch.matmul(a, b)
        sparse_ref = sparse_matmul(a, b)

    def dense_fn() -> torch.Tensor:
        with torch.no_grad():
            return torch.matmul(a, b)

    def sparse_fn() -> torch.Tensor:
        with torch.no_grad():
            return sparse_matmul(a, b)

    dense_t = timer.run(dense_fn, warmup=warmup, iters=iters)
    sparse_t = timer.run(sparse_fn, warmup=warmup, iters=iters)
    return finalize_ok_case(
        case=case,
        dense_latency_ms=float(dense_t["avg_ms"]),
        sparse_latency_ms=float(sparse_t["avg_ms"]),
        dense_out=dense_ref,
        sparse_out=sparse_ref,
    )


def _run_one_bmm_case(
    case: Dict,
    timer: BenchmarkTimer,
    device: torch.device,
    dtype: torch.dtype,
    batch: int,
    m: int,
    k: int,
    n: int,
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

    a = make_sparse_input(
        shape=(batch, m, k),
        device=device,
        dtype=dtype,
        sparsity_level=sparsity_level,
        sparsity_mode=sparsity_mode,
        structured_tile=structured_tile,
        seed=seed + 1,
    )
    b = torch.randn((batch, k, n), device=device, dtype=dtype)

    sparse_bmm = _build_sparse_bmm_module().to(device=device).eval()

    with torch.no_grad():
        dense_ref = torch.bmm(a, b)
        sparse_ref = sparse_bmm(a, b)

    def dense_fn() -> torch.Tensor:
        with torch.no_grad():
            return torch.bmm(a, b)

    def sparse_fn() -> torch.Tensor:
        with torch.no_grad():
            return sparse_bmm(a, b)

    dense_t = timer.run(dense_fn, warmup=warmup, iters=iters)
    sparse_t = timer.run(sparse_fn, warmup=warmup, iters=iters)
    return finalize_ok_case(
        case=case,
        dense_latency_ms=float(dense_t["avg_ms"]),
        sparse_latency_ms=float(sparse_t["avg_ms"]),
        dense_out=dense_ref,
        sparse_out=sparse_ref,
    )


def _build_sparse_matmul_module() -> torch.nn.Module:
    try:
        from Ops.sparse_matmul import SparseMatmul
    except Exception as exc:
        raise RuntimeError(
            "TODO hook: SparseFlow SparseMatmul is not available in Ops/sparse_matmul.py"
        ) from exc
    return SparseMatmul(threshold=1e-6, fallback_ratio=0.85, return_ms=False, profile_runtime=False)


def _build_sparse_bmm_module() -> torch.nn.Module:
    try:
        from Ops.sparse_bmm import SparseBMM
    except Exception as exc:
        raise RuntimeError(
            "TODO hook: SparseFlow SparseBMM is not available in Ops/sparse_bmm.py"
        ) from exc
    return SparseBMM(threshold=1e-6, fallback_ratio=0.85, return_ms=False, profile_runtime=False)


def dense_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)


def dense_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.bmm(a, b)


def sparse_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    threshold: float = 1e-6,
    fallback_ratio: float = 0.85,
) -> torch.Tensor:
    """SparseFlow compatibility hook: sparse_matmul(a, b)."""
    try:
        from Ops.sparse_matmul import SparseMatmul
    except Exception as exc:
        raise RuntimeError(
            "TODO hook: SparseFlow SparseMatmul is not available in Ops/sparse_matmul.py"
        ) from exc
    op = SparseMatmul(
        threshold=threshold,
        fallback_ratio=fallback_ratio,
        return_ms=False,
        profile_runtime=False,
    ).to(a.device)
    return op(a, b)


def sparse_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    threshold: float = 1e-6,
    fallback_ratio: float = 0.85,
) -> torch.Tensor:
    """SparseFlow compatibility hook: sparse_bmm(a, b)."""
    try:
        from Ops.sparse_bmm import SparseBMM
    except Exception as exc:
        raise RuntimeError(
            "TODO hook: SparseFlow SparseBMM is not available in Ops/sparse_bmm.py"
        ) from exc
    op = SparseBMM(
        threshold=threshold,
        fallback_ratio=fallback_ratio,
        return_ms=False,
        profile_runtime=False,
    ).to(a.device)
    return op(a, b)
