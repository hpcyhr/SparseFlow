from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch

try:
    from benchmark.ops.common import finalize_failed_case, finalize_ok_case, make_case_base
    from benchmark.utils.sparsity import make_sparse_input
    from benchmark.utils.timer import BenchmarkTimer
except ModuleNotFoundError:
    from ops.common import finalize_failed_case, finalize_ok_case, make_case_base  # type: ignore
    from utils.sparsity import make_sparse_input  # type: ignore
    from utils.timer import BenchmarkTimer  # type: ignore


ATTENTION_SCALES = {
    "small": {"batch": 8, "tokens": 64, "dim": 256, "num_heads": 8},
    "medium": {"batch": 16, "tokens": 128, "dim": 512, "num_heads": 8},
    "large": {"batch": 32, "tokens": 256, "dim": 512, "num_heads": 8},
}


def run_attention_suite(
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
    if scale not in ATTENTION_SCALES:
        raise ValueError(f"Unknown Attention scale: {scale}")
    cfg = ATTENTION_SCALES[scale]
    batch = int(cfg["batch"])
    tokens = int(cfg["tokens"])
    dim = int(cfg["dim"])
    num_heads = int(cfg["num_heads"])
    if dim % num_heads != 0:
        raise ValueError(f"Attention dim {dim} must be divisible by num_heads {num_heads}")
    head_dim = dim // num_heads

    case = make_case_base(
        operator="attention",
        variant=f"B{batch}_T{tokens}_C{dim}_H{num_heads}",
        scale=scale,
        sparsity_regime=sparsity_regime,
        sparsity_level=sparsity_level,
        sparsity_mode=sparsity_mode,
        seed=seed,
        shape_meta={
            "input": [batch, tokens, dim],
            "num_heads": num_heads,
            "head_dim": head_dim,
        },
        dtype=dtype,
    )

    try:
        result = _run_one_attention_case(
            case=case,
            timer=timer,
            device=device,
            dtype=dtype,
            batch=batch,
            tokens=tokens,
            dim=dim,
            num_heads=num_heads,
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


def _run_one_attention_case(
    case: Dict,
    timer: BenchmarkTimer,
    device: torch.device,
    dtype: torch.dtype,
    batch: int,
    tokens: int,
    dim: int,
    num_heads: int,
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
        shape=(batch, tokens, dim),
        device=device,
        dtype=dtype,
        sparsity_level=sparsity_level,
        sparsity_mode=sparsity_mode,
        structured_tile=structured_tile,
        seed=seed + 1,
    )
    wq = torch.randn((dim, dim), device=device, dtype=dtype)
    wk = torch.randn((dim, dim), device=device, dtype=dtype)
    wv = torch.randn((dim, dim), device=device, dtype=dtype)

    sparse_attention = _build_sparse_attention_module(num_heads=num_heads, head_dim=dim // num_heads)
    sparse_attention = sparse_attention.to(device=device).eval()
    if hasattr(sparse_attention, "collect_diag"):
        sparse_attention.collect_diag = False

    with torch.no_grad():
        dense_ref = dense_attention(x, wq, wk, wv, num_heads=num_heads)
        sparse_ref = sparse_attention_forward(sparse_attention, x, wq, wk, wv, num_heads=num_heads)

    def dense_fn() -> torch.Tensor:
        with torch.no_grad():
            return dense_attention(x, wq, wk, wv, num_heads=num_heads)

    def sparse_fn() -> torch.Tensor:
        with torch.no_grad():
            return sparse_attention_forward(sparse_attention, x, wq, wk, wv, num_heads=num_heads)

    dense_t = timer.run(dense_fn, warmup=warmup, iters=iters)
    sparse_t = timer.run(sparse_fn, warmup=warmup, iters=iters)
    return finalize_ok_case(
        case=case,
        dense_latency_ms=float(dense_t["avg_ms"]),
        sparse_latency_ms=float(sparse_t["avg_ms"]),
        dense_out=dense_ref,
        sparse_out=sparse_ref,
    )


def dense_attention(
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    q, k, v, head_dim = _qkv_project(x, wq, wk, wv, num_heads=num_heads)
    scale = 1.0 / math.sqrt(float(head_dim))
    logits = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)
    out = torch.matmul(attn, v)
    return _merge_heads(out)


def sparse_attention_forward(
    sparse_attention_module: torch.nn.Module,
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    q, k, v, _ = _qkv_project(x, wq, wk, wv, num_heads=num_heads)
    logits = sparse_attention_module.qk(q, k)
    attn = torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)
    out = sparse_attention_module.av(attn, v)
    return _merge_heads(out)


def _qkv_project(
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    num_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    b, t, c = x.shape
    head_dim = c // num_heads
    q = torch.matmul(x, wq)
    k = torch.matmul(x, wk)
    v = torch.matmul(x, wv)
    q = q.view(b, t, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    k = k.view(b, t, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    v = v.view(b, t, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    return q, k, v, head_dim


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    # x: [B, heads, tokens, head_dim]
    b, h, t, d = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(b, t, h * d)


def _build_sparse_attention_module(num_heads: int, head_dim: int) -> torch.nn.Module:
    try:
        from Ops.sparse_attention import SparseAttention
    except Exception as exc:
        raise RuntimeError(
            "TODO hook: SparseFlow SparseAttention is not available in Ops/sparse_attention.py"
        ) from exc
    return SparseAttention(
        num_heads=num_heads,
        head_dim=head_dim,
        threshold=1e-6,
        return_ms=False,
    )


def sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    threshold: float = 1e-6,
) -> torch.Tensor:
    """SparseFlow compatibility hook: sparse_attention(q, k, v)."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must be [B, heads, tokens, head_dim].")
    heads = int(q.shape[1])
    head_dim = int(q.shape[-1])
    op = _build_sparse_attention_module(num_heads=heads, head_dim=head_dim).to(q.device).eval()
    if hasattr(op, "collect_diag"):
        op.collect_diag = False
    logits = op.qk(q, k)
    attn = torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)
    out = op.av(attn, v)
    return out
