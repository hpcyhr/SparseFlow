#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conv1d Sparse Benchmark for SparseFlow

Compares:
- Dense cuDNN Conv1d (torch.nn.functional.conv1d)
- SparseFlow Conv1d sparse kernel (Kernels.conv1d.sparse_conv1d_forward)

Coverage:
- Sparsity: 90%, 95%, 97%, 98%, 99%, 99.5%
- Pattern: random element-wise sparsity, block/tile-aligned sparsity

Metrics:
- latency (CUDA Event, warmup + repeat)
- speedup (Dense / SparseFlow)
- numerical consistency (cosine, max abs err, mean abs err, allclose)

CSV columns are designed to match the requested table style.
"""

import argparse
import csv
import inspect
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


# ----------------------------
# Project import bootstrap
# ----------------------------
def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "Kernels" / "conv1d.py").exists() and (p / "Ops").exists():
            return p
    raise RuntimeError(
        "Cannot locate SparseFlow project root (expecting Kernels/conv1d.py). "
        "Run this script inside the SparseFlow repository."
    )


_THIS = Path(__file__).resolve() if "__file__" in globals() else Path.cwd()
_PROJECT_ROOT = find_project_root(_THIS)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Kernels.conv1d import sparse_conv1d_forward  # noqa: E402

_HAS_RETURN_BACKEND_META = (
    "return_backend_meta" in inspect.signature(sparse_conv1d_forward).parameters
)


# ----------------------------
# Config / cases
# ----------------------------
@dataclass(frozen=True)
class Conv1dCase:
    b: int
    c: int
    l: int
    k: int
    s: int
    p: int
    c_out: int

    @property
    def shape_str(self) -> str:
        return f"{self.b}x{self.c}x{self.l}"

    @property
    def kernel_stride_str(self) -> str:
        return f"k={self.k} / s{self.s}"


DEFAULT_CASES: List[Conv1dCase] = [
    Conv1dCase(32, 64, 256, 3, 1, 1, 64),
    Conv1dCase(32, 128, 128, 3, 1, 1, 128),
    Conv1dCase(32, 256, 64, 3, 1, 1, 256),
    Conv1dCase(32, 128, 128, 3, 2, 1, 128),
    Conv1dCase(32, 256, 64, 1, 1, 0, 256),
]

SPARSITIES = [0.90, 0.95, 0.97, 0.98, 0.99, 0.995]
PATTERNS = ["random", "tile"]


# ----------------------------
# Helpers
# ----------------------------
def parse_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def sparsity_label(s: float) -> str:
    pct = s * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return f"{int(round(pct))}%"
    return f"{pct:.1f}%"


def pattern_label(p: str) -> str:
    if p == "random":
        return "Random sparsity"
    if p == "tile":
        return "Block / tile-aligned sparsity"
    return p


def make_random_sparse_input(
    shape: Tuple[int, int, int],
    sparsity: float,
    device: torch.device,
    dtype: torch.dtype,
    gen: torch.Generator,
) -> torch.Tensor:
    x = torch.randn(shape, device=device, dtype=dtype, generator=gen)
    keep_prob = 1.0 - sparsity
    mask = (torch.rand(shape, device=device, generator=gen) < keep_prob)
    return (x * mask.to(dtype)).contiguous()


def make_tile_aligned_sparse_input(
    shape: Tuple[int, int, int],
    sparsity: float,
    tile_c: int,
    tile_l: int,
    device: torch.device,
    dtype: torch.dtype,
    gen: torch.Generator,
) -> torch.Tensor:
    b, c, l = shape
    keep_prob = 1.0 - sparsity
    c_tiles = math.ceil(c / tile_c)
    l_tiles = math.ceil(l / tile_l)

    tile_keep = (torch.rand((b, c_tiles, l_tiles), device=device, generator=gen) < keep_prob)
    mask = tile_keep.repeat_interleave(tile_c, dim=1).repeat_interleave(tile_l, dim=2)
    mask = mask[:, :c, :l]

    x = torch.randn((b, c, l), device=device, dtype=dtype, generator=gen)
    return (x * mask.to(dtype)).contiguous()


def actual_sparsity(x: torch.Tensor) -> float:
    non_zero = (x != 0).float().mean().item()
    return 1.0 - float(non_zero)


@torch.no_grad()
def cuda_event_bench_ms(fn, warmup: int, repeat: int) -> Tuple[float, float, float]:
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(repeat):
        start.record()
        _ = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    mean_ms = float(sum(times) / len(times))
    p50_ms = float(statistics.median(times))
    std_ms = float(statistics.pstdev(times)) if len(times) > 1 else 0.0
    return mean_ms, p50_ms, std_ms


@torch.no_grad()
def sparse_forward_once(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: int,
    padding: int,
    threshold: float,
    need_tile_stats: bool,
    need_backend_meta: bool,
):
    kwargs = dict(
        stride=stride,
        padding=padding,
        threshold=threshold,
        return_ms=False,
        return_tile_stats=need_tile_stats,
    )
    if _HAS_RETURN_BACKEND_META:
        kwargs["return_backend_meta"] = need_backend_meta

    out = sparse_conv1d_forward(x, w, b, **kwargs)
    if not isinstance(out, tuple):
        return out, {}, {}

    y = out[0]
    idx = 2
    stats = {}
    backend = {}

    if need_tile_stats and len(out) > idx and isinstance(out[idx], dict):
        stats = out[idx]
        idx += 1

    if _HAS_RETURN_BACKEND_META and need_backend_meta and len(out) > idx and isinstance(out[idx], dict):
        backend = out[idx]
    elif stats:
        backend = {"backend": stats.get("backend", ""), "reason": stats.get("reason", "")}

    return y, stats, backend


@torch.no_grad()
def compute_numeric_metrics(
    y_ref: torch.Tensor,
    y_sparse: torch.Tensor,
    atol: float,
    rtol: float,
) -> Tuple[float, float, float, bool]:
    a = y_ref.float().reshape(-1)
    b = y_sparse.float().reshape(-1)
    diff = (a - b).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())

    denom = float(a.norm().item() * b.norm().item())
    if denom > 0.0:
        cosine = float((a @ b).item() / denom)
    else:
        cosine = 1.0

    ok = bool(torch.allclose(a, b, atol=atol, rtol=rtol))
    return cosine, max_abs, mean_abs, ok


def make_csv_row(
    case: Conv1dCase,
    sparsity: float,
    pattern: str,
    dense_ms: float,
    sparse_ms: float,
    speedup: float,
    cosine: float,
    max_abs: float,
    mean_abs: float,
    allclose_ok: bool,
    measured_sparsity: float,
    backend: Dict[str, str],
) -> Dict[str, object]:
    return {
        "Operator": "Conv1d",
        "Shape (B, C, L)": case.shape_str,
        "Kernel / Stride": case.kernel_stride_str,
        "Sparsity": sparsity_label(sparsity),
        "Pattern": pattern_label(pattern),
        "Dense": f"{dense_ms:.4f}",
        "SparseFlow": f"{sparse_ms:.4f}",
        "Speedup": "inf" if math.isinf(speedup) else f"{speedup:.4f}",
        "Cosine": f"{cosine:.8f}",
        "MaxAbsErr": f"{max_abs:.6e}",
        "MeanAbsErr": f"{mean_abs:.6e}",
        "AllClose": int(allclose_ok),
        "ActualSparsity(%)": f"{measured_sparsity * 100.0:.3f}",
        "SparseBackend": str(backend.get("backend", "")),
        "SparseReason": str(backend.get("reason", "")),
    }


def save_csv(rows: List[Dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Operator",
        "Shape (B, C, L)",
        "Kernel / Stride",
        "Sparsity",
        "Pattern",
        "Dense",
        "SparseFlow",
        "Speedup",
        "Cosine",
        "MaxAbsErr",
        "MeanAbsErr",
        "AllClose",
        "ActualSparsity(%)",
        "SparseBackend",
        "SparseReason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------
# Main benchmark
# ----------------------------
@torch.no_grad()
def run_benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires a CUDA GPU.")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = parse_dtype(args.dtype)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)

    rows: List[Dict[str, object]] = []
    total = len(DEFAULT_CASES) * len(SPARSITIES) * len(PATTERNS)
    done = 0

    print("=" * 96)
    print("SparseFlow Conv1d Benchmark")
    print(f"device={device}, dtype={dtype}, warmup={args.warmup}, repeat={args.repeat}")
    print(f"sparsities={[sparsity_label(s) for s in SPARSITIES]}, patterns={PATTERNS}")
    print("=" * 96)

    for s_idx, sparsity in enumerate(SPARSITIES):
        for p_idx, pattern in enumerate(PATTERNS):
            for c_idx, case in enumerate(DEFAULT_CASES):
                done += 1
                local_seed = args.seed + s_idx * 1000 + p_idx * 100 + c_idx
                gen = torch.Generator(device=device)
                gen.manual_seed(local_seed)

                shape = (case.b, case.c, case.l)

                if pattern == "random":
                    x = make_random_sparse_input(shape, sparsity, device, dtype, gen)
                elif pattern == "tile":
                    x = make_tile_aligned_sparse_input(
                        shape,
                        sparsity,
                        args.tile_c,
                        args.tile_l,
                        device,
                        dtype,
                        gen,
                    )
                else:
                    raise ValueError(f"Unknown pattern: {pattern}")

                # Keep weights fixed across dense/sparse path per case instance.
                w = torch.randn(
                    (case.c_out, case.c, case.k),
                    device=device,
                    dtype=dtype,
                    generator=gen,
                ).contiguous()
                b = torch.randn((case.c_out,), device=device, dtype=dtype, generator=gen).contiguous()

                # Numerical reference (float32)
                y_ref = F.conv1d(
                    x.float(),
                    w.float(),
                    b.float(),
                    stride=case.s,
                    padding=case.p,
                ).float()

                y_sparse_chk, stats, backend = sparse_forward_once(
                    x=x,
                    w=w,
                    b=b,
                    stride=case.s,
                    padding=case.p,
                    threshold=args.threshold,
                    need_tile_stats=True,
                    need_backend_meta=True,
                )

                cosine, max_abs, mean_abs, allclose_ok = compute_numeric_metrics(
                    y_ref, y_sparse_chk, atol=args.atol, rtol=args.rtol
                )

                # Dense cuDNN timing
                def dense_fn():
                    return F.conv1d(x, w, b, stride=case.s, padding=case.p)

                # SparseFlow timing
                def sparse_fn():
                    y, _, _ = sparse_forward_once(
                        x=x,
                        w=w,
                        b=b,
                        stride=case.s,
                        padding=case.p,
                        threshold=args.threshold,
                        need_tile_stats=False,
                        need_backend_meta=False,
                    )
                    return y

                dense_ms, _, _ = cuda_event_bench_ms(dense_fn, args.warmup, args.repeat)
                sparse_ms, _, _ = cuda_event_bench_ms(sparse_fn, args.warmup, args.repeat)
                speedup = dense_ms / sparse_ms if sparse_ms > 0.0 else float("inf")

                row = make_csv_row(
                    case=case,
                    sparsity=sparsity,
                    pattern=pattern,
                    dense_ms=dense_ms,
                    sparse_ms=sparse_ms,
                    speedup=speedup,
                    cosine=cosine,
                    max_abs=max_abs,
                    mean_abs=mean_abs,
                    allclose_ok=allclose_ok,
                    measured_sparsity=actual_sparsity(x),
                    backend=backend if backend else {
                        "backend": stats.get("backend", ""),
                        "reason": stats.get("reason", ""),
                    },
                )
                rows.append(row)

                print(
                    f"[{done:02d}/{total:02d}] "
                    f"{case.shape_str:>11s} {case.kernel_stride_str:>10s} "
                    f"{sparsity_label(sparsity):>6s} | {pattern_label(pattern):<28s} "
                    f"Dense={dense_ms:8.4f} ms  Sparse={sparse_ms:8.4f} ms  "
                    f"Speedup={('inf' if math.isinf(speedup) else f'{speedup:6.3f}')}  "
                    f"Cos={cosine: .6f}  MaxErr={max_abs: .3e}  "
                    f"Backend={row['SparseBackend']}"
                )

    save_csv(rows, Path(args.csv))
    print("=" * 96)
    print(f"Done. CSV saved to: {Path(args.csv).resolve()}")
    print("=" * 96)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark Dense cuDNN Conv1d vs SparseFlow Conv1d sparse kernel."
    )
    p.add_argument("--device", type=str, default="cuda:0", help="CUDA device, e.g. cuda:0")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"])
    p.add_argument("--warmup", type=int, default=40, help="Warmup iterations")
    p.add_argument("--repeat", type=int, default=120, help="Measured iterations")
    p.add_argument("--seed", type=int, default=20260408)
    p.add_argument("--threshold", type=float, default=1e-6, help="SparseFlow threshold")
    p.add_argument("--tile-c", type=int, default=16, help="Block sparsity tile size in C")
    p.add_argument("--tile-l", type=int, default=32, help="Block sparsity tile size in L")
    p.add_argument("--atol", type=float, default=1e-3, help="allclose atol")
    p.add_argument("--rtol", type=float, default=1e-3, help="allclose rtol")
    p.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Enable TF32 for CUDA matmul/cudnn (default: False)",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=str(_PROJECT_ROOT / "Benchmark" / "outputs" / "conv1d_sparse_benchmark.csv"),
        help="CSV output path",
    )
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_benchmark(args)
