#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_sparse_linear_unified.py

Unified benchmark / correctness harness for conv-style SparseLinear.

Goals:
  - Avoid old bench script interface mismatches
  - Test 2D and 3D inputs in one place
  - Validate correctness first, then latency
  - Work with current conv-style SparseLinear implementation

Examples:
  python bench_sparse_linear_unified.py \
    --project_root /home/yhr/SparseFlow \
    --cases small2d medium2d large2d small3d large3d \
    --mode bernoulli_mask \
    --activity 0.01 \
    --iters 100 \
    --warmup 20

  python bench_sparse_linear_unified.py \
    --project_root /home/yhr/SparseFlow \
    --cases large3d \
    --mode bernoulli_mask \
    --activity 0.01 \
    --iters 50 \
    --warmup 10 \
    --profile_runtime
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PRESET_CASES: Dict[str, Tuple[int, ...]] = {
    # 2D: (B, Cin, Cout)
    "small2d":  (4, 16, 16),
    "medium2d": (32, 128, 128),
    "large2d":  (128, 512, 512),
    # 3D: (T, B, Cin, Cout)
    "small3d":  (4, 8, 128, 128),
    "large3d":  (16, 128, 512, 512),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", type=str, required=True)
    p.add_argument(
        "--cases",
        nargs="+",
        default=["small2d", "medium2d", "large2d", "small3d", "large3d"],
        help=f"Preset cases from: {list(PRESET_CASES.keys())}",
    )
    p.add_argument("--mode", type=str, default="bernoulli_mask",
                   choices=["bernoulli_mask", "manual", "single_hot", "all_ones_small"])
    p.add_argument("--activity", type=float, default=0.01)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--profile_runtime", action="store_true")
    p.add_argument("--dump_profile", action="store_true")
    p.add_argument("--threshold", type=float, default=1e-6)
    p.add_argument("--dense_threshold", type=float, default=0.25)
    p.add_argument("--warmup_steps", type=int, default=8)
    p.add_argument("--calib_every", type=int, default=32)
    p.add_argument("--ema_decay", type=float, default=0.9)
    p.add_argument("--zero_streak_needed", type=int, default=2)
    return p.parse_args()


def import_sparseflow(project_root: str):
    root = str(Path(project_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from Ops.sparse_linear import SparseLinear
    return SparseLinear


def make_dtype(name: str):
    return torch.float16 if name == "fp16" else torch.float32


def make_input_2d(B: int, Cin: int, mode: str, activity: float, device: str, dtype: torch.dtype):
    if mode == "bernoulli_mask":
        x = torch.randn(B, Cin, device=device, dtype=dtype)
        mask = (torch.rand(B, Cin, device=device) < activity)
        return x * mask.to(dtype)
    elif mode == "manual":
        x = torch.zeros(B, Cin, device=device, dtype=dtype)
        for b in range(B):
            for k in range(Cin):
                if (b + k) % 5 == 0:
                    x[b, k] = (k % 7 + 1) * 0.1
        return x
    elif mode == "single_hot":
        x = torch.zeros(B, Cin, device=device, dtype=dtype)
        for b in range(B):
            x[b, b % Cin] = 1.0
        return x
    elif mode == "all_ones_small":
        return torch.ones(B, Cin, device=device, dtype=dtype)
    raise ValueError(mode)


def make_input(case_name: str, shape: Tuple[int, ...], mode: str, activity: float, device: str, dtype: torch.dtype):
    if case_name.endswith("2d"):
        B, Cin, _ = shape
        return make_input_2d(B, Cin, mode, activity, device, dtype)
    elif case_name.endswith("3d"):
        T, B, Cin, _ = shape
        x = make_input_2d(T * B, Cin, mode, activity, device, dtype)
        return x.reshape(T, B, Cin)
    else:
        raise ValueError(f"Unsupported case type: {case_name}")


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    denom = a.norm() * b.norm()
    if float(denom.item()) == 0.0:
        return 1.0 if float((a - b).abs().max().item()) == 0.0 else 0.0
    return float(torch.dot(a, b).item() / denom.item())


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def nnz_ratio(x: torch.Tensor) -> float:
    return float(torch.count_nonzero(x).item()) / float(max(1, x.numel()))


def measure_ms(fn, x: torch.Tensor, warmup: int, iters: int) -> float:
    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = fn(x)
    if x.is_cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = fn(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / max(1, iters)


def build_modules(SparseLinear, shape: Tuple[int, ...], device: str, dtype: torch.dtype, args):
    if len(shape) == 3:
        _, Cin, Cout = shape
    else:
        _, _, Cin, Cout = shape

    dense = nn.Linear(Cin, Cout, bias=True).to(device=device, dtype=dtype).eval()
    sparse = SparseLinear.from_dense(
        dense,
        threshold=args.threshold,
        dense_threshold=args.dense_threshold,
        warmup_steps=args.warmup_steps,
        calib_every=args.calib_every,
        ema_decay=args.ema_decay,
        zero_streak_needed=args.zero_streak_needed,
        profile_runtime=args.profile_runtime,
    ).to(device).eval()
    return dense, sparse


def describe_shape(case_name: str, shape: Tuple[int, ...]) -> str:
    if case_name.endswith("2d"):
        B, Cin, Cout = shape
        return f"2D [B,C] with B={B}, Cin={Cin}, Cout={Cout}"
    T, B, Cin, Cout = shape
    return f"3D [T,B,C] with T={T}, B={B}, Cin={Cin}, Cout={Cout}"


def run_case(case_name: str, shape: Tuple[int, ...], SparseLinear, args):
    device = args.device
    dtype = make_dtype(args.dtype)

    x = make_input(case_name, shape, args.mode, args.activity, device, dtype)
    dense, sparse = build_modules(SparseLinear, shape, device, dtype, args)

    with torch.no_grad():
        y_dense = dense(x)
        y_sparse = sparse(x)

    cos = cosine(y_dense, y_sparse)
    mad = max_abs_diff(y_dense, y_sparse)

    dense_ms = measure_ms(dense, x, args.warmup, args.iters)
    sparse_ms = measure_ms(sparse, x, args.warmup, args.iters)
    speedup = dense_ms / max(sparse_ms, 1e-12)

    result = {
        "case": case_name,
        "shape_desc": describe_shape(case_name, shape),
        "input_nnz_ratio": nnz_ratio(x),
        "dense_ms": dense_ms,
        "sparse_ms": sparse_ms,
        "speedup": speedup,
        "cosine": cos,
        "max_abs_diff": mad,
    }

    profile_txt = None
    if args.dump_profile and hasattr(sparse, "get_runtime_profile_pretty"):
        profile_txt = sparse.get_runtime_profile_pretty()

    return result, profile_txt


def print_result_table(results: List[Dict[str, float]]):
    print("=" * 132)
    print(f"{'Case':<10} {'Input':<42} {'NNZ':>8} {'Dense(ms)':>12} {'Sparse(ms)':>12} {'Speedup':>10} {'Cosine':>12} {'MaxAbsErr':>12}")
    print("-" * 132)
    for r in results:
        print(f"{r['case']:<10} {r['shape_desc']:<42} "
              f"{r['input_nnz_ratio']*100:>7.3f}% "
              f"{r['dense_ms']:>12.4f} "
              f"{r['sparse_ms']:>12.4f} "
              f"{r['speedup']:>10.3f} "
              f"{r['cosine']:>12.8f} "
              f"{r['max_abs_diff']:>12.8f}")
    print("=" * 132)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    SparseLinear = import_sparseflow(args.project_root)

    results = []
    profiles = []

    print("=" * 100)
    print("Unified SparseLinear benchmark")
    print("=" * 100)
    print(f"project_root:     {args.project_root}")
    print(f"cases:            {args.cases}")
    print(f"mode:             {args.mode}")
    print(f"activity:         {args.activity}")
    print(f"dtype/device:     {args.dtype} / {args.device}")
    print(f"warmup/iters:     {args.warmup} / {args.iters}")
    print(f"profile_runtime:  {args.profile_runtime}")
    print(f"dump_profile:     {args.dump_profile}")
    print("=" * 100)

    for case_name in args.cases:
        if case_name not in PRESET_CASES:
            raise ValueError(f"Unknown preset case '{case_name}'. Available: {list(PRESET_CASES.keys())}")
        shape = PRESET_CASES[case_name]
        result, profile_txt = run_case(case_name, shape, SparseLinear, args)
        results.append(result)
        if profile_txt is not None:
            profiles.append((case_name, profile_txt))

    print_result_table(results)

    if profiles:
        print("\n[Per-case runtime profiles]")
        print("=" * 100)
        for case_name, txt in profiles:
            print(f"\n--- {case_name} ---")
            print(txt)

    bad = [r for r in results if r["cosine"] < 0.999 or r["max_abs_diff"] > 1e-2]
    if bad:
        print("\n[Summary]")
        print(f"  {len(bad)} case(s) have suspicious correctness.")
    else:
        print("\n[Summary]")
        print("  All tested cases passed basic correctness thresholds.")


if __name__ == "__main__":
    main()
