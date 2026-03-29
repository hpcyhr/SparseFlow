"""
SparseFlow Benchmark — Sparse BMM

Compares sparse_bmm_forward vs torch.bmm at various sparsity levels.

Usage:
    cd ~/SparseFlow
    python Benchmark/bench_sparse_bmm.py
    python Benchmark/bench_sparse_bmm.py --B 16 --M 128 --K 64 --N 128 --activity 0.02
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import torch


def cosine_sim(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    d = a.norm() * b.norm()
    return (a @ b / d).item() if d > 0 else 1.0


def main():
    parser = argparse.ArgumentParser(description="SparseFlow Sparse BMM Benchmark")
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--M", type=int, default=128)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--activity", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    from Kernels.bmm import sparse_bmm_forward

    a = torch.randn(args.B, args.M, args.K, device=device, dtype=torch.float16)
    mask = (torch.rand(args.B, args.M, args.K, device=device) < args.activity)
    a = a * mask.half()
    b = torch.randn(args.B, args.K, args.N, device=device, dtype=torch.float16)

    elem_sparsity = 1.0 - mask.float().mean().item()
    print(f"{'='*80}")
    print(f"Sparse BMM Benchmark")
    print(f"{'='*80}")
    print(f"  Shape: [{args.B}, {args.M}, {args.K}] × [{args.B}, {args.K}, {args.N}]")
    print(f"  Activity: {args.activity:.4f}  Element sparsity: {elem_sparsity:.4f}")

    # Correctness
    with torch.no_grad():
        y_dense = torch.bmm(a.float(), b.float())
        result = sparse_bmm_forward(a, b, return_ms=False, return_tile_stats=True)
        y_sparse = result[0]

    cos = cosine_sim(y_dense, y_sparse)
    max_abs = (y_dense - y_sparse.float()).abs().max().item()
    print(f"\n  [Correctness] cosine={cos:.8f}  max_abs_err={max_abs:.6f}  {'PASS' if cos > 0.999 else 'FAIL'}")

    if len(result) > 2 and result[2] is not None:
        stats = result[2]
        print(f"  [Tiles] total={stats['total_tiles']}  zero={stats.get('zero_tiles',0)}  "
              f"sparse={stats.get('sparse_tiles',0)}  denseish={stats.get('denseish_tiles',0)}")

    # Warmup
    for _ in range(args.warmup):
        sparse_bmm_forward(a, b)
        torch.bmm(a.float(), b.float())
    torch.cuda.synchronize(device)

    # Benchmark
    sparse_times = []
    for _ in range(args.iters):
        _, ms = sparse_bmm_forward(a, b, return_ms=True)
        sparse_times.append(ms)

    dense_times = []
    for _ in range(args.iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        torch.bmm(a.float(), b.float())
        e.record()
        torch.cuda.synchronize(device)
        dense_times.append(s.elapsed_time(e))

    avg_s = sum(sparse_times) / len(sparse_times)
    avg_d = sum(dense_times) / len(dense_times)
    speedup = avg_d / avg_s if avg_s > 0 else float("inf")

    print(f"\n  [Latency] sparse={avg_s:.3f}ms  dense={avg_d:.3f}ms  speedup={speedup:.2f}x")
    print()

if __name__ == "__main__":
    main()