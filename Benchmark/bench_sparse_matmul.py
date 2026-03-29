"""
SparseFlow Benchmark — Sparse Matmul

Compares sparse_matmul_forward vs torch.mm at various sparsity levels.
Validates correctness and measures latency.

Usage:
    cd ~/SparseFlow
    python Benchmark/bench_sparse_matmul.py
    python Benchmark/bench_sparse_matmul.py --M 1024 --K 512 --N 256 --activity 0.01
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
    parser = argparse.ArgumentParser(description="SparseFlow Sparse Matmul Benchmark")
    parser.add_argument("--M", type=int, default=512)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--activity", type=float, default=0.01, help="Fraction of nonzero elements in A")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    from Kernels.matmul import sparse_matmul_forward

    # Create sparse input
    a = torch.randn(args.M, args.K, device=device, dtype=torch.float16)
    mask = (torch.rand(args.M, args.K, device=device) < args.activity)
    a = a * mask.half()
    b = torch.randn(args.K, args.N, device=device, dtype=torch.float16)

    elem_sparsity = 1.0 - mask.float().mean().item()
    print(f"{'='*80}")
    print(f"Sparse Matmul Benchmark")
    print(f"{'='*80}")
    print(f"  Shape: [{args.M}, {args.K}] × [{args.K}, {args.N}]")
    print(f"  Activity: {args.activity:.4f}  Element sparsity: {elem_sparsity:.4f}")
    print()

    # Correctness
    with torch.no_grad():
        y_dense = torch.mm(a.float(), b.float())
        y_sparse, _ = sparse_matmul_forward(
            a, b, return_ms=False, return_tile_stats=True,
        )[:2]

    cos = cosine_sim(y_dense, y_sparse)
    max_abs = (y_dense - y_sparse.float()).abs().max().item()
    print(f"  [Correctness] cosine={cos:.8f}  max_abs_err={max_abs:.6f}  {'PASS' if cos > 0.999 else 'FAIL'}")

    # Tile stats
    _, _, stats = sparse_matmul_forward(a, b, return_ms=False, return_tile_stats=True)
    if stats:
        print(f"  [Tiles] total={stats['total_tiles']}  zero={stats.get('zero_tiles',0)}  "
              f"sparse={stats.get('sparse_tiles',0)}  denseish={stats.get('denseish_tiles',0)}")

    # Warmup
    for _ in range(args.warmup):
        sparse_matmul_forward(a, b, return_ms=False)
        torch.mm(a.float(), b.float())
    torch.cuda.synchronize(device)

    # Benchmark sparse
    sparse_times = []
    for _ in range(args.iters):
        _, ms = sparse_matmul_forward(a, b, return_ms=True)
        sparse_times.append(ms)

    # Benchmark dense
    dense_times = []
    for _ in range(args.iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.mm(a.float(), b.float())
        end.record()
        torch.cuda.synchronize(device)
        dense_times.append(start.elapsed_time(end))

    avg_sparse = sum(sparse_times) / len(sparse_times)
    avg_dense = sum(dense_times) / len(dense_times)
    speedup = avg_dense / avg_sparse if avg_sparse > 0 else float("inf")

    print(f"\n  [Latency] sparse={avg_sparse:.3f}ms  dense={avg_dense:.3f}ms  speedup={speedup:.2f}x")
    print()

if __name__ == "__main__":
    main()