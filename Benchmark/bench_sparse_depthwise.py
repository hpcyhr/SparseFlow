"""
SparseFlow Benchmark — Sparse Depthwise Conv2d

Compares sparse_depthwise_conv2d_forward vs F.conv2d(groups=C_in).

Usage:
    cd ~/SparseFlow
    python Benchmark/bench_sparse_depthwise.py
    python Benchmark/bench_sparse_depthwise.py --C 128 --H 28 --W 28 --activity 0.01
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import torch
import torch.nn.functional as F


def cosine_sim(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    d = a.norm() * b.norm()
    return (a @ b / d).item() if d > 0 else 1.0


def main():
    parser = argparse.ArgumentParser(description="SparseFlow Sparse Depthwise Conv2d Benchmark")
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--C", type=int, default=64)
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--W", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--activity", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    from Kernels.depthwise_conv2d import sparse_depthwise_conv2d_forward

    # Sparse input
    x = torch.randn(args.N, args.C, args.H, args.W, device=device, dtype=torch.float16)
    mask = (torch.rand(args.N, args.C, args.H, args.W, device=device) < args.activity)
    x = x * mask.half()

    # Depthwise weight [C, 1, K, K]
    weight = torch.randn(args.C, 1, args.kernel_size, args.kernel_size, device=device, dtype=torch.float32)
    bias = torch.randn(args.C, device=device, dtype=torch.float32)

    elem_sparsity = 1.0 - mask.float().mean().item()
    print(f"{'='*80}")
    print(f"Sparse Depthwise Conv2d Benchmark")
    print(f"{'='*80}")
    print(f"  Shape: [{args.N}, {args.C}, {args.H}, {args.W}]  K={args.kernel_size}")
    print(f"  Activity: {args.activity:.4f}  Element sparsity: {elem_sparsity:.4f}")

    # Correctness
    with torch.no_grad():
        y_dense = F.conv2d(x.float(), weight, bias, stride=1, padding=1, groups=args.C)
        result = sparse_depthwise_conv2d_forward(
            x, weight, bias, stride=1, padding=1,
            return_ms=False, return_tile_stats=True,
        )
        y_sparse = result[0]

    cos = cosine_sim(y_dense, y_sparse)
    max_abs = (y_dense - y_sparse.float()).abs().max().item()
    print(f"\n  [Correctness] cosine={cos:.8f}  max_abs_err={max_abs:.6f}  {'PASS' if cos > 0.99 else 'FAIL'}")

    if len(result) > 2 and result[2] is not None:
        stats = result[2]
        print(f"  [Tiles] total={stats['total_tiles']}  active={stats.get('active_tiles',0)}  "
              f"zero={stats.get('zero_tiles',0)}  zero_ratio={stats.get('zero_ratio',0):.4f}")

    # Warmup
    for _ in range(args.warmup):
        sparse_depthwise_conv2d_forward(x, weight, bias, stride=1, padding=1)
        F.conv2d(x.float(), weight, bias, stride=1, padding=1, groups=args.C)
    torch.cuda.synchronize(device)

    # Benchmark
    sparse_times = []
    for _ in range(args.iters):
        _, ms = sparse_depthwise_conv2d_forward(
            x, weight, bias, stride=1, padding=1, return_ms=True,
        )
        sparse_times.append(ms)

    dense_times = []
    for _ in range(args.iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        F.conv2d(x.float(), weight, bias, stride=1, padding=1, groups=args.C)
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