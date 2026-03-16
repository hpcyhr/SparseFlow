import argparse
import math
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

# Ensure project root import works when running from Benchmark/
THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def add_project_paths(project_root: Path) -> None:
    for p in [project_root, project_root / 'Ops', project_root / 'Kernels']:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def make_input(shape: Tuple[int, ...], mode: str, activity: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    x = torch.randn(*shape, device=device, dtype=dtype)
    if mode == 'dense':
        return x
    if mode == 'bernoulli_mask':
        mask = (torch.rand(*shape, device=device) < activity).to(dtype)
        return x * mask
    if mode == 'binary_spike':
        return (torch.rand(*shape, device=device) < activity).to(dtype)
    raise ValueError(f'Unknown mode: {mode}')


@torch.no_grad()
def benchmark_module(module: nn.Module, x: torch.Tensor, iters: int, warmup: int) -> Tuple[float, torch.Tensor]:
    module.eval()
    # warmup
    for _ in range(warmup):
        y = module(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True) if x.is_cuda else None
    end = torch.cuda.Event(enable_timing=True) if x.is_cuda else None
    if x.is_cuda:
        start.record()
        for _ in range(iters):
            y = module(x)
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)
    else:
        import time
        t0 = time.perf_counter()
        for _ in range(iters):
            y = module(x)
        total_ms = (time.perf_counter() - t0) * 1000.0
    return total_ms / iters, y


def main() -> None:
    parser = argparse.ArgumentParser(description='SparseLinear micro benchmark + correctness test')
    parser.add_argument('--project_root', type=str, default='.', help='SparseFlow repo root')
    parser.add_argument('--in_features', type=int, default=512)
    parser.add_argument('--out_features', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--T', type=int, default=1)
    parser.add_argument('--mode', type=str, default='bernoulli_mask', choices=['dense', 'bernoulli_mask', 'binary_spike'])
    parser.add_argument('--activity', type=float, default=0.01, help='fraction of active elements')
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'fp32'])
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--threshold', type=float, default=1e-6)
    parser.add_argument('--dense_threshold', type=float, default=0.85)
    parser.add_argument('--profile_runtime', action='store_true')
    parser.add_argument('--compare_2d', action='store_true', help='also benchmark [B,C] input')
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    add_project_paths(project_root)

    from Ops.sparse_linear import SparseLinear

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this benchmark.')
    device = torch.device('cuda')
    dtype = torch.float16 if args.dtype == 'fp16' else torch.float32

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    dense = nn.Linear(args.in_features, args.out_features, bias=True).to(device=device, dtype=dtype).eval()
    sparse = SparseLinear.from_dense(
        dense,
        threshold=args.threshold,
        dense_threshold=args.dense_threshold,
        warmup_steps=8,
        calib_every=32,
        ema_decay=0.9,
        zero_streak_needed=2,
        profile_runtime=args.profile_runtime,
    ).to(device=device).eval()

    def run_case(x: torch.Tensor, tag: str) -> None:
        print('\n' + '=' * 90)
        print(f'Case: {tag} | shape={tuple(x.shape)} | dtype={x.dtype} | activity≈{args.activity:.4f} | mode={args.mode}')
        print('=' * 90)
        dense_ms, y_dense = benchmark_module(dense, x, args.iters, args.warmup)
        sparse_ms, y_sparse = benchmark_module(sparse, x, args.iters, args.warmup)

        # correctness
        y_dense_f = y_dense.float().reshape(-1)
        y_sparse_f = y_sparse.float().reshape(-1)
        cos = torch.nn.functional.cosine_similarity(y_dense_f, y_sparse_f, dim=0).item()
        max_abs = (y_dense.float() - y_sparse.float()).abs().max().item()

        print(f'Dense:       {dense_ms:.4f} ms/iter')
        print(f'SparseLinear:{sparse_ms:.4f} ms/iter')
        print(f'Speedup:     {dense_ms / max(sparse_ms, 1e-9):.3f}x')
        print(f'Cosine sim:  {cos:.8f}')
        print(f'Max abs err: {max_abs:.8g}')

        if hasattr(sparse, 'get_runtime_profile_pretty'):
            print('\n[Runtime profile]')
            print(sparse.get_runtime_profile_pretty())

    # 3D multi-step [T, B, Cin]
    if args.T > 1:
        x3 = make_input((args.T, args.batch_size, args.in_features), args.mode, args.activity, device, dtype)
        run_case(x3, '3D multi-step [T,B,Cin]')
    else:
        x2 = make_input((args.batch_size, args.in_features), args.mode, args.activity, device, dtype)
        run_case(x2, '2D single-step [B,Cin]')

    if args.compare_2d and args.T > 1:
        x2 = make_input((args.batch_size * args.T, args.in_features), args.mode, args.activity, device, dtype)
        run_case(x2, '2D folded [T*B,Cin]')


if __name__ == '__main__':
    main()
