"""
SparseFlow cuSPARSE Baseline Benchmark

Independent script for comparing SparseFlow against PyTorch sparse CSR
(cuSPARSE-backed when available) via im2col + SpMM.

Usage:
    python bench_cusparse.py --cin 64 --cout 64 --h 56 --w 56 --n 512 --sparsity 0.95
    python bench_cusparse.py --sweep   # run over a set of typical ResNet50 layer configs

Notes:
    - im2col uses F.unfold for 3×3 conv (stride=1, padding=1)
    - Sparse matrix is constructed via threshold on the unfolded input
    - SpMM uses torch.sparse.mm (cuSPARSE-backed on CUDA when available)
    - Timing separates: im2col(ms), csr_build(ms), spmm(ms), total(ms)
    - CSR construction overhead is NOT included in spmm timing
    - PyTorch sparse backend note: "cuSPARSE-backed when available"
"""

import argparse
import torch
import torch.nn.functional as F


def time_event(device):
    """Create a pair of CUDA events for timing."""
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    return s, e


def bench_cusparse_layer(N, C_IN, C_OUT, H, W, sparsity, kernel_size=3,
                          warmup=3, repeats=10, device='cuda'):
    """
    Benchmark one layer configuration using im2col + sparse CSR + SpMM.

    Returns dict with timing results.
    """
    # Generate sparse input (simulate SNN sparsity)
    x = torch.zeros(N, C_IN, H, W, dtype=torch.float16, device=device)
    # Randomly activate (1-sparsity) fraction of elements
    mask = torch.rand(N, C_IN, H, W, device=device) > sparsity
    x[mask] = 1.0

    # Weight: [C_OUT, C_IN, K, K]
    K = kernel_size
    weight = torch.randn(C_OUT, C_IN * K * K, dtype=torch.float16, device=device)

    pad = K // 2
    results = {}

    # ── Step 1: im2col timing ──
    # Warmup
    for _ in range(warmup):
        x_col = F.unfold(x.float(), kernel_size=K, padding=pad, stride=1)
    torch.cuda.synchronize(device)

    s, e = time_event(device)
    s.record()
    for _ in range(repeats):
        # F.unfold: [N, C_IN*K*K, H*W]
        x_col = F.unfold(x.float(), kernel_size=K, padding=pad, stride=1)
    e.record()
    torch.cuda.synchronize(device)
    results['im2col_ms'] = s.elapsed_time(e) / repeats

    # Reshape to 2D for SpMM: [N*H*W, C_IN*K*K]
    x_2d = x_col.permute(0, 2, 1).reshape(N * H * W, C_IN * K * K)

    # ── Step 2: CSR build timing ──
    # Threshold sparsification + CSR conversion
    for _ in range(warmup):
        x_sparse = x_2d.to_sparse_csr()
    torch.cuda.synchronize(device)

    s, e = time_event(device)
    s.record()
    for _ in range(repeats):
        x_sparse = x_2d.to_sparse_csr()
    e.record()
    torch.cuda.synchronize(device)
    results['csr_build_ms'] = s.elapsed_time(e) / repeats

    # ── Step 3: SpMM timing ──
    # x_sparse: [N*H*W, C_IN*K*K] CSR @ weight^T: [C_IN*K*K, C_OUT]
    w_t = weight.float().t().contiguous()  # [C_IN*K*K, C_OUT]
    x_sparse_f32 = x_2d.float().to_sparse_csr()

    for _ in range(warmup):
        y = torch.sparse.mm(x_sparse_f32, w_t)
    torch.cuda.synchronize(device)

    s, e = time_event(device)
    s.record()
    for _ in range(repeats):
        y = torch.sparse.mm(x_sparse_f32, w_t)
    e.record()
    torch.cuda.synchronize(device)
    results['spmm_ms'] = s.elapsed_time(e) / repeats

    results['total_ms'] = (results['im2col_ms']
                           + results['csr_build_ms']
                           + results['spmm_ms'])

    # Actual sparsity of the unfolded matrix
    nnz = x_sparse_f32._nnz()
    total_elems = x_2d.numel()
    results['actual_sparsity'] = 1.0 - (nnz / total_elems)
    results['nnz'] = nnz

    # ── cuDNN baseline for comparison ──
    conv = torch.nn.Conv2d(C_IN, C_OUT, K, padding=pad, bias=False,
                           dtype=torch.float16, device=device)

    for _ in range(warmup):
        with torch.no_grad():
            _ = F.conv2d(x, conv.weight, padding=pad)
    torch.cuda.synchronize(device)

    s, e = time_event(device)
    s.record()
    for _ in range(repeats):
        with torch.no_grad():
            _ = F.conv2d(x, conv.weight, padding=pad)
    e.record()
    torch.cuda.synchronize(device)
    results['cudnn_ms'] = s.elapsed_time(e) / repeats

    return results


# ═══════════════════════════════════════════════════════════════════════
# Typical ResNet50 layer configurations
# ═══════════════════════════════════════════════════════════════════════

RESNET50_LAYERS = [
    # (name, C_IN, C_OUT, H, W, kernel_size)
    ("layer1.0.conv2", 64, 64, 56, 56, 3),
    ("layer1.2.conv2", 64, 64, 56, 56, 3),
    ("layer2.0.conv2", 128, 128, 28, 28, 3),
    ("layer2.3.conv2", 128, 128, 28, 28, 3),
    ("layer3.0.conv2", 256, 256, 14, 14, 3),
    ("layer3.5.conv2", 256, 256, 14, 14, 3),
    ("layer4.0.conv2", 512, 512, 7, 7, 3),
    ("layer4.2.conv2", 512, 512, 7, 7, 3),
]


def main():
    parser = argparse.ArgumentParser(
        description="cuSPARSE baseline benchmark (im2col + CSR + SpMM)")
    parser.add_argument('--cin', type=int, default=64)
    parser.add_argument('--cout', type=int, default=64)
    parser.add_argument('--h', type=int, default=56)
    parser.add_argument('--w', type=int, default=56)
    parser.add_argument('--n', type=int, default=512,
                        help="Batch size (T*B for SNN)")
    parser.add_argument('--sparsity', type=float, default=0.95)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--sweep', action='store_true',
                        help="Run over typical ResNet50 layer configs")
    args = parser.parse_args()

    device = 'cuda'
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Sparse backend: PyTorch sparse CSR "
          f"(cuSPARSE-backed when available)")
    print()

    if args.sweep:
        sparsities = [0.90, 0.95, 0.99]
        N = args.n

        for sp in sparsities:
            print(f"{'='*80}")
            print(f"Sparsity: {sp*100:.0f}%  |  N(T*B): {N}")
            print(f"{'='*80}")
            header = (f"{'Layer':<22} {'im2col':>8} {'CSR':>8} "
                      f"{'SpMM':>8} {'Total':>8} {'cuDNN':>8} "
                      f"{'Sp/cuDNN':>8}")
            print(header)
            print("-" * len(header))

            for name, cin, cout, h, w, ks in RESNET50_LAYERS:
                try:
                    r = bench_cusparse_layer(
                        N, cin, cout, h, w, sp,
                        kernel_size=ks,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        device=device)
                    ratio = r['cudnn_ms'] / r['total_ms'] if r['total_ms'] > 0 else 0
                    print(f"{name:<22} "
                          f"{r['im2col_ms']:>7.2f}ms "
                          f"{r['csr_build_ms']:>7.2f}ms "
                          f"{r['spmm_ms']:>7.2f}ms "
                          f"{r['total_ms']:>7.2f}ms "
                          f"{r['cudnn_ms']:>7.2f}ms "
                          f"{ratio:>7.2f}x")
                except Exception as e:
                    print(f"{name:<22} ERROR: {e}")

            print()
    else:
        print(f"Config: N={args.n}, C_IN={args.cin}, C_OUT={args.cout}, "
              f"H={args.h}, W={args.w}, K={args.kernel_size}, "
              f"sparsity={args.sparsity}")
        print()

        r = bench_cusparse_layer(
            args.n, args.cin, args.cout, args.h, args.w, args.sparsity,
            kernel_size=args.kernel_size,
            warmup=args.warmup,
            repeats=args.repeats,
            device=device)

        print(f"im2col:     {r['im2col_ms']:.3f} ms")
        print(f"CSR build:  {r['csr_build_ms']:.3f} ms")
        print(f"SpMM:       {r['spmm_ms']:.3f} ms")
        print(f"Total:      {r['total_ms']:.3f} ms")
        print(f"cuDNN:      {r['cudnn_ms']:.3f} ms")
        print(f"Sparsity:   {r['actual_sparsity']*100:.1f}%  "
              f"(nnz={r['nnz']})")
        ratio = r['cudnn_ms'] / r['total_ms'] if r['total_ms'] > 0 else 0
        print(f"cuDNN/Sparse ratio: {ratio:.2f}x")


if __name__ == '__main__':
    main()