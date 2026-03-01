"""Test v7 sparse_conv2d_forward correctness + timing vs cuDNN."""
import sys, time
from pathlib import Path
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

import torch, torch.nn.functional as F
from Kernels.conv2d import sparse_conv2d_forward

def test(N, C_IN, C_OUT, H, W, K, sparsity, device="cuda"):
    x = torch.randn(N, C_IN, H, W, device=device)
    if sparsity > 0:
        x[torch.rand_like(x) < sparsity] = 0.0
    w = torch.randn(C_OUT, C_IN, K, K, device=device)
    b = torch.randn(C_OUT, device=device)
    pad = (K - 1) // 2

    y_ref = F.conv2d(x, w, b, padding=pad)
    y_sp, ms = sparse_conv2d_forward(x, w, b, block_size=8, kernel_size=K)

    diff = (y_sp - y_ref).abs()
    max_abs = diff.max().item()
    cos = F.cosine_similarity(y_sp.flatten()[None], y_ref.flatten()[None]).item()

    # cuDNN timing
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        _ = F.conv2d(x, w, b, padding=pad)
    torch.cuda.synchronize()
    cudnn_ms = (time.perf_counter() - t0) / 10 * 1000

    ok = max_abs < 0.1 and cos > 0.999
    tag = "✓" if ok else "✗"
    sp_str = f"sp={sparsity:.0%}" if sparsity > 0 else "dense"
    ratio = cudnn_ms / ms if ms > 0 else float('inf')
    print(f"  {tag} {K}x{K} N={N} C={C_IN:>3}→{C_OUT:>3} H={H:>2} {sp_str:>7}: "
          f"err={max_abs:.6f} cos={cos:.6f} sparse={ms:.2f}ms cudnn={cudnn_ms:.2f}ms {ratio:.2f}x")
    return ok

if __name__ == "__main__":
    dev = "cuda"
    print("=" * 85)
    print("  SparseFlow v7 Correctness + Performance Tests")
    print("=" * 85)

    all_ok = True
    # 3x3 tests
    for sp in [0.0, 0.9, 0.95, 0.99]:
        for (N,CI,CO,H) in [(2,64,64,56), (2,128,128,28), (2,256,256,14), (4,64,128,32)]:
            all_ok &= test(N, CI, CO, H, H, 3, sp, dev)

    print()
    # 1x1 tests
    for sp in [0.0, 0.9, 0.99]:
        for (N,CI,CO,H) in [(2,64,256,56), (2,256,128,28), (2,512,256,14)]:
            all_ok &= test(N, CI, CO, H, H, 1, sp, dev)

    print()
    # Binary spike tests (like real SNN, ~5% firing rate)
    print("Binary spike (5% firing rate):")
    for (CI,CO,H) in [(64,64,56), (128,128,28), (256,256,14)]:
        x = torch.bernoulli(torch.full((2, CI, H, H), 0.05, device=dev))
        w = torch.randn(CO, CI, 3, 3, device=dev)
        b = torch.randn(CO, device=dev)
        y_ref = F.conv2d(x, w, b, padding=1)
        y_sp, ms = sparse_conv2d_forward(x, w, b, block_size=8, kernel_size=3)
        err = (y_sp - y_ref).abs().max().item()
        cos = F.cosine_similarity(y_sp.flatten()[None], y_ref.flatten()[None]).item()
        ok = err < 0.1 and cos > 0.999
        all_ok &= ok
        tag = "✓" if ok else "✗"
        print(f"  {tag} C={CI}→{CO} H={H}: err={err:.6f} cos={cos:.6f} time={ms:.2f}ms")

    print()
    print(f"{'ALL PASS' if all_ok else 'SOME FAILED'}")