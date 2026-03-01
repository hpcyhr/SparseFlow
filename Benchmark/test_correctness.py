"""
Test: v10 metadata-driven sparse_conv2d_forward vs F.conv2d
"""
import sys
from pathlib import Path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn.functional as F


def test_conv(N, C_IN, C_OUT, H, W, kernel_size, sparsity, device="cuda"):
    from Kernels.conv2d import sparse_conv2d_forward

    x = torch.randn(N, C_IN, H, W, device=device)
    if sparsity > 0:
        x[torch.rand_like(x) < sparsity] = 0.0

    K = kernel_size
    pad = K // 2
    weight = torch.randn(C_OUT, C_IN, K, K, device=device)
    bias = torch.randn(C_OUT, device=device)

    y_ref = F.conv2d(x, weight, bias, stride=1, padding=pad)
    y_sparse, ms = sparse_conv2d_forward(
        x.contiguous(), weight.contiguous(), bias,
        block_size=8, kernel_size=K, threshold=1e-6
    )

    diff = (y_sparse - y_ref).abs()
    max_abs = diff.max().item()
    cos = F.cosine_similarity(
        y_sparse.flatten().unsqueeze(0).float(),
        y_ref.flatten().unsqueeze(0).float()
    ).item()
    return max_abs, cos, ms


if __name__ == "__main__":
    device = "cuda"
    all_pass = True
    T_ABS, T_COS = 0.1, 0.999

    # ── 3×3 ──
    print("=" * 75)
    print("3×3 Conv Tests")
    print("=" * 75)
    for N, CI, CO, H, W, sp in [
        (2, 64, 64, 56, 56, 0.0),
        (2, 64, 64, 56, 56, 0.9),
        (2, 64, 64, 56, 56, 0.99),
        (2, 128, 128, 28, 28, 0.0),
        (2, 128, 128, 28, 28, 0.9),
        (2, 128, 128, 28, 28, 0.99),
        (2, 256, 256, 14, 14, 0.9),
        (4, 64, 128, 28, 28, 0.95),
    ]:
        ma, cos, ms = test_conv(N, CI, CO, H, W, 3, sp, device)
        ok = ma < T_ABS and cos > T_COS
        if not ok: all_pass = False
        s = "✓" if ok else "✗"
        print(f"  {s} N={N} C={CI}→{CO} H={H} sp={sp:.0%}: "
              f"max_abs={ma:.6f} cos={cos:.8f} t={ms:.2f}ms")

    # ── 1×1 ──
    print("\n" + "=" * 75)
    print("1×1 Conv Tests")
    print("=" * 75)
    for N, CI, CO, H, W, sp in [
        (2, 64, 256, 56, 56, 0.0),
        (2, 64, 256, 56, 56, 0.9),
        (2, 256, 64, 56, 56, 0.99),
        (2, 128, 512, 28, 28, 0.9),
    ]:
        ma, cos, ms = test_conv(N, CI, CO, H, W, 1, sp, device)
        ok = ma < T_ABS and cos > T_COS
        if not ok: all_pass = False
        s = "✓" if ok else "✗"
        print(f"  {s} N={N} C={CI}→{CO} H={H} sp={sp:.0%}: "
              f"max_abs={ma:.6f} cos={cos:.8f} t={ms:.2f}ms")

    # ── Edge cases ──
    print("\n" + "=" * 75)
    print("Edge Cases")
    print("=" * 75)

    from Kernels.conv2d import sparse_conv2d_forward

    # No bias
    for K in [3, 1]:
        x = torch.randn(2, 64, 28, 28, device=device)
        x[torch.rand_like(x) < 0.9] = 0.0
        w = torch.randn(64, 64, K, K, device=device)
        yr = F.conv2d(x, w, None, padding=K//2)
        ys, _ = sparse_conv2d_forward(x, w, None, 8, K)
        ma = (ys - yr).abs().max().item()
        cos = F.cosine_similarity(ys.flatten().unsqueeze(0).float(),
                                   yr.flatten().unsqueeze(0).float()).item()
        ok = ma < T_ABS and cos > T_COS
        if not ok: all_pass = False
        print(f"  {'✓' if ok else '✗'} {K}x{K} no_bias: max={ma:.6f} cos={cos:.8f}")

    # All zero
    for K in [3, 1]:
        x = torch.zeros(2, 64, 28, 28, device=device)
        w = torch.randn(64, 64, K, K, device=device)
        b = torch.randn(64, device=device)
        yr = F.conv2d(x, w, b, padding=K//2)
        ys, _ = sparse_conv2d_forward(x, w, b, 8, K)
        ma = (ys - yr).abs().max().item()
        ok = ma < 1e-5
        if not ok: all_pass = False
        print(f"  {'✓' if ok else '✗'} {K}x{K} all_zero: max={ma:.8f}")

    # ── Spike inputs ──
    print("\n" + "=" * 75)
    print("Binary Spike Inputs (SNN)")
    print("=" * 75)
    for CI, CO, H in [(64, 64, 56), (128, 128, 28), (256, 256, 14)]:
        x = torch.bernoulli(torch.full((2, CI, H, H), 0.05, device=device))
        w = torch.randn(CO, CI, 3, 3, device=device)
        b = torch.randn(CO, device=device)
        yr = F.conv2d(x, w, b, padding=1)
        ys, ms = sparse_conv2d_forward(x, w, b, 8, 3)
        ma = (ys - yr).abs().max().item()
        cos = F.cosine_similarity(ys.flatten().unsqueeze(0).float(),
                                   yr.flatten().unsqueeze(0).float()).item()
        sp = (x == 0).float().mean().item()
        ok = ma < T_ABS and cos > T_COS
        if not ok: all_pass = False
        print(f"  {'✓' if ok else '✗'} C={CI}→{CO} H={H} sp={sp:.0%}: "
              f"max={ma:.6f} cos={cos:.8f} t={ms:.2f}ms")

    # ── Speedup scaling ──
    print("\n" + "=" * 75)
    print("Speedup vs cuDNN at various sparsities")
    print("=" * 75)
    for sp in [0.0, 0.5, 0.9, 0.95, 0.99]:
        x = torch.randn(8, 64, 56, 56, device=device)
        if sp > 0:
            x[torch.rand_like(x) < sp] = 0.0
        w = torch.randn(64, 64, 3, 3, device=device)
        b = torch.randn(64, device=device)

        # Warmup
        for _ in range(3):
            _ = F.conv2d(x, w, b, padding=1)
            _ = sparse_conv2d_forward(x, w, b, 8, 3)
        torch.cuda.synchronize()

        # cuDNN
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(10):
            _ = F.conv2d(x, w, b, padding=1)
        e.record()
        torch.cuda.synchronize()
        cudnn_ms = s.elapsed_time(e) / 10

        # Sparse
        total_sp_ms = 0
        for _ in range(10):
            _, ms = sparse_conv2d_forward(x, w, b, 8, 3)
            total_sp_ms += ms
        sp_ms = total_sp_ms / 10

        ratio = cudnn_ms / max(sp_ms, 1e-6)
        print(f"  sp={sp:.0%}: cuDNN={cudnn_ms:.3f}ms sparse={sp_ms:.3f}ms "
              f"speedup={ratio:.2f}x")

    # ── Memory test ──
    print("\n" + "=" * 75)
    print("Memory: T=16 scale (N=512, C=64, H=56)")
    print("=" * 75)
    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)
    x_big = torch.zeros(512, 64, 56, 56, device=device)
    nz = int(0.05 * x_big.numel())
    x_big.view(-1)[torch.randperm(x_big.numel(), device=device)[:nz]] = 1.0
    w_big = torch.randn(64, 64, 3, 3, device=device)
    b_big = torch.randn(64, device=device)
    y, ms = sparse_conv2d_forward(x_big, w_big, b_big, 8, 3)
    mem_after = torch.cuda.max_memory_allocated(device)
    extra_mb = (mem_after - mem_before) / 1024**2
    print(f"  extra_mem={extra_mb:.0f}MB t={ms:.2f}ms")
    del x_big, y
    torch.cuda.empty_cache()

    print("\n" + "=" * 75)
    print("🎉 ALL PASS" if all_pass else "❌ SOME FAILED")
    print("=" * 75)