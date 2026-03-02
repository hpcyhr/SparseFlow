"""
Test: v10.3 metadata-driven sparse_conv2d_forward
Regression tests for: padding, boundary, weight alignment, T=16 memory.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F

T_ABS, T_COS = 0.1, 0.999


def test_conv(N, CI, CO, H, W, K, sp, device="cuda"):
    from Kernels.conv2d import sparse_conv2d_forward
    x = torch.randn(N, CI, H, W, device=device)
    if sp > 0:
        x[torch.rand_like(x) < sp] = 0.0
    w = torch.randn(CO, CI, K, K, device=device)
    b = torch.randn(CO, device=device)
    yr = F.conv2d(x, w, b, padding=K//2)
    ys, ms = sparse_conv2d_forward(x, w, b, 8, K)
    ma = (ys - yr).abs().max().item()
    cos = F.cosine_similarity(ys.flatten().unsqueeze(0).float(),
                               yr.flatten().unsqueeze(0).float()).item()
    return ma, cos, ms


def suite(label, cfgs, K):
    print(f"\n{'='*72}\n{label}\n{'='*72}")
    ok_all = True
    for N, CI, CO, H, W, sp in cfgs:
        try:
            ma, cos, ms = test_conv(N, CI, CO, H, W, K, sp)
            ok = ma < T_ABS and cos > T_COS
            if not ok: ok_all = False
            print(f"  {'✓' if ok else '✗'} N={N} C={CI}→{CO} H={H} sp={sp:.0%}: "
                  f"max={ma:.6f} cos={cos:.8f} t={ms:.2f}ms")
        except Exception as e:
            ok_all = False
            print(f"  ✗ N={N} C={CI}→{CO} H={H} sp={sp:.0%}: {type(e).__name__}: {e}")
    return ok_all


if __name__ == "__main__":
    from Kernels.conv2d import sparse_conv2d_forward
    device = "cuda"
    ok = True

    # ── Standard layers ──
    ok &= suite("3×3 Conv", [
        (2, 64, 64, 56, 56, 0.0),
        (2, 64, 64, 56, 56, 0.9),
        (2, 64, 64, 56, 56, 0.99),
        (2, 128, 128, 28, 28, 0.0),
        (2, 128, 128, 28, 28, 0.9),
        (2, 256, 256, 14, 14, 0.9),
        (4, 64, 128, 28, 28, 0.95),
    ], 3)

    ok &= suite("1×1 Conv", [
        (2, 64, 256, 56, 56, 0.0),
        (2, 64, 256, 56, 56, 0.9),
        (2, 256, 64, 56, 56, 0.99),
        (2, 128, 512, 28, 28, 0.9),
    ], 1)

    # ── Bug 1 regression: padding with channel 0 ──
    ok &= suite("Bug1: C_IN not multiple of BLOCK_K=16", [
        (1, 3, 16, 16, 16, 0.0),
        (1, 5, 32, 16, 16, 0.5),
        (1, 17, 32, 28, 28, 0.5),
        (1, 1, 16, 8, 8, 0.0),
    ], 3)

    # ── Bug 2 regression: boundary segfault ──
    ok &= suite("Bug2: Boundary pixels (in_h < 0)", [
        (1, 16, 16, 8, 8, 0.0),
        (1, 16, 16, 9, 9, 0.0),
        (1, 32, 32, 7, 7, 0.5),
        (1, 64, 64, 14, 14, 0.9),
    ], 3)

    # ── Bug 3 regression: weight layout ──
    ok &= suite("Bug3: Weight alignment (odd C_OUT)", [
        (1, 32, 17, 16, 16, 0.5),
        (1, 64, 33, 28, 28, 0.9),
    ], 3)

    # ── Edge cases ──
    print(f"\n{'='*72}\nEdge Cases\n{'='*72}")
    for K in [3, 1]:
        x = torch.randn(2, 64, 28, 28, device=device)
        x[torch.rand_like(x) < 0.9] = 0.0
        w = torch.randn(64, 64, K, K, device=device)
        yr = F.conv2d(x, w, None, padding=K//2)
        ys, _ = sparse_conv2d_forward(x, w, None, 8, K)
        ma = (ys-yr).abs().max().item()
        cos = F.cosine_similarity(ys.flatten().unsqueeze(0).float(),
                                   yr.flatten().unsqueeze(0).float()).item()
        o = ma < T_ABS and cos > T_COS
        if not o: ok = False
        print(f"  {'✓' if o else '✗'} {K}x{K} no_bias: max={ma:.6f} cos={cos:.8f}")

    for K in [3, 1]:
        x = torch.zeros(2, 64, 28, 28, device=device)
        w = torch.randn(64, 64, K, K, device=device)
        b = torch.randn(64, device=device)
        yr = F.conv2d(x, w, b, padding=K//2)
        ys, _ = sparse_conv2d_forward(x, w, b, 8, K)
        ma = (ys-yr).abs().max().item()
        o = ma < 1e-5
        if not o: ok = False
        print(f"  {'✓' if o else '✗'} {K}x{K} all_zero: max={ma:.8f}")

    # ── SNN spike inputs ──
    print(f"\n{'='*72}\nBinary Spike Inputs\n{'='*72}")
    for CI, CO, H in [(64, 64, 56), (128, 128, 28), (256, 256, 14)]:
        x = torch.bernoulli(torch.full((2, CI, H, H), 0.05, device=device))
        w = torch.randn(CO, CI, 3, 3, device=device)
        b = torch.randn(CO, device=device)
        yr = F.conv2d(x, w, b, padding=1)
        ys, ms = sparse_conv2d_forward(x, w, b, 8, 3)
        ma = (ys-yr).abs().max().item()
        cos = F.cosine_similarity(ys.flatten().unsqueeze(0).float(),
                                   yr.flatten().unsqueeze(0).float()).item()
        sp = (x==0).float().mean().item()
        o = ma < T_ABS and cos > T_COS
        if not o: ok = False
        print(f"  {'✓' if o else '✗'} C={CI}→{CO} H={H} sp={sp:.0%}: "
              f"max={ma:.6f} cos={cos:.8f} t={ms:.2f}ms")

    # ── Bug 4 regression: T=16 scale memory test ──
    print(f"\n{'='*72}\nBug4: T=16 Memory (N=512, C=64, H=56)\n{'='*72}")
    torch.cuda.reset_peak_memory_stats(device)
    mem0 = torch.cuda.memory_allocated(device)
    x_big = torch.zeros(512, 64, 56, 56, device=device)
    nz = int(0.05 * x_big.numel())
    x_big.view(-1)[torch.randperm(x_big.numel(), device=device)[:nz]] = 1.0
    w_big = torch.randn(64, 64, 3, 3, device=device)
    b_big = torch.randn(64, device=device)
    try:
        y_big, ms = sparse_conv2d_forward(x_big, w_big, b_big, 8, 3)
        mem1 = torch.cuda.max_memory_allocated(device)
        extra = (mem1 - mem0) / 1024**2
        print(f"  ✓ Completed: extra_mem={extra:.0f}MB t={ms:.2f}ms")
    except Exception as e:
        ok = False
        print(f"  ✗ FAILED: {type(e).__name__}: {e}")
    del x_big
    torch.cuda.empty_cache()

    print(f"\n{'='*72}")
    print("🎉 ALL PASS" if ok else "❌ SOME FAILED")
    print('='*72)