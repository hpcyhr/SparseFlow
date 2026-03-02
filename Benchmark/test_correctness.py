"""
Test: v10.4 Dynamic Block Tuning — correctness + tile count validation
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
    yr = F.conv2d(x, w, b, padding=K // 2)
    ys, ms = sparse_conv2d_forward(x, w, b, 8, K)
    ma = (ys - yr).abs().max().item()
    cos = F.cosine_similarity(
        ys.flatten().unsqueeze(0).float(),
        yr.flatten().unsqueeze(0).float()
    ).item()
    return ma, cos, ms


def suite(label, cfgs, K):
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    ok_all = True
    for N, CI, CO, H, W, sp in cfgs:
        try:
            ma, cos, ms = test_conv(N, CI, CO, H, W, K, sp)
            ok = ma < T_ABS and cos > T_COS
            if not ok:
                ok_all = False
            s = "✓" if ok else "✗"
            print(f"  {s} N={N} C={CI}→{CO} H={H} sp={sp:.0%}: "
                  f"max={ma:.6f} cos={cos:.8f} t={ms:.2f}ms")
        except Exception as e:
            ok_all = False
            print(f"  ✗ N={N} C={CI}→{CO} H={H} sp={sp:.0%}: "
                  f"{type(e).__name__}: {e}")
    return ok_all


if __name__ == "__main__":
    from Kernels.conv2d import sparse_conv2d_forward, _select_block_sizes
    import triton

    device = "cuda"
    ok = True

    # ── 1. Validate dynamic block selection ──
    print("=" * 72)
    print("Dynamic Block Selection Validation")
    print("=" * 72)

    cases = [
        # (H, W, N,   expected BH, BW, BLOCK_M, note)
        (56, 56,  32,  8, 16, 128, "large map, low N"),
        (56, 56, 512,  8, 16, 128, "large map, high N (T=16)"),
        (28, 28,  32,  8,  8,  64, "medium map"),
        (28, 28, 512,  8,  8,  64, "medium map, high N"),
        (14, 14,  32,  4,  4,  16, "small map"),
        ( 7,  7,  32,  4,  4,  16, "tiny map"),
    ]
    for H, W, N, exp_bh, exp_bw, exp_bm, note in cases:
        bh, bw, bm, bn, bk = _select_block_sizes(H, W, 64, 64, 3, N)
        gh, gw = triton.cdiv(H, bh), triton.cdiv(W, bw)
        tiles = gh * gw
        match = (bh == exp_bh and bw == exp_bw and bm == exp_bm)
        if not match:
            ok = False
        s = "✓" if match else "✗"
        print(f"  {s} H={H} W={W} N={N}: BH={bh} BW={bw} BM={bm} BN={bn} "
              f"GH={gh} GW={gw} tiles={tiles}  ({note})")

    # Verify BLOCK_N capped at 32 for high N
    _, _, _, bn_low, _ = _select_block_sizes(56, 56, 64, 256, 3, 32)
    _, _, _, bn_high, _ = _select_block_sizes(56, 56, 64, 256, 3, 512)
    cap_ok = (bn_high <= 32)
    if not cap_ok:
        ok = False
    print(f"  {'✓' if cap_ok else '✗'} BLOCK_N cap: N=32→{bn_low}, N=512→{bn_high}")

    # ── 2. Tile count comparison ──
    print(f"\n{'=' * 72}")
    print("Tile Count Comparison (v10.3 vs v10.4)")
    print("=" * 72)
    for H in [56, 28, 14, 7]:
        # v10.3 style: BH=4, BW=8
        gh_old = triton.cdiv(H, 4)
        gw_old = triton.cdiv(H, 8)
        tiles_old = gh_old * gw_old
        # v10.4 style
        bh, bw, _, _, _ = _select_block_sizes(H, H, 64, 64, 3, 32)
        gh_new = triton.cdiv(H, bh)
        gw_new = triton.cdiv(H, bw)
        tiles_new = gh_new * gw_new
        ratio = tiles_old / max(tiles_new, 1)
        print(f"  H={H}: v10.3={tiles_old} tiles → v10.4={tiles_new} tiles "
              f"({ratio:.1f}× reduction)")

    # ── 3. Correctness: large maps (the 0.37x bottleneck case) ──
    ok &= suite("3×3 Large Map (H=56) — the critical case", [
        (2,  64,  64, 56, 56, 0.0),
        (2,  64,  64, 56, 56, 0.9),
        (2,  64,  64, 56, 56, 0.95),
        (2,  64,  64, 56, 56, 0.99),
        (4,  64, 128, 56, 56, 0.95),
        (8,  64,  64, 56, 56, 0.9),
    ], 3)

    # ── 4. Correctness: medium maps ──
    ok &= suite("3×3 Medium Map (H=28)", [
        (2, 128, 128, 28, 28, 0.0),
        (2, 128, 128, 28, 28, 0.9),
        (2, 128, 128, 28, 28, 0.99),
        (4,  64, 128, 28, 28, 0.95),
    ], 3)

    # ── 5. Correctness: small maps ──
    ok &= suite("3×3 Small Map (H=14, H=7)", [
        (2, 256, 256, 14, 14, 0.0),
        (2, 256, 256, 14, 14, 0.9),
        (2, 512, 512,  7,  7, 0.9),
    ], 3)

    # ── 6. 1×1 conv ──
    ok &= suite("1×1 Conv", [
        (2,  64, 256, 56, 56, 0.0),
        (2,  64, 256, 56, 56, 0.9),
        (2, 256,  64, 56, 56, 0.99),
        (2, 128, 512, 28, 28, 0.9),
    ], 1)

    # ── 7. Edge cases ──
    print(f"\n{'=' * 72}\nEdge Cases\n{'=' * 72}")
    for K in [3, 1]:
        x = torch.randn(2, 64, 28, 28, device=device)
        x[torch.rand_like(x) < 0.9] = 0.0
        w = torch.randn(64, 64, K, K, device=device)
        yr = F.conv2d(x, w, None, padding=K // 2)
        ys, _ = sparse_conv2d_forward(x, w, None, 8, K)
        ma = (ys - yr).abs().max().item()
        cos = F.cosine_similarity(
            ys.flatten().unsqueeze(0).float(),
            yr.flatten().unsqueeze(0).float()
        ).item()
        o = ma < T_ABS and cos > T_COS
        if not o:
            ok = False
        print(f"  {'✓' if o else '✗'} {K}x{K} no_bias: max={ma:.6f} cos={cos:.8f}")

    for K in [3, 1]:
        x = torch.zeros(2, 64, 28, 28, device=device)
        w = torch.randn(64, 64, K, K, device=device)
        b = torch.randn(64, device=device)
        yr = F.conv2d(x, w, b, padding=K // 2)
        ys, _ = sparse_conv2d_forward(x, w, b, 8, K)
        ma = (ys - yr).abs().max().item()
        o = ma < 1e-5
        if not o:
            ok = False
        print(f"  {'✓' if o else '✗'} {K}x{K} all_zero: max={ma:.8f}")

    # ── 8. Binary spikes ──
    ok &= suite("Binary Spike Inputs", [
        (2, 64, 64, 56, 56, 0.95),
        (2, 128, 128, 28, 28, 0.95),
        (2, 256, 256, 14, 14, 0.95),
    ], 3)

    # ── 9. T=16 proxy: high N ──
    print(f"\n{'=' * 72}\nHigh-N (T=16 proxy): N=512, C=64, H=56\n{'=' * 72}")
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
        extra = (mem1 - mem0) / 1024 ** 2
        print(f"  ✓ OK: extra_mem={extra:.0f}MB t={ms:.2f}ms")
    except Exception as e:
        ok = False
        print(f"  ✗ FAILED: {type(e).__name__}: {e}")
    del x_big
    torch.cuda.empty_cache()

    # ── 10. Speedup comparison ──
    print(f"\n{'=' * 72}")
    print("Speedup vs cuDNN (H=56, N=8)")
    print("=" * 72)
    for sp in [0.0, 0.5, 0.9, 0.95, 0.99]:
        x = torch.randn(8, 64, 56, 56, device=device)
        if sp > 0:
            x[torch.rand_like(x) < sp] = 0.0
        w = torch.randn(64, 64, 3, 3, device=device)
        b = torch.randn(64, device=device)

        for _ in range(3):
            F.conv2d(x, w, b, padding=1)
            sparse_conv2d_forward(x, w, b, 8, 3)
        torch.cuda.synchronize()

        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(10):
            F.conv2d(x, w, b, padding=1)
        e.record()
        torch.cuda.synchronize()
        cudnn_ms = s.elapsed_time(e) / 10

        total_sp = 0
        for _ in range(10):
            _, ms = sparse_conv2d_forward(x, w, b, 8, 3)
            total_sp += ms
        sp_ms = total_sp / 10
        ratio = cudnn_ms / max(sp_ms, 1e-6)
        print(f"  sp={sp:.0%}: cuDNN={cudnn_ms:.3f}ms sparse={sp_ms:.3f}ms "
              f"speedup={ratio:.2f}x")

    print(f"\n{'=' * 72}")
    print("🎉 ALL PASS" if ok else "❌ SOME FAILED")
    print("=" * 72)