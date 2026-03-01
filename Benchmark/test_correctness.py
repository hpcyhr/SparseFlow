"""
Standalone test: verify v4 sparse_conv2d_forward (fp16 tl.dot) vs F.conv2d (fp32)
"""
import sys
from pathlib import Path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn.functional as F


def test_conv3x3(N, C_IN, C_OUT, H, W, sparsity, device="cuda"):
    from Kernels.conv2d import sparse_conv2d_forward

    x = torch.randn(N, C_IN, H, W, device=device)
    mask = torch.rand_like(x) < sparsity
    x[mask] = 0.0
    weight = torch.randn(C_OUT, C_IN, 3, 3, device=device)
    bias = torch.randn(C_OUT, device=device)

    y_ref = F.conv2d(x, weight, bias, stride=1, padding=1)

    y_sparse, ms = sparse_conv2d_forward(
        x.contiguous(), weight.contiguous(), bias,
        block_size=8, kernel_size=3, threshold=1e-6
    )

    diff = (y_sparse - y_ref).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    cos = F.cosine_similarity(
        y_sparse.flatten().unsqueeze(0),
        y_ref.flatten().unsqueeze(0)
    ).item()
    return max_abs, mean_abs, cos, ms


def test_conv1x1(N, C_IN, C_OUT, H, W, sparsity, device="cuda"):
    from Kernels.conv2d import sparse_conv2d_forward

    x = torch.randn(N, C_IN, H, W, device=device)
    mask = torch.rand_like(x) < sparsity
    x[mask] = 0.0
    weight = torch.randn(C_OUT, C_IN, 1, 1, device=device)
    bias = torch.randn(C_OUT, device=device)

    y_ref = F.conv2d(x, weight, bias, stride=1, padding=0)

    y_sparse, ms = sparse_conv2d_forward(
        x.contiguous(), weight.contiguous(), bias,
        block_size=8, kernel_size=1, threshold=1e-6
    )

    diff = (y_sparse - y_ref).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    cos = F.cosine_similarity(
        y_sparse.flatten().unsqueeze(0),
        y_ref.flatten().unsqueeze(0)
    ).item()
    return max_abs, mean_abs, cos, ms


if __name__ == "__main__":
    device = "cuda"

    print("=" * 70)
    print("3x3 Conv Tests (fp16 kernel vs fp32 cuDNN)")
    print("=" * 70)

    configs = [
        # (N, C_IN, C_OUT, H, W, sparsity)
        (2, 64, 64, 56, 56, 0.0),
        (2, 64, 64, 56, 56, 0.9),
        (2, 64, 64, 56, 56, 0.99),
        (2, 128, 128, 28, 28, 0.9),
        (2, 128, 128, 28, 28, 0.99),
        (2, 256, 256, 14, 14, 0.9),
        (2, 256, 256, 14, 14, 0.99),
        (2, 32, 64, 56, 56, 0.95),
        (4, 64, 128, 28, 28, 0.95),
    ]

    for N, C_IN, C_OUT, H, W, sp in configs:
        max_abs, mean_abs, cos, ms = test_conv3x3(N, C_IN, C_OUT, H, W, sp, device)
        status = "✓" if max_abs < 0.05 and cos > 0.999 else "✗"
        print(f"  {status} 3x3 N={N} C={C_IN}→{C_OUT} H={H} sp={sp:.0%}: "
              f"max_abs={max_abs:.6f} cos={cos:.8f} time={ms:.2f}ms")

    print()
    print("=" * 70)
    print("1x1 Conv Tests")
    print("=" * 70)

    configs_1x1 = [
        (2, 64, 256, 56, 56, 0.0),
        (2, 64, 256, 56, 56, 0.9),
        (2, 256, 64, 56, 56, 0.99),
        (2, 128, 512, 28, 28, 0.9),
        (2, 512, 128, 28, 28, 0.99),
    ]

    for N, C_IN, C_OUT, H, W, sp in configs_1x1:
        max_abs, mean_abs, cos, ms = test_conv1x1(N, C_IN, C_OUT, H, W, sp, device)
        status = "✓" if max_abs < 0.05 and cos > 0.999 else "✗"
        print(f"  {status} 1x1 N={N} C={C_IN}→{C_OUT} H={H} sp={sp:.0%}: "
              f"max_abs={max_abs:.6f} cos={cos:.8f} time={ms:.2f}ms")

    print()
    print("=" * 70)
    print("Binary spike input (like real SNN)")
    print("=" * 70)
    for C_IN, C_OUT, H in [(64, 64, 56), (128, 128, 28), (256, 256, 14)]:
        x = torch.bernoulli(torch.full((2, C_IN, H, H), 0.05, device=device))
        weight = torch.randn(C_OUT, C_IN, 3, 3, device=device)
        bias = torch.randn(C_OUT, device=device)
        y_ref = F.conv2d(x, weight, bias, stride=1, padding=1)
        from Kernels.conv2d import sparse_conv2d_forward
        y_sparse, ms = sparse_conv2d_forward(
            x.contiguous(), weight.contiguous(), bias,
            block_size=8, kernel_size=3, threshold=1e-6
        )
        diff = (y_sparse - y_ref).abs()
        max_abs = diff.max().item()
        cos = F.cosine_similarity(y_sparse.flatten().unsqueeze(0), y_ref.flatten().unsqueeze(0)).item()
        status = "✓" if max_abs < 0.05 and cos > 0.999 else "✗"
        print(f"  {status} spike 3x3 C={C_IN}→{C_OUT} H={H}: "
              f"max_abs={max_abs:.6f} cos={cos:.8f} time={ms:.2f}ms")