"""
Standalone test: verify sparse_conv2d_forward vs F.conv2d
"""
import sys
from pathlib import Path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn.functional as F

# Test the kernel at different sparsity levels
def test_kernel(N, C_IN, C_OUT, H, W, block_size, sparsity, device="cuda"):
    from Kernels.conv2d import sparse_conv2d_forward

    # Create input with controlled sparsity
    x = torch.randn(N, C_IN, H, W, device=device)
    # Zero out elements to achieve target sparsity
    mask = torch.rand_like(x) < sparsity
    x[mask] = 0.0

    # Random weights and bias
    weight = torch.randn(C_OUT, C_IN, 3, 3, device=device)
    bias = torch.randn(C_OUT, device=device)

    # Reference: F.conv2d
    y_ref = F.conv2d(x, weight, bias, stride=1, padding=1)

    # Sparse kernel
    y_sparse, _ = sparse_conv2d_forward(
        x.contiguous(), weight.contiguous(), bias,
        block_size=block_size, kernel_size=3, threshold=0.0
    )

    # Compare
    diff = (y_sparse - y_ref).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    # Cosine similarity
    cos = F.cosine_similarity(
        y_sparse.flatten().unsqueeze(0),
        y_ref.flatten().unsqueeze(0)
    ).item()

    # Check specific location
    if max_abs > 0.001:
        idx = (diff == diff.max()).nonzero()[0]
        n, c, h, w = idx.tolist()
        print(f"  Max error at [{n},{c},{h},{w}]: sparse={y_sparse[n,c,h,w]:.6f} ref={y_ref[n,c,h,w]:.6f}")

    return max_abs, mean_abs, cos


if __name__ == "__main__":
    device = "cuda"

    print("=" * 70)
    print("Test 1: Small tensor, 0% sparsity (all non-zero)")
    print("=" * 70)
    for bs in [4, 8, 16]:
        max_abs, mean_abs, cos = test_kernel(2, 4, 8, 16, 16, bs, sparsity=0.0, device=device)
        print(f"  block={bs:>2}: max_abs={max_abs:.8f}  mean_abs={mean_abs:.8f}  cos={cos:.10f}")

    print()
    print("=" * 70)
    print("Test 2: Medium tensor, 90% sparsity")
    print("=" * 70)
    for bs in [4, 8, 16]:
        max_abs, mean_abs, cos = test_kernel(2, 16, 32, 28, 28, bs, sparsity=0.9, device=device)
        print(f"  block={bs:>2}: max_abs={max_abs:.8f}  mean_abs={mean_abs:.8f}  cos={cos:.10f}")

    print()
    print("=" * 70)
    print("Test 3: Larger tensor, 99% sparsity (like SNN)")
    print("=" * 70)
    for bs in [4, 8, 16]:
        max_abs, mean_abs, cos = test_kernel(2, 64, 64, 56, 56, bs, sparsity=0.99, device=device)
        print(f"  block={bs:>2}: max_abs={max_abs:.8f}  mean_abs={mean_abs:.8f}  cos={cos:.10f}")

    print()
    print("=" * 70)
    print("Test 4: Binary spike input (0 or 1), like real SNN")
    print("=" * 70)
    for bs in [4, 8, 16]:
        x = torch.bernoulli(torch.full((2, 32, 28, 28), 0.05, device=device))
        weight = torch.randn(64, 32, 3, 3, device=device)
        bias = torch.randn(64, device=device)

        y_ref = F.conv2d(x, weight, bias, stride=1, padding=1)

        from Kernels.conv2d import sparse_conv2d_forward
        y_sparse, _ = sparse_conv2d_forward(
            x.contiguous(), weight.contiguous(), bias,
            block_size=bs, kernel_size=3, threshold=0.0
        )

        diff = (y_sparse - y_ref).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        cos = F.cosine_similarity(
            y_sparse.flatten().unsqueeze(0),
            y_ref.flatten().unsqueeze(0)
        ).item()
        print(f"  block={bs:>2}: max_abs={max_abs:.8f}  mean_abs={mean_abs:.8f}  cos={cos:.10f}")

    print()
    print("=" * 70)
    print("Test 5: H not divisible by block_size (edge case)")
    print("=" * 70)
    for H, bs in [(14, 4), (14, 8), (7, 4), (28, 16), (56, 16)]:
        max_abs, mean_abs, cos = test_kernel(2, 16, 32, H, H, bs, sparsity=0.9, device=device)
        print(f"  H={H:>2} block={bs:>2}: max_abs={max_abs:.8f}  mean_abs={mean_abs:.8f}  cos={cos:.10f}")

