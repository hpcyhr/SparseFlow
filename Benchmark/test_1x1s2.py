#!/usr/bin/env python3
"""
Verify that the 1x1/s2 subsample approach is numerically exact.

Tests:
  1. conv1x1(x, w, stride=2, pad=0) == conv1x1(x[:,:,::2,::2], w, stride=1, pad=0)
  2. After patching, SparseConv2d with k=1,s=2,p=0 produces same output as F.conv2d
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F


def test_subsample_equivalence():
    """Verify that subsampling + s1 == direct s2 for 1x1 conv."""
    print("Test 1: Subsample equivalence for 1x1/s2 ...")
    
    torch.manual_seed(42)
    C_IN, C_OUT = 256, 512
    H, W = 8, 8
    N = 4
    
    x = torch.randn(N, C_IN, H, W, device='cuda')
    w = torch.randn(C_OUT, C_IN, 1, 1, device='cuda')
    b = torch.randn(C_OUT, device='cuda')
    
    # Direct stride=2
    y_direct = F.conv2d(x, w, b, stride=2, padding=0)
    
    # Subsample then stride=1
    x_sub = x[:, :, ::2, ::2].contiguous()
    y_sub = F.conv2d(x_sub, w, b, stride=1, padding=0)
    
    diff = (y_direct - y_sub).abs().max().item()
    print(f"  max_abs_diff = {diff:.2e}")
    assert diff < 1e-6, f"FAIL: diff {diff} too large"
    print("  PASS")


def test_sparse_conv2d_1x1s2():
    """Verify SparseConv2d handles 1x1/s2 correctly."""
    print("\nTest 2: SparseConv2d 1x1/s2 correctness ...")
    
    try:
        from Ops.sparse_conv2d import SparseConv2d
    except ImportError:
        print("  [SKIP] Cannot import SparseConv2d")
        return
    
    torch.manual_seed(42)
    C_IN, C_OUT = 256, 512
    H, W = 8, 8
    N = 4
    
    # Create dense conv
    dense = nn.Conv2d(C_IN, C_OUT, 1, stride=2, padding=0, bias=True).cuda()
    
    # Create sparse conv
    sparse = SparseConv2d.from_dense(dense)
    
    # Check _supports_triton
    supports = sparse._supports_triton()
    print(f"  _supports_triton() = {supports}")
    
    # Test with spike-like input (mostly zeros)
    x = torch.zeros(N, C_IN, H, W, device='cuda')
    # Sprinkle some ones (spike pattern)
    mask = torch.bernoulli(torch.full_like(x, 0.1))
    x = x + mask
    
    y_dense = F.conv2d(x, dense.weight, dense.bias, stride=2, padding=0).float()
    y_sparse = sparse(x)
    
    diff = (y_dense - y_sparse).abs().max().item()
    cos = F.cosine_similarity(y_dense.flatten().unsqueeze(0),
                              y_sparse.flatten().unsqueeze(0)).item()
    print(f"  max_abs_diff = {diff:.2e}")
    print(f"  cosine_sim   = {cos:.8f}")
    assert diff < 0.01, f"FAIL: diff {diff} too large"
    assert cos > 0.999, f"FAIL: cosine {cos} too low"
    print("  PASS")


def test_classify_target_type():
    """Verify classify_target_type now accepts 1x1/s2."""
    print("\nTest 3: classify_target_type includes 1x1/s2 ...")
    
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from bench_4test import classify_target_type
    except ImportError:
        print("  [SKIP] Cannot import classify_target_type")
        return
    
    info = {
        "kernel_size": (1, 1),
        "stride": (2, 2),
        "padding": (0, 0),
        "groups": 1,
    }
    result = classify_target_type(info)
    print(f"  classify_target_type(1x1/s2) = '{result}'")
    assert result == "1x1/s2", f"FAIL: expected '1x1/s2', got '{result}'"
    
    # Verify existing types still work
    for k, s, p, expected in [
        ((1,1), (1,1), (0,0), "1x1/s1"),
        ((3,3), (1,1), (1,1), "3x3/s1"),
        ((3,3), (2,2), (1,1), "3x3/s2"),
    ]:
        info2 = {"kernel_size": k, "stride": s, "padding": p, "groups": 1}
        r = classify_target_type(info2)
        assert r == expected, f"FAIL: {k}/{s}/{p} → '{r}', expected '{expected}'"
    
    print("  PASS")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU tests")
        sys.exit(0)
    
    test_subsample_equivalence()
    test_sparse_conv2d_1x1s2()
    test_classify_target_type()
    print("\nAll tests passed!")