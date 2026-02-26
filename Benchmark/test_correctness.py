"""
正确性验证脚本 — 验证所有稀疏算子输出与 PyTorch 原生算子数值一致

覆盖算子: SparseConv2d, SparseLinear, SparseBatchNorm2d

用法:
    cd ~/SparseFlow
    python Benchmark/test_correctness.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from Ops.sparse_conv2d import SparseConv2d
from Ops.sparse_linear import SparseLinear
from Ops.sparse_batchnorm2d import SparseBatchNorm2d


def test_sparse_conv2d():
    """测试 SparseConv2d 的数值正确性"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return

    print("=" * 60)
    print("SparseConv2d Correctness Test")
    print("=" * 60)

    for kernel_size, padding in [(3, 1), (1, 0)]:
        for sparsity in [0.0, 0.5, 0.9, 0.99]:
            C_in, C_out, H, W = 64, 128, 28, 28
            N = 4

            x = torch.randn(N, C_in, H, W, device=device)
            mask = torch.rand(N, C_in, H, W, device=device) > sparsity
            x = x * mask.float()

            conv = nn.Conv2d(C_in, C_out, kernel_size, padding=padding, bias=True).to(device)
            sparse_conv = SparseConv2d.from_dense(conv, block_size=8)

            with torch.no_grad():
                y_dense = F.conv2d(x, conv.weight, conv.bias, padding=padding)
                y_sparse = sparse_conv(x)

            max_diff = (y_dense - y_sparse).abs().max().item()
            rel_diff = max_diff / (y_dense.abs().max().item() + 1e-8)
            passed = max_diff < 1e-3

            status = "PASS ✓" if passed else "FAIL ✗"
            print(f"  [{status}] kernel={kernel_size}x{kernel_size}, "
                  f"sparsity={sparsity:.0%}, max_diff={max_diff:.2e}")
    print()


def test_sparse_linear():
    """测试 SparseLinear 的数值正确性"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return

    print("=" * 60)
    print("SparseLinear Correctness Test")
    print("=" * 60)

    for sparsity in [0.0, 0.5, 0.9, 0.99]:
        N, C_in, C_out = 128, 512, 1000

        x = torch.randn(N, C_in, device=device)
        # 按行稀疏化
        row_mask = torch.rand(N, device=device) > sparsity
        x = x * row_mask.unsqueeze(1).float()

        linear = nn.Linear(C_in, C_out, bias=True).to(device)
        sparse_linear = SparseLinear.from_dense(linear)

        with torch.no_grad():
            y_dense = F.linear(x, linear.weight, linear.bias)
            y_sparse = sparse_linear(x)

        max_diff = (y_dense - y_sparse).abs().max().item()
        rel_diff = max_diff / (y_dense.abs().max().item() + 1e-8)
        passed = max_diff < 1e-2  # Linear 的 atomic 误差可能稍大

        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  [{status}] sparsity={sparsity:.0%}, "
              f"max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}")
    print()


def test_sparse_batchnorm2d():
    """测试 SparseBatchNorm2d 的数值正确性"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return

    print("=" * 60)
    print("SparseBatchNorm2d Correctness Test")
    print("=" * 60)

    for sparsity in [0.0, 0.5, 0.9, 0.99]:
        N, C, H, W = 8, 64, 28, 28

        x = torch.randn(N, C, H, W, device=device)
        # 按空间位置稀疏化（所有通道同时置零）
        spatial_mask = torch.rand(N, 1, H, W, device=device) > sparsity
        x = x * spatial_mask.float()

        bn = nn.BatchNorm2d(C).to(device)
        bn.eval()
        # 需要 running_mean/var 有意义的值
        with torch.no_grad():
            _ = bn(torch.randn(32, C, H, W, device=device))
        bn.eval()

        sparse_bn = SparseBatchNorm2d.from_dense(bn)
        sparse_bn.eval()

        with torch.no_grad():
            y_dense = F.batch_norm(x, bn.running_mean, bn.running_var,
                                   bn.weight, bn.bias, False, 0.0, bn.eps)
            y_sparse = sparse_bn(x)

        max_diff = (y_dense - y_sparse).abs().max().item()
        passed = max_diff < 1e-5

        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  [{status}] sparsity={sparsity:.0%}, max_diff={max_diff:.2e}")
    print()


def test_5d_input():
    """测试 5D 输入 (T, B, C, H, W) 的正确处理"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        return

    print("=" * 60)
    print("5D Input (T,B,C,H,W) Shape Test")
    print("=" * 60)

    T, B, C_in, C_out, H, W = 4, 2, 32, 64, 28, 28

    # Conv2d
    x = torch.randn(T, B, C_in, H, W, device=device) * 0.1
    conv = nn.Conv2d(C_in, C_out, 3, padding=1).to(device)
    sc = SparseConv2d.from_dense(conv, block_size=8)
    with torch.no_grad():
        y = sc(x)
    assert y.shape == (T, B, C_out, H, W)
    print(f"  [PASS ✓] Conv2d: {x.shape} -> {y.shape}")

    # BatchNorm2d
    bn = nn.BatchNorm2d(C_in).to(device)
    bn.eval()
    _ = bn(torch.randn(8, C_in, H, W, device=device))
    bn.eval()
    sbn = SparseBatchNorm2d.from_dense(bn)
    sbn.eval()
    with torch.no_grad():
        y = sbn(x)
    assert y.shape == (T, B, C_in, H, W)
    print(f"  [PASS ✓] BN2d:   {x.shape} -> {y.shape}")

    # Linear (3D)
    C_fc = 512
    x3 = torch.randn(T, B, C_fc, device=device) * 0.1
    lin = nn.Linear(C_fc, 1000).to(device)
    sl = SparseLinear.from_dense(lin)
    with torch.no_grad():
        y = sl(x3)
    assert y.shape == (T, B, 1000)
    print(f"  [PASS ✓] Linear: {x3.shape} -> {y.shape}")
    print()


if __name__ == "__main__":
    test_sparse_conv2d()
    test_sparse_linear()
    test_sparse_batchnorm2d()
    test_5d_input()
    print("All correctness tests completed.")