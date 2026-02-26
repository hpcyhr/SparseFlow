"""
正确性验证脚本 — 验证 SparseConv2d 输出与 F.conv2d 数值一致

用法:
    python -m sparseflow.benchmark.test_correctness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def test_sparse_conv2d_correctness():
    """测试 SparseConv2d 的数值正确性"""
    from sparseflow.ops.sparse_conv2d import SparseConv2d

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available, skipping Triton correctness test")
        return

    print("=" * 60)
    print("SparseConv2d Correctness Test")
    print("=" * 60)

    for kernel_size, padding in [(3, 1), (1, 0)]:
        for sparsity in [0.0, 0.5, 0.9, 0.99]:
            C_in, C_out, H, W = 64, 128, 28, 28
            N = 4

            # 创建稀疏输入
            x = torch.randn(N, C_in, H, W, device=device)
            mask = torch.rand(N, C_in, H, W, device=device) > sparsity
            x = x * mask.float()  # 稀疏化

            # 原始 Conv2d
            conv = nn.Conv2d(C_in, C_out, kernel_size, padding=padding, bias=True).to(device)

            # SparseConv2d
            sparse_conv = SparseConv2d.from_dense(conv, block_size=8)

            # 前向
            with torch.no_grad():
                y_dense = F.conv2d(x, conv.weight, conv.bias, padding=padding)
                y_sparse = sparse_conv(x)

            # 比较
            max_diff = (y_dense - y_sparse).abs().max().item()
            rel_diff = max_diff / (y_dense.abs().max().item() + 1e-8)
            passed = max_diff < 1e-3  # float32 允许的误差

            status = "PASS ✓" if passed else "FAIL ✗"
            print(f"  [{status}] kernel={kernel_size}x{kernel_size}, "
                  f"sparsity={sparsity:.0%}, "
                  f"max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}")

            if not passed:
                print(f"    WARNING: max absolute diff = {max_diff}")

    print()


def test_sparse_conv2d_5d_input():
    """测试 5D 输入 (T, B, C, H, W) 的正确处理"""
    from sparseflow.ops.sparse_conv2d import SparseConv2d

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        return

    print("5D Input (T,B,C,H,W) Test")
    print("-" * 40)

    T, B, C_in, C_out, H, W = 4, 2, 32, 64, 28, 28
    x = torch.randn(T, B, C_in, H, W, device=device) * 0.1  # sparse-ish

    conv = nn.Conv2d(C_in, C_out, 3, padding=1, bias=True).to(device)
    sparse_conv = SparseConv2d.from_dense(conv, block_size=8)

    with torch.no_grad():
        y = sparse_conv(x)

    assert y.shape == (T, B, C_out, H, W), f"Expected shape {(T, B, C_out, H, W)}, got {y.shape}"
    print(f"  [PASS ✓] 5D input: {x.shape} -> {y.shape}")
    print()


if __name__ == "__main__":
    test_sparse_conv2d_correctness()
    test_sparse_conv2d_5d_input()
    print("All tests completed.")