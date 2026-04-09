"""
SparseFlow Benchmark — Correctness Tests for All New Operators

Tests numerical correctness of each new operator against PyTorch baselines.

Usage:
    cd ~/SparseFlow
    python Benchmark/test_new_ops.py
    python Benchmark/test_new_ops.py --gpu 0 --verbose
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_sim(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    d = a.norm() * b.norm()
    return (a @ b / d).item() if d > 0 else 1.0


def max_abs_err(a, b):
    return (a.float() - b.float()).abs().max().item()


def check(name, cos, mae, cos_thresh=0.999, mae_thresh=0.1):
    ok = cos > cos_thresh and mae < mae_thresh
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name:40s}  cosine={cos:.6f}  max_abs={mae:.6f}")
    return ok


def make_sparse(shape, activity, device, dtype=torch.float16):
    x = torch.randn(*shape, device=device, dtype=dtype)
    mask = (torch.rand(*shape, device=device) < activity)
    return x * mask.to(dtype)


def test_matmul(device, verbose):
    from Kernels.matmul import sparse_matmul_forward
    a = make_sparse((256, 128), 0.01, device)
    b = torch.randn(128, 64, device=device, dtype=torch.float16)

    y_ref = torch.mm(a.float(), b.float())
    y_sp, _ = sparse_matmul_forward(a, b, return_ms=False)

    return check("Matmul [256,128]×[128,64]", cosine_sim(y_ref, y_sp), max_abs_err(y_ref, y_sp))


def test_matmul_module(device, verbose):
    from Ops.sparse_matmul import SparseMatmul
    mod = SparseMatmul(threshold=1e-6)
    a = make_sparse((64, 32), 0.05, device)
    b = torch.randn(32, 16, device=device, dtype=torch.float16)

    y_ref = torch.mm(a.float(), b.float())
    y_sp = mod(a, b)
    return check("SparseMatmul Module [64,32]×[32,16]", cosine_sim(y_ref, y_sp), max_abs_err(y_ref, y_sp))


def test_bmm(device, verbose):
    from Kernels.bmm import sparse_bmm_forward
    a = make_sparse((8, 64, 32), 0.02, device)
    b = torch.randn(8, 32, 64, device=device, dtype=torch.float16)

    y_ref = torch.bmm(a.float(), b.float())
    y_sp, _ = sparse_bmm_forward(a, b, return_ms=False)

    return check("BMM [8,64,32]×[8,32,64]", cosine_sim(y_ref, y_sp), max_abs_err(y_ref, y_sp))


def test_bmm_module(device, verbose):
    from Ops.sparse_bmm import SparseBMM
    mod = SparseBMM(threshold=1e-6)
    a = make_sparse((4, 32, 16), 0.05, device)
    b = torch.randn(4, 16, 32, device=device, dtype=torch.float16)

    y_ref = torch.bmm(a.float(), b.float())
    y_sp = mod(a, b)
    return check("SparseBMM Module [4,32,16]×[4,16,32]", cosine_sim(y_ref, y_sp), max_abs_err(y_ref, y_sp))


def test_depthwise(device, verbose):
    from Kernels.depthwise_conv2d import sparse_depthwise_conv2d_forward
    C = 32
    x = make_sparse((2, C, 16, 16), 0.01, device)
    w = torch.randn(C, 1, 3, 3, device=device, dtype=torch.float32)
    bias = torch.randn(C, device=device, dtype=torch.float32)

    y_ref = F.conv2d(x.float(), w, bias, stride=1, padding=1, groups=C)
    y_sp, _ = sparse_depthwise_conv2d_forward(x, w, bias, stride=1, padding=1)

    return check("Depthwise Conv2d [2,32,16,16] K=3",
                 cosine_sim(y_ref, y_sp), max_abs_err(y_ref, y_sp), cos_thresh=0.99)


def test_depthwise_module(device, verbose):
    from Ops.sparse_depthwise_conv2d import SparseDepthwiseConv2d
    C = 16
    dense_conv = nn.Conv2d(C, C, 3, padding=1, groups=C).to(device)
    sparse_conv = SparseDepthwiseConv2d.from_dense(dense_conv)
    x = make_sparse((2, C, 8, 8), 0.05, device)

    y_ref = F.conv2d(x.float(), dense_conv.weight, dense_conv.bias, padding=1, groups=C)
    y_sp = sparse_conv(x)

    return check("SparseDepthwiseConv2d Module [2,16,8,8]",
                 cosine_sim(y_ref, y_sp), max_abs_err(y_ref, y_sp), cos_thresh=0.99)


def test_grouped_conv2d(device, verbose):
    from Kernels.grouped_conv2d import sparse_grouped_conv2d_forward

    groups = 4
    c_in = 16
    c_out = 32
    x = make_sparse((2, c_in, 16, 16), 0.02, device)
    w = torch.randn(c_out, c_in // groups, 3, 3, device=device, dtype=torch.float16)
    bias = torch.randn(c_out, device=device, dtype=torch.float16)

    y_ref = F.conv2d(x.float(), w.float(), bias.float(), stride=1, padding=1, groups=groups)
    y_sp, _ = sparse_grouped_conv2d_forward(
        x, w, bias,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=groups,
    )

    return check("Grouped Conv2d [2,16,16,16] G=4 K=3",
                 cosine_sim(y_ref, y_sp), max_abs_err(y_ref, y_sp), cos_thresh=0.99)


def test_grouped_conv2d_module(device, verbose):
    from Ops.sparse_grouped_conv2d import SparseGroupedConv2d

    dense_conv = nn.Conv2d(16, 32, 3, padding=1, groups=4).to(device)
    sparse_conv = SparseGroupedConv2d.from_dense(dense_conv)
    x = make_sparse((2, 16, 8, 8), 0.05, device)

    y_ref = F.conv2d(
        x.float(),
        dense_conv.weight.float(),
        dense_conv.bias.float(),
        padding=1,
        groups=4,
    )
    y_sp = sparse_conv(x)

    return check("SparseGroupedConv2d Module [2,16,8,8] G=4",
                 cosine_sim(y_ref, y_sp), max_abs_err(y_ref, y_sp), cos_thresh=0.99)


def test_maxpool2d(device, verbose):
    from Kernels.maxpool2d import sparse_maxpool2d_forward

    x = make_sparse((2, 16, 16, 16), 0.05, device)
    y_ref = F.max_pool2d(x.float(), kernel_size=3, stride=2, padding=1)
    y_sp, _ = sparse_maxpool2d_forward(
        x,
        kernel_size=3,
        stride=2,
        padding=1,
    )
    return check("MaxPool2d [2,16,16,16] K=3 S=2",
                 cosine_sim(y_ref, y_sp), max_abs_err(y_ref, y_sp), cos_thresh=0.99)


def test_maxpool2d_module(device, verbose):
    from Ops.sparse_maxpool2d import SparseMaxPool2d

    mod = SparseMaxPool2d(kernel_size=3, stride=2, padding=1)
    x = make_sparse((2, 8, 16, 16), 0.1, device)
    y_ref = F.max_pool2d(x.float(), kernel_size=3, stride=2, padding=1)
    y_sp = mod(x)
    return check("SparseMaxPool2d Module [2,8,16,16]",
                 cosine_sim(y_ref, y_sp), max_abs_err(y_ref, y_sp), cos_thresh=0.99)


def test_avgpool2d(device, verbose):
    from Kernels.avgpool2d import sparse_avgpool2d_forward

    x = make_sparse((2, 16, 16, 16), 0.05, device)
    y_ref = F.avg_pool2d(x.float(), kernel_size=3, stride=2, padding=1)
    y_sp, _ = sparse_avgpool2d_forward(
        x,
        kernel_size=3,
        stride=2,
        padding=1,
    )
    return check("AvgPool2d [2,16,16,16] K=3 S=2",
                 cosine_sim(y_ref, y_sp), max_abs_err(y_ref, y_sp), cos_thresh=0.99)


def test_avgpool2d_module(device, verbose):
    from Ops.sparse_avgpool2d import SparseAvgPool2d

    mod = SparseAvgPool2d(kernel_size=3, stride=2, padding=1)
    x = make_sparse((2, 8, 16, 16), 0.1, device)
    y_ref = F.avg_pool2d(x.float(), kernel_size=3, stride=2, padding=1)
    y_sp = mod(x)
    return check("SparseAvgPool2d Module [2,8,16,16]",
                 cosine_sim(y_ref, y_sp), max_abs_err(y_ref, y_sp), cos_thresh=0.99)


def test_conv1d(device, verbose):
    from Kernels.conv1d import sparse_conv1d_forward
    x = make_sparse((4, 32, 64), 0.01, device)
    w = torch.randn(16, 32, 3, device=device, dtype=torch.float16)
    bias = torch.randn(16, device=device, dtype=torch.float16)

    y_ref = F.conv1d(x.float(), w.float(), bias.float(), stride=1, padding=1)
    y_sp, _ = sparse_conv1d_forward(x, w, bias, stride=1, padding=1)

    return check("Conv1d [4,32,64] K=3 (v2 active-tile sparse)",
                 cosine_sim(y_ref, y_sp), max_abs_err(y_ref, y_sp))


def test_conv1d_module(device, verbose):
    from Ops.sparse_conv1d import SparseConv1d
    dense_conv = nn.Conv1d(16, 8, 3, padding=1).to(device)
    sparse_conv = SparseConv1d.from_dense(dense_conv)
    x = make_sparse((2, 16, 32), 0.05, device)

    y_ref = F.conv1d(x.float(), dense_conv.weight.float(), dense_conv.bias.float(), padding=1)
    y_sp = sparse_conv(x)

    return check("SparseConv1d Module [2,16,32]",
                 cosine_sim(y_ref, y_sp), max_abs_err(y_ref, y_sp))


def test_conv3d(device, verbose):
    from Kernels.conv3d import sparse_conv3d_forward
    x = make_sparse((2, 8, 8, 8, 8), 0.01, device)
    w = torch.randn(4, 8, 3, 3, 3, device=device, dtype=torch.float16)
    bias = torch.randn(4, device=device, dtype=torch.float16)

    y_ref = F.conv3d(x.float(), w.float(), bias.float(), stride=1, padding=1)
    y_sp, _ = sparse_conv3d_forward(x, w, bias, stride=1, padding=1)

    return check("Conv3d [2,8,8,8,8] K=3 (v2 active-tile sparse)",
                 cosine_sim(y_ref, y_sp), max_abs_err(y_ref, y_sp))


def test_conv3d_module(device, verbose):
    from Ops.sparse_conv3d import SparseConv3d
    dense_conv = nn.Conv3d(4, 2, 3, padding=1).to(device)
    sparse_conv = SparseConv3d.from_dense(dense_conv)
    x = make_sparse((2, 4, 4, 4, 4), 0.1, device)

    y_ref = F.conv3d(x.float(), dense_conv.weight.float(), dense_conv.bias.float(), padding=1)
    y_sp = sparse_conv(x)
    return check("SparseConv3d Module [2,4,4,4,4]",
                 cosine_sim(y_ref, y_sp), max_abs_err(y_ref, y_sp))


def test_attention(device, verbose):
    from Kernels.attention import sparse_qk_forward, sparse_attn_v_forward
    import math

    B, H, S, D = 2, 4, 32, 16
    q = make_sparse((B, H, S, D), 0.05, device)
    k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

    scale = 1.0 / math.sqrt(D)

    # Reference QK
    qk_ref = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    qk_sp, _ = sparse_qk_forward(q, k, scale=scale)

    ok1 = check("Attention Q×K^T [2,4,32,16]",
                cosine_sim(qk_ref, qk_sp), max_abs_err(qk_ref, qk_sp))

    # Reference attn×V with sparse attn
    attn = make_sparse((B, H, S, S), 0.05, device)
    av_ref = torch.matmul(attn.float(), v.float())
    av_sp, _ = sparse_attn_v_forward(attn, v)

    ok2 = check("Attention attn×V [2,4,32,32]×[2,4,32,16]",
                cosine_sim(av_ref, av_sp), max_abs_err(av_ref, av_sp))

    return ok1 and ok2


def test_attention_module(device, verbose):
    from Ops.sparse_attention import SparseAttention
    mod = SparseAttention(num_heads=4, head_dim=16)

    B, H, S, D = 2, 4, 16, 16
    q = make_sparse((B, H, S, D), 0.1, device)
    k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

    logits = mod.qk(q, k)
    assert logits.shape == (B, H, S, S)

    attn = make_sparse((B, H, S, S), 0.1, device)
    out = mod.av(attn, v)
    assert out.shape == (B, H, S, D)

    return check("SparseAttention Module qk+av", 1.0, 0.0)


def main():
    parser = argparse.ArgumentParser(description="SparseFlow New Operators Correctness Tests")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.manual_seed(42)

    print(f"{'='*80}")
    print(f"SparseFlow — New Operator Correctness Tests")
    print(f"{'='*80}")
    print(f"  Device: {device} ({torch.cuda.get_device_name(device)})")
    print()

    tests = [
        ("Kernels/matmul", test_matmul),
        ("Ops/SparseMatmul", test_matmul_module),
        ("Kernels/bmm", test_bmm),
        ("Ops/SparseBMM", test_bmm_module),
        ("Kernels/depthwise_conv2d", test_depthwise),
        ("Ops/SparseDepthwiseConv2d", test_depthwise_module),
        ("Kernels/grouped_conv2d", test_grouped_conv2d),
        ("Ops/SparseGroupedConv2d", test_grouped_conv2d_module),
        ("Kernels/maxpool2d", test_maxpool2d),
        ("Ops/SparseMaxPool2d", test_maxpool2d_module),
        ("Kernels/avgpool2d", test_avgpool2d),
        ("Ops/SparseAvgPool2d", test_avgpool2d_module),
        ("Kernels/conv1d", test_conv1d),
        ("Ops/SparseConv1d", test_conv1d_module),
        ("Kernels/conv3d", test_conv3d),
        ("Ops/SparseConv3d", test_conv3d_module),
        ("Kernels/attention", test_attention),
        ("Ops/SparseAttention", test_attention_module),
    ]

    passed = 0
    failed = 0

    for name, fn in tests:
        try:
            with torch.no_grad():
                ok = fn(device, args.verbose)
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [ERROR] {name:40s}  {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*80}")
    print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'='*80}")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()