#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_sparse_linear.py (fixed)
- auto-detects SparseLinear cached transposed weight accessor:
  _get_w_t / _get_weight_t / _w_t
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", type=str, required=True)
    p.add_argument("--in_features", type=int, default=16)
    p.add_argument("--out_features", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--activity", type=float, default=0.25)
    p.add_argument("--mode", type=str, default="manual",
                   choices=["bernoulli_mask", "all_ones_small", "single_hot", "manual"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dump_tensors", action="store_true")
    p.add_argument("--profile_runtime", action="store_true")
    p.add_argument("--force_dense_fallback", action="store_true")
    p.add_argument("--direct_kernel", action="store_true")
    p.add_argument("--no_bias", action="store_true")
    return p.parse_args()

def import_sparseflow(project_root: str):
    project_root = str(Path(project_root).resolve())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from Ops.sparse_linear import SparseLinear
    from Kernels.linear import sparse_linear_forward
    return SparseLinear, sparse_linear_forward

def make_dtype(name: str):
    return torch.float16 if name == "fp16" else torch.float32

def make_input(B, Cin, mode, activity, device, dtype):
    if mode == "bernoulli_mask":
        x = torch.randn(B, Cin, device=device, dtype=dtype)
        mask = (torch.rand(B, Cin, device=device) < activity)
        return x * mask.to(dtype)
    elif mode == "all_ones_small":
        return torch.ones(B, Cin, device=device, dtype=dtype)
    elif mode == "single_hot":
        x = torch.zeros(B, Cin, device=device, dtype=dtype)
        for b in range(B):
            x[b, b % Cin] = 1.0
        return x
    elif mode == "manual":
        x = torch.zeros(B, Cin, device=device, dtype=dtype)
        for b in range(B):
            for k in range(Cin):
                if (b + k) % 5 == 0:
                    x[b, k] = (k % 7 + 1) * 0.1
        return x
    raise ValueError(mode)

def cosine(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    denom = a.norm() * b.norm()
    if float(denom.item()) == 0.0:
        return 1.0 if float((a - b).abs().max().item()) == 0.0 else 0.0
    return float(torch.dot(a, b).item() / denom.item())

def max_abs_diff(a, b):
    return float((a.float() - b.float()).abs().max().item())

def maybe_print_matrix(name, x, max_rows=8, max_cols=32):
    x_cpu = x.detach().cpu()
    if x_cpu.dim() == 1:
        print(f"{name}: {x_cpu[:max_cols]}")
        return
    rows = min(max_rows, x_cpu.shape[0])
    cols = min(max_cols, x_cpu.shape[1])
    print(f"{name} (first {rows}x{cols}):")
    print(x_cpu[:rows, :cols])

def active_channels_per_row(x):
    rows = []
    x_cpu = x.detach().cpu()
    for b in range(x_cpu.shape[0]):
        idx = torch.nonzero(x_cpu[b] != 0, as_tuple=False).flatten().tolist()
        rows.append(idx)
    return rows

def get_w_t_from_module(slinear):
    if hasattr(slinear, "_get_w_t") and callable(getattr(slinear, "_get_w_t")):
        return slinear._get_w_t()
    if hasattr(slinear, "_get_weight_t") and callable(getattr(slinear, "_get_weight_t")):
        return slinear._get_weight_t()
    if hasattr(slinear, "_w_t"):
        if slinear._w_t is None:
            if hasattr(slinear, "weight"):
                slinear._w_t = slinear.weight.data.half().t().contiguous()
            else:
                raise AttributeError("SparseLinear has _w_t but no weight.")
        return slinear._w_t
    raise AttributeError("SparseLinear has no known cached W_T accessor (_get_w_t/_get_weight_t/_w_t).")

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    device = args.device
    dtype = make_dtype(args.dtype)

    SparseLinear, sparse_linear_forward = import_sparseflow(args.project_root)

    B, Cin, Cout = args.batch_size, args.in_features, args.out_features
    dense = nn.Linear(Cin, Cout, bias=not args.no_bias).to(device=device, dtype=dtype).eval()
    x = make_input(B, Cin, args.mode, args.activity, device, dtype)

    slinear = SparseLinear.from_dense(
        dense,
        threshold=1e-6,
        dense_threshold=0.85,
        warmup_steps=1,
        calib_every=1,
        ema_decay=0.9,
        zero_streak_needed=2,
        profile_runtime=args.profile_runtime,
    ).to(device).eval()

    if args.force_dense_fallback and hasattr(slinear, "_supports_sparse_kernel"):
        slinear._supports_sparse_kernel = False

    with torch.no_grad():
        y_dense = dense(x)
        y_sparse = slinear(x)

    print("=" * 100)
    print("SparseLinear minimal debug case (fixed)")
    print("=" * 100)
    print(f"B={B} Cin={Cin} Cout={Cout} dtype={dtype} device={device} mode={args.mode} activity≈{args.activity:.4f}")

    print("\n[Active channels per row]")
    for b, idx in enumerate(active_channels_per_row(x)):
        print(f"  row {b}: {idx}")

    print("\n[Correctness]")
    print(f"  cosine similarity: {cosine(y_dense, y_sparse):.8f}")
    print(f"  max abs diff:      {max_abs_diff(y_dense, y_sparse):.8f}")

    if args.dump_tensors:
        print("\n[Input / outputs]")
        maybe_print_matrix("x", x)
        maybe_print_matrix("y_dense", y_dense)
        maybe_print_matrix("y_sparse", y_sparse)
        maybe_print_matrix("diff = y_sparse - y_dense", y_sparse.float() - y_dense.float())

    if args.profile_runtime and hasattr(slinear, "get_runtime_profile_pretty"):
        print("\n[Runtime profile]")
        print(slinear.get_runtime_profile_pretty())

    if args.direct_kernel:
        print("\n[Direct kernel path]")
        with torch.no_grad():
            w_t = get_w_t_from_module(slinear)
            y_kernel, sparse_ms = sparse_linear_forward(
                x=x,
                weight=slinear.weight,
                bias=slinear.bias,
                threshold=slinear.threshold,
                w_t=w_t,
                counts_buf=None,
                tile_cin_buf=None,
                return_ms=False,
            )
        print(f"  cosine similarity: {cosine(y_dense, y_kernel):.8f}")
        print(f"  max abs diff:      {max_abs_diff(y_dense, y_kernel):.8f}")
        print(f"  sparse_ms:         {sparse_ms}")
        if args.dump_tensors:
            maybe_print_matrix("y_kernel", y_kernel)
            maybe_print_matrix("kernel_diff = y_kernel - y_dense", y_kernel.float() - y_dense.float())

if __name__ == "__main__":
    main()
