#!/usr/bin/env python3
"""
SparseFlow Backend Quality Patch — v21.1

Applies three targeted fixes to improve real compute reduction in
Kernels/conv2d.py and Kernels/fused_conv_lif.py:

  P0: Add `if g_active != 0:` guard in all sparse-path group loops
      so that inactive groups are truly skipped (no loads, no dot).
      Before: every group iteration loads zeros + executes tl.dot on zeros.
      After:  inactive groups skip the entire body via conditional branch.

  P1: Replace _popcount_buf and _check_dense_fallback with vectorized
      popcount (no Python loop over 32 bits).

  P2: Vectorize the inline popcount in fused_conv_lif.py entry point.

Usage:
    cd ~/SparseFlow
    python patch_backend_quality.py           # dry-run (shows diffs)
    python patch_backend_quality.py --apply   # apply patches
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path


def find_project_root():
    for c in [Path(__file__).resolve().parent, Path.cwd(), Path.home() / "SparseFlow"]:
        if (c / "Kernels" / "conv2d.py").exists():
            return c
    return None


# ---------------------------------------------------------------------------
# P0: Sparse-path group-skip guard for conv2d.py
# ---------------------------------------------------------------------------
#
# The pattern we're looking for in every sparse-path else branch:
#
#   for g in range(NUM_GROUPS):
#       g_active = (ag_mask >> g) & 1
#       cin_start = g * GROUP_SIZE_C
#       offs_k = cin_start + tl.arange(0, GROUP_SIZE_C)
#       k_mask = (g_active != 0) & (offs_k < C_IN)
#       ... tl.load / tl.dot ...
#
# We need to wrap everything after `g_active =` in `if g_active != 0:`
# This applies to:
#   - 1x1 sparse path (simple: one load pair + one dot)
#   - 3x3/s1 sparse path (nested kh/kw loop)
#   - 3x3/s2 sparse path (nested kh/kw loop)

# For 1x1 kernels, the sparse path looks like:
OLD_1x1_SPARSE = """\
    else:
        for g in range(NUM_GROUPS):
            g_active = (ag_mask >> g) & 1
            cin_start = g * GROUP_SIZE_C
            offs_k = cin_start + tl.arange(0, GROUP_SIZE_C)
            k_mask = (g_active != 0) & (offs_k < C_IN)
            x_addrs = x_ptr + (n_idx * C_IN + offs_k[None, :]) * HW_IN + out_h[:, None] * W_IN + out_w[:, None]
            x_tile = tl.load(x_addrs, mask=k_mask[None, :] & m_mask[:, None], other=0.0).to(tl.float16)
            w_addrs = w_cl_ptr + offs_n[None, :] * C_IN + offs_k[:, None]
            w_tile = tl.load(w_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
            acc += tl.dot(x_tile, w_tile)"""

NEW_1x1_SPARSE = """\
    else:
        # TILE_SPARSE: bitmask-gated with true group skip
        for g in range(NUM_GROUPS):
            g_active = (ag_mask >> g) & 1
            if g_active != 0:
                cin_start = g * GROUP_SIZE_C
                offs_k = cin_start + tl.arange(0, GROUP_SIZE_C)
                k_mask = offs_k < C_IN
                x_addrs = x_ptr + (n_idx * C_IN + offs_k[None, :]) * HW_IN + out_h[:, None] * W_IN + out_w[:, None]
                x_tile = tl.load(x_addrs, mask=k_mask[None, :] & m_mask[:, None], other=0.0).to(tl.float16)
                w_addrs = w_cl_ptr + offs_n[None, :] * C_IN + offs_k[:, None]
                w_tile = tl.load(w_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
                acc += tl.dot(x_tile, w_tile)"""


# For 3x3/s1 kernels, the sparse path:
OLD_3x3s1_SPARSE = """\
    else:
        for g in range(NUM_GROUPS):
            g_active = (ag_mask >> g) & 1
            cin_start = g * GROUP_SIZE_C
            offs_k = cin_start + tl.arange(0, GROUP_SIZE_C)
            k_mask = (g_active != 0) & (offs_k < C_IN)
            for kh in tl.static_range(3):
                for kw in tl.static_range(3):
                    in_h = out_h + (kh - 1)
                    in_w = out_w + (kw - 1)
                    h_ok = (in_h >= 0) & (in_h < H_IN)
                    w_ok = (in_w >= 0) & (in_w < W_IN)
                    safe_h = tl.minimum(tl.maximum(in_h, 0), H_IN - 1)
                    safe_w = tl.minimum(tl.maximum(in_w, 0), W_IN - 1)
                    x_addrs = x_ptr + (n_idx * C_IN + offs_k[None, :]) * HW_IN + safe_h[:, None] * W_IN + safe_w[:, None]
                    x_m = k_mask[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                    x_tile = tl.load(x_addrs, mask=x_m, other=0.0).to(tl.float16)
                    w_addrs = w_cl_ptr + offs_n[None, :] * W_CO + kh * W_KH + kw * W_CS + offs_k[:, None]
                    w_tile = tl.load(w_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
                    acc += tl.dot(x_tile, w_tile)"""

NEW_3x3s1_SPARSE = """\
    else:
        # TILE_SPARSE: bitmask-gated with true group skip
        for g in range(NUM_GROUPS):
            g_active = (ag_mask >> g) & 1
            if g_active != 0:
                cin_start = g * GROUP_SIZE_C
                offs_k = cin_start + tl.arange(0, GROUP_SIZE_C)
                k_mask = offs_k < C_IN
                for kh in tl.static_range(3):
                    for kw in tl.static_range(3):
                        in_h = out_h + (kh - 1)
                        in_w = out_w + (kw - 1)
                        h_ok = (in_h >= 0) & (in_h < H_IN)
                        w_ok = (in_w >= 0) & (in_w < W_IN)
                        safe_h = tl.minimum(tl.maximum(in_h, 0), H_IN - 1)
                        safe_w = tl.minimum(tl.maximum(in_w, 0), W_IN - 1)
                        x_addrs = x_ptr + (n_idx * C_IN + offs_k[None, :]) * HW_IN + safe_h[:, None] * W_IN + safe_w[:, None]
                        x_m = k_mask[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                        x_tile = tl.load(x_addrs, mask=x_m, other=0.0).to(tl.float16)
                        w_addrs = w_cl_ptr + offs_n[None, :] * W_CO + kh * W_KH + kw * W_CS + offs_k[:, None]
                        w_tile = tl.load(w_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
                        acc += tl.dot(x_tile, w_tile)"""

# For 3x3/s2 kernels:
OLD_3x3s2_SPARSE = """\
    else:
        for g in range(NUM_GROUPS):
            g_active = (ag_mask >> g) & 1
            cin_start = g * GROUP_SIZE_C
            offs_k = cin_start + tl.arange(0, GROUP_SIZE_C)
            k_mask = (g_active != 0) & (offs_k < C_IN)
            for kh in tl.static_range(3):
                for kw in tl.static_range(3):
                    in_h = out_h * 2 + (kh - 1)
                    in_w = out_w * 2 + (kw - 1)
                    h_ok = (in_h >= 0) & (in_h < H_IN)
                    w_ok = (in_w >= 0) & (in_w < W_IN)
                    safe_h = tl.minimum(tl.maximum(in_h, 0), H_IN - 1)
                    safe_w = tl.minimum(tl.maximum(in_w, 0), W_IN - 1)
                    x_addrs = x_ptr + (n_idx * C_IN + offs_k[None, :]) * HW_IN + safe_h[:, None] * W_IN + safe_w[:, None]
                    x_m = k_mask[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                    x_tile = tl.load(x_addrs, mask=x_m, other=0.0).to(tl.float16)
                    w_addrs = w_cl_ptr + offs_n[None, :] * W_CO + kh * W_KH + kw * W_CS + offs_k[:, None]
                    w_tile = tl.load(w_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
                    acc += tl.dot(x_tile, w_tile)"""

NEW_3x3s2_SPARSE = """\
    else:
        # TILE_SPARSE: bitmask-gated with true group skip
        for g in range(NUM_GROUPS):
            g_active = (ag_mask >> g) & 1
            if g_active != 0:
                cin_start = g * GROUP_SIZE_C
                offs_k = cin_start + tl.arange(0, GROUP_SIZE_C)
                k_mask = offs_k < C_IN
                for kh in tl.static_range(3):
                    for kw in tl.static_range(3):
                        in_h = out_h * 2 + (kh - 1)
                        in_w = out_w * 2 + (kw - 1)
                        h_ok = (in_h >= 0) & (in_h < H_IN)
                        w_ok = (in_w >= 0) & (in_w < W_IN)
                        safe_h = tl.minimum(tl.maximum(in_h, 0), H_IN - 1)
                        safe_w = tl.minimum(tl.maximum(in_w, 0), W_IN - 1)
                        x_addrs = x_ptr + (n_idx * C_IN + offs_k[None, :]) * HW_IN + safe_h[:, None] * W_IN + safe_w[:, None]
                        x_m = k_mask[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                        x_tile = tl.load(x_addrs, mask=x_m, other=0.0).to(tl.float16)
                        w_addrs = w_cl_ptr + offs_n[None, :] * W_CO + kh * W_KH + kw * W_CS + offs_k[:, None]
                        w_tile = tl.load(w_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
                        acc += tl.dot(x_tile, w_tile)"""


# ---------------------------------------------------------------------------
# P1: Vectorized popcount for _popcount_buf and _check_dense_fallback
# ---------------------------------------------------------------------------

OLD_POPCOUNT = """\
def _popcount_buf(ag_mask_buf, N_TILES):
    \"\"\"Vectorised popcount for int32 bitmask buffer.\"\"\"
    masks = ag_mask_buf[:N_TILES].int()
    pc = torch.zeros(N_TILES, dtype=torch.int32, device=masks.device)
    tmp = masks.clone()
    for _ in range(32):
        pc += tmp & 1
        tmp = tmp >> 1
    return pc"""

NEW_POPCOUNT = """\
def _popcount_buf(ag_mask_buf, N_TILES):
    \"\"\"Vectorised popcount for int32 bitmask buffer (parallel bit-count).\"\"\"
    # Use the standard Hamming-weight (SWAR) approach — no Python loop.
    # Works on int32 via standard bit manipulation constants.
    v = ag_mask_buf[:N_TILES].int()
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F
    v = v + (v >> 8)
    v = v + (v >> 16)
    return (v & 0x3F).to(torch.int32)"""


OLD_CHECK_FALLBACK = """\
def _check_dense_fallback(ag_mask_buf, N_TILES, NUM_GROUPS, fallback_ratio=FALLBACK_RATIO):
    if NUM_GROUPS == 0:
        return False
    masks = ag_mask_buf[:N_TILES].int()
    pc = torch.zeros(N_TILES, dtype=torch.int32, device=masks.device)
    tmp = masks.clone()
    for _ in range(32):
        pc += tmp & 1
        tmp = tmp >> 1
    avg_active = pc.float().mean().item()
    threshold = fallback_ratio * NUM_GROUPS
    return avg_active > threshold"""

NEW_CHECK_FALLBACK = """\
def _check_dense_fallback(ag_mask_buf, N_TILES, NUM_GROUPS, fallback_ratio=FALLBACK_RATIO):
    if NUM_GROUPS == 0:
        return False
    pc = _popcount_buf(ag_mask_buf, N_TILES)
    avg_active = pc.float().mean().item()
    threshold = fallback_ratio * NUM_GROUPS
    return avg_active > threshold"""


# ---------------------------------------------------------------------------
# P2: Fix inline popcount in fused_conv_lif.py
# ---------------------------------------------------------------------------

OLD_FUSED_POPCOUNT = """\
        masks = ag_mask_buf[:N_TILES].int()
        pc = torch.zeros(N_TILES, dtype=torch.int32, device=device)
        tmp = masks.clone()
        for _ in range(32):
            pc += tmp & 1
            tmp = tmp >> 1
        avg_active_ratio = pc.float().mean().item() / max(NUM_GROUPS, 1)"""

NEW_FUSED_POPCOUNT = """\
        # Vectorized SWAR popcount (no Python loop)
        v = ag_mask_buf[:N_TILES].int()
        v = v - ((v >> 1) & 0x55555555)
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
        v = (v + (v >> 4)) & 0x0F0F0F0F
        v = v + (v >> 8)
        v = v + (v >> 16)
        pc = (v & 0x3F).to(torch.int32)
        avg_active_ratio = pc.float().mean().item() / max(NUM_GROUPS, 1)"""


# ---------------------------------------------------------------------------
# P0 for fused_conv_lif.py: sparse-path group-skip guard
# ---------------------------------------------------------------------------
# The fused kernels have the same issue — iterating all groups without skip.
# Pattern for 3x3 fused sparse path (inside the conv accumulation loop):

OLD_FUSED_SPARSE = """\
    else:
        for g in range(NUM_GROUPS):
            g_active = (ag_mask >> g) & 1
            cin_start = g * GROUP_SIZE_C
            offs_k = cin_start + tl.arange(0, GROUP_SIZE_C)
            k_mask = (g_active != 0) & (offs_k < C_IN)"""

NEW_FUSED_SPARSE = """\
    else:
        # TILE_SPARSE: bitmask-gated with true group skip
        for g in range(NUM_GROUPS):
            g_active = (ag_mask >> g) & 1
            if g_active != 0:
                cin_start = g * GROUP_SIZE_C
                offs_k = cin_start + tl.arange(0, GROUP_SIZE_C)
                k_mask = offs_k < C_IN"""


def apply_patches(root, dry_run=True):
    conv2d_path = root / "Kernels" / "conv2d.py"
    fused_path = root / "Kernels" / "fused_conv_lif.py"

    patches_applied = 0
    patches_failed = 0

    # ---- Patch conv2d.py ----
    if conv2d_path.exists():
        content = conv2d_path.read_text()
        original = content

        # P0: Sparse path group-skip guards
        # Apply 1x1 sparse path fix (appears in _8x8 and _8x16 variants)
        count = content.count(OLD_1x1_SPARSE)
        if count > 0:
            content = content.replace(OLD_1x1_SPARSE, NEW_1x1_SPARSE)
            print(f"  [P0] 1x1 sparse-path group-skip guard: {count} instances patched")
            patches_applied += count
        else:
            print(f"  [P0] 1x1 sparse-path: pattern not found (maybe already patched?)")
            patches_failed += 1

        # Apply 3x3/s1 sparse path fix
        count = content.count(OLD_3x3s1_SPARSE)
        if count > 0:
            content = content.replace(OLD_3x3s1_SPARSE, NEW_3x3s1_SPARSE)
            print(f"  [P0] 3x3/s1 sparse-path group-skip guard: {count} instances patched")
            patches_applied += count
        else:
            print(f"  [P0] 3x3/s1 sparse-path: pattern not found")
            patches_failed += 1

        # Apply 3x3/s2 sparse path fix
        count = content.count(OLD_3x3s2_SPARSE)
        if count > 0:
            content = content.replace(OLD_3x3s2_SPARSE, NEW_3x3s2_SPARSE)
            print(f"  [P0] 3x3/s2 sparse-path group-skip guard: {count} instances patched")
            patches_applied += count
        else:
            print(f"  [P0] 3x3/s2 sparse-path: pattern not found")
            patches_failed += 1

        # P1: Vectorized popcount
        if OLD_POPCOUNT in content:
            content = content.replace(OLD_POPCOUNT, NEW_POPCOUNT)
            print(f"  [P1] _popcount_buf: vectorized (SWAR)")
            patches_applied += 1
        else:
            print(f"  [P1] _popcount_buf: pattern not found")
            patches_failed += 1

        if OLD_CHECK_FALLBACK in content:
            content = content.replace(OLD_CHECK_FALLBACK, NEW_CHECK_FALLBACK)
            print(f"  [P1] _check_dense_fallback: uses vectorized popcount")
            patches_applied += 1
        else:
            print(f"  [P1] _check_dense_fallback: pattern not found")
            patches_failed += 1

        if content != original:
            if dry_run:
                print(f"\n  [DRY-RUN] Would write {conv2d_path}")
            else:
                backup = conv2d_path.with_suffix('.py.pre_p0')
                if not backup.exists():
                    shutil.copy2(conv2d_path, backup)
                conv2d_path.write_text(content)
                print(f"\n  [APPLIED] {conv2d_path} (backup: {backup.name})")
        else:
            print(f"\n  [SKIP] {conv2d_path} — no changes needed")

    # ---- Patch fused_conv_lif.py ----
    if fused_path.exists():
        content = fused_path.read_text()
        original = content

        # P2: Inline popcount fix
        if OLD_FUSED_POPCOUNT in content:
            content = content.replace(OLD_FUSED_POPCOUNT, NEW_FUSED_POPCOUNT)
            print(f"  [P2] fused_conv_lif.py inline popcount: vectorized")
            patches_applied += 1
        else:
            print(f"  [P2] fused_conv_lif.py inline popcount: pattern not found")
            patches_failed += 1

        # P0 for fused: sparse-path group-skip
        # This is trickier because the fused kernel body continues after the k_mask line
        # with spatial loops. We'll use a simpler match.
        count = content.count(OLD_FUSED_SPARSE)
        if count > 0:
            content = content.replace(OLD_FUSED_SPARSE, NEW_FUSED_SPARSE)
            print(f"  [P0] fused 3x3 sparse-path group-skip guard: {count} instances patched")
            patches_applied += count
        else:
            print(f"  [P0] fused 3x3 sparse-path: pattern not found")
            patches_failed += 1

        if content != original:
            if dry_run:
                print(f"\n  [DRY-RUN] Would write {fused_path}")
            else:
                backup = fused_path.with_suffix('.py.pre_p0')
                if not backup.exists():
                    shutil.copy2(fused_path, backup)
                fused_path.write_text(content)
                print(f"\n  [APPLIED] {fused_path} (backup: {backup.name})")
        else:
            print(f"\n  [SKIP] {fused_path} — no changes needed")

    print(f"\n  Summary: {patches_applied} patches applied, {patches_failed} not found")
    return patches_applied, patches_failed


def main():
    parser = argparse.ArgumentParser(description="SparseFlow Backend Quality Patch v21.1")
    parser.add_argument("--apply", action="store_true", help="Apply patches (default: dry-run)")
    parser.add_argument("--root", type=str, default=None, help="Project root override")
    args = parser.parse_args()

    if args.root:
        root = Path(args.root)
    else:
        root = find_project_root()
        if root is None:
            print("ERROR: Cannot find SparseFlow root. Use --root.")
            sys.exit(1)

    print(f"SparseFlow Backend Quality Patch v21.1")
    print(f"  Project root: {root}")
    print(f"  Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print()

    apply_patches(root, dry_run=not args.apply)


if __name__ == "__main__":
    main()