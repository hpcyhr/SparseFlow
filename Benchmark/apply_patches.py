#!/usr/bin/env python3
"""
Incremental patch for bench_4test.py — fixes the remaining 'unavail' diag entries.

Problem: layer1.0.conv1 and layer1.0.downsample.0 show n/a because:
  1. They have ~100% sparsity but fail strict zeros==total check → not in zero_layers
  2. In the sparse-only temp model, SparseConv2d's _update_policy sets _force_zero=True
     after a few warmup batches → subsequent forwards skip the Triton kernel entirely
     → _last_diag stays empty → measure_group_sparsity returns no data for these layers

Fix: Add a catch-all after the zero_layers synthetic override that detects layers with
very high element sparsity (>=99.99%) but missing diag data, and assigns synthetic zeros.

Usage:
  cd ~/SparseFlow
  python Benchmark/apply_patch_v2.py          # dry-run
  python Benchmark/apply_patch_v2.py --apply  # apply
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--root", type=str, default=None)
    args = parser.parse_args()

    if args.root:
        root = Path(args.root)
    else:
        for c in [Path(__file__).resolve().parents[1], Path.cwd(), Path.home() / "SparseFlow"]:
            if (c / "Kernels" / "conv2d.py").exists():
                root = c
                break
        else:
            print("ERROR: Cannot find SparseFlow root.")
            sys.exit(1)

    bp = root / "Benchmark" / "bench_4test.py"
    if not bp.exists():
        print(f"ERROR: {bp} not found")
        sys.exit(1)

    content = bp.read_text()

    # Find the end of the zero_layers synthetic override block
    # Look for the marker that ends the first override loop
    old_pattern = """                    '_diag_path': 'staticzero',
                }"""

    if old_pattern not in content:
        print("ERROR: Cannot find the zero_layers override block.")
        print("       Make sure the v1 patch was applied first.")
        sys.exit(1)

    # Check if catch-all is already present
    if "'_diag_path': 'zero_fastpath'" in content:
        print("Catch-all patch already applied, nothing to do.")
        return

    new_pattern = """                    '_diag_path': 'staticzero',
                }

        # Catch-all: layers with near-100% sparsity that took the zero fast-path
        # during measurement (SparseConv2d._force_zero skips Triton → empty _last_diag)
        for _t in targets:
            _n = _t["name"]
            if _n in zero_layers:
                continue  # already handled above
            _gd = group_sparsity_data.get(_n, {})
            if _gd.get('active_group_ratio', -1.0) >= 0:
                continue  # already has real diag data
            _sd = sparsity_data.get(_n, {"zeros": 0, "total": 1})
            _elem_sp = _sd["zeros"] / max(_sd["total"], 1)
            if _elem_sp >= 0.9999:
                _gd_existing = group_sparsity_data.get(_n, {})
                group_sparsity_data[_n] = {
                    'active_group_ratio': 0.0,
                    'tile_zero_ratio': 1.0,
                    'total_group_count': _gd_existing.get('total_group_count', -1.0),
                    'nonzero_group_count': 0.0,
                    'tile_zero_count': _gd_existing.get('total_tile_count', -1.0),
                    'total_tile_count': _gd_existing.get('total_tile_count', -1.0),
                    'effective_k_ratio': 0.0,
                    'sparse_compute_ms': -1.0,
                    'sparse_total_ms': -1.0,
                    '_synthetic': True,
                    '_diag_path': 'zero_fastpath',
                }"""

    new_content = content.replace(old_pattern, new_pattern)

    if new_content == content:
        print("ERROR: Replacement produced no change")
        sys.exit(1)

    if args.apply:
        backup = str(bp) + ".bak2"
        shutil.copy2(str(bp), backup)
        bp.write_text(new_content)
        print(f"DONE: {bp} patched (backup: .bak2)")
    else:
        print(f"DRY-RUN: {bp} would be patched")
        print("Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()