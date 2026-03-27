#!/usr/bin/env python3
"""
apply_bench_4test_v25.py — Apply v25 sync-gating + A/B tile launch changes to bench_4test.py

This script modifies the existing bench_4test.py by:
  1. Adding the timing_utils import
  2. Adding --launch_all_tiles, --inference_mode, --ab_compare CLI args
  3. Adding timing preparation before measure_mode() calls
  4. Adding optional A/B tile launch comparison section

Usage:
  python apply_bench_4test_v25.py               # dry-run (prints diff)
  python apply_bench_4test_v25.py --apply        # apply changes
  python apply_bench_4test_v25.py --apply --root /path/to/SparseFlow
"""

import argparse
import sys
import shutil
from pathlib import Path


# ============================================================
# PATCH 1: Add import after existing imports
# ============================================================
IMPORT_MARKER = "from Utils.dispatch_model import dispatch_all_layers, decisions_to_sets"
IMPORT_ADDITION = """from Utils.dispatch_model import dispatch_all_layers, decisions_to_sets
from Utils.timing_utils import prepare_for_timing, set_launch_mode, count_sync_state"""

# ============================================================
# PATCH 2: Add CLI args after existing parser args
# ============================================================
# We look for the last parser.add_argument in the main() function
ARGS_MARKER = '    parser.add_argument("--min_spatial_size"'
ARGS_ADDITION_AFTER = """    parser.add_argument("--min_spatial_size","""
ARGS_NEW = """
    # --- v25: sync-gating and A/B tile launch ---
    parser.add_argument("--launch_all_tiles", action="store_true",
                        help="Mode B: launch all tiles, zero tiles early-return. "
                             "Default Mode A: build active tile IDs.")
    parser.add_argument("--inference_mode", action="store_true",
                        help="Disable periodic calibration syncs during timing.")
    parser.add_argument("--ab_compare", action="store_true",
                        help="Run both Mode A and Mode B and report comparison.")
"""

# ============================================================
# PATCH 3: Add timing preparation before measure_mode
# ============================================================
TIMING_MARKER = '    print(f"\\n[5/7]'
TIMING_PREP = """
    # ── v25: timing preparation ──
    if args.inference_mode:
        n_so = prepare_for_timing(model_sparse_only)
        n_hy = prepare_for_timing(model_hybrid)
        print(f"  [v25] inference_mode: {n_so} sparse_only + {n_hy} hybrid modules configured")

    if args.launch_all_tiles:
        set_launch_mode(model_sparse_only, launch_all=True)
        set_launch_mode(model_hybrid, launch_all=True)
        print(f"  [v25] launch_all_tiles=True (Mode B)")

    _so_state = count_sync_state(model_sparse_only)
    _hy_state = count_sync_state(model_hybrid)
    print(f"  [v25] sparse_only sync state: {_so_state}")
    print(f"  [v25] hybrid sync state: {_hy_state}")

"""

# ============================================================
# PATCH 4: Add A/B comparison after consistency checks
# ============================================================
AB_MARKER = "results = {"
AB_CODE = """
    # ── v25: A/B tile launch comparison ──
    if args.ab_compare:
        print(f"\\n[A/B] Tile launch mode comparison ...")
        prepare_for_timing(model_sparse_only)
        prepare_for_timing(model_hybrid)

        set_launch_mode(model_sparse_only, launch_all=False)
        set_launch_mode(model_hybrid, launch_all=False)
        so_a = measure_mode(model_sparse_only, loader, device, args.T, args.warmup, args.spike_mode, args.power, "Sparse ModeA")
        hy_a = measure_mode(model_hybrid, loader, device, args.T, args.warmup, args.spike_mode, args.power, "Hybrid ModeA")

        set_launch_mode(model_sparse_only, launch_all=True)
        set_launch_mode(model_hybrid, launch_all=True)
        so_b = measure_mode(model_sparse_only, loader, device, args.T, args.warmup, args.spike_mode, args.power, "Sparse ModeB")
        hy_b = measure_mode(model_hybrid, loader, device, args.T, args.warmup, args.spike_mode, args.power, "Hybrid ModeB")

        print(f"\\n  {'Config':<28} {'Latency(ms)':>12}")
        print(f"  {'-'*44}")
        for r in [so_a, so_b, hy_a, hy_b]:
            print_mode_result(r)
        so_diff = so_a["avg_ms"] - so_b["avg_ms"]
        hy_diff = hy_a["avg_ms"] - hy_b["avg_ms"]
        print(f"\\n  Sparse: ModeA - ModeB = {so_diff:+.2f} ms  (positive = ModeB faster)")
        print(f"  Hybrid: ModeA - ModeB = {hy_diff:+.2f} ms  (positive = ModeB faster)")

"""


def apply_patches(content: str) -> str:
    """Apply all v25 patches to bench_4test.py content."""

    # Patch 1: import
    if "from Utils.timing_utils import" in content:
        print("  [Patch 1] timing_utils import already present, skipping.")
    else:
        if IMPORT_MARKER in content:
            content = content.replace(IMPORT_MARKER, IMPORT_ADDITION, 1)
            print("  [Patch 1] Added timing_utils import.")
        else:
            print("  [Patch 1] WARNING: Could not find import marker. Add manually:")
            print(f"    from Utils.timing_utils import prepare_for_timing, set_launch_mode, count_sync_state")

    # Patch 2: CLI args
    if "--launch_all_tiles" in content:
        print("  [Patch 2] CLI args already present, skipping.")
    else:
        # Find the last parser.add_argument before args = parser.parse_args()
        insert_pos = content.rfind('    args = parser.parse_args()')
        if insert_pos > 0:
            content = content[:insert_pos] + ARGS_NEW + "\n" + content[insert_pos:]
            print("  [Patch 2] Added --launch_all_tiles, --inference_mode, --ab_compare args.")
        else:
            print("  [Patch 2] WARNING: Could not find parse_args(). Add args manually.")

    # Patch 3: timing preparation
    if "[v25] inference_mode" in content:
        print("  [Patch 3] Timing preparation already present, skipping.")
    else:
        # Find the timing section header
        idx = content.find('[5/7]')
        if idx < 0:
            idx = content.find('[5/6]')
        if idx > 0:
            # Find the print statement containing this
            line_start = content.rfind('\n', 0, idx)
            if line_start > 0:
                content = content[:line_start] + "\n" + TIMING_PREP + content[line_start:]
                print("  [Patch 3] Added timing preparation before measure_mode.")
        else:
            print("  [Patch 3] WARNING: Could not find step 5 marker. Add timing prep manually.")

    # Patch 4: A/B comparison
    if "A/B] Tile launch mode comparison" in content:
        print("  [Patch 4] A/B comparison already present, skipping.")
    else:
        # Insert before results = { ... }
        # Find the LAST occurrence of "    results = {" in main()
        idx = content.rfind("    results = {")
        if idx > 0:
            content = content[:idx] + AB_CODE + content[idx:]
            print("  [Patch 4] Added A/B tile launch comparison section.")
        else:
            print("  [Patch 4] WARNING: Could not find results dict. Add A/B code manually.")

    return content


def main():
    parser = argparse.ArgumentParser(description="Apply v25 patches to bench_4test.py")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default: dry-run)")
    parser.add_argument("--root", type=str, default=None, help="SparseFlow root directory")
    args = parser.parse_args()

    # Find SparseFlow root
    if args.root:
        root = Path(args.root)
    else:
        for candidate in [Path(__file__).resolve().parents[1],
                          Path.cwd(),
                          Path.cwd().parent,
                          Path.home() / "SparseFlow"]:
            if (candidate / "Kernels" / "conv2d.py").exists():
                root = candidate
                break
        else:
            print("ERROR: Cannot find SparseFlow root. Use --root.")
            sys.exit(1)

    bench_path = root / "Benchmark" / "bench_4test.py"
    if not bench_path.exists():
        print(f"ERROR: {bench_path} not found")
        sys.exit(1)

    # Also ensure timing_utils.py exists
    timing_utils_path = root / "Utils" / "timing_utils.py"
    if not timing_utils_path.exists():
        print(f"WARNING: {timing_utils_path} does not exist.")
        print(f"  You need to place timing_utils.py in Utils/ first.")

    content = bench_path.read_text()
    print(f"Read {len(content)} bytes from {bench_path}")

    patched = apply_patches(content)

    if not args.apply:
        print("\n=== DRY RUN — no changes written ===")
        print(f"Use --apply to write changes to {bench_path}")
        # Show a brief diff summary
        orig_lines = content.count('\n')
        new_lines = patched.count('\n')
        print(f"Lines: {orig_lines} → {new_lines} (+{new_lines - orig_lines})")
    else:
        # Backup
        backup = bench_path.with_suffix('.py.pre_v25')
        if not backup.exists():
            shutil.copy2(bench_path, backup)
            print(f"Backup saved to {backup}")
        bench_path.write_text(patched)
        print(f"Wrote {len(patched)} bytes to {bench_path}")
        print("Done.")


if __name__ == "__main__":
    main()