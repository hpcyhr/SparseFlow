"""
Utils/timing_utils.py — Timing hygiene and A/B tile launch utilities.

Usage in bench_4test.py:
    from Utils.timing_utils import prepare_for_timing, set_launch_mode, count_sync_state

    # Before measure_mode (after warmup/calibration phase):
    prepare_for_timing(model_sparse_only)
    prepare_for_timing(model_hybrid)

    # For A/B comparison:
    set_launch_mode(model_sparse_only, launch_all=True)   # Mode B
    so_res_b = measure_mode(model_sparse_only, ...)
    set_launch_mode(model_sparse_only, launch_all=False)  # Mode A
    so_res_a = measure_mode(model_sparse_only, ...)

    # Verify clean state before timing:
    print(count_sync_state(model_sparse_only))
"""

import torch
import torch.nn as nn
from typing import Dict


def prepare_for_timing(model: nn.Module) -> int:
    """Set all SparseConv2d modules to sync-free inference mode.

    This ensures that during timed region:
    - inference_mode=True: _should_collect_ratio() → False (no periodic calibration)
    - collect_diag=False: no diagnostic data collection
    - profile_runtime=False: no per-op timing stamps

    Combined with the P0 fix in conv2d.py, this guarantees zero GPU→CPU
    synchronizations in the sparse forward path (with Mode B), or exactly
    one sync per layer (with Mode A from nonzero()).

    Returns number of modules configured.
    """
    from Ops.sparse_conv2d import SparseConv2d

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, SparseConv2d):
            module.set_inference_mode(True)
            count += 1
    return count


def set_launch_mode(model: nn.Module, launch_all: bool) -> int:
    """Set tile launch mode for all SparseConv2d modules.

    Args:
        launch_all: If True, use Mode B (launch all tiles, zero tiles early-return).
                    If False, use Mode A (build active tile IDs, launch only active).

    Mode B advantages:
    - Eliminates the nonzero() GPU→CPU sync in _build_active_tile_ids.
    - With prepare_for_timing(), achieves truly zero syncs per forward.
    - Zero tiles execute ~2 memory ops then return — very cheap.

    Mode A advantages:
    - Launches fewer kernel programs (only active tiles).
    - May win at very high sparsity (>95%) where most tiles are zero.
    - Handles the all-tiles-zero edge case without launching any tile.

    Returns number of modules configured.
    """
    from Ops.sparse_conv2d import SparseConv2d

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, SparseConv2d):
            module.set_launch_all_tiles(launch_all)
            count += 1
    return count


def count_sync_state(model: nn.Module) -> Dict[str, int]:
    """Count SparseConv2d modules by sync-relevant state.

    Use to verify timing readiness before measure_mode().

    Expected clean state: all counters zero except 'total' and 'launch_all'/'launch_active'.
    """
    from Ops.sparse_conv2d import SparseConv2d

    stats = {
        "total": 0,
        "inference_mode": 0,
        "collect_diag": 0,
        "profile_runtime": 0,
        "warmup_pending": 0,
        "force_zero": 0,
        "force_dense": 0,
        "launch_all": 0,
        "launch_active": 0,
    }

    for name, module in model.named_modules():
        if isinstance(module, SparseConv2d):
            stats["total"] += 1
            if module._inference_mode:
                stats["inference_mode"] += 1
            if module.collect_diag:
                stats["collect_diag"] += 1
            if module.profile_runtime:
                stats["profile_runtime"] += 1
            if module._warmup_left > 0:
                stats["warmup_pending"] += 1
            if module._force_zero:
                stats["force_zero"] += 1
            if module._force_dense:
                stats["force_dense"] += 1
            if module.launch_all_tiles:
                stats["launch_all"] += 1
            else:
                stats["launch_active"] += 1

    return stats


def estimate_sync_count(model: nn.Module) -> Dict[str, int]:
    """Estimate the number of GPU→CPU syncs per forward pass.

    Based on current module configuration and the v25 sync gating:
    - Each SparseConv2d with inference_mode=True and launch_all_tiles=True: 0 syncs
    - Each SparseConv2d with inference_mode=True and launch_all_tiles=False: 1 sync
    - Each SparseConv2d with inference_mode=False: up to 6 syncs (when calib fires)
    """
    from Ops.sparse_conv2d import SparseConv2d

    result = {
        "zero_sync_modules": 0,
        "one_sync_modules": 0,
        "multi_sync_modules": 0,
        "total_min_syncs_per_forward": 0,
        "total_max_syncs_per_forward": 0,
    }

    for name, module in model.named_modules():
        if isinstance(module, SparseConv2d):
            if module._force_zero or module._force_dense:
                result["zero_sync_modules"] += 1
            elif module._inference_mode:
                if module.launch_all_tiles:
                    result["zero_sync_modules"] += 1
                else:
                    result["one_sync_modules"] += 1
                    result["total_min_syncs_per_forward"] += 1
                    result["total_max_syncs_per_forward"] += 1
            else:
                # Without inference_mode, periodic calibration may trigger syncs.
                # Min: 0 (non-calib step, launch_all) or 1 (non-calib, launch_active)
                # Max: 6 (calib step)
                if module.launch_all_tiles:
                    result["zero_sync_modules"] += 1
                    result["total_max_syncs_per_forward"] += 6
                else:
                    result["one_sync_modules"] += 1
                    result["total_min_syncs_per_forward"] += 1
                    result["total_max_syncs_per_forward"] += 6

    return result