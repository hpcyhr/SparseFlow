# Utils/timing_utils.py
"""
Utils/timing_utils.py - Timing hygiene and A/B tile launch utilities.

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

v26 notes:
- prepare_for_timing now covers both SparseConv2d and SparseLinear (any module
  that implements set_inference_mode(bool)).
- set_launch_mode now applies to any module implementing
  set_launch_all_tiles(bool) (currently SparseConv2d).
- count_sync_state now reports over all sparse policy modules (conv + linear).
"""

import torch.nn as nn
from typing import Dict


def _iter_sparse_policy_modules(model: nn.Module):
    """Yield modules that expose sparse runtime policy knobs.

    Criteria are capability-based so new sparse ops can be included without
    changing this file:
    - has set_inference_mode(...), or
    - has _inference_mode / _force_dense / _force_zero state attributes.
    """
    for name, module in model.named_modules():
        if (
            hasattr(module, "set_inference_mode")
            or hasattr(module, "_inference_mode")
            or hasattr(module, "_force_dense")
            or hasattr(module, "_force_zero")
        ):
            yield name, module


def prepare_for_timing(model: nn.Module) -> int:
    """Set sparse policy modules to sync-free inference mode.

    This ensures that during timed region:
    - inference_mode=True: _should_collect_ratio() -> False (no periodic calibration)
    - collect_diag=False: no diagnostic data collection
    - profile_runtime=False: no per-op timing stamps

    Combined with the sync-gating fixes in sparse kernels, this minimizes
    GPU->CPU synchronizations during timed runs.

    Returns number of modules configured (conv + linear sparse modules).
    """
    count = 0
    for _, module in _iter_sparse_policy_modules(model):
        if hasattr(module, "set_inference_mode"):
            module.set_inference_mode(True)
            count += 1
    return count


def set_launch_mode(model: nn.Module, launch_all: bool) -> int:
    """Set tile launch mode for all modules that support it.

    Args:
        launch_all: If True, use Mode B (launch all tiles, zero tiles early-return).
                    If False, use Mode A (build active tile IDs, launch only active).

    Returns number of modules configured.
    """
    count = 0
    for _, module in model.named_modules():
        if hasattr(module, "set_launch_all_tiles"):
            module.set_launch_all_tiles(launch_all)
            count += 1
    return count


def count_sync_state(model: nn.Module) -> Dict[str, int]:
    """Count sparse policy modules by sync-relevant state.

    Use to verify timing readiness before measure_mode().

    Expected clean state: all counters zero except 'total' and
    launch mode counters.
    """
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

    for _, module in _iter_sparse_policy_modules(model):
        stats["total"] += 1

        if bool(getattr(module, "_inference_mode", False)):
            stats["inference_mode"] += 1
        if bool(getattr(module, "collect_diag", False)):
            stats["collect_diag"] += 1
        if bool(getattr(module, "profile_runtime", False)):
            stats["profile_runtime"] += 1
        if int(getattr(module, "_warmup_left", 0)) > 0:
            stats["warmup_pending"] += 1
        if bool(getattr(module, "_force_zero", False)):
            stats["force_zero"] += 1
        if bool(getattr(module, "_force_dense", False)):
            stats["force_dense"] += 1

        if hasattr(module, "launch_all_tiles"):
            if bool(getattr(module, "launch_all_tiles", False)):
                stats["launch_all"] += 1
            else:
                stats["launch_active"] += 1

    return stats


def estimate_sync_count(model: nn.Module) -> Dict[str, int]:
    """Estimate the number of GPU->CPU syncs per forward pass.

    Heuristic based on current module configuration.
    """
    result = {
        "zero_sync_modules": 0,
        "one_sync_modules": 0,
        "multi_sync_modules": 0,
        "total_min_syncs_per_forward": 0,
        "total_max_syncs_per_forward": 0,
    }

    for _, module in _iter_sparse_policy_modules(model):
        force_zero = bool(getattr(module, "_force_zero", False))
        force_dense = bool(getattr(module, "_force_dense", False))
        inference_mode = bool(getattr(module, "_inference_mode", False))
        has_launch_toggle = hasattr(module, "launch_all_tiles")
        launch_all = bool(getattr(module, "launch_all_tiles", False))

        if force_zero or force_dense:
            result["zero_sync_modules"] += 1
            continue

        if has_launch_toggle:
            if inference_mode:
                if launch_all:
                    result["zero_sync_modules"] += 1
                else:
                    result["one_sync_modules"] += 1
                    result["total_min_syncs_per_forward"] += 1
                    result["total_max_syncs_per_forward"] += 1
            else:
                if launch_all:
                    result["zero_sync_modules"] += 1
                    result["total_max_syncs_per_forward"] += 6
                else:
                    result["one_sync_modules"] += 1
                    result["total_min_syncs_per_forward"] += 1
                    result["total_max_syncs_per_forward"] += 6
        else:
            # Modules without tile-launch mode (e.g., SparseLinear):
            # inference_mode avoids periodic calibration syncs.
            if inference_mode:
                result["zero_sync_modules"] += 1
            else:
                result["multi_sync_modules"] += 1
                result["total_max_syncs_per_forward"] += 6

    return result
