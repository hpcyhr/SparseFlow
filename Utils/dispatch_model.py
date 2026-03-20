"""
Utils/dispatch_model.py — AGR / TZR-based Conv dispatch model.

Decides per-layer backend: "dense", "sparse", or "staticzero"
based on empirical rules + linear scoring.

Usage:
    from Utils.dispatch_model import conv_meta_from_target, make_dispatch_decision

    meta = conv_meta_from_target(target_dict)
    decision = make_dispatch_decision(diag, meta)
    print(decision.backend, decision.reason, decision.score_sparse)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, Tuple

# =============================================================================
# Constants — empirical coefficients (implement exactly as specified)
# =============================================================================

# Hard gating thresholds
STATICZERO_AGR_MAX = 0.03
STATICZERO_TZR_MIN = 0.97

DENSE_AGR_MIN = 0.72
DENSE_TZR_MAX = 0.18

MIN_MACS_FOR_SPARSE = 8e5

# Sparse score threshold
SPARSE_SCORE_THRESHOLD = 0.55

# Sparse score weights
SPARSE_W_BIAS = -0.10
SPARSE_W_INACTIVE = 1.25
SPARSE_W_TZR = 0.95
SPARSE_W_INTER = 0.55
SPARSE_W_LOGMACS = 0.20
SPARSE_W_GROUP_PRESSURE = -0.85
SPARSE_W_SMALL = -0.25


# =============================================================================
# Utilities
# =============================================================================

def clamp01(x: float) -> float:
    """Clamp x to [0, 1]."""
    return max(0.0, min(1.0, x))


def _pair(x) -> Tuple[int, int]:
    """Convert int or tuple to (int, int)."""
    if isinstance(x, (tuple, list)):
        if len(x) == 1:
            return (int(x[0]), int(x[0]))
        return (int(x[0]), int(x[1]))
    return (int(x), int(x))


def infer_conv_output_hw(
    h_in: int, w_in: int,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int] = (1, 1),
) -> Tuple[int, int]:
    """Compute output spatial dimensions for a Conv2d layer."""
    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    return max(h_out, 0), max(w_out, 0)


def estimate_conv_macs(
    batch_size: int,
    h_out: int, w_out: int,
    c_out: int,
    c_in: int,
    kernel_size: Tuple[int, int],
    groups: int = 1,
) -> float:
    """Standard Conv2d MACs approximation: N * Hout * Wout * Cout * Kh * Kw * (Cin / groups)."""
    kh, kw = kernel_size
    cin_per_group = c_in / max(groups, 1)
    return float(batch_size * h_out * w_out * c_out * kh * kw * cin_per_group)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class ConvMeta:
    """Static metadata about a conv layer, derived from module + input shape."""
    layer_name: str = ""
    c_in: int = 0
    c_out: int = 0
    kernel_size: Tuple[int, int] = (3, 3)
    stride: Tuple[int, int] = (1, 1)
    padding: Tuple[int, int] = (1, 1)
    dilation: Tuple[int, int] = (1, 1)
    groups: int = 1
    h_in: int = 0
    w_in: int = 0
    h_out: int = 0
    w_out: int = 0
    batch_size: int = 1
    macs: float = 0.0


@dataclass
class DispatchDecision:
    """Full decision record for one conv layer."""
    backend: str = "dense"
    reason: str = ""
    score_sparse: float = 0.0

    agr: float = -1.0
    tzr: float = -1.0
    macs: float = 0.0

    x_inactive: float = 0.0
    x_tzr: float = 0.0
    x_inter: float = 0.0
    x_log_macs: float = 0.0
    x_group_pressure: float = 0.0
    x_small: float = 0.0

    active_groups: float = 0.0
    total_groups: float = 0.0

    # Extra context preserved for logging
    layer_name: str = ""
    groups: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Meta extraction
# =============================================================================

def conv_meta_from_module(conv_module, sample_x_shape: Optional[Tuple] = None,
                          layer_name: str = "") -> ConvMeta:
    """Extract ConvMeta from a nn.Conv2d module and an optional input shape.

    Args:
        conv_module: nn.Conv2d (or compatible) module.
        sample_x_shape: Shape of input tensor. Expected NCHW or 5-D (T,N,C,H,W).
                        If None, h_in/w_in/batch_size will be 0 and MACs will be 0.
        layer_name: Optional layer name for logging.
    """
    import torch.nn as nn

    ks = _pair(conv_module.kernel_size)
    st = _pair(conv_module.stride)
    pad = _pair(conv_module.padding)
    dil = _pair(conv_module.dilation)
    groups = int(conv_module.groups)
    c_in = int(conv_module.in_channels)
    c_out = int(conv_module.out_channels)

    h_in, w_in, batch_size = 0, 0, 1

    if sample_x_shape is not None:
        shape = tuple(sample_x_shape)
        # Handle 5-D (T, N, C, H, W) — common in SNN multi-step
        if len(shape) == 5:
            batch_size = shape[0] * shape[1]  # T * N
            h_in, w_in = shape[3], shape[4]
        elif len(shape) == 4:
            batch_size = shape[0]
            h_in, w_in = shape[2], shape[3]
        elif len(shape) == 3:
            # (N, C, L) — unlikely for conv2d but handle gracefully
            batch_size = shape[0]
            h_in, w_in = shape[2], 1
        # else: leave defaults

    h_out, w_out = infer_conv_output_hw(h_in, w_in, ks, st, pad, dil)
    macs = estimate_conv_macs(batch_size, h_out, w_out, c_out, c_in, ks, groups)

    return ConvMeta(
        layer_name=layer_name,
        c_in=c_in, c_out=c_out,
        kernel_size=ks, stride=st, padding=pad, dilation=dil,
        groups=groups,
        h_in=h_in, w_in=w_in,
        h_out=h_out, w_out=w_out,
        batch_size=batch_size,
        macs=macs,
    )


def conv_meta_from_target(target: Dict[str, Any]) -> ConvMeta:
    """Extract ConvMeta directly from a bench_4test target dict.

    The target dict is expected to have fields from analyze_targets():
        name, module, kernel_size, stride, padding, dilation, groups,
        input_shape, cin, cout
    We extract what's available and fall back gracefully.
    """
    conv_module = target.get("module")
    layer_name = target.get("name", "")
    input_shape = target.get("input_shape")

    if conv_module is not None:
        return conv_meta_from_module(conv_module, input_shape, layer_name)

    # Fallback: build from target dict fields directly
    ks = _pair(target.get("kernel_size", (3, 3)))
    st = _pair(target.get("stride", (1, 1)))
    pad = _pair(target.get("padding", (1, 1)))
    dil = _pair(target.get("dilation", (1, 1)))
    groups = int(target.get("groups", 1))
    c_in = int(target.get("cin", target.get("c_in", 0)))
    c_out = int(target.get("cout", target.get("c_out", 0)))

    h_in, w_in, batch_size = 0, 0, 1
    if input_shape is not None:
        shape = tuple(input_shape) if not isinstance(input_shape, str) else ()
        if len(shape) == 5:
            batch_size = shape[0] * shape[1]
            h_in, w_in = shape[3], shape[4]
        elif len(shape) == 4:
            batch_size = shape[0]
            h_in, w_in = shape[2], shape[3]

    h_out, w_out = infer_conv_output_hw(h_in, w_in, ks, st, pad, dil)
    macs = estimate_conv_macs(batch_size, h_out, w_out, c_out, c_in, ks, groups)

    return ConvMeta(
        layer_name=layer_name,
        c_in=c_in, c_out=c_out,
        kernel_size=ks, stride=st, padding=pad, dilation=dil,
        groups=groups,
        h_in=h_in, w_in=w_in,
        h_out=h_out, w_out=w_out,
        batch_size=batch_size,
        macs=macs,
    )


# =============================================================================
# Dispatch decision logic
# =============================================================================

def _map_diag_to_agr_tzr(diag: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Map existing group-sparsity diag fields into (agr, tzr, active_groups, total_groups).

    The existing diag from measure_group_sparsity() uses:
        active_group_ratio, tile_zero_ratio,
        nonzero_group_count, total_group_count
    We map these into the unified names.
    """
    agr = diag.get('active_group_ratio', -1.0)
    tzr = diag.get('tile_zero_ratio', -1.0)
    active_groups = diag.get('nonzero_group_count', -1.0)
    total_groups = diag.get('total_group_count', -1.0)

    # If agr is missing but we have counts, derive it
    if agr < 0 and total_groups > 0 and active_groups >= 0:
        agr = active_groups / total_groups

    # If tzr is missing but we have tile counts, derive it
    if tzr < 0:
        tile_zero_count = diag.get('tile_zero_count', -1.0)
        total_tile_count = diag.get('total_tile_count', -1.0)
        if total_tile_count > 0 and tile_zero_count >= 0:
            tzr = tile_zero_count / total_tile_count

    return agr, tzr, active_groups, total_groups


def make_dispatch_decision(
    diag: Dict[str, Any],
    meta: ConvMeta,
) -> DispatchDecision:
    """Compute dispatch decision for a single conv layer.

    Args:
        diag: Per-layer sparsity diagnostics dict (from measure_group_sparsity).
              May be empty/missing if diagnostics were not collected.
        meta: Static conv metadata (from conv_meta_from_target).

    Returns:
        DispatchDecision with backend, reason, score, and all features.
    """
    agr, tzr, active_groups, total_groups = _map_diag_to_agr_tzr(diag)
    macs = meta.macs
    groups = meta.groups

    # Initialize decision with defaults
    dec = DispatchDecision(
        layer_name=meta.layer_name,
        agr=agr,
        tzr=tzr,
        macs=macs,
        active_groups=active_groups,
        total_groups=total_groups,
        groups=groups,
    )

    # --- Rule 1: groups != 1 → dense ---
    if groups != 1:
        dec.backend = "dense"
        dec.reason = "groups!=1_keep_dense"
        return dec

    # If AGR/TZR are not available (diag not collected), fall back to dense
    if agr < 0 or tzr < 0:
        dec.backend = "dense"
        dec.reason = "no_diag_fallback_dense"
        return dec

    # --- Rule 2: hard gate → staticzero ---
    if agr <= STATICZERO_AGR_MAX or tzr >= STATICZERO_TZR_MIN:
        dec.backend = "staticzero"
        dec.reason = "hard_gate_staticzero"
        return dec

    # --- Rule 3: hard gate → dense ---
    if agr >= DENSE_AGR_MIN and tzr <= DENSE_TZR_MAX:
        dec.backend = "dense"
        dec.reason = "hard_gate_dense"
        return dec

    # --- Rule 4: tiny layer → dense ---
    if macs < MIN_MACS_FOR_SPARSE and tzr < 0.90:
        dec.backend = "dense"
        dec.reason = "tiny_layer_dense"
        return dec

    # --- Rule 5: compute sparse score ---
    x_inactive = 1.0 - agr
    x_tzr = tzr
    x_inter = x_inactive * x_tzr
    x_log_macs = clamp01(math.log2(macs / 1e6 + 1.0) / 6.0)
    x_group_pressure = agr * clamp01(active_groups / 4096.0 if active_groups >= 0 else 0.0)
    x_small = 1.0 - x_log_macs

    score_sparse = (
        SPARSE_W_BIAS
        + SPARSE_W_INACTIVE * x_inactive
        + SPARSE_W_TZR * x_tzr
        + SPARSE_W_INTER * x_inter
        + SPARSE_W_LOGMACS * x_log_macs
        + SPARSE_W_GROUP_PRESSURE * x_group_pressure
        + SPARSE_W_SMALL * x_small
    )

    dec.score_sparse = score_sparse
    dec.x_inactive = x_inactive
    dec.x_tzr = x_tzr
    dec.x_inter = x_inter
    dec.x_log_macs = x_log_macs
    dec.x_group_pressure = x_group_pressure
    dec.x_small = x_small

    if score_sparse >= SPARSE_SCORE_THRESHOLD:
        dec.backend = "sparse"
        dec.reason = f"score={score_sparse:.4f}>=threshold"
    else:
        dec.backend = "dense"
        dec.reason = f"score={score_sparse:.4f}<threshold"

    return dec


# =============================================================================
# Batch dispatch helper
# =============================================================================

def dispatch_all_layers(
    targets,
    group_sparsity_data: Dict[str, Dict],
    zero_layers=None,
) -> Dict[str, DispatchDecision]:
    """Run dispatch on all target layers.

    Args:
        targets: list of target dicts from analyze_targets().
        group_sparsity_data: {layer_name: diag_dict} from measure_group_sparsity().
        zero_layers: set of layer names with 100% element-level zero input.
                     These are forced to staticzero regardless of dispatch model.

    Returns:
        {layer_name: DispatchDecision}
    """
    if zero_layers is None:
        zero_layers = set()

    decisions = {}
    for t in targets:
        name = t["name"]

        # Layers already identified as 100% zero → staticzero (override)
        if name in zero_layers:
            dec = DispatchDecision(
                layer_name=name,
                backend="staticzero",
                reason="element_level_all_zero",
                agr=0.0,
                tzr=1.0,
            )
            decisions[name] = dec
            continue

        diag = group_sparsity_data.get(name, {})
        meta = conv_meta_from_target(t)
        dec = make_dispatch_decision(diag, meta)
        decisions[name] = dec

    return decisions


def decisions_to_sets(decisions: Dict[str, DispatchDecision]):
    """Convert decision dict into static_zero_set and sparse_set for replace_model.

    Returns:
        (static_zero_set, sparse_set)
        - static_zero_set: set of layer names to replace with StaticZeroConv2d
        - sparse_set: set of layer names to replace with SparseConv2d
        - (layers with backend="dense" are in neither set → kept dense)
    """
    static_zero_set = set()
    sparse_set = set()
    for name, dec in decisions.items():
        if dec.backend == "staticzero":
            static_zero_set.add(name)
        elif dec.backend == "sparse":
            sparse_set.add(name)
    return static_zero_set, sparse_set