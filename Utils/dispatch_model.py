"""
Utils/dispatch_model.py — conservative AGR / TZR / denseish-tile based Conv dispatch model.

Decides per-layer backend: "dense", "sparse", or "staticzero".
Important semantic rule:
  - Only exact zero-input layers should become "staticzero".
  - Non-exact layers are classified only between "dense" and "sparse".

This version makes the policy more conservative on early / dense-ish layers,
and uses dense-ish tile ratio explicitly to avoid over-routing front layers to
SparseConv.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

# =============================================================================
# Constants — conservative routing thresholds
# =============================================================================

# Exact-zero layers are handled outside this file via dispatch_all_layers(zero_layers=...).
# For non-exact layers, ultra-sparse patterns should prefer SparseConv, not StaticZero.
ULTRA_SPARSE_AGR_MAX = 0.03
ULTRA_SPARSE_TZR_MIN = 0.97

DENSE_AGR_MIN = 0.72
DENSE_TZR_MAX = 0.18

# Additional conservative dense gates
DENSEISH_RATIO_HIGH = 0.20
DENSEISH_RATIO_MID = 0.12
FRONT_DENSE_AGR_MIN = 0.35
FRONT_DENSE_TZR_MAX = 0.45
CONV1x1_DENSE_AGR_MIN = 0.20
CONV1x1_DENSEISH_MIN = 0.08
CONV1x1_TZR_MAX = 0.35

MIN_MACS_FOR_SPARSE = 8e5

# Sparse score threshold — raised vs previous version to force real DenseKeep.
SPARSE_SCORE_THRESHOLD = 1.05

# Sparse score weights
SPARSE_W_BIAS = -0.22
SPARSE_W_INACTIVE = 1.05
SPARSE_W_TZR = 0.85
SPARSE_W_INTER = 0.45
SPARSE_W_LOGMACS = 0.22
SPARSE_W_GROUP_PRESSURE = -1.05
SPARSE_W_SMALL = -0.28
SPARSE_W_DENSEISH = -1.10

# =============================================================================
# Utilities
# =============================================================================

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _pair(x) -> Tuple[int, int]:
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
    kh, kw = kernel_size
    cin_per_group = c_in / max(groups, 1)
    return float(batch_size * h_out * w_out * c_out * kh * kw * cin_per_group)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class ConvMeta:
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
    x_denseish: float = 0.0

    active_groups: float = 0.0
    total_groups: float = 0.0
    denseish_ratio: float = -1.0
    sparse_tile_ratio: float = -1.0

    layer_name: str = ""
    groups: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Meta extraction
# =============================================================================

def conv_meta_from_module(conv_module, sample_x_shape: Optional[Tuple] = None,
                          layer_name: str = "") -> ConvMeta:
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
        if len(shape) == 5:
            batch_size = shape[0] * shape[1]
            h_in, w_in = shape[3], shape[4]
        elif len(shape) == 4:
            batch_size = shape[0]
            h_in, w_in = shape[2], shape[3]
        elif len(shape) == 3:
            batch_size = shape[0]
            h_in, w_in = shape[2], 1

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
    conv_module = target.get("module")
    layer_name = target.get("name", "")
    input_shape = target.get("input_shape")

    if conv_module is not None:
        return conv_meta_from_module(conv_module, input_shape, layer_name)

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
# Diagnostic mapping helpers
# =============================================================================

def _map_diag_to_agr_tzr(diag: Dict[str, Any]) -> Tuple[float, float, float, float]:
    agr = diag.get('active_group_ratio', -1.0)
    tzr = diag.get('tile_zero_ratio', -1.0)
    active_groups = diag.get('nonzero_group_count', -1.0)
    total_groups = diag.get('total_group_count', -1.0)

    if agr < 0 and total_groups > 0 and active_groups >= 0:
        agr = active_groups / total_groups

    if tzr < 0:
        tile_zero_count = diag.get('tile_zero_count', -1.0)
        total_tile_count = diag.get('total_tile_count', -1.0)
        if total_tile_count > 0 and tile_zero_count >= 0:
            tzr = tile_zero_count / total_tile_count

    return agr, tzr, active_groups, total_groups


def _map_diag_tile_mix(diag: Dict[str, Any]) -> Tuple[float, float]:
    total_tile_count = diag.get('total_tile_count', -1.0)
    denseish_tiles = diag.get('denseish_tiles', -1.0)
    sparse_tiles = diag.get('sparse_tiles', -1.0)

    denseish_ratio = -1.0
    sparse_tile_ratio = -1.0
    if total_tile_count and total_tile_count > 0:
        if denseish_tiles >= 0:
            denseish_ratio = denseish_tiles / total_tile_count
        if sparse_tiles >= 0:
            sparse_tile_ratio = sparse_tiles / total_tile_count
    return denseish_ratio, sparse_tile_ratio


# =============================================================================
# Dispatch decision logic
# =============================================================================

def make_dispatch_decision(diag: Dict[str, Any], meta: ConvMeta) -> DispatchDecision:
    agr, tzr, active_groups, total_groups = _map_diag_to_agr_tzr(diag)
    denseish_ratio, sparse_tile_ratio = _map_diag_tile_mix(diag)
    macs = meta.macs
    groups = meta.groups
    ks = meta.kernel_size
    st = meta.stride
    is_1x1 = (ks == (1, 1) and groups == 1)

    dec = DispatchDecision(
        layer_name=meta.layer_name,
        agr=agr,
        tzr=tzr,
        macs=macs,
        active_groups=active_groups,
        total_groups=total_groups,
        denseish_ratio=denseish_ratio,
        sparse_tile_ratio=sparse_tile_ratio,
        groups=groups,
    )

    if groups != 1:
        dec.backend = "dense"
        dec.reason = "groups!=1_keep_dense"
        return dec

    if agr < 0 or tzr < 0:
        dec.backend = "dense"
        dec.reason = "no_diag_fallback_dense"
        return dec

    # Non-exact ultra-sparse layers should prefer SparseConv, not StaticZero.
    if agr <= ULTRA_SPARSE_AGR_MAX or tzr >= ULTRA_SPARSE_TZR_MIN:
        dec.backend = "sparse"
        dec.reason = "ultra_sparse_sparse"
        return dec

    # Strong dense gate for very active, low-zero layers.
    if agr >= DENSE_AGR_MIN and tzr <= DENSE_TZR_MAX:
        dec.backend = "dense"
        dec.reason = "hard_gate_dense"
        return dec

    # Dense-ish front layers: high AGR + many dense-ish tiles => keep dense.
    if denseish_ratio >= 0 and denseish_ratio >= DENSEISH_RATIO_HIGH and agr >= FRONT_DENSE_AGR_MIN and tzr <= FRONT_DENSE_TZR_MAX:
        dec.backend = "dense"
        dec.reason = "denseish_front_keep_dense"
        return dec

    # Slightly weaker dense-ish gate for very active early layers.
    if denseish_ratio >= 0 and denseish_ratio >= DENSEISH_RATIO_MID and agr >= 0.45 and tzr <= 0.55:
        dec.backend = "dense"
        dec.reason = "high_agr_denseish_keep_dense"
        return dec

    # 1x1 layers are more conservative: if reasonably active and not zero-heavy, keep dense.
    if is_1x1 and denseish_ratio >= 0 and denseish_ratio >= CONV1x1_DENSEISH_MIN and agr >= CONV1x1_DENSE_AGR_MIN and tzr <= CONV1x1_TZR_MAX:
        dec.backend = "dense"
        dec.reason = "conv1x1_denseish_keep_dense"
        return dec

    if macs < MIN_MACS_FOR_SPARSE and tzr < 0.90:
        dec.backend = "dense"
        dec.reason = "tiny_layer_dense"
        return dec

    x_inactive = 1.0 - agr
    x_tzr = tzr
    x_inter = x_inactive * x_tzr
    x_log_macs = clamp01(math.log2(macs / 1e6 + 1.0) / 6.0)
    x_group_pressure = agr * clamp01(active_groups / 4096.0 if active_groups >= 0 else 0.0)
    x_small = 1.0 - x_log_macs
    x_denseish = clamp01(denseish_ratio if denseish_ratio >= 0 else 0.0)

    score_sparse = (
        SPARSE_W_BIAS
        + SPARSE_W_INACTIVE * x_inactive
        + SPARSE_W_TZR * x_tzr
        + SPARSE_W_INTER * x_inter
        + SPARSE_W_LOGMACS * x_log_macs
        + SPARSE_W_GROUP_PRESSURE * x_group_pressure
        + SPARSE_W_SMALL * x_small
        + SPARSE_W_DENSEISH * x_denseish
    )

    dec.score_sparse = score_sparse
    dec.x_inactive = x_inactive
    dec.x_tzr = x_tzr
    dec.x_inter = x_inter
    dec.x_log_macs = x_log_macs
    dec.x_group_pressure = x_group_pressure
    dec.x_small = x_small
    dec.x_denseish = x_denseish

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

def dispatch_all_layers(targets, group_sparsity_data: Dict[str, Dict], zero_layers=None) -> Dict[str, DispatchDecision]:
    if zero_layers is None:
        zero_layers = set()

    decisions: Dict[str, DispatchDecision] = {}
    for t in targets:
        name = t["name"]
        if name in zero_layers:
            decisions[name] = DispatchDecision(
                layer_name=name,
                backend="staticzero",
                reason="element_level_all_zero",
                agr=0.0,
                tzr=1.0,
                denseish_ratio=0.0,
                sparse_tile_ratio=0.0,
            )
            continue

        diag = group_sparsity_data.get(name, {})
        meta = conv_meta_from_target(t)
        decisions[name] = make_dispatch_decision(diag, meta)

    return decisions


def decisions_to_sets(decisions: Dict[str, DispatchDecision]):
    static_zero_set = set()
    sparse_set = set()
    for name, dec in decisions.items():
        if dec.backend == "staticzero":
            static_zero_set.add(name)
        elif dec.backend == "sparse":
            sparse_set.add(name)
    return static_zero_set, sparse_set
