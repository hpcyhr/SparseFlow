"""
Utils/dispatch_model.py — Execution-Grounded Dispatch (EGD), v29.

Decides per-layer backend: "dense", "sparse", or "staticzero".

Design principle
================
The dispatch score is derived directly from what SparseFlow's sparse kernel
does at runtime, not from a hand-crafted cost model with many coefficients.

SparseFlow execution semantics
------------------------------
  1. Zero tiles     → early return (cost ≈ 0)
  2. Non-zero tiles → iterate groups, skip inactive groups via bitmask

Therefore, the fraction of dense work saved by sparse execution is

    R_l = z_l + (1 - z_l) × (1 - a_l)

where z_l = TZR (tile-zero ratio) and a_l = AGR_nz (active group ratio over
non-zero tiles). Algebraically, R_l = 1 - AGR_overall.

Per-tile compute intensity

    I_l = M_l / N_l

where M_l = dense MACs and N_l = total tile count.

Dispatch score (saved MACs per tile)

    S_l = R_l × I_l

Decision
--------
  - exact-zero input  → staticzero
  - R_l  < R_min      → dense (savings too thin for per-group bitmask overhead)
  - S_l  > τ_k        → sparse (saved MACs per tile exceed prescan/metadata cost)
  - else              → dense

Thresholds are execution-grounded but empirically calibrated:
R_min, τ_3x3, τ_1x1, τ_linear, τ_attn_linear, τ_attn_matmul.

Round 3 changes from v28 (no semantic changes to decision logic)
----------------------------------------------------------------
  - Deleted ~18 DispatchDecision fields that were written but never read by
    any code in the repository (CBD v27 feature-vector remnants: x_inactive,
    x_tzr, x_inter, x_log_macs, x_group_pressure, x_small, x_denseish,
    nz_fraction, variable_overhead, divergence_overhead, fixed_overhead,
    eta_base, denseish_is_fallback, hysteresis_threshold, prior_backend,
    effective_benefit, total_overhead, net_benefit, efficiency).
  - Deleted unused public helpers with zero callers:
        conv_meta_from_target, op_meta_from_module,
        decisions_to_backend_sets, summarize_decisions,
        format_egd_report_header, format_egd_report_row,
        print_dispatch_decision_report, print_cbd_dispatch_report,
        conv_meta_from_module_legacy, conv_meta_from_target_legacy.
    (bench_4test.py defines its own local print_dispatch_decision_report and
    imports only `dispatch_all_layers` and `decisions_to_sets`.)
  - Deleted unused parameters of dispatch_all_layers:
        prior_decisions (docstring already said "unused in v28"),
        return_summary  (depended on the now-deleted summarize_decisions).
  - Deleted fused_conv op_type branches in meta extraction; fused operators
    are out of scope for the current codebase.
  - Fixed ~30 mojibake'd characters throughout the file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

from Utils.config import (
    NUMERIC_EPS,
    DISPATCH_MIN_CONFIDENCE,
    DISPATCH_R_MARGIN,
    DISPATCH_R_MIN,
    DISPATCH_TAU_1X1,
    DISPATCH_TAU_3X3,
    DISPATCH_TAU_ATTN_LINEAR,
    DISPATCH_TAU_ATTN_MATMUL,
    DISPATCH_TAU_LINEAR,
    DISPATCH_TAU_MARGIN,
)

# =============================================================================
# Dispatch thresholds and guards
# =============================================================================
#
# These thresholds are semantically grounded in SparseFlow's execution path
# (tile skip + group skip), but are empirically calibrated rather than strict
# first-principles hardware costs.
#
# τ_3x3 (TAU_3x3)
#   Minimum saved MACs per tile for 3×3 (and larger) kernels.
#   Physical meaning: the per-tile cost of prescan + metadata + bitmask
#   dispatch. Prescan reads ~C_in × BLOCK_M values per tile and performs
#   per-group activity classification. For typical layers (C_in=64..256,
#   BLOCK_M=16..64), this is ~200K..1M MAC-equivalent memory + compute
#   overhead. 500K is a mid-range estimate.
#   Calibrate: profile prescan latency on representative layers, convert to
#   MACs at GPU peak throughput, and set τ_3x3 to that value.
TAU_3x3 = DISPATCH_TAU_3X3

# τ_1x1 (TAU_1x1)
#   Minimum saved MACs per tile for 1×1 kernels.
#   Higher than τ_3x3 because cuDNN reduces 1×1 conv to a highly-optimized
#   GEMM call with near-peak throughput. The sparse kernel must save
#   proportionally more to overcome this stronger baseline.
TAU_1x1 = DISPATCH_TAU_1X1

# Linear and attention thresholds (execution-grounded, empirically calibrated)
TAU_LINEAR = DISPATCH_TAU_LINEAR
TAU_ATTN_LINEAR = DISPATCH_TAU_ATTN_LINEAR
TAU_ATTN_MATMUL = DISPATCH_TAU_ATTN_MATMUL

# R_min
#   Minimum saved work fraction for sparse to be viable. At very low R_l
#   (high AGR), the sparse kernel iterates almost all groups with per-group
#   bitmask checks, and the savings from the few skipped groups are consumed
#   by this overhead. Below R_min=0.15, the sparse kernel does ≥85% of the
#   dense kernel's work plus bitmask overhead — guaranteed slower.
#   Calibrate: find the AGR at which sparse kernel latency equals dense
#   kernel latency on a compute-heavy layer; R_min = 1 - that AGR.
R_MIN = DISPATCH_R_MIN

# Margins are kept explicit for future stability tuning (default none).
R_MARGIN = DISPATCH_R_MARGIN
TAU_MARGIN = DISPATCH_TAU_MARGIN


# =============================================================================
# Utilities
# =============================================================================

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


_EPS = NUMERIC_EPS


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


def _estimate_block_m(h_out: int, w_out: int) -> int:
    """Estimate BLOCK_M (output pixels per tile) from output spatial size.

    Matches the kernel's _select_block_sizes logic approximately.
    Only used when total_tile_count is not available from diagnostics.
    """
    pixels = h_out * w_out
    if pixels >= 512:
        return 64
    if pixels >= 128:
        return 32
    if pixels >= 32:
        return 16
    return max(pixels, 1)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class OpMeta:
    layer_name: str = ""
    op_type: str = "unknown"
    meta_source: str = "measured"
    c_in: int = 0
    c_out: int = 0
    in_features: int = 0
    out_features: int = 0
    num_heads: int = 0
    seq_len: int = 0
    head_dim: int = 0
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


# Backward-compatible alias — used repo-wide as the type annotation for meta.
ConvMeta = OpMeta


@dataclass
class DispatchDecision:
    """Per-layer dispatch decision.

    Fields are grouped into (1) the decision outcome, (2) the v28 score
    triple, (3) raw input measurements, and (4) provenance metadata used by
    diagnostic reports and confidence gating. Fields that were set by v27
    but never read by any code path have been removed in v29.
    """

    # --- decision outcome ---
    backend: str = "dense"
    reason: str = ""
    reason_code: str = "init"
    confidence: float = 1.0
    fallback_reason: str = ""

    # --- provenance ---
    meta_source: str = "measured"       # measured | estimated | fallback | shortcut
    diag_source: str = "measured"       # measured | missing | shortcut
    support_status: str = "supported"   # supported | unsupported_groups | unsupported_op | exact_zero_shortcut
    score_family: str = "unknown"       # conv | linear | attn_linear | attn_matmul | matmul | bmm | none | unknown
    tile_source: str = "diag"           # diag | meta_geometry | estimated_default | unknown

    # --- v28 core score triple ---
    R_l: float = 0.0           # saved work fraction: tzr + (1-tzr)(1-agr_nz)
    I_l: float = 0.0           # per-tile intensity: macs / n_tiles
    S_l: float = 0.0           # dispatch score: R_l × I_l (saved MACs per tile)
    agr_nz: float = 0.0        # AGR over non-zero tiles: agr / (1 - tzr)
    n_tiles: float = 0.0       # total tile count (measured or estimated)
    tau: float = 0.0           # threshold used for this layer's kernel family

    # --- worst-case R guard (calibration-based, optional) ---
    agr_p90: float = -1.0      # p90 AGR from calibration batches (if available)
    r_worst: float = -1.0      # conservative saved-work lower bound (1 - agr_p90)

    # --- bench-facing convenience field ---
    # score_sparse = S_l / tau. Read by Benchmark/bench_4test.py for report
    # ordering and by format_egd_report_row-style printers.
    score_sparse: float = 0.0

    # --- raw input measurements (kept for JSON dumps) ---
    agr: float = -1.0          # overall AGR (active_group_ratio from kernel)
    tzr: float = -1.0          # tile zero ratio
    macs: float = 0.0
    denseish_ratio: float = -1.0
    sparse_tile_ratio: float = -1.0
    active_groups: float = 0.0
    total_groups: float = 0.0

    # --- identity ---
    layer_name: str = ""
    groups: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Meta extraction
# =============================================================================

def conv_meta_from_module(conv_module, sample_x_shape: Optional[Tuple] = None,
                          layer_name: str = "") -> ConvMeta:
    """Build a ConvMeta from a live nn.Conv2d (and an optional sample shape)."""
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
        op_type="conv",
        meta_source="measured",
        c_in=c_in, c_out=c_out,
        kernel_size=ks, stride=st, padding=pad, dilation=dil,
        groups=groups,
        h_in=h_in, w_in=w_in,
        h_out=h_out, w_out=w_out,
        batch_size=batch_size,
        macs=macs,
    )


def _target_get(target: Any, key: str, default=None):
    if isinstance(target, dict):
        return target.get(key, default)
    return getattr(target, key, default)


def _shape_tuple(input_shape: Optional[Tuple]) -> Tuple[int, ...]:
    if input_shape is None:
        return ()
    try:
        return tuple(int(x) for x in input_shape)
    except Exception:
        return ()


def _prod(values) -> int:
    out = 1
    for v in values:
        try:
            out *= int(v)
        except Exception:
            return 0
    return out


def _infer_conv_batch_from_shape(input_shape: Optional[Tuple]) -> int:
    shape = _shape_tuple(input_shape)
    if len(shape) == 5:
        return max(int(shape[0]) * int(shape[1]), 1)
    if len(shape) == 4:
        return max(int(shape[0]), 1)
    if len(shape) == 3:
        return max(int(shape[0]), 1)
    return 0


def _infer_linear_rows_from_shape(input_shape: Optional[Tuple]) -> int:
    shape = _shape_tuple(input_shape)
    if len(shape) == 0:
        return 0
    # Interpret as [..., features]: rows = prod of all leading dims.
    if len(shape) == 1:
        return 1
    return max(_prod(shape[:-1]), 1)


def _infer_attention_batch_from_shape(input_shape: Optional[Tuple]) -> int:
    shape = _shape_tuple(input_shape)
    if len(shape) == 4:
        return max(int(shape[0]) * int(shape[1]), 1)
    if len(shape) == 3:
        return max(int(shape[0]), 1)
    return 1


def _infer_batch_from_shape(input_shape: Optional[Tuple]) -> int:
    shape = _shape_tuple(input_shape)
    if len(shape) == 0:
        return 1
    return max(int(shape[0]), 1)


def _is_attention_like_module(module: Any) -> bool:
    if module is None:
        return False
    for attr in ("q", "k", "v", "proj"):
        if not hasattr(module, attr):
            return False
    return True


def _attention_meta_from_target(
    target: Any,
    conv_module: Any,
    layer_name: str,
    input_shape: Optional[Tuple],
    op_type: str,
) -> ConvMeta:
    shape = _shape_tuple(input_shape)
    seq_len = int(shape[-2]) if len(shape) >= 2 else int(_target_get(target, "input_h", 0))
    batch_size = _infer_attention_batch_from_shape(input_shape)

    dim = int(_target_get(target, "input_w", 0))
    num_heads = int(_target_get(target, "num_heads", 1))
    head_dim = int(_target_get(target, "head_dim", 0))

    if conv_module is not None:
        if hasattr(conv_module, "dim"):
            dim = int(getattr(conv_module, "dim"))
        elif hasattr(conv_module, "q") and hasattr(conv_module.q, "in_features"):
            dim = int(conv_module.q.in_features)
        if hasattr(conv_module, "num_heads"):
            num_heads = int(getattr(conv_module, "num_heads"))
        if hasattr(conv_module, "head_dim"):
            head_dim = int(getattr(conv_module, "head_dim"))

    dim = max(int(dim), 1)
    num_heads = max(int(num_heads), 1)
    if head_dim <= 0:
        head_dim = max(dim // num_heads, 1)
    seq_len = max(int(seq_len), 1)

    score_op_type = "attention_matmul"
    if op_type in ("attention_qkav", "attention_matmul"):
        macs = float(batch_size * num_heads * 2 * seq_len * seq_len * head_dim)
    elif op_type in ("attention_linear", "attention_proj_linear"):
        score_op_type = "attention_proj_linear"
        macs = float(batch_size * num_heads * 2 * seq_len * head_dim * head_dim)
    elif op_type == "attention_qkmix":
        score_op_type = "attention_proj_linear"
        macs = float(batch_size * num_heads * 4 * seq_len * head_dim * head_dim)
    else:
        # Conservative fallback for unknown attention variants.
        score_op_type = "attention_matmul"
        macs = float(batch_size * dim * dim)

    meta_source = "measured" if batch_size > 0 and seq_len > 0 and dim > 0 else "estimated"
    return ConvMeta(
        layer_name=layer_name,
        op_type=score_op_type,
        meta_source=meta_source,
        c_in=dim,
        c_out=dim,
        in_features=dim,
        out_features=dim,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        h_in=1,
        w_in=1,
        h_out=1,
        w_out=1,
        batch_size=batch_size,
        macs=macs,
    )


def _target_layer_name(target: Any) -> str:
    name = _target_get(target, "name", "")
    if not name:
        name = _target_get(target, "conv_name", "")
    return str(name)


def _linear_meta_from_target(
    target: Any,
    linear_module: Optional[Any],
    layer_name: str,
    input_shape: Optional[Tuple],
) -> ConvMeta:
    if linear_module is not None:
        c_in = int(getattr(linear_module, "in_features", 0))
        c_out = int(getattr(linear_module, "out_features", 0))
    else:
        c_in = int(_target_get(target, "in_features", _target_get(target, "cin", _target_get(target, "c_in", 0))))
        c_out = int(_target_get(target, "out_features", _target_get(target, "cout", _target_get(target, "c_out", 0))))

    batch_size = _infer_linear_rows_from_shape(input_shape)
    meta_source = "measured" if batch_size > 0 else "estimated"
    if batch_size <= 0:
        batch_size = int(_target_get(target, "batch_size", 0))
    if batch_size <= 0:
        batch_size = 1
        meta_source = "fallback"

    macs = float(batch_size * c_in * c_out) if c_in > 0 and c_out > 0 else 0.0
    return ConvMeta(
        layer_name=layer_name,
        op_type="linear",
        meta_source=meta_source,
        c_in=c_in,
        c_out=c_out,
        in_features=c_in,
        out_features=c_out,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        h_in=1,
        w_in=1,
        h_out=1,
        w_out=1,
        batch_size=batch_size,
        macs=macs,
    )


def _matmul_like_meta_from_target(
    target: Any,
    layer_name: str,
    input_shape: Optional[Tuple],
    op_type: str,
) -> ConvMeta:
    in_shape = _shape_tuple(input_shape)
    out_shape = _shape_tuple(_target_get(target, "output_shape"))

    # Expect [..., M, K] @ [..., K, N] -> [..., M, N]
    k_dim = int(in_shape[-1]) if len(in_shape) >= 1 else int(_target_get(target, "k", 0))
    m_dim = int(in_shape[-2]) if len(in_shape) >= 2 else int(_target_get(target, "m", 0))
    n_dim = int(out_shape[-1]) if len(out_shape) >= 1 else int(_target_get(target, "n", 0))

    if n_dim <= 0:
        n_dim = int(_target_get(target, "out_features", _target_get(target, "c_out", 0)))

    if len(out_shape) >= 3:
        batch = max(_prod(out_shape[:-2]), 1)
    elif len(in_shape) >= 3:
        batch = max(_prod(in_shape[:-2]), 1)
    else:
        batch = int(_target_get(target, "batch_size", 1))
        if batch <= 0:
            batch = 1

    if m_dim <= 0 and len(out_shape) >= 2:
        m_dim = int(out_shape[-2])
    if m_dim <= 0:
        m_dim = 1
    if k_dim <= 0:
        k_dim = 1
    if n_dim <= 0:
        n_dim = 1

    macs = float(batch * m_dim * k_dim * n_dim)
    meta_source = "measured" if len(in_shape) > 0 and len(out_shape) > 0 else "estimated"
    return ConvMeta(
        layer_name=layer_name,
        op_type=op_type,
        meta_source=meta_source,
        c_in=k_dim,
        c_out=n_dim,
        in_features=k_dim,
        out_features=n_dim,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        h_in=1,
        w_in=1,
        h_out=1,
        w_out=1,
        batch_size=batch,
        macs=macs,
    )


def _conv_meta_from_target(
    target: Any,
    conv_module: Optional[Any],
    layer_name: str,
    input_shape: Optional[Tuple],
) -> ConvMeta:
    op_hint = str(_target_get(target, "op_type", "conv"))
    if conv_module is not None:
        meta = conv_meta_from_module(conv_module, input_shape, layer_name)
        h_in, w_in = meta.h_in, meta.w_in
        batch_size = int(meta.batch_size)
        c_in, c_out = meta.c_in, meta.c_out
        ks, st, pad, dil, groups = meta.kernel_size, meta.stride, meta.padding, meta.dilation, meta.groups
        meta_source = "measured"
    else:
        ks = _pair(_target_get(target, "kernel_size", (3, 3)))
        st = _pair(_target_get(target, "stride", (1, 1)))
        pad = _pair(_target_get(target, "padding", (1, 1)))
        dil = _pair(_target_get(target, "dilation", (1, 1)))
        groups = int(_target_get(target, "groups", 1))
        c_in = int(_target_get(target, "cin", _target_get(target, "c_in", 0)))
        c_out = int(_target_get(target, "cout", _target_get(target, "c_out", 0)))
        h_in, w_in = 0, 0
        batch_size = 0
        meta_source = "fallback"

    if h_in <= 0 or w_in <= 0:
        h_t = int(_target_get(target, "input_h", 0))
        w_t = int(_target_get(target, "input_w", 0))
        if h_t > 0 and w_t > 0:
            h_in, w_in = h_t, w_t
            if meta_source == "measured":
                meta_source = "estimated"

    if batch_size <= 0:
        batch_size = _infer_conv_batch_from_shape(input_shape)
        if batch_size > 0 and meta_source == "fallback":
            meta_source = "estimated"
    if batch_size <= 0:
        batch_size = int(_target_get(target, "batch_size", 0))
        if batch_size > 0 and meta_source == "fallback":
            meta_source = "estimated"
    if batch_size <= 0:
        batch_size = 1
        meta_source = "fallback"

    h_out, w_out = infer_conv_output_hw(h_in, w_in, ks, st, pad, dil)
    macs = 0.0
    if c_in > 0 and c_out > 0 and h_out > 0 and w_out > 0:
        macs = estimate_conv_macs(batch_size, h_out, w_out, c_out, c_in, ks, groups)

    if op_hint in ("depthwise_conv2d", "grouped_conv2d"):
        op_type = op_hint
    else:
        op_type = "conv"

    return ConvMeta(
        layer_name=layer_name,
        op_type=op_type,
        meta_source=meta_source,
        c_in=c_in,
        c_out=c_out,
        kernel_size=ks,
        stride=st,
        padding=pad,
        dilation=dil,
        groups=groups,
        h_in=h_in,
        w_in=w_in,
        h_out=h_out,
        w_out=w_out,
        batch_size=batch_size,
        macs=macs,
    )


def _fallback_meta_from_target(target: Any, layer_name: str, input_shape: Optional[Tuple]) -> ConvMeta:
    # Conservative fallback keeps dispatch in the dense path unless diagnostics
    # are strongly favorable.
    return _conv_meta_from_target(target, None, layer_name, input_shape)


def op_meta_from_target(target: Any) -> ConvMeta:
    """Dispatch-time entry point: extract a ConvMeta from a target record."""
    conv_module = _target_get(target, "module")
    if conv_module is None:
        conv_module = _target_get(target, "conv_module")
    layer_name = _target_layer_name(target)
    input_shape = _target_get(target, "input_shape")
    op_type = str(_target_get(target, "op_type", ""))
    attention_ops = {
        "attention_qkav",
        "attention_linear",
        "attention_qkmix",
        "attention_matmul",
        "attention_proj_linear",
    }
    matmul_ops = {"matmul", "bmm"}

    if conv_module is not None:
        if hasattr(conv_module, "in_features") and hasattr(conv_module, "out_features"):
            return _linear_meta_from_target(target, conv_module, layer_name, input_shape)
        if op_type in attention_ops or _is_attention_like_module(conv_module):
            return _attention_meta_from_target(target, conv_module, layer_name, input_shape, op_type)
        if hasattr(conv_module, "kernel_size"):
            return _conv_meta_from_target(target, conv_module, layer_name, input_shape)

    if op_type in attention_ops:
        return _attention_meta_from_target(target, None, layer_name, input_shape, op_type)
    if op_type in matmul_ops:
        return _matmul_like_meta_from_target(target, layer_name, input_shape, op_type)
    if op_type == "linear":
        return _linear_meta_from_target(target, None, layer_name, input_shape)
    if op_type.startswith("conv") or op_type in ("depthwise_conv2d", "grouped_conv2d"):
        return _conv_meta_from_target(target, None, layer_name, input_shape)
    return _fallback_meta_from_target(target, layer_name, input_shape)


# =============================================================================
# Diagnostic mapping helpers
# =============================================================================

def _map_diag_to_agr_tzr(diag: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Extract overall AGR and TZR from kernel diagnostics.

    AGR = nonzero_group_count / total_group_count. This is the overall active
    group ratio across ALL tiles (including zero tiles that contribute 0
    active groups).
    """
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


def _get_n_tiles(diag: Dict[str, Any], meta: ConvMeta) -> Tuple[float, str]:
    """Get total tile count from diagnostics, or estimate from geometry.

    Prefers the measured value from kernel diagnostics (total_tile_count).
    Falls back to a geometric estimate using output spatial size and an
    approximate BLOCK_M.
    """
    n = diag.get('total_tile_count', -1.0)
    if n > 0:
        return float(n), "diag"

    h_out, w_out = meta.h_out, meta.w_out
    if h_out <= 0 or w_out <= 0:
        # No spatial info — return 1 so I_l = MACs (conservative: high per-tile
        # intensity makes sparse easier to justify, but R_min and the S_l
        # threshold still guard against bad decisions).
        return 1.0, "estimated_default"

    block_m = _estimate_block_m(h_out, w_out)
    pixels = h_out * w_out
    n_spatial = max((pixels + block_m - 1) // block_m, 1)
    batch = int(meta.batch_size) if int(meta.batch_size) > 0 else 1
    return float(batch * n_spatial), "meta_geometry"


def _select_tau(meta: ConvMeta, diag: Dict[str, Any]) -> Tuple[float, str]:
    """Pick the threshold (τ) and score family for a layer.

    Resolution order:
      1. explicit op_type on the meta (linear / matmul / bmm / attention_*)
      2. kernel_type hint on the diag dict
      3. convolution footprint (1x1 vs 3x3+) as a final default
    """
    op_type = str(getattr(meta, "op_type", "unknown")).lower()
    kernel_type = str(diag.get("kernel_type", "")).lower()

    if op_type == "matmul" or kernel_type == "matmul":
        return TAU_ATTN_MATMUL, "matmul"
    if op_type == "bmm" or kernel_type == "bmm":
        return TAU_ATTN_MATMUL, "bmm"
    if op_type == "linear":
        return TAU_LINEAR, "linear"
    if op_type == "attention_proj_linear":
        return TAU_ATTN_LINEAR, "attn_linear"
    if op_type == "attention_matmul":
        return TAU_ATTN_MATMUL, "attn_matmul"
    if "attention" in kernel_type and "linear" in kernel_type:
        return TAU_ATTN_LINEAR, "attn_linear"
    if "attention" in kernel_type:
        return TAU_ATTN_MATMUL, "attn_matmul"
    if kernel_type == "linear":
        return TAU_LINEAR, "linear"

    # Fallback by kernel footprint.
    ks = meta.kernel_size
    if ks == (1, 1):
        return TAU_1x1, "conv"
    return TAU_3x3, "conv"


# =============================================================================
# Core dispatch logic
# =============================================================================

def make_dispatch_decision(diag: Dict[str, Any], meta: ConvMeta) -> DispatchDecision:
    """Execution-grounded dispatch for a single non-exact-zero layer."""
    agr, tzr, active_groups, total_groups = _map_diag_to_agr_tzr(diag)
    denseish_ratio, sparse_tile_ratio = _map_diag_tile_mix(diag)
    macs = meta.macs
    groups = meta.groups
    meta_source = str(getattr(meta, "meta_source", "measured"))
    op_type = str(getattr(meta, "op_type", "unknown"))

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
        meta_source=meta_source,
        diag_source="measured",
        support_status="supported",
        score_family="unknown",
        tile_source="diag",
    )

    # Support / sanity gates
    if groups != 1 and op_type not in ("grouped_conv2d", "depthwise_conv2d"):
        dec.backend = "dense"
        dec.support_status = "unsupported_groups"
        dec.reason_code = "unsupported_groups"
        dec.reason = "unsupported_groups!=1"
        dec.fallback_reason = dec.reason_code
        dec.confidence = 1.0
        return dec

    if op_type == "unknown":
        dec.backend = "dense"
        dec.support_status = "unsupported_op"
        dec.reason_code = "unsupported_op"
        dec.reason = "unsupported_op_fallback_dense"
        dec.fallback_reason = dec.reason_code
        dec.confidence = 0.7
        return dec

    if agr < 0 or tzr < 0:
        dec.backend = "dense"
        dec.diag_source = "missing"
        dec.reason_code = "missing_diag"
        dec.reason = "missing_diag_fallback_dense"
        dec.fallback_reason = dec.reason_code
        dec.confidence = 0.2
        return dec

    # ---- v28 score: R_l, I_l, S_l ----
    nz_frac = max(1.0 - tzr, _EPS)
    agr_nz = min(agr / nz_frac, 1.0)
    R_l_exec = tzr + nz_frac * (1.0 - agr_nz)
    R_l_simple = 1.0 - agr
    R_l = R_l_exec
    r_mismatch = abs(R_l_exec - R_l_simple)

    agr_p90 = float(diag.get("active_group_ratio_p90", -1.0))
    r_worst = (1.0 - agr_p90) if agr_p90 >= 0 else -1.0

    n_tiles, tile_source = _get_n_tiles(diag, meta)
    dec.tile_source = tile_source

    # Fallback MACs estimation when meta.macs wasn't filled in (e.g. linear
    # meta with unknown batch, or conv meta with unknown H/W).
    kernel_type = str(diag.get("kernel_type", "")).lower()
    is_linear_diag = kernel_type == "linear"
    is_linear_meta = (
        meta.kernel_size == (1, 1)
        and meta.h_out <= 1
        and meta.w_out <= 1
        and meta.groups == 1
    )
    macs_estimated = False
    if macs <= 0 and is_linear_diag and is_linear_meta and meta.c_in > 0 and meta.c_out > 0:
        block_m = float(diag.get("block_m", -1))
        if block_m <= 0:
            block_m = 32.0
        n_rows_est = max(n_tiles * block_m, 1.0)
        macs = float(n_rows_est * meta.c_in * meta.c_out)
        macs_estimated = True
    elif macs <= 0 and meta.c_in > 0 and meta.c_out > 0 and meta.h_out > 0 and meta.w_out > 0:
        block_m = float(diag.get("block_m", -1))
        if block_m <= 0:
            block_m = float(_estimate_block_m(meta.h_out, meta.w_out))
        n_spatial = max(math.ceil((meta.h_out * meta.w_out) / max(block_m, 1.0)), 1.0)
        batch_est = max(n_tiles / n_spatial, 1.0)
        kh, kw = meta.kernel_size
        cin_per_group = meta.c_in / max(meta.groups, 1)
        macs = float(batch_est * meta.h_out * meta.w_out * meta.c_out * kh * kw * cin_per_group)
        macs_estimated = True

    if macs_estimated:
        dec.meta_source = "estimated" if dec.meta_source == "measured" else dec.meta_source
    dec.macs = macs

    if macs <= 0:
        dec.backend = "dense"
        dec.reason_code = "missing_meta"
        dec.reason = "missing_meta_fallback_dense"
        dec.fallback_reason = dec.reason_code
        dec.confidence = 0.2
        dec.meta_source = "fallback" if dec.meta_source == "measured" else dec.meta_source
        return dec

    I_l = macs / max(n_tiles, 1.0)
    S_l = R_l * I_l
    tau, tau_family = _select_tau(meta, diag)
    tau_gate = tau * (1.0 + TAU_MARGIN)
    r_gate = R_MIN + R_MARGIN
    dec.score_family = tau_family

    dec.R_l = R_l
    dec.I_l = I_l
    dec.S_l = S_l
    dec.agr_nz = agr_nz
    dec.agr_p90 = agr_p90
    dec.r_worst = r_worst
    dec.n_tiles = n_tiles
    dec.tau = tau
    dec.score_sparse = S_l / tau if tau > 0 else 0.0

    reason_suffix = ""
    if macs_estimated:
        reason_suffix += "|macs_estimated"
    if r_mismatch > 1e-4:
        reason_suffix += "|R_mismatch_check"

    # ---- decision gates ----
    if r_worst >= 0 and r_worst < r_gate:
        dec.backend = "dense"
        dec.reason_code = "R_guard_worst"
        dec.reason = f"R_worst={r_worst:.3f}<R_min={r_gate:.3f}{reason_suffix}"
        dec.fallback_reason = dec.reason_code
    elif R_l < r_gate:
        dec.backend = "dense"
        dec.reason_code = "R_guard"
        dec.reason = f"R={R_l:.3f}<R_min={r_gate:.3f}{reason_suffix}"
        dec.fallback_reason = dec.reason_code
    elif S_l > tau_gate:
        dec.backend = "sparse"
        dec.reason_code = "score_pass"
        dec.reason = f"S={S_l:.0f}>tau={tau_gate:.0f}{reason_suffix}"
        dec.fallback_reason = ""
    else:
        dec.backend = "dense"
        dec.reason_code = "score_fail"
        dec.reason = f"S={S_l:.0f}<=tau={tau_gate:.0f}{reason_suffix}"
        dec.fallback_reason = dec.reason_code

    # ---- confidence + low-confidence gate ----
    confidence = 1.0
    if dec.diag_source != "measured":
        confidence *= 0.5
    if dec.meta_source != "measured":
        confidence *= 0.75
    if dec.tile_source != "diag":
        confidence *= 0.9
    if r_mismatch > 1e-4:
        confidence *= 0.8
    dec.confidence = clamp01(confidence)

    # Estimated/fallback metadata rows can be printed and analyzed, but they
    # should not force sparse execution when confidence is too low.
    if dec.backend == "sparse" and dec.confidence < DISPATCH_MIN_CONFIDENCE:
        dec.backend = "dense"
        dec.reason_code = "low_confidence"
        dec.reason = f"low_confidence<{DISPATCH_MIN_CONFIDENCE:.2f}{reason_suffix}"
        dec.fallback_reason = dec.reason_code

    return dec


# =============================================================================
# StaticZero decision helper
# =============================================================================

def _make_staticzero_decision(layer_name: str) -> DispatchDecision:
    """DispatchDecision for an exact-zero-input layer.

    StaticZero is the extreme of the dispatch surface:
      R_l = 1.0 (all work saved), S_l → ∞ for any finite I_l.
    """
    return DispatchDecision(
        layer_name=layer_name,
        backend="staticzero",
        reason="exact_zero_input_shortcut",
        reason_code="exact_zero",
        fallback_reason="exact_zero_shortcut",
        meta_source="shortcut",
        diag_source="shortcut",
        support_status="exact_zero_shortcut",
        score_family="none",
        tile_source="unknown",
        confidence=1.0,
        agr=0.0,
        tzr=1.0,
        R_l=1.0,
        agr_nz=0.0,
        score_sparse=float('inf'),
        denseish_ratio=0.0,
        sparse_tile_ratio=0.0,
    )


def _make_pool_decision(
    layer_name: str,
    op_type: str,
    pool_module: Optional[Any] = None,
) -> DispatchDecision:
    """DispatchDecision for MaxPool2d / AvgPool2d layers (Round 5.5b).

    Pool operators have zero MACs, so the MAC-centric EGD cost model
    (`S_l = R_l × I_l` with I_l = MACs / N_tiles) is undefined for them.
    Rather than extending the cost model, we short-circuit pool layers to
    always prefer the sparse kernel backend — the sparse pool kernels have
    a cheap tile-zero fast path and degrade gracefully to dense on tiles
    with activity. There is no hysteresis and no per-layer calibration.
    Current policy: dispatch pool layers to dense by default unless an explicit
    sparse-pool benchmark justifies enabling them.

    Only the currently supported sparse-pool semantics are even eligible for
    sparse replacement:
      - `ceil_mode=False` for both MaxPool2d and AvgPool2d
      - `return_indices=False` for MaxPool2d
    Unsupported pool variants are forced to dense so reports match runtime.
    """
    if pool_module is not None:
        if bool(getattr(pool_module, "ceil_mode", False)):
            return DispatchDecision(
                layer_name=layer_name,
                backend="dense",
                reason=f"unsupported_pool_ceil_mode_{op_type}",
                reason_code="unsupported_pool_ceil_mode",
                fallback_reason="unsupported_pool_ceil_mode",
                meta_source="shortcut",
                diag_source="shortcut",
                support_status="unsupported_op",
                score_family="pool",
                tile_source="unknown",
                confidence=1.0,
                agr=0.0,
                tzr=0.0,
                R_l=0.0,
                agr_nz=0.0,
                score_sparse=0.0,
                denseish_ratio=0.0,
                sparse_tile_ratio=0.0,
            )
        if op_type == "maxpool2d" and bool(getattr(pool_module, "return_indices", False)):
            return DispatchDecision(
                layer_name=layer_name,
                backend="dense",
                reason="unsupported_pool_return_indices_maxpool2d",
                reason_code="unsupported_pool_return_indices",
                fallback_reason="unsupported_pool_return_indices",
                meta_source="shortcut",
                diag_source="shortcut",
                support_status="unsupported_op",
                score_family="pool",
                tile_source="unknown",
                confidence=1.0,
                agr=0.0,
                tzr=0.0,
                R_l=0.0,
                agr_nz=0.0,
                score_sparse=0.0,
                denseish_ratio=0.0,
                sparse_tile_ratio=0.0,
            )

    return DispatchDecision(
        layer_name=layer_name,
        backend="dense",
        reason=f"pool_default_dense_{op_type}",
        reason_code="pool_default_dense",
        fallback_reason="pool_default_dense",
        meta_source="shortcut",
        diag_source="shortcut",
        support_status="supported",
        score_family="pool",
        tile_source="unknown",
        confidence=1.0,
        agr=0.0,
        tzr=0.0,
        R_l=0.0,
        agr_nz=0.0,
        score_sparse=0.0,
        denseish_ratio=0.0,
        sparse_tile_ratio=0.0,
    )


# =============================================================================
# Batch dispatch
# =============================================================================

def dispatch_all_layers(
    targets,
    group_sparsity_data: Dict[str, Dict],
    zero_layers=None,
) -> Dict[str, DispatchDecision]:
    """Run dispatch for all target layers.

    Args:
        targets: list of target dicts from analyze_targets().
        group_sparsity_data: per-layer diagnostic dict from measure_group_sparsity().
        zero_layers: set of layer names with exact-zero input (→ staticzero).

    Returns:
        Dict mapping layer_name → DispatchDecision.
    """
    if zero_layers is None:
        zero_layers = set()

    decisions: Dict[str, DispatchDecision] = {}
    _POOL_OPS = {"maxpool2d", "avgpool2d"}
    for t in targets:
        name = _target_layer_name(t)
        if not name:
            continue

        op_type = str(_target_get(t, "op_type", ""))
        if op_type in _POOL_OPS:
            pool_module = _target_get(t, "module")
            if pool_module is None:
                pool_module = _target_get(t, "conv_module")
            decisions[name] = _make_pool_decision(name, op_type, pool_module)
            continue

        if name in zero_layers:
            decisions[name] = _make_staticzero_decision(name)
            continue

        diag = group_sparsity_data.get(name, {})
        meta = op_meta_from_target(t)
        decisions[name] = make_dispatch_decision(diag, meta)

    return decisions


def decisions_to_sets(decisions: Dict[str, DispatchDecision]):
    """Convert decisions dict to (static_zero_set, sparse_set) for model building."""
    static_zero_set = set()
    sparse_set = set()
    for name, dec in decisions.items():
        if dec.backend == "staticzero":
            static_zero_set.add(name)
        elif dec.backend == "sparse":
            sparse_set.add(name)
    return static_zero_set, sparse_set
