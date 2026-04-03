"""
Utils/dispatch_model.py — Execution-Grounded Dispatch (EGD), v28.

Decides per-layer backend: "dense", "sparse", or "staticzero".

Design principle:
  The dispatch score is derived directly from what SparseFlow's sparse kernel
  does at runtime, not from a hand-crafted cost model with many coefficients.

SparseFlow execution semantics:
  1. Zero tiles → early return (cost ≈ 0)
  2. Non-zero tiles → iterate groups, skip inactive groups via bitmask

Therefore, the fraction of dense work saved by sparse execution is:

    R_l = z_l + (1 - z_l) × (1 - a_l)

  where z_l = TZR (tile-zero ratio) and a_l = AGR_nz (active group ratio
  over non-zero tiles).  Algebraically, R_l = 1 - AGR_overall.

Per-tile compute intensity (how much dense work each tile represents):

    I_l = M_l / N_l

  where M_l = dense MACs and N_l = total tile count.

Dispatch score (saved MACs per tile):

    S_l = R_l × I_l

Decision:
  - exact-zero input → staticzero
  - R_l < R_min → dense  (savings too thin for per-group bitmask overhead)
  - S_l > τ_k → sparse   (saved MACs per tile exceed prescan/metadata cost)
  - else → dense

Only 3 constants: R_min, τ_3x3, τ_1x1.
All have direct physical meaning tied to the sparse kernel's execution cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

# =============================================================================
# Dispatch constants  (3 total)
# =============================================================================
#
# τ_3x3 (TAU_3x3):
#   Minimum saved MACs per tile for 3×3 (and larger) kernels.
#   Physical meaning: the per-tile cost of prescan + metadata + bitmask
#   dispatch.  Prescan reads ~C_in × BLOCK_M values per tile and performs
#   per-group activity classification.  For typical layers (C_in=64-256,
#   BLOCK_M=16-64), this is ~200K-1M MACs equivalent in memory + compute
#   overhead.  500K is a mid-range estimate.
#   Calibrate: profile prescan latency on representative layers, convert to
#   MACs at GPU peak throughput, and set τ_3x3 to that value.
TAU_3x3 = 500_000

# τ_1x1 (TAU_1x1):
#   Minimum saved MACs per tile for 1×1 kernels.
#   Higher than τ_3x3 because cuDNN reduces 1×1 conv to a highly-optimized
#   GEMM call with near-peak throughput.  The sparse kernel must save
#   proportionally more to overcome this stronger baseline.
#   Calibrate: same procedure as τ_3x3 but comparing against cuDNN 1×1 perf.
TAU_1x1 = 1_000_000

# R_min (R_MIN):
#   Minimum saved work fraction for sparse to be viable.
#   Physical meaning: at very low R_l (high AGR), the sparse kernel iterates
#   almost all groups with per-group bitmask checks, and the savings from
#   the few skipped groups are consumed by this overhead.  Below R_min=0.15,
#   the sparse kernel does ≥85% of the dense kernel's work plus bitmask
#   overhead — guaranteed slower.
#   Calibrate: find the AGR at which sparse kernel latency equals dense
#   kernel latency on a compute-heavy layer; R_min = 1 - that AGR.
R_MIN = 0.15


# =============================================================================
# Utilities
# =============================================================================

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


_EPS = 1e-6


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

    # ── Core dispatch score fields (v28) ──
    R_l: float = 0.0           # saved work fraction: TZR + (1-TZR)(1-AGR_nz)
    I_l: float = 0.0           # per-tile intensity: MACs / N_tiles
    S_l: float = 0.0           # dispatch score: R_l × I_l (saved MACs per tile)
    agr_nz: float = 0.0        # AGR over non-zero tiles: AGR / (1 - TZR)
    n_tiles: float = 0.0       # total tile count (measured or estimated)
    tau: float = 0.0           # threshold used for this layer's kernel family

    # ── Backward-compatible fields ──
    score_sparse: float = 0.0  # S_l / tau (normalized: >1 → sparse)

    effective_benefit: float = 0.0   # alias for R_l
    total_overhead: float = 0.0      # tau / I_l (overhead as fraction of work)
    net_benefit: float = 0.0         # R_l - tau/I_l

    # ── Input measurements ──
    agr: float = -1.0          # overall AGR (active_group_ratio from kernel)
    tzr: float = -1.0          # tile zero ratio
    macs: float = 0.0
    denseish_ratio: float = -1.0
    sparse_tile_ratio: float = -1.0
    active_groups: float = 0.0
    total_groups: float = 0.0

    # ── Derived features for backward compat with bench reports ──
    efficiency: float = 0.0
    nz_fraction: float = 0.0
    denseish_among_nz: float = 0.0
    variable_overhead: float = 0.0
    divergence_overhead: float = 0.0
    fixed_overhead: float = 0.0
    eta_base: float = 0.0
    denseish_is_fallback: bool = False
    hysteresis_threshold: float = 0.0
    prior_backend: str = ""

    x_inactive: float = 0.0
    x_tzr: float = 0.0
    x_inter: float = 0.0
    x_log_macs: float = 0.0
    x_group_pressure: float = 0.0
    x_small: float = 0.0
    x_denseish: float = 0.0

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


def _target_get(target: Any, key: str, default=None):
    if isinstance(target, dict):
        return target.get(key, default)
    return getattr(target, key, default)


def _infer_batch_from_shape(input_shape: Optional[Tuple]) -> int:
    if input_shape is None:
        return 0
    shape = tuple(input_shape) if not isinstance(input_shape, str) else ()
    if len(shape) >= 5:
        return int(shape[0]) * int(shape[1])
    if len(shape) == 4:
        # Transformer blocks often use [T, B, N, C].
        return int(shape[0]) * int(shape[1])
    if len(shape) >= 1:
        return int(shape[0])
    return 0


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
    shape = tuple(input_shape) if (input_shape is not None and not isinstance(input_shape, str)) else ()
    seq_len = int(shape[-2]) if len(shape) >= 2 else int(_target_get(target, "input_h", 0))
    batch_size = _infer_batch_from_shape(input_shape)

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

    if op_type == "attention_qkav":
        macs = float(batch_size * num_heads * 2 * seq_len * seq_len * head_dim)
    elif op_type == "attention_linear":
        macs = float(batch_size * num_heads * 2 * seq_len * head_dim * head_dim)
    elif op_type == "attention_qkmix":
        macs = float(batch_size * num_heads * 4 * seq_len * head_dim * head_dim)
    else:
        # Conservative fallback for unknown attention variants.
        macs = float(batch_size * dim * dim)

    return ConvMeta(
        layer_name=layer_name,
        c_in=dim,
        c_out=dim,
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


def conv_meta_from_target(target: Any) -> ConvMeta:
    conv_module = _target_get(target, "module")
    if conv_module is None:
        conv_module = _target_get(target, "conv_module")
    layer_name = _target_get(target, "name", "")
    if not layer_name:
        layer_name = _target_get(target, "conv_name", "")
    input_shape = _target_get(target, "input_shape")
    op_type = _target_get(target, "op_type", "")
    attention_ops = {"attention_qkav", "attention_linear", "attention_qkmix"}

    if conv_module is not None:
        # Linear target from Core ReplacementTarget / dict target
        if hasattr(conv_module, "in_features") and hasattr(conv_module, "out_features"):
            c_in = int(conv_module.in_features)
            c_out = int(conv_module.out_features)
            batch_size = _infer_batch_from_shape(input_shape)
            macs = float(batch_size * c_in * c_out) if batch_size > 0 else 0.0
            return ConvMeta(
                layer_name=layer_name,
                c_in=c_in,
                c_out=c_out,
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
        if op_type in attention_ops or _is_attention_like_module(conv_module):
            return _attention_meta_from_target(target, conv_module, layer_name, input_shape, op_type)
        return conv_meta_from_module(conv_module, input_shape, layer_name)

    # Dict target (legacy bench path) or attribute-only Core target fallback.
    if op_type in attention_ops:
        return _attention_meta_from_target(target, None, layer_name, input_shape, op_type)

    if op_type == "linear":
        c_in = int(_target_get(target, "in_features", _target_get(target, "cin", _target_get(target, "c_in", 0))))
        c_out = int(_target_get(target, "out_features", _target_get(target, "cout", _target_get(target, "c_out", 0))))
        batch_size = _infer_batch_from_shape(input_shape)
        macs = float(batch_size * c_in * c_out) if batch_size > 0 else 0.0
        return ConvMeta(
            layer_name=layer_name,
            c_in=c_in,
            c_out=c_out,
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

    ks = _pair(_target_get(target, "kernel_size", (3, 3)))
    st = _pair(_target_get(target, "stride", (1, 1)))
    pad = _pair(_target_get(target, "padding", (1, 1)))
    dil = _pair(_target_get(target, "dilation", (1, 1)))
    groups = int(_target_get(target, "groups", 1))
    c_in = int(_target_get(target, "cin", _target_get(target, "c_in", 0)))
    c_out = int(_target_get(target, "cout", _target_get(target, "c_out", 0)))

    h_in, w_in, batch_size = 0, 0, 1
    if input_shape is not None:
        shape = tuple(input_shape) if not isinstance(input_shape, str) else ()
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


# =============================================================================
# Diagnostic mapping helpers
# =============================================================================

def _map_diag_to_agr_tzr(diag: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Extract AGR (overall) and TZR from kernel diagnostics.

    AGR = nonzero_group_count / total_group_count.
    This is the overall active group ratio across ALL tiles (including zero
    tiles that contribute 0 active groups).
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


def _get_n_tiles(diag: Dict[str, Any], meta: ConvMeta) -> float:
    """Get total tile count from diagnostics, or estimate from geometry.

    Prefers the measured value from kernel diagnostics (total_tile_count).
    Falls back to a geometric estimate using output spatial size and an
    approximate BLOCK_M.
    """
    n = diag.get('total_tile_count', -1.0)
    if n > 0:
        return float(n)

    # Geometric estimate
    h_out, w_out = meta.h_out, meta.w_out
    if h_out <= 0 or w_out <= 0:
        # No spatial info — return 1 to make I_l = MACs (conservative: high
        # per-tile intensity makes sparse easier to justify, but R_min and
        # the S_l threshold still guard against bad decisions)
        return 1.0

    block_m = _estimate_block_m(h_out, w_out)
    pixels = h_out * w_out
    n_spatial = max((pixels + block_m - 1) // block_m, 1)
    return float(meta.batch_size * n_spatial)


# =============================================================================
# Core dispatch logic
# =============================================================================

def make_dispatch_decision(diag: Dict[str, Any], meta: ConvMeta) -> DispatchDecision:
    """Execution-Grounded Dispatch for a single non-exact-zero layer.

    Computes:
        R_l = z + (1 - z)(1 - a)     saved work fraction
        I_l = MACs / N_tiles          per-tile intensity
        S_l = R_l × I_l              saved MACs per tile (dispatch score)

    Decision:
        R_l < R_min → dense   (savings consumed by per-group overhead)
        S_l > τ_k   → sparse  (saved MACs exceed per-tile dispatch cost)
        else        → dense
    """
    agr, tzr, active_groups, total_groups = _map_diag_to_agr_tzr(diag)
    denseish_ratio, sparse_tile_ratio = _map_diag_tile_mix(diag)
    macs = meta.macs
    groups = meta.groups

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

    # ── Structural guard ──

    if groups != 1:
        dec.backend = "dense"
        dec.reason = "groups!=1_unsupported"
        return dec

    if agr < 0 or tzr < 0:
        dec.backend = "dense"
        dec.reason = "no_diag_fallback_dense"
        return dec

    # ── Compute dispatch score ──

    # Saved work fraction R_l, derived from execution semantics:
    #   z tiles → fully skipped
    #   (1-z) tiles → skip (1-a) fraction of groups
    # where a = AGR_nz = average active group ratio over non-zero tiles
    nz_frac = max(1.0 - tzr, _EPS)
    agr_nz = min(agr / nz_frac, 1.0)
    R_l = tzr + nz_frac * (1.0 - agr_nz)
    # Note: algebraically R_l = 1 - agr, but the expanded form shows the
    # direct mapping to SparseFlow's two-level skip (tile + group).

    # Per-tile compute intensity
    n_tiles = _get_n_tiles(diag, meta)

    # For linear targets in Core-all-ops flow, input_shape can be unavailable,
    # which makes meta.macs = 0. Recover MACs using measured tile geometry.
    is_linear_diag = str(diag.get("kernel_type", "")).lower() == "linear"
    is_linear_meta = (
        meta.kernel_size == (1, 1)
        and meta.h_out <= 1
        and meta.w_out <= 1
        and meta.groups == 1
    )
    if macs <= 0 and is_linear_diag and is_linear_meta and meta.c_in > 0 and meta.c_out > 0:
        block_m = float(diag.get("block_m", -1))
        if block_m <= 0:
            block_m = 32.0
        n_rows_est = max(n_tiles * block_m, 1.0)
        macs = float(n_rows_est * meta.c_in * meta.c_out)
        dec.macs = macs

    I_l = macs / max(n_tiles, 1.0)

    # Dispatch score: saved MACs per tile
    S_l = R_l * I_l

    # Kernel-family threshold
    ks = meta.kernel_size
    is_1x1 = (ks == (1, 1) and groups == 1)
    tau = TAU_1x1 if is_1x1 else TAU_3x3

    # ── Record fields ──

    dec.R_l = R_l
    dec.I_l = I_l
    dec.S_l = S_l
    dec.agr_nz = agr_nz
    dec.n_tiles = n_tiles
    dec.nz_fraction = nz_frac
    dec.tau = tau

    # Normalized score: >1 means sparse, ≤1 means dense (when R_l ≥ R_min)
    dec.score_sparse = S_l / tau if tau > 0 else 0.0

    # Backward-compatible benefit/overhead view:
    # "benefit" = R_l, "overhead" = tau/I_l (threshold as fraction of work)
    dec.effective_benefit = R_l
    if I_l > 0:
        dec.total_overhead = tau / I_l
        dec.net_benefit = R_l - tau / I_l
    else:
        dec.total_overhead = float('inf')
        dec.net_benefit = -1.0

    # Legacy compat fields
    dec.x_inactive = 1.0 - agr
    dec.x_tzr = tzr
    dec.x_inter = (1.0 - agr) * tzr
    dec.x_log_macs = clamp01(math.log2(macs / 1e6 + 1.0) / 6.0)
    dec.x_group_pressure = agr * clamp01(
        active_groups / 4096.0 if active_groups >= 0 else 0.0
    )
    dec.x_small = 1.0 - dec.x_log_macs
    dec.x_denseish = clamp01(denseish_ratio if denseish_ratio >= 0 else 0.0)
    dec.efficiency = R_l  # legacy: closest concept to "efficiency"

    # ── Decision ──

    if R_l < R_MIN:
        dec.backend = "dense"
        dec.reason = f"R={R_l:.3f}<R_min={R_MIN}"
    elif S_l > tau:
        dec.backend = "sparse"
        dec.reason = f"S={S_l:.0f}>tau={tau:.0f}"
    else:
        dec.backend = "dense"
        dec.reason = f"S={S_l:.0f}<=tau={tau:.0f}"

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
        reason="exact_zero_input",
        agr=0.0,
        tzr=1.0,
        R_l=1.0,
        agr_nz=0.0,
        effective_benefit=1.0,
        net_benefit=1.0,
        score_sparse=float('inf'),
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
    prior_decisions: Optional[Dict[str, DispatchDecision]] = None,
) -> Dict[str, DispatchDecision]:
    """Run dispatch for all target layers.

    Args:
        targets: list of target dicts from analyze_targets().
        group_sparsity_data: per-layer diagnostic dict from measure_group_sparsity().
        zero_layers: set of layer names with exact-zero input (→ staticzero).
        prior_decisions: accepted for interface compatibility (unused in v28;
            the simplified score model is deterministic and does not require
            hysteresis for stability).

    Returns:
        Dict mapping layer_name → DispatchDecision.
    """
    if zero_layers is None:
        zero_layers = set()

    decisions: Dict[str, DispatchDecision] = {}
    for t in targets:
        name = _target_get(t, "name", "")
        if not name:
            name = _target_get(t, "conv_name", "")
        if not name:
            continue

        if name in zero_layers:
            decisions[name] = _make_staticzero_decision(name)
            continue

        diag = group_sparsity_data.get(name, {})
        meta = conv_meta_from_target(t)
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


# =============================================================================
# Reporting
# =============================================================================

def format_egd_report_header() -> str:
    """Column header for the EGD dispatch report."""
    return (
        f"  {'Layer':<40} {'AGR':>6} {'TZR':>6} {'AGR_nz':>7} "
        f"{'R_l':>6} {'I_l':>10} {'S_l':>10} {'τ':>10} {'S/τ':>6} "
        f"{'Dec':>8} {'Reason':>24}"
    )


def format_egd_report_row(dec: DispatchDecision) -> str:
    """Format one row of the EGD dispatch report."""
    name = dec.layer_name
    short = name if len(name) <= 39 else "..." + name[-36:]
    agr_s = f"{dec.agr:.3f}" if dec.agr >= 0 else "n/a"
    tzr_s = f"{dec.tzr:.3f}" if dec.tzr >= 0 else "n/a"
    agrnz_s = f"{dec.agr_nz:.3f}" if dec.agr_nz >= 0 else "n/a"
    rl_s = f"{dec.R_l:.3f}"
    il_s = f"{dec.I_l:.0f}" if dec.I_l > 0 else "n/a"
    sl_s = f"{dec.S_l:.0f}" if dec.S_l > 0 else "0"
    tau_s = f"{dec.tau:.0f}" if dec.tau > 0 else "n/a"
    ratio_s = f"{dec.score_sparse:.2f}" if dec.score_sparse < 1e6 else "inf"
    reason = dec.reason
    if len(reason) > 24:
        reason = reason[:21] + "..."
    return (
        f"  {short:<40} {agr_s:>6} {tzr_s:>6} {agrnz_s:>7} "
        f"{rl_s:>6} {il_s:>10} {sl_s:>10} {tau_s:>10} {ratio_s:>6} "
        f"{dec.backend:>8} {reason:>24}"
    )


def print_dispatch_decision_report(targets, group_sparsity_data, dispatch_decisions):
    """Print a detailed EGD dispatch report to stdout.

    Named to match the existing bench_4test.py call site.
    """
    print(f"\n  Execution-Grounded Dispatch Report (EGD v28)")
    print(f"  Constants: τ_3x3={TAU_3x3:,}  τ_1x1={TAU_1x1:,}  R_min={R_MIN}")
    print(f"  Score: S_l = R_l × I_l  (saved MACs per tile)")
    print(format_egd_report_header())
    print(f"  {'-'*150}")

    n_sparse = 0
    n_dense = 0
    n_staticzero = 0
    for t in targets:
        name = _target_get(t, "name", "")
        if not name:
            name = _target_get(t, "conv_name", "")
        if not name:
            continue
        dec = dispatch_decisions.get(name)
        if dec is None:
            continue
        print(format_egd_report_row(dec))
        if dec.backend == "sparse":
            n_sparse += 1
        elif dec.backend == "staticzero":
            n_staticzero += 1
        else:
            n_dense += 1

    total = n_sparse + n_dense + n_staticzero
    print(f"\n  Summary: {n_sparse} sparse + {n_dense} dense + {n_staticzero} staticzero "
          f"= {total} layers")


# Alias for v26/v27 compatibility — bench code may call either name
print_cbd_dispatch_report = print_dispatch_decision_report
