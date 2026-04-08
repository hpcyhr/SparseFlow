"""
Utils/dispatch_model.py 鈥?Execution-Grounded Dispatch (EGD), v28.

Decides per-layer backend: "dense", "sparse", or "staticzero".

Design principle:
  The dispatch score is derived directly from what SparseFlow's sparse kernel
  does at runtime, not from a hand-crafted cost model with many coefficients.

SparseFlow execution semantics:
  1. Zero tiles 鈫?early return (cost 鈮?0)
  2. Non-zero tiles 鈫?iterate groups, skip inactive groups via bitmask

Therefore, the fraction of dense work saved by sparse execution is:

    R_l = z_l + (1 - z_l) 脳 (1 - a_l)

  where z_l = TZR (tile-zero ratio) and a_l = AGR_nz (active group ratio
  over non-zero tiles).  Algebraically, R_l = 1 - AGR_overall.

Per-tile compute intensity (how much dense work each tile represents):

    I_l = M_l / N_l

  where M_l = dense MACs and N_l = total tile count.

Dispatch score (saved MACs per tile):

    S_l = R_l 脳 I_l

Decision:
  - exact-zero input 鈫?staticzero
  - R_l < R_min 鈫?dense  (savings too thin for per-group bitmask overhead)
  - S_l > 蟿_k 鈫?sparse   (saved MACs per tile exceed prescan/metadata cost)
  - else 鈫?dense

Thresholds are execution-grounded but empirically calibrated:
R_min, 蟿_3x3, 蟿_1x1, 蟿_linear, 蟿_attn_*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, Union

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
# NOTE:
# These thresholds are semantically grounded in SparseFlow's execution path
# (tile skip + group skip), but are empirically calibrated rather than strict
# first-principles hardware costs.
#
# 蟿_3x3 (TAU_3x3):
#   Minimum saved MACs per tile for 3脳3 (and larger) kernels.
#   Physical meaning: the per-tile cost of prescan + metadata + bitmask
#   dispatch.  Prescan reads ~C_in 脳 BLOCK_M values per tile and performs
#   per-group activity classification.  For typical layers (C_in=64-256,
#   BLOCK_M=16-64), this is ~200K-1M MACs equivalent in memory + compute
#   overhead.  500K is a mid-range estimate.
#   Calibrate: profile prescan latency on representative layers, convert to
#   MACs at GPU peak throughput, and set 蟿_3x3 to that value.
TAU_3x3 = DISPATCH_TAU_3X3

# 蟿_1x1 (TAU_1x1):
#   Minimum saved MACs per tile for 1脳1 kernels.
#   Higher than 蟿_3x3 because cuDNN reduces 1脳1 conv to a highly-optimized
#   GEMM call with near-peak throughput.  The sparse kernel must save
#   proportionally more to overcome this stronger baseline.
#   Calibrate: same procedure as 蟿_3x3 but comparing against cuDNN 1脳1 perf.
TAU_1x1 = DISPATCH_TAU_1X1

# Linear thresholds (execution-grounded, empirically calibrated)
TAU_LINEAR = DISPATCH_TAU_LINEAR

# Attention thresholds (split by family)
TAU_ATTN_LINEAR = DISPATCH_TAU_ATTN_LINEAR
TAU_ATTN_MATMUL = DISPATCH_TAU_ATTN_MATMUL

# R_min (R_MIN):
#   Minimum saved work fraction for sparse to be viable.
#   Physical meaning: at very low R_l (high AGR), the sparse kernel iterates
#   almost all groups with per-group bitmask checks, and the savings from
#   the few skipped groups are consumed by this overhead.  Below R_min=0.15,
#   the sparse kernel does 鈮?5% of the dense kernel's work plus bitmask
#   overhead 鈥?guaranteed slower.
#   Calibrate: find the AGR at which sparse kernel latency equals dense
#   kernel latency on a compute-heavy layer; R_min = 1 - that AGR.
R_MIN = DISPATCH_R_MIN

# Keep margins explicit for future stability tuning (default no extra margin).
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


# Backward-compatible alias.
ConvMeta = OpMeta


@dataclass
class DispatchDecision:
    backend: str = "dense"
    reason: str = ""
    reason_code: str = "init"
    confidence: float = 1.0
    meta_source: str = "measured"       # measured | estimated | fallback | shortcut
    diag_source: str = "measured"       # measured | missing | shortcut
    support_status: str = "supported"   # supported | unsupported_groups | unsupported_op | exact_zero_shortcut
    score_family: str = "unknown"       # conv | linear | attn_linear | attn_matmul | matmul | bmm | none | unknown
    tile_source: str = "diag"           # diag | meta_geometry | estimated_default | unknown
    fallback_reason: str = ""           # short fallback/debug code

    # 鈹€鈹€ Core dispatch score fields (v28) 鈹€鈹€
    R_l: float = 0.0           # saved work fraction: TZR + (1-TZR)(1-AGR_nz)
    I_l: float = 0.0           # per-tile intensity: MACs / N_tiles
    S_l: float = 0.0           # dispatch score: R_l 脳 I_l (saved MACs per tile)
    agr_nz: float = 0.0        # AGR over non-zero tiles: AGR / (1 - TZR)
    n_tiles: float = 0.0       # total tile count (measured or estimated)
    tau: float = 0.0           # threshold used for this layer's kernel family
    agr_p90: float = -1.0      # p90 AGR from calibration batches (if available)
    r_worst: float = -1.0      # conservative saved-work lower bound (1 - AGR_p90)

    # 鈹€鈹€ Backward-compatible fields 鈹€鈹€
    score_sparse: float = 0.0  # S_l / tau (normalized: >1 鈫?sparse)

    effective_benefit: float = 0.0   # alias for R_l
    total_overhead: float = 0.0      # tau / I_l (overhead as fraction of work)
    net_benefit: float = 0.0         # R_l - tau/I_l

    # 鈹€鈹€ Input measurements 鈹€鈹€
    agr: float = -1.0          # overall AGR (active_group_ratio from kernel)
    tzr: float = -1.0          # tile zero ratio
    macs: float = 0.0
    denseish_ratio: float = -1.0
    sparse_tile_ratio: float = -1.0
    active_groups: float = 0.0
    total_groups: float = 0.0

    # 鈹€鈹€ Derived features for backward compat with bench reports 鈹€鈹€
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


def op_meta_from_module(conv_module, sample_x_shape: Optional[Tuple] = None,
                        layer_name: str = "") -> OpMeta:
    """Generalized metadata entry-point.

    Backward compatible wrapper around the historical conv-oriented helper.
    """
    return conv_meta_from_module(conv_module, sample_x_shape=sample_x_shape, layer_name=layer_name)


def _target_get(target: Any, key: str, default=None):
    if isinstance(target, dict):
        return target.get(key, default)
    return getattr(target, key, default)


def _shape_tuple(input_shape: Optional[Tuple]) -> Tuple[int, ...]:
    if input_shape is None or isinstance(input_shape, str):
        return ()
    try:
        return tuple(int(v) for v in tuple(input_shape))
    except Exception:
        return ()


def _prod(values) -> int:
    vals = [int(v) for v in values]
    if not vals:
        return 0
    out = 1
    for v in vals:
        out *= v
    return int(out)


def _infer_conv_batch_from_shape(input_shape: Optional[Tuple]) -> int:
    shape = _shape_tuple(input_shape)
    if len(shape) >= 5:
        # [T, B, C, H, W] or similar: treat all leading dims before C/H/W as batch.
        return max(_prod(shape[:-3]), 0)
    if len(shape) == 4:
        # [B, C, H, W]
        return max(int(shape[0]), 0)
    if len(shape) == 3:
        # [B, C, L] for conv1d-like tensors.
        return max(int(shape[0]), 0)
    return 0


def _infer_linear_rows_from_shape(input_shape: Optional[Tuple]) -> int:
    shape = _shape_tuple(input_shape)
    if len(shape) >= 2:
        # nn.Linear applies over last dim; all leading dims are batch rows.
        return max(_prod(shape[:-1]), 0)
    if len(shape) == 1:
        return max(int(shape[0]), 0)
    return 0


def _infer_attention_batch_from_shape(input_shape: Optional[Tuple]) -> int:
    shape = _shape_tuple(input_shape)
    if len(shape) >= 3:
        # Attention uses [..., N, C], so leading dims are batch-like.
        return max(_prod(shape[:-2]), 0)
    if len(shape) == 2:
        return max(int(shape[0]), 0)
    return 0


def _infer_batch_from_shape(input_shape: Optional[Tuple]) -> int:
    # Backward-compatible alias.
    return _infer_linear_rows_from_shape(input_shape)


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

    if op_hint.startswith("conv") or op_hint.startswith("fused_conv") or op_hint == "depthwise_conv2d":
        op_type = "conv"
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
    # Conservative fallback keeps dispatch in dense path unless diagnostics are strongly favorable.
    return _conv_meta_from_target(target, None, layer_name, input_shape)


def op_meta_from_target(target: Any) -> ConvMeta:
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
    if op_type.startswith("conv") or op_type.startswith("fused_conv") or op_type == "depthwise_conv2d":
        return _conv_meta_from_target(target, None, layer_name, input_shape)
    return _fallback_meta_from_target(target, layer_name, input_shape)


# Backward-compatible name.
def conv_meta_from_target(target: Any) -> ConvMeta:
    return op_meta_from_target(target)


# Backward-compatible aliases while migrating naming to OpMeta/op_meta_*.
conv_meta_from_module_legacy = conv_meta_from_module
conv_meta_from_target_legacy = conv_meta_from_target


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


def _get_n_tiles(diag: Dict[str, Any], meta: ConvMeta) -> Tuple[float, str]:
    """Get total tile count from diagnostics, or estimate from geometry.

    Prefers the measured value from kernel diagnostics (total_tile_count).
    Falls back to a geometric estimate using output spatial size and an
    approximate BLOCK_M.
    """
    n = diag.get('total_tile_count', -1.0)
    if n > 0:
        return float(n), "diag"

    # Geometric estimate
    h_out, w_out = meta.h_out, meta.w_out
    if h_out <= 0 or w_out <= 0:
        # No spatial info 鈥?return 1 to make I_l = MACs (conservative: high
        # per-tile intensity makes sparse easier to justify, but R_min and
        # the S_l threshold still guard against bad decisions)
        return 1.0, "estimated_default"

    block_m = _estimate_block_m(h_out, w_out)
    pixels = h_out * w_out
    n_spatial = max((pixels + block_m - 1) // block_m, 1)
    batch = int(meta.batch_size) if int(meta.batch_size) > 0 else 1
    return float(batch * n_spatial), "meta_geometry"


def _select_tau(meta: ConvMeta, diag: Dict[str, Any]) -> Tuple[float, str]:
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
    groups = meta.groups
    if ks == (1, 1) and groups == 1:
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

    if groups != 1:
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
    dec.nz_fraction = nz_frac
    dec.tau = tau
    dec.score_sparse = S_l / tau if tau > 0 else 0.0

    dec.effective_benefit = R_l
    if I_l > 0:
        dec.total_overhead = tau / I_l
        dec.net_benefit = R_l - tau / I_l
    else:
        dec.total_overhead = float("inf")
        dec.net_benefit = -1.0

    dec.x_inactive = 1.0 - agr
    dec.x_tzr = tzr
    dec.x_inter = (1.0 - agr) * tzr
    dec.x_log_macs = clamp01(math.log2(macs / 1e6 + 1.0) / 6.0)
    dec.x_group_pressure = agr * clamp01(
        active_groups / 4096.0 if active_groups >= 0 else 0.0
    )
    dec.x_small = 1.0 - dec.x_log_macs
    dec.x_denseish = clamp01(denseish_ratio if denseish_ratio >= 0 else 0.0)
    dec.efficiency = R_l

    reason_suffix = ""
    if macs_estimated:
        reason_suffix += "|macs_estimated"
    if r_mismatch > 1e-4:
        reason_suffix += "|R_mismatch_check"

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

    # Confidence gate:
    # estimated/fallback metadata rows can be printed and analyzed, but they
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
      R_l = 1.0 (all work saved), S_l 鈫?鈭?for any finite I_l.
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
    return_summary: bool = False,
) -> Union[Dict[str, DispatchDecision], Tuple[Dict[str, DispatchDecision], Dict[str, int]]]:
    """Run dispatch for all target layers.

    Args:
        targets: list of target dicts from analyze_targets().
        group_sparsity_data: per-layer diagnostic dict from measure_group_sparsity().
        zero_layers: set of layer names with exact-zero input (鈫?staticzero).
        prior_decisions: accepted for interface compatibility (unused in v28;
            the simplified score model is deterministic and does not require
            hysteresis for stability).

    Returns:
        Dict mapping layer_name 鈫?DispatchDecision.
    """
    if zero_layers is None:
        zero_layers = set()

    decisions: Dict[str, DispatchDecision] = {}
    for t in targets:
        name = _target_layer_name(t)
        if not name:
            continue

        if name in zero_layers:
            decisions[name] = _make_staticzero_decision(name)
            continue

        diag = group_sparsity_data.get(name, {})
        meta = op_meta_from_target(t)
        decisions[name] = make_dispatch_decision(diag, meta)

    if return_summary:
        return decisions, summarize_decisions(decisions)
    return decisions


def summarize_decisions(decisions: Dict[str, DispatchDecision]) -> Dict[str, int]:
    summary = {
        "n_sparse": 0,
        "n_dense": 0,
        "n_staticzero": 0,
        "n_unsupported": 0,
        "n_missing_diag": 0,
        "n_estimated_meta": 0,
    }
    for dec in decisions.values():
        if dec.backend == "sparse":
            summary["n_sparse"] += 1
        elif dec.backend == "staticzero":
            summary["n_staticzero"] += 1
        else:
            summary["n_dense"] += 1

        if str(dec.support_status).startswith("unsupported"):
            summary["n_unsupported"] += 1
        if dec.diag_source == "missing":
            summary["n_missing_diag"] += 1
        if dec.meta_source != "measured":
            summary["n_estimated_meta"] += 1
    return summary


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


def decisions_to_backend_sets(decisions: Dict[str, DispatchDecision]) -> Dict[str, set]:
    out = {"staticzero": set(), "sparse": set(), "dense": set()}
    for name, dec in decisions.items():
        backend = dec.backend if dec.backend in out else "dense"
        out[backend].add(name)
    return out


# =============================================================================
# Reporting
# =============================================================================

def format_egd_report_header() -> str:
    """Column header for the EGD dispatch report."""
    return (
        f"  {'Layer':<40} {'AGR':>6} {'TZR':>6} {'AGR_nz':>7} "
        f"{'R_l':>6} {'I_l':>10} {'S_l':>10} {'蟿':>10} {'S/蟿':>6} "
        f"{'Dec':>8} {'RC':>14} {'Src':>12}"
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
    rc = dec.reason_code or "n/a"
    if len(rc) > 14:
        rc = rc[:11] + "..."
    src = f"{dec.meta_source[:1]}/{dec.diag_source[:1]}/{dec.tile_source[:1]}"
    return (
        f"  {short:<40} {agr_s:>6} {tzr_s:>6} {agrnz_s:>7} "
        f"{rl_s:>6} {il_s:>10} {sl_s:>10} {tau_s:>10} {ratio_s:>6} "
        f"{dec.backend:>8} {rc:>14} {src:>12}"
    )


def print_dispatch_decision_report(targets, group_sparsity_data, dispatch_decisions):
    """Print a detailed EGD dispatch report to stdout.

    Named to match the existing bench_4test.py call site.
    """
    print(f"\n  Execution-Grounded Dispatch Report (EGD v28)")
    print(
        f"  Constants: 蟿_3x3={TAU_3x3:,} 蟿_1x1={TAU_1x1:,} "
        f"蟿_linear={TAU_LINEAR:,} 蟿_attn_lin={TAU_ATTN_LINEAR:,} "
        f"蟿_attn_mm={TAU_ATTN_MATMUL:,} R_min={R_MIN}"
    )
    print(f"  Score: S_l = R_l 脳 I_l  (saved MACs per tile)")
    print(format_egd_report_header())
    print(f"  {'-'*150}")

    n_sparse = 0
    n_dense = 0
    n_staticzero = 0
    n_dense_score = 0
    n_dense_unsupported = 0
    n_dense_missing_diag = 0
    n_estimated_meta = 0
    for t in targets:
        name = _target_layer_name(t)
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
            if str(dec.support_status).startswith("unsupported"):
                n_dense_unsupported += 1
            elif dec.diag_source == "missing":
                n_dense_missing_diag += 1
            else:
                n_dense_score += 1
        if dec.meta_source != "measured":
            n_estimated_meta += 1

    total = n_sparse + n_dense + n_staticzero
    print(
        f"\n  Summary: sparse={n_sparse} dense={n_dense} staticzero={n_staticzero} total={total}"
    )
    print(
        f"           dense(score)={n_dense_score} dense(unsupported)={n_dense_unsupported} "
        f"dense(missing_diag)={n_dense_missing_diag} estimated_meta={n_estimated_meta}"
    )


# Alias for v26/v27 compatibility 鈥?bench code may call either name
print_cbd_dispatch_report = print_dispatch_decision_report

