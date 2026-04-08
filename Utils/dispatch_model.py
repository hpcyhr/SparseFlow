"""
Utils/dispatch_model.py

Execution-Grounded Dispatch (EGD) for SparseFlow.

This module decides per layer backend: "dense", "sparse", or "staticzero".
It is intentionally conservative and keeps backward compatibility with older
bench scripts and field names.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Calibrated dispatch constants
# ---------------------------------------------------------------------------
# These constants are execution-grounded (tile skip + group skip semantics),
# but they are still empirically calibrated thresholds.
TAU_3x3 = 500_000.0
TAU_1x1 = 1_000_000.0
TAU_LINEAR = 900_000.0
TAU_ATTN_LINEAR = 900_000.0
TAU_ATTN_MATMUL = 1_200_000.0
TAU_MATMUL = 1_100_000.0
TAU_BMM = 1_200_000.0

R_MIN = 0.15
R_MARGIN = 0.0
TAU_MARGIN = 0.0

_EPS = 1e-6


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _pair(x) -> Tuple[int, int]:
    if isinstance(x, (tuple, list)):
        if len(x) == 1:
            return int(x[0]), int(x[0])
        return int(x[0]), int(x[1])
    return int(x), int(x)


def infer_conv_output_hw(
    h_in: int,
    w_in: int,
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
    h_out: int,
    w_out: int,
    c_out: int,
    c_in: int,
    kernel_size: Tuple[int, int],
    groups: int = 1,
) -> float:
    kh, kw = kernel_size
    cin_per_group = c_in / max(groups, 1)
    return float(batch_size * h_out * w_out * c_out * kh * kw * cin_per_group)


def _estimate_block_m(h_out: int, w_out: int) -> int:
    pixels = h_out * w_out
    if pixels >= 512:
        return 64
    if pixels >= 128:
        return 32
    if pixels >= 32:
        return 16
    return max(pixels, 1)


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
        return int(shape[0])
    if len(shape) >= 1:
        return int(shape[0])
    return 0


def _infer_seq_from_shape(input_shape: Optional[Tuple]) -> int:
    if input_shape is None:
        return 0
    shape = tuple(input_shape) if not isinstance(input_shape, str) else ()
    if len(shape) >= 2:
        return int(shape[-2])
    return 0


def _is_attention_like_module(module: Any) -> bool:
    if module is None:
        return False
    for attr in ("q", "k", "v", "proj"):
        if not hasattr(module, attr):
            return False
    return True


# ---------------------------------------------------------------------------
# Metadata abstractions
# ---------------------------------------------------------------------------
@dataclass
class OpMeta:
    layer_name: str = ""
    op_type: str = "unknown"
    groups: int = 1
    kernel_size: Tuple[int, int] = (1, 1)
    stride: Tuple[int, int] = (1, 1)
    padding: Tuple[int, int] = (0, 0)
    dilation: Tuple[int, int] = (1, 1)

    c_in: int = 0
    c_out: int = 0
    h_in: int = 0
    w_in: int = 0
    h_out: int = 0
    w_out: int = 0
    batch_size: int = 0
    macs: float = 0.0

    in_features: int = 0
    out_features: int = 0
    seq_len: int = 0
    num_heads: int = 0
    head_dim: int = 0

    meta_source: str = "measured"  # measured | estimated | fallback | shortcut


# Backward compatibility alias
ConvMeta = OpMeta


@dataclass
class DispatchDecision:
    backend: str = "dense"
    reason: str = ""
    reason_code: str = ""

    # Core EGD score fields
    R_l: float = 0.0
    I_l: float = 0.0
    S_l: float = 0.0
    agr_nz: float = 0.0
    n_tiles: float = 0.0
    tau: float = 0.0
    score_sparse: float = 0.0

    # Inputs and compatibility fields
    agr: float = -1.0
    tzr: float = -1.0
    macs: float = 0.0
    denseish_ratio: float = -1.0
    sparse_tile_ratio: float = -1.0
    active_groups: float = 0.0
    total_groups: float = 0.0
    groups: int = 1
    layer_name: str = ""

    # Legacy compatibility fields used by existing exports
    effective_benefit: float = 0.0
    total_overhead: float = 0.0
    net_benefit: float = 0.0
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

    # Legacy CBD feature vector (deprecated but kept for compatibility)
    x_inactive: float = 0.0
    x_tzr: float = 0.0
    x_inter: float = 0.0
    x_log_macs: float = 0.0
    x_group_pressure: float = 0.0
    x_small: float = 0.0
    x_denseish: float = 0.0

    # Provenance / observability fields
    confidence: float = 1.0
    meta_source: str = "measured"
    diag_source: str = "measured"
    support_status: str = "supported"
    score_family: str = "unknown"
    tile_source: str = "unknown"
    fallback_reason: str = ""
    r_worst: float = -1.0
    agr_p90: float = -1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _conv_meta_from_target(
    target: Any,
    module: Any,
    layer_name: str,
    input_shape: Optional[Tuple],
    op_type: str,
) -> OpMeta:
    if module is not None:
        ks = _pair(getattr(module, "kernel_size", (3, 3)))
        st = _pair(getattr(module, "stride", (1, 1)))
        pad = _pair(getattr(module, "padding", (1, 1)))
        dil = _pair(getattr(module, "dilation", (1, 1)))
        groups = int(getattr(module, "groups", 1))
        c_in = int(getattr(module, "in_channels", 0))
        c_out = int(getattr(module, "out_channels", 0))
    else:
        ks = _pair(_target_get(target, "kernel_size", (3, 3)))
        st = _pair(_target_get(target, "stride", (1, 1)))
        pad = _pair(_target_get(target, "padding", (1, 1)))
        dil = _pair(_target_get(target, "dilation", (1, 1)))
        groups = int(_target_get(target, "groups", 1))
        c_in = int(_target_get(target, "cin", _target_get(target, "c_in", 0)))
        c_out = int(_target_get(target, "cout", _target_get(target, "c_out", 0)))

    h_in = int(_target_get(target, "input_h", 0))
    w_in = int(_target_get(target, "input_w", 0))
    batch_size = _infer_batch_from_shape(input_shape)
    if input_shape is not None and not isinstance(input_shape, str):
        s = tuple(input_shape)
        if len(s) >= 4:
            h_in = int(s[-2])
            w_in = int(s[-1])

    h_out, w_out = infer_conv_output_hw(h_in, w_in, ks, st, pad, dil)
    macs = estimate_conv_macs(batch_size, h_out, w_out, c_out, c_in, ks, groups)
    source = "measured" if (h_in > 0 and w_in > 0 and batch_size > 0) else "estimated"

    return OpMeta(
        layer_name=layer_name,
        op_type=op_type or "conv",
        groups=groups,
        kernel_size=ks,
        stride=st,
        padding=pad,
        dilation=dil,
        c_in=c_in,
        c_out=c_out,
        h_in=h_in,
        w_in=w_in,
        h_out=h_out,
        w_out=w_out,
        batch_size=batch_size,
        macs=macs,
        meta_source=source,
    )


def _linear_meta_from_target(
    target: Any,
    module: Any,
    layer_name: str,
    input_shape: Optional[Tuple],
    op_type: str,
) -> OpMeta:
    if module is not None:
        in_f = int(getattr(module, "in_features", 0))
        out_f = int(getattr(module, "out_features", 0))
    else:
        in_f = int(_target_get(target, "in_features", _target_get(target, "cin", _target_get(target, "c_in", 0))))
        out_f = int(_target_get(target, "out_features", _target_get(target, "cout", _target_get(target, "c_out", 0))))

    batch_size = _infer_batch_from_shape(input_shape)
    seq_len = _infer_seq_from_shape(input_shape)
    rows = max(batch_size, 0)
    if rows <= 0:
        rows = max(int(_target_get(target, "input_h", 0)), 0)
    if rows <= 0 and seq_len > 0:
        rows = seq_len
    macs = float(rows * in_f * out_f) if rows > 0 else 0.0
    source = "measured" if rows > 0 else "estimated"

    return OpMeta(
        layer_name=layer_name,
        op_type=op_type or "linear",
        groups=1,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        c_in=in_f,
        c_out=out_f,
        h_in=1,
        w_in=1,
        h_out=1,
        w_out=1,
        batch_size=max(rows, 0),
        macs=macs,
        in_features=in_f,
        out_features=out_f,
        seq_len=seq_len,
        meta_source=source,
    )


def _attention_meta_from_target(
    target: Any,
    module: Any,
    layer_name: str,
    input_shape: Optional[Tuple],
    op_type: str,
) -> OpMeta:
    seq_len = max(_infer_seq_from_shape(input_shape), int(_target_get(target, "input_h", 0)), 1)
    batch_size = max(_infer_batch_from_shape(input_shape), 1)
    dim = int(_target_get(target, "input_w", 0))
    num_heads = int(_target_get(target, "num_heads", 1))
    head_dim = int(_target_get(target, "head_dim", 0))

    if module is not None:
        if hasattr(module, "dim"):
            dim = int(getattr(module, "dim"))
        elif hasattr(module, "q") and hasattr(module.q, "in_features"):
            dim = int(module.q.in_features)
        if hasattr(module, "num_heads"):
            num_heads = int(getattr(module, "num_heads"))
        if hasattr(module, "head_dim"):
            head_dim = int(getattr(module, "head_dim"))

    dim = max(dim, 1)
    num_heads = max(num_heads, 1)
    if head_dim <= 0:
        head_dim = max(dim // num_heads, 1)

    if op_type in ("attention_qkav", "attention_qkmix", "attention_matmul"):
        macs = float(batch_size * num_heads * 2 * seq_len * seq_len * head_dim)
    else:
        macs = float(batch_size * num_heads * 2 * seq_len * head_dim * head_dim)

    return OpMeta(
        layer_name=layer_name,
        op_type=op_type,
        groups=1,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        c_in=dim,
        c_out=dim,
        h_in=1,
        w_in=1,
        h_out=1,
        w_out=1,
        batch_size=batch_size,
        macs=macs,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        meta_source="estimated",
    )


def _matmul_like_meta_from_target(
    target: Any,
    layer_name: str,
    input_shape: Optional[Tuple],
    op_type: str,
) -> OpMeta:
    m = int(_target_get(target, "m", _target_get(target, "input_h", 0)))
    k = int(_target_get(target, "k", _target_get(target, "input_w", 0)))
    n = int(_target_get(target, "n", _target_get(target, "out_features", 0)))
    batch = int(_target_get(target, "batch", 1))
    if input_shape is not None and not isinstance(input_shape, str):
        s = tuple(input_shape)
        if len(s) >= 2:
            m = max(m, int(s[-2]))
            k = max(k, int(s[-1]))
        if len(s) >= 3:
            batch = max(batch, int(s[0]))
    if n <= 0:
        n = k
    m = max(m, 1)
    k = max(k, 1)
    n = max(n, 1)
    batch = max(batch, 1)
    macs = float(batch * m * k * n) if op_type == "bmm" else float(m * k * n)

    return OpMeta(
        layer_name=layer_name,
        op_type=op_type,
        groups=1,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        c_in=k,
        c_out=n,
        h_in=1,
        w_in=1,
        h_out=1,
        w_out=1,
        batch_size=batch,
        macs=macs,
        in_features=k,
        out_features=n,
        meta_source="estimated",
    )


def _fallback_meta_from_target(target: Any, layer_name: str, op_type: str) -> OpMeta:
    return OpMeta(layer_name=layer_name, op_type=op_type or "unknown", meta_source="fallback")


def op_meta_from_target(target: Any) -> OpMeta:
    module = _target_get(target, "module")
    if module is None:
        module = _target_get(target, "conv_module")

    layer_name = _target_get(target, "name", "") or _target_get(target, "conv_name", "")
    op_type = str(_target_get(target, "op_type", ""))
    input_shape = _target_get(target, "input_shape")

    if op_type in ("attention_qkav", "attention_linear", "attention_qkmix", "attention_matmul", "attention_proj_linear"):
        return _attention_meta_from_target(target, module, layer_name, input_shape, op_type)
    if op_type in ("matmul", "bmm"):
        return _matmul_like_meta_from_target(target, layer_name, input_shape, op_type)

    if module is not None:
        if hasattr(module, "in_features") and hasattr(module, "out_features"):
            return _linear_meta_from_target(target, module, layer_name, input_shape, op_type or "linear")
        if hasattr(module, "kernel_size") and hasattr(module, "in_channels"):
            return _conv_meta_from_target(target, module, layer_name, input_shape, op_type or "conv")
        if _is_attention_like_module(module):
            return _attention_meta_from_target(target, module, layer_name, input_shape, op_type or "attention_matmul")

    if op_type == "linear":
        return _linear_meta_from_target(target, module, layer_name, input_shape, op_type)
    if op_type.startswith("conv") or "fused_conv" in op_type or op_type == "depthwise_conv2d":
        return _conv_meta_from_target(target, module, layer_name, input_shape, op_type)
    if op_type:
        return _fallback_meta_from_target(target, layer_name, op_type)
    return _fallback_meta_from_target(target, layer_name, "unknown")


# Backward compatibility aliases
def conv_meta_from_target(target: Any) -> OpMeta:
    return op_meta_from_target(target)


def conv_meta_from_module(conv_module, sample_x_shape: Optional[Tuple] = None, layer_name: str = "") -> OpMeta:
    tmp = {"module": conv_module, "name": layer_name, "input_shape": sample_x_shape}
    return op_meta_from_target(tmp)


# ---------------------------------------------------------------------------
# Diagnostic mapping helpers
# ---------------------------------------------------------------------------
def _map_diag_to_agr_tzr(diag: Dict[str, Any]) -> Tuple[float, float, float, float]:
    agr = diag.get("active_group_ratio", -1.0)
    tzr = diag.get("tile_zero_ratio", -1.0)
    active_groups = diag.get("nonzero_group_count", -1.0)
    total_groups = diag.get("total_group_count", -1.0)

    if agr < 0 and total_groups > 0 and active_groups >= 0:
        agr = active_groups / total_groups

    if tzr < 0:
        zero_cnt = diag.get("tile_zero_count", -1.0)
        total_tiles = diag.get("total_tile_count", -1.0)
        if total_tiles > 0 and zero_cnt >= 0:
            tzr = zero_cnt / total_tiles

    return float(agr), float(tzr), float(active_groups), float(total_groups)


def _map_diag_tile_mix(diag: Dict[str, Any]) -> Tuple[float, float]:
    total_tiles = diag.get("total_tile_count", -1.0)
    denseish_tiles = diag.get("denseish_tiles", -1.0)
    sparse_tiles = diag.get("sparse_tiles", -1.0)
    denseish_ratio = -1.0
    sparse_ratio = -1.0
    if total_tiles and total_tiles > 0:
        if denseish_tiles >= 0:
            denseish_ratio = denseish_tiles / total_tiles
        if sparse_tiles >= 0:
            sparse_ratio = sparse_tiles / total_tiles
    return float(denseish_ratio), float(sparse_ratio)


def _get_n_tiles(diag: Dict[str, Any], meta: OpMeta) -> Tuple[float, str]:
    measured = float(diag.get("total_tile_count", -1.0))
    if measured > 0:
        return measured, "diag"

    if meta.h_out > 0 and meta.w_out > 0 and meta.batch_size > 0:
        block_m = _estimate_block_m(meta.h_out, meta.w_out)
        pixels = meta.h_out * meta.w_out
        n_spatial = max((pixels + block_m - 1) // block_m, 1)
        return float(meta.batch_size * n_spatial), "meta_geometry"

    return 1.0, "estimated_default"


def _select_tau(meta: OpMeta) -> Tuple[float, str]:
    op_type = (meta.op_type or "").lower()
    if op_type in ("attention_linear", "attention_proj_linear"):
        return TAU_ATTN_LINEAR, "attn_linear"
    if op_type in ("attention_qkav", "attention_qkmix", "attention_matmul"):
        return TAU_ATTN_MATMUL, "attn_matmul"
    if op_type == "linear":
        return TAU_LINEAR, "linear"
    if op_type == "matmul":
        return TAU_MATMUL, "matmul"
    if op_type == "bmm":
        return TAU_BMM, "bmm"
    if meta.kernel_size == (1, 1) and meta.groups == 1:
        return TAU_1x1, "conv"
    return TAU_3x3, "conv"


def _estimate_missing_macs(meta: OpMeta, n_tiles: float, diag: Dict[str, Any]) -> Tuple[float, str]:
    if meta.macs > 0:
        return float(meta.macs), meta.meta_source

    op_type = (meta.op_type or "").lower()
    if op_type == "linear" and meta.in_features > 0 and meta.out_features > 0:
        block_m = float(diag.get("block_m", 32))
        rows = max(n_tiles * max(block_m, 1.0), 1.0)
        return float(rows * meta.in_features * meta.out_features), "estimated"
    if op_type in ("matmul", "bmm") and meta.in_features > 0 and meta.out_features > 0:
        rows = max(meta.batch_size, 1)
        return float(rows * meta.in_features * meta.out_features), "estimated"
    if op_type.startswith("attention_"):
        seq = max(meta.seq_len, 1)
        h = max(meta.num_heads, 1)
        d = max(meta.head_dim, 1)
        if op_type in ("attention_qkav", "attention_qkmix", "attention_matmul"):
            return float(max(meta.batch_size, 1) * h * 2 * seq * seq * d), "estimated"
        return float(max(meta.batch_size, 1) * h * 2 * seq * d * d), "estimated"
    if meta.c_in > 0 and meta.c_out > 0 and meta.h_out > 0 and meta.w_out > 0 and meta.batch_size > 0:
        return estimate_conv_macs(meta.batch_size, meta.h_out, meta.w_out, meta.c_out, meta.c_in, meta.kernel_size, meta.groups), "estimated"
    return 0.0, "fallback"


# ---------------------------------------------------------------------------
# Core dispatch logic
# ---------------------------------------------------------------------------
def make_dispatch_decision(diag: Dict[str, Any], meta: OpMeta) -> DispatchDecision:
    agr, tzr, active_groups, total_groups = _map_diag_to_agr_tzr(diag)
    denseish_ratio, sparse_tile_ratio = _map_diag_tile_mix(diag)

    dec = DispatchDecision(
        layer_name=meta.layer_name,
        groups=meta.groups,
        agr=agr,
        tzr=tzr,
        macs=float(meta.macs),
        active_groups=active_groups,
        total_groups=total_groups,
        denseish_ratio=denseish_ratio,
        sparse_tile_ratio=sparse_tile_ratio,
        meta_source=meta.meta_source,
        diag_source="measured" if (agr >= 0 and tzr >= 0) else "missing",
        support_status="supported",
        score_family="unknown",
        tile_source="unknown",
        fallback_reason="",
    )

    # Structural support guard.
    if meta.groups != 1:
        dec.backend = "dense"
        dec.reason_code = "unsupported_groups"
        dec.reason = "unsupported_groups!=1"
        dec.support_status = "unsupported_groups"
        dec.fallback_reason = dec.reason_code
        dec.diag_source = "missing" if dec.diag_source != "measured" else dec.diag_source
        return dec

    if agr < 0 or tzr < 0:
        dec.backend = "dense"
        dec.reason_code = "missing_diag"
        dec.reason = "missing_diag_fallback_dense"
        dec.diag_source = "missing"
        dec.fallback_reason = dec.reason_code
        return dec

    nz_frac = max(1.0 - tzr, _EPS)
    agr_nz = min(agr / nz_frac, 1.0)
    r_exec = tzr + nz_frac * (1.0 - agr_nz)
    r_simple = 1.0 - agr
    if abs(r_exec - r_simple) > 1e-4:
        # Keep result stable but leave a traceable reason suffix.
        dec.fallback_reason = "R_mismatch_check"
    R_l = r_exec

    n_tiles, tile_source = _get_n_tiles(diag, meta)
    macs, macs_source = _estimate_missing_macs(meta, n_tiles, diag)
    tau, score_family = _select_tau(meta)

    I_l = macs / max(n_tiles, 1.0)
    S_l = R_l * I_l
    score_sparse = (S_l / tau) if tau > 0 else 0.0

    dec.R_l = float(R_l)
    dec.I_l = float(I_l)
    dec.S_l = float(S_l)
    dec.agr_nz = float(agr_nz)
    dec.n_tiles = float(n_tiles)
    dec.nz_fraction = float(nz_frac)
    dec.tau = float(tau)
    dec.score_sparse = float(score_sparse)
    dec.score_family = score_family
    dec.tile_source = tile_source
    dec.macs = float(macs)
    if score_family == "unknown":
        dec.support_status = "unsupported_op"
    if macs_source != meta.meta_source:
        dec.meta_source = macs_source

    dec.effective_benefit = dec.R_l
    if dec.I_l > 0:
        dec.total_overhead = dec.tau / dec.I_l
        dec.net_benefit = dec.R_l - dec.total_overhead
    else:
        dec.total_overhead = float("inf")
        dec.net_benefit = -1.0

    # Deprecated compatibility fields (kept for old report consumers).
    dec.x_inactive = 1.0 - agr
    dec.x_tzr = tzr
    dec.x_inter = (1.0 - agr) * tzr
    dec.x_log_macs = clamp01(math.log2(max(macs, 0.0) / 1e6 + 1.0) / 6.0)
    dec.x_group_pressure = agr * clamp01(active_groups / 4096.0 if active_groups >= 0 else 0.0)
    dec.x_small = 1.0 - dec.x_log_macs
    dec.x_denseish = clamp01(denseish_ratio if denseish_ratio >= 0 else 0.0)
    dec.efficiency = dec.R_l

    agr_p90 = float(diag.get("active_group_ratio_p90", -1.0))
    dec.agr_p90 = agr_p90
    dec.r_worst = (1.0 - agr_p90) if agr_p90 >= 0 else dec.R_l

    confidence = 1.0
    if dec.meta_source in ("estimated", "fallback"):
        confidence -= 0.25
    if dec.diag_source != "measured":
        confidence -= 0.25
    if dec.tile_source in ("meta_geometry", "estimated_default"):
        confidence -= 0.15
    if dec.score_family == "unknown":
        confidence -= 0.15
    dec.confidence = clamp01(confidence)

    if dec.support_status == "unsupported_op":
        dec.backend = "dense"
        dec.reason_code = "unsupported_op"
        dec.reason = "unsupported_op_fallback_dense"
        dec.fallback_reason = dec.reason_code
        return dec

    if dec.confidence < 0.6:
        dec.backend = "dense"
        dec.reason_code = "low_confidence"
        dec.reason = "low_confidence_fallback_dense"
        dec.fallback_reason = dec.reason_code
        return dec

    if dec.r_worst >= 0 and dec.r_worst < (R_MIN + R_MARGIN):
        dec.backend = "dense"
        dec.reason_code = "R_guard_worst"
        dec.reason = f"R_worst={dec.r_worst:.3f}<R_min={R_MIN + R_MARGIN:.3f}"
        dec.fallback_reason = dec.reason_code
        return dec

    if dec.R_l < (R_MIN + R_MARGIN):
        dec.backend = "dense"
        dec.reason_code = "R_guard"
        dec.reason = f"R={dec.R_l:.3f}<R_min={R_MIN + R_MARGIN:.3f}"
        dec.fallback_reason = dec.reason_code
        return dec

    tau_guard = dec.tau * (1.0 + TAU_MARGIN)
    if dec.S_l > tau_guard:
        dec.backend = "sparse"
        dec.reason_code = "score_pass"
        dec.reason = f"S={dec.S_l:.0f}>tau={tau_guard:.0f}"
        return dec

    dec.backend = "dense"
    dec.reason_code = "score_fail"
    dec.reason = f"S={dec.S_l:.0f}<=tau={tau_guard:.0f}"
    dec.fallback_reason = dec.reason_code
    return dec


def _make_staticzero_decision(layer_name: str) -> DispatchDecision:
    return DispatchDecision(
        layer_name=layer_name,
        backend="staticzero",
        reason="exact_zero_input_shortcut",
        reason_code="exact_zero",
        fallback_reason="exact_zero",
        agr=0.0,
        tzr=1.0,
        R_l=1.0,
        agr_nz=0.0,
        effective_benefit=1.0,
        net_benefit=1.0,
        score_sparse=float("inf"),
        denseish_ratio=0.0,
        sparse_tile_ratio=0.0,
        confidence=1.0,
        meta_source="shortcut",
        diag_source="shortcut",
        support_status="exact_zero_shortcut",
        score_family="none",
        tile_source="shortcut",
    )


# ---------------------------------------------------------------------------
# Batch dispatch and summaries
# ---------------------------------------------------------------------------
def dispatch_all_layers(
    targets,
    group_sparsity_data: Dict[str, Dict],
    zero_layers=None,
    prior_decisions: Optional[Dict[str, DispatchDecision]] = None,  # kept for compat
) -> Dict[str, DispatchDecision]:
    if zero_layers is None:
        zero_layers = set()

    decisions: Dict[str, DispatchDecision] = {}
    for t in targets:
        name = _target_get(t, "name", "") or _target_get(t, "conv_name", "")
        if not name:
            continue
        if name in zero_layers:
            decisions[name] = _make_staticzero_decision(name)
            continue
        diag = group_sparsity_data.get(name, {})
        meta = op_meta_from_target(t)
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


def decisions_to_backend_sets(decisions: Dict[str, DispatchDecision]) -> Dict[str, set]:
    out = {"staticzero": set(), "sparse": set(), "dense": set()}
    for name, dec in decisions.items():
        if dec.backend == "staticzero":
            out["staticzero"].add(name)
        elif dec.backend == "sparse":
            out["sparse"].add(name)
        else:
            out["dense"].add(name)
    return out


def summarize_decisions(decisions: Dict[str, DispatchDecision]) -> Dict[str, int]:
    summary = {
        "n_sparse": 0,
        "n_staticzero": 0,
        "n_dense": 0,
        "n_dense_unsupported": 0,
        "n_dense_missing_diag": 0,
        "n_estimated_meta": 0,
    }
    for dec in decisions.values():
        if dec.meta_source in ("estimated", "fallback"):
            summary["n_estimated_meta"] += 1
        if dec.backend == "sparse":
            summary["n_sparse"] += 1
            continue
        if dec.backend == "staticzero":
            summary["n_staticzero"] += 1
            continue
        summary["n_dense"] += 1
        if dec.support_status != "supported":
            summary["n_dense_unsupported"] += 1
        if dec.diag_source != "measured":
            summary["n_dense_missing_diag"] += 1
    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def format_egd_report_header() -> str:
    return (
        f"  {'Layer':<40} {'AGR':>6} {'TZR':>6} {'AGR_nz':>7} "
        f"{'R_l':>6} {'I_l':>10} {'S_l':>10} {'tau':>10} {'S/t':>6} "
        f"{'Dec':>8} {'Code':>12}"
    )


def format_egd_report_row(dec: DispatchDecision) -> str:
    short = dec.layer_name if len(dec.layer_name) <= 39 else "..." + dec.layer_name[-36:]
    agr_s = f"{dec.agr:.3f}" if dec.agr >= 0 else "n/a"
    tzr_s = f"{dec.tzr:.3f}" if dec.tzr >= 0 else "n/a"
    agrnz_s = f"{dec.agr_nz:.3f}" if dec.agr_nz >= 0 else "n/a"
    rl_s = f"{dec.R_l:.3f}"
    il_s = f"{dec.I_l:.0f}" if dec.I_l > 0 else "n/a"
    sl_s = f"{dec.S_l:.0f}" if dec.S_l > 0 else "0"
    tau_s = f"{dec.tau:.0f}" if dec.tau > 0 else "n/a"
    ratio_s = f"{dec.score_sparse:.2f}" if dec.score_sparse < 1e6 else "inf"
    code = (dec.reason_code or "n/a")
    if len(code) > 12:
        code = code[:12]
    return (
        f"  {short:<40} {agr_s:>6} {tzr_s:>6} {agrnz_s:>7} "
        f"{rl_s:>6} {il_s:>10} {sl_s:>10} {tau_s:>10} {ratio_s:>6} "
        f"{dec.backend:>8} {code:>12}"
    )


def print_dispatch_decision_report(targets, group_sparsity_data, dispatch_decisions):
    print("\n  Execution-Grounded Dispatch Report (EGD)")
    print(
        f"  Constants: tau_3x3={int(TAU_3x3):,} tau_1x1={int(TAU_1x1):,} "
        f"tau_linear={int(TAU_LINEAR):,} R_min={R_MIN}"
    )
    print(format_egd_report_header())
    print(f"  {'-' * 140}")

    for t in targets:
        name = _target_get(t, "name", "") or _target_get(t, "conv_name", "")
        if not name:
            continue
        dec = dispatch_decisions.get(name)
        if dec is None:
            continue
        print(format_egd_report_row(dec))

    summary = summarize_decisions(dispatch_decisions)
    print(
        "\n  Summary: "
        f"sparse={summary['n_sparse']} "
        f"staticzero={summary['n_staticzero']} "
        f"dense={summary['n_dense']} "
        f"dense(unsupported)={summary['n_dense_unsupported']} "
        f"dense(missing_diag)={summary['n_dense_missing_diag']} "
        f"estimated_meta={summary['n_estimated_meta']}"
    )


# Deprecated compatibility alias for older call sites.
print_cbd_dispatch_report = print_dispatch_decision_report
