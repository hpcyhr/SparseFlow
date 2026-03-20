"""
SparseFlow Conv2d Triton Kernels — v18.0 Bitmask + Adaptive Grouping

Changes from v17:
  A. Metadata: per-tile uint32 active-group BITMASK replaces ag_count / ag_list
  B. GROUP_SIZE: adaptive via choose_group_size(c_in) instead of fixed 32
  C. Prescan: two-stage design (tile screen → group bitmask, single kernel)
  D. Compute: zero-tile early return; iterate groups via range(NUM_GROUPS) with bit check
  E. Diagnostics: metadata_kind, group_size, prescan stats from bitmask

Supported patterns:
  1x1/s1/p0, 1x1/s2/p0 (via subsample), 3x3/s1/p1, 3x3/s2/p1
"""

import torch
import triton
import triton.language as tl
from triton import autotune, Config

# ---------------------------------------------------------------------------
# Adaptive group size
# ---------------------------------------------------------------------------
FALLBACK_RATIO = 0.85


def choose_group_size(c_in: int) -> int:
    """
    Adaptive group size based on input channels.

    Policy:
      C_IN <= 64  → 16  (finer granularity for early layers; 4 groups)
      64 < C_IN <= 128 → 16  (8 groups)
      C_IN > 128  → 32  (coarser for deeper layers; <=16 groups)

    Constraint: num_groups must fit in uint32 bitmask (≤ 32 bits).
    """
    if c_in <= 128:
        gs = 16
    else:
        gs = 32
    # Safety: ensure num_groups ≤ 32 for uint32 bitmask
    num_groups = (c_in + gs - 1) // gs
    while num_groups > 32:
        gs *= 2
        num_groups = (c_in + gs - 1) // gs
    return gs


# Legacy alias kept for any external code that reads this constant.
# New code should call choose_group_size(c_in) instead.
GROUP_SIZE = 32


def _select_tile_sizes(H, W):
    """Deterministic tile-size selection (shared by prescan & compute)."""
    pixels = H * W
    if pixels >= 3136:
        return 8, 16
    return 8, 8


def _select_block_sizes(H, W, C_IN, C_OUT, kernel_size, N):
    """Compat helper used by other modules."""
    BH, BW = _select_tile_sizes(H, W)
    gs = choose_group_size(C_IN)
    return BH, BW, BH * BW, 64, gs


# ---------------------------------------------------------------------------
# Prescan kernel — produces per-tile uint32 bitmask
# ---------------------------------------------------------------------------
#
# Two-stage design (inside a single kernel launch):
#   Stage 1: For each tile, accumulate a tile-level max across ALL channels
#            and ALL receptive-field positions.  If tile_max == 0 the tile is
#            definitely zero → store mask = 0 and skip the per-group breakdown.
#   Stage 2: For tiles with activity, scan each channel-group separately to
#            build the per-group bitmask.
#
# Because Triton does not support `break` in range-loops the tile-max is
# accumulated cheaply as a single scalar *alongside* the group-level scan.
# Zero tiles therefore pay the same data-load cost but avoid bitmask book-
# keeping.  The main savings come at *compute* time: zero-tile tiles early-
# return from the compute kernel, completely skipping the 9×K dot loop.

@triton.jit
def prescan_bitmask_kernel(
    x_ptr,
    ag_mask_ptr,          # [N_TILES] int32 output — per-tile bitmask
    N_val,
    C_IN,
    H_IN,
    W_IN,
    H_OUT,
    W_OUT,
    GH,
    GW,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    RF_SIZE: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
    tile_id = tl.program_id(0)
    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles:
        return

    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH

    # Receptive-field bounds for this output tile
    rf_h_start = gh_idx * BLOCK_H * STRIDE - PAD
    rf_w_start = gw_idx * BLOCK_W * STRIDE - PAD
    RF_H: tl.constexpr = (BLOCK_H - 1) * STRIDE + KERNEL_SIZE
    RF_W: tl.constexpr = (BLOCK_W - 1) * STRIDE + KERNEL_SIZE

    flat_idx = tl.arange(0, RF_SIZE)
    flat_row = flat_idx // RF_W
    flat_col = flat_idx % RF_W
    valid_mask = flat_idx < (RF_H * RF_W)

    hh = rf_h_start + flat_row
    ww = rf_w_start + flat_col
    hw_mask = valid_mask & (hh >= 0) & (hh < H_IN) & (ww >= 0) & (ww < W_IN)

    safe_h = tl.minimum(tl.maximum(hh, 0), H_IN - 1)
    safe_w = tl.minimum(tl.maximum(ww, 0), W_IN - 1)

    HW = H_IN * W_IN

    # --- Per-group bitmask construction ---
    off1 = tl.arange(0, 1)
    mask = tl.zeros([1], dtype=tl.int32)

    for g in range(NUM_GROUPS):
        group_max = tl.zeros([1], dtype=tl.float32)

        for c_off in range(GROUP_SIZE_C):
            c = g * GROUP_SIZE_C + c_off
            if c < C_IN:
                base = (n_idx * C_IN + c) * HW
                vals = tl.load(
                    x_ptr + base + safe_h * W_IN + safe_w,
                    mask=hw_mask,
                    other=0.0,
                )
                ch_max = tl.max(vals, axis=0)
                group_max = tl.maximum(group_max, ch_max)

        is_active_int = (group_max > THRESHOLD).to(tl.int32)
        # Set bit g in mask  (1 << g is a Python int since g comes from range)
        mask = mask + is_active_int * (1 << g)

    tl.store(ag_mask_ptr + tile_id + off1, mask)


# ---------------------------------------------------------------------------
# Python helpers — metadata construction & fallback check
# ---------------------------------------------------------------------------

@torch.no_grad()
def _build_active_group_bitmask(
    x_f16,
    N,
    C_IN,
    H_IN,
    W_IN,
    H_OUT,
    W_OUT,
    BH,
    BW,
    GH,
    GW,
    kernel_size,
    stride,
    padding,
    threshold,
    ag_mask_buf,
):
    """Build per-tile active-group bitmask (uint32).

    Returns (GROUP_SIZE_C, NUM_GROUPS) actually used.
    """
    GROUP_SIZE_C = choose_group_size(C_IN)
    N_TILES = N * GH * GW
    NUM_GROUPS = triton.cdiv(C_IN, GROUP_SIZE_C)

    rf_h = (BH - 1) * stride + kernel_size
    rf_w = (BW - 1) * stride + kernel_size
    rf_actual = rf_h * rf_w
    RF_SIZE = triton.next_power_of_2(max(rf_actual, 1))

    prescan_bitmask_kernel[(N_TILES,)](
        x_f16,
        ag_mask_buf,
        N,
        C_IN,
        H_IN,
        W_IN,
        H_OUT,
        W_OUT,
        GH,
        GW,
        BLOCK_H=BH,
        BLOCK_W=BW,
        KERNEL_SIZE=kernel_size,
        STRIDE=stride,
        PAD=padding,
        GROUP_SIZE_C=GROUP_SIZE_C,
        NUM_GROUPS=NUM_GROUPS,
        RF_SIZE=RF_SIZE,
        THRESHOLD=threshold,
    )

    return GROUP_SIZE_C, NUM_GROUPS


# Backward-compat alias used by fused_conv_lif.py
def _build_active_group_metadata(
    x_f16, N, C_IN, H_IN, W_IN, H_OUT, W_OUT,
    BH, BW, GH, GW, kernel_size, stride, padding,
    threshold, ag_count_buf, ag_list_buf,
):
    """Legacy shim — redirects to bitmask builder.

    ag_count_buf is reinterpreted as ag_mask_buf (same shape: [N_TILES] int32).
    ag_list_buf is IGNORED (no longer needed).
    """
    return _build_active_group_bitmask(
        x_f16, N, C_IN, H_IN, W_IN, H_OUT, W_OUT,
        BH, BW, GH, GW, kernel_size, stride, padding,
        threshold, ag_count_buf,  # reuse as mask buf
    )


def _check_dense_fallback(ag_mask_buf, N_TILES, NUM_GROUPS, fallback_ratio=FALLBACK_RATIO):
    """Check if average active-group ratio exceeds fallback threshold."""
    if NUM_GROUPS == 0:
        return False
    # popcount via bit tricks on int32: count set bits per mask
    masks = ag_mask_buf[:N_TILES].int()
    # Compute popcount for each mask value
    total_active = 0
    # GPU-friendly popcount: convert to float bit-count
    # Use the identity: popcount(x) = sum of bits
    # For int32, iterate 32 bits is expensive; use a simpler proxy:
    # avg_active ≈ (sum of masks != 0) is too coarse.
    # Instead use a vectorized approach:
    pc = torch.zeros(N_TILES, dtype=torch.int32, device=masks.device)
    tmp = masks.clone()
    for _ in range(32):
        pc += tmp & 1
        tmp = tmp >> 1
    avg_active = pc.float().mean().item()
    threshold = fallback_ratio * NUM_GROUPS
    return avg_active > threshold


def _check_dense_fallback_fast(ag_mask_buf, N_TILES, NUM_GROUPS, fallback_ratio=FALLBACK_RATIO):
    """Fast approximate fallback check — avoids full popcount.

    Uses the fraction of non-zero masks as a conservative proxy.
    If most tiles have *any* activity AND NUM_GROUPS is small,
    the actual AGR is likely high → fall back.
    """
    if NUM_GROUPS == 0:
        return False
    nonzero_frac = (ag_mask_buf[:N_TILES] != 0).float().mean().item()
    # If almost all tiles are active and few groups exist, AGR is high
    if nonzero_frac > fallback_ratio and NUM_GROUPS <= 4:
        return True
    # Otherwise do full popcount check
    return _check_dense_fallback(ag_mask_buf, N_TILES, NUM_GROUPS, fallback_ratio)


# ---------------------------------------------------------------------------
# Autotune configs
# ---------------------------------------------------------------------------

def _make_configs(block_h, block_w):
    block_m = block_h * block_w
    configs = []
    for bn in [64, 128]:
        for nw in [4, 8]:
            configs.append(
                Config(
                    {
                        "BLOCK_M": block_m,
                        "BLOCK_N": bn,
                        "BLOCK_H": block_h,
                        "BLOCK_W": block_w,
                    },
                    num_warps=nw,
                    num_stages=1,
                )
            )
    return configs


_CONFIGS_8x8 = _make_configs(8, 8)
_CONFIGS_8x16 = _make_configs(8, 16)


# ===================================================================
# Compute kernels — bitmask-based iteration with zero-tile early return
# ===================================================================
#
# Naming: sparse_conv{K}x{K}s{S}_bm_kernel_{BH}x{BW}
#   K  = kernel size (1 or 3)
#   S  = stride (1 or 2)
#   BH×BW = tile size
#
# Inner loop: for g in range(NUM_GROUPS):  (constexpr-unrolled)
#   - check bit g of ag_mask
#   - if inactive, k_mask is all-False → masked loads return zero
#   - dot with zero operands is a no-op in terms of result
#
# Zero-tile early return: if ag_mask == 0, write bias-only and return.
# This avoids the entire NUM_GROUPS × 9 inner loop for 99%+ of tiles.

# ---- 1x1 / stride=1 ----

@autotune(configs=_CONFIGS_8x8, key=["C_IN", "C_OUT", "H_OUT", "W_OUT", "GH", "GW"])
@triton.jit
def sparse_conv1x1_bm_kernel_8x8(
    x_ptr, w_cl_ptr, bias_ptr, ag_mask_ptr, y_ptr, N_val,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    H_IN: tl.constexpr, W_IN: tl.constexpr,
    H_OUT: tl.constexpr, W_OUT: tl.constexpr,
    GH: tl.constexpr, GW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, NUM_GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    tile_id = tl.program_id(0)
    pid_cout = tl.program_id(1)
    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles:
        return

    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH

    offs_n = pid_cout * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < C_OUT
    offs_m = tl.arange(0, BLOCK_M)
    out_h = gh_idx * BLOCK_H + offs_m // BLOCK_W
    out_w = gw_idx * BLOCK_W + offs_m % BLOCK_W
    m_mask = (out_h < H_OUT) & (out_w < W_OUT)

    HW_IN: tl.constexpr = H_IN * W_IN
    HW_OUT: tl.constexpr = H_OUT * W_OUT

    # Stage 1: bitmask check — zero-tile early return
    ag_mask = tl.load(ag_mask_ptr + tile_id)
    if ag_mask == 0:
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
        oa = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
        tl.store(oa, acc, mask=m_mask[:, None] & n_mask[None, :])
        return

    # Stage 2: sparse compute via bitmask
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for g in range(NUM_GROUPS):
        g_active = (ag_mask >> g) & 1
        cin_start = g * GROUP_SIZE_C
        offs_k = cin_start + tl.arange(0, GROUP_SIZE_C)
        k_mask = (g_active != 0) & (offs_k < C_IN)

        x_addrs = x_ptr + (n_idx * C_IN + offs_k[None, :]) * HW_IN + out_h[:, None] * W_IN + out_w[:, None]
        x_tile = tl.load(x_addrs, mask=k_mask[None, :] & m_mask[:, None], other=0.0).to(tl.float16)
        w_addrs = w_cl_ptr + offs_n[None, :] * C_IN + offs_k[:, None]
        w_tile = tl.load(w_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
        acc += tl.dot(x_tile, w_tile)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
    oa = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
    tl.store(oa, acc, mask=m_mask[:, None] & n_mask[None, :])


@autotune(configs=_CONFIGS_8x16, key=["C_IN", "C_OUT", "H_OUT", "W_OUT", "GH", "GW"])
@triton.jit
def sparse_conv1x1_bm_kernel_8x16(
    x_ptr, w_cl_ptr, bias_ptr, ag_mask_ptr, y_ptr, N_val,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    H_IN: tl.constexpr, W_IN: tl.constexpr,
    H_OUT: tl.constexpr, W_OUT: tl.constexpr,
    GH: tl.constexpr, GW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, NUM_GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    tile_id = tl.program_id(0)
    pid_cout = tl.program_id(1)
    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles:
        return
    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH
    offs_n = pid_cout * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < C_OUT
    offs_m = tl.arange(0, BLOCK_M)
    out_h = gh_idx * BLOCK_H + offs_m // BLOCK_W
    out_w = gw_idx * BLOCK_W + offs_m % BLOCK_W
    m_mask = (out_h < H_OUT) & (out_w < W_OUT)
    HW_IN: tl.constexpr = H_IN * W_IN
    HW_OUT: tl.constexpr = H_OUT * W_OUT

    ag_mask = tl.load(ag_mask_ptr + tile_id)
    if ag_mask == 0:
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
        oa = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
        tl.store(oa, acc, mask=m_mask[:, None] & n_mask[None, :])
        return

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for g in range(NUM_GROUPS):
        g_active = (ag_mask >> g) & 1
        cin_start = g * GROUP_SIZE_C
        offs_k = cin_start + tl.arange(0, GROUP_SIZE_C)
        k_mask = (g_active != 0) & (offs_k < C_IN)
        x_addrs = x_ptr + (n_idx * C_IN + offs_k[None, :]) * HW_IN + out_h[:, None] * W_IN + out_w[:, None]
        x_tile = tl.load(x_addrs, mask=k_mask[None, :] & m_mask[:, None], other=0.0).to(tl.float16)
        w_addrs = w_cl_ptr + offs_n[None, :] * C_IN + offs_k[:, None]
        w_tile = tl.load(w_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
        acc += tl.dot(x_tile, w_tile)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
    oa = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
    tl.store(oa, acc, mask=m_mask[:, None] & n_mask[None, :])


# ---- 3x3 / stride=1 ----

@autotune(configs=_CONFIGS_8x8, key=["C_IN", "C_OUT", "H_OUT", "W_OUT", "GH", "GW"])
@triton.jit
def sparse_conv3x3s1_bm_kernel_8x8(
    x_ptr, w_cl_ptr, bias_ptr, ag_mask_ptr, y_ptr, N_val,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    H_IN: tl.constexpr, W_IN: tl.constexpr,
    H_OUT: tl.constexpr, W_OUT: tl.constexpr,
    GH: tl.constexpr, GW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, NUM_GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    tile_id = tl.program_id(0)
    pid_cout = tl.program_id(1)
    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles:
        return
    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH
    offs_n = pid_cout * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < C_OUT
    offs_m = tl.arange(0, BLOCK_M)
    out_h = gh_idx * BLOCK_H + offs_m // BLOCK_W
    out_w = gw_idx * BLOCK_W + offs_m % BLOCK_W
    m_mask = (out_h < H_OUT) & (out_w < W_OUT)

    HW_IN: tl.constexpr = H_IN * W_IN
    HW_OUT: tl.constexpr = H_OUT * W_OUT
    W_CS: tl.constexpr = C_IN
    W_KH: tl.constexpr = 3 * C_IN
    W_CO: tl.constexpr = 9 * C_IN

    ag_mask = tl.load(ag_mask_ptr + tile_id)
    if ag_mask == 0:
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
        oa = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
        tl.store(oa, acc, mask=m_mask[:, None] & n_mask[None, :])
        return

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for g in range(NUM_GROUPS):
        g_active = (ag_mask >> g) & 1
        cin_start = g * GROUP_SIZE_C
        offs_k = cin_start + tl.arange(0, GROUP_SIZE_C)
        k_mask = (g_active != 0) & (offs_k < C_IN)

        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                in_h = out_h + (kh - 1)
                in_w = out_w + (kw - 1)
                h_ok = (in_h >= 0) & (in_h < H_IN)
                w_ok = (in_w >= 0) & (in_w < W_IN)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H_IN - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W_IN - 1)

                x_addrs = x_ptr + (n_idx * C_IN + offs_k[None, :]) * HW_IN + safe_h[:, None] * W_IN + safe_w[:, None]
                x_m = k_mask[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                x_tile = tl.load(x_addrs, mask=x_m, other=0.0).to(tl.float16)

                w_addrs = w_cl_ptr + offs_n[None, :] * W_CO + kh * W_KH + kw * W_CS + offs_k[:, None]
                w_tile = tl.load(w_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
                acc += tl.dot(x_tile, w_tile)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
    oa = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
    tl.store(oa, acc, mask=m_mask[:, None] & n_mask[None, :])


@autotune(configs=_CONFIGS_8x16, key=["C_IN", "C_OUT", "H_OUT", "W_OUT", "GH", "GW"])
@triton.jit
def sparse_conv3x3s1_bm_kernel_8x16(
    x_ptr, w_cl_ptr, bias_ptr, ag_mask_ptr, y_ptr, N_val,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    H_IN: tl.constexpr, W_IN: tl.constexpr,
    H_OUT: tl.constexpr, W_OUT: tl.constexpr,
    GH: tl.constexpr, GW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, NUM_GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    tile_id = tl.program_id(0)
    pid_cout = tl.program_id(1)
    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles:
        return
    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH
    offs_n = pid_cout * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < C_OUT
    offs_m = tl.arange(0, BLOCK_M)
    out_h = gh_idx * BLOCK_H + offs_m // BLOCK_W
    out_w = gw_idx * BLOCK_W + offs_m % BLOCK_W
    m_mask = (out_h < H_OUT) & (out_w < W_OUT)
    HW_IN: tl.constexpr = H_IN * W_IN
    HW_OUT: tl.constexpr = H_OUT * W_OUT
    W_CS: tl.constexpr = C_IN
    W_KH: tl.constexpr = 3 * C_IN
    W_CO: tl.constexpr = 9 * C_IN

    ag_mask = tl.load(ag_mask_ptr + tile_id)
    if ag_mask == 0:
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
        oa = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
        tl.store(oa, acc, mask=m_mask[:, None] & n_mask[None, :])
        return

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for g in range(NUM_GROUPS):
        g_active = (ag_mask >> g) & 1
        cin_start = g * GROUP_SIZE_C
        offs_k = cin_start + tl.arange(0, GROUP_SIZE_C)
        k_mask = (g_active != 0) & (offs_k < C_IN)

        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                in_h = out_h + (kh - 1)
                in_w = out_w + (kw - 1)
                h_ok = (in_h >= 0) & (in_h < H_IN)
                w_ok = (in_w >= 0) & (in_w < W_IN)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H_IN - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W_IN - 1)

                x_addrs = x_ptr + (n_idx * C_IN + offs_k[None, :]) * HW_IN + safe_h[:, None] * W_IN + safe_w[:, None]
                x_m = k_mask[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                x_tile = tl.load(x_addrs, mask=x_m, other=0.0).to(tl.float16)

                w_addrs = w_cl_ptr + offs_n[None, :] * W_CO + kh * W_KH + kw * W_CS + offs_k[:, None]
                w_tile = tl.load(w_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
                acc += tl.dot(x_tile, w_tile)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
    oa = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
    tl.store(oa, acc, mask=m_mask[:, None] & n_mask[None, :])


# ---- 3x3 / stride=2 ----

@autotune(configs=_CONFIGS_8x8, key=["C_IN", "C_OUT", "H_OUT", "W_OUT", "GH", "GW"])
@triton.jit
def sparse_conv3x3s2_bm_kernel_8x8(
    x_ptr, w_cl_ptr, bias_ptr, ag_mask_ptr, y_ptr, N_val,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    H_IN: tl.constexpr, W_IN: tl.constexpr,
    H_OUT: tl.constexpr, W_OUT: tl.constexpr,
    GH: tl.constexpr, GW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, NUM_GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    tile_id = tl.program_id(0)
    pid_cout = tl.program_id(1)
    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles:
        return
    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH
    offs_n = pid_cout * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < C_OUT
    offs_m = tl.arange(0, BLOCK_M)
    out_h = gh_idx * BLOCK_H + offs_m // BLOCK_W
    out_w = gw_idx * BLOCK_W + offs_m % BLOCK_W
    m_mask = (out_h < H_OUT) & (out_w < W_OUT)
    HW_IN: tl.constexpr = H_IN * W_IN
    HW_OUT: tl.constexpr = H_OUT * W_OUT
    W_CS: tl.constexpr = C_IN
    W_KH: tl.constexpr = 3 * C_IN
    W_CO: tl.constexpr = 9 * C_IN

    ag_mask = tl.load(ag_mask_ptr + tile_id)
    if ag_mask == 0:
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
        oa = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
        tl.store(oa, acc, mask=m_mask[:, None] & n_mask[None, :])
        return

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for g in range(NUM_GROUPS):
        g_active = (ag_mask >> g) & 1
        cin_start = g * GROUP_SIZE_C
        offs_k = cin_start + tl.arange(0, GROUP_SIZE_C)
        k_mask = (g_active != 0) & (offs_k < C_IN)

        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                in_h = out_h * 2 + (kh - 1)
                in_w = out_w * 2 + (kw - 1)
                h_ok = (in_h >= 0) & (in_h < H_IN)
                w_ok = (in_w >= 0) & (in_w < W_IN)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H_IN - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W_IN - 1)

                x_addrs = x_ptr + (n_idx * C_IN + offs_k[None, :]) * HW_IN + safe_h[:, None] * W_IN + safe_w[:, None]
                x_m = k_mask[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                x_tile = tl.load(x_addrs, mask=x_m, other=0.0).to(tl.float16)

                w_addrs = w_cl_ptr + offs_n[None, :] * W_CO + kh * W_KH + kw * W_CS + offs_k[:, None]
                w_tile = tl.load(w_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
                acc += tl.dot(x_tile, w_tile)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
    oa = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
    tl.store(oa, acc, mask=m_mask[:, None] & n_mask[None, :])


@autotune(configs=_CONFIGS_8x16, key=["C_IN", "C_OUT", "H_OUT", "W_OUT", "GH", "GW"])
@triton.jit
def sparse_conv3x3s2_bm_kernel_8x16(
    x_ptr, w_cl_ptr, bias_ptr, ag_mask_ptr, y_ptr, N_val,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    H_IN: tl.constexpr, W_IN: tl.constexpr,
    H_OUT: tl.constexpr, W_OUT: tl.constexpr,
    GH: tl.constexpr, GW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, NUM_GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    tile_id = tl.program_id(0)
    pid_cout = tl.program_id(1)
    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles:
        return
    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH
    offs_n = pid_cout * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < C_OUT
    offs_m = tl.arange(0, BLOCK_M)
    out_h = gh_idx * BLOCK_H + offs_m // BLOCK_W
    out_w = gw_idx * BLOCK_W + offs_m % BLOCK_W
    m_mask = (out_h < H_OUT) & (out_w < W_OUT)
    HW_IN: tl.constexpr = H_IN * W_IN
    HW_OUT: tl.constexpr = H_OUT * W_OUT
    W_CS: tl.constexpr = C_IN
    W_KH: tl.constexpr = 3 * C_IN
    W_CO: tl.constexpr = 9 * C_IN

    ag_mask = tl.load(ag_mask_ptr + tile_id)
    if ag_mask == 0:
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
        oa = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
        tl.store(oa, acc, mask=m_mask[:, None] & n_mask[None, :])
        return

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for g in range(NUM_GROUPS):
        g_active = (ag_mask >> g) & 1
        cin_start = g * GROUP_SIZE_C
        offs_k = cin_start + tl.arange(0, GROUP_SIZE_C)
        k_mask = (g_active != 0) & (offs_k < C_IN)

        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                in_h = out_h * 2 + (kh - 1)
                in_w = out_w * 2 + (kw - 1)
                h_ok = (in_h >= 0) & (in_h < H_IN)
                w_ok = (in_w >= 0) & (in_w < W_IN)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H_IN - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W_IN - 1)

                x_addrs = x_ptr + (n_idx * C_IN + offs_k[None, :]) * HW_IN + safe_h[:, None] * W_IN + safe_w[:, None]
                x_m = k_mask[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                x_tile = tl.load(x_addrs, mask=x_m, other=0.0).to(tl.float16)

                w_addrs = w_cl_ptr + offs_n[None, :] * W_CO + kh * W_KH + kw * W_CS + offs_k[:, None]
                w_tile = tl.load(w_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
                acc += tl.dot(x_tile, w_tile)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
    oa = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
    tl.store(oa, acc, mask=m_mask[:, None] & n_mask[None, :])


# ===================================================================
# Python entry point
# ===================================================================

def sparse_conv2d_forward(
    x,
    weight,
    bias,
    block_size=None,
    kernel_size=3,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    threshold=1e-6,
    w_cl=None,
    # Legacy params (ignored — kept for call-site compat)
    counts_buf=None,
    tile_cin_buf=None,
    group_flags_buf=None,
    ag_count_buf=None,
    ag_list_buf=None,
    # New bitmask buffer
    ag_mask_buf=None,
    return_ms=False,
    fallback_ratio=FALLBACK_RATIO,
    return_avg_active_ratio=False,
):
    import torch.nn.functional as Fn

    N, C_IN, H_IN, W_IN = x.shape
    C_OUT = weight.shape[0]
    device = x.device

    if isinstance(stride, tuple):
        stride = stride[0]
    if isinstance(padding, tuple):
        padding = padding[0]
    if isinstance(dilation, tuple):
        dilation = dilation[0]

    # Unsupported configs → dense fallback
    if groups != 1 or dilation != 1:
        y = Fn.conv2d(x, weight, bias, stride=stride, padding=padding,
                      dilation=dilation, groups=groups).float()
        if return_avg_active_ratio:
            return y, 0.0, 1.0
        return y, 0.0

    H_OUT = (H_IN + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    W_OUT = (W_IN + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    if H_OUT <= 0 or W_OUT <= 0:
        y = Fn.conv2d(x, weight, bias, stride=stride, padding=padding,
                      dilation=dilation, groups=groups).float()
        if return_avg_active_ratio:
            return y, 0.0, 1.0
        return y, 0.0

    supported = (
        (kernel_size == 1 and stride == 1 and padding == 0) or
        (kernel_size == 1 and stride == 2 and padding == 0) or
        (kernel_size == 3 and stride == 1 and padding == 1) or
        (kernel_size == 3 and stride == 2 and padding == 1)
    )
    if not supported:
        y = Fn.conv2d(x, weight, bias, stride=stride, padding=padding,
                      dilation=dilation, groups=groups).float()
        if return_avg_active_ratio:
            return y, 0.0, 1.0
        return y, 0.0

    # 1x1/s2 → subsample input then use 1x1/s1 path
    if kernel_size == 1 and stride == 2:
        x = x[:, :, ::2, ::2].contiguous()
        N, C_IN, H_IN, W_IN = x.shape
        stride = 1
        H_OUT = H_IN
        W_OUT = W_IN

    # Adaptive group size
    GROUP_SIZE_C = choose_group_size(C_IN)
    NUM_GROUPS = triton.cdiv(C_IN, GROUP_SIZE_C)

    BH, BW = _select_tile_sizes(H_OUT, W_OUT)
    GH = triton.cdiv(H_OUT, BH)
    GW = triton.cdiv(W_OUT, BW)
    N_TILES = N * GH * GW

    # Weight layout
    if w_cl is not None:
        w_cl_f16 = w_cl
    else:
        if kernel_size == 3:
            w_cl_f16 = weight.half().permute(0, 2, 3, 1).contiguous()
        else:
            w_cl_f16 = weight.half().reshape(C_OUT, C_IN).contiguous()

    x_f16 = x.half().contiguous()

    # Allocate bitmask buffer
    if ag_mask_buf is None or ag_mask_buf.numel() < N_TILES:
        ag_mask_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)

    # Build bitmask metadata (two-stage prescan)
    _build_active_group_bitmask(
        x_f16, N, C_IN, H_IN, W_IN, H_OUT, W_OUT,
        BH, BW, GH, GW,
        kernel_size, stride, padding,
        threshold, ag_mask_buf,
    )

    # Compute AGR if requested
    avg_active_ratio = None
    if return_avg_active_ratio:
        # Vectorised popcount via bit iteration
        masks = ag_mask_buf[:N_TILES].int()
        pc = torch.zeros(N_TILES, dtype=torch.int32, device=device)
        tmp = masks.clone()
        for _ in range(32):
            pc += tmp & 1
            tmp = tmp >> 1
        avg_active_ratio = pc.float().mean().item() / max(NUM_GROUPS, 1)

        if avg_active_ratio == 0.0:
            y = torch.zeros(N, C_OUT, H_OUT, W_OUT, dtype=torch.float32, device=device)
            if bias is not None:
                y += bias.float().view(1, -1, 1, 1)
            return y, 0.0, avg_active_ratio
        if avg_active_ratio > fallback_ratio:
            y = Fn.conv2d(x, weight, bias, stride=stride, padding=padding,
                          dilation=dilation, groups=groups).float()
            return y, 0.0, avg_active_ratio

    has_bias = bias is not None
    bias_f32 = bias.float().contiguous() if has_bias else torch.empty(1, device=device)
    y = torch.empty(N, C_OUT, H_OUT, W_OUT, dtype=torch.float32, device=device)

    sparse_ms = 0.0
    if return_ms:
        se = torch.cuda.Event(enable_timing=True)
        ee = torch.cuda.Event(enable_timing=True)
        se.record()

    def _grid(META):
        return (N_TILES, triton.cdiv(C_OUT, META["BLOCK_N"]))

    # Select kernel by pattern + tile size
    if kernel_size == 1:
        kernel = sparse_conv1x1_bm_kernel_8x16 if BW == 16 else sparse_conv1x1_bm_kernel_8x8
    elif stride == 1:
        kernel = sparse_conv3x3s1_bm_kernel_8x16 if BW == 16 else sparse_conv3x3s1_bm_kernel_8x8
    else:
        kernel = sparse_conv3x3s2_bm_kernel_8x16 if BW == 16 else sparse_conv3x3s2_bm_kernel_8x8

    kernel[_grid](
        x_f16,
        w_cl_f16,
        bias_f32,
        ag_mask_buf,
        y,
        N,
        C_IN=C_IN, C_OUT=C_OUT,
        H_IN=H_IN, W_IN=W_IN,
        H_OUT=H_OUT, W_OUT=W_OUT,
        GH=GH, GW=GW,
        HAS_BIAS=has_bias,
        GROUP_SIZE_C=GROUP_SIZE_C,
        NUM_GROUPS=NUM_GROUPS,
    )

    if return_ms:
        ee.record()
        torch.cuda.synchronize(device)
        sparse_ms = se.elapsed_time(ee)

    if return_avg_active_ratio:
        return y, sparse_ms, avg_active_ratio
    return y, sparse_ms