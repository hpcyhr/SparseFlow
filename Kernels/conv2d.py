"""
SparseFlow Conv2d Triton Kernels — v21.0 Refined Two-Stage Prescan

Changes from v20:
  A. Stage 1 is now genuinely cheap for ALL tiles:
     - Checks only the first channel per group across RF positions
     - Classifies as DENSEISH (all groups' rep channels active) or UNCERTAIN
     - NO expensive exact-zero verification inline — that moves to Stage 2
     - Cost: O(NUM_GROUPS × RF_SIZE) per tile, uniformly
  B. Stage 2 handles all refinement for UNCERTAIN tiles:
     - Full per-group scan → exact bitmask
     - Classifies as ZERO (mask==0), DENSEISH (mask==ALL_ONES), or SPARSE
     - Includes early-exit when all groups found active
  C. Dense-ish tile execution path is more truly dense-like:
     - 3x3 kernels: spatial-outer / channel-inner loop order
       → spatial addresses computed once per (kh,kw), reused across channel groups
     - 1x1 kernels: pre-computed spatial base, flat channel iteration
     - Structurally distinct from the sparse bitmask-gated path
  D. Tile classification uses named constants throughout:
     - TILE_ZERO = 0
     - TILE_SPARSE = 1
     - TILE_DENSEISH = 2
     - TILE_UNCERTAIN = 3  (no more magic number)

Supported patterns:
  1x1/s1/p0, 1x1/s2/p0 (via subsample), 3x3/s1/p1, 3x3/s2/p1
"""

import torch
import triton
import triton.language as tl
from triton import autotune, Config

# ---------------------------------------------------------------------------
# Tile classification constants
# ---------------------------------------------------------------------------
TILE_ZERO = 0
TILE_SPARSE = 1
TILE_DENSEISH = 2
TILE_UNCERTAIN = 3

FALLBACK_RATIO = 0.85


def choose_group_size(c_in: int) -> int:
    if c_in <= 128:
        gs = 16
    else:
        gs = 32
    num_groups = (c_in + gs - 1) // gs
    while num_groups > 32:
        gs *= 2
        num_groups = (c_in + gs - 1) // gs
    return gs


# Legacy alias
GROUP_SIZE = 32


def _select_tile_sizes(H, W):
    pixels = H * W
    if pixels >= 3136:
        return 8, 16
    return 8, 8


def _select_block_sizes(H, W, C_IN, C_OUT, kernel_size, N):
    BH, BW = _select_tile_sizes(H, W)
    gs = choose_group_size(C_IN)
    return BH, BW, BH * BW, 64, gs


# ===========================================================================
# STAGE 1: Coarse tile classification — cheap for ALL tiles
# ===========================================================================
# Checks ONE representative channel (first) per group across RF positions.
# Cost: O(NUM_GROUPS × RF_SIZE) per tile — uniformly cheap.
#
# Classification:
#   - ALL groups' rep channels active → DENSEISH
#     (safe: dense path iterates all channels unconditionally)
#   - Otherwise → UNCERTAIN (includes possible-zero and mixed)
#     (Stage 2 will refine with full per-group scan)
#
# KEY DESIGN POINT:
#   Stage 1 does NOT attempt to confirm zero tiles.
#   The expensive full-channel verification is entirely in Stage 2.
#   This keeps Stage 1 uniformly cheap with no data-dependent branches.
# ===========================================================================

@triton.jit
def tile_coarse_classify_kernel(
    x_ptr,
    tile_class_ptr,       # [N_TILES] int32 output
    ag_mask_ptr,          # [N_TILES] int32 output: preliminary rough mask
    N_val,
    C_IN,
    H_IN,
    W_IN,
    GH,
    GW,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
    RF_SIZE: tl.constexpr,
    THRESHOLD: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    ALL_ONES_MASK: tl.constexpr,
    UNCERTAIN_CLASS: tl.constexpr,
):
    """Stage 1: Coarse classification for 3x3 convolutions.

    Checks first channel per group across all RF positions.
    Only classifies DENSEISH (all groups active) vs UNCERTAIN.
    No expensive zero verification — that is Stage 2's job.
    """
    tile_id = tl.program_id(0)
    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles:
        return

    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH

    # Compute receptive field positions
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
    off1 = tl.arange(0, 1)

    # ---- Coarse group check: first channel per group ----
    rough_mask = tl.zeros([1], dtype=tl.int32)
    for g in range(NUM_GROUPS):
        c_rep = g * GROUP_SIZE_C  # first channel of group
        if c_rep < C_IN:
            base = (n_idx * C_IN + c_rep) * HW
            vals = tl.load(
                x_ptr + base + safe_h * W_IN + safe_w,
                mask=hw_mask,
                other=0.0,
            )
            ch_max = tl.max(vals, axis=0)
            is_active = (ch_max > THRESHOLD).to(tl.int32)
            rough_mask = rough_mask + is_active * (1 << g)

    # ---- Classify ----
    if tl.sum(rough_mask == ALL_ONES_MASK) > 0:
        # All groups have active first channel → DENSEISH
        tl.store(tile_class_ptr + tile_id + off1,
                 tl.full([1], 2, dtype=tl.int32))  # TILE_DENSEISH
        tl.store(ag_mask_ptr + tile_id + off1,
                 tl.full([1], ALL_ONES_MASK, dtype=tl.int32))
    else:
        # UNCERTAIN — Stage 2 will do the full group scan
        tl.store(tile_class_ptr + tile_id + off1,
                 tl.full([1], UNCERTAIN_CLASS, dtype=tl.int32))
        tl.store(ag_mask_ptr + tile_id + off1, rough_mask)


@triton.jit
def tile_coarse_classify_1x1_kernel(
    x_ptr,
    tile_class_ptr,
    ag_mask_ptr,
    N_val,
    C_IN,
    H_IN,
    W_IN,
    GH,
    GW,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    THRESHOLD: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    ALL_ONES_MASK: tl.constexpr,
    UNCERTAIN_CLASS: tl.constexpr,
):
    """Stage 1 for 1x1 conv: same cheap coarse classification."""
    tile_id = tl.program_id(0)
    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles:
        return

    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH

    offs_m = tl.arange(0, BLOCK_M)
    out_h = gh_idx * BLOCK_H + offs_m // BLOCK_W
    out_w = gw_idx * BLOCK_W + offs_m % BLOCK_W
    m_mask = (out_h < H_IN) & (out_w < W_IN)
    HW = H_IN * W_IN
    off1 = tl.arange(0, 1)

    # ---- Coarse group check: first channel per group ----
    rough_mask = tl.zeros([1], dtype=tl.int32)
    for g in range(NUM_GROUPS):
        c_rep = g * GROUP_SIZE_C
        if c_rep < C_IN:
            addrs = (n_idx * C_IN + c_rep) * HW + out_h * W_IN + out_w
            vals = tl.load(x_ptr + addrs, mask=m_mask, other=0.0)
            ch_max = tl.max(vals, axis=0)
            is_active = (ch_max > THRESHOLD).to(tl.int32)
            rough_mask = rough_mask + is_active * (1 << g)

    # ---- Classify ----
    if tl.sum(rough_mask == ALL_ONES_MASK) > 0:
        tl.store(tile_class_ptr + tile_id + off1,
                 tl.full([1], 2, dtype=tl.int32))  # TILE_DENSEISH
        tl.store(ag_mask_ptr + tile_id + off1,
                 tl.full([1], ALL_ONES_MASK, dtype=tl.int32))
    else:
        tl.store(tile_class_ptr + tile_id + off1,
                 tl.full([1], UNCERTAIN_CLASS, dtype=tl.int32))
        tl.store(ag_mask_ptr + tile_id + off1, rough_mask)


# ===========================================================================
# STAGE 2: Exact group bitmask refinement — ONLY for UNCERTAIN tiles
# ===========================================================================
# Tiles classified DENSEISH by Stage 1 are SKIPPED (early return).
# UNCERTAIN tiles get the full per-group channel scan.
# After building exact bitmask:
#   mask == 0         → TILE_ZERO
#   mask == ALL_ONES  → TILE_DENSEISH
#   otherwise         → TILE_SPARSE
#
# This is where the expensive work lives — but only for tiles that need it.
# ===========================================================================

@triton.jit
def group_bitmask_refine_kernel(
    x_ptr,
    tile_class_ptr,       # [N_TILES] int32 — read: skip if != UNCERTAIN
    ag_mask_ptr,          # [N_TILES] int32 — overwrite with exact bitmask
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
    ALL_ONES_MASK: tl.constexpr,
    UNCERTAIN_CLASS: tl.constexpr,
):
    """Stage 2: Exact per-group bitmask for UNCERTAIN tiles only."""
    tile_id = tl.program_id(0)
    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles:
        return

    off1 = tl.arange(0, 1)

    # ---- Skip non-uncertain tiles ----
    tc = tl.load(tile_class_ptr + tile_id + off1)
    if tl.sum(tc) != UNCERTAIN_CLASS:
        return

    # ---- Compute RF positions ----
    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH

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

    # ---- Full group scan with early exit on all-active ----
    mask = tl.zeros([1], dtype=tl.int32)
    g = tl.zeros([1], dtype=tl.int32)

    while tl.sum(g) < NUM_GROUPS:
        g_val = tl.sum(g)
        group_max = tl.zeros([1], dtype=tl.float32)

        for c_off in range(GROUP_SIZE_C):
            c = g_val * GROUP_SIZE_C + c_off
            if c < C_IN:
                base = (n_idx * C_IN + c) * HW
                vals = tl.load(
                    x_ptr + base + safe_h * W_IN + safe_w,
                    mask=hw_mask,
                    other=0.0,
                )
                ch_max = tl.max(vals, axis=0)
                group_max = tl.maximum(group_max, ch_max)

        is_active = (group_max > THRESHOLD).to(tl.int32)
        mask = mask + is_active * (1 << g_val)

        # Early exit: all groups found active
        if tl.sum(mask == ALL_ONES_MASK) > 0:
            g = tl.full([1], NUM_GROUPS, dtype=tl.int32)
        else:
            g = g + 1

    # ---- Store exact bitmask and final classification ----
    tl.store(ag_mask_ptr + tile_id + off1, mask)

    if tl.sum(mask) == 0:
        # TILE_ZERO
        tl.store(tile_class_ptr + tile_id + off1,
                 tl.zeros([1], dtype=tl.int32))
    else:
        is_denseish = (mask == ALL_ONES_MASK).to(tl.int32)
        # TILE_SPARSE=1, TILE_DENSEISH=2
        tile_class = 1 + is_denseish
        tl.store(tile_class_ptr + tile_id + off1, tile_class)


@triton.jit
def group_bitmask_refine_1x1_kernel(
    x_ptr,
    tile_class_ptr,
    ag_mask_ptr,
    N_val,
    C_IN,
    H_IN,
    W_IN,
    GH,
    GW,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    THRESHOLD: tl.constexpr,
    ALL_ONES_MASK: tl.constexpr,
    UNCERTAIN_CLASS: tl.constexpr,
):
    """Stage 2 for 1x1 conv: exact bitmask for UNCERTAIN tiles only."""
    tile_id = tl.program_id(0)
    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles:
        return

    off1 = tl.arange(0, 1)
    tc = tl.load(tile_class_ptr + tile_id + off1)
    if tl.sum(tc) != UNCERTAIN_CLASS:
        return

    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH

    offs_m = tl.arange(0, BLOCK_M)
    out_h = gh_idx * BLOCK_H + offs_m // BLOCK_W
    out_w = gw_idx * BLOCK_W + offs_m % BLOCK_W
    m_mask = (out_h < H_IN) & (out_w < W_IN)
    HW = H_IN * W_IN

    # Full group scan with early exit
    mask = tl.zeros([1], dtype=tl.int32)
    g = tl.zeros([1], dtype=tl.int32)

    while tl.sum(g) < NUM_GROUPS:
        g_val = tl.sum(g)
        group_max = tl.zeros([1], dtype=tl.float32)
        for c_off in range(GROUP_SIZE_C):
            c = g_val * GROUP_SIZE_C + c_off
            if c < C_IN:
                addrs = (n_idx * C_IN + c) * HW + out_h * W_IN + out_w
                vals = tl.load(x_ptr + addrs, mask=m_mask, other=0.0)
                ch_max = tl.max(vals, axis=0)
                group_max = tl.maximum(group_max, ch_max)

        is_active = (group_max > THRESHOLD).to(tl.int32)
        mask = mask + is_active * (1 << g_val)

        if tl.sum(mask == ALL_ONES_MASK) > 0:
            g = tl.full([1], NUM_GROUPS, dtype=tl.int32)
        else:
            g = g + 1

    tl.store(ag_mask_ptr + tile_id + off1, mask)

    if tl.sum(mask) == 0:
        tl.store(tile_class_ptr + tile_id + off1,
                 tl.zeros([1], dtype=tl.int32))  # TILE_ZERO
    else:
        is_denseish = (mask == ALL_ONES_MASK).to(tl.int32)
        tile_class = 1 + is_denseish
        tl.store(tile_class_ptr + tile_id + off1, tile_class)


# ---------------------------------------------------------------------------
# Legacy Stage 1 kernels — kept for backward compat with fused_conv_lif
# ---------------------------------------------------------------------------

@triton.jit
def tile_screen_kernel(
    x_ptr,
    tile_alive_ptr,
    N_val,
    C_IN,
    H_IN,
    W_IN,
    GH,
    GW,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
    RF_SIZE: tl.constexpr,
    THRESHOLD: tl.constexpr,
    C_BATCH: tl.constexpr,
):
    """Legacy compat: zero/alive classification with early exit."""
    tile_id = tl.program_id(0)
    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles:
        return

    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH

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
    c_idx = 0
    found = tl.zeros([1], dtype=tl.int32)

    while (c_idx < C_IN) & (tl.sum(found) == 0):
        base = (n_idx * C_IN + c_idx) * HW
        vals = tl.load(
            x_ptr + base + safe_h * W_IN + safe_w,
            mask=hw_mask,
            other=0.0,
        )
        ch_max = tl.max(vals, axis=0)
        found = found + (ch_max > THRESHOLD).to(tl.int32)
        c_idx += 1

    off1 = tl.arange(0, 1)
    tl.store(tile_alive_ptr + tile_id + off1, found)


@triton.jit
def tile_screen_1x1_kernel(
    x_ptr,
    tile_alive_ptr,
    N_val,
    C_IN,
    H_IN,
    W_IN,
    GH,
    GW,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
    """Legacy compat: 1x1 zero/alive classification."""
    tile_id = tl.program_id(0)
    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles:
        return

    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH

    offs_m = tl.arange(0, BLOCK_M)
    out_h = gh_idx * BLOCK_H + offs_m // BLOCK_W
    out_w = gw_idx * BLOCK_W + offs_m % BLOCK_W
    m_mask = (out_h < H_IN) & (out_w < W_IN)
    HW = H_IN * W_IN

    c_idx = 0
    found = tl.zeros([1], dtype=tl.int32)

    while (c_idx < C_IN) & (tl.sum(found) == 0):
        addrs = (n_idx * C_IN + c_idx) * HW + out_h * W_IN + out_w
        vals = tl.load(x_ptr + addrs, mask=m_mask, other=0.0)
        ch_max = tl.max(vals, axis=0)
        found = found + (ch_max > THRESHOLD).to(tl.int32)
        c_idx += 1

    off1 = tl.arange(0, 1)
    tl.store(tile_alive_ptr + tile_id + off1, found)


# ---------------------------------------------------------------------------
# Python helpers — two-stage metadata construction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _build_two_stage_metadata(
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
    tile_alive_buf,    # unused in v21 (kept for buffer compat)
    ag_mask_buf,
    tile_class_buf,
):
    """Real two-stage prescan: cheap coarse classify → conditional exact refinement.

    Stage 1: O(NUM_GROUPS × RF_SIZE) — classifies DENSEISH or UNCERTAIN
    Stage 2: O(C_IN × RF_SIZE) — only for UNCERTAIN tiles

    Returns (GROUP_SIZE_C, NUM_GROUPS).
    """
    GROUP_SIZE_C = choose_group_size(C_IN)
    N_TILES = N * GH * GW
    NUM_GROUPS = triton.cdiv(C_IN, GROUP_SIZE_C)
    ALL_ONES_MASK = (1 << NUM_GROUPS) - 1

    # --- Stage 1: coarse tile classification (cheap) ---
    if kernel_size == 1:
        BM = BH * BW
        tile_coarse_classify_1x1_kernel[(N_TILES,)](
            x_f16,
            tile_class_buf,
            ag_mask_buf,
            N, C_IN, H_IN, W_IN, GH, GW,
            BLOCK_H=BH, BLOCK_W=BW, BLOCK_M=BM,
            THRESHOLD=threshold,
            GROUP_SIZE_C=GROUP_SIZE_C,
            NUM_GROUPS=NUM_GROUPS,
            ALL_ONES_MASK=ALL_ONES_MASK,
            UNCERTAIN_CLASS=TILE_UNCERTAIN,
        )
    else:
        rf_h = (BH - 1) * stride + kernel_size
        rf_w = (BW - 1) * stride + kernel_size
        RF_SIZE = triton.next_power_of_2(max(rf_h * rf_w, 1))

        tile_coarse_classify_kernel[(N_TILES,)](
            x_f16,
            tile_class_buf,
            ag_mask_buf,
            N, C_IN, H_IN, W_IN, GH, GW,
            BLOCK_H=BH, BLOCK_W=BW,
            KERNEL_SIZE=kernel_size, STRIDE=stride, PAD=padding,
            RF_SIZE=RF_SIZE, THRESHOLD=threshold,
            GROUP_SIZE_C=GROUP_SIZE_C,
            NUM_GROUPS=NUM_GROUPS,
            ALL_ONES_MASK=ALL_ONES_MASK,
            UNCERTAIN_CLASS=TILE_UNCERTAIN,
        )

    # --- Stage 2: exact bitmask for UNCERTAIN tiles only ---
    if kernel_size == 1:
        BM = BH * BW
        group_bitmask_refine_1x1_kernel[(N_TILES,)](
            x_f16,
            tile_class_buf,
            ag_mask_buf,
            N, C_IN, H_IN, W_IN, GH, GW,
            BLOCK_H=BH, BLOCK_W=BW, BLOCK_M=BM,
            GROUP_SIZE_C=GROUP_SIZE_C, NUM_GROUPS=NUM_GROUPS,
            THRESHOLD=threshold, ALL_ONES_MASK=ALL_ONES_MASK,
            UNCERTAIN_CLASS=TILE_UNCERTAIN,
        )
    else:
        rf_h = (BH - 1) * stride + kernel_size
        rf_w = (BW - 1) * stride + kernel_size
        RF_SIZE = triton.next_power_of_2(max(rf_h * rf_w, 1))

        group_bitmask_refine_kernel[(N_TILES,)](
            x_f16,
            tile_class_buf,
            ag_mask_buf,
            N, C_IN, H_IN, W_IN, H_OUT, W_OUT, GH, GW,
            BLOCK_H=BH, BLOCK_W=BW,
            KERNEL_SIZE=kernel_size, STRIDE=stride, PAD=padding,
            GROUP_SIZE_C=GROUP_SIZE_C, NUM_GROUPS=NUM_GROUPS,
            RF_SIZE=RF_SIZE, THRESHOLD=threshold,
            ALL_ONES_MASK=ALL_ONES_MASK,
            UNCERTAIN_CLASS=TILE_UNCERTAIN,
        )

    return GROUP_SIZE_C, NUM_GROUPS


# Backward-compat aliases for fused_conv_lif.py and other callers
def _build_active_group_bitmask(
    x_f16, N, C_IN, H_IN, W_IN, H_OUT, W_OUT,
    BH, BW, GH, GW, kernel_size, stride, padding,
    threshold, ag_mask_buf,
):
    """Compat shim — allocates temp buffers and calls two-stage builder."""
    N_TILES = N * GH * GW
    device = x_f16.device
    tile_alive_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
    tile_class_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
    return _build_two_stage_metadata(
        x_f16, N, C_IN, H_IN, W_IN, H_OUT, W_OUT,
        BH, BW, GH, GW, kernel_size, stride, padding,
        threshold, tile_alive_buf, ag_mask_buf, tile_class_buf,
    )


def _build_active_group_metadata(
    x_f16, N, C_IN, H_IN, W_IN, H_OUT, W_OUT,
    BH, BW, GH, GW, kernel_size, stride, padding,
    threshold, ag_count_buf, ag_list_buf,
):
    """Legacy shim for fused_conv_lif.py."""
    return _build_active_group_bitmask(
        x_f16, N, C_IN, H_IN, W_IN, H_OUT, W_OUT,
        BH, BW, GH, GW, kernel_size, stride, padding,
        threshold, ag_count_buf,
    )


def _check_dense_fallback(ag_mask_buf, N_TILES, NUM_GROUPS, fallback_ratio=FALLBACK_RATIO):
    if NUM_GROUPS == 0:
        return False
    masks = ag_mask_buf[:N_TILES].int()
    pc = torch.zeros(N_TILES, dtype=torch.int32, device=masks.device)
    tmp = masks.clone()
    for _ in range(32):
        pc += tmp & 1
        tmp = tmp >> 1
    avg_active = pc.float().mean().item()
    threshold = fallback_ratio * NUM_GROUPS
    return avg_active > threshold


def _popcount_buf(ag_mask_buf, N_TILES):
    """Vectorised popcount for int32 bitmask buffer."""
    masks = ag_mask_buf[:N_TILES].int()
    pc = torch.zeros(N_TILES, dtype=torch.int32, device=masks.device)
    tmp = masks.clone()
    for _ in range(32):
        pc += tmp & 1
        tmp = tmp >> 1
    return pc


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
# Compute kernels — three tile execution paths:
#
#   TILE_ZERO (ag_mask==0):
#     → bias-only early return
#
#   TILE_DENSEISH (ag_mask==ALL_ONES):
#     → dense-like tile path
#     → 3x3: spatial-outer / channel-inner loop order
#       (spatial addresses & masks computed once per kh/kw, reused
#        across all channel groups — eliminates redundant spatial math)
#     → 1x1: pre-computed spatial base, flat channel iteration
#     → Structurally distinct from the sparse bitmask-gated path
#
#   TILE_SPARSE (ag_mask is partial):
#     → bitmask-gated group iteration (group-outer, spatial-inner)
# ===================================================================

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
    ALL_ONES_MASK: tl.constexpr,
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

    # --- TILE_ZERO: bias-only ---
    if ag_mask == 0:
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
        oa = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
        tl.store(oa, acc, mask=m_mask[:, None] & n_mask[None, :])
        return

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # --- TILE_DENSEISH: flat channel iteration, pre-computed spatial base ---
    if ag_mask == ALL_ONES_MASK:
        x_spatial = n_idx * C_IN * HW_IN + out_h * W_IN + out_w  # [BLOCK_M]
        for k_idx in range(NUM_GROUPS):
            k_start = k_idx * GROUP_SIZE_C
            offs_k = k_start + tl.arange(0, GROUP_SIZE_C)
            k_mask = offs_k < C_IN
            x_addrs = x_ptr + x_spatial[:, None] + offs_k[None, :] * HW_IN
            x_tile = tl.load(x_addrs, mask=k_mask[None, :] & m_mask[:, None], other=0.0).to(tl.float16)
            w_addrs = w_cl_ptr + offs_n[None, :] * C_IN + offs_k[:, None]
            w_tile = tl.load(w_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
            acc += tl.dot(x_tile, w_tile)
    else:
        # --- TILE_SPARSE: bitmask-gated ---
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
    ALL_ONES_MASK: tl.constexpr,
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

    if ag_mask == ALL_ONES_MASK:
        x_spatial = n_idx * C_IN * HW_IN + out_h * W_IN + out_w
        for k_idx in range(NUM_GROUPS):
            k_start = k_idx * GROUP_SIZE_C
            offs_k = k_start + tl.arange(0, GROUP_SIZE_C)
            k_mask = offs_k < C_IN
            x_addrs = x_ptr + x_spatial[:, None] + offs_k[None, :] * HW_IN
            x_tile = tl.load(x_addrs, mask=k_mask[None, :] & m_mask[:, None], other=0.0).to(tl.float16)
            w_addrs = w_cl_ptr + offs_n[None, :] * C_IN + offs_k[:, None]
            w_tile = tl.load(w_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
            acc += tl.dot(x_tile, w_tile)
    else:
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
    ALL_ONES_MASK: tl.constexpr,
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

    # --- TILE_ZERO ---
    if ag_mask == 0:
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
        oa = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
        tl.store(oa, acc, mask=m_mask[:, None] & n_mask[None, :])
        return

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # --- TILE_DENSEISH: spatial-outer, channel-inner ---
    if ag_mask == ALL_ONES_MASK:
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                in_h = out_h + (kh - 1)
                in_w = out_w + (kw - 1)
                h_ok = (in_h >= 0) & (in_h < H_IN)
                w_ok = (in_w >= 0) & (in_w < W_IN)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H_IN - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W_IN - 1)
                sp_ok = m_mask & h_ok & w_ok
                x_hw = safe_h * W_IN + safe_w
                w_kpos = kh * W_KH + kw * W_CS
                for k_idx in range(NUM_GROUPS):
                    k_start = k_idx * GROUP_SIZE_C
                    offs_k = k_start + tl.arange(0, GROUP_SIZE_C)
                    k_mask = offs_k < C_IN
                    x_addrs = x_ptr + (n_idx * C_IN + offs_k[None, :]) * HW_IN + x_hw[:, None]
                    x_tile = tl.load(x_addrs, mask=k_mask[None, :] & sp_ok[:, None], other=0.0).to(tl.float16)
                    w_addrs = w_cl_ptr + offs_n[None, :] * W_CO + w_kpos + offs_k[:, None]
                    w_tile = tl.load(w_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
                    acc += tl.dot(x_tile, w_tile)
    else:
        # --- TILE_SPARSE: group-outer, bitmask-gated ---
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
    ALL_ONES_MASK: tl.constexpr,
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

    if ag_mask == ALL_ONES_MASK:
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                in_h = out_h + (kh - 1)
                in_w = out_w + (kw - 1)
                h_ok = (in_h >= 0) & (in_h < H_IN)
                w_ok = (in_w >= 0) & (in_w < W_IN)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H_IN - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W_IN - 1)
                sp_ok = m_mask & h_ok & w_ok
                x_hw = safe_h * W_IN + safe_w
                w_kpos = kh * W_KH + kw * W_CS
                for k_idx in range(NUM_GROUPS):
                    k_start = k_idx * GROUP_SIZE_C
                    offs_k = k_start + tl.arange(0, GROUP_SIZE_C)
                    k_mask = offs_k < C_IN
                    x_addrs = x_ptr + (n_idx * C_IN + offs_k[None, :]) * HW_IN + x_hw[:, None]
                    x_tile = tl.load(x_addrs, mask=k_mask[None, :] & sp_ok[:, None], other=0.0).to(tl.float16)
                    w_addrs = w_cl_ptr + offs_n[None, :] * W_CO + w_kpos + offs_k[:, None]
                    w_tile = tl.load(w_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
                    acc += tl.dot(x_tile, w_tile)
    else:
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
    ALL_ONES_MASK: tl.constexpr,
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

    if ag_mask == ALL_ONES_MASK:
        # Dense-ish: spatial-outer, channel-inner (stride=2)
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                in_h = out_h * 2 + (kh - 1)
                in_w = out_w * 2 + (kw - 1)
                h_ok = (in_h >= 0) & (in_h < H_IN)
                w_ok = (in_w >= 0) & (in_w < W_IN)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H_IN - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W_IN - 1)
                sp_ok = m_mask & h_ok & w_ok
                x_hw = safe_h * W_IN + safe_w
                w_kpos = kh * W_KH + kw * W_CS
                for k_idx in range(NUM_GROUPS):
                    k_start = k_idx * GROUP_SIZE_C
                    offs_k = k_start + tl.arange(0, GROUP_SIZE_C)
                    k_mask = offs_k < C_IN
                    x_addrs = x_ptr + (n_idx * C_IN + offs_k[None, :]) * HW_IN + x_hw[:, None]
                    x_tile = tl.load(x_addrs, mask=k_mask[None, :] & sp_ok[:, None], other=0.0).to(tl.float16)
                    w_addrs = w_cl_ptr + offs_n[None, :] * W_CO + w_kpos + offs_k[:, None]
                    w_tile = tl.load(w_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
                    acc += tl.dot(x_tile, w_tile)
    else:
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
    ALL_ONES_MASK: tl.constexpr,
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

    if ag_mask == ALL_ONES_MASK:
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                in_h = out_h * 2 + (kh - 1)
                in_w = out_w * 2 + (kw - 1)
                h_ok = (in_h >= 0) & (in_h < H_IN)
                w_ok = (in_w >= 0) & (in_w < W_IN)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H_IN - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W_IN - 1)
                sp_ok = m_mask & h_ok & w_ok
                x_hw = safe_h * W_IN + safe_w
                w_kpos = kh * W_KH + kw * W_CS
                for k_idx in range(NUM_GROUPS):
                    k_start = k_idx * GROUP_SIZE_C
                    offs_k = k_start + tl.arange(0, GROUP_SIZE_C)
                    k_mask = offs_k < C_IN
                    x_addrs = x_ptr + (n_idx * C_IN + offs_k[None, :]) * HW_IN + x_hw[:, None]
                    x_tile = tl.load(x_addrs, mask=k_mask[None, :] & sp_ok[:, None], other=0.0).to(tl.float16)
                    w_addrs = w_cl_ptr + offs_n[None, :] * W_CO + w_kpos + offs_k[:, None]
                    w_tile = tl.load(w_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
                    acc += tl.dot(x_tile, w_tile)
    else:
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
    # Legacy params (kept for compat)
    counts_buf=None,
    tile_cin_buf=None,
    group_flags_buf=None,
    ag_count_buf=None,
    ag_list_buf=None,
    ag_mask_buf=None,
    # Two-stage buffers
    tile_alive_buf=None,
    tile_class_buf=None,
    return_ms=False,
    fallback_ratio=FALLBACK_RATIO,
    return_avg_active_ratio=False,
    return_tile_stats=False,
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

    if groups != 1 or dilation != 1:
        y = Fn.conv2d(x, weight, bias, stride=stride, padding=padding,
                      dilation=dilation, groups=groups).float()
        ret = (y, 0.0)
        if return_avg_active_ratio:
            ret = ret + (1.0,)
        if return_tile_stats:
            ret = ret + (None,)
        return ret

    H_OUT = (H_IN + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    W_OUT = (W_IN + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    if H_OUT <= 0 or W_OUT <= 0:
        y = Fn.conv2d(x, weight, bias, stride=stride, padding=padding,
                      dilation=dilation, groups=groups).float()
        ret = (y, 0.0)
        if return_avg_active_ratio:
            ret = ret + (1.0,)
        if return_tile_stats:
            ret = ret + (None,)
        return ret

    supported = (
        (kernel_size == 1 and stride == 1 and padding == 0) or
        (kernel_size == 1 and stride == 2 and padding == 0) or
        (kernel_size == 3 and stride == 1 and padding == 1) or
        (kernel_size == 3 and stride == 2 and padding == 1)
    )
    if not supported:
        y = Fn.conv2d(x, weight, bias, stride=stride, padding=padding,
                      dilation=dilation, groups=groups).float()
        ret = (y, 0.0)
        if return_avg_active_ratio:
            ret = ret + (1.0,)
        if return_tile_stats:
            ret = ret + (None,)
        return ret

    # 1x1/s2 → subsample then 1x1/s1
    actual_kernel_1x1s2 = (kernel_size == 1 and stride == 2)
    if actual_kernel_1x1s2:
        x = x[:, :, ::2, ::2].contiguous()
        N, C_IN, H_IN, W_IN = x.shape
        stride = 1
        H_OUT = H_IN
        W_OUT = W_IN

    GROUP_SIZE_C = choose_group_size(C_IN)
    NUM_GROUPS = triton.cdiv(C_IN, GROUP_SIZE_C)
    ALL_ONES_MASK = (1 << NUM_GROUPS) - 1

    BH, BW = _select_tile_sizes(H_OUT, W_OUT)
    GH = triton.cdiv(H_OUT, BH)
    GW = triton.cdiv(W_OUT, BW)
    N_TILES = N * GH * GW

    if w_cl is not None:
        w_cl_f16 = w_cl
    else:
        if kernel_size == 3:
            w_cl_f16 = weight.half().permute(0, 2, 3, 1).contiguous()
        else:
            w_cl_f16 = weight.half().reshape(C_OUT, C_IN).contiguous()

    x_f16 = x.half().contiguous()

    # Allocate buffers
    if tile_alive_buf is None or tile_alive_buf.numel() < N_TILES:
        tile_alive_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
    if ag_mask_buf is None or ag_mask_buf.numel() < N_TILES:
        ag_mask_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
    if tile_class_buf is None or tile_class_buf.numel() < N_TILES:
        tile_class_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)

    # Two-stage prescan
    _build_two_stage_metadata(
        x_f16, N, C_IN, H_IN, W_IN, H_OUT, W_OUT,
        BH, BW, GH, GW,
        kernel_size, stride, padding,
        threshold, tile_alive_buf, ag_mask_buf, tile_class_buf,
    )

    # Compute tile stats and AGR
    tile_stats = None
    avg_active_ratio = None

    if return_avg_active_ratio or return_tile_stats:
        tc = tile_class_buf[:N_TILES]
        zero_count = int((tc == TILE_ZERO).sum().item())
        sparse_count = int((tc == TILE_SPARSE).sum().item())
        denseish_count = int((tc == TILE_DENSEISH).sum().item())

        if return_tile_stats:
            tile_stats = {
                'stage1_zero_tiles': 0,  # Stage 1 no longer classifies zeros
                'stage1_denseish_tiles': -1,  # not tracked without extra sync
                'stage2_tiles': zero_count + sparse_count,  # lower bound
                'zero_tiles': zero_count,
                'sparse_tiles': sparse_count,
                'denseish_tiles': denseish_count,
                'total_tiles': N_TILES,
                'prescan_mode': 'two_stage_v21',
            }

        if NUM_GROUPS > 0:
            pc = _popcount_buf(ag_mask_buf, N_TILES)
            avg_active_ratio = pc.float().mean().item() / max(NUM_GROUPS, 1)
        else:
            avg_active_ratio = 0.0

        if avg_active_ratio == 0.0:
            y = torch.zeros(N, C_OUT, H_OUT, W_OUT, dtype=torch.float32, device=device)
            if bias is not None:
                y += bias.float().view(1, -1, 1, 1)
            ret = (y, 0.0)
            if return_avg_active_ratio:
                ret = ret + (avg_active_ratio,)
            if return_tile_stats:
                ret = ret + (tile_stats,)
            return ret

        if avg_active_ratio > fallback_ratio:
            y = Fn.conv2d(x, weight, bias, stride=stride, padding=padding,
                          dilation=dilation, groups=groups).float()
            ret = (y, 0.0)
            if return_avg_active_ratio:
                ret = ret + (avg_active_ratio,)
            if return_tile_stats:
                ret = ret + (tile_stats,)
            return ret

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
        ALL_ONES_MASK=ALL_ONES_MASK,
    )

    if return_ms:
        ee.record()
        torch.cuda.synchronize(device)
        sparse_ms = se.elapsed_time(ee)

    ret = (y, sparse_ms)
    if return_avg_active_ratio:
        if avg_active_ratio is None:
            pc = _popcount_buf(ag_mask_buf, N_TILES)
            avg_active_ratio = pc.float().mean().item() / max(NUM_GROUPS, 1)
        ret = ret + (avg_active_ratio,)
    if return_tile_stats:
        ret = ret + (tile_stats,)
    return ret