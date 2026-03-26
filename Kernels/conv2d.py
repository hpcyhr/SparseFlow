"""
SparseFlow Conv2d Triton Kernels — v24.0

Changes from v23:
  A. Zero-candidate refine: skips c_off=0 per group (already confirmed zero
     by Stage 1 representative channel check). Saves NUM_GROUPS × RF_SIZE
     loads for every zero-candidate tile. Group-level early exit retained.
  B. Dense-ish path uses wider DENSE_K = 2*GROUP_SIZE_C for tl.dot,
     halving K-loop iterations. Autotune adds num_stages=2 and BLOCK_N=32.
  C. sparse_conv2d_forward accepts pre-converted fp16 input to avoid
     redundant .half() when called from the wrapper.

Supported patterns: 1x1/s1/p0, 1x1/s2/p0, 3x3/s1/p1, 3x3/s2/p1
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
TILE_ZERO_CANDIDATE = 4

FALLBACK_RATIO = 0.85

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _select_tile_sizes(H, W):
    pixels = H * W
    if pixels >= 3136:
        return 8, 16
    return 8, 8


def _select_block_sizes(H, W, C_IN, C_OUT, kernel_size, N):
    BH, BW = _select_tile_sizes(H, W)
    gs = choose_group_size(C_IN)
    return BH, BW, BH * BW, 64, gs


def _popcount_buf(ag_mask_buf, N_TILES):
    v = ag_mask_buf[:N_TILES].int()
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F
    v = v + (v >> 8)
    v = v + (v >> 16)
    return (v & 0x3F).to(torch.int32)


def _check_dense_fallback(ag_mask_buf, N_TILES, NUM_GROUPS, fallback_ratio=FALLBACK_RATIO):
    if NUM_GROUPS == 0:
        return False
    pc = _popcount_buf(ag_mask_buf, N_TILES)
    avg_active = pc.float().mean().item()
    return avg_active > fallback_ratio * NUM_GROUPS


def _build_active_tile_ids(tile_class_buf, N_TILES):
    tc = tile_class_buf[:N_TILES]
    active = torch.nonzero(tc != TILE_ZERO, as_tuple=False).flatten()
    if active.numel() == 0:
        return active.to(dtype=torch.int32), 0
    return active.to(dtype=torch.int32).contiguous(), int(active.numel())


# ===========================================================================
# STAGE 1: Coarse 3-way classification (NCHW)
# ===========================================================================

@triton.jit
def tile_coarse_classify_kernel(
    x_ptr, tile_class_ptr, ag_mask_ptr,
    N_val, C_IN, H_IN, W_IN, GH, GW,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    KERNEL_SIZE: tl.constexpr, STRIDE: tl.constexpr, PAD: tl.constexpr,
    RF_SIZE: tl.constexpr, THRESHOLD: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, NUM_GROUPS: tl.constexpr,
    ALL_ONES_MASK: tl.constexpr,
    UNCERTAIN_CLASS: tl.constexpr, ZERO_CANDIDATE_CLASS: tl.constexpr,
):
    tile_id = tl.program_id(0)
    if tile_id >= N_val * GH * GW:
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
    hh = rf_h_start + flat_idx // RF_W
    ww = rf_w_start + flat_idx % RF_W
    hw_mask = (flat_idx < RF_H * RF_W) & (hh >= 0) & (hh < H_IN) & (ww >= 0) & (ww < W_IN)
    safe_h = tl.minimum(tl.maximum(hh, 0), H_IN - 1)
    safe_w = tl.minimum(tl.maximum(ww, 0), W_IN - 1)
    HW = H_IN * W_IN
    off1 = tl.arange(0, 1)

    rough_mask = tl.zeros([1], dtype=tl.int32)
    for g in range(NUM_GROUPS):
        c_rep = g * GROUP_SIZE_C
        if c_rep < C_IN:
            vals = tl.load(x_ptr + (n_idx * C_IN + c_rep) * HW + safe_h * W_IN + safe_w, mask=hw_mask, other=0.0)
            is_active = (tl.max(vals, axis=0) > THRESHOLD).to(tl.int32)
            rough_mask = rough_mask + is_active * (1 << g)

    if tl.sum(rough_mask == ALL_ONES_MASK) > 0:
        tl.store(tile_class_ptr + tile_id + off1, tl.full([1], 2, dtype=tl.int32))
        tl.store(ag_mask_ptr + tile_id + off1, tl.full([1], ALL_ONES_MASK, dtype=tl.int32))
    else:
        if tl.sum(rough_mask) == 0:
            tl.store(tile_class_ptr + tile_id + off1, tl.full([1], ZERO_CANDIDATE_CLASS, dtype=tl.int32))
            tl.store(ag_mask_ptr + tile_id + off1, tl.zeros([1], dtype=tl.int32))
        else:
            tl.store(tile_class_ptr + tile_id + off1, tl.full([1], UNCERTAIN_CLASS, dtype=tl.int32))
            tl.store(ag_mask_ptr + tile_id + off1, rough_mask)


@triton.jit
def tile_coarse_classify_1x1_kernel(
    x_ptr, tile_class_ptr, ag_mask_ptr,
    N_val, C_IN, H_IN, W_IN, GH, GW,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_M: tl.constexpr,
    THRESHOLD: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, NUM_GROUPS: tl.constexpr,
    ALL_ONES_MASK: tl.constexpr,
    UNCERTAIN_CLASS: tl.constexpr, ZERO_CANDIDATE_CLASS: tl.constexpr,
):
    tile_id = tl.program_id(0)
    if tile_id >= N_val * GH * GW:
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

    rough_mask = tl.zeros([1], dtype=tl.int32)
    for g in range(NUM_GROUPS):
        c_rep = g * GROUP_SIZE_C
        if c_rep < C_IN:
            vals = tl.load(x_ptr + (n_idx * C_IN + c_rep) * HW + out_h * W_IN + out_w, mask=m_mask, other=0.0)
            is_active = (tl.max(vals, axis=0) > THRESHOLD).to(tl.int32)
            rough_mask = rough_mask + is_active * (1 << g)

    if tl.sum(rough_mask == ALL_ONES_MASK) > 0:
        tl.store(tile_class_ptr + tile_id + off1, tl.full([1], 2, dtype=tl.int32))
        tl.store(ag_mask_ptr + tile_id + off1, tl.full([1], ALL_ONES_MASK, dtype=tl.int32))
    else:
        if tl.sum(rough_mask) == 0:
            tl.store(tile_class_ptr + tile_id + off1, tl.full([1], ZERO_CANDIDATE_CLASS, dtype=tl.int32))
            tl.store(ag_mask_ptr + tile_id + off1, tl.zeros([1], dtype=tl.int32))
        else:
            tl.store(tile_class_ptr + tile_id + off1, tl.full([1], UNCERTAIN_CLASS, dtype=tl.int32))
            tl.store(ag_mask_ptr + tile_id + off1, rough_mask)


# ===========================================================================
# STAGE 2a: Zero-candidate refinement [Priority A — skip rep channel]
# ===========================================================================
# Stage 1 already confirmed c_off=0 (representative) is zero for every group.
# We only need to scan c_off=1..GROUP_SIZE_C-1 to confirm full-zero or find
# a false positive. This saves NUM_GROUPS × RF_SIZE loads per tile.

@triton.jit
def zero_candidate_refine_kernel(
    x_ptr, tile_class_ptr, ag_mask_ptr,
    N_val, C_IN, H_IN, W_IN, H_OUT, W_OUT, GH, GW,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    KERNEL_SIZE: tl.constexpr, STRIDE: tl.constexpr, PAD: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, NUM_GROUPS: tl.constexpr,
    RF_SIZE: tl.constexpr, THRESHOLD: tl.constexpr,
    ALL_ONES_MASK: tl.constexpr, ZERO_CANDIDATE_CLASS: tl.constexpr,
):
    tile_id = tl.program_id(0)
    if tile_id >= N_val * GH * GW:
        return
    off1 = tl.arange(0, 1)
    tc = tl.load(tile_class_ptr + tile_id + off1)
    if tl.sum(tc) != ZERO_CANDIDATE_CLASS:
        return

    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH
    RF_H: tl.constexpr = (BLOCK_H - 1) * STRIDE + KERNEL_SIZE
    RF_W: tl.constexpr = (BLOCK_W - 1) * STRIDE + KERNEL_SIZE
    flat_idx = tl.arange(0, RF_SIZE)
    hh = gh_idx * BLOCK_H * STRIDE - PAD + flat_idx // RF_W
    ww = gw_idx * BLOCK_W * STRIDE - PAD + flat_idx % RF_W
    hw_mask = (flat_idx < RF_H * RF_W) & (hh >= 0) & (hh < H_IN) & (ww >= 0) & (ww < W_IN)
    safe_h = tl.minimum(tl.maximum(hh, 0), H_IN - 1)
    safe_w = tl.minimum(tl.maximum(ww, 0), W_IN - 1)
    HW = H_IN * W_IN

    # [Priority A] Group-level scan, skip c_off=0 (already confirmed zero)
    mask = tl.zeros([1], dtype=tl.int32)
    found_nz = tl.zeros([1], dtype=tl.int32)
    g_idx = tl.zeros([1], dtype=tl.int32)

    while (tl.sum(g_idx) < NUM_GROUPS) & (tl.sum(found_nz) == 0):
        g_val = tl.sum(g_idx)
        group_max = tl.zeros([1], dtype=tl.float32)
        # Start from c_off=1: c_off=0 was the rep channel, already zero
        for c_off in range(1, GROUP_SIZE_C):
            c = g_val * GROUP_SIZE_C + c_off
            if c < C_IN:
                vals = tl.load(x_ptr + (n_idx * C_IN + c) * HW + safe_h * W_IN + safe_w,
                               mask=hw_mask, other=0.0)
                group_max = tl.maximum(group_max, tl.max(vals, axis=0))
        is_active = (group_max > THRESHOLD).to(tl.int32)
        mask = mask + is_active * (1 << g_val)
        found_nz = found_nz + is_active
        g_idx = g_idx + 1

    if tl.sum(found_nz) == 0:
        tl.store(tile_class_ptr + tile_id + off1, tl.zeros([1], dtype=tl.int32))
        tl.store(ag_mask_ptr + tile_id + off1, tl.zeros([1], dtype=tl.int32))
        return

    # False positive: continue remaining groups for full bitmask
    while tl.sum(g_idx) < NUM_GROUPS:
        g_val = tl.sum(g_idx)
        group_max = tl.zeros([1], dtype=tl.float32)
        for c_off in range(1, GROUP_SIZE_C):
            c = g_val * GROUP_SIZE_C + c_off
            if c < C_IN:
                vals = tl.load(x_ptr + (n_idx * C_IN + c) * HW + safe_h * W_IN + safe_w,
                               mask=hw_mask, other=0.0)
                group_max = tl.maximum(group_max, tl.max(vals, axis=0))
        is_active = (group_max > THRESHOLD).to(tl.int32)
        mask = mask + is_active * (1 << g_val)
        if tl.sum(mask == ALL_ONES_MASK) > 0:
            g_idx = tl.full([1], NUM_GROUPS, dtype=tl.int32)
        else:
            g_idx = g_idx + 1

    tl.store(ag_mask_ptr + tile_id + off1, mask)
    if tl.sum(mask == ALL_ONES_MASK) > 0:
        tl.store(tile_class_ptr + tile_id + off1, tl.full([1], 2, dtype=tl.int32))
    else:
        tl.store(tile_class_ptr + tile_id + off1, tl.full([1], 1, dtype=tl.int32))


@triton.jit
def zero_candidate_refine_1x1_kernel(
    x_ptr, tile_class_ptr, ag_mask_ptr,
    N_val, C_IN, H_IN, W_IN, GH, GW,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_M: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, NUM_GROUPS: tl.constexpr,
    THRESHOLD: tl.constexpr, ALL_ONES_MASK: tl.constexpr,
    ZERO_CANDIDATE_CLASS: tl.constexpr,
):
    tile_id = tl.program_id(0)
    if tile_id >= N_val * GH * GW:
        return
    off1 = tl.arange(0, 1)
    tc = tl.load(tile_class_ptr + tile_id + off1)
    if tl.sum(tc) != ZERO_CANDIDATE_CLASS:
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

    mask = tl.zeros([1], dtype=tl.int32)
    found_nz = tl.zeros([1], dtype=tl.int32)
    g_idx = tl.zeros([1], dtype=tl.int32)

    while (tl.sum(g_idx) < NUM_GROUPS) & (tl.sum(found_nz) == 0):
        g_val = tl.sum(g_idx)
        group_max = tl.zeros([1], dtype=tl.float32)
        for c_off in range(1, GROUP_SIZE_C):
            c = g_val * GROUP_SIZE_C + c_off
            if c < C_IN:
                vals = tl.load(x_ptr + (n_idx * C_IN + c) * HW + out_h * W_IN + out_w, mask=m_mask, other=0.0)
                group_max = tl.maximum(group_max, tl.max(vals, axis=0))
        is_active = (group_max > THRESHOLD).to(tl.int32)
        mask = mask + is_active * (1 << g_val)
        found_nz = found_nz + is_active
        g_idx = g_idx + 1

    if tl.sum(found_nz) == 0:
        tl.store(tile_class_ptr + tile_id + off1, tl.zeros([1], dtype=tl.int32))
        tl.store(ag_mask_ptr + tile_id + off1, tl.zeros([1], dtype=tl.int32))
        return

    while tl.sum(g_idx) < NUM_GROUPS:
        g_val = tl.sum(g_idx)
        group_max = tl.zeros([1], dtype=tl.float32)
        for c_off in range(1, GROUP_SIZE_C):
            c = g_val * GROUP_SIZE_C + c_off
            if c < C_IN:
                vals = tl.load(x_ptr + (n_idx * C_IN + c) * HW + out_h * W_IN + out_w, mask=m_mask, other=0.0)
                group_max = tl.maximum(group_max, tl.max(vals, axis=0))
        is_active = (group_max > THRESHOLD).to(tl.int32)
        mask = mask + is_active * (1 << g_val)
        if tl.sum(mask == ALL_ONES_MASK) > 0:
            g_idx = tl.full([1], NUM_GROUPS, dtype=tl.int32)
        else:
            g_idx = g_idx + 1

    tl.store(ag_mask_ptr + tile_id + off1, mask)
    if tl.sum(mask == ALL_ONES_MASK) > 0:
        tl.store(tile_class_ptr + tile_id + off1, tl.full([1], 2, dtype=tl.int32))
    else:
        tl.store(tile_class_ptr + tile_id + off1, tl.full([1], 1, dtype=tl.int32))


# ===========================================================================
# STAGE 2b: Exact bitmask for UNCERTAIN tiles (NCHW)
# ===========================================================================

@triton.jit
def group_bitmask_refine_kernel(
    x_ptr, tile_class_ptr, ag_mask_ptr,
    N_val, C_IN, H_IN, W_IN, H_OUT, W_OUT, GH, GW,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    KERNEL_SIZE: tl.constexpr, STRIDE: tl.constexpr, PAD: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, NUM_GROUPS: tl.constexpr,
    RF_SIZE: tl.constexpr, THRESHOLD: tl.constexpr,
    ALL_ONES_MASK: tl.constexpr, UNCERTAIN_CLASS: tl.constexpr,
):
    tile_id = tl.program_id(0)
    if tile_id >= N_val * GH * GW:
        return
    off1 = tl.arange(0, 1)
    tc = tl.load(tile_class_ptr + tile_id + off1)
    if tl.sum(tc) != UNCERTAIN_CLASS:
        return
    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH
    RF_H: tl.constexpr = (BLOCK_H - 1) * STRIDE + KERNEL_SIZE
    RF_W: tl.constexpr = (BLOCK_W - 1) * STRIDE + KERNEL_SIZE
    flat_idx = tl.arange(0, RF_SIZE)
    hh = gh_idx * BLOCK_H * STRIDE - PAD + flat_idx // RF_W
    ww = gw_idx * BLOCK_W * STRIDE - PAD + flat_idx % RF_W
    hw_mask = (flat_idx < RF_H * RF_W) & (hh >= 0) & (hh < H_IN) & (ww >= 0) & (ww < W_IN)
    safe_h = tl.minimum(tl.maximum(hh, 0), H_IN - 1)
    safe_w = tl.minimum(tl.maximum(ww, 0), W_IN - 1)
    HW = H_IN * W_IN

    mask = tl.zeros([1], dtype=tl.int32)
    g = tl.zeros([1], dtype=tl.int32)
    while tl.sum(g) < NUM_GROUPS:
        g_val = tl.sum(g)
        group_max = tl.zeros([1], dtype=tl.float32)
        for c_off in range(GROUP_SIZE_C):
            c = g_val * GROUP_SIZE_C + c_off
            if c < C_IN:
                vals = tl.load(x_ptr + (n_idx * C_IN + c) * HW + safe_h * W_IN + safe_w,
                               mask=hw_mask, other=0.0)
                group_max = tl.maximum(group_max, tl.max(vals, axis=0))
        mask = mask + (group_max > THRESHOLD).to(tl.int32) * (1 << g_val)
        if tl.sum(mask == ALL_ONES_MASK) > 0:
            g = tl.full([1], NUM_GROUPS, dtype=tl.int32)
        else:
            g = g + 1
    tl.store(ag_mask_ptr + tile_id + off1, mask)
    if tl.sum(mask) == 0:
        tl.store(tile_class_ptr + tile_id + off1, tl.zeros([1], dtype=tl.int32))
    else:
        tl.store(tile_class_ptr + tile_id + off1,
                 tl.full([1], 1, dtype=tl.int32) + (mask == ALL_ONES_MASK).to(tl.int32))


@triton.jit
def group_bitmask_refine_1x1_kernel(
    x_ptr, tile_class_ptr, ag_mask_ptr,
    N_val, C_IN, H_IN, W_IN, GH, GW,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_M: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, NUM_GROUPS: tl.constexpr,
    THRESHOLD: tl.constexpr, ALL_ONES_MASK: tl.constexpr,
    UNCERTAIN_CLASS: tl.constexpr,
):
    tile_id = tl.program_id(0)
    if tile_id >= N_val * GH * GW:
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

    mask = tl.zeros([1], dtype=tl.int32)
    g = tl.zeros([1], dtype=tl.int32)
    while tl.sum(g) < NUM_GROUPS:
        g_val = tl.sum(g)
        group_max = tl.zeros([1], dtype=tl.float32)
        for c_off in range(GROUP_SIZE_C):
            c = g_val * GROUP_SIZE_C + c_off
            if c < C_IN:
                vals = tl.load(x_ptr + (n_idx * C_IN + c) * HW + out_h * W_IN + out_w, mask=m_mask, other=0.0)
                group_max = tl.maximum(group_max, tl.max(vals, axis=0))
        mask = mask + (group_max > THRESHOLD).to(tl.int32) * (1 << g_val)
        if tl.sum(mask == ALL_ONES_MASK) > 0:
            g = tl.full([1], NUM_GROUPS, dtype=tl.int32)
        else:
            g = g + 1
    tl.store(ag_mask_ptr + tile_id + off1, mask)
    if tl.sum(mask) == 0:
        tl.store(tile_class_ptr + tile_id + off1, tl.zeros([1], dtype=tl.int32))
    else:
        tl.store(tile_class_ptr + tile_id + off1,
                 tl.full([1], 1, dtype=tl.int32) + (mask == ALL_ONES_MASK).to(tl.int32))


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------

@torch.no_grad()
def _build_two_stage_metadata(
    x_f16, N, C_IN, H_IN, W_IN, H_OUT, W_OUT,
    BH, BW, GH, GW,
    kernel_size, stride, padding, threshold,
    ag_mask_buf, tile_class_buf,
    prescan_stats=None,
):
    GROUP_SIZE_C = choose_group_size(C_IN)
    N_TILES = N * GH * GW
    NUM_GROUPS = triton.cdiv(C_IN, GROUP_SIZE_C)
    ALL_ONES_MASK = (1 << NUM_GROUPS) - 1

    # Stage 1
    if kernel_size == 1:
        BM = BH * BW
        tile_coarse_classify_1x1_kernel[(N_TILES,)](
            x_f16, tile_class_buf, ag_mask_buf,
            N, C_IN, H_IN, W_IN, GH, GW,
            BLOCK_H=BH, BLOCK_W=BW, BLOCK_M=BM,
            THRESHOLD=threshold, GROUP_SIZE_C=GROUP_SIZE_C,
            NUM_GROUPS=NUM_GROUPS, ALL_ONES_MASK=ALL_ONES_MASK,
            UNCERTAIN_CLASS=TILE_UNCERTAIN, ZERO_CANDIDATE_CLASS=TILE_ZERO_CANDIDATE,
        )
    else:
        rf_h = (BH - 1) * stride + kernel_size
        rf_w = (BW - 1) * stride + kernel_size
        RF_SIZE = triton.next_power_of_2(max(rf_h * rf_w, 1))
        tile_coarse_classify_kernel[(N_TILES,)](
            x_f16, tile_class_buf, ag_mask_buf,
            N, C_IN, H_IN, W_IN, GH, GW,
            BLOCK_H=BH, BLOCK_W=BW,
            KERNEL_SIZE=kernel_size, STRIDE=stride, PAD=padding,
            RF_SIZE=RF_SIZE, THRESHOLD=threshold, GROUP_SIZE_C=GROUP_SIZE_C,
            NUM_GROUPS=NUM_GROUPS, ALL_ONES_MASK=ALL_ONES_MASK,
            UNCERTAIN_CLASS=TILE_UNCERTAIN, ZERO_CANDIDATE_CLASS=TILE_ZERO_CANDIDATE,
        )

    if prescan_stats is not None:
        tc = tile_class_buf[:N_TILES]
        prescan_stats['stage1_zero_candidate'] = int((tc == TILE_ZERO_CANDIDATE).sum().item())
        prescan_stats['stage1_denseish'] = int((tc == TILE_DENSEISH).sum().item())
        prescan_stats['stage1_uncertain'] = int((tc == TILE_UNCERTAIN).sum().item())

    # Stage 2a: zero-candidate refine
    if kernel_size == 1:
        BM = BH * BW
        zero_candidate_refine_1x1_kernel[(N_TILES,)](
            x_f16, tile_class_buf, ag_mask_buf,
            N, C_IN, H_IN, W_IN, GH, GW,
            BLOCK_H=BH, BLOCK_W=BW, BLOCK_M=BM,
            GROUP_SIZE_C=GROUP_SIZE_C, NUM_GROUPS=NUM_GROUPS,
            THRESHOLD=threshold, ALL_ONES_MASK=ALL_ONES_MASK,
            ZERO_CANDIDATE_CLASS=TILE_ZERO_CANDIDATE,
        )
    else:
        rf_h = (BH - 1) * stride + kernel_size
        rf_w = (BW - 1) * stride + kernel_size
        RF_SIZE = triton.next_power_of_2(max(rf_h * rf_w, 1))
        zero_candidate_refine_kernel[(N_TILES,)](
            x_f16, tile_class_buf, ag_mask_buf,
            N, C_IN, H_IN, W_IN, H_OUT, W_OUT, GH, GW,
            BLOCK_H=BH, BLOCK_W=BW,
            KERNEL_SIZE=kernel_size, STRIDE=stride, PAD=padding,
            GROUP_SIZE_C=GROUP_SIZE_C, NUM_GROUPS=NUM_GROUPS,
            RF_SIZE=RF_SIZE, THRESHOLD=threshold,
            ALL_ONES_MASK=ALL_ONES_MASK, ZERO_CANDIDATE_CLASS=TILE_ZERO_CANDIDATE,
        )

    # Stage 2b: uncertain bitmask
    if kernel_size == 1:
        BM = BH * BW
        group_bitmask_refine_1x1_kernel[(N_TILES,)](
            x_f16, tile_class_buf, ag_mask_buf,
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
            x_f16, tile_class_buf, ag_mask_buf,
            N, C_IN, H_IN, W_IN, H_OUT, W_OUT, GH, GW,
            BLOCK_H=BH, BLOCK_W=BW,
            KERNEL_SIZE=kernel_size, STRIDE=stride, PAD=padding,
            GROUP_SIZE_C=GROUP_SIZE_C, NUM_GROUPS=NUM_GROUPS,
            RF_SIZE=RF_SIZE, THRESHOLD=threshold,
            ALL_ONES_MASK=ALL_ONES_MASK, UNCERTAIN_CLASS=TILE_UNCERTAIN,
        )

    if prescan_stats is not None:
        tc = tile_class_buf[:N_TILES]
        prescan_stats['final_zero'] = int((tc == TILE_ZERO).sum().item())
        prescan_stats['final_sparse'] = int((tc == TILE_SPARSE).sum().item())
        prescan_stats['final_denseish'] = int((tc == TILE_DENSEISH).sum().item())
        prescan_stats['stage2_zero_refine_tiles'] = prescan_stats.get('stage1_zero_candidate', 0)
        prescan_stats['stage2_uncertain_tiles'] = prescan_stats.get('stage1_uncertain', 0)

    return GROUP_SIZE_C, NUM_GROUPS


# ---------------------------------------------------------------------------
# [Priority B] Autotune — wider configs with num_stages=2 and BLOCK_N=32
# ---------------------------------------------------------------------------

def _make_configs(block_h, block_w):
    block_m = block_h * block_w
    configs = []
    for bn in [32, 64, 128]:
        for nw in [4, 8]:
            for ns in [1, 2]:
                if bn == 32 and nw == 8:
                    continue
                configs.append(Config(
                    {"BLOCK_M": block_m, "BLOCK_N": bn,
                     "BLOCK_H": block_h, "BLOCK_W": block_w},
                    num_warps=nw, num_stages=ns,
                ))
    return configs

_CONFIGS_8x8 = _make_configs(8, 8)
_CONFIGS_8x16 = _make_configs(8, 16)


# ===================================================================
# [Priority B+C] NHWC Compute Kernels — wider dense-ish K
#
# Dense-ish: uses DENSE_K (= 2×GROUP_SIZE_C) for wider tl.dot,
#   halving K-loop iterations. No bitmask, no group abstraction.
# Sparse: uses GROUP_SIZE_C with bitmask-gated group skip.
# ===================================================================

# ---- 1x1 / s1, 8x8 ----
@autotune(configs=_CONFIGS_8x8, key=["C_IN", "C_OUT", "H_OUT", "W_OUT", "GH", "GW"])
@triton.jit
def sparse_conv1x1_nhwc_kernel_8x8(
    x_nhwc_ptr, w_cl_ptr, bias_ptr, ag_mask_ptr, y_ptr, N_val,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    H_IN: tl.constexpr, W_IN: tl.constexpr,
    H_OUT: tl.constexpr, W_OUT: tl.constexpr,
    GH: tl.constexpr, GW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, NUM_GROUPS: tl.constexpr,
    ALL_ONES_MASK: tl.constexpr,
    DENSE_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    USE_TILE_IDS: tl.constexpr,
):
    pid_tile = tl.program_id(0)
    tile_id = tl.load(tile_ids_ptr + pid_tile) if USE_TILE_IDS else pid_tile
    pid_cout = tl.program_id(1)
    if tile_id >= N_val * GH * GW:
        return
    gw_idx = tile_id % GW; tmp = tile_id // GW
    gh_idx = tmp % GH; n_idx = tmp // GH
    offs_n = pid_cout * BLOCK_N + tl.arange(0, BLOCK_N); n_mask = offs_n < C_OUT
    offs_m = tl.arange(0, BLOCK_M)
    out_h = gh_idx * BLOCK_H + offs_m // BLOCK_W
    out_w = gw_idx * BLOCK_W + offs_m % BLOCK_W
    m_mask = (out_h < H_OUT) & (out_w < W_OUT)
    HW_OUT: tl.constexpr = H_OUT * W_OUT
    WC: tl.constexpr = W_IN * C_IN
    x_base = n_idx * H_IN * WC + out_h * WC + out_w * C_IN
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
        # [Priority B] Dense-ish: wider DENSE_K tiles
        for cin_base in range(0, NUM_GROUPS * GROUP_SIZE_C, DENSE_K):
            offs_k = cin_base + tl.arange(0, DENSE_K)
            k_v = offs_k < C_IN
            x_t = tl.load(x_nhwc_ptr + x_base[:, None] + offs_k[None, :], mask=k_v[None, :] & m_mask[:, None], other=0.0).to(tl.float16)
            w_t = tl.load(w_cl_ptr + offs_n[None, :] * C_IN + offs_k[:, None], mask=k_v[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
            acc += tl.dot(x_t, w_t)
    else:
        for g in range(NUM_GROUPS):
            g_active = (ag_mask >> g) & 1
            if g_active != 0:
                cs = g * GROUP_SIZE_C
                offs_k = cs + tl.arange(0, GROUP_SIZE_C)
                k_m = offs_k < C_IN
                x_t = tl.load(x_nhwc_ptr + x_base[:, None] + offs_k[None, :], mask=k_m[None, :] & m_mask[:, None], other=0.0).to(tl.float16)
                w_t = tl.load(w_cl_ptr + offs_n[None, :] * C_IN + offs_k[:, None], mask=k_m[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
                acc += tl.dot(x_t, w_t)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
    oa = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
    tl.store(oa, acc, mask=m_mask[:, None] & n_mask[None, :])


# ---- 1x1 / s1, 8x16 ----
@autotune(configs=_CONFIGS_8x16, key=["C_IN", "C_OUT", "H_OUT", "W_OUT", "GH", "GW"])
@triton.jit
def sparse_conv1x1_nhwc_kernel_8x16(
    x_nhwc_ptr, w_cl_ptr, bias_ptr, ag_mask_ptr, y_ptr, N_val,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    H_IN: tl.constexpr, W_IN: tl.constexpr,
    H_OUT: tl.constexpr, W_OUT: tl.constexpr,
    GH: tl.constexpr, GW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, NUM_GROUPS: tl.constexpr,
    ALL_ONES_MASK: tl.constexpr,
    DENSE_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    USE_TILE_IDS: tl.constexpr,
):
    pid_tile = tl.program_id(0)
    tile_id = tl.load(tile_ids_ptr + pid_tile) if USE_TILE_IDS else pid_tile
    pid_cout = tl.program_id(1)
    if tile_id >= N_val * GH * GW:
        return
    gw_idx = tile_id % GW; tmp = tile_id // GW
    gh_idx = tmp % GH; n_idx = tmp // GH
    offs_n = pid_cout * BLOCK_N + tl.arange(0, BLOCK_N); n_mask = offs_n < C_OUT
    offs_m = tl.arange(0, BLOCK_M)
    out_h = gh_idx * BLOCK_H + offs_m // BLOCK_W
    out_w = gw_idx * BLOCK_W + offs_m % BLOCK_W
    m_mask = (out_h < H_OUT) & (out_w < W_OUT)
    HW_OUT: tl.constexpr = H_OUT * W_OUT
    WC: tl.constexpr = W_IN * C_IN
    x_base = n_idx * H_IN * WC + out_h * WC + out_w * C_IN
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
        for cin_base in range(0, NUM_GROUPS * GROUP_SIZE_C, DENSE_K):
            offs_k = cin_base + tl.arange(0, DENSE_K)
            k_v = offs_k < C_IN
            x_t = tl.load(x_nhwc_ptr + x_base[:, None] + offs_k[None, :], mask=k_v[None, :] & m_mask[:, None], other=0.0).to(tl.float16)
            w_t = tl.load(w_cl_ptr + offs_n[None, :] * C_IN + offs_k[:, None], mask=k_v[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
            acc += tl.dot(x_t, w_t)
    else:
        for g in range(NUM_GROUPS):
            g_active = (ag_mask >> g) & 1
            if g_active != 0:
                cs = g * GROUP_SIZE_C
                offs_k = cs + tl.arange(0, GROUP_SIZE_C)
                k_m = offs_k < C_IN
                x_t = tl.load(x_nhwc_ptr + x_base[:, None] + offs_k[None, :], mask=k_m[None, :] & m_mask[:, None], other=0.0).to(tl.float16)
                w_t = tl.load(w_cl_ptr + offs_n[None, :] * C_IN + offs_k[:, None], mask=k_m[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
                acc += tl.dot(x_t, w_t)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
    oa = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
    tl.store(oa, acc, mask=m_mask[:, None] & n_mask[None, :])


# ---- helper for 3x3 compute body (avoids 4× copy-paste) ----
# Triton doesn't support function calls across @triton.jit, so we
# use the same inline structure in each variant.

# ---- 3x3 / s1, 8x8 ----
@autotune(configs=_CONFIGS_8x8, key=["C_IN", "C_OUT", "H_OUT", "W_OUT", "GH", "GW"])
@triton.jit
def sparse_conv3x3s1_nhwc_kernel_8x8(
    x_nhwc_ptr, w_cl_ptr, bias_ptr, ag_mask_ptr, tile_ids_ptr, y_ptr, N_val,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    H_IN: tl.constexpr, W_IN: tl.constexpr,
    H_OUT: tl.constexpr, W_OUT: tl.constexpr,
    GH: tl.constexpr, GW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, NUM_GROUPS: tl.constexpr,
    ALL_ONES_MASK: tl.constexpr,
    DENSE_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    USE_TILE_IDS: tl.constexpr,
):
    pid_tile = tl.program_id(0)
    tile_id = tl.load(tile_ids_ptr + pid_tile) if USE_TILE_IDS else pid_tile
    pid_cout = tl.program_id(1)
    if tile_id >= N_val * GH * GW:
        return
    gw_idx = tile_id % GW; tmp = tile_id // GW
    gh_idx = tmp % GH; n_idx = tmp // GH
    offs_n = pid_cout * BLOCK_N + tl.arange(0, BLOCK_N); n_mask = offs_n < C_OUT
    offs_m = tl.arange(0, BLOCK_M)
    out_h = gh_idx * BLOCK_H + offs_m // BLOCK_W
    out_w = gw_idx * BLOCK_W + offs_m % BLOCK_W
    m_mask = (out_h < H_OUT) & (out_w < W_OUT)
    HW_OUT: tl.constexpr = H_OUT * W_OUT
    WC: tl.constexpr = W_IN * C_IN
    W_CS: tl.constexpr = C_IN; W_KH: tl.constexpr = 3 * C_IN; W_CO: tl.constexpr = 9 * C_IN
    n_base = n_idx * H_IN * WC
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
        # [Priority B] Dense-ish: spatial-outer, wider DENSE_K flat channel
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                in_h = out_h + (kh - 1); in_w = out_w + (kw - 1)
                h_ok = (in_h >= 0) & (in_h < H_IN); w_ok = (in_w >= 0) & (in_w < W_IN)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H_IN - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W_IN - 1)
                hw_ok = m_mask & h_ok & w_ok
                x_hw = n_base + safe_h * WC + safe_w * C_IN
                w_off = kh * W_KH + kw * W_CS
                for cin_base in range(0, NUM_GROUPS * GROUP_SIZE_C, DENSE_K):
                    offs_k = cin_base + tl.arange(0, DENSE_K)
                    k_v = offs_k < C_IN
                    x_t = tl.load(x_nhwc_ptr + x_hw[:, None] + offs_k[None, :], mask=k_v[None, :] & hw_ok[:, None], other=0.0).to(tl.float16)
                    w_t = tl.load(w_cl_ptr + offs_n[None, :] * W_CO + w_off + offs_k[:, None], mask=k_v[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
                    acc += tl.dot(x_t, w_t)
    else:
        # Sparse: spatial-outer, bitmask-gated GROUP_SIZE_C
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                in_h = out_h + (kh - 1); in_w = out_w + (kw - 1)
                h_ok = (in_h >= 0) & (in_h < H_IN); w_ok = (in_w >= 0) & (in_w < W_IN)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H_IN - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W_IN - 1)
                hw_ok = m_mask & h_ok & w_ok
                x_hw = n_base + safe_h * WC + safe_w * C_IN
                w_off = kh * W_KH + kw * W_CS
                for g in range(NUM_GROUPS):
                    g_active = (ag_mask >> g) & 1
                    if g_active != 0:
                        cs = g * GROUP_SIZE_C
                        offs_k = cs + tl.arange(0, GROUP_SIZE_C)
                        k_m = offs_k < C_IN
                        x_t = tl.load(x_nhwc_ptr + x_hw[:, None] + offs_k[None, :], mask=k_m[None, :] & hw_ok[:, None], other=0.0).to(tl.float16)
                        w_t = tl.load(w_cl_ptr + offs_n[None, :] * W_CO + w_off + offs_k[:, None], mask=k_m[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
                        acc += tl.dot(x_t, w_t)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
    oa = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
    tl.store(oa, acc, mask=m_mask[:, None] & n_mask[None, :])


# ---- 3x3 / s1, 8x16 ----
@autotune(configs=_CONFIGS_8x16, key=["C_IN", "C_OUT", "H_OUT", "W_OUT", "GH", "GW"])
@triton.jit
def sparse_conv3x3s1_nhwc_kernel_8x16(
    x_nhwc_ptr, w_cl_ptr, bias_ptr, ag_mask_ptr, tile_ids_ptr, y_ptr, N_val,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    H_IN: tl.constexpr, W_IN: tl.constexpr,
    H_OUT: tl.constexpr, W_OUT: tl.constexpr,
    GH: tl.constexpr, GW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, NUM_GROUPS: tl.constexpr,
    ALL_ONES_MASK: tl.constexpr,
    DENSE_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    USE_TILE_IDS: tl.constexpr,
):
    pid_tile = tl.program_id(0)
    tile_id = tl.load(tile_ids_ptr + pid_tile) if USE_TILE_IDS else pid_tile
    pid_cout = tl.program_id(1)
    if tile_id >= N_val * GH * GW:
        return
    gw_idx = tile_id % GW; tmp = tile_id // GW
    gh_idx = tmp % GH; n_idx = tmp // GH
    offs_n = pid_cout * BLOCK_N + tl.arange(0, BLOCK_N); n_mask = offs_n < C_OUT
    offs_m = tl.arange(0, BLOCK_M)
    out_h = gh_idx * BLOCK_H + offs_m // BLOCK_W
    out_w = gw_idx * BLOCK_W + offs_m % BLOCK_W
    m_mask = (out_h < H_OUT) & (out_w < W_OUT)
    HW_OUT: tl.constexpr = H_OUT * W_OUT
    WC: tl.constexpr = W_IN * C_IN
    W_CS: tl.constexpr = C_IN; W_KH: tl.constexpr = 3 * C_IN; W_CO: tl.constexpr = 9 * C_IN
    n_base = n_idx * H_IN * WC
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
                in_h = out_h + (kh - 1); in_w = out_w + (kw - 1)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H_IN - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W_IN - 1)
                hw_ok = m_mask & (in_h >= 0) & (in_h < H_IN) & (in_w >= 0) & (in_w < W_IN)
                x_hw = n_base + safe_h * WC + safe_w * C_IN
                w_off = kh * W_KH + kw * W_CS
                for cin_base in range(0, NUM_GROUPS * GROUP_SIZE_C, DENSE_K):
                    offs_k = cin_base + tl.arange(0, DENSE_K); k_v = offs_k < C_IN
                    x_t = tl.load(x_nhwc_ptr + x_hw[:, None] + offs_k[None, :], mask=k_v[None, :] & hw_ok[:, None], other=0.0).to(tl.float16)
                    w_t = tl.load(w_cl_ptr + offs_n[None, :] * W_CO + w_off + offs_k[:, None], mask=k_v[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
                    acc += tl.dot(x_t, w_t)
    else:
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                in_h = out_h + (kh - 1); in_w = out_w + (kw - 1)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H_IN - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W_IN - 1)
                hw_ok = m_mask & (in_h >= 0) & (in_h < H_IN) & (in_w >= 0) & (in_w < W_IN)
                x_hw = n_base + safe_h * WC + safe_w * C_IN
                w_off = kh * W_KH + kw * W_CS
                for g in range(NUM_GROUPS):
                    g_active = (ag_mask >> g) & 1
                    if g_active != 0:
                        cs = g * GROUP_SIZE_C; offs_k = cs + tl.arange(0, GROUP_SIZE_C); k_m = offs_k < C_IN
                        x_t = tl.load(x_nhwc_ptr + x_hw[:, None] + offs_k[None, :], mask=k_m[None, :] & hw_ok[:, None], other=0.0).to(tl.float16)
                        w_t = tl.load(w_cl_ptr + offs_n[None, :] * W_CO + w_off + offs_k[:, None], mask=k_m[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
                        acc += tl.dot(x_t, w_t)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
    oa = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
    tl.store(oa, acc, mask=m_mask[:, None] & n_mask[None, :])


# ---- 3x3 / s2, 8x8 ----
@autotune(configs=_CONFIGS_8x8, key=["C_IN", "C_OUT", "H_OUT", "W_OUT", "GH", "GW"])
@triton.jit
def sparse_conv3x3s2_nhwc_kernel_8x8(
    x_nhwc_ptr, w_cl_ptr, bias_ptr, ag_mask_ptr, tile_ids_ptr, y_ptr, N_val,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    H_IN: tl.constexpr, W_IN: tl.constexpr,
    H_OUT: tl.constexpr, W_OUT: tl.constexpr,
    GH: tl.constexpr, GW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, NUM_GROUPS: tl.constexpr,
    ALL_ONES_MASK: tl.constexpr,
    DENSE_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    tile_id = tl.program_id(0)
    pid_cout = tl.program_id(1)
    if tile_id >= N_val * GH * GW:
        return
    gw_idx = tile_id % GW; tmp = tile_id // GW
    gh_idx = tmp % GH; n_idx = tmp // GH
    offs_n = pid_cout * BLOCK_N + tl.arange(0, BLOCK_N); n_mask = offs_n < C_OUT
    offs_m = tl.arange(0, BLOCK_M)
    out_h = gh_idx * BLOCK_H + offs_m // BLOCK_W
    out_w = gw_idx * BLOCK_W + offs_m % BLOCK_W
    m_mask = (out_h < H_OUT) & (out_w < W_OUT)
    HW_OUT: tl.constexpr = H_OUT * W_OUT
    WC: tl.constexpr = W_IN * C_IN
    W_CS: tl.constexpr = C_IN; W_KH: tl.constexpr = 3 * C_IN; W_CO: tl.constexpr = 9 * C_IN
    n_base = n_idx * H_IN * WC
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
                in_h = out_h * 2 + (kh - 1); in_w = out_w * 2 + (kw - 1)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H_IN - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W_IN - 1)
                hw_ok = m_mask & (in_h >= 0) & (in_h < H_IN) & (in_w >= 0) & (in_w < W_IN)
                x_hw = n_base + safe_h * WC + safe_w * C_IN
                w_off = kh * W_KH + kw * W_CS
                for cin_base in range(0, NUM_GROUPS * GROUP_SIZE_C, DENSE_K):
                    offs_k = cin_base + tl.arange(0, DENSE_K); k_v = offs_k < C_IN
                    x_t = tl.load(x_nhwc_ptr + x_hw[:, None] + offs_k[None, :], mask=k_v[None, :] & hw_ok[:, None], other=0.0).to(tl.float16)
                    w_t = tl.load(w_cl_ptr + offs_n[None, :] * W_CO + w_off + offs_k[:, None], mask=k_v[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
                    acc += tl.dot(x_t, w_t)
    else:
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                in_h = out_h * 2 + (kh - 1); in_w = out_w * 2 + (kw - 1)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H_IN - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W_IN - 1)
                hw_ok = m_mask & (in_h >= 0) & (in_h < H_IN) & (in_w >= 0) & (in_w < W_IN)
                x_hw = n_base + safe_h * WC + safe_w * C_IN
                w_off = kh * W_KH + kw * W_CS
                for g in range(NUM_GROUPS):
                    g_active = (ag_mask >> g) & 1
                    if g_active != 0:
                        cs = g * GROUP_SIZE_C; offs_k = cs + tl.arange(0, GROUP_SIZE_C); k_m = offs_k < C_IN
                        x_t = tl.load(x_nhwc_ptr + x_hw[:, None] + offs_k[None, :], mask=k_m[None, :] & hw_ok[:, None], other=0.0).to(tl.float16)
                        w_t = tl.load(w_cl_ptr + offs_n[None, :] * W_CO + w_off + offs_k[:, None], mask=k_m[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
                        acc += tl.dot(x_t, w_t)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
    oa = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
    tl.store(oa, acc, mask=m_mask[:, None] & n_mask[None, :])


# ---- 3x3 / s2, 8x16 ----
@autotune(configs=_CONFIGS_8x16, key=["C_IN", "C_OUT", "H_OUT", "W_OUT", "GH", "GW"])
@triton.jit
def sparse_conv3x3s2_nhwc_kernel_8x16(
    x_nhwc_ptr, w_cl_ptr, bias_ptr, ag_mask_ptr, tile_ids_ptr, y_ptr, N_val,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    H_IN: tl.constexpr, W_IN: tl.constexpr,
    H_OUT: tl.constexpr, W_OUT: tl.constexpr,
    GH: tl.constexpr, GW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, NUM_GROUPS: tl.constexpr,
    ALL_ONES_MASK: tl.constexpr,
    DENSE_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    tile_id = tl.program_id(0)
    pid_cout = tl.program_id(1)
    if tile_id >= N_val * GH * GW:
        return
    gw_idx = tile_id % GW; tmp = tile_id // GW
    gh_idx = tmp % GH; n_idx = tmp // GH
    offs_n = pid_cout * BLOCK_N + tl.arange(0, BLOCK_N); n_mask = offs_n < C_OUT
    offs_m = tl.arange(0, BLOCK_M)
    out_h = gh_idx * BLOCK_H + offs_m // BLOCK_W
    out_w = gw_idx * BLOCK_W + offs_m % BLOCK_W
    m_mask = (out_h < H_OUT) & (out_w < W_OUT)
    HW_OUT: tl.constexpr = H_OUT * W_OUT
    WC: tl.constexpr = W_IN * C_IN
    W_CS: tl.constexpr = C_IN; W_KH: tl.constexpr = 3 * C_IN; W_CO: tl.constexpr = 9 * C_IN
    n_base = n_idx * H_IN * WC
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
                in_h = out_h * 2 + (kh - 1); in_w = out_w * 2 + (kw - 1)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H_IN - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W_IN - 1)
                hw_ok = m_mask & (in_h >= 0) & (in_h < H_IN) & (in_w >= 0) & (in_w < W_IN)
                x_hw = n_base + safe_h * WC + safe_w * C_IN
                w_off = kh * W_KH + kw * W_CS
                for cin_base in range(0, NUM_GROUPS * GROUP_SIZE_C, DENSE_K):
                    offs_k = cin_base + tl.arange(0, DENSE_K); k_v = offs_k < C_IN
                    x_t = tl.load(x_nhwc_ptr + x_hw[:, None] + offs_k[None, :], mask=k_v[None, :] & hw_ok[:, None], other=0.0).to(tl.float16)
                    w_t = tl.load(w_cl_ptr + offs_n[None, :] * W_CO + w_off + offs_k[:, None], mask=k_v[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
                    acc += tl.dot(x_t, w_t)
    else:
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                in_h = out_h * 2 + (kh - 1); in_w = out_w * 2 + (kw - 1)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H_IN - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W_IN - 1)
                hw_ok = m_mask & (in_h >= 0) & (in_h < H_IN) & (in_w >= 0) & (in_w < W_IN)
                x_hw = n_base + safe_h * WC + safe_w * C_IN
                w_off = kh * W_KH + kw * W_CS
                for g in range(NUM_GROUPS):
                    g_active = (ag_mask >> g) & 1
                    if g_active != 0:
                        cs = g * GROUP_SIZE_C; offs_k = cs + tl.arange(0, GROUP_SIZE_C); k_m = offs_k < C_IN
                        x_t = tl.load(x_nhwc_ptr + x_hw[:, None] + offs_k[None, :], mask=k_m[None, :] & hw_ok[:, None], other=0.0).to(tl.float16)
                        w_t = tl.load(w_cl_ptr + offs_n[None, :] * W_CO + w_off + offs_k[:, None], mask=k_m[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
                        acc += tl.dot(x_t, w_t)

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
    kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
    threshold=1e-6,
    w_cl=None,
    ag_mask_buf=None, tile_class_buf=None,
    return_ms=False, fallback_ratio=FALLBACK_RATIO,
    return_avg_active_ratio=False, return_tile_stats=False,
    return_backend_meta=False,
    # [Priority C] Pre-converted NHWC tensor
    x_nhwc=None,
    active_tile_ids_buf=None,
    # Legacy compat
    block_size=None, counts_buf=None, tile_cin_buf=None,
    group_flags_buf=None, ag_count_buf=None, ag_list_buf=None,
    tile_alive_buf=None,
):
    import torch.nn.functional as Fn

    N, C_IN, H_IN, W_IN = x.shape
    C_OUT = weight.shape[0]
    device = x.device
    if isinstance(stride, tuple): stride = stride[0]
    if isinstance(padding, tuple): padding = padding[0]
    if isinstance(dilation, tuple): dilation = dilation[0]

    def _finalize_return(y, ms, avg_active_ratio_val=None, tile_stats_val=None, backend_meta_val=None):
        ret = (y, ms)
        if return_avg_active_ratio:
            ret = ret + (avg_active_ratio_val,)
        if return_tile_stats:
            ret = ret + (tile_stats_val,)
        if return_backend_meta:
            ret = ret + (backend_meta_val,)
        return ret

    def _dense_fallback(reason="dense_fallback", avg_active_ratio_val=1.0, tile_stats_val=None, backend_meta_extra=None):
        dense_ms = 0.0
        if return_ms:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
        y = Fn.conv2d(x, weight, bias, stride=stride, padding=padding,
                      dilation=dilation, groups=groups).float()
        if return_ms:
            end_ev.record()
            torch.cuda.synchronize(device)
            dense_ms = start_ev.elapsed_time(end_ev)
        backend_meta_val = {
            "backend": "dense_fallback",
            "reason": reason,
        }
        if backend_meta_extra:
            backend_meta_val.update(backend_meta_extra)
        return _finalize_return(y, dense_ms, avg_active_ratio_val, tile_stats_val, backend_meta_val)

    def _zero_tiles_output(reason, tile_stats_val=None, backend_meta_extra=None):
        y = torch.zeros(N, C_OUT, H_OUT, W_OUT, dtype=torch.float32, device=device)
        if bias is not None:
            y = y + bias.detach().float().view(1, -1, 1, 1)
        backend_meta_val = {
            "backend": "zero_tiles_only",
            "reason": reason,
        }
        if backend_meta_extra:
            backend_meta_val.update(backend_meta_extra)
        return _finalize_return(y, 0.0, 0.0, tile_stats_val, backend_meta_val)

    if groups != 1 or dilation != 1:
        return _dense_fallback(reason="unsupported_groups_or_dilation")

    H_OUT = (H_IN + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    W_OUT = (W_IN + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    if H_OUT <= 0 or W_OUT <= 0:
        return _dense_fallback(reason="invalid_output_shape")

    supported = (
        (kernel_size == 1 and stride == 1 and padding == 0) or
        (kernel_size == 1 and stride == 2 and padding == 0) or
        (kernel_size == 3 and stride == 1 and padding == 1) or
        (kernel_size == 3 and stride == 2 and padding == 1)
    )
    if not supported:
        return _dense_fallback(reason="unsupported_pattern")

    actual_1x1s2 = (kernel_size == 1 and stride == 2)
    if actual_1x1s2:
        return _dense_fallback(reason="split_1x1s2_downsample")

    GROUP_SIZE_C = choose_group_size(C_IN)
    NUM_GROUPS = triton.cdiv(C_IN, GROUP_SIZE_C)
    ALL_ONES_MASK = (1 << NUM_GROUPS) - 1
    # [Priority B] Dense-ish uses wider K
    DENSE_K = min(GROUP_SIZE_C * 2, 64)
    # Ensure DENSE_K is power-of-2 and >= 16
    if DENSE_K < 16:
        DENSE_K = 16

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

    # [Priority C] Prepare fp16 NCHW — avoid redundant .half() if already fp16
    if x.dtype == torch.float16 and x.is_contiguous():
        x_f16 = x
    else:
        x_f16 = x.half().contiguous()

    # Metadata buffers
    if ag_mask_buf is None or ag_mask_buf.numel() < N_TILES:
        ag_mask_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
    if tile_class_buf is None or tile_class_buf.numel() < N_TILES:
        tile_class_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)

    # Prescan (NCHW)
    prescan_stats = {} if return_tile_stats else None
    _build_two_stage_metadata(
        x_f16, N, C_IN, H_IN, W_IN, H_OUT, W_OUT,
        BH, BW, GH, GW,
        kernel_size, stride, padding, threshold,
        ag_mask_buf, tile_class_buf,
        prescan_stats=prescan_stats,
    )

    tc = tile_class_buf[:N_TILES]
    zc = int((tc == TILE_ZERO).sum().item())
    sc = int((tc == TILE_SPARSE).sum().item())
    dc = int((tc == TILE_DENSEISH).sum().item())
    total_nonzero_tiles = sc + dc
    denseish_ratio = float(dc) / max(float(total_nonzero_tiles), 1.0)

    if NUM_GROUPS > 0:
        pc = _popcount_buf(ag_mask_buf, N_TILES)
        avg_active_ratio = float(pc.sum().item()) / max(float(N_TILES * NUM_GROUPS), 1.0)
    else:
        avg_active_ratio = 1.0

    tile_stats_base = None
    if return_tile_stats:
        tile_stats_base = {
            'zero_tiles': zc, 'sparse_tiles': sc, 'denseish_tiles': dc,
            'total_tiles': N_TILES, 'prescan_mode': 'three_stage_nhwc_v25',
            'active_tiles': total_nonzero_tiles,
            'active_tile_ratio': float(total_nonzero_tiles) / max(float(N_TILES), 1.0),
            'denseish_ratio_nonzero': denseish_ratio,
        }
        if prescan_stats:
            tile_stats_base.update(prescan_stats)

    # [Self-rescue] Metadata says this is effectively dense: stop before sparse compute.
    if _check_dense_fallback(ag_mask_buf, N_TILES, NUM_GROUPS, fallback_ratio=fallback_ratio):
        return _dense_fallback(
            reason="post_metadata_dense_fallback",
            avg_active_ratio_val=avg_active_ratio,
            tile_stats_val=tile_stats_base,
            backend_meta_extra={
                "active_tiles": total_nonzero_tiles,
                "total_tiles": N_TILES,
                "denseish_ratio_nonzero": denseish_ratio,
            },
        )

    active_tile_ids, active_tile_count = _build_active_tile_ids(tile_class_buf, N_TILES)
    if active_tile_count == 0:
        return _zero_tiles_output(
            reason="all_tiles_zero_after_metadata",
            tile_stats_val=tile_stats_base,
            backend_meta_extra={"active_tiles": 0, "total_tiles": N_TILES},
        )

    # [Priority C] NHWC for compute — reuse if provided, else derive from x_f16
    if x_nhwc is None:
        x_nhwc = x_f16.permute(0, 2, 3, 1).contiguous()

    y = torch.empty(N, C_OUT, H_OUT, W_OUT, dtype=torch.float32, device=device)

    if return_ms:
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()

    bias_ptr = bias if bias is not None else x_nhwc

    def _grid(meta):
        return (active_tile_count, triton.cdiv(C_OUT, meta["BLOCK_N"]))

    common = dict(
        C_IN=C_IN, C_OUT=C_OUT,
        H_IN=H_IN, W_IN=W_IN, H_OUT=H_OUT, W_OUT=W_OUT,
        GH=GH, GW=GW, HAS_BIAS=(bias is not None),
        GROUP_SIZE_C=GROUP_SIZE_C, NUM_GROUPS=NUM_GROUPS,
        ALL_ONES_MASK=ALL_ONES_MASK, DENSE_K=DENSE_K,
        USE_TILE_IDS=True,
    )

    if kernel_size == 1:
        fn = sparse_conv1x1_nhwc_kernel_8x16 if (BH == 8 and BW == 16) else sparse_conv1x1_nhwc_kernel_8x8
        # 1x1/s1 remains supported for explicit opt-in callers; it still uses full launch.
        def _grid_1x1(meta):
            return (N_TILES, triton.cdiv(C_OUT, meta["BLOCK_N"]))
        common_1x1 = dict(common)
        common_1x1.pop("USE_TILE_IDS")
        fn[_grid_1x1](x_nhwc, w_cl_f16, bias_ptr, ag_mask_buf, y, N, **common_1x1)
    elif kernel_size == 3 and stride == 1:
        fn = sparse_conv3x3s1_nhwc_kernel_8x16 if (BH == 8 and BW == 16) else sparse_conv3x3s1_nhwc_kernel_8x8
        fn[_grid](x_nhwc, w_cl_f16, bias_ptr, ag_mask_buf, active_tile_ids, y, N, **common)
    elif kernel_size == 3 and stride == 2:
        fn = sparse_conv3x3s2_nhwc_kernel_8x16 if (BH == 8 and BW == 16) else sparse_conv3x3s2_nhwc_kernel_8x8
        fn[_grid](x_nhwc, w_cl_f16, bias_ptr, ag_mask_buf, active_tile_ids, y, N, **common)

    sparse_ms = 0.0
    if return_ms:
        end_ev.record()
        torch.cuda.synchronize(device)
        sparse_ms = start_ev.elapsed_time(end_ev)

    tile_stats = tile_stats_base
    backend_meta = {
        "backend": "sparse_triton",
        "reason": "active_tile_compaction",
        "active_tiles": active_tile_count,
        "total_tiles": N_TILES,
        "denseish_ratio_nonzero": denseish_ratio,
    }
    ret = (y, sparse_ms)
    if return_avg_active_ratio: ret = ret + (avg_active_ratio,)
    if return_tile_stats: ret = ret + (tile_stats,)
    return ret