"""
SparseFlow Conv2d Triton Kernels — v25.1

Maturity: main_path (production-facing sparse kernel).

Changes from v25.0 (Round 2 cleanup — no semantic changes):
  - Fixed mojibake in docstrings.
  - Removed dead helper _build_active_group_metadata (zero callers).
  - Removed 7 unused "legacy compat" parameters from sparse_conv2d_forward
    (block_size, counts_buf, tile_cin_buf, group_flags_buf, ag_count_buf,
    ag_list_buf, tile_alive_buf). None of them were read by the function body
    and no caller in the repository passed any of them.
  - Deduplicated rf_h/rf_w/RF_SIZE computation in _build_two_stage_metadata
    (previously recomputed three times in the kernel_size!=1 path).

Changes from v24 (already in v25.0, retained):
  [P0] sparse_conv2d_forward: all .item() syncs gated behind need_stats flag.
       When return_avg_active_ratio=False AND return_tile_stats=False, the
       function performs ZERO GPU→CPU synchronizations.
  [P1] launch_all_tiles parameter for A/B tile launch comparison.
       Mode A (False): active-tile-ID launch via nonzero() (1 sync).
       Mode B (True):  launch all N_TILES, zero tiles early-return (0 sync).
  All Triton JIT kernels unchanged from v24.

Supported patterns: 1x1/s1/p0, 1x1/s2/p0, 3x3/s1/p1, 3x3/s2/p1
"""

# v26 notes:
# - Added a fused prescan+compute persistent backend for 1x1/s1/p0 when
#   statistics are not requested.
# - v26.1 shares receptive-field prescan with conv1d/conv3d via
#   Kernels/_prescan_common.py.

import torch
import triton
import triton.language as tl
from triton import autotune, Config
from Kernels._prescan_common import _build_rf_prescan_metadata_impl
from Utils.config import PRESCAN_ACTIVITY_EPS, SPARSE_DENSE_RATIO_THRESHOLD

# ---------------------------------------------------------------------------
# Tile classification constants
# ---------------------------------------------------------------------------
TILE_ZERO = 0
TILE_SPARSE = 1
TILE_DENSEISH = 2
TILE_UNCERTAIN = 3
TILE_ZERO_CANDIDATE = 4

FALLBACK_RATIO = SPARSE_DENSE_RATIO_THRESHOLD
AUTO_LAUNCH_ALL_ACTIVE_TILE_RATIO = 0.75
PERSISTENT_OVERSUB = 2
USE_FUSED_1X1_PERSISTENT = True

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


def _popcount_buf(ag_mask_buf, N_TILES):
    """Vectorised SWAR popcount for int32 bitmask buffer."""
    v = ag_mask_buf[:N_TILES].int()
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F
    v = v + (v >> 8)
    v = v + (v >> 16)
    return (v & 0x3F).to(torch.int32)


def _check_dense_fallback(ag_mask_buf, N_TILES, NUM_GROUPS, fallback_ratio=FALLBACK_RATIO):
    """NOTE: calls .mean().item() → GPU→CPU sync. Only use in gated paths."""
    if NUM_GROUPS == 0:
        return False
    pc = _popcount_buf(ag_mask_buf, N_TILES)
    avg_active = pc.float().mean().item()
    return avg_active > fallback_ratio * NUM_GROUPS


def _build_active_tile_ids(tile_class_buf, N_TILES, active_tile_ids_buf=None):
    """NOTE: calls torch.nonzero() → GPU→CPU sync. Only use in Mode A path."""
    tc = tile_class_buf[:N_TILES]
    active = torch.nonzero(tc != TILE_ZERO, as_tuple=False).flatten()
    if active.numel() == 0:
        return active.to(dtype=torch.int32), 0
    active = active.to(dtype=torch.int32).contiguous()
    if active_tile_ids_buf is not None and active_tile_ids_buf.numel() >= active.numel():
        active_tile_ids_buf[: active.numel()].copy_(active)
        return active_tile_ids_buf[: active.numel()], int(active.numel())
    return active, int(active.numel())


def _summarize_stage1_metadata(tile_class_buf, ag_mask_buf, N_TILES, NUM_GROUPS):
    """Summarize Stage-1 coarse metadata.

    The Stage-1 group mask is a lower bound on the final active-group mask for
    ZERO_CANDIDATE / UNCERTAIN tiles. If this lower bound is already dense-ish
    enough, we can safely bail out to the dense path without paying for the
    later refinement stages.
    """
    tc = tile_class_buf[:N_TILES]
    summary = {
        "stage1_zero_candidate": int((tc == TILE_ZERO_CANDIDATE).sum().item()),
        "stage1_denseish": int((tc == TILE_DENSEISH).sum().item()),
        "stage1_uncertain": int((tc == TILE_UNCERTAIN).sum().item()),
    }
    if N_TILES <= 0:
        summary["stage1_avg_active_group_ratio_lower_bound"] = 0.0
        return summary, 0.0
    if NUM_GROUPS <= 0:
        summary["stage1_avg_active_group_ratio_lower_bound"] = 1.0
        return summary, 1.0

    pc = _popcount_buf(ag_mask_buf, N_TILES)
    avg_active_ratio = float(pc.sum().item()) / max(float(N_TILES * NUM_GROUPS), 1.0)
    summary["stage1_avg_active_group_ratio_lower_bound"] = avg_active_ratio
    return summary, avg_active_ratio


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
            is_active = (tl.max(tl.abs(vals), axis=0) > THRESHOLD).to(tl.int32)
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
            is_active = (tl.max(tl.abs(vals), axis=0) > THRESHOLD).to(tl.int32)
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
# STAGE 2a: Zero-candidate refinement
# ===========================================================================

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

    mask = tl.zeros([1], dtype=tl.int32)
    found_nz = tl.zeros([1], dtype=tl.int32)
    g_idx = tl.zeros([1], dtype=tl.int32)

    while (tl.sum(g_idx) < NUM_GROUPS) & (tl.sum(found_nz) == 0):
        g_val = tl.sum(g_idx)
        group_max = tl.zeros([1], dtype=tl.float32)
        # NOTE:
        # c_off starts at 1 by design. Stage-1 coarse pass already inspected
        # the representative channel (offset 0) for each group. Stage-2a only
        # refines the remaining channels to avoid redundant reads.
        for c_off in range(1, GROUP_SIZE_C):
            c = g_val * GROUP_SIZE_C + c_off
            if c < C_IN:
                vals = tl.load(x_ptr + (n_idx * C_IN + c) * HW + safe_h * W_IN + safe_w,
                               mask=hw_mask, other=0.0)
                group_max = tl.maximum(group_max, tl.max(tl.abs(vals), axis=0))
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
        # See note above: offset 0 is intentionally skipped in Stage-2a.
        for c_off in range(1, GROUP_SIZE_C):
            c = g_val * GROUP_SIZE_C + c_off
            if c < C_IN:
                vals = tl.load(x_ptr + (n_idx * C_IN + c) * HW + safe_h * W_IN + safe_w,
                               mask=hw_mask, other=0.0)
                group_max = tl.maximum(group_max, tl.max(tl.abs(vals), axis=0))
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
        # NOTE:
        # c_off starts at 1 by design. Stage-1 coarse pass already inspected
        # the representative channel (offset 0) for each group. Stage-2a only
        # refines the remaining channels to avoid redundant reads.
        for c_off in range(1, GROUP_SIZE_C):
            c = g_val * GROUP_SIZE_C + c_off
            if c < C_IN:
                vals = tl.load(x_ptr + (n_idx * C_IN + c) * HW + out_h * W_IN + out_w, mask=m_mask, other=0.0)
                group_max = tl.maximum(group_max, tl.max(tl.abs(vals), axis=0))
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
        # See note above: offset 0 is intentionally skipped in Stage-2a.
        for c_off in range(1, GROUP_SIZE_C):
            c = g_val * GROUP_SIZE_C + c_off
            if c < C_IN:
                vals = tl.load(x_ptr + (n_idx * C_IN + c) * HW + out_h * W_IN + out_w, mask=m_mask, other=0.0)
                group_max = tl.maximum(group_max, tl.max(tl.abs(vals), axis=0))
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
# STAGE 2b: Exact bitmask for UNCERTAIN tiles
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
                vals = tl.load(x_ptr + (n_idx * C_IN + c) * HW + safe_h * W_IN + safe_w, mask=hw_mask, other=0.0)
                group_max = tl.maximum(group_max, tl.max(tl.abs(vals), axis=0))
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
                group_max = tl.maximum(group_max, tl.max(tl.abs(vals), axis=0))
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
    allow_stage1_dense_fallback=False,
    fallback_ratio=FALLBACK_RATIO,
):
    GROUP_SIZE_C = choose_group_size(C_IN)
    N_TILES = N * GH * GW
    NUM_GROUPS = triton.cdiv(C_IN, GROUP_SIZE_C)
    stage1_summary = None
    x_nhwc = x_f16.permute(0, 2, 3, 1)
    tile_class, ag_mask, debug_stats = _build_rf_prescan_metadata_impl(
        x_channels_last=x_nhwc,
        spatial_dims=(H_OUT, W_OUT),
        kernel_dims=(kernel_size, kernel_size),
        stride=stride,
        padding=padding,
        block_dims=(BH, BW),
        group_size_c=GROUP_SIZE_C,
        num_groups=NUM_GROUPS,
        threshold=threshold,
        return_debug_stats=(prescan_stats is not None or allow_stage1_dense_fallback),
        tile_class_out=tile_class_buf,
        ag_mask_out=ag_mask_buf,
    )

    if debug_stats is not None:
        stage1_summary = {
            "stage1_zero_candidate": int(debug_stats["stage1_zero_candidate"]),
            "stage1_denseish": int(debug_stats["stage1_denseish"]),
            "stage1_uncertain": int(debug_stats["stage1_uncertain"]),
            "stage1_avg_active_group_ratio_lower_bound": float(
                debug_stats["stage1_avg_active_group_ratio_lower_bound"]
            ),
        }
        if prescan_stats is not None:
            prescan_stats.update(debug_stats)
        if (
            allow_stage1_dense_fallback
            and float(debug_stats["stage1_avg_active_group_ratio_lower_bound"]) > fallback_ratio
        ):
            if prescan_stats is not None:
                prescan_stats["stage1_dense_fallback"] = 1
            return GROUP_SIZE_C, NUM_GROUPS, True, stage1_summary

    return GROUP_SIZE_C, NUM_GROUPS, False, stage1_summary


# ---------------------------------------------------------------------------
# Autotune configs
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
# NHWC Compute Kernels
# ===================================================================

# ---- 1x1 / s1, 8x8 ----
@autotune(configs=_CONFIGS_8x8, key=["C_IN", "C_OUT", "H_OUT", "W_OUT", "GH", "GW"])
@triton.jit
def sparse_conv1x1_nhwc_kernel_8x8(
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
            offs_k = cin_base + tl.arange(0, DENSE_K); k_v = offs_k < C_IN
            x_t = tl.load(x_nhwc_ptr + x_base[:, None] + offs_k[None, :], mask=k_v[None, :] & m_mask[:, None], other=0.0).to(tl.float16)
            w_t = tl.load(w_cl_ptr + offs_n[None, :] * C_IN + offs_k[:, None], mask=k_v[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
            acc += tl.dot(x_t, w_t)
    else:
        for g in range(NUM_GROUPS):
            g_active = (ag_mask >> g) & 1
            if g_active != 0:
                cs = g * GROUP_SIZE_C; offs_k = cs + tl.arange(0, GROUP_SIZE_C); k_m = offs_k < C_IN
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
            offs_k = cin_base + tl.arange(0, DENSE_K); k_v = offs_k < C_IN
            x_t = tl.load(x_nhwc_ptr + x_base[:, None] + offs_k[None, :], mask=k_v[None, :] & m_mask[:, None], other=0.0).to(tl.float16)
            w_t = tl.load(w_cl_ptr + offs_n[None, :] * C_IN + offs_k[:, None], mask=k_v[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
            acc += tl.dot(x_t, w_t)
    else:
        for g in range(NUM_GROUPS):
            g_active = (ag_mask >> g) & 1
            if g_active != 0:
                cs = g * GROUP_SIZE_C; offs_k = cs + tl.arange(0, GROUP_SIZE_C); k_m = offs_k < C_IN
                x_t = tl.load(x_nhwc_ptr + x_base[:, None] + offs_k[None, :], mask=k_m[None, :] & m_mask[:, None], other=0.0).to(tl.float16)
                w_t = tl.load(w_cl_ptr + offs_n[None, :] * C_IN + offs_k[:, None], mask=k_m[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
                acc += tl.dot(x_t, w_t)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
    oa = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
    tl.store(oa, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _fused_prescan_compute_conv1x1_persistent(
    x_nhwc_ptr, w_ptr, bias_ptr, y_ptr,
    N_val, C_IN, C_OUT, H_IN, W_IN,
    SPATIAL_TILES, N_COUT_TILES, TOTAL_WORK,
    THRESHOLD,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, NUM_GROUPS: tl.constexpr, NUM_SMS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    hw = H_IN * W_IN
    hw_out = H_IN * W_IN
    wc = W_IN * C_IN
    all_ones_mask: tl.constexpr = (1 << NUM_GROUPS) - 1
    linear = pid

    while linear < TOTAL_WORK:
        pid_cout = linear % N_COUT_TILES
        tmp = linear // N_COUT_TILES
        n_idx = tmp // SPATIAL_TILES
        spatial_tile = tmp % SPATIAL_TILES

        if n_idx < N_val:
            offs_n = pid_cout * BLOCK_N + tl.arange(0, BLOCK_N)
            n_mask = offs_n < C_OUT

            offs_m = spatial_tile * BLOCK_M + tl.arange(0, BLOCK_M)
            m_mask = offs_m < hw
            out_h = offs_m // W_IN
            out_w = offs_m % W_IN
            x_base = n_idx * H_IN * wc + out_h * wc + out_w * C_IN

            ag_mask = tl.zeros([1], dtype=tl.int32)
            for g in range(NUM_GROUPS):
                cs = g * GROUP_SIZE_C
                group_has_nz = tl.zeros([1], dtype=tl.int32)
                for k_off in range(0, GROUP_SIZE_C, BLOCK_K):
                    offs_k = cs + k_off + tl.arange(0, BLOCK_K)
                    k_mask = offs_k < C_IN
                    x_t = tl.load(
                        x_nhwc_ptr + x_base[:, None] + offs_k[None, :],
                        mask=m_mask[:, None] & k_mask[None, :],
                        other=0.0,
                    )
                    chunk_active = (tl.max(tl.abs(x_t), axis=0) > THRESHOLD).to(tl.int32)
                    group_has_nz = tl.maximum(
                        group_has_nz,
                        (tl.sum(chunk_active, axis=0) > 0).to(tl.int32),
                    )
                ag_mask = ag_mask + group_has_nz * (1 << g)

            acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            if tl.sum(ag_mask) != 0:
                if tl.sum(ag_mask == all_ones_mask) > 0:
                    for cin_base in range(0, NUM_GROUPS * GROUP_SIZE_C, BLOCK_K):
                        offs_k = cin_base + tl.arange(0, BLOCK_K)
                        k_mask = offs_k < C_IN
                        x_t = tl.load(
                            x_nhwc_ptr + x_base[:, None] + offs_k[None, :],
                            mask=m_mask[:, None] & k_mask[None, :],
                            other=0.0,
                        ).to(tl.float16)
                        w_t = tl.load(
                            w_ptr + offs_n[None, :] * C_IN + offs_k[:, None],
                            mask=k_mask[:, None] & n_mask[None, :],
                            other=0.0,
                        ).to(tl.float16)
                        acc += tl.dot(x_t, w_t)
                else:
                    ag_bits = tl.sum(ag_mask)
                    for g in range(NUM_GROUPS):
                        g_active = (ag_bits >> g) & 1
                        if g_active != 0:
                            cs = g * GROUP_SIZE_C
                            for k_off in range(0, GROUP_SIZE_C, BLOCK_K):
                                offs_k = cs + k_off + tl.arange(0, BLOCK_K)
                                k_mask = offs_k < C_IN
                                x_t = tl.load(
                                    x_nhwc_ptr + x_base[:, None] + offs_k[None, :],
                                    mask=m_mask[:, None] & k_mask[None, :],
                                    other=0.0,
                                ).to(tl.float16)
                                w_t = tl.load(
                                    w_ptr + offs_n[None, :] * C_IN + offs_k[:, None],
                                    mask=k_mask[:, None] & n_mask[None, :],
                                    other=0.0,
                                ).to(tl.float16)
                                acc += tl.dot(x_t, w_t)

            if HAS_BIAS:
                acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]

            oa = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * hw_out + out_h[:, None] * W_IN + out_w[:, None]
            tl.store(oa, acc, mask=m_mask[:, None] & n_mask[None, :])

        linear += NUM_SMS


def _fused_1x1_persistent_path(
    x_f16,
    x_nhwc,
    weight,
    bias,
    w_cl,
    N,
    C_IN,
    C_OUT,
    H_OUT,
    W_OUT,
    BH,
    BW,
    GROUP_SIZE_C,
    DENSE_K,
    threshold,
    return_ms,
):
    device = x_f16.device
    if x_nhwc is None:
        x_nhwc = x_f16.permute(0, 2, 3, 1).contiguous()
    if w_cl is not None:
        w_cl_f16 = w_cl
    else:
        w_cl_f16 = weight.half().reshape(C_OUT, C_IN).contiguous()

    block_m = BH * BW
    block_n = 64 if C_OUT >= 64 else 32
    block_k = 32 if DENSE_K >= 32 else 16
    spatial_tiles = triton.cdiv(H_OUT * W_OUT, block_m)
    n_cout_tiles = triton.cdiv(C_OUT, block_n)
    total_work = int(N * spatial_tiles * n_cout_tiles)

    if total_work <= 0:
        y = torch.zeros(N, C_OUT, H_OUT, W_OUT, dtype=torch.float32, device=device)
        if bias is not None:
            y = y + bias.detach().float().view(1, -1, 1, 1)
        return y, 0.0, {
            "backend": "zero_tiles_only",
            "reason": "empty_fused_1x1_work",
            "total_tiles": int(N * spatial_tiles),
            "launch_count": 0,
            "launch_mode": "persistent",
            "launch_mode_reason": "persistent_fused_1x1",
        }

    num_programs = max(
        1,
        int(torch.cuda.get_device_properties(device).multi_processor_count) * PERSISTENT_OVERSUB,
    )
    y = torch.empty(N, C_OUT, H_OUT, W_OUT, dtype=torch.float32, device=device)

    sparse_ms = 0.0
    if return_ms:
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()

    bias_ptr = bias if bias is not None else x_nhwc
    _fused_prescan_compute_conv1x1_persistent[(num_programs,)](
        x_nhwc,
        w_cl_f16,
        bias_ptr,
        y,
        N,
        C_IN,
        C_OUT,
        H_OUT,
        W_OUT,
        spatial_tiles,
        n_cout_tiles,
        total_work,
        float(threshold),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_SIZE_C=GROUP_SIZE_C,
        NUM_GROUPS=triton.cdiv(C_IN, GROUP_SIZE_C),
        NUM_SMS=num_programs,
        HAS_BIAS=(bias is not None),
    )

    if return_ms:
        end_ev.record()
        torch.cuda.synchronize(device)
        sparse_ms = start_ev.elapsed_time(end_ev)

    backend_meta = {
        "backend": "sparse_triton",
        "reason": "fused_1x1_persistent_v26",
        "total_tiles": int(N * spatial_tiles),
        "launch_count": int(total_work),
        "launch_mode": "persistent",
        "launch_mode_reason": "persistent_fused_1x1",
        "persistent_programs": int(num_programs),
    }
    return y, sparse_ms, backend_meta


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
                    offs_k = cin_base + tl.arange(0, DENSE_K); k_v = offs_k < C_IN
                    x_t = tl.load(x_nhwc_ptr + x_hw[:, None] + offs_k[None, :], mask=k_v[None, :] & hw_ok[:, None], other=0.0).to(tl.float16)
                    w_t = tl.load(w_cl_ptr + offs_n[None, :] * W_CO + w_off + offs_k[:, None], mask=k_v[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
                    acc += tl.dot(x_t, w_t)
    else:
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
                        cs = g * GROUP_SIZE_C; offs_k = cs + tl.arange(0, GROUP_SIZE_C); k_m = offs_k < C_IN
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
# Python entry point — v25: sync-gated + A/B tile launch switch
# ===================================================================

def sparse_conv2d_forward(
    x, weight, bias,
    kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
    threshold=PRESCAN_ACTIVITY_EPS, w_cl=None,
    ag_mask_buf=None, tile_class_buf=None,
    return_ms=False, fallback_ratio=FALLBACK_RATIO,
    return_avg_active_ratio=False, return_tile_stats=False,
    return_backend_meta=False,
    x_nhwc=None, active_tile_ids_buf=None,
    launch_all_tiles=False,
):
    import torch.nn.functional as Fn

    N, C_IN, H_IN, W_IN = x.shape
    C_OUT = weight.shape[0]
    device = x.device
    if isinstance(stride, tuple): stride = stride[0]
    if isinstance(padding, tuple): padding = padding[0]
    if isinstance(dilation, tuple): dilation = dilation[0]

    need_stats = return_tile_stats or return_avg_active_ratio

    def _finalize_return(y, ms, avg_active_ratio_val=None, tile_stats_val=None, backend_meta_val=None):
        ret = (y, ms)
        if return_avg_active_ratio: ret = ret + (avg_active_ratio_val,)
        if return_tile_stats: ret = ret + (tile_stats_val,)
        if return_backend_meta: ret = ret + (backend_meta_val,)
        return ret

    def _dense_fallback(reason="dense_fallback", avg_active_ratio_val=1.0, tile_stats_val=None, backend_meta_extra=None):
        dense_ms = 0.0
        if return_ms:
            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            se.record()

        # dtype-align for fallback path
        x_dense = x.float()
        w_dense = weight.float()
        b_dense = bias.float() if bias is not None else None

        y = Fn.conv2d(
            x_dense, w_dense, b_dense,
            stride=stride, padding=padding, dilation=dilation, groups=groups
        ).float()

        if return_ms:
            ee.record()
            torch.cuda.synchronize(device)
            dense_ms = se.elapsed_time(ee)

        bm = {"backend": "dense_fallback", "reason": reason}
        if backend_meta_extra:
            bm.update(backend_meta_extra)
        return _finalize_return(y, dense_ms, avg_active_ratio_val, tile_stats_val, bm)

    def _zero_tiles_output(reason, tile_stats_val=None, backend_meta_extra=None):
        y = torch.zeros(N, C_OUT, H_OUT, W_OUT, dtype=torch.float32, device=device)
        if bias is not None: y = y + bias.detach().float().view(1, -1, 1, 1)
        bm = {"backend": "zero_tiles_only", "reason": reason}
        if backend_meta_extra: bm.update(backend_meta_extra)
        return _finalize_return(y, 0.0, 0.0, tile_stats_val, bm)

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

    if kernel_size == 1 and stride == 2:
        return _dense_fallback(reason="split_1x1s2_downsample")

    GROUP_SIZE_C = choose_group_size(C_IN)
    NUM_GROUPS = triton.cdiv(C_IN, GROUP_SIZE_C)
    ALL_ONES_MASK = (1 << NUM_GROUPS) - 1
    DENSE_K = min(GROUP_SIZE_C * 2, 64)
    if DENSE_K < 16: DENSE_K = 16

    BH, BW = _select_tile_sizes(H_OUT, W_OUT)
    GH = triton.cdiv(H_OUT, BH)
    GW = triton.cdiv(W_OUT, BW)
    N_TILES = N * GH * GW

    x_f16 = x if (x.dtype == torch.float16 and x.is_contiguous()) else x.half().contiguous()

    if (
        kernel_size == 1
        and stride == 1
        and padding == 0
        and groups == 1
        and dilation == 1
        and USE_FUSED_1X1_PERSISTENT
        and not need_stats
    ):
        y, sparse_ms, backend_meta = _fused_1x1_persistent_path(
            x_f16=x_f16,
            x_nhwc=x_nhwc,
            weight=weight,
            bias=bias,
            w_cl=w_cl,
            N=N,
            C_IN=C_IN,
            C_OUT=C_OUT,
            H_OUT=H_OUT,
            W_OUT=W_OUT,
            BH=BH,
            BW=BW,
            GROUP_SIZE_C=GROUP_SIZE_C,
            DENSE_K=DENSE_K,
            threshold=threshold,
            return_ms=return_ms,
        )
        return _finalize_return(y, sparse_ms, None, None, backend_meta)

    if ag_mask_buf is None or ag_mask_buf.numel() < N_TILES:
        ag_mask_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
    if tile_class_buf is None or tile_class_buf.numel() < N_TILES:
        tile_class_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)

    prescan_stats = {} if return_tile_stats else None
    stage1_dense_fallback = False
    stage1_summary = None
    GROUP_SIZE_C, NUM_GROUPS, stage1_dense_fallback, stage1_summary = _build_two_stage_metadata(
        x_f16, N, C_IN, H_IN, W_IN, H_OUT, W_OUT,
        BH, BW, GH, GW, kernel_size, stride, padding, threshold,
        ag_mask_buf, tile_class_buf,
        prescan_stats=prescan_stats,
        allow_stage1_dense_fallback=(return_avg_active_ratio and not return_tile_stats),
        fallback_ratio=fallback_ratio,
    )

    if stage1_dense_fallback:
        stage1_avg_active_ratio = 1.0
        if stage1_summary is not None:
            stage1_avg_active_ratio = float(
                stage1_summary.get("stage1_avg_active_group_ratio_lower_bound", 1.0)
            )
        return _dense_fallback(
            reason="stage1_metadata_dense_fallback",
            avg_active_ratio_val=stage1_avg_active_ratio,
            tile_stats_val=None,
            backend_meta_extra={
                "stage1_dense_fallback": True,
                "total_tiles": N_TILES,
            },
        )

    # ====== [P0] Sync-gated: only compute stats when requested ======
    avg_active_ratio = None
    tile_stats_base = None
    active_tiles_for_meta = None
    launch_mode_reason = "all_tiles_requested" if launch_all_tiles else "active_only_default"

    if need_stats:
        tc = tile_class_buf[:N_TILES]
        zc = int((tc == TILE_ZERO).sum().item())
        sc = int((tc == TILE_SPARSE).sum().item())
        dc = int((tc == TILE_DENSEISH).sum().item())
        total_nonzero = sc + dc
        denseish_ratio = float(dc) / max(float(total_nonzero), 1.0)
        active_tiles_for_meta = int(total_nonzero)

        if NUM_GROUPS > 0:
            pc = _popcount_buf(ag_mask_buf, N_TILES)
            avg_active_ratio = float(pc.sum().item()) / max(float(N_TILES * NUM_GROUPS), 1.0)
        else:
            avg_active_ratio = 1.0

        if return_tile_stats:
            tile_stats_base = {
                'zero_tiles': zc, 'sparse_tiles': sc, 'denseish_tiles': dc,
                'total_tiles': N_TILES, 'prescan_mode': 'three_stage_nhwc_v25',
                'active_tiles': total_nonzero,
                'active_tile_ratio': float(total_nonzero) / max(float(N_TILES), 1.0),
                'denseish_ratio_nonzero': denseish_ratio,
            }
            if prescan_stats: tile_stats_base.update(prescan_stats)

        if _check_dense_fallback(ag_mask_buf, N_TILES, NUM_GROUPS, fallback_ratio=fallback_ratio):
            return _dense_fallback(
                reason="post_metadata_dense_fallback",
                avg_active_ratio_val=avg_active_ratio,
                tile_stats_val=tile_stats_base,
                backend_meta_extra={"active_tiles": total_nonzero, "total_tiles": N_TILES, "denseish_ratio_nonzero": denseish_ratio},
            )
        if (
            not launch_all_tiles
            and N_TILES > 0
            and (float(total_nonzero) / float(N_TILES)) >= AUTO_LAUNCH_ALL_ACTIVE_TILE_RATIO
        ):
            launch_all_tiles = True
            launch_mode_reason = "auto_active_tile_ratio"

    # ====== [P1] A/B tile launch switch ======
    if launch_all_tiles:
        launch_count = N_TILES
        use_tile_ids = False
        tile_ids_ptr = ag_mask_buf  # unused placeholder
    else:
        active_tile_ids, active_tile_count = _build_active_tile_ids(
            tile_class_buf,
            N_TILES,
            active_tile_ids_buf=active_tile_ids_buf,
        )
        if active_tile_count == 0:
            return _zero_tiles_output(
                reason="all_tiles_zero_after_metadata",
                tile_stats_val=tile_stats_base,
                backend_meta_extra={"active_tiles": 0, "total_tiles": N_TILES},
            )
        launch_count = active_tile_count
        use_tile_ids = True
        tile_ids_ptr = active_tile_ids
        if active_tiles_for_meta is None:
            active_tiles_for_meta = int(active_tile_count)

    if x_nhwc is None:
        x_nhwc = x_f16.permute(0, 2, 3, 1).contiguous()
    if w_cl is not None:
        w_cl_f16 = w_cl
    else:
        w_cl_f16 = (
            weight.half().permute(0, 2, 3, 1).contiguous()
            if kernel_size == 3
            else weight.half().reshape(C_OUT, C_IN).contiguous()
        )

    y = torch.empty(N, C_OUT, H_OUT, W_OUT, dtype=torch.float32, device=device)

    if return_ms:
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()

    bias_ptr = bias if bias is not None else x_nhwc

    def _grid(meta):
        return (launch_count, triton.cdiv(C_OUT, meta["BLOCK_N"]))

    common = dict(
        C_IN=C_IN, C_OUT=C_OUT, H_IN=H_IN, W_IN=W_IN, H_OUT=H_OUT, W_OUT=W_OUT,
        GH=GH, GW=GW, HAS_BIAS=(bias is not None),
        GROUP_SIZE_C=GROUP_SIZE_C, NUM_GROUPS=NUM_GROUPS,
        ALL_ONES_MASK=ALL_ONES_MASK, DENSE_K=DENSE_K,
        USE_TILE_IDS=use_tile_ids,
    )

    if kernel_size == 1:
        fn = sparse_conv1x1_nhwc_kernel_8x16 if (BH == 8 and BW == 16) else sparse_conv1x1_nhwc_kernel_8x8
        fn[_grid](x_nhwc, w_cl_f16, bias_ptr, ag_mask_buf, tile_ids_ptr, y, N, **common)
    elif stride == 1:
        fn = sparse_conv3x3s1_nhwc_kernel_8x16 if (BH == 8 and BW == 16) else sparse_conv3x3s1_nhwc_kernel_8x8
        fn[_grid](x_nhwc, w_cl_f16, bias_ptr, ag_mask_buf, tile_ids_ptr, y, N, **common)
    else:
        fn = sparse_conv3x3s2_nhwc_kernel_8x16 if (BH == 8 and BW == 16) else sparse_conv3x3s2_nhwc_kernel_8x8
        fn[_grid](x_nhwc, w_cl_f16, bias_ptr, ag_mask_buf, tile_ids_ptr, y, N, **common)

    sparse_ms = 0.0
    if return_ms:
        end_ev.record(); torch.cuda.synchronize(device); sparse_ms = start_ev.elapsed_time(end_ev)

    backend_meta = {
        "backend": "sparse_triton", "reason": "v25_sync_gated",
        "total_tiles": N_TILES, "launch_count": launch_count,
        "launch_mode": "all_tiles" if launch_all_tiles else "active_only",
        "launch_mode_reason": launch_mode_reason,
    }
    if active_tiles_for_meta is not None:
        backend_meta["active_tiles"] = int(active_tiles_for_meta)

    return _finalize_return(y, sparse_ms, avg_active_ratio, tile_stats_base, backend_meta)
