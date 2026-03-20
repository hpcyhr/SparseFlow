"""
SparseFlow Conv2d Triton Kernels — v19.0 Two-Stage Prescan + Tile Classification

Changes from v18:
  A. Prescan: genuine two-stage design
     - Stage 1 (tile_screen): cheap per-tile max across ALL channels (no per-group tracking)
       → classifies tiles as ZERO or ALIVE
     - Stage 2 (group_bitmask): only runs on ALIVE tiles, builds per-group bitmask
     - Zero tiles (~99% in SNN) skip Stage 2 entirely
  B. Tile classification: TILE_ZERO / TILE_SPARSE / TILE_DENSEISH
     - Dense-ish tiles (all groups active) use simplified compute path
  C. Compute kernels: per-tile adaptive path
     - Zero tiles: bias-only early return
     - Dense-ish tiles: iterate all groups unconditionally (no bit-check overhead)
     - Sparse tiles: bitmask-gated iteration (existing path)
  D. 1x1 specialization: simpler prescan (no spatial RF expansion)
  E. Diagnostics: stage1_zero_tiles, stage2_tiles, denseish_tiles, sparse_tiles

Supported patterns:
  1x1/s1/p0, 1x1/s2/p0 (via subsample), 3x3/s1/p1, 3x3/s2/p1
"""

import torch
import triton
import triton.language as tl
from triton import autotune, Config

# ---------------------------------------------------------------------------
# Tile classification constants (used in tile_class buffer)
# ---------------------------------------------------------------------------
TILE_ZERO = 0
TILE_SPARSE = 1
TILE_DENSEISH = 2

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


# ---------------------------------------------------------------------------
# Stage 1: Tile screen — cheap per-tile zero/alive classification
# ---------------------------------------------------------------------------
# Single scalar accumulator per tile. No per-group bookkeeping.
# For zero tiles (99%+), this is the ONLY prescan work done.

@triton.jit
def tile_screen_kernel(
    x_ptr,
    tile_alive_ptr,       # [N_TILES] int32 output: 0=zero, 1=alive
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
    # Number of channels to check per iteration for early-exit
    C_BATCH: tl.constexpr,
):
    """Stage 1: Quick tile-level zero/alive classification.

    Scans ALL channels × RF positions using a single scalar accumulator.
    Uses while-loop with early exit: as soon as ANY nonzero value is found,
    the tile is classified as alive and we stop scanning.
    """
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

    # Scan channels with early exit via while loop
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


# ---------------------------------------------------------------------------
# Stage 1 specialized for 1x1 conv: no spatial RF expansion needed
# ---------------------------------------------------------------------------

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
    """Stage 1 for 1x1 conv: tile positions correspond directly to input positions."""
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
# Stage 2: Group bitmask — only runs on alive tiles
# ---------------------------------------------------------------------------

@triton.jit
def group_bitmask_kernel(
    x_ptr,
    tile_alive_ptr,       # [N_TILES] int32 input from Stage 1
    ag_mask_ptr,          # [N_TILES] int32 output — per-tile bitmask
    tile_class_ptr,       # [N_TILES] int32 output — TILE_ZERO/SPARSE/DENSEISH
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
    MAX_AG: tl.constexpr,
    RF_SIZE: tl.constexpr,
    THRESHOLD: tl.constexpr,
    ALL_ONES_MASK: tl.constexpr,
):
    """Stage 2: Per-group bitmask construction for ALIVE tiles only.

    Dead tiles (tile_alive==0) get mask=0 and class=TILE_ZERO immediately.
    Alive tiles get full group scan and classification:
      - All groups active → TILE_DENSEISH
      - Some groups active → TILE_SPARSE
    """
    tile_id = tl.program_id(0)
    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles:
        return

    off1 = tl.arange(0, 1)

    # Check Stage 1 result — dead tiles skip entirely
    alive = tl.load(tile_alive_ptr + tile_id + off1)
    if tl.sum(alive) == 0:
        tl.store(ag_mask_ptr + tile_id + off1, tl.zeros([1], dtype=tl.int32))
        tl.store(tile_class_ptr + tile_id + off1, tl.zeros([1], dtype=tl.int32))  # TILE_ZERO=0
        return

    # Alive tile — full per-group bitmask construction
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
        mask = mask + is_active_int * (1 << g)

    tl.store(ag_mask_ptr + tile_id + off1, mask)

    # Classify: denseish if all groups active, else sparse
    is_denseish = (mask == ALL_ONES_MASK).to(tl.int32)
    # TILE_SPARSE=1, TILE_DENSEISH=2
    tile_class = 1 + is_denseish
    tl.store(tile_class_ptr + tile_id + off1, tile_class)


# ---------------------------------------------------------------------------
# Stage 2 specialized for 1x1 conv
# ---------------------------------------------------------------------------

@triton.jit
def group_bitmask_1x1_kernel(
    x_ptr,
    tile_alive_ptr,
    ag_mask_ptr,
    tile_class_ptr,
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
):
    tile_id = tl.program_id(0)
    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles:
        return

    off1 = tl.arange(0, 1)
    alive = tl.load(tile_alive_ptr + tile_id + off1)
    if tl.sum(alive) == 0:
        tl.store(ag_mask_ptr + tile_id + off1, tl.zeros([1], dtype=tl.int32))
        tl.store(tile_class_ptr + tile_id + off1, tl.zeros([1], dtype=tl.int32))
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
    for g in range(NUM_GROUPS):
        group_max = tl.zeros([1], dtype=tl.float32)
        for c_off in range(GROUP_SIZE_C):
            c = g * GROUP_SIZE_C + c_off
            if c < C_IN:
                addrs = (n_idx * C_IN + c) * HW + out_h * W_IN + out_w
                vals = tl.load(x_ptr + addrs, mask=m_mask, other=0.0)
                ch_max = tl.max(vals, axis=0)
                group_max = tl.maximum(group_max, ch_max)
        is_active_int = (group_max > THRESHOLD).to(tl.int32)
        mask = mask + is_active_int * (1 << g)

    tl.store(ag_mask_ptr + tile_id + off1, mask)
    is_denseish = (mask == ALL_ONES_MASK).to(tl.int32)
    tile_class = 1 + is_denseish
    tl.store(tile_class_ptr + tile_id + off1, tile_class)


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
    tile_alive_buf,
    ag_mask_buf,
    tile_class_buf,
):
    """Two-stage prescan: tile_screen → conditional group_bitmask.

    Returns (GROUP_SIZE_C, NUM_GROUPS).
    """
    GROUP_SIZE_C = choose_group_size(C_IN)
    N_TILES = N * GH * GW
    NUM_GROUPS = triton.cdiv(C_IN, GROUP_SIZE_C)
    ALL_ONES_MASK = (1 << NUM_GROUPS) - 1

    # --- Stage 1: tile screening ---
    if kernel_size == 1:
        BM = BH * BW
        tile_screen_1x1_kernel[(N_TILES,)](
            x_f16,
            tile_alive_buf,
            N, C_IN, H_IN, W_IN, GH, GW,
            BLOCK_H=BH, BLOCK_W=BW, BLOCK_M=BM,
            THRESHOLD=threshold,
        )
    else:
        rf_h = (BH - 1) * stride + kernel_size
        rf_w = (BW - 1) * stride + kernel_size
        RF_SIZE = triton.next_power_of_2(max(rf_h * rf_w, 1))

        tile_screen_kernel[(N_TILES,)](
            x_f16,
            tile_alive_buf,
            N, C_IN, H_IN, W_IN, GH, GW,
            BLOCK_H=BH, BLOCK_W=BW,
            KERNEL_SIZE=kernel_size, STRIDE=stride, PAD=padding,
            RF_SIZE=RF_SIZE, THRESHOLD=threshold,
            C_BATCH=1,
        )

    # --- Stage 2: group bitmask (only meaningful for alive tiles) ---
    if kernel_size == 1:
        BM = BH * BW
        group_bitmask_1x1_kernel[(N_TILES,)](
            x_f16,
            tile_alive_buf,
            ag_mask_buf,
            tile_class_buf,
            N, C_IN, H_IN, W_IN, GH, GW,
            BLOCK_H=BH, BLOCK_W=BW, BLOCK_M=BM,
            GROUP_SIZE_C=GROUP_SIZE_C, NUM_GROUPS=NUM_GROUPS,
            THRESHOLD=threshold, ALL_ONES_MASK=ALL_ONES_MASK,
        )
    else:
        rf_h = (BH - 1) * stride + kernel_size
        rf_w = (BW - 1) * stride + kernel_size
        RF_SIZE = triton.next_power_of_2(max(rf_h * rf_w, 1))

        group_bitmask_kernel[(N_TILES,)](
            x_f16,
            tile_alive_buf,
            ag_mask_buf,
            tile_class_buf,
            N, C_IN, H_IN, W_IN, H_OUT, W_OUT, GH, GW,
            BLOCK_H=BH, BLOCK_W=BW,
            KERNEL_SIZE=kernel_size, STRIDE=stride, PAD=padding,
            GROUP_SIZE_C=GROUP_SIZE_C, NUM_GROUPS=NUM_GROUPS,
            MAX_AG=NUM_GROUPS,
            RF_SIZE=RF_SIZE, THRESHOLD=threshold,
            ALL_ONES_MASK=ALL_ONES_MASK,
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
# Compute kernels — three-path per tile:
#   TILE_ZERO (mask==0): bias-only early return
#   TILE_DENSEISH (all groups active): unconditional group iteration
#   TILE_SPARSE: bitmask-gated iteration
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

    # --- Path 1: Zero tile early return ---
    if ag_mask == 0:
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
        oa = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
        tl.store(oa, acc, mask=m_mask[:, None] & n_mask[None, :])
        return

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # --- Path 2: Dense-ish tile (all groups active) — no bit-check ---
    if ag_mask == ALL_ONES_MASK:
        for g in range(NUM_GROUPS):
            cin_start = g * GROUP_SIZE_C
            offs_k = cin_start + tl.arange(0, GROUP_SIZE_C)
            k_mask = offs_k < C_IN
            x_addrs = x_ptr + (n_idx * C_IN + offs_k[None, :]) * HW_IN + out_h[:, None] * W_IN + out_w[:, None]
            x_tile = tl.load(x_addrs, mask=k_mask[None, :] & m_mask[:, None], other=0.0).to(tl.float16)
            w_addrs = w_cl_ptr + offs_n[None, :] * C_IN + offs_k[:, None]
            w_tile = tl.load(w_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
            acc += tl.dot(x_tile, w_tile)
    else:
        # --- Path 3: Sparse tile — bitmask-gated ---
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
        for g in range(NUM_GROUPS):
            cin_start = g * GROUP_SIZE_C
            offs_k = cin_start + tl.arange(0, GROUP_SIZE_C)
            k_mask = offs_k < C_IN
            x_addrs = x_ptr + (n_idx * C_IN + offs_k[None, :]) * HW_IN + out_h[:, None] * W_IN + out_w[:, None]
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

    # Zero tile
    if ag_mask == 0:
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
        oa = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
        tl.store(oa, acc, mask=m_mask[:, None] & n_mask[None, :])
        return

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Dense-ish tile: skip per-group bit check
    if ag_mask == ALL_ONES_MASK:
        for g in range(NUM_GROUPS):
            cin_start = g * GROUP_SIZE_C
            offs_k = cin_start + tl.arange(0, GROUP_SIZE_C)
            k_mask = offs_k < C_IN
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
    else:
        # Sparse tile: bitmask-gated
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
        for g in range(NUM_GROUPS):
            cin_start = g * GROUP_SIZE_C
            offs_k = cin_start + tl.arange(0, GROUP_SIZE_C)
            k_mask = offs_k < C_IN
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
        for g in range(NUM_GROUPS):
            cin_start = g * GROUP_SIZE_C
            offs_k = cin_start + tl.arange(0, GROUP_SIZE_C)
            k_mask = offs_k < C_IN
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
        for g in range(NUM_GROUPS):
            cin_start = g * GROUP_SIZE_C
            offs_k = cin_start + tl.arange(0, GROUP_SIZE_C)
            k_mask = offs_k < C_IN
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
    # New two-stage buffers
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
        alive_count = sparse_count + denseish_count

        if return_tile_stats:
            tile_stats = {
                'stage1_zero_tiles': zero_count,
                'stage2_tiles': alive_count,
                'zero_tiles': zero_count,
                'sparse_tiles': sparse_count,
                'denseish_tiles': denseish_count,
                'total_tiles': N_TILES,
                'prescan_mode': 'two_stage',
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