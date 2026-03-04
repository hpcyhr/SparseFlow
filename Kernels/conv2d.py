"""
SparseFlow Conv2d Triton Kernels — v12.1 Per-Tile Channel Compaction

═══════════════════════════════════════════════════════════════════════
Architecture:
  Stage-1  (3-step GPU pipeline):
    1. prescan_count_kernel: grid=(N_TILES,), scans C_IN channels per
       tile → tile_counts[tile_id]
    2. torch.cumsum on GPU → tile_ptr (CSR row-pointer)
    3. prescan_write_kernel: grid=(N_TILES,), writes active Cin indices
       into tile_cin[] at positions given by tile_ptr

  Stage-2  (per-tile sparse GEMM):
    Grid = (N_TILES, cdiv(C_OUT, BLOCK_N))
    Each program reads its active Cin list from tile_ptr/tile_cin,
    iterates in BLOCK_K chunks up to MAX_K_ITERS (constexpr),
    loads x patches + weight slices via indirect indexing,
    accumulates via tl.dot (Tensor Core).

  Supports 3×3 and 1×1 kernels.

Tile strategy (改动 B — dynamic):
  _select_block_sizes adapts BH/BW based on H*W AND C_IN.

Stability (改动 E):
  All kernel pointer arithmetic uses explicit masks.
  All reshape/flatten assumes contiguous() has been called.
  All Triton for-loops use constexpr bounds.

Triton 2.x safe: no continue, no 0-d tensors, no scalar .to(),
  all range() arguments are tl.constexpr.
═══════════════════════════════════════════════════════════════════════
"""

import torch
import triton
import triton.language as tl

# ═══════════════════════════════════════════════════════════════════════
# Dynamic tile config (改动 B)
# ═══════════════════════════════════════════════════════════════════════

def _select_block_sizes(H, W, C_IN, C_OUT, kernel_size, N):
    """
    Dynamic block selection based on spatial size, channel count, batch.

    H>=56 with large C_IN: BH=16,BW=16 → BLOCK_M=256, GH=4,GW=4 (16 tiles)
    H>=56 with small C_IN: BH=8,BW=16  → BLOCK_M=128, GH=7,GW=4 (28 tiles)
    H>=28:                 BH=8,BW=8   → BLOCK_M=64
    Otherwise:             BH=4,BW=4   → BLOCK_M=16

    Tensor Core: BLOCK_M, BLOCK_N, BLOCK_K all >= 16, powers of 2.
    """
    pixels = H * W

    if pixels >= 3136:              # H*W ≥ 56*56
        if C_IN > 64:
            BH, BW = 16, 16        # BLOCK_M = 256, fewer tiles
        else:
            BH, BW = 8, 16         # BLOCK_M = 128, finer granularity
    elif pixels >= 784:             # H*W ≥ 28*28
        BH, BW = 8, 8              # BLOCK_M = 64
    elif min(H, W) >= 8:
        BH, BW = 4, 4              # BLOCK_M = 16
    else:
        BH, BW = 4, 4

    BLOCK_M = BH * BW

    # C_OUT tile — register-safe
    high_pressure = (N >= 256)
    if high_pressure:
        BLOCK_N = 32 if C_OUT >= 32 else 16
    else:
        if C_OUT >= 128:
            BLOCK_N = 64
        elif C_OUT >= 32:
            BLOCK_N = 32
        else:
            BLOCK_N = 16

    BLOCK_K = 16
    return BH, BW, BLOCK_M, BLOCK_N, BLOCK_K


# ═══════════════════════════════════════════════════════════════════════
# Stage-1 Step 1: Count active channels per tile
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def prescan_count_kernel(
    x_ptr,              # [N, C_IN, H, W] fp16 contiguous
    counts_ptr,         # [N_TILES] int32 output
    N_val, C_IN, H, W,
    GH, GW,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    MAX_C: tl.constexpr,      # >= C_IN, must be constexpr for loop
    THRESHOLD: tl.constexpr,
):
    """
    For each tile (n, gh, gw), count how many input channels are active.
    Grid: (N_TILES,) where N_TILES = N * GH * GW

    MAX_C is a constexpr upper bound on C_IN (next power of 2).
    The loop runs MAX_C iterations; channels >= C_IN are skipped via mask.
    """
    tile_id = tl.program_id(0)
    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles:
        return

    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH

    h_start = gh_idx * BLOCK_H
    w_start = gw_idx * BLOCK_W

    offs_h = h_start + tl.arange(0, BLOCK_H)
    offs_w = w_start + tl.arange(0, BLOCK_W)
    hh = offs_h[:, None]
    ww = offs_w[None, :]
    hw_mask = (hh < H) & (ww < W)

    HW = H * W
    count = 0

    for c_idx in range(MAX_C):
        if c_idx < C_IN:
            base = (n_idx * C_IN + c_idx) * HW
            vals = tl.load(x_ptr + base + hh * W + ww,
                           mask=hw_mask, other=0.0)
            is_nz = tl.max(tl.abs(vals)) > THRESHOLD
            count += is_nz.to(tl.int32)

    tl.store(counts_ptr + tile_id, count)


# ═══════════════════════════════════════════════════════════════════════
# Stage-1 Step 3: Write active channel indices per tile
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def prescan_write_kernel(
    x_ptr,              # [N, C_IN, H, W] fp16 contiguous
    tile_ptr_data,      # [N_TILES + 1] int32 (prefix sum)
    tile_cin_ptr,       # [total_active] int32 output
    total_active,       # scalar: total active count
    N_val, C_IN, H, W,
    GH, GW,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    MAX_C: tl.constexpr,      # >= C_IN, constexpr for loop
    THRESHOLD: tl.constexpr,
):
    """
    For each tile, write its active Cin indices into tile_cin.
    Grid: (N_TILES,)
    """
    tile_id = tl.program_id(0)
    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles:
        return

    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH

    h_start = gh_idx * BLOCK_H
    w_start = gw_idx * BLOCK_W

    offs_h = h_start + tl.arange(0, BLOCK_H)
    offs_w = w_start + tl.arange(0, BLOCK_W)
    hh = offs_h[:, None]
    ww = offs_w[None, :]
    hw_mask = (hh < H) & (ww < W)

    HW = H * W

    write_pos = tl.load(tile_ptr_data + tile_id)
    idx = 0

    for c_idx in range(MAX_C):
        if c_idx < C_IN:
            base = (n_idx * C_IN + c_idx) * HW
            vals = tl.load(x_ptr + base + hh * W + ww,
                           mask=hw_mask, other=0.0)
            is_nz = tl.max(tl.abs(vals)) > THRESHOLD

            if is_nz:
                out_pos = write_pos + idx
                if out_pos < total_active:
                    tl.store(tile_cin_ptr + out_pos, c_idx)
                idx += 1


# ═══════════════════════════════════════════════════════════════════════
# Stage-1 Python orchestration
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _build_tile_csr(x_f16, N, C_IN, H, W, BH, BW, GH, GW, threshold,
                    counts_buf=None):
    """
    Build per-tile CSR structure entirely on GPU.
    One CPU sync after cumsum to read total_active for allocation.

    Returns:
        tile_ptr: [N_TILES + 1] int32 — prefix sum
        tile_cin: [total_active] int32 — concatenated active Cin indices
        total_active: int
    """
    device = x_f16.device
    N_TILES = N * GH * GW

    # Constexpr upper bound for channel loop
    MAX_C = triton.next_power_of_2(max(C_IN, 1))

    # Step 1: Count active channels per tile
    if counts_buf is not None and counts_buf.numel() >= N_TILES:
        tile_counts = counts_buf[:N_TILES]
    else:
        tile_counts = torch.empty(N_TILES, dtype=torch.int32, device=device)

    prescan_count_kernel[(N_TILES,)](
        x_f16, tile_counts,
        N, C_IN, H, W, GH, GW,
        BLOCK_H=BH, BLOCK_W=BW,
        MAX_C=MAX_C,
        THRESHOLD=threshold,
    )

    # Step 2: Prefix sum on GPU → tile_ptr
    cumsum = torch.cumsum(tile_counts, dim=0, dtype=torch.int32)
    tile_ptr = torch.empty(N_TILES + 1, dtype=torch.int32, device=device)
    tile_ptr[0] = 0
    tile_ptr[1:] = cumsum

    # Need total_active to allocate tile_cin — one sync
    torch.cuda.synchronize(device)
    total_active = int(tile_ptr[N_TILES].item())

    if total_active == 0:
        tile_cin = torch.empty(0, dtype=torch.int32, device=device)
        return tile_ptr, tile_cin, 0

    # Step 3: Write active channel indices
    tile_cin = torch.empty(total_active, dtype=torch.int32, device=device)

    prescan_write_kernel[(N_TILES,)](
        x_f16, tile_ptr, tile_cin, total_active,
        N, C_IN, H, W, GH, GW,
        BLOCK_H=BH, BLOCK_W=BW,
        MAX_C=MAX_C,
        THRESHOLD=threshold,
    )

    return tile_ptr, tile_cin, total_active


# ═══════════════════════════════════════════════════════════════════════
# Stage-2: Per-tile sparse 3×3 conv (CSR + indirect weight)
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def sparse_conv3x3_csr_kernel(
    x_ptr,              # [N, C_IN, H, W] fp16 contiguous
    w_ptr,              # [C_OUT, C_IN, 3, 3] fp16 contiguous
    bias_ptr,           # [C_OUT] fp32 or dummy
    tile_ptr_data,      # [N_TILES + 1] int32
    tile_cin_ptr,       # [total_active] int32
    y_ptr,              # [N, C_OUT, H, W] fp32
    N_val, C_IN, C_OUT, H, W,
    GH, GW,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,   # BH * BW
    BLOCK_N: tl.constexpr,   # C_OUT chunk
    BLOCK_K: tl.constexpr,   # K-dim chunk for tl.dot
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    MAX_K_ITERS: tl.constexpr,  # = cdiv(C_IN, BLOCK_K), constexpr loop bound
):
    """
    Per-tile sparse 3×3 conv.
    Grid: (N_TILES, cdiv(C_OUT, BLOCK_N))

    MAX_K_ITERS is the upper bound on K iterations. Tiles with fewer
    active channels simply mask out the excess iterations (loads return 0,
    tl.dot accumulates nothing).
    """
    tile_id = tl.program_id(0)
    pid_cout = tl.program_id(1)

    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles:
        return

    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH

    c_out_start = pid_cout * BLOCK_N
    offs_n = c_out_start + tl.arange(0, BLOCK_N)
    n_mask = offs_n < C_OUT

    offs_m = tl.arange(0, BLOCK_M)
    tile_bh = offs_m // BLOCK_W
    tile_bw = offs_m % BLOCK_W
    tile_h = gh_idx * BLOCK_H + tile_bh
    tile_w = gw_idx * BLOCK_W + tile_bw
    m_mask = (tile_h < H) & (tile_w < W)

    HW = H * W
    H_max = H - 1
    W_max = W - 1

    tile_start = tl.load(tile_ptr_data + tile_id)
    tile_end = tl.load(tile_ptr_data + tile_id + 1)
    active_K = tile_end - tile_start

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_iter in range(MAX_K_ITERS):
        k_start = k_iter * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < active_K

        cin_global_idx = tl.load(
            tile_cin_ptr + tile_start + offs_k,
            mask=k_mask, other=0)

        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                raw_h = tile_h + (kh - 1)
                raw_w = tile_w + (kw - 1)
                h_ok = (raw_h >= 0) & (raw_h < H)
                w_ok = (raw_w >= 0) & (raw_w < W)
                safe_h = tl.minimum(tl.maximum(raw_h, 0), H_max)
                safe_w = tl.minimum(tl.maximum(raw_w, 0), W_max)

                # x[n, cin, h, w] — [BLOCK_M, BLOCK_K]
                x_addrs = (x_ptr
                           + (n_idx * C_IN + cin_global_idx[None, :]) * HW
                           + safe_h[:, None] * W
                           + safe_w[:, None])
                x_load_mask = (k_mask[None, :] & m_mask[:, None]
                               & h_ok[:, None] & w_ok[:, None])
                x_tile = tl.load(x_addrs, mask=x_load_mask, other=0.0)
                x_tile = x_tile.to(tl.float16)

                # w[cout, cin, kh, kw] — [BLOCK_K, BLOCK_N]
                # Layout: [C_OUT, C_IN, 3, 3] → stride co=C_IN*9, ci=9
                w_addrs = (w_ptr
                           + offs_n[None, :] * (C_IN * 9)
                           + cin_global_idx[:, None] * 9
                           + kh * 3 + kw)
                w_load_mask = k_mask[:, None] & n_mask[None, :]
                w_tile = tl.load(w_addrs, mask=w_load_mask, other=0.0)
                w_tile = w_tile.to(tl.float16)

                acc += tl.dot(x_tile, w_tile)

    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        acc += bias_vals[None, :]

    out_addrs = (y_ptr
                 + (n_idx * C_OUT + offs_n[None, :]) * HW
                 + tile_h[:, None] * W
                 + tile_w[:, None])
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_addrs, acc, mask=out_mask)


# ═══════════════════════════════════════════════════════════════════════
# Stage-2: Per-tile sparse 1×1 conv (CSR + indirect weight)
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def sparse_conv1x1_csr_kernel(
    x_ptr,              # [N, C_IN, H, W] fp16 contiguous
    w_ptr,              # [C_OUT, C_IN] fp16 contiguous (reshaped from [C_OUT,C_IN,1,1])
    bias_ptr,           # [C_OUT] fp32 or dummy
    tile_ptr_data,      # [N_TILES + 1] int32
    tile_cin_ptr,       # [total_active] int32
    y_ptr,              # [N, C_OUT, H, W] fp32
    N_val, C_IN, C_OUT, H, W,
    GH, GW,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    MAX_K_ITERS: tl.constexpr,
):
    """Per-tile sparse 1×1 conv. Same CSR, no spatial filter loop."""
    tile_id = tl.program_id(0)
    pid_cout = tl.program_id(1)

    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles:
        return

    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH

    c_out_start = pid_cout * BLOCK_N
    offs_n = c_out_start + tl.arange(0, BLOCK_N)
    n_mask = offs_n < C_OUT

    offs_m = tl.arange(0, BLOCK_M)
    tile_bh = offs_m // BLOCK_W
    tile_bw = offs_m % BLOCK_W
    tile_h = gh_idx * BLOCK_H + tile_bh
    tile_w = gw_idx * BLOCK_W + tile_bw
    m_mask = (tile_h < H) & (tile_w < W)
    safe_h = tl.minimum(tile_h, H - 1)
    safe_w = tl.minimum(tile_w, W - 1)

    HW = H * W

    tile_start = tl.load(tile_ptr_data + tile_id)
    tile_end = tl.load(tile_ptr_data + tile_id + 1)
    active_K = tile_end - tile_start

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_iter in range(MAX_K_ITERS):
        k_start = k_iter * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < active_K

        cin_global_idx = tl.load(
            tile_cin_ptr + tile_start + offs_k,
            mask=k_mask, other=0)

        # x[n, cin, h, w] — [BLOCK_M, BLOCK_K]
        x_addrs = (x_ptr
                   + (n_idx * C_IN + cin_global_idx[None, :]) * HW
                   + safe_h[:, None] * W
                   + safe_w[:, None])
        x_load_mask = k_mask[None, :] & m_mask[:, None]
        x_tile = tl.load(x_addrs, mask=x_load_mask, other=0.0)
        x_tile = x_tile.to(tl.float16)

        # w[cout, cin] — [BLOCK_K, BLOCK_N]
        w_addrs = (w_ptr
                   + offs_n[None, :] * C_IN
                   + cin_global_idx[:, None])
        w_load_mask = k_mask[:, None] & n_mask[None, :]
        w_tile = tl.load(w_addrs, mask=w_load_mask, other=0.0)
        w_tile = w_tile.to(tl.float16)

        acc += tl.dot(x_tile, w_tile)

    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        acc += bias_vals[None, :]

    out_addrs = (y_ptr
                 + (n_idx * C_OUT + offs_n[None, :]) * HW
                 + tile_h[:, None] * W
                 + tile_w[:, None])
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_addrs, acc, mask=out_mask)


# ═══════════════════════════════════════════════════════════════════════
# Legacy kernels — retained for Diy/test scripts backward compat
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def prescan_kernel(
    x_ptr, flags_ptr, N, C, H, W, GRID_H, GRID_W,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
    """Legacy per-block prescan."""
    pid = tl.program_id(0)
    gw = pid % GRID_W; tmp = pid // GRID_W
    gh = tmp % GRID_H; tmp2 = tmp // GRID_H
    c = tmp2 % C; n = tmp2 // C
    hh = (gh * BLOCK_H + tl.arange(0, BLOCK_H))[:, None]
    ww = (gw * BLOCK_W + tl.arange(0, BLOCK_W))[None, :]
    mask = (hh < H) & (ww < W)
    base = (n * C + c) * H
    val = tl.load(x_ptr + (base + hh) * W + ww, mask=mask, other=0.0)
    is_nz = tl.max(tl.abs(val)) > THRESHOLD
    tl.store(flags_ptr + pid, is_nz.to(tl.int32))


@triton.jit
def dense_conv3x3_kernel(
    x_ptr, y_ptr, N, C, H, W, GRID_H, GRID_W,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    """Dense 3×3 box filter baseline."""
    pid = tl.program_id(0)
    total = N * C * GRID_H * GRID_W
    if pid >= total:
        return
    gw = pid % GRID_W; tmp = pid // GRID_W
    gh = tmp % GRID_H; tmp2 = tmp // GRID_H
    c = tmp2 % C; n = tmp2 // C
    hh = (gh * BLOCK_H + tl.arange(0, BLOCK_H))[:, None]
    ww = (gw * BLOCK_W + tl.arange(0, BLOCK_W))[None, :]
    mask = (hh < H) & (ww < W)
    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    for kh in range(-1, 2):
        for kw in range(-1, 2):
            h_idx = hh + kh; w_idx = ww + kw
            m = mask & (h_idx >= 0) & (h_idx < H) & (w_idx >= 0) & (w_idx < W)
            acc += tl.load(x_ptr + ((n*C+c)*H+h_idx)*W+w_idx, mask=m, other=0.0)
    tl.store(y_ptr + ((n*C+c)*H+hh)*W+ww, acc, mask=mask)


# ═══════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════

def sparse_conv2d_forward(x, weight, bias, block_size=None,
                          kernel_size=3, threshold=1e-6,
                          counts_buf=None, return_ms=False):
    """
    Per-tile channel-compacted sparse conv2d.

    Args:
        x: [N, C_IN, H, W] input
        weight: [C_OUT, C_IN, K, K]
        bias: [C_OUT] or None
        block_size: None (dynamic) or int (legacy hint)
        kernel_size: 3 or 1
        threshold: zero-detection threshold
        counts_buf: optional pre-allocated int32 buffer for tile_counts
            (numel >= N*GH*GW). Avoids per-call allocation.
        return_ms: if True, wraps Stage-2 in CUDA events for timing.
            If False (default), no sync, no timing overhead.

    Returns:
        y: [N, C_OUT, H, W] fp32
        sparse_ms: Stage-2 time in ms (0.0 when return_ms=False)
    """
    N, C_IN, H, W = x.shape
    C_OUT = weight.shape[0]
    device = x.device

    # ── Dynamic block selection ──
    pixels = H * W
    if block_size is None or pixels >= 3136:
        BH, BW, BLOCK_M, BLOCK_N, BLOCK_K = _select_block_sizes(
            H, W, C_IN, C_OUT, kernel_size, N)
    else:
        bs = max(block_size, 4)
        BH, BW = bs, bs
        BLOCK_M = BH * BW
        BLOCK_N = 32 if C_OUT >= 32 else 16
        BLOCK_K = 16

    GH = triton.cdiv(H, BH)
    GW = triton.cdiv(W, BW)
    N_TILES = N * GH * GW

    x_f16 = x.half().contiguous()
    w_f16 = weight.half().contiguous()

    # ── Stage-1: Build per-tile CSR ──
    tile_ptr, tile_cin, total_active = _build_tile_csr(
        x_f16, N, C_IN, H, W, BH, BW, GH, GW, threshold,
        counts_buf=counts_buf)

    # All-zero fast path
    if total_active == 0:
        y = torch.zeros(N, C_OUT, H, W, dtype=torch.float32, device=device)
        if bias is not None:
            y += bias.float().view(1, -1, 1, 1)
        return y, 0.0

    # ── Stage-2: Per-tile sparse conv ──
    has_bias = bias is not None
    bias_f32 = (bias.float().contiguous() if has_bias
                else torch.empty(1, device=device))
    y = torch.zeros(N, C_OUT, H, W, dtype=torch.float32, device=device)

    grid_cout = triton.cdiv(C_OUT, BLOCK_N)

    # MAX_K_ITERS: constexpr upper bound for the K-dim loop inside Stage-2.
    # This is cdiv(C_IN, BLOCK_K) — the worst case where all channels active.
    MAX_K_ITERS = triton.cdiv(C_IN, BLOCK_K)

    sparse_ms = 0.0
    if return_ms:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()

    if kernel_size == 3:
        sparse_conv3x3_csr_kernel[(N_TILES, grid_cout)](
            x_f16, w_f16, bias_f32,
            tile_ptr, tile_cin,
            y,
            N, C_IN, C_OUT, H, W,
            GH, GW,
            HAS_BIAS=has_bias,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            BLOCK_H=BH, BLOCK_W=BW,
            MAX_K_ITERS=MAX_K_ITERS,
        )
    else:  # 1×1
        w_1x1 = w_f16.reshape(C_OUT, C_IN).contiguous()
        sparse_conv1x1_csr_kernel[(N_TILES, grid_cout)](
            x_f16, w_1x1, bias_f32,
            tile_ptr, tile_cin,
            y,
            N, C_IN, C_OUT, H, W,
            GH, GW,
            HAS_BIAS=has_bias,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            BLOCK_H=BH, BLOCK_W=BW,
            MAX_K_ITERS=MAX_K_ITERS,
        )

    if return_ms:
        end_evt.record()
        torch.cuda.synchronize(device)
        sparse_ms = start_evt.elapsed_time(end_evt)

    return y, sparse_ms