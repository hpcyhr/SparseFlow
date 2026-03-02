"""
SparseFlow Conv2d Triton Kernels — v10.3 Metadata-Driven (production)

Architecture:
  Stage-1:   Prescan → flags [N, C_IN, GH, GW] int32
  Stage-1.5: Active channel detection + weight gather (vectorized torch)
  Stage-2:   Branch-free Tensor Core GEMM over shrunk K dimension

Design decisions and safety guarantees:

  1. PADDING SAFETY: active_cin padded with 0 (valid channel index).
     Why this is safe (triple protection):
       a) k_mask = (offs_k < active_K_raw) → padding lanes masked
       b) x_tile load uses k_mask → returns 0.0 for padding
       c) w_gathered has zero rows for padding positions
     Even if Triton speculatively executes the masked path, channel 0
     is always a valid address (no segfault), and the product x*w = 0*w = 0.

  2. MEMORY SAFETY: Spatial coords clamped to [0, H-1] / [0, W-1] BEFORE
     address computation. Real boundary handling via x_valid mask (→ other=0.0).
     Addresses are ALWAYS within the x tensor regardless of kh/kw shift.

  3. WEIGHT ALIGNMENT: w_gathered shape = [C_OUT, AK_PAD, 3, 3] for 3×3,
     [C_OUT, AK_PAD] for 1×1. Python computes strides and passes to kernel.
     Kernel uses these strides directly — no recomputation.

  4. T=16 MEMORY: BLOCK_M reduced from 64 to 32 (BH=BW=4 for spatial≥8).
     This halves per-thread register pressure: acc is [32, BLOCK_N] instead
     of [64, BLOCK_N]. Reduces register spilling → avoids segfault at T=16.
     GH/GW passed as kernel args (no tl.cdiv inside kernel).
     NUM_K_ITERS is runtime int (not constexpr) to avoid recompilation per
     sparsity level.

Triton 2.x safe: no continue, no 0-d tensors, no scalar .to()
"""

import torch
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════
# Stage-1: Prescan — per-block zero detection
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def prescan_kernel(
    x_ptr, flags_ptr,
    N, C, H, W,
    GRID_H, GRID_W,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
    pid = tl.program_id(0)
    gw = pid % GRID_W
    tmp = pid // GRID_W
    gh = tmp % GRID_H
    tmp2 = tmp // GRID_H
    c = tmp2 % C
    n = tmp2 // C

    offs_h = gh * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = gw * BLOCK_W + tl.arange(0, BLOCK_W)
    hh = offs_h[:, None]
    ww = offs_w[None, :]
    mask = (hh < H) & (ww < W)

    base = (n * C + c) * H
    val = tl.load(x_ptr + (base + hh) * W + ww, mask=mask, other=0.0)
    is_nz = tl.max(tl.abs(val)) > THRESHOLD
    tl.store(flags_ptr + pid, is_nz.to(tl.int32))


# ═══════════════════════════════════════════════════════════════════════
# Stage-2a: Sparse 3×3 Conv — branch-free GEMM
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def sparse_conv3x3_v10_kernel(
    x_ptr,              # [N, C_IN, H, W] fp16
    w_gathered_ptr,     # [C_OUT, AK_PAD, 3, 3] fp16 contiguous
    bias_ptr,           # [C_OUT] fp32 or dummy
    active_cin_ptr,     # [AK_PAD] int32
    y_ptr,              # [N, C_OUT, H, W] fp32
    N_val, C_IN, C_OUT, H, W,
    GH, GW,             # spatial grid dims (from Python, no tl.cdiv)
    active_K_raw,       # real active count (for k_mask)
    num_k_iters,        # = ceil(AK_PAD / BLOCK_K), runtime int
    w_stride_co,        # = AK_PAD * 9
    w_stride_k,         # = 9
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Grid: (N * GH * GW, cdiv(C_OUT, BLOCK_N))
    Branch-free K-loop over pre-filtered active channels.
    All addresses clamped; k_mask guards padding entries.
    """
    pid_spatial = tl.program_id(0)
    pid_cout = tl.program_id(1)

    # Decode spatial position using Python-provided GH, GW
    gw_idx = pid_spatial % GW
    tmp = pid_spatial // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH

    # C_OUT tile
    c_out_start = pid_cout * BLOCK_N
    offs_n = c_out_start + tl.arange(0, BLOCK_N)
    n_mask = offs_n < C_OUT

    # Spatial pixel tile
    offs_m = tl.arange(0, BLOCK_M)
    tile_bh = offs_m // BLOCK_W
    tile_bw = offs_m % BLOCK_W
    tile_h = gh_idx * BLOCK_H + tile_bh
    tile_w = gw_idx * BLOCK_W + tile_bw
    m_mask = (tile_h < H) & (tile_w < W)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    HW = H * W
    H_max = H - 1
    W_max = W - 1

    # ── K loop (runtime bound — no recompilation per sparsity level) ──
    for k_iter in range(num_k_iters):
        k_start = k_iter * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < active_K_raw  # masks padding entries

        # Channel indices (padding slots → 0, a valid index; masked below)
        cin_idx = tl.load(active_cin_ptr + offs_k, mask=k_mask, other=0)

        # 9 sub-GEMMs for (kh, kw) ∈ {0,1,2}²
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                # Shifted spatial coordinates
                raw_h = tile_h + (kh - 1)  # may be -1 .. H
                raw_w = tile_w + (kw - 1)  # may be -1 .. W

                # Validity mask (real boundary logic)
                h_ok = (raw_h >= 0) & (raw_h < H)
                w_ok = (raw_w >= 0) & (raw_w < W)

                # CLAMP to safe range → address always within buffer
                safe_h = tl.minimum(tl.maximum(raw_h, 0), H_max)
                safe_w = tl.minimum(tl.maximum(raw_w, 0), W_max)

                # X_tile[BLOCK_M, BLOCK_K]
                x_addrs = (x_ptr
                           + (n_idx * C_IN + cin_idx[None, :]) * HW
                           + safe_h[:, None] * W
                           + safe_w[:, None])
                x_load_mask = (k_mask[None, :] & m_mask[:, None]
                               & h_ok[:, None] & w_ok[:, None])
                x_tile = tl.load(x_addrs, mask=x_load_mask, other=0.0)
                x_tile = x_tile.to(tl.float16)

                # W_tile[BLOCK_K, BLOCK_N]
                w_addrs = (w_gathered_ptr
                           + offs_n[None, :] * w_stride_co
                           + offs_k[:, None] * w_stride_k
                           + kh * 3 + kw)
                w_load_mask = k_mask[:, None] & n_mask[None, :]
                w_tile = tl.load(w_addrs, mask=w_load_mask, other=0.0)
                w_tile = w_tile.to(tl.float16)

                acc += tl.dot(x_tile, w_tile)

    # Bias
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        acc += bias_vals[None, :]

    # Store fp32
    out_addrs = (y_ptr
                 + (n_idx * C_OUT + offs_n[None, :]) * HW
                 + tile_h[:, None] * W
                 + tile_w[:, None])
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_addrs, acc, mask=out_mask)


# ═══════════════════════════════════════════════════════════════════════
# Stage-2b: Sparse 1×1 Conv — branch-free GEMM
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def sparse_conv1x1_v10_kernel(
    x_ptr,              # [N, C_IN, H, W] fp16
    w_gathered_ptr,     # [C_OUT, AK_PAD] fp16 contiguous
    bias_ptr,           # [C_OUT] fp32 or dummy
    active_cin_ptr,     # [AK_PAD] int32
    y_ptr,              # [N, C_OUT, H, W] fp32
    N_val, C_IN, C_OUT, H, W,
    GH, GW,
    active_K_raw,
    num_k_iters,
    w_stride_co,        # = AK_PAD
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Grid: (N * GH * GW, cdiv(C_OUT, BLOCK_N))
    Branch-free 1×1 GEMM. Reads directly from NCHW x_ptr.
    """
    pid_spatial = tl.program_id(0)
    pid_cout = tl.program_id(1)

    gw_idx = pid_spatial % GW
    tmp = pid_spatial // GW
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

    # Clamp spatial for safe addressing (1×1 has no shift, but tile edges
    # may exceed H/W when not evenly divisible)
    safe_h = tl.minimum(tile_h, H - 1)
    safe_w = tl.minimum(tile_w, W - 1)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    HW = H * W

    for k_iter in range(num_k_iters):
        k_start = k_iter * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < active_K_raw

        cin_idx = tl.load(active_cin_ptr + offs_k, mask=k_mask, other=0)

        # X_tile[BLOCK_M, BLOCK_K]
        x_addrs = (x_ptr
                   + (n_idx * C_IN + cin_idx[None, :]) * HW
                   + safe_h[:, None] * W
                   + safe_w[:, None])
        x_load_mask = k_mask[None, :] & m_mask[:, None]
        x_tile = tl.load(x_addrs, mask=x_load_mask, other=0.0)
        x_tile = x_tile.to(tl.float16)

        # W_tile[BLOCK_K, BLOCK_N]
        w_addrs = (w_gathered_ptr
                   + offs_n[None, :] * w_stride_co
                   + offs_k[:, None])
        w_load_mask = k_mask[:, None] & n_mask[None, :]
        w_tile = tl.load(w_addrs, mask=w_load_mask, other=0.0)
        w_tile = w_tile.to(tl.float16)

        acc += tl.dot(x_tile, w_tile)

    # Bias
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        acc += bias_vals[None, :]

    # Store fp32
    out_addrs = (y_ptr
                 + (n_idx * C_OUT + offs_n[None, :]) * HW
                 + tile_h[:, None] * W
                 + tile_w[:, None])
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_addrs, acc, mask=out_mask)


# ═══════════════════════════════════════════════════════════════════════
# Dense Conv 3×3 (baseline reference)
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def dense_conv3x3_kernel(
    x_ptr, y_ptr, N, C, H, W, GRID_H, GRID_W,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
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
# Tile config — T=16 optimized
# ═══════════════════════════════════════════════════════════════════════

def _select_block_sizes(H, W, C_IN, C_OUT, kernel_size):
    """
    All tl.dot dims: powers-of-2, >= 16.

    T=16 optimization: BLOCK_M = 32 (BH=BW=4) reduces register pressure.
    The 3×3 kernel holds acc[BLOCK_M, BLOCK_N] + 2 tiles [BLOCK_M, BLOCK_K]
    + [BLOCK_K, BLOCK_N] in registers. At BLOCK_M=64, BLOCK_N=64 this is
    64*64*4 = 16KB for acc alone, causing spills on A100 (255 registers/thread,
    32 threads/warp → ~32KB total). BLOCK_M=32 halves this to 8KB.
    """
    # Spatial tile: BLOCK_M = BH * BW
    # Use 4×4=16 minimum (tl.dot requires ≥16), 4×8=32 for larger spatial
    spatial = min(H, W)
    if spatial >= 8:
        BH, BW = 4, 8      # BLOCK_M = 32
    else:
        BH, BW = 4, 4      # BLOCK_M = 16

    BLOCK_M = BH * BW

    # C_OUT tile
    if C_OUT >= 128:
        BLOCK_N = 64
    elif C_OUT >= 32:
        BLOCK_N = 32
    else:
        BLOCK_N = 16

    # C_IN tile
    BLOCK_K = 16

    return BH, BW, BLOCK_M, BLOCK_N, BLOCK_K


# ═══════════════════════════════════════════════════════════════════════
# Stage-1.5: Metadata build (vectorized torch, GPU)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _build_active_channels_3x3(flags_4d, N, C_IN, GH, GW, BLOCK_K, device):
    """
    Dilate flags by 3×3 (neighbor influence), reduce to global active set.

    Returns:
        active_cin:      [AK_PAD] int32 — padded with 0
        active_K_raw:    int — real active count
        active_K_padded: int — padded to multiple of BLOCK_K
    """
    f = flags_4d.float()
    # Dilate: pad spatial dims by 1, then max_pool 3×3 stride 1
    padded = torch.nn.functional.pad(f, (1, 1, 1, 1), value=0.0)
    dilated = torch.nn.functional.max_pool2d(padded, 3, stride=1, padding=0)
    # dilated: [N, C_IN, GH, GW]

    # Global reduction: channel active if ANY (n, gh, gw) flags it
    channel_active = (dilated > 0).any(dim=0).any(dim=-1).any(dim=-1)  # [C_IN]
    nz = channel_active.nonzero(as_tuple=False)  # [num_active, 1]
    active_cin = nz.reshape(-1).to(torch.int32)
    active_K_raw = int(active_cin.numel())

    if active_K_raw == 0:
        z = torch.zeros(BLOCK_K, dtype=torch.int32, device=device)
        return z, 0, BLOCK_K

    # Pad to next multiple of BLOCK_K
    active_K_padded = ((active_K_raw + BLOCK_K - 1) // BLOCK_K) * BLOCK_K
    if active_K_padded > active_K_raw:
        # Pad with 0. Safety:
        #   - k_mask = offs_k < active_K_raw → padding lanes masked
        #   - x_tile load returns 0.0 for masked lanes
        #   - w_gathered has zero weight for padding rows
        #   - Even if speculatively executed, channel 0 is valid memory
        pad_t = torch.zeros(active_K_padded - active_K_raw,
                            dtype=torch.int32, device=device)
        active_cin = torch.cat([active_cin, pad_t])

    return active_cin.contiguous(), active_K_raw, active_K_padded


@torch.no_grad()
def _build_active_channels_1x1(flags_4d, N, C_IN, GH, GW, BLOCK_K, device):
    """For 1×1: no dilation. Channel active if any (n, gh, gw) flags it."""
    channel_active = (flags_4d > 0).any(dim=0).any(dim=-1).any(dim=-1)
    nz = channel_active.nonzero(as_tuple=False)
    active_cin = nz.reshape(-1).to(torch.int32)
    active_K_raw = int(active_cin.numel())

    if active_K_raw == 0:
        z = torch.zeros(BLOCK_K, dtype=torch.int32, device=device)
        return z, 0, BLOCK_K

    active_K_padded = ((active_K_raw + BLOCK_K - 1) // BLOCK_K) * BLOCK_K
    if active_K_padded > active_K_raw:
        pad_t = torch.zeros(active_K_padded - active_K_raw,
                            dtype=torch.int32, device=device)
        active_cin = torch.cat([active_cin, pad_t])

    return active_cin.contiguous(), active_K_raw, active_K_padded


@torch.no_grad()
def _gather_weight_3x3(weight_f16, active_cin, active_K_raw, active_K_padded,
                        C_OUT):
    """
    [C_OUT, C_IN, 3, 3] → [C_OUT, AK_PAD, 3, 3]
    Padding rows are zero (k_mask + zero weight = zero contribution).
    """
    idx = active_cin[:active_K_raw].long()
    w_active = weight_f16[:, idx, :, :]  # [C_OUT, active_K_raw, 3, 3]

    if active_K_padded > active_K_raw:
        pad_size = active_K_padded - active_K_raw
        pad = torch.zeros(C_OUT, pad_size, 3, 3,
                          dtype=weight_f16.dtype, device=weight_f16.device)
        w_active = torch.cat([w_active, pad], dim=1)

    return w_active.contiguous()


@torch.no_grad()
def _gather_weight_1x1(weight_f16, active_cin, active_K_raw, active_K_padded,
                        C_OUT, C_IN):
    """[C_OUT, C_IN, 1, 1] → [C_OUT, AK_PAD]"""
    w_2d = weight_f16.reshape(C_OUT, C_IN)
    idx = active_cin[:active_K_raw].long()
    w_active = w_2d[:, idx]  # [C_OUT, active_K_raw]

    if active_K_padded > active_K_raw:
        pad_size = active_K_padded - active_K_raw
        pad = torch.zeros(C_OUT, pad_size,
                          dtype=weight_f16.dtype, device=weight_f16.device)
        w_active = torch.cat([w_active, pad], dim=1)

    return w_active.contiguous()


# ═══════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════

def sparse_conv2d_forward(x, weight, bias, block_size,
                          kernel_size=3, threshold=1e-6):
    """
    Three-stage sparse conv2d:
      Stage 1:   Prescan → flags
      Stage 1.5: Active channel metadata + weight gather
      Stage 2:   Branch-free Tensor Core GEMM over shrunk K

    Returns:
        y: [N, C_OUT, H, W] float32
        sparse_ms: Stage-2 kernel time in ms
    """
    N, C_IN, H, W = x.shape
    C_OUT = weight.shape[0]
    device = x.device

    BH, BW, BLOCK_M, BLOCK_N, BLOCK_K = _select_block_sizes(
        H, W, C_IN, C_OUT, kernel_size)
    GH = triton.cdiv(H, BH)
    GW = triton.cdiv(W, BW)

    x_f16 = x.half().contiguous()
    w_f16 = weight.half().contiguous()

    # ── Stage-1: Prescan ──
    total_blocks = N * C_IN * GH * GW
    flags = torch.empty(total_blocks, dtype=torch.int32, device=device)
    prescan_kernel[(total_blocks,)](
        x_f16, flags, N, C_IN, H, W, GH, GW,
        BLOCK_H=BH, BLOCK_W=BW, THRESHOLD=threshold,
    )
    torch.cuda.synchronize(device)

    # All-zero fast path
    if flags.sum().item() == 0:
        y = torch.zeros(N, C_OUT, H, W, dtype=torch.float32, device=device)
        if bias is not None:
            y += bias.float().view(1, -1, 1, 1)
        return y, 0.0

    # ── Stage-1.5: Metadata ──
    flags_4d = flags.reshape(N, C_IN, GH, GW)
    del flags  # free early

    has_bias = bias is not None
    bias_f32 = bias.float().contiguous() if has_bias else torch.empty(1, device=device)

    y = torch.zeros(N, C_OUT, H, W, dtype=torch.float32, device=device)

    grid_spatial = N * GH * GW
    grid_cout = triton.cdiv(C_OUT, BLOCK_N)

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    if kernel_size == 3:
        active_cin, active_K_raw, ak_pad = _build_active_channels_3x3(
            flags_4d, N, C_IN, GH, GW, BLOCK_K, device)
        del flags_4d

        if active_K_raw == 0:
            if has_bias:
                y += bias_f32.view(1, -1, 1, 1)
            return y, 0.0

        w_gathered = _gather_weight_3x3(w_f16, active_cin, active_K_raw,
                                        ak_pad, C_OUT)
        del w_f16  # free original weight copy

        # Strides computed in Python — guaranteed to match contiguous layout
        w_stride_co = ak_pad * 9   # [C_OUT, AK_PAD, 3, 3] dim-0 stride
        w_stride_k = 9             # dim-1 stride
        num_k_iters = ak_pad // BLOCK_K  # exact since ak_pad is multiple of BLOCK_K

        start_evt.record()
        sparse_conv3x3_v10_kernel[(grid_spatial, grid_cout)](
            x_f16, w_gathered, bias_f32, active_cin, y,
            N, C_IN, C_OUT, H, W,
            GH, GW,
            active_K_raw,
            num_k_iters,
            w_stride_co,
            w_stride_k,
            HAS_BIAS=has_bias,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            BLOCK_H=BH, BLOCK_W=BW,
        )
        end_evt.record()

    else:  # 1×1
        active_cin, active_K_raw, ak_pad = _build_active_channels_1x1(
            flags_4d, N, C_IN, GH, GW, BLOCK_K, device)
        del flags_4d

        if active_K_raw == 0:
            if has_bias:
                y += bias_f32.view(1, -1, 1, 1)
            return y, 0.0

        w_gathered = _gather_weight_1x1(w_f16, active_cin, active_K_raw,
                                        ak_pad, C_OUT, C_IN)
        del w_f16

        w_stride_co = ak_pad  # [C_OUT, AK_PAD] dim-0 stride
        num_k_iters = ak_pad // BLOCK_K

        start_evt.record()
        sparse_conv1x1_v10_kernel[(grid_spatial, grid_cout)](
            x_f16, w_gathered, bias_f32, active_cin, y,
            N, C_IN, C_OUT, H, W,
            GH, GW,
            active_K_raw,
            num_k_iters,
            w_stride_co,
            HAS_BIAS=has_bias,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            BLOCK_H=BH, BLOCK_W=BW,
        )
        end_evt.record()

    torch.cuda.synchronize(device)
    sparse_ms = start_evt.elapsed_time(end_evt)

    return y, sparse_ms