"""
SparseFlow Conv2d Triton Kernels — v11.0 Channel-First Prescan

═══════════════════════════════════════════════════════════════════════
v11.0 changes vs v10.4:
═══════════════════════════════════════════════════════════════════════

1. **Channel-first prescan** (改动 1):
   Old: prescan_kernel grid = N * C_IN * GH * GW (one program per block)
   New: channel_prescan_kernel grid = N * C_IN (one program per channel)
        Each program iterates over all tiles (GH × GW) internally,
        doing a simple OR per tile. This massively reduces kernel launch
        overhead — e.g. for N=512, C=64, GH=7, GW=4:
          Old: 512 * 64 * 7 * 4 = 917,504 programs
          New: 512 * 64         = 32,768  programs  (28× fewer)
   The old prescan_kernel is retained for backward compatibility
   (Diy/test scripts still import it).

2. **Time-union prescan** (改动 2):
   When merge_time_steps > 0 is passed to sparse_conv2d_forward, the
   prescan is run on a time-OR-compressed version of x:
     x_merged = (x.view(T, B, C, H, W).abs() > threshold).any(dim=0)
   This reduces the prescan input from N=T*B to just B samples.
   Stage-2 still runs on the full N=T*B input with original data.
   merge_time_steps=0 (default) disables this optimization.

3. **Stage-2 interface unchanged** (改动 3):
   sparse_conv3x3_v10_kernel / sparse_conv1x1_v10_kernel signatures
   are 100% unchanged. Only the metadata source (active_cin, etc.)
   changes via the new prescan + simplified _build_active_channels.

4. **flags caching + return_ms** (改动 4):
   - sparse_conv2d_forward(..., flags=None, return_ms=False)
   - flags: optional pre-allocated int32 buffer for prescan output
   - return_ms: only creates CUDA events and synchronizes when True

═══════════════════════════════════════════════════════════════════════

Performance: Dynamic BLOCK_M based on spatial resolution.
  H=56 → BH=8, BW=16 → BLOCK_M=128 → GH×GW = 7×4 = 28 tiles
  H=28 → BH=8, BW=8  → BLOCK_M=64  → GH×GW = 4×4 = 16 tiles
  H<16 → BH=4, BW=4  → BLOCK_M=16  → fine-grained sparsity pruning

Safety: BLOCK_N capped at 32 when N≥256 to limit register pressure.

Architecture:
  Stage-1:   Channel-first prescan → flags_4d [N, C_IN, GH, GW]
  Stage-1.5: Active channel detection + weight gather (simplified)
  Stage-2:   Branch-free Tensor Core GEMM (interface unchanged)

Triton 2.x safe: no continue, no 0-d tensors, no scalar .to()
"""

import torch
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════
# Stage-1: Channel-first prescan — one program per (n, c)
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def channel_prescan_kernel(
    x_ptr,          # [N, C, H, W] fp16, contiguous
    flags_ptr,      # [N, C, GH, GW] int32, contiguous (or >= that size)
    N, C, H, W,
    GH, GW,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
    """
    Channel-first prescan: each program handles one (n, c) pair and
    iterates over all spatial tiles (gh, gw).

    Grid: (N * C,)

    For each tile, loads the BH × BW patch, checks if any element is
    nonzero, and writes 1/0 to flags_ptr[n, c, gh, gw].

    Memory access pattern: for a given (n, c), consecutive tiles in gw
    direction access adjacent memory → reasonably coalesced.
    """
    pid = tl.program_id(0)
    c = pid % C
    n = pid // C

    # Base pointer for x[n, c, :, :]
    x_base = x_ptr + (n * C + c) * H * W

    # Base pointer for flags[n, c, :, :]
    flags_base = flags_ptr + (n * C + c) * GH * GW

    for gh in range(GH):
        for gw in range(GW):
            # Tile origin
            h_start = gh * BLOCK_H
            w_start = gw * BLOCK_W

            # Load BH × BW patch
            offs_h = h_start + tl.arange(0, BLOCK_H)
            offs_w = w_start + tl.arange(0, BLOCK_W)
            hh = offs_h[:, None]
            ww = offs_w[None, :]
            mask = (hh < H) & (ww < W)

            val = tl.load(x_base + hh * W + ww, mask=mask, other=0.0)
            is_nz = tl.max(tl.abs(val)) > THRESHOLD

            tl.store(flags_base + gh * GW + gw, is_nz.to(tl.int32))


# ═══════════════════════════════════════════════════════════════════════
# Legacy prescan — retained for backward compatibility (Diy/test scripts)
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
# Stage-2a: Sparse 3×3 Conv — branch-free GEMM (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def sparse_conv3x3_v10_kernel(
    x_ptr,              # [N, C_IN, H, W] fp16
    w_gathered_ptr,     # [C_OUT, AK_PAD, 3, 3] fp16 contiguous
    bias_ptr,           # [C_OUT] fp32 or dummy
    active_cin_ptr,     # [AK_PAD] int32
    y_ptr,              # [N, C_OUT, H, W] fp32
    N_val, C_IN, C_OUT, H, W,
    GH, GW,
    active_K_raw,
    num_k_iters,
    w_stride_co,        # = AK_PAD * 9
    w_stride_k,         # = 9
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
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

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    HW = H * W
    H_max = H - 1
    W_max = W - 1

    for k_iter in range(num_k_iters):
        k_start = k_iter * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < active_K_raw

        cin_idx = tl.load(active_cin_ptr + offs_k, mask=k_mask, other=0)

        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                raw_h = tile_h + (kh - 1)
                raw_w = tile_w + (kw - 1)

                h_ok = (raw_h >= 0) & (raw_h < H)
                w_ok = (raw_w >= 0) & (raw_w < W)

                safe_h = tl.minimum(tl.maximum(raw_h, 0), H_max)
                safe_w = tl.minimum(tl.maximum(raw_w, 0), W_max)

                x_addrs = (x_ptr
                           + (n_idx * C_IN + cin_idx[None, :]) * HW
                           + safe_h[:, None] * W
                           + safe_w[:, None])
                x_load_mask = (k_mask[None, :] & m_mask[:, None]
                               & h_ok[:, None] & w_ok[:, None])
                x_tile = tl.load(x_addrs, mask=x_load_mask, other=0.0)
                x_tile = x_tile.to(tl.float16)

                w_addrs = (w_gathered_ptr
                           + offs_n[None, :] * w_stride_co
                           + offs_k[:, None] * w_stride_k
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
# Stage-2b: Sparse 1×1 Conv — branch-free GEMM (UNCHANGED)
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

    safe_h = tl.minimum(tile_h, H - 1)
    safe_w = tl.minimum(tile_w, W - 1)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    HW = H * W

    for k_iter in range(num_k_iters):
        k_start = k_iter * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < active_K_raw

        cin_idx = tl.load(active_cin_ptr + offs_k, mask=k_mask, other=0)

        x_addrs = (x_ptr
                   + (n_idx * C_IN + cin_idx[None, :]) * HW
                   + safe_h[:, None] * W
                   + safe_w[:, None])
        x_load_mask = k_mask[None, :] & m_mask[:, None]
        x_tile = tl.load(x_addrs, mask=x_load_mask, other=0.0)
        x_tile = x_tile.to(tl.float16)

        w_addrs = (w_gathered_ptr
                   + offs_n[None, :] * w_stride_co
                   + offs_k[:, None])
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
# Dense Conv 3×3 (baseline reference, UNCHANGED)
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
# Dynamic tile config (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════

def _select_block_sizes(H, W, C_IN, C_OUT, kernel_size, N):
    """
    Dynamic block selection based on pixel count and batch size.

    All tl.dot dimensions must be powers-of-2, >= 16.

    Strategy:
      Large maps (H*W ≥ 56*56 = 3136):
        BH=8, BW=16 → BLOCK_M=128 → GH*GW = 7*4 = 28 for H=56

      Medium maps (H*W ≥ 28*28 = 784):
        BH=8, BW=8 → BLOCK_M=64 → GH*GW = 4*4 = 16 for H=28

      Small maps (H*W < 784):
        BH=4, BW=4 → BLOCK_M=16

    Register pressure guard:
      When N ≥ 256, cap BLOCK_N at 32.
    """
    pixels = H * W

    if pixels >= 3136:          # H*W ≥ 56*56
        BH, BW = 8, 16         # BLOCK_M = 128
    elif pixels >= 784:         # H*W ≥ 28*28
        BH, BW = 8, 8          # BLOCK_M = 64
    elif min(H, W) >= 8:
        BH, BW = 4, 4          # BLOCK_M = 16
    else:
        BH, BW = 4, 4          # BLOCK_M = 16

    BLOCK_M = BH * BW

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
# Stage-1.5: Active channel detection
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _build_active_channels_3x3(flags_4d, N, C_IN, GH, GW, BLOCK_K, device):
    """
    Build global active channel set for 3×3 conv.

    For 3×3 conv, a channel c is needed at spatial tile (gh, gw) if c is
    nonzero in any of the 9 neighbor tiles (because the 3×3 filter window
    extends ±1 in each spatial direction on the tile grid). We use
    max_pool2d with kernel_size=3 as an efficient dilation operator.

    After dilation, reduce over (N, GH, GW) → per-channel active mask.
    """
    # Dilate flags_4d [N, C_IN, GH, GW] by 3×3 neighbor influence
    f = flags_4d.float()
    padded = torch.nn.functional.pad(f, (1, 1, 1, 1), value=0.0)
    dilated = torch.nn.functional.max_pool2d(padded, 3, stride=1, padding=0)

    # Reduce: any batch, any tile → channel active
    channel_active = (dilated > 0).any(dim=0).any(dim=-1).any(dim=-1)  # [C_IN]
    nz = channel_active.nonzero(as_tuple=False)
    active_cin = nz.reshape(-1).to(torch.int32)
    active_K_raw = int(active_cin.numel())

    if active_K_raw == 0:
        z = torch.zeros(BLOCK_K, dtype=torch.int32, device=device)
        return z, 0, BLOCK_K

    # Pad to multiple of BLOCK_K
    active_K_padded = ((active_K_raw + BLOCK_K - 1) // BLOCK_K) * BLOCK_K
    if active_K_padded > active_K_raw:
        pad_t = torch.zeros(active_K_padded - active_K_raw,
                            dtype=torch.int32, device=device)
        active_cin = torch.cat([active_cin, pad_t])

    return active_cin.contiguous(), active_K_raw, active_K_padded


@torch.no_grad()
def _build_active_channels_1x1(flags_4d, N, C_IN, GH, GW, BLOCK_K, device):
    """
    Build global active channel set for 1×1 conv.

    For 1×1, no neighbor influence: just reduce flags over (N, GH, GW).
    """
    channel_active = (flags_4d > 0).any(dim=0).any(dim=-1).any(dim=-1)  # [C_IN]
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


# ═══════════════════════════════════════════════════════════════════════
# Weight gather helpers (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _gather_weight_3x3(weight_f16, active_cin, active_K_raw, active_K_padded,
                        C_OUT):
    """
    [C_OUT, C_IN, 3, 3] → [C_OUT, AK_PAD, 3, 3]
    Padding rows are zero. No redundant alloc: cat only if padding needed.
    """
    idx = active_cin[:active_K_raw].long()
    w_active = weight_f16[:, idx, :, :]

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
    w_active = w_2d[:, idx]

    if active_K_padded > active_K_raw:
        pad_size = active_K_padded - active_K_raw
        pad = torch.zeros(C_OUT, pad_size,
                          dtype=weight_f16.dtype, device=weight_f16.device)
        w_active = torch.cat([w_active, pad], dim=1)

    return w_active.contiguous()


# ═══════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════

def sparse_conv2d_forward(x, weight, bias, block_size=None,
                          kernel_size=3, threshold=1e-6,
                          flags=None, return_ms=False,
                          merge_time_steps=0):
    """
    Three-stage sparse conv2d with channel-first prescan.

    Args:
        x: [N, C_IN, H, W] input tensor (N is typically T*B for SNN)
        weight: [C_OUT, C_IN, K, K] convolution weight
        bias: [C_OUT] or None
        block_size: None (fully dynamic) or int (legacy hint)
        kernel_size: 3 or 1
        threshold: zero-detection threshold
        flags: optional pre-allocated int32 buffer for prescan output.
            If provided, must have numel() >= prescan_N * C_IN * GH * GW.
            If None, allocated internally.
        return_ms: if True, wraps Stage-2 in CUDA events for timing.
            If False (default), no events, no cudaDeviceSynchronize, returns 0.0.
        merge_time_steps: int, > 0 enables time-union prescan optimization.
            When enabled, x is treated as [T, B, C, H, W] where T = merge_time_steps.
            Prescan runs on OR-compressed [B, C, H, W] (T× fewer programs).
            Stage-2 still computes on full x. Default 0 = disabled.

    Returns:
        y: [N, C_OUT, H, W] float32
        sparse_ms: Stage-2 kernel time in ms (0.0 when return_ms=False)
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

    x_f16 = x.half().contiguous()
    w_f16 = weight.half().contiguous()

    # ══════════════════════════════════════════════════════════════════
    # Stage-1: Channel-first prescan
    # ══════════════════════════════════════════════════════════════════

    # Determine prescan input: optionally OR-compress time dimension
    if merge_time_steps > 0 and N > merge_time_steps:
        T = merge_time_steps
        B = N // T
        # OR across time: any(dim=0) over [T, B, C, H, W]
        # Result: [B, C, H, W] in fp16 (binary 0/1 values cast to fp16)
        x_for_prescan = (x_f16.view(T, B, C_IN, H, W)
                         .abs().gt(threshold)
                         .any(dim=0)
                         .to(x_f16.dtype))  # [B, C_IN, H, W]
        prescan_N = B
    else:
        x_for_prescan = x_f16
        prescan_N = N

    total_flags = prescan_N * C_IN * GH * GW

    # Use caller-supplied flags buffer if available, else allocate
    if flags is not None:
        assert flags.numel() >= total_flags, (
            f"flags buffer too small: {flags.numel()} < {total_flags}")
    else:
        flags = torch.empty(total_flags, dtype=torch.int32, device=device)

    # Launch channel-first prescan: grid = prescan_N * C_IN
    channel_prescan_kernel[(prescan_N * C_IN,)](
        x_for_prescan, flags, prescan_N, C_IN, H, W, GH, GW,
        BLOCK_H=BH, BLOCK_W=BW, THRESHOLD=threshold,
    )

    # Sync needed to read flags on CPU for all-zero fast path
    torch.cuda.synchronize(device)

    # Reshape to 4D for metadata building
    flags_4d = flags[:total_flags].reshape(prescan_N, C_IN, GH, GW)

    # All-zero fast path
    if flags_4d.sum().item() == 0:
        y = torch.zeros(N, C_OUT, H, W, dtype=torch.float32, device=device)
        if bias is not None:
            y += bias.float().view(1, -1, 1, 1)
        return y, 0.0

    # ══════════════════════════════════════════════════════════════════
    # Stage-1.5: Active channel metadata
    # ══════════════════════════════════════════════════════════════════

    has_bias = bias is not None
    bias_f32 = bias.float().contiguous() if has_bias else torch.empty(1, device=device)

    y = torch.zeros(N, C_OUT, H, W, dtype=torch.float32, device=device)

    # Stage-2 grid runs on FULL N (not prescan_N)
    grid_spatial = N * GH * GW
    grid_cout = triton.cdiv(C_OUT, BLOCK_N)

    # Timing: only create events when return_ms=True
    sparse_ms = 0.0
    if return_ms:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

    if kernel_size == 3:
        active_cin, active_K_raw, ak_pad = _build_active_channels_3x3(
            flags_4d, prescan_N, C_IN, GH, GW, BLOCK_K, device)
        del flags_4d

        if active_K_raw == 0:
            if has_bias:
                y += bias_f32.view(1, -1, 1, 1)
            return y, 0.0

        w_gathered = _gather_weight_3x3(w_f16, active_cin, active_K_raw,
                                        ak_pad, C_OUT)
        del w_f16

        w_stride_co = ak_pad * 9
        w_stride_k = 9
        num_k_iters = ak_pad // BLOCK_K

        if return_ms:
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

        if return_ms:
            end_evt.record()

    else:  # 1×1
        active_cin, active_K_raw, ak_pad = _build_active_channels_1x1(
            flags_4d, prescan_N, C_IN, GH, GW, BLOCK_K, device)
        del flags_4d

        if active_K_raw == 0:
            if has_bias:
                y += bias_f32.view(1, -1, 1, 1)
            return y, 0.0

        w_gathered = _gather_weight_1x1(w_f16, active_cin, active_K_raw,
                                        ak_pad, C_OUT, C_IN)
        del w_f16

        w_stride_co = ak_pad
        num_k_iters = ak_pad // BLOCK_K

        if return_ms:
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

        if return_ms:
            end_evt.record()

    # Timing: only synchronize when return_ms=True
    if return_ms:
        torch.cuda.synchronize(device)
        sparse_ms = start_evt.elapsed_time(end_evt)

    return y, sparse_ms