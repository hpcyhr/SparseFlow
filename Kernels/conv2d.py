"""
SparseFlow Conv2d Triton Kernels — v10.4 Dynamic Block Tuning

Performance: Dynamic BLOCK_M based on spatial resolution.
  H=56 → BH=8, BW=16 → BLOCK_M=128 → GH×GW = 7×4 = 28 tiles (vs 98 in v10.3)
  H=28 → BH=8, BW=8  → BLOCK_M=64  → GH×GW = 4×4 = 16 tiles
  H<16 → BH=4, BW=4  → BLOCK_M=16  → fine-grained sparsity pruning

  This 3.5× reduction in spatial tiles directly reduces:
    - Grid programs launched (less kernel overhead)
    - Python-side metadata ops (flags_4d is smaller)
    - Each program does MORE work → better Tensor Core utilization

Safety: BLOCK_N capped at 32 when N≥256 (T=16 proxy) to limit register
  pressure. acc[128, 32] = 128×32×4 = 16KB fits comfortably in A100 regs.

Correctness: Prescan and kernel share identical BH/BW → flags_4d shape
  [N, C_IN, GH, GW] is consistent. max_pool2d 3×3 dilation on the coarser
  grid still correctly captures neighbor block influence because each flag
  represents a larger spatial region.

Architecture unchanged:
  Stage-1:   Prescan → flags
  Stage-1.5: Active channel detection + weight gather
  Stage-2:   Branch-free Tensor Core GEMM over shrunk K

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
# Dynamic tile config
# ═══════════════════════════════════════════════════════════════════════

def _select_block_sizes(H, W, C_IN, C_OUT, kernel_size, N):
    """
    Dynamic block selection based on pixel count and batch size.

    All tl.dot dimensions must be powers-of-2, >= 16.

    Strategy:
      Large maps (H*W ≥ 56*56 = 3136):
        BH=8, BW=16 → BLOCK_M=128 → GH*GW = 7*4 = 28 for H=56
        Maximizes per-program work, amortizes kernel launch + metadata.

      Medium maps (H*W ≥ 28*28 = 784):
        BH=8, BW=8 → BLOCK_M=64 → GH*GW = 4*4 = 16 for H=28
        Good balance of work per program and sparsity granularity.

      Small maps (H*W < 784):
        BH=4, BW=4 → BLOCK_M=16
        Fine-grained sparsity pruning for small feature maps.

    Register pressure guard:
      When N ≥ 256 (proxy for T ≥ 16 with BS ≥ 16, or T ≥ 8 with BS ≥ 32),
      cap BLOCK_N at 32. acc[128, 32] = 16KB < register budget.
      Otherwise allow BLOCK_N=64 for larger C_OUT.

    Tile count comparison at H=56, W=56:
      v10.3: BH=4, BW=8 → GH=14, GW=7 → 98 tiles
      v10.4: BH=8, BW=16 → GH=7, GW=4  → 28 tiles  (3.5× fewer)
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

    # C_OUT tile — register-safe selection
    high_pressure = (N >= 256)  # proxy for large T

    if high_pressure:
        # Cap at 32 to keep acc[BLOCK_M, 32] manageable
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
# Stage-1.5: Metadata build (vectorized torch, GPU)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _build_active_channels_3x3(flags_4d, N, C_IN, GH, GW, BLOCK_K, device):
    """
    Dilate flags by 3×3 (neighbor block influence), reduce to global active set.

    Correctness at larger BH/BW: Each flag now represents a larger spatial block.
    The max_pool2d 3×3 dilation still correctly marks channel c as needed for
    tile (gh, gw) if c is non-zero in any of the 9 neighbor BLOCKS. This is
    conservative: it may include some channels that are zero in the specific
    pixel sub-region within the block, but never misses a needed channel.
    At high sparsity this conservative estimate is still very effective.
    """
    f = flags_4d.float()
    padded = torch.nn.functional.pad(f, (1, 1, 1, 1), value=0.0)
    dilated = torch.nn.functional.max_pool2d(padded, 3, stride=1, padding=0)

    channel_active = (dilated > 0).any(dim=0).any(dim=-1).any(dim=-1)
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
                          kernel_size=3, threshold=1e-6):
    """
    Three-stage sparse conv2d with dynamic block tuning.

    block_size semantics:
      None  → fully dynamic: _select_block_sizes(H, W, ...) decides everything
      int   → legacy hint, but OVERRIDDEN for large maps (H*W >= 3136)
              This ensures H=56 always gets BLOCK_M=128 regardless of caller

    Returns:
        y: [N, C_OUT, H, W] float32
        sparse_ms: Stage-2 kernel time in ms
    """
    N, C_IN, H, W = x.shape
    C_OUT = weight.shape[0]
    device = x.device

    # Dynamic block selection — always uses _select_block_sizes for optimal
    # tile config. Even if block_size was passed, large maps get upgraded.
    pixels = H * W
    if block_size is None or pixels >= 3136:
        # Full dynamic mode: kernel decides optimal BH/BW/BLOCK_M/BLOCK_N
        BH, BW, BLOCK_M, BLOCK_N, BLOCK_K = _select_block_sizes(
            H, W, C_IN, C_OUT, kernel_size, N)
    else:
        # Legacy fallback: use passed block_size as BH=BW
        # Ensure minimum tile dims for tl.dot (>= 4)
        bs = max(block_size, 4)
        BH, BW = bs, bs
        BLOCK_M = BH * BW
        BLOCK_N = 32 if C_OUT >= 32 else 16
        BLOCK_K = 16

    GH = triton.cdiv(H, BH)
    GW = triton.cdiv(W, BW)

    x_f16 = x.half().contiguous()
    w_f16 = weight.half().contiguous()

    # ── Stage-1: Prescan (uses same BH, BW as kernel) ──
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
    del flags

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
        del w_f16

        w_stride_co = ak_pad * 9
        w_stride_k = 9
        num_k_iters = ak_pad // BLOCK_K

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

        w_stride_co = ak_pad
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