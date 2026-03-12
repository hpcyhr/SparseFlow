"""
SparseFlow Fused Sparse Conv + LIF Triton Kernels

Eliminates intermediate VRAM write for conv output by fusing LIF neuron
dynamics directly into the sparse conv kernel epilogue.

In-register flow (after conv accumulation):
  1. I = acc + bias
  2. V_temp = V_prev * decay + I
  3. Spike = 1.0 if V_temp > V_th else 0.0
  4. V_next = V_temp * (1.0 - Spike)   (soft reset)

Memory IO per output element:
  Read:  x (sparse), w, bias, V_prev
  Write: Spike_out, V_next
  Skip:  intermediate conv result (never touches VRAM)

Reuses conv2d.py v15.1 prescan infrastructure (_build_tile_csr, _select_tile_sizes).
"""

import torch
import triton
import triton.language as tl
from triton import autotune, Config


# ═══════════════════════════════════════════════════════════════════════
# Autotune configs
# ═══════════════════════════════════════════════════════════════════════

def _make_fused_configs(block_h, block_w):
    block_m = block_h * block_w
    configs = []
    for bn in [64, 128]:
        for bk in [32, 64]:
            for nw in [4, 8]:
                configs.append(Config(
                    {'BLOCK_M': block_m, 'BLOCK_N': bn, 'BLOCK_K': bk,
                     'BLOCK_H': block_h, 'BLOCK_W': block_w},
                    num_warps=nw, num_stages=1))
    return configs

_FUSED_3X3_CONFIGS_8x8  = _make_fused_configs(8, 8)
_FUSED_3X3_CONFIGS_8x16 = _make_fused_configs(8, 16)
_FUSED_1X1_CONFIGS_8x8  = _make_fused_configs(8, 8)
_FUSED_1X1_CONFIGS_8x16 = _make_fused_configs(8, 16)


# ═══════════════════════════════════════════════════════════════════════
# Shared LIF epilogue macro (called at end of each kernel)
# ═══════════════════════════════════════════════════════════════════════
# Not a separate function due to Triton JIT constraints — inlined below.


# ═══════════════════════════════════════════════════════════════════════
# Fused Sparse 3x3 Conv + LIF — 8x8
# ═══════════════════════════════════════════════════════════════════════

@autotune(configs=_FUSED_3X3_CONFIGS_8x8,
          key=['C_IN', 'C_OUT', 'H', 'W', 'GH', 'GW'])
@triton.jit
def fused_conv3x3_lif_kernel_8x8(
    x_ptr, w_cl_ptr, bias_ptr, tile_ptr_data, tile_cin_ptr,
    v_prev_ptr, spike_ptr, v_next_ptr, N_val,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
    GH: tl.constexpr, GW: tl.constexpr,
    HAS_BIAS: tl.constexpr, DECAY: tl.constexpr, V_TH: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
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

    c_out_start = pid_cout * BLOCK_N
    offs_n = c_out_start + tl.arange(0, BLOCK_N)
    n_mask = offs_n < C_OUT

    offs_m = tl.arange(0, BLOCK_M)
    out_h = gh_idx * BLOCK_H + offs_m // BLOCK_W
    out_w = gw_idx * BLOCK_W + offs_m % BLOCK_W
    m_mask = (out_h < H) & (out_w < W)
    HW: tl.constexpr = H * W

    tile_start = tl.load(tile_ptr_data + tile_id)
    tile_end = tl.load(tile_ptr_data + tile_id + 1)
    active_K = tile_end - tile_start
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    W_CIN_STRIDE: tl.constexpr = C_IN
    W_KH_STRIDE: tl.constexpr = 3 * C_IN
    W_CO_STRIDE: tl.constexpr = 9 * C_IN

    k_start = 0
    while k_start < active_K:
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < active_K
        cin_global = tl.load(tile_cin_ptr + tile_start + offs_k, mask=k_mask, other=0)
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                in_h = out_h + (kh - 1)
                in_w = out_w + (kw - 1)
                h_ok = (in_h >= 0) & (in_h < H)
                w_ok = (in_w >= 0) & (in_w < W)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W - 1)
                x_addrs = (x_ptr + (n_idx * C_IN + cin_global[None, :]) * HW
                           + safe_h[:, None] * W + safe_w[:, None])
                x_lm = k_mask[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                x_tile = tl.load(x_addrs, mask=x_lm, other=0.0).to(tl.float16)
                w_addrs = (w_cl_ptr + offs_n[None, :] * W_CO_STRIDE
                           + kh * W_KH_STRIDE + kw * W_CIN_STRIDE + cin_global[:, None])
                w_lm = k_mask[:, None] & n_mask[None, :]
                w_tile = tl.load(w_addrs, mask=w_lm, other=0.0).to(tl.float16)
                acc += tl.dot(x_tile, w_tile)
        k_start += BLOCK_K

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]

    # ═══ FUSED LIF EPILOGUE ═══
    out_addrs = (n_idx * C_OUT + offs_n[None, :]) * HW + out_h[:, None] * W + out_w[:, None]
    out_mask = m_mask[:, None] & n_mask[None, :]
    v_prev = tl.load(v_prev_ptr + out_addrs, mask=out_mask, other=0.0)
    v_temp = v_prev * DECAY + acc
    spike = (v_temp > V_TH).to(tl.float32)
    v_next = v_temp * (1.0 - spike)
    tl.store(spike_ptr + out_addrs, spike, mask=out_mask)
    tl.store(v_next_ptr + out_addrs, v_next, mask=out_mask)


@autotune(configs=_FUSED_3X3_CONFIGS_8x16,
          key=['C_IN', 'C_OUT', 'H', 'W', 'GH', 'GW'])
@triton.jit
def fused_conv3x3_lif_kernel_8x16(
    x_ptr, w_cl_ptr, bias_ptr, tile_ptr_data, tile_cin_ptr,
    v_prev_ptr, spike_ptr, v_next_ptr, N_val,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
    GH: tl.constexpr, GW: tl.constexpr,
    HAS_BIAS: tl.constexpr, DECAY: tl.constexpr, V_TH: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    # Body identical to 8x8 — autotuner handles BH=8,BW=16
    tile_id = tl.program_id(0)
    pid_cout = tl.program_id(1)
    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles:
        return
    gw_idx = tile_id % GW; tmp = tile_id // GW
    gh_idx = tmp % GH; n_idx = tmp // GH
    c_out_start = pid_cout * BLOCK_N
    offs_n = c_out_start + tl.arange(0, BLOCK_N); n_mask = offs_n < C_OUT
    offs_m = tl.arange(0, BLOCK_M)
    out_h = gh_idx * BLOCK_H + offs_m // BLOCK_W
    out_w = gw_idx * BLOCK_W + offs_m % BLOCK_W
    m_mask = (out_h < H) & (out_w < W)
    HW: tl.constexpr = H * W
    tile_start = tl.load(tile_ptr_data + tile_id)
    tile_end = tl.load(tile_ptr_data + tile_id + 1)
    active_K = tile_end - tile_start
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    W_CIN_STRIDE: tl.constexpr = C_IN
    W_KH_STRIDE: tl.constexpr = 3 * C_IN
    W_CO_STRIDE: tl.constexpr = 9 * C_IN
    k_start = 0
    while k_start < active_K:
        offs_k = k_start + tl.arange(0, BLOCK_K); k_mask = offs_k < active_K
        cin_global = tl.load(tile_cin_ptr + tile_start + offs_k, mask=k_mask, other=0)
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                in_h = out_h + (kh - 1); in_w = out_w + (kw - 1)
                h_ok = (in_h >= 0) & (in_h < H); w_ok = (in_w >= 0) & (in_w < W)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W - 1)
                x_addrs = x_ptr + (n_idx * C_IN + cin_global[None, :]) * HW + safe_h[:, None] * W + safe_w[:, None]
                x_lm = k_mask[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                x_tile = tl.load(x_addrs, mask=x_lm, other=0.0).to(tl.float16)
                w_addrs = w_cl_ptr + offs_n[None, :] * W_CO_STRIDE + kh * W_KH_STRIDE + kw * W_CIN_STRIDE + cin_global[:, None]
                w_tile = tl.load(w_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
                acc += tl.dot(x_tile, w_tile)
        k_start += BLOCK_K
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
    out_addrs = (n_idx * C_OUT + offs_n[None, :]) * HW + out_h[:, None] * W + out_w[:, None]
    out_mask = m_mask[:, None] & n_mask[None, :]
    v_prev = tl.load(v_prev_ptr + out_addrs, mask=out_mask, other=0.0)
    v_temp = v_prev * DECAY + acc
    spike = (v_temp > V_TH).to(tl.float32)
    tl.store(spike_ptr + out_addrs, spike, mask=out_mask)
    tl.store(v_next_ptr + out_addrs, v_temp * (1.0 - spike), mask=out_mask)


@autotune(configs=_FUSED_1X1_CONFIGS_8x8,
          key=['C_IN', 'C_OUT', 'H', 'W', 'GH', 'GW'])
@triton.jit
def fused_conv1x1_lif_kernel_8x8(
    x_ptr, w_ptr, bias_ptr, tile_ptr_data, tile_cin_ptr,
    v_prev_ptr, spike_ptr, v_next_ptr, N_val,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
    GH: tl.constexpr, GW: tl.constexpr,
    HAS_BIAS: tl.constexpr, DECAY: tl.constexpr, V_TH: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    tile_id = tl.program_id(0); pid_cout = tl.program_id(1)
    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles: return
    gw_idx = tile_id % GW; tmp = tile_id // GW
    gh_idx = tmp % GH; n_idx = tmp // GH
    offs_n = pid_cout * BLOCK_N + tl.arange(0, BLOCK_N); n_mask = offs_n < C_OUT
    offs_m = tl.arange(0, BLOCK_M)
    out_h = gh_idx * BLOCK_H + offs_m // BLOCK_W
    out_w = gw_idx * BLOCK_W + offs_m % BLOCK_W
    m_mask = (out_h < H) & (out_w < W)
    safe_h = tl.minimum(out_h, H - 1); safe_w = tl.minimum(out_w, W - 1)
    HW: tl.constexpr = H * W
    tile_start = tl.load(tile_ptr_data + tile_id)
    active_K = tl.load(tile_ptr_data + tile_id + 1) - tile_start
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    k_start = 0
    while k_start < active_K:
        offs_k = k_start + tl.arange(0, BLOCK_K); k_mask = offs_k < active_K
        cin_g = tl.load(tile_cin_ptr + tile_start + offs_k, mask=k_mask, other=0)
        x_tile = tl.load(x_ptr + (n_idx * C_IN + cin_g[None, :]) * HW + safe_h[:, None] * W + safe_w[:, None],
                         mask=k_mask[None, :] & m_mask[:, None], other=0.0).to(tl.float16)
        w_tile = tl.load(w_ptr + offs_n[None, :] * C_IN + cin_g[:, None],
                         mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
        acc += tl.dot(x_tile, w_tile)
        k_start += BLOCK_K
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
    out_addrs = (n_idx * C_OUT + offs_n[None, :]) * HW + out_h[:, None] * W + out_w[:, None]
    out_mask = m_mask[:, None] & n_mask[None, :]
    v_prev = tl.load(v_prev_ptr + out_addrs, mask=out_mask, other=0.0)
    v_temp = v_prev * DECAY + acc
    spike = (v_temp > V_TH).to(tl.float32)
    tl.store(spike_ptr + out_addrs, spike, mask=out_mask)
    tl.store(v_next_ptr + out_addrs, v_temp * (1.0 - spike), mask=out_mask)


@autotune(configs=_FUSED_1X1_CONFIGS_8x16,
          key=['C_IN', 'C_OUT', 'H', 'W', 'GH', 'GW'])
@triton.jit
def fused_conv1x1_lif_kernel_8x16(
    x_ptr, w_ptr, bias_ptr, tile_ptr_data, tile_cin_ptr,
    v_prev_ptr, spike_ptr, v_next_ptr, N_val,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
    GH: tl.constexpr, GW: tl.constexpr,
    HAS_BIAS: tl.constexpr, DECAY: tl.constexpr, V_TH: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    tile_id = tl.program_id(0); pid_cout = tl.program_id(1)
    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles: return
    gw_idx = tile_id % GW; tmp = tile_id // GW
    gh_idx = tmp % GH; n_idx = tmp // GH
    offs_n = pid_cout * BLOCK_N + tl.arange(0, BLOCK_N); n_mask = offs_n < C_OUT
    offs_m = tl.arange(0, BLOCK_M)
    out_h = gh_idx * BLOCK_H + offs_m // BLOCK_W
    out_w = gw_idx * BLOCK_W + offs_m % BLOCK_W
    m_mask = (out_h < H) & (out_w < W)
    safe_h = tl.minimum(out_h, H - 1); safe_w = tl.minimum(out_w, W - 1)
    HW: tl.constexpr = H * W
    tile_start = tl.load(tile_ptr_data + tile_id)
    active_K = tl.load(tile_ptr_data + tile_id + 1) - tile_start
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    k_start = 0
    while k_start < active_K:
        offs_k = k_start + tl.arange(0, BLOCK_K); k_mask = offs_k < active_K
        cin_g = tl.load(tile_cin_ptr + tile_start + offs_k, mask=k_mask, other=0)
        x_tile = tl.load(x_ptr + (n_idx * C_IN + cin_g[None, :]) * HW + safe_h[:, None] * W + safe_w[:, None],
                         mask=k_mask[None, :] & m_mask[:, None], other=0.0).to(tl.float16)
        w_tile = tl.load(w_ptr + offs_n[None, :] * C_IN + cin_g[:, None],
                         mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
        acc += tl.dot(x_tile, w_tile)
        k_start += BLOCK_K
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
    out_addrs = (n_idx * C_OUT + offs_n[None, :]) * HW + out_h[:, None] * W + out_w[:, None]
    out_mask = m_mask[:, None] & n_mask[None, :]
    v_prev = tl.load(v_prev_ptr + out_addrs, mask=out_mask, other=0.0)
    v_temp = v_prev * DECAY + acc
    spike = (v_temp > V_TH).to(tl.float32)
    tl.store(spike_ptr + out_addrs, spike, mask=out_mask)
    tl.store(v_next_ptr + out_addrs, v_temp * (1.0 - spike), mask=out_mask)


# ═══════════════════════════════════════════════════════════════════════
# Python entry
# ═══════════════════════════════════════════════════════════════════════

def fused_sparse_conv_lif_forward(
    x, weight, bias, v_prev,
    kernel_size=3, decay=0.5, v_threshold=1.0, threshold=1e-6,
    w_cl=None, counts_buf=None, tile_cin_buf=None,
    return_ms=False,
):
    """
    Fused Sparse Conv + LIF. Reuses conv2d.py prescan infrastructure.

    Returns: (spike_out, v_next, sparse_ms)
    """
    from Kernels.conv2d import _select_tile_sizes, _build_tile_csr

    N, C_IN, H, W = x.shape
    C_OUT = weight.shape[0]
    device = x.device

    BH, BW = _select_tile_sizes(H, W)
    GH = triton.cdiv(H, BH)
    GW = triton.cdiv(W, BW)
    N_TILES = N * GH * GW

    x_f16 = x.half().contiguous()

    if w_cl is not None:
        w_cl_f16 = w_cl
    else:
        if kernel_size == 3:
            w_cl_f16 = weight.half().permute(0, 2, 3, 1).contiguous()
        else:
            w_cl_f16 = weight.half().reshape(C_OUT, C_IN).contiguous()

    if counts_buf is None or counts_buf.numel() < N_TILES:
        counts_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
    if tile_cin_buf is None or tile_cin_buf.numel() < N_TILES * C_IN:
        tile_cin_buf = torch.empty(N_TILES * C_IN, dtype=torch.int32, device=device)

    tile_ptr = _build_tile_csr(
        x_f16, N, C_IN, H, W, BH, BW, GH, GW,
        kernel_size, threshold, counts_buf=counts_buf, tile_cin_buf=tile_cin_buf)

    has_bias = bias is not None
    bias_f32 = bias.float().contiguous() if has_bias else torch.empty(1, device=device)
    v_prev_f32 = v_prev.float().contiguous()
    spike_out = torch.zeros(N, C_OUT, H, W, dtype=torch.float32, device=device)
    v_next = torch.zeros(N, C_OUT, H, W, dtype=torch.float32, device=device)

    sparse_ms = 0.0
    if return_ms:
        se = torch.cuda.Event(enable_timing=True)
        ee = torch.cuda.Event(enable_timing=True)
        se.record()

    def _grid(META):
        return (N_TILES, triton.cdiv(C_OUT, META['BLOCK_N']))

    if BW == 16:
        _k3x3 = fused_conv3x3_lif_kernel_8x16
        _k1x1 = fused_conv1x1_lif_kernel_8x16
    else:
        _k3x3 = fused_conv3x3_lif_kernel_8x8
        _k1x1 = fused_conv1x1_lif_kernel_8x8

    kfn = _k3x3 if kernel_size == 3 else _k1x1
    kfn[_grid](
        x_f16, w_cl_f16, bias_f32, tile_ptr, tile_cin_buf,
        v_prev_f32, spike_out, v_next, N,
        C_IN, C_OUT, H, W, GH, GW,
        HAS_BIAS=has_bias, DECAY=decay, V_TH=v_threshold,
    )

    if return_ms:
        ee.record()
        torch.cuda.synchronize(device)
        sparse_ms = se.elapsed_time(ee)

    return spike_out, v_next, sparse_ms