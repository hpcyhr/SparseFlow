"""
SparseFlow Fused Sparse Conv + LIF — v16.4 Compact Active-Group

改动：
1. 与 conv2d.py 统一使用 active_group_count + active_group_list
2. 与 Ops 层接口对齐，入口函数名统一为 `sparse_fused_conv_lif_forward`
3. 去掉 tl.full_like，兼容较老 Triton 版本
4. 新增 `return_avg_active_ratio`，仅在需要时才同步读取
"""

import torch
import triton
import triton.language as tl
from triton import autotune, Config

from Kernels.conv2d import (
    GROUP_SIZE,
    FALLBACK_RATIO,
    _select_tile_sizes,
    _build_active_group_metadata,
    _check_dense_fallback,
)


def _select_block_sizes(H, W, C_IN, C_OUT, kernel_size, N):
    BH, BW = _select_tile_sizes(H, W)
    return BH, BW, BH * BW, 64, GROUP_SIZE


def _make_fused_configs(bh, bw):
    bm = bh * bw
    cfgs = []
    for bn in [64, 128]:
        for nw in [4, 8]:
            cfgs.append(
                Config(
                    {
                        "BLOCK_M": bm,
                        "BLOCK_N": bn,
                        "BLOCK_H": bh,
                        "BLOCK_W": bw,
                    },
                    num_warps=nw,
                    num_stages=1,
                )
            )
    return cfgs


_FC_8x8 = _make_fused_configs(8, 8)
_FC_8x16 = _make_fused_configs(8, 16)


@autotune(configs=_FC_8x8, key=["C_IN", "C_OUT", "H", "W", "GH", "GW"])
@triton.jit
def fused_ag_conv3x3_lif_8x8(
    x_ptr,
    w_cl_ptr,
    bias_ptr,
    ag_count_ptr,
    ag_list_ptr,
    v_prev_ptr,
    spike_ptr,
    v_next_ptr,
    N_val,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    GH: tl.constexpr,
    GW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    DECAY: tl.constexpr,
    RECIP_TAU: tl.constexpr,
    V_TH: tl.constexpr,
    HAS_V_RESET: tl.constexpr,
    V_RESET: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    MAX_AG: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
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
    m_mask = (out_h < H) & (out_w < W)

    HW: tl.constexpr = H * W
    W_CS: tl.constexpr = C_IN
    W_KH: tl.constexpr = 3 * C_IN
    W_CO: tl.constexpr = 9 * C_IN

    active_count = tl.load(ag_count_ptr + tile_id)
    list_base = tile_id * MAX_AG

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    g_idx = 0
    while g_idx < active_count:
        gid = tl.load(ag_list_ptr + list_base + g_idx)
        cin_start = gid * GROUP_SIZE_C
        offs_k = cin_start + tl.arange(0, GROUP_SIZE_C)
        k_mask = offs_k < C_IN

        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                in_h = out_h + (kh - 1)
                in_w = out_w + (kw - 1)
                h_ok = (in_h >= 0) & (in_h < H)
                w_ok = (in_w >= 0) & (in_w < W)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W - 1)

                x_addrs = (
                    x_ptr
                    + (n_idx * C_IN + offs_k[None, :]) * HW
                    + safe_h[:, None] * W
                    + safe_w[:, None]
                )
                x_mask = k_mask[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                x_tile = tl.load(x_addrs, mask=x_mask, other=0.0).to(tl.float16)

                w_addrs = (
                    w_cl_ptr
                    + offs_n[None, :] * W_CO
                    + kh * W_KH
                    + kw * W_CS
                    + offs_k[:, None]
                )
                w_mask = k_mask[:, None] & n_mask[None, :]
                w_tile = tl.load(w_addrs, mask=w_mask, other=0.0).to(tl.float16)

                acc += tl.dot(x_tile, w_tile)

        g_idx += 1

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]

    oa = (n_idx * C_OUT + offs_n[None, :]) * HW + out_h[:, None] * W + out_w[:, None]
    om = m_mask[:, None] & n_mask[None, :]

    vp = tl.load(v_prev_ptr + oa, mask=om, other=0.0)
    vt = vp * DECAY + acc * RECIP_TAU
    sp = (vt >= V_TH).to(tl.float32)

    if HAS_V_RESET:
        v_reset_tensor = vt * 0.0 + V_RESET
        vn = tl.where(sp > 0.0, v_reset_tensor, vt)
    else:
        vn = vt - sp * V_TH

    tl.store(spike_ptr + oa, sp, mask=om)
    tl.store(v_next_ptr + oa, vn, mask=om)


@autotune(configs=_FC_8x16, key=["C_IN", "C_OUT", "H", "W", "GH", "GW"])
@triton.jit
def fused_ag_conv3x3_lif_8x16(
    x_ptr,
    w_cl_ptr,
    bias_ptr,
    ag_count_ptr,
    ag_list_ptr,
    v_prev_ptr,
    spike_ptr,
    v_next_ptr,
    N_val,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    GH: tl.constexpr,
    GW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    DECAY: tl.constexpr,
    RECIP_TAU: tl.constexpr,
    V_TH: tl.constexpr,
    HAS_V_RESET: tl.constexpr,
    V_RESET: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    MAX_AG: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
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
    m_mask = (out_h < H) & (out_w < W)

    HW: tl.constexpr = H * W
    W_CS: tl.constexpr = C_IN
    W_KH: tl.constexpr = 3 * C_IN
    W_CO: tl.constexpr = 9 * C_IN

    active_count = tl.load(ag_count_ptr + tile_id)
    list_base = tile_id * MAX_AG

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    g_idx = 0
    while g_idx < active_count:
        gid = tl.load(ag_list_ptr + list_base + g_idx)
        cin_start = gid * GROUP_SIZE_C
        offs_k = cin_start + tl.arange(0, GROUP_SIZE_C)
        k_mask = offs_k < C_IN

        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                in_h = out_h + (kh - 1)
                in_w = out_w + (kw - 1)
                h_ok = (in_h >= 0) & (in_h < H)
                w_ok = (in_w >= 0) & (in_w < W)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W - 1)

                x_addrs = (
                    x_ptr
                    + (n_idx * C_IN + offs_k[None, :]) * HW
                    + safe_h[:, None] * W
                    + safe_w[:, None]
                )
                x_mask = k_mask[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                x_tile = tl.load(x_addrs, mask=x_mask, other=0.0).to(tl.float16)

                w_addrs = (
                    w_cl_ptr
                    + offs_n[None, :] * W_CO
                    + kh * W_KH
                    + kw * W_CS
                    + offs_k[:, None]
                )
                w_mask = k_mask[:, None] & n_mask[None, :]
                w_tile = tl.load(w_addrs, mask=w_mask, other=0.0).to(tl.float16)

                acc += tl.dot(x_tile, w_tile)

        g_idx += 1

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]

    oa = (n_idx * C_OUT + offs_n[None, :]) * HW + out_h[:, None] * W + out_w[:, None]
    om = m_mask[:, None] & n_mask[None, :]

    vp = tl.load(v_prev_ptr + oa, mask=om, other=0.0)
    vt = vp * DECAY + acc * RECIP_TAU
    sp = (vt >= V_TH).to(tl.float32)

    if HAS_V_RESET:
        v_reset_tensor = vt * 0.0 + V_RESET
        vn = tl.where(sp > 0.0, v_reset_tensor, vt)
    else:
        vn = vt - sp * V_TH

    tl.store(spike_ptr + oa, sp, mask=om)
    tl.store(v_next_ptr + oa, vn, mask=om)


def sparse_fused_conv_lif_forward(
    x,
    v_prev,
    weight,
    bias,
    tau=2.0,
    v_threshold=1.0,
    v_reset=0.0,
    decay_input=True,
    block_size=None,
    kernel_size=3,
    threshold=1e-6,
    w_cl=None,
    counts_buf=None,       # 兼容旧接口，忽略
    tile_cin_buf=None,     # 兼容旧接口，忽略
    group_flags_buf=None,  # 兼容旧接口，忽略
    ag_count_buf=None,
    ag_list_buf=None,
    return_ms=False,
    fallback_ratio=FALLBACK_RATIO,
    return_avg_active_ratio=False,
):
    import torch.nn.functional as Fn

    N, C_IN, H, W = x.shape
    C_OUT = weight.shape[0]
    device = x.device

    stride = 1
    padding = 1 if kernel_size == 3 else 0
    H_OUT = (H + 2 * padding - kernel_size) // stride + 1
    W_OUT = (W + 2 * padding - kernel_size) // stride + 1

    BH, BW = _select_tile_sizes(H_OUT, W_OUT)
    GH = triton.cdiv(H_OUT, BH)
    GW = triton.cdiv(W_OUT, BW)
    N_TILES = N * GH * GW
    NUM_GROUPS = triton.cdiv(C_IN, GROUP_SIZE)
    MAX_AG = NUM_GROUPS

    if w_cl is not None:
        w_cl_f16 = w_cl
    else:
        if kernel_size == 3:
            w_cl_f16 = weight.half().permute(0, 2, 3, 1).contiguous()
        else:
            w_cl_f16 = weight.half().reshape(C_OUT, C_IN).contiguous()

    # 1x1 直接 dense fallback
    if kernel_size != 3:
        y = Fn.conv2d(x, weight, bias, stride=1, padding=0).float()
        vp = v_prev.float()
        if decay_input:
            vt = vp + (y - (vp - (0.0 if v_reset is None else float(v_reset)))) / float(tau)
        else:
            vt = vp - (vp - (0.0 if v_reset is None else float(v_reset))) / float(tau) + y
        sp = (vt >= v_threshold).float()
        if v_reset is None:
            vn = vt - sp * v_threshold
        else:
            vn = torch.where(sp.bool(), torch.full_like(vt, float(v_reset)), vt)
        if return_avg_active_ratio:
            return sp, vn, 0.0, 1.0
        return sp, vn, 0.0

    x_f16 = x.half().contiguous()

    if ag_count_buf is None or ag_count_buf.numel() < N_TILES:
        ag_count_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
    if ag_list_buf is None or ag_list_buf.numel() < N_TILES * MAX_AG:
        ag_list_buf = torch.empty(N_TILES * MAX_AG, dtype=torch.int32, device=device)

    _build_active_group_metadata(
        x_f16,
        N,
        C_IN,
        H,
        W,
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
        ag_count_buf,
        ag_list_buf,
    )

    avg_active_ratio = None
    if return_avg_active_ratio:
        avg_active_ratio = (
            ag_count_buf[:N_TILES].float().mean().item() / max(NUM_GROUPS, 1)
        )

    if (avg_active_ratio is not None and avg_active_ratio > fallback_ratio) or (
        avg_active_ratio is None and _check_dense_fallback(ag_count_buf, N_TILES, NUM_GROUPS, fallback_ratio)
    ):
        y = Fn.conv2d(x, weight, bias, stride=1, padding=1).float()
        vp = v_prev.float()
        if decay_input:
            vt = vp + (y - (vp - (0.0 if v_reset is None else float(v_reset)))) / float(tau)
        else:
            vt = vp - (vp - (0.0 if v_reset is None else float(v_reset))) / float(tau) + y
        sp = (vt >= v_threshold).float()
        if v_reset is None:
            vn = vt - sp * v_threshold
        else:
            vn = torch.where(sp.bool(), torch.full_like(vt, float(v_reset)), vt)
        if return_avg_active_ratio:
            return sp, vn, 0.0, avg_active_ratio
        return sp, vn, 0.0

    has_bias = bias is not None
    bias_f32 = bias.float().contiguous() if has_bias else torch.empty(1, device=device)
    v_prev_f32 = v_prev.float().contiguous()
    spike_out = torch.empty(N, C_OUT, H_OUT, W_OUT, dtype=torch.float32, device=device)
    v_next = torch.empty(N, C_OUT, H_OUT, W_OUT, dtype=torch.float32, device=device)

    sparse_ms = 0.0
    if return_ms:
        se = torch.cuda.Event(enable_timing=True)
        ee = torch.cuda.Event(enable_timing=True)
        se.record()

    def _grid(META):
        return (N_TILES, triton.cdiv(C_OUT, META["BLOCK_N"]))

    decay = 1.0 - 1.0 / float(tau)
    recip_tau = 1.0 / float(tau)
    has_v_reset = v_reset is not None
    v_reset_val = 0.0 if v_reset is None else float(v_reset)

    kernel = fused_ag_conv3x3_lif_8x16 if BW == 16 else fused_ag_conv3x3_lif_8x8
    kernel[_grid](
        x_f16,
        w_cl_f16,
        bias_f32,
        ag_count_buf,
        ag_list_buf,
        v_prev_f32,
        spike_out,
        v_next,
        N,
        C_IN, C_OUT, H_OUT, W_OUT, GH, GW,
        HAS_BIAS=has_bias,
        DECAY=decay,
        RECIP_TAU=recip_tau,
        V_TH=float(v_threshold),
        HAS_V_RESET=has_v_reset,
        V_RESET=v_reset_val,
        GROUP_SIZE_C=GROUP_SIZE,
        MAX_AG=MAX_AG,
    )

    if return_ms:
        ee.record()
        torch.cuda.synchronize(device)
        sparse_ms = se.elapsed_time(ee)

    if return_avg_active_ratio:
        return spike_out, v_next, sparse_ms, avg_active_ratio
    return spike_out, v_next, sparse_ms