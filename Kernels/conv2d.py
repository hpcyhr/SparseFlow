"""
SparseFlow Conv2d Triton Kernels — v17.0 Multi-Pattern Compact Active-Group

实现：
1. 1x1, stride=1, pad=0, groups=1 的 sparse kernel
2. 3x3, stride=1, pad=1, groups=1 的 sparse kernel
3. 3x3, stride=2, pad=1, groups=1 的 sparse kernel
4. 统一 active-group metadata 构建与 dispatch
5. 非支持配置安全回退到 dense conv2d
"""

import torch
import triton
import triton.language as tl
from triton import autotune, Config

GROUP_SIZE = 32
BLOCK_G = 4
FALLBACK_RATIO = 0.85


def _select_tile_sizes(H, W):
    pixels = H * W
    if pixels >= 3136:
        return 8, 16
    return 8, 8


def _select_block_sizes(H, W, C_IN, C_OUT, kernel_size, N):
    BH, BW = _select_tile_sizes(H, W)
    return BH, BW, BH * BW, 64, GROUP_SIZE


@triton.jit
def prescan_active_groups_kernel(
    x_ptr,
    ag_count_ptr,
    ag_list_ptr,
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
):
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
    list_base = tile_id * MAX_AG

    off1 = tl.arange(0, 1)
    count = tl.zeros([1], dtype=tl.int32)

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

        is_active = group_max > THRESHOLD
        can_write = count < MAX_AG
        write_mask = is_active & can_write

        write_ptr = ag_list_ptr + list_base + count
        g_val = tl.full([1], g, dtype=tl.int32)
        tl.store(write_ptr, g_val, mask=write_mask)
        count = count + is_active.to(tl.int32)

    tl.store(ag_count_ptr + tile_id + off1, count)


@torch.no_grad()
def _build_active_group_metadata(
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
    ag_count_buf,
    ag_list_buf,
):
    N_TILES = N * GH * GW
    NUM_GROUPS = triton.cdiv(C_IN, GROUP_SIZE)
    MAX_AG = NUM_GROUPS

    rf_h = (BH - 1) * stride + kernel_size
    rf_w = (BW - 1) * stride + kernel_size
    rf_actual = rf_h * rf_w
    RF_SIZE = triton.next_power_of_2(rf_actual)

    prescan_active_groups_kernel[(N_TILES,)](
        x_f16,
        ag_count_buf,
        ag_list_buf,
        N,
        C_IN,
        H_IN,
        W_IN,
        H_OUT,
        W_OUT,
        GH,
        GW,
        BLOCK_H=BH,
        BLOCK_W=BW,
        KERNEL_SIZE=kernel_size,
        STRIDE=stride,
        PAD=padding,
        GROUP_SIZE_C=GROUP_SIZE,
        NUM_GROUPS=NUM_GROUPS,
        MAX_AG=MAX_AG,
        RF_SIZE=RF_SIZE,
        THRESHOLD=threshold,
    )

    return NUM_GROUPS, MAX_AG


def _check_dense_fallback(ag_count_buf, N_TILES, NUM_GROUPS, fallback_ratio=FALLBACK_RATIO):
    avg_active = ag_count_buf[:N_TILES].float().mean()
    threshold = fallback_ratio * NUM_GROUPS
    return avg_active.item() > threshold


# autotune configs

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


@autotune(configs=_CONFIGS_8x8, key=["C_IN", "C_OUT", "H_OUT", "W_OUT", "GH", "GW"])
@triton.jit
def sparse_conv1x1_ag_kernel_8x8(
    x_ptr, w_cl_ptr, bias_ptr, ag_count_ptr, ag_list_ptr, y_ptr, N_val,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    H_IN: tl.constexpr, W_IN: tl.constexpr,
    H_OUT: tl.constexpr, W_OUT: tl.constexpr,
    GH: tl.constexpr, GW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, MAX_AG: tl.constexpr,
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

    active_count = tl.load(ag_count_ptr + tile_id)
    list_base = tile_id * MAX_AG
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    g_idx = 0
    while g_idx < active_count:
        offs_g = g_idx + tl.arange(0, BLOCK_G)
        g_mask = offs_g < active_count
        gids = tl.load(ag_list_ptr + list_base + offs_g, mask=g_mask, other=0)

        gid0 = tl.load(ag_list_ptr + list_base + g_idx + 0, mask=(g_idx + 0) < active_count, other=0)
        gid1 = tl.load(ag_list_ptr + list_base + g_idx + 1, mask=(g_idx + 1) < active_count, other=0)
        gid2 = tl.load(ag_list_ptr + list_base + g_idx + 2, mask=(g_idx + 2) < active_count, other=0)
        gid3 = tl.load(ag_list_ptr + list_base + g_idx + 3, mask=(g_idx + 3) < active_count, other=0)

        valid0 = (g_idx + 0) < active_count
        valid1 = (g_idx + 1) < active_count
        valid2 = (g_idx + 2) < active_count
        valid3 = (g_idx + 3) < active_count

        cin_start0 = gid0 * GROUP_SIZE_C
        offs_k0 = cin_start0 + tl.arange(0, GROUP_SIZE_C)
        k_mask0 = valid0 & (offs_k0 < C_IN)
        x_addrs0 = (
            x_ptr
            + (n_idx * C_IN + offs_k0[None, :]) * HW_IN
            + out_h[:, None] * W_IN
            + out_w[:, None]
        )
        x_mask0 = k_mask0[None, :] & m_mask[:, None]
        x_tile0 = tl.load(x_addrs0, mask=x_mask0, other=0.0).to(tl.float16)
        w_addrs0 = w_cl_ptr + offs_n[None, :] * C_IN + offs_k0[:, None]
        w_mask0 = k_mask0[:, None] & n_mask[None, :]
        w_tile0 = tl.load(w_addrs0, mask=w_mask0, other=0.0).to(tl.float16)
        acc += tl.dot(x_tile0, w_tile0)

        cin_start1 = gid1 * GROUP_SIZE_C
        offs_k1 = cin_start1 + tl.arange(0, GROUP_SIZE_C)
        k_mask1 = valid1 & (offs_k1 < C_IN)
        x_addrs1 = (
            x_ptr
            + (n_idx * C_IN + offs_k1[None, :]) * HW_IN
            + out_h[:, None] * W_IN
            + out_w[:, None]
        )
        x_mask1 = k_mask1[None, :] & m_mask[:, None]
        x_tile1 = tl.load(x_addrs1, mask=x_mask1, other=0.0).to(tl.float16)
        w_addrs1 = w_cl_ptr + offs_n[None, :] * C_IN + offs_k1[:, None]
        w_mask1 = k_mask1[:, None] & n_mask[None, :]
        w_tile1 = tl.load(w_addrs1, mask=w_mask1, other=0.0).to(tl.float16)
        acc += tl.dot(x_tile1, w_tile1)

        cin_start2 = gid2 * GROUP_SIZE_C
        offs_k2 = cin_start2 + tl.arange(0, GROUP_SIZE_C)
        k_mask2 = valid2 & (offs_k2 < C_IN)
        x_addrs2 = (
            x_ptr
            + (n_idx * C_IN + offs_k2[None, :]) * HW_IN
            + out_h[:, None] * W_IN
            + out_w[:, None]
        )
        x_mask2 = k_mask2[None, :] & m_mask[:, None]
        x_tile2 = tl.load(x_addrs2, mask=x_mask2, other=0.0).to(tl.float16)
        w_addrs2 = w_cl_ptr + offs_n[None, :] * C_IN + offs_k2[:, None]
        w_mask2 = k_mask2[:, None] & n_mask[None, :]
        w_tile2 = tl.load(w_addrs2, mask=w_mask2, other=0.0).to(tl.float16)
        acc += tl.dot(x_tile2, w_tile2)

        cin_start3 = gid3 * GROUP_SIZE_C
        offs_k3 = cin_start3 + tl.arange(0, GROUP_SIZE_C)
        k_mask3 = valid3 & (offs_k3 < C_IN)
        x_addrs3 = (
            x_ptr
            + (n_idx * C_IN + offs_k3[None, :]) * HW_IN
            + out_h[:, None] * W_IN
            + out_w[:, None]
        )
        x_mask3 = k_mask3[None, :] & m_mask[:, None]
        x_tile3 = tl.load(x_addrs3, mask=x_mask3, other=0.0).to(tl.float16)
        w_addrs3 = w_cl_ptr + offs_n[None, :] * C_IN + offs_k3[:, None]
        w_mask3 = k_mask3[:, None] & n_mask[None, :]
        w_tile3 = tl.load(w_addrs3, mask=w_mask3, other=0.0).to(tl.float16)
        acc += tl.dot(x_tile3, w_tile3)

        g_idx += BLOCK_G

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]

    out_addrs = (
        y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
    )
    tl.store(out_addrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@autotune(configs=_CONFIGS_8x16, key=["C_IN", "C_OUT", "H_OUT", "W_OUT", "GH", "GW"])
@triton.jit
def sparse_conv1x1_ag_kernel_8x16(
    x_ptr, w_cl_ptr, bias_ptr, ag_count_ptr, ag_list_ptr, y_ptr, N_val,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    H_IN: tl.constexpr, W_IN: tl.constexpr,
    H_OUT: tl.constexpr, W_OUT: tl.constexpr,
    GH: tl.constexpr, GW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, MAX_AG: tl.constexpr,
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

    active_count = tl.load(ag_count_ptr + tile_id)
    list_base = tile_id * MAX_AG
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    g_idx = 0
    while g_idx < active_count:
        offs_g = g_idx + tl.arange(0, BLOCK_G)
        g_mask = offs_g < active_count
        gids = tl.load(ag_list_ptr + list_base + offs_g, mask=g_mask, other=0)

        gid0 = tl.load(ag_list_ptr + list_base + g_idx + 0, mask=(g_idx + 0) < active_count, other=0)
        gid1 = tl.load(ag_list_ptr + list_base + g_idx + 1, mask=(g_idx + 1) < active_count, other=0)
        gid2 = tl.load(ag_list_ptr + list_base + g_idx + 2, mask=(g_idx + 2) < active_count, other=0)
        gid3 = tl.load(ag_list_ptr + list_base + g_idx + 3, mask=(g_idx + 3) < active_count, other=0)

        valid0 = (g_idx + 0) < active_count
        valid1 = (g_idx + 1) < active_count
        valid2 = (g_idx + 2) < active_count
        valid3 = (g_idx + 3) < active_count

        cin_start0 = gid0 * GROUP_SIZE_C
        offs_k0 = cin_start0 + tl.arange(0, GROUP_SIZE_C)
        k_mask0 = valid0 & (offs_k0 < C_IN)
        x_addrs0 = x_ptr + (n_idx * C_IN + offs_k0[None, :]) * HW_IN + out_h[:, None] * W_IN + out_w[:, None]
        x_mask0 = k_mask0[None, :] & m_mask[:, None]
        x_tile0 = tl.load(x_addrs0, mask=x_mask0, other=0.0).to(tl.float16)
        w_addrs0 = w_cl_ptr + offs_n[None, :] * C_IN + offs_k0[:, None]
        w_mask0 = k_mask0[:, None] & n_mask[None, :]
        w_tile0 = tl.load(w_addrs0, mask=w_mask0, other=0.0).to(tl.float16)
        acc += tl.dot(x_tile0, w_tile0)

        cin_start1 = gid1 * GROUP_SIZE_C
        offs_k1 = cin_start1 + tl.arange(0, GROUP_SIZE_C)
        k_mask1 = valid1 & (offs_k1 < C_IN)
        x_addrs1 = x_ptr + (n_idx * C_IN + offs_k1[None, :]) * HW_IN + out_h[:, None] * W_IN + out_w[:, None]
        x_mask1 = k_mask1[None, :] & m_mask[:, None]
        x_tile1 = tl.load(x_addrs1, mask=x_mask1, other=0.0).to(tl.float16)
        w_addrs1 = w_cl_ptr + offs_n[None, :] * C_IN + offs_k1[:, None]
        w_mask1 = k_mask1[:, None] & n_mask[None, :]
        w_tile1 = tl.load(w_addrs1, mask=w_mask1, other=0.0).to(tl.float16)
        acc += tl.dot(x_tile1, w_tile1)

        cin_start2 = gid2 * GROUP_SIZE_C
        offs_k2 = cin_start2 + tl.arange(0, GROUP_SIZE_C)
        k_mask2 = valid2 & (offs_k2 < C_IN)
        x_addrs2 = x_ptr + (n_idx * C_IN + offs_k2[None, :]) * HW_IN + out_h[:, None] * W_IN + out_w[:, None]
        x_mask2 = k_mask2[None, :] & m_mask[:, None]
        x_tile2 = tl.load(x_addrs2, mask=x_mask2, other=0.0).to(tl.float16)
        w_addrs2 = w_cl_ptr + offs_n[None, :] * C_IN + offs_k2[:, None]
        w_mask2 = k_mask2[:, None] & n_mask[None, :]
        w_tile2 = tl.load(w_addrs2, mask=w_mask2, other=0.0).to(tl.float16)
        acc += tl.dot(x_tile2, w_tile2)

        cin_start3 = gid3 * GROUP_SIZE_C
        offs_k3 = cin_start3 + tl.arange(0, GROUP_SIZE_C)
        k_mask3 = valid3 & (offs_k3 < C_IN)
        x_addrs3 = x_ptr + (n_idx * C_IN + offs_k3[None, :]) * HW_IN + out_h[:, None] * W_IN + out_w[:, None]
        x_mask3 = k_mask3[None, :] & m_mask[:, None]
        x_tile3 = tl.load(x_addrs3, mask=x_mask3, other=0.0).to(tl.float16)
        w_addrs3 = w_cl_ptr + offs_n[None, :] * C_IN + offs_k3[:, None]
        w_mask3 = k_mask3[:, None] & n_mask[None, :]
        w_tile3 = tl.load(w_addrs3, mask=w_mask3, other=0.0).to(tl.float16)
        acc += tl.dot(x_tile3, w_tile3)
        g_idx += BLOCK_G

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]

    out_addrs = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
    tl.store(out_addrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@autotune(configs=_CONFIGS_8x8, key=["C_IN", "C_OUT", "H_OUT", "W_OUT", "GH", "GW"])
@triton.jit
def sparse_conv3x3s1_ag_kernel_8x8(
    x_ptr, w_cl_ptr, bias_ptr, ag_count_ptr, ag_list_ptr, y_ptr, N_val,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    H_IN: tl.constexpr, W_IN: tl.constexpr,
    H_OUT: tl.constexpr, W_OUT: tl.constexpr,
    GH: tl.constexpr, GW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, MAX_AG: tl.constexpr,
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

    active_count = tl.load(ag_count_ptr + tile_id)
    list_base = tile_id * MAX_AG
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    g_idx = 0
    while g_idx < active_count:
        offs_g = g_idx + tl.arange(0, BLOCK_G)
        g_mask = offs_g < active_count
        gids = tl.load(ag_list_ptr + list_base + offs_g, mask=g_mask, other=0)

        gid0 = tl.load(ag_list_ptr + list_base + g_idx + 0, mask=(g_idx + 0) < active_count, other=0)
        gid1 = tl.load(ag_list_ptr + list_base + g_idx + 1, mask=(g_idx + 1) < active_count, other=0)
        gid2 = tl.load(ag_list_ptr + list_base + g_idx + 2, mask=(g_idx + 2) < active_count, other=0)
        gid3 = tl.load(ag_list_ptr + list_base + g_idx + 3, mask=(g_idx + 3) < active_count, other=0)

        valid0 = (g_idx + 0) < active_count
        valid1 = (g_idx + 1) < active_count
        valid2 = (g_idx + 2) < active_count
        valid3 = (g_idx + 3) < active_count

        cin_start0 = gid0 * GROUP_SIZE_C
        offs_k0 = cin_start0 + tl.arange(0, GROUP_SIZE_C)
        k_mask0 = valid0 & (offs_k0 < C_IN)
        cin_start1 = gid1 * GROUP_SIZE_C
        offs_k1 = cin_start1 + tl.arange(0, GROUP_SIZE_C)
        k_mask1 = valid1 & (offs_k1 < C_IN)
        cin_start2 = gid2 * GROUP_SIZE_C
        offs_k2 = cin_start2 + tl.arange(0, GROUP_SIZE_C)
        k_mask2 = valid2 & (offs_k2 < C_IN)
        cin_start3 = gid3 * GROUP_SIZE_C
        offs_k3 = cin_start3 + tl.arange(0, GROUP_SIZE_C)
        k_mask3 = valid3 & (offs_k3 < C_IN)

        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                in_h = out_h + (kh - 1)
                in_w = out_w + (kw - 1)
                h_ok = (in_h >= 0) & (in_h < H_IN)
                w_ok = (in_w >= 0) & (in_w < W_IN)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H_IN - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W_IN - 1)

                x_addrs0 = x_ptr + (n_idx * C_IN + offs_k0[None, :]) * HW_IN + safe_h[:, None] * W_IN + safe_w[:, None]
                x_mask0 = k_mask0[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                x_tile0 = tl.load(x_addrs0, mask=x_mask0, other=0.0).to(tl.float16)
                w_addrs0 = w_cl_ptr + offs_n[None, :] * W_CO + kh * W_KH + kw * W_CS + offs_k0[:, None]
                w_mask0 = k_mask0[:, None] & n_mask[None, :]
                w_tile0 = tl.load(w_addrs0, mask=w_mask0, other=0.0).to(tl.float16)
                acc += tl.dot(x_tile0, w_tile0)

                x_addrs1 = x_ptr + (n_idx * C_IN + offs_k1[None, :]) * HW_IN + safe_h[:, None] * W_IN + safe_w[:, None]
                x_mask1 = k_mask1[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                x_tile1 = tl.load(x_addrs1, mask=x_mask1, other=0.0).to(tl.float16)
                w_addrs1 = w_cl_ptr + offs_n[None, :] * W_CO + kh * W_KH + kw * W_CS + offs_k1[:, None]
                w_mask1 = k_mask1[:, None] & n_mask[None, :]
                w_tile1 = tl.load(w_addrs1, mask=w_mask1, other=0.0).to(tl.float16)
                acc += tl.dot(x_tile1, w_tile1)

                x_addrs2 = x_ptr + (n_idx * C_IN + offs_k2[None, :]) * HW_IN + safe_h[:, None] * W_IN + safe_w[:, None]
                x_mask2 = k_mask2[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                x_tile2 = tl.load(x_addrs2, mask=x_mask2, other=0.0).to(tl.float16)
                w_addrs2 = w_cl_ptr + offs_n[None, :] * W_CO + kh * W_KH + kw * W_CS + offs_k2[:, None]
                w_mask2 = k_mask2[:, None] & n_mask[None, :]
                w_tile2 = tl.load(w_addrs2, mask=w_mask2, other=0.0).to(tl.float16)
                acc += tl.dot(x_tile2, w_tile2)

                x_addrs3 = x_ptr + (n_idx * C_IN + offs_k3[None, :]) * HW_IN + safe_h[:, None] * W_IN + safe_w[:, None]
                x_mask3 = k_mask3[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                x_tile3 = tl.load(x_addrs3, mask=x_mask3, other=0.0).to(tl.float16)
                w_addrs3 = w_cl_ptr + offs_n[None, :] * W_CO + kh * W_KH + kw * W_CS + offs_k3[:, None]
                w_mask3 = k_mask3[:, None] & n_mask[None, :]
                w_tile3 = tl.load(w_addrs3, mask=w_mask3, other=0.0).to(tl.float16)
                acc += tl.dot(x_tile3, w_tile3)
        g_idx += BLOCK_G

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]

    out_addrs = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
    tl.store(out_addrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@autotune(configs=_CONFIGS_8x16, key=["C_IN", "C_OUT", "H_OUT", "W_OUT", "GH", "GW"])
@triton.jit
def sparse_conv3x3s1_ag_kernel_8x16(
    x_ptr, w_cl_ptr, bias_ptr, ag_count_ptr, ag_list_ptr, y_ptr, N_val,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    H_IN: tl.constexpr, W_IN: tl.constexpr,
    H_OUT: tl.constexpr, W_OUT: tl.constexpr,
    GH: tl.constexpr, GW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, MAX_AG: tl.constexpr,
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

    active_count = tl.load(ag_count_ptr + tile_id)
    list_base = tile_id * MAX_AG
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    g_idx = 0
    while g_idx < active_count:
        offs_g = g_idx + tl.arange(0, BLOCK_G)
        g_mask = offs_g < active_count
        gids = tl.load(ag_list_ptr + list_base + offs_g, mask=g_mask, other=0)

        gid0 = tl.load(ag_list_ptr + list_base + g_idx + 0, mask=(g_idx + 0) < active_count, other=0)
        gid1 = tl.load(ag_list_ptr + list_base + g_idx + 1, mask=(g_idx + 1) < active_count, other=0)
        gid2 = tl.load(ag_list_ptr + list_base + g_idx + 2, mask=(g_idx + 2) < active_count, other=0)
        gid3 = tl.load(ag_list_ptr + list_base + g_idx + 3, mask=(g_idx + 3) < active_count, other=0)

        valid0 = (g_idx + 0) < active_count
        valid1 = (g_idx + 1) < active_count
        valid2 = (g_idx + 2) < active_count
        valid3 = (g_idx + 3) < active_count

        cin_start0 = gid0 * GROUP_SIZE_C
        offs_k0 = cin_start0 + tl.arange(0, GROUP_SIZE_C)
        k_mask0 = valid0 & (offs_k0 < C_IN)
        cin_start1 = gid1 * GROUP_SIZE_C
        offs_k1 = cin_start1 + tl.arange(0, GROUP_SIZE_C)
        k_mask1 = valid1 & (offs_k1 < C_IN)
        cin_start2 = gid2 * GROUP_SIZE_C
        offs_k2 = cin_start2 + tl.arange(0, GROUP_SIZE_C)
        k_mask2 = valid2 & (offs_k2 < C_IN)
        cin_start3 = gid3 * GROUP_SIZE_C
        offs_k3 = cin_start3 + tl.arange(0, GROUP_SIZE_C)
        k_mask3 = valid3 & (offs_k3 < C_IN)

        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                in_h = out_h + (kh - 1)
                in_w = out_w + (kw - 1)
                h_ok = (in_h >= 0) & (in_h < H_IN)
                w_ok = (in_w >= 0) & (in_w < W_IN)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H_IN - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W_IN - 1)

                x_addrs0 = x_ptr + (n_idx * C_IN + offs_k0[None, :]) * HW_IN + safe_h[:, None] * W_IN + safe_w[:, None]
                x_mask0 = k_mask0[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                x_tile0 = tl.load(x_addrs0, mask=x_mask0, other=0.0).to(tl.float16)
                w_addrs0 = w_cl_ptr + offs_n[None, :] * W_CO + kh * W_KH + kw * W_CS + offs_k0[:, None]
                w_mask0 = k_mask0[:, None] & n_mask[None, :]
                w_tile0 = tl.load(w_addrs0, mask=w_mask0, other=0.0).to(tl.float16)
                acc += tl.dot(x_tile0, w_tile0)

                x_addrs1 = x_ptr + (n_idx * C_IN + offs_k1[None, :]) * HW_IN + safe_h[:, None] * W_IN + safe_w[:, None]
                x_mask1 = k_mask1[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                x_tile1 = tl.load(x_addrs1, mask=x_mask1, other=0.0).to(tl.float16)
                w_addrs1 = w_cl_ptr + offs_n[None, :] * W_CO + kh * W_KH + kw * W_CS + offs_k1[:, None]
                w_mask1 = k_mask1[:, None] & n_mask[None, :]
                w_tile1 = tl.load(w_addrs1, mask=w_mask1, other=0.0).to(tl.float16)
                acc += tl.dot(x_tile1, w_tile1)

                x_addrs2 = x_ptr + (n_idx * C_IN + offs_k2[None, :]) * HW_IN + safe_h[:, None] * W_IN + safe_w[:, None]
                x_mask2 = k_mask2[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                x_tile2 = tl.load(x_addrs2, mask=x_mask2, other=0.0).to(tl.float16)
                w_addrs2 = w_cl_ptr + offs_n[None, :] * W_CO + kh * W_KH + kw * W_CS + offs_k2[:, None]
                w_mask2 = k_mask2[:, None] & n_mask[None, :]
                w_tile2 = tl.load(w_addrs2, mask=w_mask2, other=0.0).to(tl.float16)
                acc += tl.dot(x_tile2, w_tile2)

                x_addrs3 = x_ptr + (n_idx * C_IN + offs_k3[None, :]) * HW_IN + safe_h[:, None] * W_IN + safe_w[:, None]
                x_mask3 = k_mask3[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                x_tile3 = tl.load(x_addrs3, mask=x_mask3, other=0.0).to(tl.float16)
                w_addrs3 = w_cl_ptr + offs_n[None, :] * W_CO + kh * W_KH + kw * W_CS + offs_k3[:, None]
                w_mask3 = k_mask3[:, None] & n_mask[None, :]
                w_tile3 = tl.load(w_addrs3, mask=w_mask3, other=0.0).to(tl.float16)
                acc += tl.dot(x_tile3, w_tile3)
        g_idx += BLOCK_G

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]

    out_addrs = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
    tl.store(out_addrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@autotune(configs=_CONFIGS_8x8, key=["C_IN", "C_OUT", "H_OUT", "W_OUT", "GH", "GW"])
@triton.jit
def sparse_conv3x3s2_ag_kernel_8x8(
    x_ptr, w_cl_ptr, bias_ptr, ag_count_ptr, ag_list_ptr, y_ptr, N_val,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    H_IN: tl.constexpr, W_IN: tl.constexpr,
    H_OUT: tl.constexpr, W_OUT: tl.constexpr,
    GH: tl.constexpr, GW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, MAX_AG: tl.constexpr,
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

    active_count = tl.load(ag_count_ptr + tile_id)
    list_base = tile_id * MAX_AG
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    g_idx = 0
    while g_idx < active_count:
        offs_g = g_idx + tl.arange(0, BLOCK_G)
        g_mask = offs_g < active_count
        gids = tl.load(ag_list_ptr + list_base + offs_g, mask=g_mask, other=0)

        gid0 = tl.load(ag_list_ptr + list_base + g_idx + 0, mask=(g_idx + 0) < active_count, other=0)
        gid1 = tl.load(ag_list_ptr + list_base + g_idx + 1, mask=(g_idx + 1) < active_count, other=0)
        gid2 = tl.load(ag_list_ptr + list_base + g_idx + 2, mask=(g_idx + 2) < active_count, other=0)
        gid3 = tl.load(ag_list_ptr + list_base + g_idx + 3, mask=(g_idx + 3) < active_count, other=0)

        valid0 = (g_idx + 0) < active_count
        valid1 = (g_idx + 1) < active_count
        valid2 = (g_idx + 2) < active_count
        valid3 = (g_idx + 3) < active_count

        cin_start0 = gid0 * GROUP_SIZE_C
        offs_k0 = cin_start0 + tl.arange(0, GROUP_SIZE_C)
        k_mask0 = valid0 & (offs_k0 < C_IN)
        cin_start1 = gid1 * GROUP_SIZE_C
        offs_k1 = cin_start1 + tl.arange(0, GROUP_SIZE_C)
        k_mask1 = valid1 & (offs_k1 < C_IN)
        cin_start2 = gid2 * GROUP_SIZE_C
        offs_k2 = cin_start2 + tl.arange(0, GROUP_SIZE_C)
        k_mask2 = valid2 & (offs_k2 < C_IN)
        cin_start3 = gid3 * GROUP_SIZE_C
        offs_k3 = cin_start3 + tl.arange(0, GROUP_SIZE_C)
        k_mask3 = valid3 & (offs_k3 < C_IN)

        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                in_h = out_h * 2 + (kh - 1)
                in_w = out_w * 2 + (kw - 1)
                h_ok = (in_h >= 0) & (in_h < H_IN)
                w_ok = (in_w >= 0) & (in_w < W_IN)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H_IN - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W_IN - 1)

                x_addrs0 = x_ptr + (n_idx * C_IN + offs_k0[None, :]) * HW_IN + safe_h[:, None] * W_IN + safe_w[:, None]
                x_mask0 = k_mask0[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                x_tile0 = tl.load(x_addrs0, mask=x_mask0, other=0.0).to(tl.float16)
                w_addrs0 = w_cl_ptr + offs_n[None, :] * W_CO + kh * W_KH + kw * W_CS + offs_k0[:, None]
                w_mask0 = k_mask0[:, None] & n_mask[None, :]
                w_tile0 = tl.load(w_addrs0, mask=w_mask0, other=0.0).to(tl.float16)
                acc += tl.dot(x_tile0, w_tile0)

                x_addrs1 = x_ptr + (n_idx * C_IN + offs_k1[None, :]) * HW_IN + safe_h[:, None] * W_IN + safe_w[:, None]
                x_mask1 = k_mask1[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                x_tile1 = tl.load(x_addrs1, mask=x_mask1, other=0.0).to(tl.float16)
                w_addrs1 = w_cl_ptr + offs_n[None, :] * W_CO + kh * W_KH + kw * W_CS + offs_k1[:, None]
                w_mask1 = k_mask1[:, None] & n_mask[None, :]
                w_tile1 = tl.load(w_addrs1, mask=w_mask1, other=0.0).to(tl.float16)
                acc += tl.dot(x_tile1, w_tile1)

                x_addrs2 = x_ptr + (n_idx * C_IN + offs_k2[None, :]) * HW_IN + safe_h[:, None] * W_IN + safe_w[:, None]
                x_mask2 = k_mask2[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                x_tile2 = tl.load(x_addrs2, mask=x_mask2, other=0.0).to(tl.float16)
                w_addrs2 = w_cl_ptr + offs_n[None, :] * W_CO + kh * W_KH + kw * W_CS + offs_k2[:, None]
                w_mask2 = k_mask2[:, None] & n_mask[None, :]
                w_tile2 = tl.load(w_addrs2, mask=w_mask2, other=0.0).to(tl.float16)
                acc += tl.dot(x_tile2, w_tile2)

                x_addrs3 = x_ptr + (n_idx * C_IN + offs_k3[None, :]) * HW_IN + safe_h[:, None] * W_IN + safe_w[:, None]
                x_mask3 = k_mask3[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                x_tile3 = tl.load(x_addrs3, mask=x_mask3, other=0.0).to(tl.float16)
                w_addrs3 = w_cl_ptr + offs_n[None, :] * W_CO + kh * W_KH + kw * W_CS + offs_k3[:, None]
                w_mask3 = k_mask3[:, None] & n_mask[None, :]
                w_tile3 = tl.load(w_addrs3, mask=w_mask3, other=0.0).to(tl.float16)
                acc += tl.dot(x_tile3, w_tile3)
        g_idx += BLOCK_G

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]

    out_addrs = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
    tl.store(out_addrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@autotune(configs=_CONFIGS_8x16, key=["C_IN", "C_OUT", "H_OUT", "W_OUT", "GH", "GW"])
@triton.jit
def sparse_conv3x3s2_ag_kernel_8x16(
    x_ptr, w_cl_ptr, bias_ptr, ag_count_ptr, ag_list_ptr, y_ptr, N_val,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    H_IN: tl.constexpr, W_IN: tl.constexpr,
    H_OUT: tl.constexpr, W_OUT: tl.constexpr,
    GH: tl.constexpr, GW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, MAX_AG: tl.constexpr,
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

    active_count = tl.load(ag_count_ptr + tile_id)
    list_base = tile_id * MAX_AG
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    g_idx = 0
    while g_idx < active_count:
        offs_g = g_idx + tl.arange(0, BLOCK_G)
        g_mask = offs_g < active_count
        gids = tl.load(ag_list_ptr + list_base + offs_g, mask=g_mask, other=0)

        gid0 = tl.load(ag_list_ptr + list_base + g_idx + 0, mask=(g_idx + 0) < active_count, other=0)
        gid1 = tl.load(ag_list_ptr + list_base + g_idx + 1, mask=(g_idx + 1) < active_count, other=0)
        gid2 = tl.load(ag_list_ptr + list_base + g_idx + 2, mask=(g_idx + 2) < active_count, other=0)
        gid3 = tl.load(ag_list_ptr + list_base + g_idx + 3, mask=(g_idx + 3) < active_count, other=0)

        valid0 = (g_idx + 0) < active_count
        valid1 = (g_idx + 1) < active_count
        valid2 = (g_idx + 2) < active_count
        valid3 = (g_idx + 3) < active_count

        cin_start0 = gid0 * GROUP_SIZE_C
        offs_k0 = cin_start0 + tl.arange(0, GROUP_SIZE_C)
        k_mask0 = valid0 & (offs_k0 < C_IN)
        cin_start1 = gid1 * GROUP_SIZE_C
        offs_k1 = cin_start1 + tl.arange(0, GROUP_SIZE_C)
        k_mask1 = valid1 & (offs_k1 < C_IN)
        cin_start2 = gid2 * GROUP_SIZE_C
        offs_k2 = cin_start2 + tl.arange(0, GROUP_SIZE_C)
        k_mask2 = valid2 & (offs_k2 < C_IN)
        cin_start3 = gid3 * GROUP_SIZE_C
        offs_k3 = cin_start3 + tl.arange(0, GROUP_SIZE_C)
        k_mask3 = valid3 & (offs_k3 < C_IN)

        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                in_h = out_h * 2 + (kh - 1)
                in_w = out_w * 2 + (kw - 1)
                h_ok = (in_h >= 0) & (in_h < H_IN)
                w_ok = (in_w >= 0) & (in_w < W_IN)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H_IN - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W_IN - 1)

                x_addrs0 = x_ptr + (n_idx * C_IN + offs_k0[None, :]) * HW_IN + safe_h[:, None] * W_IN + safe_w[:, None]
                x_mask0 = k_mask0[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                x_tile0 = tl.load(x_addrs0, mask=x_mask0, other=0.0).to(tl.float16)
                w_addrs0 = w_cl_ptr + offs_n[None, :] * W_CO + kh * W_KH + kw * W_CS + offs_k0[:, None]
                w_mask0 = k_mask0[:, None] & n_mask[None, :]
                w_tile0 = tl.load(w_addrs0, mask=w_mask0, other=0.0).to(tl.float16)
                acc += tl.dot(x_tile0, w_tile0)

                x_addrs1 = x_ptr + (n_idx * C_IN + offs_k1[None, :]) * HW_IN + safe_h[:, None] * W_IN + safe_w[:, None]
                x_mask1 = k_mask1[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                x_tile1 = tl.load(x_addrs1, mask=x_mask1, other=0.0).to(tl.float16)
                w_addrs1 = w_cl_ptr + offs_n[None, :] * W_CO + kh * W_KH + kw * W_CS + offs_k1[:, None]
                w_mask1 = k_mask1[:, None] & n_mask[None, :]
                w_tile1 = tl.load(w_addrs1, mask=w_mask1, other=0.0).to(tl.float16)
                acc += tl.dot(x_tile1, w_tile1)

                x_addrs2 = x_ptr + (n_idx * C_IN + offs_k2[None, :]) * HW_IN + safe_h[:, None] * W_IN + safe_w[:, None]
                x_mask2 = k_mask2[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                x_tile2 = tl.load(x_addrs2, mask=x_mask2, other=0.0).to(tl.float16)
                w_addrs2 = w_cl_ptr + offs_n[None, :] * W_CO + kh * W_KH + kw * W_CS + offs_k2[:, None]
                w_mask2 = k_mask2[:, None] & n_mask[None, :]
                w_tile2 = tl.load(w_addrs2, mask=w_mask2, other=0.0).to(tl.float16)
                acc += tl.dot(x_tile2, w_tile2)

                x_addrs3 = x_ptr + (n_idx * C_IN + offs_k3[None, :]) * HW_IN + safe_h[:, None] * W_IN + safe_w[:, None]
                x_mask3 = k_mask3[None, :] & m_mask[:, None] & h_ok[:, None] & w_ok[:, None]
                x_tile3 = tl.load(x_addrs3, mask=x_mask3, other=0.0).to(tl.float16)
                w_addrs3 = w_cl_ptr + offs_n[None, :] * W_CO + kh * W_KH + kw * W_CS + offs_k3[:, None]
                w_mask3 = k_mask3[:, None] & n_mask[None, :]
                w_tile3 = tl.load(w_addrs3, mask=w_mask3, other=0.0).to(tl.float16)
                acc += tl.dot(x_tile3, w_tile3)
        g_idx += BLOCK_G

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]

    out_addrs = y_ptr + (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]
    tl.store(out_addrs, acc, mask=m_mask[:, None] & n_mask[None, :])


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
    counts_buf=None,
    tile_cin_buf=None,
    group_flags_buf=None,
    ag_count_buf=None,
    ag_list_buf=None,
    return_ms=False,
    fallback_ratio=FALLBACK_RATIO,
    return_avg_active_ratio=False,
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
        y = Fn.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups).float()
        if return_avg_active_ratio:
            return y, 0.0, 1.0
        return y, 0.0

    H_OUT = (H_IN + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    W_OUT = (W_IN + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    if H_OUT <= 0 or W_OUT <= 0:
        y = Fn.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups).float()
        if return_avg_active_ratio:
            return y, 0.0, 1.0
        return y, 0.0

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

    supported = (
        (kernel_size == 1 and stride == 1 and padding == 0) or
        (kernel_size == 3 and stride == 1 and padding == 1) or
        (kernel_size == 3 and stride == 2 and padding == 1)
    )

    if not supported:
        y = Fn.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups).float()
        if return_avg_active_ratio:
            return y, 0.0, 1.0
        return y, 0.0

    x_f16 = x.half().contiguous()

    if ag_count_buf is None or ag_count_buf.numel() < N_TILES:
        ag_count_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
    if ag_list_buf is None or ag_list_buf.numel() < N_TILES * MAX_AG:
        ag_list_buf = torch.empty(N_TILES * MAX_AG, dtype=torch.int32, device=device)

    _build_active_group_metadata(
        x_f16,
        N, C_IN, H_IN, W_IN, H_OUT, W_OUT,
        BH, BW, GH, GW,
        kernel_size, stride, padding,
        threshold,
        ag_count_buf, ag_list_buf,
    )

    avg_active_ratio = None
    if return_avg_active_ratio:
        avg_active_ratio = ag_count_buf[:N_TILES].float().mean().item() / max(NUM_GROUPS, 1)
        if avg_active_ratio == 0.0:
            y = torch.zeros(N, C_OUT, H_OUT, W_OUT, dtype=torch.float32, device=device)
            if bias is not None:
                y += bias.float().view(1, -1, 1, 1)
            return y, 0.0, avg_active_ratio
        if avg_active_ratio > fallback_ratio:
            y = Fn.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups).float()
            return y, 0.0, avg_active_ratio

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
        kernel = sparse_conv1x1_ag_kernel_8x16 if BW == 16 else sparse_conv1x1_ag_kernel_8x8
    elif stride == 1:
        kernel = sparse_conv3x3s1_ag_kernel_8x16 if BW == 16 else sparse_conv3x3s1_ag_kernel_8x8
    else:
        kernel = sparse_conv3x3s2_ag_kernel_8x16 if BW == 16 else sparse_conv3x3s2_ag_kernel_8x8

    kernel[_grid](
        x_f16,
        w_cl_f16,
        bias_f32,
        ag_count_buf,
        ag_list_buf,
        y,
        N,
        C_IN, C_OUT,
        H_IN, W_IN,
        H_OUT, W_OUT,
        GH, GW,
        HAS_BIAS=has_bias,
        GROUP_SIZE_C=GROUP_SIZE,
        MAX_AG=MAX_AG,
    )

    if return_ms:
        ee.record()
        torch.cuda.synchronize(device)
        sparse_ms = se.elapsed_time(ee)

    if return_avg_active_ratio:
        return y, sparse_ms, avg_active_ratio
    return y, sparse_ms
