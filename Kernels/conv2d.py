"""
稀疏 Conv2d Triton Kernels — Tensor Core 加速 (v4)

3×3 conv 分解为 9 次子 GEMM:
  output += W[:,:,kh,kw] @ shift(input, kh-1, kw-1)
  每次子 GEMM: W_slice[BLOCK_M, BLOCK_K] × X_shift[BLOCK_K, TILE_N]
  对 C_IN 分块, 每个 chunk 先检查 flags 跳过全零。

1×1 conv 直接 GEMM:
  W[BLOCK_M, BLOCK_K] × X[BLOCK_K, TILE_N]

所有 tl.dot 使用 fp16 输入 fp32 累加。
"""

import torch
import triton
import triton.language as tl


# =============================================================================
# Stage-1: Prescan
# =============================================================================

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
    tmp = tmp // GRID_H
    c = tmp % C
    n = tmp // C

    offs_h = gh * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = gw * BLOCK_W + tl.arange(0, BLOCK_W)
    hh = offs_h[:, None]
    ww = offs_w[None, :]
    mask = (hh < H) & (ww < W)

    base = (n * C + c) * H
    val = tl.load(x_ptr + (base + hh) * W + ww, mask=mask, other=0.0)
    is_nz = tl.max(tl.abs(val)) > THRESHOLD
    tl.store(flags_ptr + pid, is_nz.to(tl.int32))


# =============================================================================
# Stage-2: Sparse 3×3 Conv — 9-GEMM 分解 + tl.dot
# =============================================================================

@triton.jit
def sparse_conv3x3_dot_kernel(
    x_ptr,           # [N, C_IN, H, W] fp16
    w_ptr,           # [C_OUT, C_IN, 3, 3] fp16 (原始布局)
    flags_ptr,       # [N * C_IN * GRID_H * GRID_W] int32
    y_ptr,           # [N, C_OUT, H, W] fp16
    N, C_IN, C_OUT,
    H, W,
    GRID_H, GRID_W,
    BLOCK_M: tl.constexpr,       # C_OUT tile (>= 16)
    BLOCK_K: tl.constexpr,       # C_IN tile (>= 16)
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    grid: (N * GRID_H * GRID_W, cdiv(C_OUT, BLOCK_M))
    每个 program 输出 [BLOCK_M, TILE_N] 子块, TILE_N = BLOCK_H * BLOCK_W
    """
    pid_spatial = tl.program_id(0)
    pid_m = tl.program_id(1)

    gw_idx = pid_spatial % GRID_W
    tmp = pid_spatial // GRID_W
    gh_idx = tmp % GRID_H
    n_idx = tmp // GRID_H

    c_out_start = pid_m * BLOCK_M

    TILE_N: tl.constexpr = BLOCK_H * BLOCK_W

    # 空间坐标
    offs_tile = tl.arange(0, TILE_N)  # [TILE_N]
    tile_h = gh_idx * BLOCK_H + (offs_tile // BLOCK_W)  # [TILE_N]
    tile_w = gw_idx * BLOCK_W + (offs_tile % BLOCK_W)   # [TILE_N]

    # C_OUT offsets
    offs_m = c_out_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]

    # 累加器
    acc = tl.zeros([BLOCK_M, TILE_N], dtype=tl.float32)

    # flags strides
    flags_stride_n = C_IN * GRID_H * GRID_W
    flags_stride_c = GRID_H * GRID_W
    flag_hw = gh_idx * GRID_W + gw_idx

    num_k_iters = tl.cdiv(C_IN, BLOCK_K)

    for k_iter in range(num_k_iters):
        c_in_start = k_iter * BLOCK_K
        offs_k = c_in_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # --- 稀疏检查: chunk 内所有通道在当前 block 的 flags ---
        # 加载所有 BLOCK_K 个 flag（越界通道的 flag 用 safe index 读取后 mask）
        flag_addrs = (flags_ptr + n_idx * flags_stride_n
                      + offs_k * flags_stride_c + flag_hw)
        flag_mask = offs_k < C_IN
        # 越界通道 load 0
        chunk_flags = tl.load(flag_addrs, mask=flag_mask, other=0)
        any_nz = tl.max(chunk_flags)

        if any_nz != 0:
            # --- 9 次子 GEMM: 每次对应一个 (kh, kw) 位置 ---
            for kh in tl.static_range(3):
                for kw in tl.static_range(3):
                    h_shifted = tile_h + kh - 1  # [TILE_N]
                    w_shifted = tile_w + kw - 1  # [TILE_N]

                    x_addrs = (x_ptr
                               + (n_idx * C_IN + offs_k[:, None]) * (H * W)
                               + h_shifted[None, :] * W
                               + w_shifted[None, :])
                    x_mask = ((offs_k[:, None] < C_IN)
                              & (h_shifted[None, :] >= 0) & (h_shifted[None, :] < H)
                              & (w_shifted[None, :] >= 0) & (w_shifted[None, :] < W))
                    x_tile = tl.load(x_addrs, mask=x_mask, other=0.0).to(tl.float16)

                    w_addrs = (w_ptr
                               + ((offs_m[:, None] * C_IN + offs_k[None, :]) * 3 + kh) * 3 + kw)
                    w_mask = (offs_m[:, None] < C_OUT) & (offs_k[None, :] < C_IN)
                    w_tile = tl.load(w_addrs, mask=w_mask, other=0.0).to(tl.float16)

                    acc += tl.dot(w_tile, x_tile)

    # --- 写回 ---
    out_addrs = (y_ptr
                 + (n_idx * C_OUT + offs_m[:, None]) * (H * W)
                 + tile_h[None, :] * W
                 + tile_w[None, :])
    out_mask = (offs_m[:, None] < C_OUT) & (tile_h[None, :] < H) & (tile_w[None, :] < W)
    tl.store(out_addrs, acc.to(tl.float16), mask=out_mask)


# =============================================================================
# Stage-2: Sparse 1×1 Conv — tl.dot
# =============================================================================

@triton.jit
def sparse_conv1x1_dot_kernel(
    x_ptr, w_ptr, flags_ptr, y_ptr,
    N, C_IN, C_OUT,
    H, W,
    GRID_H, GRID_W,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_spatial = tl.program_id(0)
    pid_m = tl.program_id(1)

    gw_idx = pid_spatial % GRID_W
    tmp = pid_spatial // GRID_W
    gh_idx = tmp % GRID_H
    n_idx = tmp // GRID_H

    c_out_start = pid_m * BLOCK_M

    TILE_N: tl.constexpr = BLOCK_H * BLOCK_W

    offs_tile = tl.arange(0, TILE_N)
    tile_h = gh_idx * BLOCK_H + (offs_tile // BLOCK_W)
    tile_w = gw_idx * BLOCK_W + (offs_tile % BLOCK_W)

    offs_m = c_out_start + tl.arange(0, BLOCK_M)
    acc = tl.zeros([BLOCK_M, TILE_N], dtype=tl.float32)

    flags_stride_n = C_IN * GRID_H * GRID_W
    flags_stride_c = GRID_H * GRID_W
    flag_hw = gh_idx * GRID_W + gw_idx

    num_k_iters = tl.cdiv(C_IN, BLOCK_K)

    for k_iter in range(num_k_iters):
        c_in_start = k_iter * BLOCK_K
        offs_k = c_in_start + tl.arange(0, BLOCK_K)

        # 向量化 flag 检查
        flag_addrs = (flags_ptr + n_idx * flags_stride_n
                      + offs_k * flags_stride_c + flag_hw)
        flag_mask = offs_k < C_IN
        chunk_flags = tl.load(flag_addrs, mask=flag_mask, other=0)
        any_nz = tl.max(chunk_flags)

        if any_nz != 0:
            # X tile [BLOCK_K, TILE_N]
            x_addrs = (x_ptr
                       + (n_idx * C_IN + offs_k[:, None]) * (H * W)
                       + tile_h[None, :] * W
                       + tile_w[None, :])
            x_mask = ((offs_k[:, None] < C_IN)
                      & (tile_h[None, :] < H) & (tile_w[None, :] < W))
            x_tile = tl.load(x_addrs, mask=x_mask, other=0.0).to(tl.float16)

            # W tile [BLOCK_M, BLOCK_K]
            w_addrs = w_ptr + offs_m[:, None] * C_IN + offs_k[None, :]
            w_mask = (offs_m[:, None] < C_OUT) & (offs_k[None, :] < C_IN)
            w_tile = tl.load(w_addrs, mask=w_mask, other=0.0).to(tl.float16)

            acc += tl.dot(w_tile, x_tile)

    # 写回
    out_addrs = (y_ptr
                 + (n_idx * C_OUT + offs_m[:, None]) * (H * W)
                 + tile_h[None, :] * W
                 + tile_w[None, :])
    out_mask = (offs_m[:, None] < C_OUT) & (tile_h[None, :] < H) & (tile_w[None, :] < W)
    tl.store(out_addrs, acc.to(tl.float16), mask=out_mask)


# =============================================================================
# Dense Conv 3×3 (基准)
# =============================================================================

@triton.jit
def dense_conv3x3_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    GRID_H, GRID_W,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * C * GRID_H * GRID_W
    if pid >= total:
        return
    gw = pid % GRID_W
    tmp = pid // GRID_W
    gh = tmp % GRID_H
    tmp = tmp // GRID_H
    c = tmp % C
    n = tmp // C
    offs_h = gh * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = gw * BLOCK_W + tl.arange(0, BLOCK_W)
    hh = offs_h[:, None]
    ww = offs_w[None, :]
    mask = (hh < H) & (ww < W)
    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    for kh in range(-1, 2):
        for kw in range(-1, 2):
            h_idx = hh + kh
            w_idx = ww + kw
            m = mask & (h_idx >= 0) & (h_idx < H) & (w_idx >= 0) & (w_idx < W)
            acc += tl.load(x_ptr + ((n * C + c) * H + h_idx) * W + w_idx, mask=m, other=0.0)
    tl.store(y_ptr + ((n * C + c) * H + hh) * W + ww, acc, mask=mask)


# =============================================================================
# Python 封装
# =============================================================================

def _select_tile_config(H, W, C_IN, C_OUT, kernel_size):
    """选择 tile, 确保所有维度 >= 16 (tl.dot 要求)。"""
    spatial = min(H, W)

    # TILE_N = BH * BW 必须 >= 16
    if spatial >= 32:
        BH, BW = 16, 16   # TILE_N = 256
    elif spatial >= 16:
        BH, BW = 16, 16   # TILE_N = 256
    elif spatial >= 8:
        BH, BW = 8, 8     # TILE_N = 64
    else:
        BH, BW = 4, 4     # TILE_N = 16

    # BLOCK_M: C_OUT tile, >= 16
    if C_OUT >= 128:
        BLOCK_M = 64
    elif C_OUT >= 64:
        BLOCK_M = 32
    else:
        BLOCK_M = 16

    # BLOCK_K: C_IN tile, >= 16
    if kernel_size == 3:
        # 3x3: BLOCK_K 是 C_IN 方向 tile, 也作为 tl.dot 的 K 维
        # tl.dot 要求 K >= 16, 这里直接用 >= 16
        if C_IN >= 128:
            BLOCK_K = 32
        elif C_IN >= 64:
            BLOCK_K = 16
        else:
            BLOCK_K = 16
    else:
        if C_IN >= 128:
            BLOCK_K = 32
        elif C_IN >= 64:
            BLOCK_K = 16
        else:
            BLOCK_K = 16

    return BH, BW, BLOCK_M, BLOCK_K


def sparse_conv2d_forward(x, weight, bias, block_size,
                          kernel_size=3, threshold=1e-6):
    """两阶段稀疏卷积 (Tensor Core, fp16 compute, fp32 accumulate)。"""
    N, C_IN, H, W = x.shape
    C_OUT = weight.shape[0]

    x_f16 = x.half().contiguous()
    w_f16 = weight.half().contiguous()

    BH, BW, BLOCK_M, BLOCK_K = _select_tile_config(H, W, C_IN, C_OUT, kernel_size)
    GH = triton.cdiv(H, BH)
    GW = triton.cdiv(W, BW)
    total_in = N * C_IN * GH * GW

    # Stage-1: Prescan
    flags = torch.empty(total_in, dtype=torch.int32, device=x.device)
    prescan_kernel[(total_in,)](
        x_f16, flags, N, C_IN, H, W, GH, GW,
        BLOCK_H=BH, BLOCK_W=BW, THRESHOLD=threshold,
    )
    torch.cuda.synchronize(x.device)

    num_nz = flags.sum().item()
    if num_nz == 0:
        y = torch.zeros(N, C_OUT, H, W, dtype=x.dtype, device=x.device)
        if bias is not None:
            y += bias.view(1, -1, 1, 1)
        return y, 0.0

    y = torch.empty(N, C_OUT, H, W, dtype=torch.float16, device=x.device)

    grid_spatial = N * GH * GW
    grid_m = triton.cdiv(C_OUT, BLOCK_M)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    if kernel_size == 3:
        sparse_conv3x3_dot_kernel[(grid_spatial, grid_m)](
            x_f16, w_f16, flags, y,
            N, C_IN, C_OUT, H, W, GH, GW,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            BLOCK_H=BH, BLOCK_W=BW,
        )
    else:
        # 1x1: 权重 reshape [C_OUT, C_IN, 1, 1] → [C_OUT, C_IN]
        w_1x1 = w_f16.reshape(C_OUT, C_IN).contiguous()
        sparse_conv1x1_dot_kernel[(grid_spatial, grid_m)](
            x_f16, w_1x1, flags, y,
            N, C_IN, C_OUT, H, W, GH, GW,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            BLOCK_H=BH, BLOCK_W=BW,
        )

    end.record()
    torch.cuda.synchronize(x.device)

    if bias is not None:
        y = y + bias.half().view(1, -1, 1, 1)

    return y.float(), start.elapsed_time(end)