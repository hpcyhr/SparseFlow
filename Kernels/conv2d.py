"""
稀疏 Conv2d Triton Kernels

两阶段设计：
  Stage-1 prescan:  扫描每个 block，标记是否非零 → 生成 nz_idx
  Stage-2 sparse:   只对非零 block 执行卷积，零 block 跳过

支持的卷积类型：
  - 3×3, stride=1, padding=1, groups=1  (标准卷积)
  - 1×1, stride=1, padding=0, groups=1  (瓶颈层/降采样)

重要：Stage-2 kernel 包含真实卷积权重参与计算（非 box filter）。
"""

import torch
import triton
import triton.language as tl


# =============================================================================
# Stage-1: Prescan Kernel（通用，与卷积类型无关）
# =============================================================================

@triton.jit
def prescan_kernel(
    x_ptr,          # 输入特征图 [N, C_in, H, W]
    flags_ptr,      # 输出标志 [total_blocks]
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    GRID_H: tl.constexpr,
    GRID_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
    """
    扫描输入的每个空间 block，判断是否全零。
    grid 维度: (N * C * GRID_H * GRID_W,)
    每个 program 负责一个 (n, c, gh, gw) block。
    """
    pid = tl.program_id(0)
    total = N * C * GRID_H * GRID_W
    if pid >= total:
        return

    # 解码 flat index -> (n, c, gh, gw)
    gw = pid % GRID_W
    tmp = pid // GRID_W
    gh = tmp % GRID_H
    tmp = tmp // GRID_H
    c = tmp % C
    n = tmp // C

    # block 内的 H/W 偏移
    offs_h = gh * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = gw * BLOCK_W + tl.arange(0, BLOCK_W)
    hh = offs_h[:, None]
    ww = offs_w[None, :]
    mask = (hh < H) & (ww < W)

    # 加载 block 数据，检查是否全零
    val = tl.load(x_ptr + ((n * C + c) * H + hh) * W + ww, mask=mask, other=0.0)
    is_nz = tl.max(tl.abs(val)) > THRESHOLD
    tl.store(flags_ptr + pid, is_nz.to(tl.int32))


# =============================================================================
# Stage-2: Sparse Conv 3×3 Kernel (带权重)
# =============================================================================

@triton.jit
def sparse_conv3x3_weighted_kernel(
    x_ptr,          # 输入 [N, C_in, H, W]
    w_ptr,          # 权重 [C_out, C_in, 3, 3]
    b_ptr,          # 偏置 [C_out] 或 nullptr
    y_ptr,          # 输出 [N, C_out, H, W]
    idx_ptr,        # 非零 block 索引 [num_nz] — 按 (n, c_in, gh, gw) 编码
    num_nz,
    HAS_BIAS: tl.constexpr,
    N: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    GRID_H: tl.constexpr,
    GRID_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    对非零 block 执行 3×3 卷积。

    每个 program 处理一个非零 block (n, c_in, gh, gw)，
    将其对所有 C_OUT 个输出通道的贡献 atomically 累加到 y。

    注意：这是 scatter 模式 — 每个输入 block 贡献到所有输出通道。
    对于高稀疏率场景，这比 gather 模式（遍历所有 c_in）更高效，
    因为大量 c_in 的 block 为零，可以直接跳过。
    """
    pid = tl.program_id(0)
    if pid >= num_nz:
        return

    # 解码非零 block 的原始 flat index
    orig = tl.load(idx_ptr + pid)
    gw = orig % GRID_W
    tmp = orig // GRID_W
    gh = tmp % GRID_H
    tmp = tmp // GRID_H
    c_in = tmp % C_IN
    n = tmp // C_IN

    # block 内坐标
    offs_h = gh * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = gw * BLOCK_W + tl.arange(0, BLOCK_W)
    hh = offs_h[:, None]   # [BLOCK_H, 1]
    ww = offs_w[None, :]   # [1, BLOCK_W]
    mask = (hh < H) & (ww < W)

    # 对每个输出通道累加
    for c_out in range(C_OUT):
        acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)

        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                # 卷积权重 w[c_out, c_in, kh, kw]
                w_val = tl.load(w_ptr + ((c_out * C_IN + c_in) * 3 + kh) * 3 + kw)

                # 输入坐标 (padding=1 → offset = kh-1, kw-1)
                h_idx = hh + (kh - 1)
                w_idx = ww + (kw - 1)
                m = mask & (h_idx >= 0) & (h_idx < H) & (w_idx >= 0) & (w_idx < W)

                x_val = tl.load(
                    x_ptr + ((n * C_IN + c_in) * H + h_idx) * W + w_idx,
                    mask=m, other=0.0
                )
                acc += x_val * w_val

        # 累加到输出（需要 atomic 因为多个 c_in 会写同一个位置）
        tl.atomic_add(
            y_ptr + ((n * C_OUT + c_out) * H + hh) * W + ww,
            acc, mask=mask
        )


# =============================================================================
# Stage-2: Sparse Conv 1×1 Kernel (带权重)
# =============================================================================

@triton.jit
def sparse_conv1x1_weighted_kernel(
    x_ptr,          # 输入 [N, C_in, H, W]
    w_ptr,          # 权重 [C_out, C_in, 1, 1] → 实际就是 [C_out, C_in]
    b_ptr,          # 偏置 [C_out] 或 nullptr
    y_ptr,          # 输出 [N, C_out, H, W]
    idx_ptr,        # 非零 block 索引
    num_nz,
    HAS_BIAS: tl.constexpr,
    N: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    GRID_H: tl.constexpr,
    GRID_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    对非零 block 执行 1×1 卷积（本质是逐像素矩阵乘）。
    """
    pid = tl.program_id(0)
    if pid >= num_nz:
        return

    orig = tl.load(idx_ptr + pid)
    gw = orig % GRID_W
    tmp = orig // GRID_W
    gh = tmp % GRID_H
    tmp = tmp // GRID_H
    c_in = tmp % C_IN
    n = tmp // C_IN

    offs_h = gh * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = gw * BLOCK_W + tl.arange(0, BLOCK_W)
    hh = offs_h[:, None]
    ww = offs_w[None, :]
    mask = (hh < H) & (ww < W)

    # 加载输入 block
    x_val = tl.load(
        x_ptr + ((n * C_IN + c_in) * H + hh) * W + ww,
        mask=mask, other=0.0
    )

    for c_out in range(C_OUT):
        # 1×1 权重就是一个标量 w[c_out, c_in]
        w_val = tl.load(w_ptr + c_out * C_IN + c_in)
        acc = x_val * w_val

        tl.atomic_add(
            y_ptr + ((n * C_OUT + c_out) * H + hh) * W + ww,
            acc, mask=mask
        )


# =============================================================================
# Stage-2: Dense Conv 3×3 Kernel (基准对照，无权重 box filter)
# =============================================================================

@triton.jit
def dense_conv3x3_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    GRID_H, GRID_W,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """基准对照：稠密 3×3 box filter (无权重)"""
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
            acc += tl.load(
                x_ptr + ((n * C + c) * H + h_idx) * W + w_idx,
                mask=m, other=0.0
            )
    tl.store(y_ptr + ((n * C + c) * H + hh) * W + ww, acc, mask=mask)


# =============================================================================
# Python 封装函数
# =============================================================================

def sparse_conv2d_forward(x: torch.Tensor, weight: torch.Tensor,
                          bias: torch.Tensor, block_size: int,
                          kernel_size: int = 3, threshold: float = 1e-6):
    """
    两阶段稀疏卷积前向传播。

    Args:
        x: 输入 [N, C_in, H, W]
        weight: 权重 [C_out, C_in, K, K]
        bias: 偏置 [C_out] or None
        block_size: prescan block 大小
        kernel_size: 3 或 1
        threshold: 零判断阈值

    Returns:
        y: 输出 [N, C_out, H, W]
        sparse_ms: Stage-2 计时 (ms)
    """
    N, C_IN, H, W = x.shape
    C_OUT = weight.shape[0]
    BLOCK_H = block_size
    BLOCK_W = block_size
    GRID_H = triton.cdiv(H, BLOCK_H)
    GRID_W = triton.cdiv(W, BLOCK_W)
    total_blocks = N * C_IN * GRID_H * GRID_W

    # --- Stage-1: Prescan ---
    flags = torch.empty(total_blocks, dtype=torch.int32, device=x.device)
    prescan_kernel[(total_blocks,)](
        x, flags,
        N, C_IN, H, W, GRID_H, GRID_W,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        THRESHOLD=threshold,
    )
    torch.cuda.synchronize()

    nz_idx = flags.nonzero(as_tuple=False).squeeze(1).int()
    num_nz = nz_idx.numel()

    # 输出初始化为 0（稀疏模式需要 atomic_add 累加）
    y = torch.zeros(N, C_OUT, H, W, dtype=x.dtype, device=x.device)

    if num_nz == 0:
        # 全零输入 → 只加 bias
        if bias is not None:
            y += bias.view(1, -1, 1, 1)
        return y, 0.0

    # --- Stage-2: Sparse Conv ---
    has_bias = bias is not None
    b_ptr = bias if has_bias else x  # dummy, 不会被用到

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    if kernel_size == 3:
        sparse_conv3x3_weighted_kernel[(num_nz,)](
            x, weight, b_ptr, y, nz_idx, num_nz,
            HAS_BIAS=has_bias,
            N=N, C_IN=C_IN, C_OUT=C_OUT, H=H, W=W,
            GRID_H=GRID_H, GRID_W=GRID_W,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        )
    elif kernel_size == 1:
        sparse_conv1x1_weighted_kernel[(num_nz,)](
            x, weight, b_ptr, y, nz_idx, num_nz,
            HAS_BIAS=has_bias,
            N=N, C_IN=C_IN, C_OUT=C_OUT, H=H, W=W,
            GRID_H=GRID_H, GRID_W=GRID_W,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        )

    end.record()
    torch.cuda.synchronize()

    # 加 bias
    if has_bias:
        y += bias.view(1, -1, 1, 1)

    return y, start.elapsed_time(end)