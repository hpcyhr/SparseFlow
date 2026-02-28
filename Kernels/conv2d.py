"""
稀疏 Conv2d Triton Kernels — Gather 模式 (v2 修正版)

两阶段设计：
  Stage-1 prescan:  扫描每个 (n, c_in, gh, gw) block，标记是否非零
  Stage-2 gather conv: 以输出 block 为中心，遍历所有 c_in，跳过全零 c_in block

修正内容 (v2 相比 v1):
  1. 邻域 flag 检查改用 tl.where + mask 方式
     - v1 bug: 使用 Python-level `if ngh >= 0 and ngh < GRID_H` 对运行时值做判断
       在 Triton JIT 编译中，运行时变量 (gh, gw) 是符号值，
       Python `if` 无法在 GPU 上正确执行分支判断
     - v2 fix: 使用 `tl.where(valid, ngh, 0)` 做安全索引 + mask 过滤
  2. any_nz 标量类型修正
     - v1 bug: `any_nz = 0` (Python int) 与 `flag_val` (Triton tensor) 做 `|` 运算
     - v2 fix: 声明 `any_nz` 为 `tl.int32` 类型
  3. 保持 prescan / 1x1 kernel / dense_conv3x3_kernel 不变（它们没有此类问题）

支持的卷积类型：
  - 3×3, stride=1, padding=1, groups=1
  - 1×1, stride=1, padding=0, groups=1
"""

import torch
import triton
import triton.language as tl


# =============================================================================
# Stage-1: Prescan Kernel
# =============================================================================

@triton.jit
def prescan_kernel(
    x_ptr,          # 输入 [N, C_in, H, W]
    flags_ptr,      # 输出 [N * C_in * GRID_H * GRID_W]
    N, C, H, W,
    GRID_H, GRID_W,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
    """每个 program 扫描一个 (n, c, gh, gw) block，判断是否全零。"""
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
# Stage-2: Gather Conv 3×3 Kernel (v2 — 修正边界检查)
# =============================================================================

@triton.jit
def gather_conv3x3_kernel(
    x_ptr,          # 输入 [N, C_in, H, W]
    w_ptr,          # 权重 [C_out, C_in, 3, 3]
    flags_ptr,      # prescan flags [N * C_in * GRID_H * GRID_W]
    y_ptr,          # 输出 [N, C_out, H, W]
    N, C_IN, C_OUT,
    H, W,
    GRID_H, GRID_W,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Gather 模式 3×3 卷积 (v2)。

    每个 program 负责计算一个输出 block: (n, c_out, gh, gw)。
    遍历所有 c_in，通过 flags 判断邻域是否有非零 block 来决定是否跳过。

    grid: (N * C_OUT * GRID_H * GRID_W,)
    """
    pid = tl.program_id(0)

    gw = pid % GRID_W
    tmp = pid // GRID_W
    gh = tmp % GRID_H
    tmp = tmp // GRID_H
    c_out = tmp % C_OUT
    n = tmp // C_OUT

    # 输出 block 内的像素坐标
    offs_h = gh * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = gw * BLOCK_W + tl.arange(0, BLOCK_W)
    hh = offs_h[:, None]   # [BLOCK_H, 1]
    ww = offs_w[None, :]   # [1, BLOCK_W]
    out_mask = (hh < H) & (ww < W)

    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)

    # flags 布局: flags[n * C_IN * GRID_H * GRID_W + c_in * GRID_H * GRID_W + gh * GRID_W + gw]
    flags_stride_n = C_IN * GRID_H * GRID_W
    flags_stride_c = GRID_H * GRID_W

    for c_in in range(C_IN):
        # === 检查该 c_in 在当前 block 的 3x3 邻域是否有非零 block ===
        # 对 3x3 邻域的 block 进行检查。用 clamp 确保索引合法，
        # 越界的 block 读到的是合法位置的值，但用 valid 标记过滤。
        base_flag = n * flags_stride_n + c_in * flags_stride_c
        any_nz = 0

        # 展开 3x3 邻域 (tl.static_range 在编译时展开)
        for dh in tl.static_range(3):     # 0, 1, 2
            for dw in tl.static_range(3):
                ngh = gh + dh - 1
                ngw = gw + dw - 1

                # 边界检查
                is_valid_h = (ngh >= 0) & (ngh < GRID_H)
                is_valid_w = (ngw >= 0) & (ngw < GRID_W)
                is_valid = is_valid_h & is_valid_w
                # 越界时使用 gh, gw（总是合法的）作为安全索引
                # 读到的 flag 不影响结果，因为会被 is_valid 过滤
                safe_ngh = ngh * is_valid.to(tl.int32) + gh * (1 - is_valid.to(tl.int32))
                safe_ngw = ngw * is_valid.to(tl.int32) + gw * (1 - is_valid.to(tl.int32))

                flag_val = tl.load(flags_ptr + base_flag + safe_ngh * GRID_W + safe_ngw)

                # 越界时 flag 强制视为 0
                flag_val = flag_val * is_valid.to(tl.int32)
                any_nz = any_nz | flag_val

        if any_nz == 0:
            continue  # 该 c_in 在邻域内全零，跳过

        # === 计算该 c_in 对当前输出 block 的贡献 ===
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                w_val = tl.load(w_ptr + ((c_out * C_IN + c_in) * 3 + kh) * 3 + kw)

                h_idx = hh + (kh - 1)
                w_idx = ww + (kw - 1)
                m = out_mask & (h_idx >= 0) & (h_idx < H) & (w_idx >= 0) & (w_idx < W)

                x_val = tl.load(
                    x_ptr + ((n * C_IN + c_in) * H + h_idx) * W + w_idx,
                    mask=m, other=0.0
                )
                acc += x_val * w_val

    # 写出（无 atomic，每个 output block 唯一一个 program 负责）
    tl.store(
        y_ptr + ((n * C_OUT + c_out) * H + hh) * W + ww,
        acc, mask=out_mask
    )


# =============================================================================
# Stage-2: Gather Conv 1×1 Kernel (v2)
# =============================================================================

@triton.jit
def gather_conv1x1_kernel(
    x_ptr,          # 输入 [N, C_in, H, W]
    w_ptr,          # 权重 [C_out, C_in]
    flags_ptr,      # prescan flags
    y_ptr,          # 输出 [N, C_out, H, W]
    N, C_IN, C_OUT,
    H, W,
    GRID_H, GRID_W,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Gather 模式 1×1 卷积 (v2)。
    grid: (N * C_OUT * GRID_H * GRID_W,)
    """
    pid = tl.program_id(0)

    gw = pid % GRID_W
    tmp = pid // GRID_W
    gh = tmp % GRID_H
    tmp = tmp // GRID_H
    c_out = tmp % C_OUT
    n = tmp // C_OUT

    offs_h = gh * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = gw * BLOCK_W + tl.arange(0, BLOCK_W)
    hh = offs_h[:, None]
    ww = offs_w[None, :]
    out_mask = (hh < H) & (ww < W)

    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)

    flags_stride_n = C_IN * GRID_H * GRID_W
    flags_stride_c = GRID_H * GRID_W
    flag_offset_hw = gh * GRID_W + gw

    for c_in in range(C_IN):
        flag_val = tl.load(
            flags_ptr + n * flags_stride_n + c_in * flags_stride_c + flag_offset_hw
        )
        if flag_val == 0:
            continue

        w_val = tl.load(w_ptr + c_out * C_IN + c_in)

        x_val = tl.load(
            x_ptr + ((n * C_IN + c_in) * H + hh) * W + ww,
            mask=out_mask, other=0.0
        )
        acc += x_val * w_val

    tl.store(
        y_ptr + ((n * C_OUT + c_out) * H + hh) * W + ww,
        acc, mask=out_mask
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
    两阶段稀疏卷积前向传播 (Gather 模式 v2)。

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
    total_input_blocks = N * C_IN * GRID_H * GRID_W
    total_output_blocks = N * C_OUT * GRID_H * GRID_W

    # --- Stage-1: Prescan ---
    flags = torch.empty(total_input_blocks, dtype=torch.int32, device=x.device)
    prescan_kernel[(total_input_blocks,)](
        x, flags,
        N, C_IN, H, W, GRID_H, GRID_W,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        THRESHOLD=threshold,
    )
    torch.cuda.synchronize(x.device)

    # 快速路径：全零输入
    num_nz = flags.sum().item()
    if num_nz == 0:
        y = torch.zeros(N, C_OUT, H, W, dtype=x.dtype, device=x.device)
        if bias is not None:
            y += bias.view(1, -1, 1, 1)
        return y, 0.0

    # --- Stage-2: Gather Conv ---
    y = torch.empty(N, C_OUT, H, W, dtype=x.dtype, device=x.device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    if kernel_size == 3:
        gather_conv3x3_kernel[(total_output_blocks,)](
            x, weight, flags, y,
            N, C_IN, C_OUT, H, W,
            GRID_H, GRID_W,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        )
    elif kernel_size == 1:
        gather_conv1x1_kernel[(total_output_blocks,)](
            x, weight, flags, y,
            N, C_IN, C_OUT, H, W,
            GRID_H, GRID_W,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        )

    end.record()
    torch.cuda.synchronize(x.device)

    # 加 bias
    if bias is not None:
        y += bias.view(1, -1, 1, 1)

    return y, start.elapsed_time(end)

