"""
稀疏 Linear Triton Kernels

两阶段设计（与 Conv2d 一致）：
  Stage-1 prescan:  按行扫描输入 [N, C_in]，标记全零行
  Stage-2 sparse:   只对非零行执行矩阵乘，零行跳过

适用场景：
  - SNN 全连接层（LIF 输出 → Linear）
  - Spiking Transformer 的 QKV 投影
  - ResNet 最后的分类头（fc 层）

输入是 LIF 脉冲输出 [N, C_in]，N = T*B 展平后的 batch 维度。
大量行为全零，可直接跳过。
"""

import torch
import triton
import triton.language as tl


# =============================================================================
# Stage-1: Prescan — 按行扫描，标记非零行
# =============================================================================

@triton.jit
def linear_prescan_kernel(
    x_ptr,          # 输入 [N, C_in]
    flags_ptr,      # 输出标志 [N]
    N: tl.constexpr,
    C_IN: tl.constexpr,
    BLOCK_C: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
    """
    扫描输入的每一行，判断是否全零。
    grid: (N,)
    """
    row = tl.program_id(0)
    if row >= N:
        return

    is_nz = tl.constexpr(0) != 0  # False

    for c_start in range(0, C_IN, BLOCK_C):
        offs = c_start + tl.arange(0, BLOCK_C)
        mask = offs < C_IN
        val = tl.load(x_ptr + row * C_IN + offs, mask=mask, other=0.0)
        if tl.max(tl.abs(val)) > THRESHOLD:
            is_nz = True

    tl.store(flags_ptr + row, is_nz.to(tl.int32))


# =============================================================================
# Stage-2: Sparse Linear — 只对非零行做矩阵乘
# =============================================================================

@triton.jit
def sparse_linear_kernel(
    x_ptr,          # 输入 [N, C_in]
    w_ptr,          # 权重 [C_out, C_in]（注意 Linear 的 weight 是 [C_out, C_in]）
    y_ptr,          # 输出 [N, C_out]
    idx_ptr,        # 非零行索引 [num_nz]
    num_nz,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    对非零行执行 y[n] = x[n] @ W^T。
    grid: (num_nz, cdiv(C_OUT, BLOCK_C))
    每个 program 负责一个非零行的一段输出。
    """
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    if pid_row >= num_nz:
        return

    n = tl.load(idx_ptr + pid_row)

    # 输出通道偏移
    out_offs = pid_col * BLOCK_C + tl.arange(0, BLOCK_C)
    out_mask = out_offs < C_OUT

    # 累加 dot product: y[n, c_out] = sum_k x[n, k] * w[c_out, k]
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    for k in range(C_IN):
        x_val = tl.load(x_ptr + n * C_IN + k)
        # w[c_out, k] for c_out in out_offs
        w_val = tl.load(w_ptr + out_offs * C_IN + k, mask=out_mask, other=0.0)
        acc += x_val * w_val

    tl.store(y_ptr + n * C_OUT + out_offs, acc, mask=out_mask)


# =============================================================================
# Python 封装
# =============================================================================

def sparse_linear_forward(x: torch.Tensor, weight: torch.Tensor,
                          bias: torch.Tensor, threshold: float = 1e-6):
    """
    两阶段稀疏 Linear 前向传播。

    Args:
        x: 输入 [N, C_in]
        weight: 权重 [C_out, C_in]
        bias: 偏置 [C_out] or None
        threshold: 零判断阈值

    Returns:
        y: 输出 [N, C_out]
        sparse_ms: Stage-2 计时 (ms)
    """
    N, C_IN = x.shape
    C_OUT = weight.shape[0]

    BLOCK_C = min(128, triton.next_power_of_2(max(C_IN, C_OUT)))

    # --- Stage-1: Prescan ---
    flags = torch.empty(N, dtype=torch.int32, device=x.device)
    linear_prescan_kernel[(N,)](
        x, flags, N, C_IN,
        BLOCK_C=BLOCK_C,
        THRESHOLD=threshold,
    )
    torch.cuda.synchronize(x.device)

    nz_idx = flags.nonzero(as_tuple=False).squeeze(1).int()
    num_nz = nz_idx.numel()

    y = torch.zeros(N, C_OUT, dtype=x.dtype, device=x.device)

    if num_nz == 0:
        if bias is not None:
            y += bias.unsqueeze(0)
        return y, 0.0

    # --- Stage-2: Sparse Linear ---
    BLOCK_OUT = min(128, triton.next_power_of_2(C_OUT))
    grid = (num_nz, triton.cdiv(C_OUT, BLOCK_OUT))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    sparse_linear_kernel[grid](
        x, weight, y, nz_idx, num_nz,
        C_IN=C_IN, C_OUT=C_OUT,
        BLOCK_C=BLOCK_OUT,
    )

    end.record()
    torch.cuda.synchronize(x.device)

    if bias is not None:
        # 只对非零行加 bias（零行保持全零）
        y[nz_idx.long()] += bias.unsqueeze(0)

    return y, start.elapsed_time(end)