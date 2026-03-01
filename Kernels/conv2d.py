"""
SparseFlow Conv2d v7 — Sparse im2col-GEMM with Tensor Core acceleration

3x3: F.unfold → detect NZ channel-groups → gather → Triton GEMM
1x1: reshape  → detect NZ channels       → gather → Triton GEMM

Triton kernel is a standard tiled GEMM:  C = A @ B + bias
  fp16 loads, fp32 accumulate, fp16 store → fp32 output.
"""

import torch
import torch.nn.functional as F_torch
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════
#  Triton GEMM:  C[M,N] = A[M,K] @ B[K,N] + bias[M]
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def _gemm_bias_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n
    pid_n = pid % num_n

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
                     mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0).to(tl.float16)
        b = tl.load(b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
                     mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0).to(tl.float16)
        acc += tl.dot(a, b)

    if HAS_BIAS:
        bias_val = tl.load(bias_ptr + rm, mask=rm < M, other=0.0).to(tl.float32)
        acc += bias_val[:, None]

    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
             acc.to(tl.float16),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def _triton_gemm(A, B, bias, BM=64, BN=64, BK=32):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, dtype=torch.float16, device=A.device)
    has_bias = bias is not None
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), )
    _gemm_bias_kernel[grid](
        A, B, C, bias if has_bias else A,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        HAS_BIAS=has_bias,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
    )
    return C


# ═══════════════════════════════════════════════════════════════════════
#  Sparsity Detection
# ═══════════════════════════════════════════════════════════════════════

def _nz_channels(x_2d, threshold=0.0):
    return (x_2d.abs().max(dim=1).values > threshold).nonzero(as_tuple=False).view(-1)

def _nz_channel_groups(x_2d, group_size, threshold=0.0):
    C_total, L = x_2d.shape
    n_groups = C_total // group_size
    grouped = x_2d[:n_groups * group_size].view(n_groups, group_size, L)
    return (grouped.abs().amax(dim=(1, 2)) > threshold).nonzero(as_tuple=False).view(-1)


# ═══════════════════════════════════════════════════════════════════════
#  Tile Config
# ═══════════════════════════════════════════════════════════════════════

def _gemm_config(M, N, K):
    def pick(dim, choices):
        for c in choices:
            if dim >= 2 * c:
                return c
        return choices[-1]
    BM = pick(M, (128, 64, 32, 16))
    BN = pick(N, (128, 64, 32, 16))
    BK = pick(K, (64, 32, 16))
    return BM, BN, BK


# ═══════════════════════════════════════════════════════════════════════
#  Legacy kernels (benchmark compat)
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def prescan_kernel(
    x_ptr, flags_ptr, N, C, H, W, GRID_H, GRID_W,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, THRESHOLD: tl.constexpr,
):
    pid = tl.program_id(0)
    gw = pid % GRID_W;  tmp = pid // GRID_W
    gh = tmp % GRID_H;  tmp = tmp // GRID_H
    c  = tmp % C;        n   = tmp // C
    hh = (gh * BLOCK_H + tl.arange(0, BLOCK_H))[:, None]
    ww = (gw * BLOCK_W + tl.arange(0, BLOCK_W))[None, :]
    mask = (hh < H) & (ww < W)
    val = tl.load(x_ptr + ((n * C + c) * H + hh) * W + ww, mask=mask, other=0.0)
    tl.store(flags_ptr + pid, (tl.max(tl.abs(val)) > THRESHOLD).to(tl.int32))

@triton.jit
def dense_conv3x3_kernel(
    x_ptr, y_ptr, N, C, H, W, GRID_H, GRID_W,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N * C * GRID_H * GRID_W:
        return
    gw = pid % GRID_W;  tmp = pid // GRID_W
    gh = tmp % GRID_H;  tmp = tmp // GRID_H
    c  = tmp % C;        n   = tmp // C
    hh = (gh * BLOCK_H + tl.arange(0, BLOCK_H))[:, None]
    ww = (gw * BLOCK_W + tl.arange(0, BLOCK_W))[None, :]
    mask = (hh < H) & (ww < W)
    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    for kh in range(-1, 2):
        for kw in range(-1, 2):
            hi = hh + kh; wi = ww + kw
            m = mask & (hi >= 0) & (hi < H) & (wi >= 0) & (wi < W)
            acc += tl.load(x_ptr + ((n*C+c)*H + hi)*W + wi, mask=m, other=0.0)
    tl.store(y_ptr + ((n*C+c)*H + hh)*W + ww, acc, mask=mask)


# ═══════════════════════════════════════════════════════════════════════
#  Main Entry
# ═══════════════════════════════════════════════════════════════════════

def sparse_conv2d_forward(x, weight, bias, block_size,
                          kernel_size=3, threshold=0.0):
    """
    Sparse Conv2d via im2col + channel-pruned GEMM.
    Returns: (y [N,C_OUT,H,W] fp32,  sparse_ms float)
    """
    N, C_IN, H, W = x.shape
    C_OUT = weight.shape[0]
    HW = H * W
    device = x.device

    x_f16 = x.half().contiguous()
    w_f16 = weight.half().contiguous()
    b_f16 = bias.half().contiguous() if bias is not None else None

    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)

    if kernel_size == 1:
        x_2d = x_f16.permute(1, 0, 2, 3).reshape(C_IN, N * HW).contiguous()
        nz = _nz_channels(x_2d, threshold)
        if nz.numel() == 0:
            y = torch.zeros(N, C_OUT, H, W, dtype=x.dtype, device=device)
            if bias is not None: y += bias.view(1, -1, 1, 1)
            return y, 0.0

        x_nz = x_2d[nz].contiguous()
        w_nz = w_f16.reshape(C_OUT, C_IN)[:, nz].contiguous()
        BM, BN, BK = _gemm_config(C_OUT, N * HW, nz.numel())

        ev_s.record()
        y_2d = _triton_gemm(w_nz, x_nz, b_f16, BM, BN, BK)
        ev_e.record()
        torch.cuda.synchronize(device)

        y = y_2d.view(C_OUT, N, H, W).permute(1, 0, 2, 3).contiguous()
        return y.float(), ev_s.elapsed_time(ev_e)

    else:
        col = F_torch.unfold(x_f16, kernel_size=3, padding=1, stride=1)
        CK = C_IN * 9
        col_2d = col.permute(1, 0, 2).reshape(CK, N * HW).contiguous()

        nz_grp = _nz_channel_groups(col_2d, group_size=9, threshold=threshold)
        if nz_grp.numel() == 0:
            y = torch.zeros(N, C_OUT, H, W, dtype=x.dtype, device=device)
            if bias is not None: y += bias.view(1, -1, 1, 1)
            return y, 0.0

        nz_rows = (nz_grp[:, None] * 9 + torch.arange(9, device=device)).reshape(-1)
        col_nz = col_2d[nz_rows].contiguous()
        w_nz = w_f16.reshape(C_OUT, CK)[:, nz_rows].contiguous()
        NZ_K = nz_rows.numel()
        BM, BN, BK = _gemm_config(C_OUT, N * HW, NZ_K)

        ev_s.record()
        y_2d = _triton_gemm(w_nz, col_nz, b_f16, BM, BN, BK)
        ev_e.record()
        torch.cuda.synchronize(device)

        y = y_2d.view(C_OUT, N, H, W).permute(1, 0, 2, 3).contiguous()
        return y.float(), ev_s.elapsed_time(ev_e)