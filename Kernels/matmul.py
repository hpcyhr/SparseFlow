"""
SparseFlow Kernels/matmul.py - Sparse Matmul Triton kernel.

Maturity: main_path (production-facing sparse kernel).

Grouped-bitmask sparse matmul for [M, K] x [K, N] -> [M, N].
Follows the same two-stage prescan + tile-dispatch pattern as conv2d/linear,
adapted for functional matmul APIs.
"""

import torch
import triton
import triton.language as tl

import sys
from pathlib import Path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Utils.sparse_helpers import (
    TILE_ZERO, TILE_SPARSE, TILE_DENSEISH,
    choose_group_size, select_row_tile_sizes,
    build_row_metadata, popcount_buf,
)
from Utils.config import PRESCAN_ACTIVITY_EPS, SPARSE_DENSE_RATIO_THRESHOLD

FALLBACK_RATIO = SPARSE_DENSE_RATIO_THRESHOLD
TRITON_MAX_TENSOR_NUMEL = 131072

_MATMUL_CONFIGS = [
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8),
]

# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.autotune(configs=_MATMUL_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def _sparse_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    ag_mask_ptr, tile_class_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, NUM_GROUPS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N

    # Tile classification from prescan (per M-tile)
    off1 = tl.arange(0, 1)
    tile_cls = tl.load(tile_class_ptr + pid_m + off1)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if tl.sum(tile_cls) == TILE_ZERO:
        # All-zero tile in A 鈫?output is zero
        pass

    elif tl.sum(tile_cls) == TILE_DENSEISH:
        # Dense path 鈥?iterate all K groups
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            a_addrs = offs_m[:, None] * K + offs_k[None, :]
            a_vals = tl.load(a_ptr + a_addrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

            b_addrs = offs_k[:, None] * N + offs_n[None, :]
            b_vals = tl.load(b_ptr + b_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)

            acc += tl.dot(a_vals, b_vals)

    else:
        # Sparse path 鈥?bitmask-gated groups
        ag = tl.load(ag_mask_ptr + pid_m + off1)
        for g in range(NUM_GROUPS):
            g_active = (ag >> g) & 1
            if tl.sum(g_active) != 0:
                g_base = g * GROUP_SIZE_C
                for k_off in range(0, GROUP_SIZE_C, BLOCK_K):
                    offs_k = g_base + k_off + tl.arange(0, BLOCK_K)
                    k_mask = offs_k < K

                    a_addrs = offs_m[:, None] * K + offs_k[None, :]
                    a_vals = tl.load(a_ptr + a_addrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

                    b_addrs = offs_k[:, None] * N + offs_n[None, :]
                    b_vals = tl.load(b_ptr + b_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)

                    acc += tl.dot(a_vals, b_vals)

    out_addrs = offs_m[:, None] * N + offs_n[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptr + out_addrs, acc, mask=out_mask)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def sparse_matmul_forward(
    a: torch.Tensor,       # [M, K]
    b: torch.Tensor,       # [K, N]
    threshold: float = PRESCAN_ACTIVITY_EPS,
    ag_mask_buf: torch.Tensor = None,
    tile_class_buf: torch.Tensor = None,
    return_ms: bool = False,
    return_avg_active_ratio: bool = False,
    return_tile_stats: bool = False,
    fallback_ratio: float = FALLBACK_RATIO,
):
    """
    Sparse matmul: C = A @ B where A is expected to be sparse (SNN spikes).

    Returns:
        (C, ms) + optional (avg_active_ratio,) + optional (tile_stats,)
    """
    assert a.ndim == 2 and b.ndim == 2, f"Expected 2D inputs, got {a.shape}, {b.shape}"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"K mismatch: {K} vs {K2}"
    device = a.device

    need_stats = return_avg_active_ratio or return_tile_stats

    def _finalize(c, ms, ratio=None, stats=None):
        ret = (c, ms)
        if return_avg_active_ratio:
            ret = ret + (ratio,)
        if return_tile_stats:
            ret = ret + (stats,)
        return ret

    # Minimum size guard
    if M * K < 1024 or M * N < 1024:
        c = torch.mm(a.float(), b.float())
        return _finalize(c, 0.0, 1.0 if return_avg_active_ratio else None,
                         None)

    GROUP_SIZE_C = choose_group_size(K)
    NUM_GROUPS = triton.cdiv(K, GROUP_SIZE_C)
    BM, BN = select_row_tile_sizes(M, N)
    N_TILES_M = triton.cdiv(M, BM)

    max_block_n = max(cfg.kwargs['BLOCK_N'] for cfg in _MATMUL_CONFIGS)
    if GROUP_SIZE_C * max_block_n > TRITON_MAX_TENSOR_NUMEL:
        c = torch.mm(a.float(), b.float())
        tile_stats = {
            'total_tiles': N_TILES_M,
            'fallback': True,
            'reason': f'group_size_too_large(gs={GROUP_SIZE_C}, max_bn={max_block_n})',
        } if return_tile_stats else None
        ratio = 1.0 if return_avg_active_ratio else None
        return _finalize(c, 0.0, ratio, tile_stats)

    max_block_k = max(cfg.kwargs['BLOCK_K'] for cfg in _MATMUL_CONFIGS)
    est_smem_bytes = ((BM * max_block_k) + (max_block_k * max_block_n) + (BM * max_block_n)) * 4
    if est_smem_bytes > 160_000:
        c = torch.mm(a.float(), b.float())
        tile_stats = {
            'total_tiles': N_TILES_M,
            'fallback': True,
            'reason': f'estimated_shared_memory_too_large(smem={est_smem_bytes})',
        } if return_tile_stats else None
        ratio = 1.0 if return_avg_active_ratio else None
        return _finalize(c, 0.0, ratio, tile_stats)

    a_f16 = a.half().contiguous()
    b_f16 = b.half().contiguous()

    # Prescan
    try:
        ag_mask_buf, tile_class_buf, _ = build_row_metadata(
            a_f16, M, K, BM, GROUP_SIZE_C, threshold,
            ag_mask_buf, tile_class_buf,
        )
    except Exception:
        c = torch.mm(a.float(), b.float())
        tile_stats = {
            'total_tiles': N_TILES_M,
            'fallback': True,
            'reason': 'prescan_failed',
        } if return_tile_stats else None
        ratio = 1.0 if return_avg_active_ratio else None
        return _finalize(c, 0.0, ratio, tile_stats)

    # Dense fallback check (only if we need stats or ratio is high)
    if need_stats:
        pc = popcount_buf(ag_mask_buf, N_TILES_M)
        avg_active = pc.float().mean().item()
        avg_ratio = avg_active / max(NUM_GROUPS, 1)
    else:
        avg_ratio = None

    if avg_ratio is not None and avg_ratio > fallback_ratio:
        c = torch.mm(a.float(), b.float())
        tile_stats = {
            'total_tiles': N_TILES_M,
            'fallback': True,
            'avg_active_ratio': avg_ratio,
        } if return_tile_stats else None
        return _finalize(c, 0.0, avg_ratio, tile_stats)

    # Allocate output
    c = torch.empty(M, N, dtype=torch.float32, device=device)

    # Timing
    if return_ms:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    grid = (N_TILES_M, triton.cdiv(N, BN))
    try:
        _sparse_matmul_kernel[grid](
            a_f16, b_f16, c,
            ag_mask_buf, tile_class_buf,
            M=M, N=N, K=K,
            GROUP_SIZE_C=GROUP_SIZE_C, NUM_GROUPS=NUM_GROUPS,
        )
    except Exception:
        c = torch.mm(a.float(), b.float())
        tile_stats = {
            'total_tiles': N_TILES_M,
            'fallback': True,
            'reason': 'kernel_launch_failed',
            'avg_active_ratio': avg_ratio,
        } if return_tile_stats else None
        ratio = avg_ratio if return_avg_active_ratio else None
        if return_avg_active_ratio and ratio is None:
            ratio = 1.0
        return _finalize(c, 0.0, ratio, tile_stats)

    ms = 0.0
    if return_ms:
        end.record()
        torch.cuda.synchronize(device)
        ms = start.elapsed_time(end)

    tile_stats = None
    if return_tile_stats:
        tc = tile_class_buf[:N_TILES_M]
        tile_stats = {
            'total_tiles': N_TILES_M,
            'zero_tiles': int((tc == TILE_ZERO).sum().item()),
            'sparse_tiles': int((tc == TILE_SPARSE).sum().item()),
            'denseish_tiles': int((tc == TILE_DENSEISH).sum().item()),
            'fallback': False,
        }

    return _finalize(c, ms, avg_ratio, tile_stats)

