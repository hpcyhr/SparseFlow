"""
SparseFlow Kernels/bmm.py — Sparse Batched Matmul Triton Kernel v1.0

Sparse BMM for [B, M, K] × [B, K, N] → [B, M, N].
Per-batch prescan on the A tensor, then dispatches per (batch, m-tile, n-tile).

Critical for Spikeformer attention (Q×K^T, attn×V) where Q/attn carry
spike sparsity from upstream LIF neurons.
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
    choose_group_size, popcount_buf,
)

FALLBACK_RATIO = 0.85

# ---------------------------------------------------------------------------
# Prescan kernel — per batch element
# ---------------------------------------------------------------------------

@triton.jit
def _prescan_bmm_kernel(
    a_ptr,             # [B, M, K] row-major
    ag_mask_ptr,       # [B * N_TILES_M]
    tile_class_ptr,    # [B * N_TILES_M]
    B: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    N_TILES_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    ALL_ONES: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
    # pid encodes (batch_idx, tile_m_idx)
    pid = tl.program_id(0)
    batch_idx = pid // N_TILES_M
    tile_idx = pid % N_TILES_M

    row_start = tile_idx * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < M

    batch_offset = batch_idx * M * K
    ag_mask = tl.zeros([], dtype=tl.int32)
    any_nonzero = tl.zeros([], dtype=tl.int32)

    for g in range(NUM_GROUPS):
        col_start = g * GROUP_SIZE_C
        cols = col_start + tl.arange(0, GROUP_SIZE_C)
        col_mask = cols < K

        addrs = batch_offset + rows[:, None] * K + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]
        vals = tl.load(a_ptr + addrs, mask=mask, other=0.0)

        has_nonzero = tl.sum(tl.abs(vals) > THRESHOLD) > 0
        if has_nonzero:
            ag_mask = ag_mask | (1 << g)
            any_nonzero = 1

    out_idx = batch_idx * N_TILES_M + tile_idx
    tl.store(ag_mask_ptr + out_idx, ag_mask)

    cls = TILE_ZERO
    if any_nonzero != 0:
        if ag_mask == ALL_ONES:
            cls = TILE_DENSEISH
        else:
            cls = TILE_SPARSE
    tl.store(tile_class_ptr + out_idx, cls)


# ---------------------------------------------------------------------------
# Compute kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _sparse_bmm_kernel(
    a_ptr, b_ptr, c_ptr,
    ag_mask_ptr, tile_class_ptr,
    B_dim: tl.constexpr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    N_TILES_M: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr, NUM_GROUPS: tl.constexpr,
):
    # Grid: (B * N_TILES_M, N_TILES_N)
    pid_bm = tl.program_id(0)
    pid_n = tl.program_id(1)

    batch_idx = pid_bm // N_TILES_M
    tile_m = pid_bm % N_TILES_M

    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N

    a_batch = batch_idx * M * K
    b_batch = batch_idx * K * N
    c_batch = batch_idx * M * N

    meta_idx = batch_idx * N_TILES_M + tile_m
    tile_cls = tl.load(tile_class_ptr + meta_idx)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if tile_cls == TILE_ZERO:
        pass

    elif tile_cls == TILE_DENSEISH:
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            a_addrs = a_batch + offs_m[:, None] * K + offs_k[None, :]
            a_vals = tl.load(a_ptr + a_addrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

            b_addrs = b_batch + offs_k[:, None] * N + offs_n[None, :]
            b_vals = tl.load(b_ptr + b_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)

            acc += tl.dot(a_vals, b_vals)

    else:
        ag = tl.load(ag_mask_ptr + meta_idx)
        for g in range(NUM_GROUPS):
            g_active = (ag >> g) & 1
            if g_active != 0:
                offs_k = g * GROUP_SIZE_C + tl.arange(0, GROUP_SIZE_C)
                k_mask = offs_k < K

                a_addrs = a_batch + offs_m[:, None] * K + offs_k[None, :]
                a_vals = tl.load(a_ptr + a_addrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

                b_addrs = b_batch + offs_k[:, None] * N + offs_n[None, :]
                b_vals = tl.load(b_ptr + b_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)

                acc += tl.dot(a_vals, b_vals)

    out_addrs = c_batch + offs_m[:, None] * N + offs_n[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptr + out_addrs, acc, mask=out_mask)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def sparse_bmm_forward(
    a: torch.Tensor,       # [B, M, K]
    b: torch.Tensor,       # [B, K, N]
    threshold: float = 1e-6,
    ag_mask_buf: torch.Tensor = None,
    tile_class_buf: torch.Tensor = None,
    return_ms: bool = False,
    return_avg_active_ratio: bool = False,
    return_tile_stats: bool = False,
    fallback_ratio: float = FALLBACK_RATIO,
):
    """
    Sparse BMM: C[b] = A[b] @ B[b], where A is expected sparse.

    Returns:
        (C, ms) + optional (avg_active_ratio,) + optional (tile_stats,)
    """
    assert a.ndim == 3 and b.ndim == 3
    B_dim, M, K = a.shape
    B2, K2, N = b.shape
    assert B_dim == B2 and K == K2
    device = a.device

    need_stats = return_avg_active_ratio or return_tile_stats

    def _finalize(c, ms, ratio=None, stats=None):
        ret = (c, ms)
        if return_avg_active_ratio:
            ret = ret + (ratio,)
        if return_tile_stats:
            ret = ret + (stats,)
        return ret

    # Small-problem fallback
    if M * K < 512 or B_dim * M * N < 4096:
        c = torch.bmm(a.float(), b.float())
        return _finalize(c, 0.0, 1.0 if return_avg_active_ratio else None, None)

    GROUP_SIZE_C = choose_group_size(K)
    NUM_GROUPS = triton.cdiv(K, GROUP_SIZE_C)
    ALL_ONES = (1 << NUM_GROUPS) - 1

    # Tile sizes
    BM = 32 if M >= 32 else 16
    BN = 32 if N >= 32 else 16
    N_TILES_M = triton.cdiv(M, BM)
    N_TILES_N = triton.cdiv(N, BN)

    TOTAL_META = B_dim * N_TILES_M
    a_f16 = a.half().contiguous()
    b_f16 = b.half().contiguous()

    # Allocate metadata
    if ag_mask_buf is None or ag_mask_buf.numel() < TOTAL_META:
        ag_mask_buf = torch.empty(TOTAL_META, dtype=torch.int32, device=device)
    if tile_class_buf is None or tile_class_buf.numel() < TOTAL_META:
        tile_class_buf = torch.empty(TOTAL_META, dtype=torch.int32, device=device)

    # Prescan
    _prescan_bmm_kernel[(TOTAL_META,)](
        a_f16, ag_mask_buf, tile_class_buf,
        B=B_dim, M=M, K=K, N_TILES_M=N_TILES_M,
        BLOCK_M=BM, GROUP_SIZE_C=GROUP_SIZE_C,
        NUM_GROUPS=NUM_GROUPS, ALL_ONES=ALL_ONES,
        THRESHOLD=threshold,
    )

    # Ratio check
    avg_ratio = None
    if need_stats:
        pc = popcount_buf(ag_mask_buf, TOTAL_META)
        avg_active = pc.float().mean().item()
        avg_ratio = avg_active / max(NUM_GROUPS, 1)

    if avg_ratio is not None and avg_ratio > fallback_ratio:
        c = torch.bmm(a.float(), b.float())
        stats = {'total_tiles': TOTAL_META, 'fallback': True} if return_tile_stats else None
        return _finalize(c, 0.0, avg_ratio, stats)

    # Output
    c = torch.empty(B_dim, M, N, dtype=torch.float32, device=device)

    if return_ms:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    grid = (B_dim * N_TILES_M, N_TILES_N)
    _sparse_bmm_kernel[grid](
        a_f16, b_f16, c,
        ag_mask_buf, tile_class_buf,
        B_dim=B_dim, M=M, N=N, K=K,
        N_TILES_M=N_TILES_M,
        GROUP_SIZE_C=GROUP_SIZE_C, NUM_GROUPS=NUM_GROUPS,
    )

    ms = 0.0
    if return_ms:
        end.record()
        torch.cuda.synchronize(device)
        ms = start.elapsed_time(end)

    stats = None
    if return_tile_stats:
        tc = tile_class_buf[:TOTAL_META]
        stats = {
            'total_tiles': TOTAL_META,
            'zero_tiles': int((tc == TILE_ZERO).sum().item()),
            'sparse_tiles': int((tc == TILE_SPARSE).sum().item()),
            'denseish_tiles': int((tc == TILE_DENSEISH).sum().item()),
            'fallback': False,
        }

    return _finalize(c, ms, avg_ratio, stats)