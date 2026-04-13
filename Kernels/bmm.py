"""
SparseFlow Kernels/bmm.py - Sparse Batched Matmul Triton kernel.

Maturity: main_path (production-facing sparse kernel).

Sparse BMM for [B, M, K] x [B, K, N] -> [B, M, N].
Per-batch prescan on A, then dispatch per (batch, m-tile, n-tile).

Critical for spike-transformer attention stages (QK^T and AttnV).

Change-log
----------
  - Round 7: fixed autotune/prescan BLOCK_M misalignment by per-BM kernel
    specialization. The launch grid also uses meta["BLOCK_N"] so BLOCK_N
    autotune remains aligned with the output-column launch geometry.
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Utils.sparse_helpers import (
    TILE_ZERO, TILE_SPARSE, TILE_DENSEISH,
    choose_group_size, popcount_buf,
)
from Utils.config import (
    ENABLE_RUNTIME_FALLBACK_POLICY,
    PRESCAN_ACTIVITY_EPS,
    SPARSE_DENSE_RATIO_THRESHOLD,
)

FALLBACK_RATIO = SPARSE_DENSE_RATIO_THRESHOLD
TRITON_MAX_TENSOR_NUMEL = 131072


def _make_bmm_configs(block_m: int):
    configs = []
    for block_n in (16, 32, 64):
        for block_k in (32, 64):
            for num_warps in (4, 8):
                for num_stages in (1, 2):
                    if block_n == 16 and num_warps == 8:
                        continue
                    configs.append(
                        triton.Config(
                            {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k},
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                    )
    return configs


_BMM_CONFIGS_BM16 = _make_bmm_configs(16)
_BMM_CONFIGS_BM32 = _make_bmm_configs(32)
_BMM_CONFIGS_BM64 = _make_bmm_configs(64)


@triton.jit
def _prescan_bmm_kernel(
    a_ptr,
    ag_mask_ptr,
    tile_class_ptr,
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
    pid = tl.program_id(0)
    batch_idx = pid // N_TILES_M
    tile_idx = pid % N_TILES_M

    row_start = tile_idx * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < M

    batch_offset = batch_idx * M * K
    off1 = tl.arange(0, 1)
    ag_mask = tl.zeros([1], dtype=tl.int32)
    any_nonzero = tl.zeros([1], dtype=tl.int32)

    for g in range(NUM_GROUPS):
        col_start = g * GROUP_SIZE_C
        cols = col_start + tl.arange(0, GROUP_SIZE_C)
        col_mask = cols < K

        addrs = batch_offset + rows[:, None] * K + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]
        vals = tl.load(a_ptr + addrs, mask=mask, other=0.0)

        has_nonzero = (tl.sum((tl.abs(vals) > THRESHOLD).to(tl.int32), axis=0) > 0).to(tl.int32)
        ag_mask = ag_mask + has_nonzero * (1 << g)
        any_nonzero = tl.maximum(any_nonzero, has_nonzero)

    out_idx = batch_idx * N_TILES_M + tile_idx
    tl.store(ag_mask_ptr + out_idx + off1, ag_mask)

    if tl.sum(any_nonzero) == 0:
        tl.store(tile_class_ptr + out_idx + off1, tl.zeros([1], dtype=tl.int32))
    else:
        if tl.sum(ag_mask == ALL_ONES) > 0:
            tl.store(tile_class_ptr + out_idx + off1, tl.full([1], TILE_DENSEISH, dtype=tl.int32))
        else:
            tl.store(tile_class_ptr + out_idx + off1, tl.full([1], TILE_SPARSE, dtype=tl.int32))


def _make_sparse_bmm_kernel(configs):
    @triton.autotune(configs=configs, key=['M', 'N', 'K'])
    @triton.jit
    def _kernel(
        a_ptr, b_ptr, c_ptr,
        ag_mask_ptr, tile_class_ptr,
        B_dim: tl.constexpr,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        N_TILES_M: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_SIZE_C: tl.constexpr, NUM_GROUPS: tl.constexpr,
    ):
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
        off1 = tl.arange(0, 1)
        tile_cls = tl.load(tile_class_ptr + meta_idx + off1)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        if tl.sum(tile_cls) == TILE_ZERO:
            pass
        elif tl.sum(tile_cls) == TILE_DENSEISH:
            for k_start in range(0, K, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)
                k_mask = offs_k < K

                a_addrs = a_batch + offs_m[:, None] * K + offs_k[None, :]
                a_vals = tl.load(a_ptr + a_addrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

                b_addrs = b_batch + offs_k[:, None] * N + offs_n[None, :]
                b_vals = tl.load(b_ptr + b_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)

                acc += tl.dot(a_vals, b_vals)
        else:
            ag = tl.load(ag_mask_ptr + meta_idx + off1)
            for g in range(NUM_GROUPS):
                g_active = (ag >> g) & 1
                if tl.sum(g_active) != 0:
                    g_base = g * GROUP_SIZE_C
                    for k_off in range(0, GROUP_SIZE_C, BLOCK_K):
                        offs_k = g_base + k_off + tl.arange(0, BLOCK_K)
                        k_mask = offs_k < K

                        a_addrs = a_batch + offs_m[:, None] * K + offs_k[None, :]
                        a_vals = tl.load(a_ptr + a_addrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

                        b_addrs = b_batch + offs_k[:, None] * N + offs_n[None, :]
                        b_vals = tl.load(b_ptr + b_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)

                        acc += tl.dot(a_vals, b_vals)

        out_addrs = c_batch + offs_m[:, None] * N + offs_n[None, :]
        out_mask = m_mask[:, None] & n_mask[None, :]
        tl.store(c_ptr + out_addrs, acc, mask=out_mask)

    return _kernel


_sparse_bmm_kernel_bm16 = _make_sparse_bmm_kernel(_BMM_CONFIGS_BM16)
_sparse_bmm_kernel_bm32 = _make_sparse_bmm_kernel(_BMM_CONFIGS_BM32)
_sparse_bmm_kernel_bm64 = _make_sparse_bmm_kernel(_BMM_CONFIGS_BM64)


def sparse_bmm_forward(
    a: torch.Tensor,
    b: torch.Tensor,
    threshold: float = PRESCAN_ACTIVITY_EPS,
    ag_mask_buf: torch.Tensor = None,
    tile_class_buf: torch.Tensor = None,
    return_ms: bool = False,
    return_avg_active_ratio: bool = False,
    return_tile_stats: bool = False,
    fallback_ratio: float = FALLBACK_RATIO,
):
    """Sparse BMM: C[b] = A[b] @ B[b], where A is expected sparse."""
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

    if M * K < 512 or B_dim * M * N < 4096:
        c = torch.bmm(a.float(), b.float())
        return _finalize(c, 0.0, 1.0 if return_avg_active_ratio else None, None)

    GROUP_SIZE_C = choose_group_size(K)
    NUM_GROUPS = triton.cdiv(K, GROUP_SIZE_C)
    ALL_ONES = (1 << NUM_GROUPS) - 1

    if M >= 128:
        BM = 64
    elif M >= 32:
        BM = 32
    else:
        BM = 16
    N_TILES_M = triton.cdiv(M, BM)

    if BM == 16:
        configs = _BMM_CONFIGS_BM16
        kernel = _sparse_bmm_kernel_bm16
    elif BM == 32:
        configs = _BMM_CONFIGS_BM32
        kernel = _sparse_bmm_kernel_bm32
    elif BM == 64:
        configs = _BMM_CONFIGS_BM64
        kernel = _sparse_bmm_kernel_bm64
    else:
        raise ValueError(f"unsupported BM={BM}")

    max_block_n = max(cfg.kwargs['BLOCK_N'] for cfg in configs)
    if GROUP_SIZE_C * max_block_n > TRITON_MAX_TENSOR_NUMEL:
        c = torch.bmm(a.float(), b.float())
        stats = {
            'total_tiles': B_dim * N_TILES_M,
            'fallback': True,
            'reason': f'group_size_too_large(gs={GROUP_SIZE_C}, max_bn={max_block_n})',
        } if return_tile_stats else None
        ratio = 1.0 if return_avg_active_ratio else None
        return _finalize(c, 0.0, ratio, stats)

    max_block_k = max(cfg.kwargs['BLOCK_K'] for cfg in configs)
    est_smem_bytes = ((BM * max_block_k) + (max_block_k * max_block_n) + (BM * max_block_n)) * 4
    if est_smem_bytes > 160_000:
        c = torch.bmm(a.float(), b.float())
        stats = {
            'total_tiles': B_dim * N_TILES_M,
            'fallback': True,
            'reason': f'estimated_shared_memory_too_large(smem={est_smem_bytes})',
        } if return_tile_stats else None
        ratio = 1.0 if return_avg_active_ratio else None
        return _finalize(c, 0.0, ratio, stats)

    TOTAL_META = B_dim * N_TILES_M
    a_f16 = a.half().contiguous()

    if ag_mask_buf is None or ag_mask_buf.numel() < TOTAL_META:
        ag_mask_buf = torch.empty(TOTAL_META, dtype=torch.int32, device=device)
    if tile_class_buf is None or tile_class_buf.numel() < TOTAL_META:
        tile_class_buf = torch.empty(TOTAL_META, dtype=torch.int32, device=device)

    try:
        _prescan_bmm_kernel[(TOTAL_META,)](
            a_f16, ag_mask_buf, tile_class_buf,
            B=B_dim, M=M, K=K, N_TILES_M=N_TILES_M,
            BLOCK_M=BM, GROUP_SIZE_C=GROUP_SIZE_C,
            NUM_GROUPS=NUM_GROUPS, ALL_ONES=ALL_ONES,
            THRESHOLD=threshold,
        )
    except Exception:
        c = torch.bmm(a.float(), b.float())
        stats = {
            'total_tiles': TOTAL_META,
            'fallback': True,
            'reason': 'prescan_failed',
        } if return_tile_stats else None
        ratio = 1.0 if return_avg_active_ratio else None
        return _finalize(c, 0.0, ratio, stats)

    avg_ratio = None
    if need_stats:
        pc = popcount_buf(ag_mask_buf, TOTAL_META)
        avg_active = pc.float().mean().item()
        avg_ratio = avg_active / max(NUM_GROUPS, 1)

    if ENABLE_RUNTIME_FALLBACK_POLICY and avg_ratio is not None and avg_ratio > fallback_ratio:
        c = torch.bmm(a.float(), b.float())
        stats = {'total_tiles': TOTAL_META, 'fallback': True} if return_tile_stats else None
        return _finalize(c, 0.0, avg_ratio, stats)

    b_f16 = b.half().contiguous()
    c = torch.empty(B_dim, M, N, dtype=torch.float16, device=device)

    if return_ms:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    def _grid(meta):
        return (B_dim * N_TILES_M, triton.cdiv(N, meta["BLOCK_N"]))

    try:
        kernel[_grid](
            a_f16, b_f16, c,
            ag_mask_buf, tile_class_buf,
            B_dim=B_dim, M=M, N=N, K=K,
            N_TILES_M=N_TILES_M,
            GROUP_SIZE_C=GROUP_SIZE_C, NUM_GROUPS=NUM_GROUPS,
        )
    except Exception:
        c = torch.bmm(a.float(), b.float())
        stats = {
            'total_tiles': TOTAL_META,
            'fallback': True,
            'reason': 'kernel_launch_failed',
            'avg_active_ratio': avg_ratio,
        } if return_tile_stats else None
        ratio = avg_ratio if return_avg_active_ratio else None
        if return_avg_active_ratio and ratio is None:
            ratio = 1.0
        return _finalize(c, 0.0, ratio, stats)

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
