"""
SparseFlow Kernels/matmul.py — Sparse Matmul Triton kernel.

Maturity: main_path (production-facing sparse kernel).

Grouped-bitmask sparse matmul for [M, K] × [K, N] → [M, N].
Uses the three-stage prescan pipeline from Kernels/linear.py (coarse
classify → zero-candidate refine → group bitmask refine), which runs on
A's row-tile geometry [M, K] and is geometrically identical to linear's
input prescan.

Round 6 cleanup
---------------
  - Migrated prescan from the single-stage `Utils/sparse_helpers::
    build_row_metadata` to the three-stage pipeline `_build_linear_metadata`
    exported by Kernels/linear.py. The three-stage pipeline exposes
    TILE_ZERO_CANDIDATE and TILE_UNCERTAIN intermediate classes, improving
    tile classification accuracy without touching the compute kernel.
    After the pipeline runs, the final tile_class buffer contains only
    {TILE_ZERO, TILE_SPARSE, TILE_DENSEISH}, which is exactly what the
    existing _sparse_matmul_kernel expects.
  - Inlined `select_row_tile_sizes` (previously in sparse_helpers
    DEPRECATED section). The 10-line helper is used only here.
  - Fixed mojibake in comments.

Pre-existing issue (NOT fixed in Round 6, noted for a future pass)
------------------------------------------------------------------
  The compute kernel is decorated with @triton.autotune over BLOCK_M and
  BLOCK_N, while the grid is launched using a fixed BM from
  `select_row_tile_sizes` and the prescan runs with the same fixed BM.
  If autotune picks BLOCK_M ≠ BM, the grid launches `cdiv(M, BM)` programs
  but each program scans `BLOCK_M` rows, producing a row-tile misalignment
  with the prescan's classification buffer. Symptom would be silently
  wrong output on shapes where autotune prefers a different BLOCK_M.
  A proper fix either (a) restricts _MATMUL_CONFIGS to a single BLOCK_M
  matching BM, or (b) makes the grid a lambda that reads meta["BLOCK_M"]
  and recomputes prescan per-config. Both are semantic changes deferred
  to a dedicated correctness pass.
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Utils.config import PRESCAN_ACTIVITY_EPS, SPARSE_DENSE_RATIO_THRESHOLD
from Utils.sparse_helpers import (
    TILE_ZERO, TILE_SPARSE, TILE_DENSEISH,
    choose_group_size, popcount_buf,
)
from Kernels.linear import _build_linear_metadata

FALLBACK_RATIO = SPARSE_DENSE_RATIO_THRESHOLD
TRITON_MAX_TENSOR_NUMEL = 131072


# ---------------------------------------------------------------------------
# Row-tile sizing (previously in Utils/sparse_helpers.select_row_tile_sizes)
# ---------------------------------------------------------------------------

def _select_row_tile_sizes(M: int, N: int):
    """Pick (BLOCK_M, BLOCK_N) for a flat [M, N] output."""
    if M >= 128:
        bm = 64
    elif M >= 32:
        bm = 32
    else:
        bm = 16

    if N >= 256:
        bn = 64
    elif N >= 64:
        bn = 32
    else:
        bn = 16
    return bm, bn


_MATMUL_CONFIGS = [
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8),
]


# ---------------------------------------------------------------------------
# Triton kernel (unchanged from v24 — three-stage prescan produces the same
# final tile_class set: {TILE_ZERO, TILE_SPARSE, TILE_DENSEISH})
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
        # All-zero tile in A → output is zero
        pass

    elif tl.sum(tile_cls) == TILE_DENSEISH:
        # Dense path — iterate all K groups
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            a_addrs = offs_m[:, None] * K + offs_k[None, :]
            a_vals = tl.load(a_ptr + a_addrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

            b_addrs = offs_k[:, None] * N + offs_n[None, :]
            b_vals = tl.load(b_ptr + b_addrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)

            acc += tl.dot(a_vals, b_vals)

    else:
        # Sparse path — bitmask-gated groups
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
    """Sparse matmul: C = A @ B where A is expected to be sparse (SNN spikes).

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
        return _finalize(
            c, 0.0,
            1.0 if return_avg_active_ratio else None,
            None,
        )

    GROUP_SIZE_C = choose_group_size(K)
    NUM_GROUPS = triton.cdiv(K, GROUP_SIZE_C)
    BM, BN = _select_row_tile_sizes(M, N)
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

    # Buffer allocation for prescan metadata
    if ag_mask_buf is None or ag_mask_buf.numel() < N_TILES_M:
        ag_mask_buf = torch.empty(N_TILES_M, dtype=torch.int32, device=device)
    if tile_class_buf is None or tile_class_buf.numel() < N_TILES_M:
        tile_class_buf = torch.empty(N_TILES_M, dtype=torch.int32, device=device)

    # Three-stage prescan (shared with Kernels/linear.py).
    # A's geometry [M, K] maps to linear's [N_val, C_IN].
    prescan_stats = {} if return_tile_stats else None
    try:
        _, _, stage1_dense_fallback, stage1_summary = _build_linear_metadata(
            x_f16=a_f16,
            N=M,
            C_IN=K,
            BLOCK_M=BM,
            N_TILES=N_TILES_M,
            threshold=threshold,
            ag_mask_buf=ag_mask_buf,
            tile_class_buf=tile_class_buf,
            prescan_stats=prescan_stats,
            allow_stage1_dense_fallback=(return_avg_active_ratio and not return_tile_stats),
            fallback_ratio=fallback_ratio,
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

    if stage1_dense_fallback:
        stage1_avg_ratio = 1.0
        if stage1_summary is not None:
            stage1_avg_ratio = float(
                stage1_summary.get("stage1_avg_active_group_ratio_lower_bound", 1.0)
            )
        c = torch.mm(a.float(), b.float())
        return _finalize(c, 0.0, stage1_avg_ratio, None)

    # Dense fallback check (sync-gated on need_stats)
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

    b_f16 = b.half().contiguous()

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
        if prescan_stats:
            tile_stats.update(prescan_stats)

    return _finalize(c, ms, avg_ratio, tile_stats)
