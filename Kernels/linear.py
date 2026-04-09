"""
SparseFlow Kernels/linear.py

Maturity: main_path (production-facing sparse kernel).
"""

import torch
import triton
import triton.language as tl
from triton import autotune, Config
from Utils.config import PRESCAN_ACTIVITY_EPS, SPARSE_DENSE_RATIO_THRESHOLD
from Utils.sparse_helpers import (
    TILE_ZERO,
    TILE_SPARSE,
    TILE_DENSEISH,
    TILE_UNCERTAIN,
    TILE_ZERO_CANDIDATE,
    choose_group_size,
    popcount_buf,
)

TRITON_MAX_TENSOR_NUMEL = 131072
FALLBACK_RATIO = SPARSE_DENSE_RATIO_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_linear_block_m(n_rows: int) -> int:
    if n_rows >= 1024:
        return 128
    if n_rows >= 256:
        return 64
    return 32


def _check_dense_fallback(
    ag_mask_buf: torch.Tensor,
    n_tiles: int,
    num_groups: int,
    fallback_ratio: float = FALLBACK_RATIO,
) -> bool:
    """
    NOTE: calls .item() and syncs GPU->CPU. Only call this in need_stats paths.
    """
    if num_groups == 0:
        return False
    pc = popcount_buf(ag_mask_buf, n_tiles)
    avg_active = pc.float().mean().item()
    return avg_active > float(fallback_ratio) * float(num_groups)


def _build_active_tile_ids(tile_class_buf: torch.Tensor, n_tiles: int):
    """
    NOTE: calls torch.nonzero() and syncs GPU->CPU. Only call in active_only mode.
    """
    tc = tile_class_buf[:n_tiles]
    active = torch.nonzero(tc != TILE_ZERO, as_tuple=False).flatten()
    if active.numel() == 0:
        return active.to(dtype=torch.int32), 0
    return active.to(dtype=torch.int32).contiguous(), int(active.numel())


# ---------------------------------------------------------------------------
# Stage 1: coarse tile classification over grouped input channels
# ---------------------------------------------------------------------------

@triton.jit
def linear_tile_coarse_classify_kernel(
    x_ptr, tile_class_ptr, ag_mask_ptr,
    N_val, C_IN,
    BLOCK_M: tl.constexpr,
    THRESHOLD: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    ALL_ONES_MASK: tl.constexpr,
    UNCERTAIN_CLASS: tl.constexpr,
    ZERO_CANDIDATE_CLASS: tl.constexpr,
):
    tile_id = tl.program_id(0)
    row_start = tile_id * BLOCK_M
    if row_start >= N_val:
        return

    row_offs = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offs < N_val
    off1 = tl.arange(0, 1)

    rough_mask = tl.zeros([1], dtype=tl.int32)
    for g in range(NUM_GROUPS):
        c_rep = g * GROUP_SIZE_C
        if c_rep < C_IN:
            vals = tl.load(x_ptr + row_offs * C_IN + c_rep, mask=row_mask, other=0.0)
            is_active = (tl.max(tl.abs(vals), axis=0) > THRESHOLD).to(tl.int32)
            rough_mask = rough_mask + is_active * (1 << g)

    if tl.sum(rough_mask == ALL_ONES_MASK) > 0:
        tl.store(tile_class_ptr + tile_id + off1, tl.full([1], TILE_DENSEISH, dtype=tl.int32))
        tl.store(ag_mask_ptr + tile_id + off1, tl.full([1], ALL_ONES_MASK, dtype=tl.int32))
    else:
        if tl.sum(rough_mask) == 0:
            tl.store(tile_class_ptr + tile_id + off1, tl.full([1], ZERO_CANDIDATE_CLASS, dtype=tl.int32))
            tl.store(ag_mask_ptr + tile_id + off1, tl.zeros([1], dtype=tl.int32))
        else:
            tl.store(tile_class_ptr + tile_id + off1, tl.full([1], UNCERTAIN_CLASS, dtype=tl.int32))
            tl.store(ag_mask_ptr + tile_id + off1, rough_mask)


# ---------------------------------------------------------------------------
# Stage 2a: refine zero-candidate tiles
# ---------------------------------------------------------------------------

@triton.jit
def linear_zero_candidate_refine_kernel(
    x_ptr, tile_class_ptr, ag_mask_ptr,
    N_val, C_IN,
    BLOCK_M: tl.constexpr,
    THRESHOLD: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    ALL_ONES_MASK: tl.constexpr,
    ZERO_CANDIDATE_CLASS: tl.constexpr,
):
    tile_id = tl.program_id(0)
    row_start = tile_id * BLOCK_M
    if row_start >= N_val:
        return

    off1 = tl.arange(0, 1)
    tc = tl.load(tile_class_ptr + tile_id + off1)
    if tl.sum(tc) != ZERO_CANDIDATE_CLASS:
        return

    row_offs = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offs < N_val

    mask = tl.zeros([1], dtype=tl.int32)
    found_nz = tl.zeros([1], dtype=tl.int32)
    g_idx = tl.zeros([1], dtype=tl.int32)

    while (tl.sum(g_idx) < NUM_GROUPS) & (tl.sum(found_nz) == 0):
        g_val = tl.sum(g_idx)
        group_max = tl.zeros([1], dtype=tl.float32)
        for c_off in range(1, GROUP_SIZE_C):
            c = g_val * GROUP_SIZE_C + c_off
            if c < C_IN:
                vals = tl.load(x_ptr + row_offs * C_IN + c, mask=row_mask, other=0.0)
                group_max = tl.maximum(group_max, tl.max(tl.abs(vals), axis=0))
        is_active = (group_max > THRESHOLD).to(tl.int32)
        mask = mask + is_active * (1 << g_val)
        found_nz = found_nz + is_active
        g_idx = g_idx + 1

    if tl.sum(found_nz) == 0:
        tl.store(tile_class_ptr + tile_id + off1, tl.zeros([1], dtype=tl.int32))
        tl.store(ag_mask_ptr + tile_id + off1, tl.zeros([1], dtype=tl.int32))
        return

    while tl.sum(g_idx) < NUM_GROUPS:
        g_val = tl.sum(g_idx)
        group_max = tl.zeros([1], dtype=tl.float32)
        for c_off in range(1, GROUP_SIZE_C):
            c = g_val * GROUP_SIZE_C + c_off
            if c < C_IN:
                vals = tl.load(x_ptr + row_offs * C_IN + c, mask=row_mask, other=0.0)
                group_max = tl.maximum(group_max, tl.max(tl.abs(vals), axis=0))
        is_active = (group_max > THRESHOLD).to(tl.int32)
        mask = mask + is_active * (1 << g_val)
        if tl.sum(mask == ALL_ONES_MASK) > 0:
            g_idx = tl.full([1], NUM_GROUPS, dtype=tl.int32)
        else:
            g_idx = g_idx + 1

    tl.store(ag_mask_ptr + tile_id + off1, mask)
    if tl.sum(mask == ALL_ONES_MASK) > 0:
        tl.store(tile_class_ptr + tile_id + off1, tl.full([1], TILE_DENSEISH, dtype=tl.int32))
    else:
        tl.store(tile_class_ptr + tile_id + off1, tl.full([1], TILE_SPARSE, dtype=tl.int32))


# ---------------------------------------------------------------------------
# Stage 2b: exact bitmask for uncertain tiles
# ---------------------------------------------------------------------------

@triton.jit
def linear_group_bitmask_refine_kernel(
    x_ptr, tile_class_ptr, ag_mask_ptr,
    N_val, C_IN,
    BLOCK_M: tl.constexpr,
    THRESHOLD: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    ALL_ONES_MASK: tl.constexpr,
    UNCERTAIN_CLASS: tl.constexpr,
):
    tile_id = tl.program_id(0)
    row_start = tile_id * BLOCK_M
    if row_start >= N_val:
        return

    off1 = tl.arange(0, 1)
    tc = tl.load(tile_class_ptr + tile_id + off1)
    if tl.sum(tc) != UNCERTAIN_CLASS:
        return

    row_offs = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offs < N_val

    mask = tl.zeros([1], dtype=tl.int32)
    g_idx = tl.zeros([1], dtype=tl.int32)

    while tl.sum(g_idx) < NUM_GROUPS:
        g_val = tl.sum(g_idx)
        group_max = tl.zeros([1], dtype=tl.float32)
        for c_off in range(GROUP_SIZE_C):
            c = g_val * GROUP_SIZE_C + c_off
            if c < C_IN:
                vals = tl.load(x_ptr + row_offs * C_IN + c, mask=row_mask, other=0.0)
                group_max = tl.maximum(group_max, tl.max(tl.abs(vals), axis=0))
        mask = mask + (group_max > THRESHOLD).to(tl.int32) * (1 << g_val)
        if tl.sum(mask == ALL_ONES_MASK) > 0:
            g_idx = tl.full([1], NUM_GROUPS, dtype=tl.int32)
        else:
            g_idx = g_idx + 1

    tl.store(ag_mask_ptr + tile_id + off1, mask)
    if tl.sum(mask) == 0:
        tl.store(tile_class_ptr + tile_id + off1, tl.zeros([1], dtype=tl.int32))
    else:
        out_cls = tl.full([1], TILE_SPARSE, dtype=tl.int32) + (mask == ALL_ONES_MASK).to(tl.int32)
        tl.store(tile_class_ptr + tile_id + off1, out_cls)


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------

@torch.no_grad()
def _build_linear_metadata(
    x_f16: torch.Tensor,
    N: int,
    C_IN: int,
    BLOCK_M: int,
    N_TILES: int,
    threshold: float,
    ag_mask_buf: torch.Tensor,
    tile_class_buf: torch.Tensor,
    prescan_stats=None,
):
    GROUP_SIZE_C = choose_group_size(C_IN)
    NUM_GROUPS = triton.cdiv(C_IN, GROUP_SIZE_C)
    ALL_ONES_MASK = (1 << NUM_GROUPS) - 1

    linear_tile_coarse_classify_kernel[(N_TILES,)](
        x_f16, tile_class_buf, ag_mask_buf,
        N, C_IN,
        BLOCK_M=BLOCK_M,
        THRESHOLD=threshold,
        GROUP_SIZE_C=GROUP_SIZE_C,
        NUM_GROUPS=NUM_GROUPS,
        ALL_ONES_MASK=ALL_ONES_MASK,
        UNCERTAIN_CLASS=TILE_UNCERTAIN,
        ZERO_CANDIDATE_CLASS=TILE_ZERO_CANDIDATE,
    )

    if prescan_stats is not None:
        tc = tile_class_buf[:N_TILES]
        prescan_stats['stage1_zero_candidate'] = int((tc == TILE_ZERO_CANDIDATE).sum().item())
        prescan_stats['stage1_denseish'] = int((tc == TILE_DENSEISH).sum().item())
        prescan_stats['stage1_uncertain'] = int((tc == TILE_UNCERTAIN).sum().item())

    linear_zero_candidate_refine_kernel[(N_TILES,)](
        x_f16, tile_class_buf, ag_mask_buf,
        N, C_IN,
        BLOCK_M=BLOCK_M,
        THRESHOLD=threshold,
        GROUP_SIZE_C=GROUP_SIZE_C,
        NUM_GROUPS=NUM_GROUPS,
        ALL_ONES_MASK=ALL_ONES_MASK,
        ZERO_CANDIDATE_CLASS=TILE_ZERO_CANDIDATE,
    )

    linear_group_bitmask_refine_kernel[(N_TILES,)](
        x_f16, tile_class_buf, ag_mask_buf,
        N, C_IN,
        BLOCK_M=BLOCK_M,
        THRESHOLD=threshold,
        GROUP_SIZE_C=GROUP_SIZE_C,
        NUM_GROUPS=NUM_GROUPS,
        ALL_ONES_MASK=ALL_ONES_MASK,
        UNCERTAIN_CLASS=TILE_UNCERTAIN,
    )

    return GROUP_SIZE_C, NUM_GROUPS


# ---------------------------------------------------------------------------
# Compute kernel: zero / denseish / sparse
# ---------------------------------------------------------------------------

_LINEAR_CONFIGS = [
    # Conservative set to avoid OOR on large C_IN/C_OUT (e.g., VGG classifier).
    Config({'BLOCK_N': 32, 'DENSE_K': 32}, num_warps=4, num_stages=1),
    Config({'BLOCK_N': 64, 'DENSE_K': 32}, num_warps=4, num_stages=1),
    Config({'BLOCK_N': 64, 'DENSE_K': 64}, num_warps=4, num_stages=1),
]


@autotune(configs=_LINEAR_CONFIGS, key=['C_IN', 'C_OUT', 'N_TILES_KEY', 'BLOCK_M_KEY', 'GROUP_SIZE_C_KEY'])
@triton.jit
def sparse_linear_grouped_kernel(
    x_ptr,
    w_t_ptr,
    tile_class_ptr,
    ag_mask_ptr,
    tile_ids_ptr,
    y_ptr,
    N_val,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    N_TILES_KEY: tl.constexpr,
    BLOCK_M_KEY: tl.constexpr,
    GROUP_SIZE_C_KEY: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DENSE_K: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    USE_TILE_IDS: tl.constexpr,
):
    pid_tile = tl.program_id(0)
    tile_id = tl.load(tile_ids_ptr + pid_tile) if USE_TILE_IDS else pid_tile
    pid_cout = tl.program_id(1)

    if tile_id >= N_TILES_KEY:
        return

    row_start = tile_id * BLOCK_M
    cout_start = pid_cout * BLOCK_N

    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_n = cout_start + tl.arange(0, BLOCK_N)
    m_mask = offs_m < N_val
    n_mask = offs_n < C_OUT

    off1 = tl.arange(0, 1)
    tc = tl.load(tile_class_ptr + tile_id + off1)
    tile_cls = tl.sum(tc)
    if tile_cls == TILE_ZERO:
        return

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    if tile_cls == TILE_DENSEISH:
        for cin_base in range(0, C_IN, DENSE_K):
            offs_k = cin_base + tl.arange(0, DENSE_K)
            k_mask = offs_k < C_IN

            x_addrs = offs_m[:, None] * C_IN + offs_k[None, :]
            x_mask = m_mask[:, None] & k_mask[None, :]
            x_tile = tl.load(x_ptr + x_addrs, mask=x_mask, other=0.0).to(tl.float16)

            w_addrs = offs_k[:, None] * C_OUT + offs_n[None, :]
            w_mask = k_mask[:, None] & n_mask[None, :]
            w_tile = tl.load(w_t_ptr + w_addrs, mask=w_mask, other=0.0).to(tl.float16)

            acc += tl.dot(x_tile, w_tile)

    elif tile_cls == TILE_SPARSE:
        ag = tl.load(ag_mask_ptr + tile_id + off1)
        for g in range(NUM_GROUPS):
            g_active = (ag >> g) & 1
            if tl.sum(g_active) != 0:
                # Chunk active group by DENSE_K to avoid large temporary tiles
                # like [GROUP_SIZE_C, BLOCK_N], which can exceed shared memory.
                g_base = g * GROUP_SIZE_C
                for k_off in range(0, GROUP_SIZE_C, DENSE_K):
                    offs_k = g_base + k_off + tl.arange(0, DENSE_K)
                    k_mask = offs_k < C_IN

                    x_addrs = offs_m[:, None] * C_IN + offs_k[None, :]
                    x_mask = m_mask[:, None] & k_mask[None, :]
                    x_tile = tl.load(x_ptr + x_addrs, mask=x_mask, other=0.0).to(tl.float16)

                    w_addrs = offs_k[:, None] * C_OUT + offs_n[None, :]
                    w_mask = k_mask[:, None] & n_mask[None, :]
                    w_tile = tl.load(w_t_ptr + w_addrs, mask=w_mask, other=0.0).to(tl.float16)

                    acc += tl.dot(x_tile, w_tile)

    out_addrs = offs_m[:, None] * C_OUT + offs_n[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    out_old = tl.load(y_ptr + out_addrs, mask=out_mask, other=0.0)
    tl.store(y_ptr + out_addrs, out_old + acc, mask=out_mask)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def sparse_linear_forward(
    x,
    weight,
    bias=None,
    threshold=PRESCAN_ACTIVITY_EPS,
    w_t=None,
    ag_mask_buf=None,
    tile_class_buf=None,
    return_ms=False,
    fallback_ratio=FALLBACK_RATIO,
    return_avg_active_ratio=False,
    return_tile_stats=False,
    return_backend_meta=False,
    active_tile_ids_buf=None,
    launch_all_tiles=False,
):
    import torch.nn.functional as Fn

    N, C_IN = x.shape
    C_OUT = weight.shape[0]
    device = x.device

    need_stats = return_avg_active_ratio or return_tile_stats

    def _finalize_return(
        y,
        ms,
        avg_active_ratio_val=None,
        tile_stats_val=None,
        backend_meta_val=None,
    ):
        ret = (y, ms)
        if return_avg_active_ratio:
            ret = ret + (avg_active_ratio_val,)
        if return_tile_stats:
            ret = ret + (tile_stats_val,)
        if return_backend_meta:
            ret = ret + (backend_meta_val,)
        return ret

    def _dense_fallback(
        reason="dense_fallback",
        avg_active_ratio_val=1.0,
        tile_stats_val=None,
        backend_meta_extra=None,
    ):
        dense_ms = 0.0
        if return_ms:
            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            se.record()

        y = Fn.linear(
            x.float(),
            weight.float(),
            bias.float() if bias is not None else None,
        ).float()

        if return_ms:
            ee.record()
            torch.cuda.synchronize(device)
            dense_ms = se.elapsed_time(ee)

        bm = {"backend": "dense_fallback", "reason": reason}
        if backend_meta_extra:
            bm.update(backend_meta_extra)
        return _finalize_return(y, dense_ms, avg_active_ratio_val, tile_stats_val, bm)

    def _zero_tiles_output(reason, tile_stats_val=None, backend_meta_extra=None):
        y = torch.zeros(N, C_OUT, dtype=torch.float32, device=device)
        if bias is not None:
            y = y + bias.detach().float().view(1, -1)
        bm = {"backend": "zero_tiles_only", "reason": reason}
        if backend_meta_extra:
            bm.update(backend_meta_extra)
        return _finalize_return(y, 0.0, 0.0, tile_stats_val, bm)

    # shape / support guards
    if x.dim() != 2:
        return _dense_fallback(reason="expected_2d_input")

    if C_IN <= 0 or C_OUT <= 0 or N <= 0:
        y = torch.zeros(max(N, 0), max(C_OUT, 0), dtype=torch.float32, device=device)
        bm = {"backend": "zero_tiles_only", "reason": "empty_shape", "total_tiles": 0}
        return _finalize_return(y, 0.0, 0.0, None, bm)

    BLOCK_M = _select_linear_block_m(N)
    N_TILES = triton.cdiv(N, BLOCK_M)

    x_f16 = x if (x.dtype == torch.float16 and x.is_contiguous()) else x.half().contiguous()
    w_t_f16 = w_t if w_t is not None else weight.half().t().contiguous()

    if ag_mask_buf is None or ag_mask_buf.numel() < N_TILES:
        ag_mask_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
    if tile_class_buf is None or tile_class_buf.numel() < N_TILES:
        tile_class_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)

    prescan_stats = {} if return_tile_stats else None
    group_size_c, num_groups = _build_linear_metadata(
        x_f16,
        N,
        C_IN,
        BLOCK_M,
        N_TILES,
        threshold,
        ag_mask_buf,
        tile_class_buf,
        prescan_stats=prescan_stats,
    )

    avg_active_ratio = None
    tile_stats = None
    active_tiles_for_meta = None

    if need_stats:
        tc = tile_class_buf[:N_TILES]
        zc = int((tc == TILE_ZERO).sum().item())
        sc = int((tc == TILE_SPARSE).sum().item())
        dc = int((tc == TILE_DENSEISH).sum().item())
        total_nonzero = sc + dc
        denseish_ratio = float(dc) / max(float(total_nonzero), 1.0)
        active_tiles_for_meta = int(total_nonzero)

        if num_groups > 0:
            pc = popcount_buf(ag_mask_buf, N_TILES)
            avg_active_ratio = float(pc.sum().item()) / max(float(N_TILES * num_groups), 1.0)
        else:
            avg_active_ratio = 1.0

        if return_tile_stats:
            tile_stats = {
                "zero_tiles": zc,
                "sparse_tiles": sc,
                "denseish_tiles": dc,
                "total_tiles": int(N_TILES),
                "prescan_mode": "three_stage_grouped_linear_v4",
                "active_tiles": int(total_nonzero),
                "active_tile_ratio": float(total_nonzero) / max(float(N_TILES), 1.0),
                "denseish_ratio_nonzero": denseish_ratio,
                "avg_active_group_ratio": avg_active_ratio,
                "block_m": int(BLOCK_M),
                "group_size_c": int(group_size_c),
                "num_groups": int(num_groups),
            }
            if prescan_stats:
                tile_stats.update(prescan_stats)

        if _check_dense_fallback(ag_mask_buf, N_TILES, num_groups, fallback_ratio=fallback_ratio):
            return _dense_fallback(
                reason="post_metadata_dense_fallback",
                avg_active_ratio_val=avg_active_ratio,
                tile_stats_val=tile_stats,
                backend_meta_extra={
                    "active_tiles": total_nonzero,
                    "total_tiles": int(N_TILES),
                    "denseish_ratio_nonzero": denseish_ratio,
                },
            )

        if launch_all_tiles and total_nonzero == 0:
            return _zero_tiles_output(
                reason="all_tiles_zero_after_metadata",
                tile_stats_val=tile_stats,
                backend_meta_extra={"active_tiles": 0, "total_tiles": int(N_TILES)},
            )

    # Safety guard against Triton per-tensor numel limit in kernel temporaries.
    # The sparse path constructs tensors with shape [GROUP_SIZE_C, BLOCK_N].
    # Require:
    #   GROUP_SIZE_C * max(BLOCK_N) <= TRITON_MAX_TENSOR_NUMEL
    max_block_n = max(cfg.kwargs["BLOCK_N"] for cfg in _LINEAR_CONFIGS)
    if group_size_c * max_block_n > TRITON_MAX_TENSOR_NUMEL:
        return _dense_fallback(
            reason=f"group_size_too_large_for_sparse_kernel(gs={group_size_c}, max_bn={max_block_n})",
            avg_active_ratio_val=avg_active_ratio if avg_active_ratio is not None else 1.0,
            tile_stats_val=tile_stats,
            backend_meta_extra={
                "group_size_c": int(group_size_c),
                "num_groups": int(num_groups),
                "total_tiles": int(N_TILES),
            },
        )

    # Guard against shared-memory heavy configs. Triton autotune can still pick
    # configs that exceed HW SMEM budget for large GEMM tiles on some GPUs.
    # Use a conservative estimate and fallback to dense before kernel launch.
    max_dense_k = max(cfg.kwargs["DENSE_K"] for cfg in _LINEAR_CONFIGS)
    est_smem_bytes = (
        (BLOCK_M * max_dense_k) + (max_dense_k * max_block_n) + (BLOCK_M * max_block_n)
    ) * 4
    if est_smem_bytes > 160_000:
        return _dense_fallback(
            reason=(
                "estimated_shared_memory_too_large_for_sparse_kernel"
                f"(block_m={BLOCK_M}, max_bn={max_block_n}, max_dk={max_dense_k}, smem={est_smem_bytes})"
            ),
            avg_active_ratio_val=avg_active_ratio if avg_active_ratio is not None else 1.0,
            tile_stats_val=tile_stats,
            backend_meta_extra={
                "group_size_c": int(group_size_c),
                "num_groups": int(num_groups),
                "total_tiles": int(N_TILES),
                "estimated_shared_mem_bytes": int(est_smem_bytes),
            },
        )

    if launch_all_tiles:
        launch_count = N_TILES
        use_tile_ids = False
        tile_ids_ptr = ag_mask_buf  # placeholder, ignored in all_tiles mode
    else:
        active_tile_ids, active_tile_count = _build_active_tile_ids(tile_class_buf, N_TILES)
        if active_tile_count == 0:
            return _zero_tiles_output(
                reason="all_tiles_zero_after_metadata",
                tile_stats_val=tile_stats,
                backend_meta_extra={"active_tiles": 0, "total_tiles": int(N_TILES)},
            )
        if active_tile_ids_buf is not None and active_tile_ids_buf.numel() >= active_tile_count:
            active_tile_ids_buf[:active_tile_count].copy_(active_tile_ids)
            tile_ids_ptr = active_tile_ids_buf[:active_tile_count]
        else:
            tile_ids_ptr = active_tile_ids
        launch_count = active_tile_count
        use_tile_ids = True
        if active_tiles_for_meta is None:
            active_tiles_for_meta = int(active_tile_count)

    y = torch.zeros(N, C_OUT, dtype=torch.float32, device=device)
    if bias is not None:
        y = y + bias.detach().float().view(1, -1)

    sparse_ms = 0.0
    if return_ms:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()

    def _grid(META):
        return (launch_count, triton.cdiv(C_OUT, META["BLOCK_N"]))

    sparse_linear_grouped_kernel[_grid](
        x_f16,
        w_t_f16,
        tile_class_buf,
        ag_mask_buf,
        tile_ids_ptr,
        y,
        N,
        C_IN,
        C_OUT,
        N_TILES,
        BLOCK_M,
        group_size_c,
        BLOCK_M=BLOCK_M,
        GROUP_SIZE_C=group_size_c,
        NUM_GROUPS=num_groups,
        USE_TILE_IDS=use_tile_ids,
    )

    if return_ms:
        end_evt.record()
        torch.cuda.synchronize(device)
        sparse_ms = start_evt.elapsed_time(end_evt)

    backend_meta = {
        "backend": "sparse_triton",
        "reason": "linear_unified_v1",
        "total_tiles": int(N_TILES),
        "launch_count": int(launch_count),
        "launch_mode": "all_tiles" if launch_all_tiles else "active_only",
    }
    if active_tiles_for_meta is not None:
        backend_meta["active_tiles"] = int(active_tiles_for_meta)
    if avg_active_ratio is not None:
        backend_meta["avg_active_group_ratio"] = float(avg_active_ratio)

    return _finalize_return(y, sparse_ms, avg_active_ratio, tile_stats, backend_meta)