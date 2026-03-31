import torch
import triton
import triton.language as tl
from triton import autotune, Config


TILE_ZERO = 0
TILE_SPARSE = 1
TILE_DENSEISH = 2
TILE_UNCERTAIN = 3
TILE_ZERO_CANDIDATE = 4
TRITON_MAX_TENSOR_NUMEL = 131072


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def choose_group_size(c_in: int) -> int:
    if c_in <= 128:
        gs = 16
    else:
        gs = 32
    num_groups = (c_in + gs - 1) // gs
    while num_groups > 32:
        gs *= 2
        num_groups = (c_in + gs - 1) // gs
    return gs


def _select_linear_block_m(n_rows: int) -> int:
    if n_rows >= 1024:
        return 128
    if n_rows >= 256:
        return 64
    return 32


def _popcount_buf(ag_mask_buf: torch.Tensor, n_tiles: int) -> torch.Tensor:
    v = ag_mask_buf[:n_tiles].int()
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F
    v = v + (v >> 8)
    v = v + (v >> 16)
    return (v & 0x3F).to(torch.int32)


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
    Config({'BLOCK_N': 64,  'DENSE_K': 32},  num_warps=4, num_stages=1),
    Config({'BLOCK_N': 64,  'DENSE_K': 64},  num_warps=4, num_stages=2),
    Config({'BLOCK_N': 128, 'DENSE_K': 32},  num_warps=4, num_stages=1),
    Config({'BLOCK_N': 128, 'DENSE_K': 64},  num_warps=8, num_stages=2),
    Config({'BLOCK_N': 128, 'DENSE_K': 128}, num_warps=8, num_stages=2),
]


@autotune(configs=_LINEAR_CONFIGS, key=['C_IN', 'C_OUT', 'N_TILES_KEY', 'BLOCK_M_KEY', 'GROUP_SIZE_C_KEY'])
@triton.jit
def sparse_linear_grouped_kernel(
    x_ptr,
    w_t_ptr,
    bias_ptr,
    tile_class_ptr,
    ag_mask_ptr,
    y_ptr,
    N_val,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    N_TILES_KEY: tl.constexpr,
    BLOCK_M_KEY: tl.constexpr,
    GROUP_SIZE_C_KEY: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DENSE_K: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    tile_id = tl.program_id(0)
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

    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        acc += bias_vals[None, :]

    out_addrs = offs_m[:, None] * C_OUT + offs_n[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptr + out_addrs, acc, mask=out_mask)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def sparse_linear_forward(
    x,
    weight,
    bias=None,
    threshold=1e-6,
    w_t=None,
    ag_mask_buf=None,
    tile_class_buf=None,
    return_ms=False,
    return_avg_active_ratio=False,
    return_tile_stats=False,
    return_backend_meta=False,
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

    # shape / support guards
    if x.dim() != 2:
        return _dense_fallback(reason="expected_2d_input")

    if C_IN <= 0 or C_OUT <= 0 or N <= 0:
        y = torch.zeros(max(N, 0), max(C_OUT, 0), dtype=torch.float32, device=device)
        bm = {"backend": "zero_tiles_only", "reason": "empty_shape"}
        return _finalize_return(y, 0.0, 0.0, None, bm)

    BLOCK_M = _select_linear_block_m(N)
    N_TILES = triton.cdiv(N, BLOCK_M)

    x_f16 = x if (x.dtype == torch.float16 and x.is_contiguous()) else x.half().contiguous()
    w_t_f16 = w_t if w_t is not None else weight.half().t().contiguous()

    if ag_mask_buf is None or ag_mask_buf.numel() < N_TILES:
        ag_mask_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
    if tile_class_buf is None or tile_class_buf.numel() < N_TILES:
        tile_class_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)

    tile_stats = {} if return_tile_stats else None
    group_size_c, num_groups = _build_linear_metadata(
        x_f16,
        N,
        C_IN,
        BLOCK_M,
        N_TILES,
        threshold,
        ag_mask_buf,
        tile_class_buf,
        prescan_stats=tile_stats,
    )

    # Safety guard against Triton per-tensor numel limit in kernel temporaries.
    # The sparse path constructs tensors with shape [GROUP_SIZE_C, BLOCK_N].
    # With current configs BLOCK_N can be up to 128, so require:
    #   GROUP_SIZE_C * 128 <= TRITON_MAX_TENSOR_NUMEL
    max_block_n = max(cfg.kwargs["BLOCK_N"] for cfg in _LINEAR_CONFIGS)
    if group_size_c * max_block_n > TRITON_MAX_TENSOR_NUMEL:
        return _dense_fallback(
            reason=f"group_size_too_large_for_sparse_kernel(gs={group_size_c}, max_bn={max_block_n})",
            avg_active_ratio_val=1.0,
            tile_stats_val=tile_stats,
            backend_meta_extra={
                "group_size_c": int(group_size_c),
                "num_groups": int(num_groups),
                "total_tiles": int(N_TILES),
            },
        )

    has_bias = bias is not None
    bias_f32 = bias.float().contiguous() if has_bias else torch.empty(1, dtype=torch.float32, device=device)
    y = torch.zeros(N, C_OUT, dtype=torch.float32, device=device)

    sparse_ms = 0.0
    if return_ms:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()

    def _grid(META):
        return (N_TILES, triton.cdiv(C_OUT, META["BLOCK_N"]))

    sparse_linear_grouped_kernel[_grid](
        x_f16,
        w_t_f16,
        bias_f32,
        tile_class_buf,
        ag_mask_buf,
        y,
        N,
        C_IN,
        C_OUT,
        N_TILES,
        BLOCK_M,
        group_size_c,
        HAS_BIAS=has_bias,
        BLOCK_M=BLOCK_M,
        GROUP_SIZE_C=group_size_c,
        NUM_GROUPS=num_groups,
    )

    if return_ms:
        end_evt.record()
        torch.cuda.synchronize(device)
        sparse_ms = start_evt.elapsed_time(end_evt)

    avg_active_ratio = None
    if return_avg_active_ratio:
        pc = _popcount_buf(ag_mask_buf, N_TILES)
        avg_active_ratio = float(pc.float().mean().item()) / max(float(num_groups), 1.0)

    backend_meta = {
        "backend": "sparse_triton",
        "reason": "three_stage_grouped_linear",
        "total_tiles": int(N_TILES),
    }

    if return_tile_stats:
        tc = tile_class_buf[:N_TILES]
        tile_stats["final_zero"] = int((tc == TILE_ZERO).sum().item())
        tile_stats["final_sparse"] = int((tc == TILE_SPARSE).sum().item())
        tile_stats["final_denseish"] = int((tc == TILE_DENSEISH).sum().item())
        tile_stats["zero_tiles"] = tile_stats["final_zero"]
        tile_stats["sparse_tiles"] = tile_stats["final_sparse"]
        tile_stats["denseish_tiles"] = tile_stats["final_denseish"]
        tile_stats["stage2_zero_refine_tiles"] = tile_stats.get("stage1_zero_candidate", 0)
        tile_stats["stage2_uncertain_tiles"] = tile_stats.get("stage1_uncertain", 0)

        if tile_stats["final_zero"] == N_TILES:
            backend_meta = {
                "backend": "zero_tiles_only",
                "reason": "all_tiles_zero_after_prescan",
                "total_tiles": int(N_TILES),
            }
        else:
            backend_meta["active_tiles"] = int(N_TILES - tile_stats["final_zero"])

    return _finalize_return(y, sparse_ms, avg_active_ratio, tile_stats, backend_meta)
