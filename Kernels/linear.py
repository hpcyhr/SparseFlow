"""
SparseFlow Sparse Linear Triton Kernels — Tile-level Dynamic-K (Tile-level Dynamic-K)

Architecture mirrors conv2d.py:
  Stage-1: Per-tile channel prescan (count → cumsum → write, zero sync)
  Stage-2: Autotuned sparse GEMM with while-loop Dynamic-K

Key difference from Kernels/linear.py (v1, row-level prescan):
  - v1 skips entire zero ROWS → coarse-grained
  - Groups rows into tiles, skips zero CHANNELS per tile → fine-grained
  - Matches conv1x1_pertile_kernel pattern exactly:
      M = batch rows (BLOCK_M)
      N = C_OUT
      K = active C_IN channels (Dynamic-K via prescan)

Weight layout: W_T [C_IN, C_OUT] ("channel-last") for coalesced BLOCK_N access.
"""

import torch
import triton
import triton.language as tl
from triton import autotune, Config


# ═══════════════════════════════════════════════════════════════════════
# Stage-1 Step 1: Count active channels per batch tile
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def linear_prescan_count_kernel(
    x_ptr,              # [N, C_IN] fp16
    counts_ptr,         # [N_TILES] int32
    N_val,
    C_IN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MAX_C: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
    """
    For each batch tile (BLOCK_M rows), count how many C_IN channels
    have at least one non-zero value across the tile.
    Grid: (N_TILES,)
    """
    tile_id = tl.program_id(0)
    row_start = tile_id * BLOCK_M

    row_offs = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offs < N_val

    count = 0

    for c_idx in range(MAX_C):
        if c_idx < C_IN:
            # Load x[row_start:row_start+BLOCK_M, c_idx]
            vals = tl.load(
                x_ptr + row_offs * C_IN + c_idx,
                mask=row_mask, other=0.0)
            is_nz = tl.max(tl.abs(vals)) > THRESHOLD
            count += is_nz.to(tl.int32)

    tl.store(counts_ptr + tile_id, count)


# ═══════════════════════════════════════════════════════════════════════
# Stage-1 Step 2: Write active channel indices per batch tile
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def linear_prescan_write_kernel(
    x_ptr,              # [N, C_IN] fp16
    tile_ptr_data,      # [N_TILES + 1] int32  (CSR offsets)
    tile_cin_ptr,       # [max_entries] int32   (active channel indices)
    cin_buf_size,
    N_val,
    C_IN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MAX_C: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
    """
    For each batch tile, write the indices of active channels into
    tile_cin_buf at the CSR offset position.
    Grid: (N_TILES,)
    """
    tile_id = tl.program_id(0)
    row_start = tile_id * BLOCK_M

    row_offs = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offs < N_val

    write_pos = tl.load(tile_ptr_data + tile_id)
    idx = 0

    for c_idx in range(MAX_C):
        if c_idx < C_IN:
            vals = tl.load(
                x_ptr + row_offs * C_IN + c_idx,
                mask=row_mask, other=0.0)
            is_nz = tl.max(tl.abs(vals)) > THRESHOLD

            if is_nz:
                out_pos = write_pos + idx
                if out_pos < cin_buf_size:
                    tl.store(tile_cin_ptr + out_pos, c_idx)
                idx += 1


# ═══════════════════════════════════════════════════════════════════════
# Stage-1 Python orchestration: count → cumsum → write (zero sync)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _build_linear_tile_csr(x_f16, N, C_IN, BLOCK_M, N_TILES, threshold,
                           counts_buf, tile_cin_buf):
    """
    Build per-tile CSR structure for Linear. Zero host-device sync.

    Returns:
        tile_ptr: [N_TILES + 1] int32
    """
    device = x_f16.device
    MAX_C = triton.next_power_of_2(max(C_IN, 1))

    tile_counts = counts_buf[:N_TILES]

    linear_prescan_count_kernel[(N_TILES,)](
        x_f16, tile_counts,
        N, C_IN=C_IN, BLOCK_M=BLOCK_M,
        MAX_C=MAX_C, THRESHOLD=threshold,
    )

    cumsum = torch.cumsum(tile_counts, dim=0, dtype=torch.int32)
    tile_ptr = torch.empty(N_TILES + 1, dtype=torch.int32, device=device)
    tile_ptr[0] = 0
    tile_ptr[1:] = cumsum

    cin_buf_size = tile_cin_buf.numel()

    linear_prescan_write_kernel[(N_TILES,)](
        x_f16, tile_ptr, tile_cin_buf,
        cin_buf_size,
        N, C_IN=C_IN, BLOCK_M=BLOCK_M,
        MAX_C=MAX_C, THRESHOLD=threshold,
    )

    return tile_ptr


# ═══════════════════════════════════════════════════════════════════════
# Autotune configuration pool
# ═══════════════════════════════════════════════════════════════════════

_LINEAR_CONFIGS = [
    # BLOCK_M: batch tile, BLOCK_N: output channels, BLOCK_K: active input channels
    Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=1),
    Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=1),
    Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=1),
    Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=1),
    Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=1),
    Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=8, num_stages=1),
    Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=1),
    Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=1),
    Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=1),
    Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=8, num_stages=1),
    Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=1),
    Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=1),
]


# ═══════════════════════════════════════════════════════════════════════
# Stage-2: Autotuned Sparse Linear with Dynamic-K while loop
# ═══════════════════════════════════════════════════════════════════════

@autotune(configs=_LINEAR_CONFIGS, key=['C_IN', 'C_OUT', 'N_TILES_KEY'])
@triton.jit
def sparse_linear_pertile_kernel(
    x_ptr,              # [N, C_IN] fp16
    w_t_ptr,            # [C_IN, C_OUT] fp16  (transposed for coalesced access)
    bias_ptr,           # [C_OUT] fp32 or dummy
    tile_ptr_data,      # [N_TILES + 1] int32
    tile_cin_ptr,       # [max_entries] int32
    y_ptr,              # [N, C_OUT] fp32
    N_val,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    N_TILES_KEY: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Per-tile sparse Linear — autotune + while-loop Dynamic-K.

    Each program handles one batch tile × one C_OUT chunk.
    Only active C_IN channels (from prescan CSR) participate in the matmul.
    """
    tile_id = tl.program_id(0)
    pid_cout = tl.program_id(1)

    if tile_id >= N_TILES_KEY:
        return

    row_start = tile_id * BLOCK_M
    cout_start = pid_cout * BLOCK_N

    # Row offsets within this batch tile
    offs_m = row_start + tl.arange(0, BLOCK_M)
    m_mask = offs_m < N_val

    # Output channel offsets
    offs_n = cout_start + tl.arange(0, BLOCK_N)
    n_mask = offs_n < C_OUT

    # CSR bounds for this tile
    tile_start = tl.load(tile_ptr_data + tile_id)
    tile_end = tl.load(tile_ptr_data + tile_id + 1)
    active_K = tile_end - tile_start

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # ── Dynamic-K while loop ──
    k_start = 0
    while k_start < active_K:
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < active_K

        # Gather active channel indices
        cin_global = tl.load(
            tile_cin_ptr + tile_start + offs_k,
            mask=k_mask, other=0)

        # Load x tile: [BLOCK_M, BLOCK_K]
        # x[row, cin_global] for each row in tile
        x_addrs = offs_m[:, None] * C_IN + cin_global[None, :]
        x_load_mask = m_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptr + x_addrs, mask=x_load_mask, other=0.0)
        x_tile = x_tile.to(tl.float16)

        # Load w_t tile: [BLOCK_K, BLOCK_N]
        # w_t[cin_global, cout] — transposed layout for coalesced N access
        w_addrs = cin_global[:, None] * C_OUT + offs_n[None, :]
        w_load_mask = k_mask[:, None] & n_mask[None, :]
        w_tile = tl.load(w_t_ptr + w_addrs, mask=w_load_mask, other=0.0)
        w_tile = w_tile.to(tl.float16)

        acc += tl.dot(x_tile, w_tile)
        k_start += BLOCK_K

    # ── Bias ──
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        acc += bias_vals[None, :]

    # ── Store output ──
    out_addrs = offs_m[:, None] * C_OUT + offs_n[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptr + out_addrs, acc, mask=out_mask)


# ═══════════════════════════════════════════════════════════════════════
# Batch tile size selection
# ═══════════════════════════════════════════════════════════════════════

def _select_linear_block_m(N):
    """Select batch tile size based on total batch count."""
    if N >= 512:
        return 128
    elif N >= 128:
        return 64
    else:
        return 32


# ═══════════════════════════════════════════════════════════════════════
# Main entry: sparse_linear_forward
# ═══════════════════════════════════════════════════════════════════════

def sparse_linear_forward(x, weight, bias=None, threshold=1e-6,
                             w_t=None, counts_buf=None, tile_cin_buf=None,
                             return_ms=False):
    """
    Tile-level Dynamic-K sparse Linear forward.

    Pipeline:
      1. prescan_count → cumsum → prescan_write  (all GPU, zero sync)
      2. autotuned sparse GEMM with while-loop Dynamic-K

    Args:
        x: [N, C_IN] input (fp16 or fp32, auto-converted)
        weight: [C_OUT, C_IN] original Linear weight
        bias: [C_OUT] or None
        w_t: [C_IN, C_OUT] pre-transposed weight (fp16), cached by module
        counts_buf, tile_cin_buf: pre-allocated buffers
        return_ms: True → CUDA event timing for Stage-2

    Returns:
        y: [N, C_OUT] fp32
        sparse_ms: float
    """
    N, C_IN = x.shape
    C_OUT = weight.shape[0]
    device = x.device

    # ── Tile size ──
    BLOCK_M = _select_linear_block_m(N)
    N_TILES = triton.cdiv(N, BLOCK_M)

    x_f16 = x.half().contiguous()

    # ── Weight transposed layout: [C_IN, C_OUT] for coalesced access ──
    if w_t is not None:
        w_t_f16 = w_t
    else:
        w_t_f16 = weight.half().t().contiguous()  # [C_OUT, C_IN]^T = [C_IN, C_OUT]

    # ── Buffer preparation ──
    if counts_buf is None or counts_buf.numel() < N_TILES:
        counts_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
    if tile_cin_buf is None or tile_cin_buf.numel() < N_TILES * C_IN:
        tile_cin_buf = torch.empty(N_TILES * C_IN, dtype=torch.int32, device=device)

    # ── Stage-1: Per-tile CSR (zero sync) ──
    tile_ptr = _build_linear_tile_csr(
        x_f16, N, C_IN, BLOCK_M, N_TILES, threshold,
        counts_buf, tile_cin_buf)

    # ── Stage-2: Autotuned sparse Linear ──
    has_bias = bias is not None
    bias_f32 = (bias.float().contiguous() if has_bias
                else torch.empty(1, device=device))
    y = torch.zeros(N, C_OUT, dtype=torch.float32, device=device)

    sparse_ms = 0.0
    if return_ms:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()

    def _grid(META):
        return (N_TILES, triton.cdiv(C_OUT, META['BLOCK_N']))

    sparse_linear_pertile_kernel[_grid](
        x_f16, w_t_f16, bias_f32,
        tile_ptr, tile_cin_buf,
        y,
        N,
        C_IN, C_OUT, N_TILES,
        HAS_BIAS=has_bias,
    )

    if return_ms:
        end_evt.record()
        torch.cuda.synchronize(device)
        sparse_ms = start_evt.elapsed_time(end_evt)

    return y, sparse_ms