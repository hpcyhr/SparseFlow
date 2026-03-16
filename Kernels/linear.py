"""
SparseFlow sparse Linear Triton kernels — conv-style operator backend.

Design goals:
  - Keep the kernel math specialized for Linear/GEMM.
  - Keep the Python/Kernels interface stable and simple.
  - Match SparseConv2d style: explicit buffer reuse, optional timing, folded 2D execution.

Kernel strategy:
  Stage-1: per-row-tile active-channel prescan (count -> cumsum -> write), all on GPU.
  Stage-2: sparse GEMM over dynamic K (active Cin only), with autotuned tile sizes.

Main entry:
  sparse_linear_forward(
      x,                      # [N, Cin]
      weight,                 # [Cout, Cin]
      bias=None,
      threshold=1e-6,
      w_t=None,               # optional cached [Cin, Cout]
      counts_buf=None,        # optional preallocated int32 buffer
      tile_cin_buf=None,      # optional preallocated int32 buffer
      return_ms=False,
  ) -> (y_fp32, kernel_ms)
"""

import torch
import triton
import triton.language as tl
from triton import autotune, Config


# -----------------------------------------------------------------------------
# Stage-1: tile-level active-channel metadata
# -----------------------------------------------------------------------------

@triton.jit
def linear_prescan_count_kernel(
    x_ptr,              # [N, Cin] fp16
    counts_ptr,         # [N_TILES] int32
    N_val,
    C_IN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MAX_C: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
    tile_id = tl.program_id(0)
    row_start = tile_id * BLOCK_M

    row_offs = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offs < N_val

    count = 0
    for c_idx in range(MAX_C):
        if c_idx < C_IN:
            vals = tl.load(
                x_ptr + row_offs * C_IN + c_idx,
                mask=row_mask,
                other=0.0,
            )
            is_nz = tl.max(tl.abs(vals)) > THRESHOLD
            count += is_nz.to(tl.int32)

    tl.store(counts_ptr + tile_id, count)


@triton.jit
def linear_prescan_write_kernel(
    x_ptr,              # [N, Cin] fp16
    tile_ptr_data,      # [N_TILES + 1] int32
    tile_cin_ptr,       # [max_entries] int32
    cin_buf_size,
    N_val,
    C_IN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MAX_C: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
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
                mask=row_mask,
                other=0.0,
            )
            is_nz = tl.max(tl.abs(vals)) > THRESHOLD
            if is_nz:
                out_pos = write_pos + idx
                if out_pos < cin_buf_size:
                    tl.store(tile_cin_ptr + out_pos, c_idx)
                idx += 1


@torch.no_grad()
def _build_linear_tile_csr(
    x_f16: torch.Tensor,
    N: int,
    C_IN: int,
    BLOCK_M: int,
    N_TILES: int,
    threshold: float,
    counts_buf: torch.Tensor,
    tile_cin_buf: torch.Tensor,
):
    device = x_f16.device
    MAX_C = triton.next_power_of_2(max(C_IN, 1))

    tile_counts = counts_buf[:N_TILES]

    linear_prescan_count_kernel[(N_TILES,)](
        x_f16,
        tile_counts,
        N,
        C_IN=C_IN,
        BLOCK_M=BLOCK_M,
        MAX_C=MAX_C,
        THRESHOLD=threshold,
    )

    cumsum = torch.cumsum(tile_counts, dim=0, dtype=torch.int32)
    tile_ptr = torch.empty(N_TILES + 1, dtype=torch.int32, device=device)
    tile_ptr[0] = 0
    tile_ptr[1:] = cumsum

    cin_buf_size = tile_cin_buf.numel()

    linear_prescan_write_kernel[(N_TILES,)](
        x_f16,
        tile_ptr,
        tile_cin_buf,
        cin_buf_size,
        N,
        C_IN=C_IN,
        BLOCK_M=BLOCK_M,
        MAX_C=MAX_C,
        THRESHOLD=threshold,
    )

    return tile_ptr


# -----------------------------------------------------------------------------
# Stage-2: sparse GEMM
# -----------------------------------------------------------------------------

_LINEAR_CONFIGS = [
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


@autotune(configs=_LINEAR_CONFIGS, key=['C_IN', 'C_OUT', 'N_TILES_KEY'])
@triton.jit
def sparse_linear_pertile_kernel(
    x_ptr,              # [N, Cin] fp16
    w_t_ptr,            # [Cin, Cout] fp16
    bias_ptr,           # [Cout] fp32 or dummy
    tile_ptr_data,      # [N_TILES + 1] int32
    tile_cin_ptr,       # [max_entries] int32
    y_ptr,              # [N, Cout] fp32
    N_val,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    N_TILES_KEY: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_id = tl.program_id(0)
    pid_cout = tl.program_id(1)

    if tile_id >= N_TILES_KEY:
        return

    row_start = tile_id * BLOCK_M
    cout_start = pid_cout * BLOCK_N

    offs_m = row_start + tl.arange(0, BLOCK_M)
    m_mask = offs_m < N_val

    offs_n = cout_start + tl.arange(0, BLOCK_N)
    n_mask = offs_n < C_OUT

    tile_start = tl.load(tile_ptr_data + tile_id)
    tile_end = tl.load(tile_ptr_data + tile_id + 1)
    active_K = tile_end - tile_start

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    k_start = 0
    while k_start < active_K:
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < active_K

        cin_global = tl.load(
            tile_cin_ptr + tile_start + offs_k,
            mask=k_mask,
            other=0
        )

        x_addrs = offs_m[:, None] * C_IN + cin_global[None, :]
        x_mask = m_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptr + x_addrs, mask=x_mask, other=0.0).to(tl.float16)

        w_addrs = cin_global[:, None] * C_OUT + offs_n[None, :]
        w_mask = k_mask[:, None] & n_mask[None, :]
        w_tile = tl.load(w_t_ptr + w_addrs, mask=w_mask, other=0.0).to(tl.float16)

        acc += tl.dot(x_tile, w_tile)
        k_start += BLOCK_K

    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        acc += bias_vals[None, :]

    out_addrs = offs_m[:, None] * C_OUT + offs_n[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptr + out_addrs, acc, mask=out_mask)


# -----------------------------------------------------------------------------
# Python helpers
# -----------------------------------------------------------------------------

def _select_linear_block_m(N: int) -> int:
    if N >= 1024:
        return 128
    elif N >= 256:
        return 64
    else:
        return 32


def sparse_linear_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    threshold: float = 1e-6,
    w_t: torch.Tensor = None,
    counts_buf: torch.Tensor = None,
    tile_cin_buf: torch.Tensor = None,
    return_ms: bool = False,
):
    """
    Folded 2D sparse linear forward.

    Args:
        x: [N, Cin]
        weight: [Cout, Cin]
        bias: [Cout] or None
        threshold: activity threshold
        w_t: optional cached [Cin, Cout]
        counts_buf: optional [>=N_TILES] int32
        tile_cin_buf: optional [>=N_TILES * Cin] int32
        return_ms: whether to time stage-2 sparse kernel via CUDA events

    Returns:
        y: [N, Cout] fp32
        sparse_ms: float
    """
    assert x.dim() == 2, f"Expected 2D x [N, Cin], got {tuple(x.shape)}"
    assert weight.dim() == 2, f"Expected weight [Cout, Cin], got {tuple(weight.shape)}"

    N, C_IN = x.shape
    C_OUT, C_IN_W = weight.shape
    assert C_IN == C_IN_W, f"x Cin={C_IN} but weight Cin={C_IN_W}"

    device = x.device
    BLOCK_M = _select_linear_block_m(N)
    N_TILES = triton.cdiv(N, BLOCK_M)

    x_f16 = x.half().contiguous()
    w_t_f16 = w_t if w_t is not None else weight.half().t().contiguous()

    if counts_buf is None or counts_buf.numel() < N_TILES:
        counts_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)

    needed_cin = N_TILES * C_IN
    if tile_cin_buf is None or tile_cin_buf.numel() < needed_cin:
        tile_cin_buf = torch.empty(needed_cin, dtype=torch.int32, device=device)

    tile_ptr = _build_linear_tile_csr(
        x_f16=x_f16,
        N=N,
        C_IN=C_IN,
        BLOCK_M=BLOCK_M,
        N_TILES=N_TILES,
        threshold=threshold,
        counts_buf=counts_buf,
        tile_cin_buf=tile_cin_buf,
    )

    has_bias = bias is not None
    bias_f32 = bias.float().contiguous() if has_bias else torch.empty(1, device=device, dtype=torch.float32)
    y = torch.zeros(N, C_OUT, dtype=torch.float32, device=device)

    sparse_ms = 0.0
    if return_ms:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()

    def _grid(META):
        return (N_TILES, triton.cdiv(C_OUT, META['BLOCK_N']))

    sparse_linear_pertile_kernel[_grid](
        x_f16,
        w_t_f16,
        bias_f32,
        tile_ptr,
        tile_cin_buf,
        y,
        N,
        C_IN=C_IN,
        C_OUT=C_OUT,
        N_TILES_KEY=N_TILES,
        HAS_BIAS=has_bias,
    )

    if return_ms:
        end_evt.record()
        torch.cuda.synchronize(device)
        sparse_ms = start_evt.elapsed_time(end_evt)

    return y, sparse_ms
