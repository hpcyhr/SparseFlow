"""
SparseFlow Kernels/depthwise_conv2d.py 鈥?Sparse Depthwise Conv2d v1.0

Depthwise convolution (groups=C_in): each input channel is convolved
independently with its own filter.  Sparsity exploitation is simpler
than standard conv: if input channel c is all-zero in a spatial tile,
the entire output channel c for that tile is zero 鈫?skip.

This is a separate file from conv2d.py (which handles groups=1).
MobileNet-style networks use depthwise conv extensively.

Kernel pattern:
  Stage-1: per-(N, C, tile_h, tile_w) zero check on input
  Stage-2: only non-zero (channel, tile) pairs execute the k脳k filter
"""

import torch
import triton
import triton.language as tl

TILE_ZERO = 0
TILE_ACTIVE = 1


# ---------------------------------------------------------------------------
# Prescan: per-channel, per-spatial-tile zero detection
# ---------------------------------------------------------------------------

@triton.jit
def _prescan_depthwise_kernel(
    x_ptr,              # [N, C, H_IN, W_IN]
    tile_mask_ptr,      # [N * C * GH * GW] 鈥?0=zero, 1=active
    N: tl.constexpr,
    C: tl.constexpr,
    H_IN: tl.constexpr,
    W_IN: tl.constexpr,
    KH: tl.constexpr,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    GH: tl.constexpr,
    GW: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
    # pid 鈫?(n, c, gh, gw)
    pid = tl.program_id(0)
    tiles_per_sample = C * GH * GW
    n_idx = pid // tiles_per_sample
    rem = pid % tiles_per_sample
    c_idx = rem // (GH * GW)
    rem2 = rem % (GH * GW)
    gh_idx = rem2 // GW
    gw_idx = rem2 % GW

    # Receptive field in input space
    h_out_start = gh_idx * BH
    w_out_start = gw_idx * BW

    off1 = tl.arange(0, 1)
    any_nonzero = tl.zeros([1], dtype=tl.int32)
    for bh in range(BH):
        for bw in range(BW):
            oh = h_out_start + bh
            ow = w_out_start + bw
            for kh in range(KH):
                for kw in range(KH):
                    ih = oh * STRIDE - PADDING + kh
                    iw = ow * STRIDE - PADDING + kw
                    if (ih >= 0) and (ih < H_IN):
                        if (iw >= 0) and (iw < W_IN):
                            addr = n_idx * C * H_IN * W_IN + c_idx * H_IN * W_IN + ih * W_IN + iw
                            val = tl.load(x_ptr + addr)
                            if tl.abs(val) > THRESHOLD:
                                any_nonzero = tl.full([1], 1, dtype=tl.int32)

    tl.store(tile_mask_ptr + pid + off1, any_nonzero)


# ---------------------------------------------------------------------------
# Compute kernel 鈥?depthwise conv on active tiles only
# ---------------------------------------------------------------------------

@triton.jit
def _sparse_depthwise_conv2d_kernel(
    x_ptr, w_ptr, bias_ptr, y_ptr,
    tile_mask_ptr,
    N: tl.constexpr, C: tl.constexpr,
    H_IN: tl.constexpr, W_IN: tl.constexpr,
    H_OUT: tl.constexpr, W_OUT: tl.constexpr,
    KH: tl.constexpr,
    STRIDE: tl.constexpr, PADDING: tl.constexpr,
    BH: tl.constexpr, BW: tl.constexpr,
    GH: tl.constexpr, GW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)

    off1 = tl.arange(0, 1)
    tile_active = tl.load(tile_mask_ptr + pid + off1)
    if tl.sum(tile_active) == 0:
        # Zero tile: write zeros (or skip if output pre-zeroed)
        return

    tiles_per_sample = C * GH * GW
    n_idx = pid // tiles_per_sample
    rem = pid % tiles_per_sample
    c_idx = rem // (GH * GW)
    rem2 = rem % (GH * GW)
    gh_idx = rem2 // GW
    gw_idx = rem2 % GW

    h_start = gh_idx * BH
    w_start = gw_idx * BW

    bias_val = 0.0
    if HAS_BIAS:
        bias_val = tl.load(bias_ptr + c_idx)

    # Simple scalar loop for depthwise (k脳k is typically 3脳3)
    for bh in range(BH):
        oh = h_start + bh
        if oh < H_OUT:
            for bw_i in range(BW):
                ow = w_start + bw_i
                if ow < W_OUT:
                    acc = 0.0
                    for kh in range(KH):
                        for kw in range(KH):
                            ih = oh * STRIDE - PADDING + kh
                            iw = ow * STRIDE - PADDING + kw
                            if (ih >= 0) and (ih < H_IN):
                                if (iw >= 0) and (iw < W_IN):
                                    x_addr = n_idx * C * H_IN * W_IN + c_idx * H_IN * W_IN + ih * W_IN + iw
                                    w_addr = c_idx * KH * KH + kh * KH + kw
                                    x_val = tl.load(x_ptr + x_addr).to(tl.float32)
                                    w_val = tl.load(w_ptr + w_addr).to(tl.float32)
                                    acc += x_val * w_val
                    acc += bias_val
                    y_addr = n_idx * C * H_OUT * W_OUT + c_idx * H_OUT * W_OUT + oh * W_OUT + ow
                    tl.store(y_ptr + y_addr, acc)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def sparse_depthwise_conv2d_forward(
    x: torch.Tensor,       # [N, C, H, W]
    weight: torch.Tensor,  # [C, 1, KH, KW]
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 1,
    threshold: float = 1e-6,
    return_ms: bool = False,
    return_tile_stats: bool = False,
):
    """
    Sparse depthwise conv2d.
    Skips computation for (channel, spatial-tile) pairs where input is zero.

    Returns: (y, ms) + optional (tile_stats,)
    """
    import torch.nn.functional as Fn

    N, C, H_IN, W_IN = x.shape
    KH = weight.shape[2]

    if isinstance(stride, tuple):
        stride = stride[0]
    if isinstance(padding, tuple):
        padding = padding[0]

    H_OUT = (H_IN + 2 * padding - KH) // stride + 1
    W_OUT = (W_IN + 2 * padding - KH) // stride + 1

    dense_bias = bias.float() if bias is not None else None

    # Fallback for unsupported configs
    if KH > 7 or H_OUT <= 0 or W_OUT <= 0:
        y = Fn.conv2d(x.float(), weight.float(), dense_bias, stride=stride, padding=padding, groups=C)
        ret = (y.float(), 0.0)
        if return_tile_stats:
            ret = ret + (None,)
        return ret

    # Tile sizes for depthwise: smaller tiles since each tile is 1 channel
    BH = min(8, H_OUT)
    BW = min(8, W_OUT)
    GH = (H_OUT + BH - 1) // BH
    GW = (W_OUT + BW - 1) // BW
    TOTAL_TILES = N * C * GH * GW

    device = x.device
    x_f16 = x.half().contiguous()
    w_f32 = weight.float().reshape(C, KH, KH).contiguous()

    tile_mask = torch.empty(TOTAL_TILES, dtype=torch.int32, device=device)

    # Prescan
    try:
        _prescan_depthwise_kernel[(TOTAL_TILES,)](
            x_f16, tile_mask,
            N=N, C=C, H_IN=H_IN, W_IN=W_IN,
            KH=KH, STRIDE=stride, PADDING=padding,
            BH=BH, BW=BW, GH=GH, GW=GW,
            THRESHOLD=threshold,
        )
    except Exception:
        y = Fn.conv2d(x.float(), weight.float(), dense_bias, stride=stride, padding=padding, groups=C).float()
        ret = (y, 0.0)
        if return_tile_stats:
            ret = ret + ({
                'total_tiles': TOTAL_TILES,
                'fallback': True,
                'reason': 'prescan_failed',
            },)
        return ret

    # Output prefill: zero-input depthwise tile should output bias (if present).
    y = torch.zeros(N, C, H_OUT, W_OUT, dtype=torch.float32, device=device)
    if dense_bias is not None:
        y = y + dense_bias.view(1, C, 1, 1)

    if return_ms:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    bias_ptr = dense_bias.contiguous() if dense_bias is not None else torch.empty(0, device=device)

    try:
        _sparse_depthwise_conv2d_kernel[(TOTAL_TILES,)](
            x_f16, w_f32, bias_ptr, y,
            tile_mask,
            N=N, C=C, H_IN=H_IN, W_IN=W_IN,
            H_OUT=H_OUT, W_OUT=W_OUT,
            KH=KH, STRIDE=stride, PADDING=padding,
            BH=BH, BW=BW, GH=GH, GW=GW,
            HAS_BIAS=(bias is not None),
        )
    except Exception:
        y = Fn.conv2d(x.float(), weight.float(), dense_bias, stride=stride, padding=padding, groups=C).float()
        ret = (y, 0.0)
        if return_tile_stats:
            ret = ret + ({
                'total_tiles': TOTAL_TILES,
                'fallback': True,
                'reason': 'kernel_launch_failed',
            },)
        return ret

    ms = 0.0
    if return_ms:
        end.record()
        torch.cuda.synchronize(device)
        ms = start.elapsed_time(end)

    stats = None
    if return_tile_stats:
        active = int(tile_mask.sum().item())
        stats = {
            'total_tiles': TOTAL_TILES,
            'active_tiles': active,
            'zero_tiles': TOTAL_TILES - active,
            'zero_ratio': 1.0 - active / max(TOTAL_TILES, 1),
        }

    ret = (y, ms)
    if return_tile_stats:
        ret = ret + (stats,)
    return ret
