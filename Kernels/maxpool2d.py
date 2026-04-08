"""
SparseFlow Kernels/maxpool2d.py

Sparse MaxPool2d with metadata-first zero-tile skipping.

This follows the same high-level SparseFlow method:
  1. output-space tiling
  2. metadata build first
  3. zero-tile short circuit
  4. active-only or all-tiles launch
  5. structured stats / backend meta
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

from Utils.config import PRESCAN_ACTIVITY_EPS, SPARSE_DENSE_RATIO_THRESHOLD
from Utils.sparse_helpers import TILE_ZERO, TILE_SPARSE

FALLBACK_RATIO = SPARSE_DENSE_RATIO_THRESHOLD
NEG_INF = -3.402823466e38


def _pair(v) -> Tuple[int, int]:
    if isinstance(v, (tuple, list)):
        if len(v) == 1:
            return int(v[0]), int(v[0])
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _select_pool_tile_sizes(h_out: int, w_out: int) -> Tuple[int, int]:
    pixels = h_out * w_out
    if pixels >= 3136:
        return 8, 16
    return 8, 8


def _ensure_metadata_buffers(
    ag_mask_buf: Optional[torch.Tensor],
    tile_class_buf: Optional[torch.Tensor],
    total_tiles: int,
    device: torch.device,
):
    if ag_mask_buf is None or ag_mask_buf.numel() < total_tiles:
        ag_mask_buf = torch.empty(total_tiles, dtype=torch.int32, device=device)
    if tile_class_buf is None or tile_class_buf.numel() < total_tiles:
        tile_class_buf = torch.empty(total_tiles, dtype=torch.int32, device=device)
    return ag_mask_buf, tile_class_buf


def _build_active_tile_ids(tile_class_buf: torch.Tensor, total_tiles: int):
    tc = tile_class_buf[:total_tiles]
    active = torch.nonzero(tc != TILE_ZERO, as_tuple=False).flatten()
    if active.numel() == 0:
        return active.to(dtype=torch.int32), 0
    return active.to(dtype=torch.int32).contiguous(), int(active.numel())


@triton.jit
def _prescan_maxpool2d_kernel(
    x_ptr,
    ag_mask_ptr,
    tile_class_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H_IN: tl.constexpr,
    W_IN: tl.constexpr,
    H_OUT: tl.constexpr,
    W_OUT: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    GH: tl.constexpr,
    GW: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles_per_sample = C * GH * GW
    n_idx = pid // tiles_per_sample
    rem = pid % tiles_per_sample
    c_idx = rem // (GH * GW)
    rem2 = rem % (GH * GW)
    gh_idx = rem2 // GW
    gw_idx = rem2 % GW

    h_out_start = gh_idx * BH
    w_out_start = gw_idx * BW

    off1 = tl.arange(0, 1)
    any_nonzero = tl.zeros([1], dtype=tl.int32)

    for bh in range(BH):
        oh = h_out_start + bh
        if oh < H_OUT:
            for bw in range(BW):
                ow = w_out_start + bw
                if ow < W_OUT:
                    for kh in range(KH):
                        ih = oh * STRIDE_H - PAD_H + kh * DIL_H
                        if (ih >= 0) and (ih < H_IN):
                            for kw in range(KW):
                                iw = ow * STRIDE_W - PAD_W + kw * DIL_W
                                if (iw >= 0) and (iw < W_IN):
                                    addr = n_idx * C * H_IN * W_IN + c_idx * H_IN * W_IN + ih * W_IN + iw
                                    val = tl.load(x_ptr + addr)
                                    if tl.abs(val) > THRESHOLD:
                                        any_nonzero = tl.full([1], 1, dtype=tl.int32)

    tl.store(ag_mask_ptr + pid + off1, any_nonzero)
    if tl.sum(any_nonzero) == 0:
        tl.store(tile_class_ptr + pid + off1, tl.zeros([1], dtype=tl.int32))
    else:
        tl.store(tile_class_ptr + pid + off1, tl.full([1], TILE_SPARSE, dtype=tl.int32))


@triton.jit
def _sparse_maxpool2d_kernel(
    x_ptr,
    y_ptr,
    tile_class_ptr,
    tile_ids_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H_IN: tl.constexpr,
    W_IN: tl.constexpr,
    H_OUT: tl.constexpr,
    W_OUT: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    GH: tl.constexpr,
    GW: tl.constexpr,
    USE_TILE_IDS: tl.constexpr,
):
    pid_tile = tl.program_id(0)
    tile_id = tl.load(tile_ids_ptr + pid_tile) if USE_TILE_IDS else pid_tile

    tiles_per_sample = C * GH * GW
    if tile_id >= N * tiles_per_sample:
        return

    off1 = tl.arange(0, 1)
    tc = tl.load(tile_class_ptr + tile_id + off1)
    if tl.sum(tc) == TILE_ZERO:
        return

    n_idx = tile_id // tiles_per_sample
    rem = tile_id % tiles_per_sample
    c_idx = rem // (GH * GW)
    rem2 = rem % (GH * GW)
    gh_idx = rem2 // GW
    gw_idx = rem2 % GW

    h_out_start = gh_idx * BH
    w_out_start = gw_idx * BW

    for bh in range(BH):
        oh = h_out_start + bh
        if oh < H_OUT:
            for bw in range(BW):
                ow = w_out_start + bw
                if ow < W_OUT:
                    max_val = tl.full([1], NEG_INF, dtype=tl.float32)
                    has_valid = tl.zeros([1], dtype=tl.int32)
                    for kh in range(KH):
                        ih = oh * STRIDE_H - PAD_H + kh * DIL_H
                        if (ih >= 0) and (ih < H_IN):
                            for kw in range(KW):
                                iw = ow * STRIDE_W - PAD_W + kw * DIL_W
                                if (iw >= 0) and (iw < W_IN):
                                    x_addr = n_idx * C * H_IN * W_IN + c_idx * H_IN * W_IN + ih * W_IN + iw
                                    x_val = tl.load(x_ptr + x_addr).to(tl.float32)
                                    max_val = tl.maximum(max_val, x_val)
                                    has_valid = tl.full([1], 1, dtype=tl.int32)
                    out_val = max_val
                    if tl.sum(has_valid) == 0:
                        out_val = tl.zeros([1], dtype=tl.float32)
                    y_addr = n_idx * C * H_OUT * W_OUT + c_idx * H_OUT * W_OUT + oh * W_OUT + ow
                    tl.store(y_ptr + y_addr, out_val)


def sparse_maxpool2d_forward(
    x: torch.Tensor,
    kernel_size=2,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode: bool = False,
    threshold: float = PRESCAN_ACTIVITY_EPS,
    ag_mask_buf: Optional[torch.Tensor] = None,
    tile_class_buf: Optional[torch.Tensor] = None,
    return_ms: bool = False,
    fallback_ratio: float = FALLBACK_RATIO,  # kept for interface uniformity
    return_avg_active_ratio: bool = False,
    return_tile_stats: bool = False,
    return_backend_meta: bool = False,
    active_tile_ids_buf: Optional[torch.Tensor] = None,
    launch_all_tiles: bool = False,
):
    del fallback_ratio
    import torch.nn.functional as Fn

    if stride is None:
        stride = kernel_size

    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)

    N, C, H_IN, W_IN = x.shape
    device = x.device
    need_stats = return_tile_stats or return_avg_active_ratio

    def _finalize_return(y, ms, avg_active_ratio_val=None, tile_stats_val=None, backend_meta_val=None):
        ret = (y, ms)
        if return_avg_active_ratio:
            ret = ret + (avg_active_ratio_val,)
        if return_tile_stats:
            ret = ret + (tile_stats_val,)
        if return_backend_meta:
            ret = ret + (backend_meta_val,)
        return ret

    def _dense_fallback(reason: str):
        dense_ms = 0.0
        if return_ms:
            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            se.record()
        y = Fn.max_pool2d(
            x.float(),
            kernel_size=(kh, kw),
            stride=(sh, sw),
            padding=(ph, pw),
            dilation=(dh, dw),
            ceil_mode=ceil_mode,
        ).float()
        if return_ms:
            ee.record()
            torch.cuda.synchronize(device)
            dense_ms = se.elapsed_time(ee)
        backend_meta = {"backend": "dense_fallback", "reason": reason}
        return _finalize_return(y, dense_ms, 1.0, None, backend_meta)

    if x.ndim != 4:
        return _dense_fallback("invalid_tensor_rank")
    if ceil_mode:
        return _dense_fallback("unsupported_ceil_mode")

    H_OUT = (H_IN + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    W_OUT = (W_IN + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    if H_OUT <= 0 or W_OUT <= 0:
        return _dense_fallback("invalid_output_shape")

    bh, bw = _select_pool_tile_sizes(H_OUT, W_OUT)
    gh = triton.cdiv(H_OUT, bh)
    gw = triton.cdiv(W_OUT, bw)
    total_tiles = N * C * gh * gw

    x_f16 = x if (x.dtype == torch.float16 and x.is_contiguous()) else x.half().contiguous()
    ag_mask_buf, tile_class_buf = _ensure_metadata_buffers(ag_mask_buf, tile_class_buf, total_tiles, device)

    try:
        _prescan_maxpool2d_kernel[(total_tiles,)](
            x_f16,
            ag_mask_buf,
            tile_class_buf,
            N=N,
            C=C,
            H_IN=H_IN,
            W_IN=W_IN,
            H_OUT=H_OUT,
            W_OUT=W_OUT,
            KH=kh,
            KW=kw,
            STRIDE_H=sh,
            STRIDE_W=sw,
            PAD_H=ph,
            PAD_W=pw,
            DIL_H=dh,
            DIL_W=dw,
            BH=bh,
            BW=bw,
            GH=gh,
            GW=gw,
            THRESHOLD=float(threshold),
        )
    except Exception:
        return _dense_fallback("prescan_failed")

    avg_active_ratio = None
    tile_stats = None
    active_tiles = None

    if need_stats:
        tc = tile_class_buf[:total_tiles]
        zero_tiles = int((tc == TILE_ZERO).sum().item())
        sparse_tiles = int((tc == TILE_SPARSE).sum().item())
        active_tiles = int(sparse_tiles)
        avg_active_ratio = float(active_tiles) / max(float(total_tiles), 1.0)
        if return_tile_stats:
            tile_stats = {
                "zero_tiles": int(zero_tiles),
                "sparse_tiles": int(sparse_tiles),
                "denseish_tiles": 0,
                "total_tiles": int(total_tiles),
                "active_tiles": int(active_tiles),
                "active_tile_ratio": float(active_tiles) / max(float(total_tiles), 1.0),
                "avg_active_group_ratio": float(avg_active_ratio),
                "prescan_mode": "pool2d_zero_skip_v1",
                "group_size_c": 1,
                "num_groups": 1,
            }

    if launch_all_tiles:
        launch_count = total_tiles
        use_tile_ids = False
        tile_ids_ptr = ag_mask_buf
    else:
        active_tile_ids, active_tile_count = _build_active_tile_ids(tile_class_buf, total_tiles)
        if active_tile_count == 0:
            y = torch.zeros(N, C, H_OUT, W_OUT, dtype=torch.float32, device=device)
            backend_meta = {
                "backend": "zero_tiles_only",
                "reason": "all_tiles_zero_after_metadata",
                "total_tiles": int(total_tiles),
                "launch_count": 0,
                "launch_mode": "active_only",
                "active_tiles": 0,
            }
            return _finalize_return(y, 0.0, 0.0, tile_stats, backend_meta)
        if active_tile_ids_buf is not None and active_tile_ids_buf.numel() >= active_tile_count:
            active_tile_ids_buf[:active_tile_count].copy_(active_tile_ids)
            tile_ids_ptr = active_tile_ids_buf[:active_tile_count]
        else:
            tile_ids_ptr = active_tile_ids
        launch_count = active_tile_count
        use_tile_ids = True
        if active_tiles is None:
            active_tiles = int(active_tile_count)

    y = torch.zeros(N, C, H_OUT, W_OUT, dtype=torch.float32, device=device)

    sparse_ms = 0.0
    if return_ms:
        se = torch.cuda.Event(enable_timing=True)
        ee = torch.cuda.Event(enable_timing=True)
        se.record()

    try:
        _sparse_maxpool2d_kernel[(launch_count,)](
            x_f16,
            y,
            tile_class_buf,
            tile_ids_ptr,
            N=N,
            C=C,
            H_IN=H_IN,
            W_IN=W_IN,
            H_OUT=H_OUT,
            W_OUT=W_OUT,
            KH=kh,
            KW=kw,
            STRIDE_H=sh,
            STRIDE_W=sw,
            PAD_H=ph,
            PAD_W=pw,
            DIL_H=dh,
            DIL_W=dw,
            BH=bh,
            BW=bw,
            GH=gh,
            GW=gw,
            USE_TILE_IDS=use_tile_ids,
        )
    except Exception:
        return _dense_fallback("kernel_launch_failed")

    if return_ms:
        ee.record()
        torch.cuda.synchronize(device)
        sparse_ms = se.elapsed_time(ee)

    if avg_active_ratio is None and total_tiles > 0:
        avg_active_ratio = float(active_tiles or 0) / float(total_tiles)

    backend_meta = {
        "backend": "sparse_triton",
        "reason": "maxpool2d_zero_skip_v1",
        "total_tiles": int(total_tiles),
        "launch_count": int(launch_count),
        "launch_mode": "all_tiles" if launch_all_tiles else "active_only",
        "active_tiles": int(active_tiles or 0),
        "avg_active_group_ratio": float(avg_active_ratio or 0.0),
    }
    return _finalize_return(y, sparse_ms, avg_active_ratio, tile_stats, backend_meta)
