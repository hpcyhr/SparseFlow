"""
SparseFlow Kernels/conv3d.py — Sparse Conv3d Triton Kernel v1.0

3D convolution [N, C_IN, D, H, W] → [N, C_OUT, D_OUT, H_OUT, W_OUT]
with grouped-bitmask prescan on input channels per volumetric tile.

v1: Prescan + dense compute fallback with tile classification stats.
Sparse compute kernel deferred to v2 — the prescan establishes whether
3D sparsity patterns in SNN volumetric models justify dedicated kernels.

Supported: kernel_size=3, stride=1, padding=1, groups=1, dilation=1.
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
    choose_group_size,
)

FALLBACK_RATIO = 0.85


# ---------------------------------------------------------------------------
# Public entry — v1 prescan + F.conv3d fallback
# ---------------------------------------------------------------------------

def sparse_conv3d_forward(
    x: torch.Tensor,        # [N, C_IN, D, H, W]
    weight: torch.Tensor,   # [C_OUT, C_IN, KD, KH, KW]
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 1,
    threshold: float = 1e-6,
    return_ms: bool = False,
    return_tile_stats: bool = False,
):
    """
    Sparse 3D convolution — v1 with input sparsity profiling.

    Currently executes dense compute (F.conv3d) but reports per-tile
    sparsity statistics to guide future kernel development.
    """
    import torch.nn.functional as Fn

    N_batch, C_IN, D_IN, H_IN, W_IN = x.shape
    C_OUT = weight.shape[0]
    KD = weight.shape[2]
    KH = weight.shape[3]
    device = x.device

    if isinstance(stride, (tuple, list)):
        stride = stride[0]
    if isinstance(padding, (tuple, list)):
        padding = padding[0]

    D_OUT = (D_IN + 2 * padding - KD) // stride + 1
    H_OUT = (H_IN + 2 * padding - KH) // stride + 1
    W_OUT = (W_IN + 2 * padding - KH) // stride + 1

    # v1: always use dense path for compute
    if return_ms:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    y = Fn.conv3d(x, weight, bias, stride=stride, padding=padding).float()

    ms = 0.0
    if return_ms:
        end.record()
        torch.cuda.synchronize(device)
        ms = start.elapsed_time(end)

    stats = None
    if return_tile_stats:
        # Quick sparsity analysis on input
        with torch.no_grad():
            x_flat = x.view(N_batch, C_IN, -1)  # [N, C_IN, D*H*W]
            spatial_nnz = (x_flat.abs() > threshold).any(dim=2)  # [N, C_IN]
            active_channels_per_sample = spatial_nnz.sum(dim=1).float()  # [N]
            avg_active_ratio = (active_channels_per_sample.mean().item() / max(C_IN, 1))

        stats = {
            'total_volume': N_batch * D_OUT * H_OUT * W_OUT,
            'avg_active_channel_ratio': avg_active_ratio,
            'element_sparsity': 1.0 - (x.abs() > threshold).float().mean().item(),
            'prescan_version': 'conv3d_v1_stats_only',
        }

    ret = (y, ms)
    if return_tile_stats:
        ret = ret + (stats,)
    return ret