"""
SparseFlow Kernels/batchnorm2d.py - Sparse-aware BatchNorm2d inference helper.

Maturity: prototype/stats_only (non-primary path).

For exact-zero spatial locations in spike tensors, BatchNorm output is a
channel-wise constant. This helper uses that property to avoid repeated full
elementwise BN work for zero regions while preserving exact inference math.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def precompute_zero_bias(bn: nn.BatchNorm2d) -> torch.Tensor:
    """Compute BN output for x==0 per channel.

    y_zero = (-running_mean / sqrt(running_var + eps)) * weight + bias
    """
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    gamma = bn.weight if bn.weight is not None else torch.ones_like(mean)
    beta = bn.bias if bn.bias is not None else torch.zeros_like(mean)
    inv_std = 1.0 / torch.sqrt(var + eps)
    return (-mean * inv_std) * gamma + beta


def sparse_batchnorm2d_forward(
    x: torch.Tensor,
    bn: nn.BatchNorm2d,
    threshold: float = 1e-6,
):
    """Sparse-aware BN2d inference.

    Args:
        x: [N, C, H, W]
        bn: BatchNorm2d in eval mode
        threshold: zero detection threshold for spatial location activity

    Returns:
        y: [N, C, H, W]
        elapsed_ms: kernel timing (CUDA event, 0 on CPU)
    """
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor [N,C,H,W], got {tuple(x.shape)}")

    device = x.device
    use_cuda_timing = device.type == "cuda"
    if use_cuda_timing:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    n_dim, c_dim, _, _ = x.shape
    zero_bias = precompute_zero_bias(bn)  # [C]

    # Detect (n,h,w) positions that are all-zero across channels.
    spatial_sum = x.abs().sum(dim=1)      # [N,H,W]
    is_zero = spatial_sum <= threshold    # [N,H,W]

    # Dense BN for all positions, then overwrite zero positions with constants.
    y = nn.functional.batch_norm(
        x,
        bn.running_mean,
        bn.running_var,
        bn.weight,
        bn.bias,
        False,
        0.0,
        bn.eps,
    )

    zero_mask = is_zero.unsqueeze(1)  # [N,1,H,W]
    y = torch.where(zero_mask, zero_bias.view(1, c_dim, 1, 1), y)

    elapsed_ms = 0.0
    if use_cuda_timing:
        end.record()
        torch.cuda.synchronize(device)
        elapsed_ms = float(start.elapsed_time(end))

    return y, elapsed_ms
