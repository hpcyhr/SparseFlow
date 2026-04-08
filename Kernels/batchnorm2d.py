"""
Kernels/batchnorm2d.py

Experimental / prototype helper.

Important:
- This path is not part of SparseFlow's mature main-line acceleration.
- It runs dense BN first, then applies a zero-location overwrite pass.
- It is intended for diagnostics and research experiments, not for claiming
  speedup over native BatchNorm2d.
"""

import torch
import torch.nn as nn


def precompute_zero_bias(bn: nn.BatchNorm2d) -> torch.Tensor:
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    gamma = bn.weight if bn.weight is not None else torch.ones_like(mean)
    beta = bn.bias if bn.bias is not None else torch.zeros_like(mean)
    inv_std = 1.0 / torch.sqrt(var + eps)
    return (-mean * inv_std) * gamma + beta


def sparse_batchnorm2d_forward(x: torch.Tensor, bn: nn.BatchNorm2d, threshold: float = 1e-6):
    """
    Experimental BN helper:
    1) run dense BN
    2) overwrite fully zero spatial locations with precomputed channel bias
    """
    n, c, h, w = x.shape

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    zero_bias = precompute_zero_bias(bn)
    spatial_sum = x.abs().sum(dim=1)
    is_zero = spatial_sum <= threshold

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

    zero_mask = is_zero.unsqueeze(1)
    y = torch.where(zero_mask, zero_bias.view(1, c, 1, 1), y)

    end.record()
    torch.cuda.synchronize(x.device)
    return y, start.elapsed_time(end)

