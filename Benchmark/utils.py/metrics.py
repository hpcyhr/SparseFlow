from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a_flat = a.detach().float().reshape(-1)
    b_flat = b.detach().float().reshape(-1)
    na = float(a_flat.norm())
    nb = float(b_flat.norm())
    if na < eps and nb < eps:
        return 1.0
    if na < eps or nb < eps:
        return 0.0
    cos = F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0), dim=1)
    return float(cos.item())


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def correctness_metrics(dense_out: torch.Tensor, sparse_out: torch.Tensor) -> Dict[str, float]:
    return {
        "cosine_similarity": cosine_similarity(dense_out, sparse_out),
        "max_abs_error": max_abs_error(dense_out, sparse_out),
    }

