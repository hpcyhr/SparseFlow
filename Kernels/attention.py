"""
SparseFlow Kernels/attention.py - Sparse attention helper kernels.

Maturity: experimental (usable, still evolving).

This file wraps attention-specific matmul flows around Kernels/bmm.py:
- qk stage: Q @ K^T
- av stage: Attn @ V

Softmax, masking, and residual logic remain in model/Ops wrappers.
"""


import math
import torch

import sys
from pathlib import Path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Kernels.bmm import sparse_bmm_forward
from Utils.config import PRESCAN_ACTIVITY_EPS


def sparse_qk_forward(
    q: torch.Tensor,        # [B, num_heads, seq_len, head_dim]
    k: torch.Tensor,        # [B, num_heads, seq_len, head_dim]
    scale: float = None,
    threshold: float = PRESCAN_ACTIVITY_EPS,
    return_ms: bool = False,
    return_tile_stats: bool = False,
):
    """
    Sparse Q 脳 K^T for attention score computation.

    Q is expected sparse (spike-derived queries).  K^T is transposed internally.
    Result: [B, num_heads, seq_len, seq_len] attention logits.

    Args:
        q: Query tensor [B, num_heads, seq_len, head_dim]
        k: Key tensor   [B, num_heads, seq_len, head_dim]
        scale: Scaling factor (default: 1/sqrt(head_dim))

    Returns:
        (attn_logits, ms) + optional (tile_stats,)
    """
    B, num_heads, seq_len, head_dim = q.shape
    assert k.shape == q.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Reshape to [B*num_heads, seq_len, head_dim]
    q_flat = q.reshape(B * num_heads, seq_len, head_dim)
    k_flat = k.reshape(B * num_heads, seq_len, head_dim)

    # Tier 0 P7: rely on sparse_bmm_forward to handle non-contiguous K^T via masked loads, avoid full copy.
    k_t = k_flat.transpose(1, 2)

    # Sparse BMM: Q @ K^T 鈫?[B*H, seq_len, seq_len]
    result = sparse_bmm_forward(
        a=q_flat,
        b=k_t,
        threshold=threshold,
        return_ms=return_ms,
        return_tile_stats=return_tile_stats,
    )

    attn_logits = result[0] * scale
    ms = result[1]

    # Reshape back to [B, num_heads, seq_len, seq_len]
    attn_logits = attn_logits.reshape(B, num_heads, seq_len, seq_len)

    ret = (attn_logits, ms)
    if return_tile_stats:
        ret = ret + (result[2] if len(result) > 2 else None,)
    return ret


def sparse_attn_v_forward(
    attn: torch.Tensor,      # [B, num_heads, seq_len, seq_len]
    v: torch.Tensor,         # [B, num_heads, seq_len, head_dim]
    threshold: float = PRESCAN_ACTIVITY_EPS,
    return_ms: bool = False,
    return_tile_stats: bool = False,
):
    """
    Sparse attn 脳 V for attention output computation.

    attn is expected sparse when using spike-based thresholding instead
    of softmax (common in Spikeformer variants).

    Returns:
        (output, ms) + optional (tile_stats,)
        output shape: [B, num_heads, seq_len, head_dim]
    """
    B, num_heads, seq_len_q, seq_len_k = attn.shape
    B2, H2, seq_len_k2, head_dim = v.shape
    assert B == B2 and num_heads == H2 and seq_len_k == seq_len_k2

    attn_flat = attn.reshape(B * num_heads, seq_len_q, seq_len_k)
    v_flat = v.reshape(B * num_heads, seq_len_k, head_dim)

    result = sparse_bmm_forward(
        a=attn_flat,
        b=v_flat,
        threshold=threshold,
        return_ms=return_ms,
        return_tile_stats=return_tile_stats,
    )

    output = result[0].reshape(B, num_heads, seq_len_q, head_dim)
    ms = result[1]

    ret = (output, ms)
    if return_tile_stats:
        ret = ret + (result[2] if len(result) > 2 else None,)
    return ret
