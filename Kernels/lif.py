"""
SparseFlow Kernels/lif.py — Standalone LIF Neuron Triton Kernel v1.0

Leaky Integrate-and-Fire neuron dynamics as a standalone kernel:
    V_temp = V_prev * decay + I
    spike  = 1.0 if V_temp >= v_threshold else 0.0
    V_next = V_temp - spike * v_threshold  (soft reset)
         or  spike * v_reset + (1-spike) * V_temp  (hard reset)

This kernel is NOT fused with convolution — it operates on pre-computed
current tensors.  The fused Conv+LIF kernel lives in fused_conv_lif.py.

Implementation note:
- This kernel currently applies the same update for all elements and does not
  implement a dedicated zero-input short-circuit branch.

Supports: flat [N] and batched [*, N] inputs via reshape.
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _lif_kernel(
    current_ptr,    # [TOTAL] input current
    v_prev_ptr,     # [TOTAL] previous membrane potential
    spike_ptr,      # [TOTAL] output spikes
    v_next_ptr,     # [TOTAL] next membrane potential
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
    DECAY: tl.constexpr,
    V_TH: tl.constexpr,
    V_RESET: tl.constexpr,      # NaN → soft reset
    USE_SOFT_RESET: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL

    I = tl.load(current_ptr + offs, mask=mask, other=0.0)
    v_prev = tl.load(v_prev_ptr + offs, mask=mask, other=0.0)

    v_temp = v_prev * DECAY + I
    spike = (v_temp >= V_TH).to(tl.float32)

    if USE_SOFT_RESET:
        v_next = v_temp - spike * V_TH
    else:
        v_next = spike * V_RESET + (1.0 - spike) * v_temp

    tl.store(spike_ptr + offs, spike, mask=mask)
    tl.store(v_next_ptr + offs, v_next, mask=mask)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def lif_forward(
    current: torch.Tensor,       # Input current, any shape
    v_prev: torch.Tensor,        # Previous membrane potential, same shape
    decay: float = 0.5,          # τ decay factor (1/τ typically)
    v_threshold: float = 1.0,
    v_reset: float = None,       # None → soft reset
    return_ms: bool = False,
) -> tuple:
    """
    Standalone LIF neuron forward pass.

    Args:
        current:     Input current tensor (any shape, will be flattened internally)
        v_prev:      Previous membrane potential (same shape as current)
        decay:       Decay factor for membrane potential
        v_threshold: Firing threshold
        v_reset:     Reset voltage (None for soft reset)
        return_ms:   Whether to return kernel timing

    Returns:
        (spike, v_next, ms) — spike and v_next have same shape as input
    """
    orig_shape = current.shape
    device = current.device

    current_flat = current.reshape(-1).float().contiguous()
    v_prev_flat = v_prev.reshape(-1).float().contiguous()
    TOTAL = current_flat.numel()

    spike_flat = torch.empty(TOTAL, dtype=torch.float32, device=device)
    v_next_flat = torch.empty(TOTAL, dtype=torch.float32, device=device)

    BLOCK = 1024
    grid = ((TOTAL + BLOCK - 1) // BLOCK,)

    use_soft = v_reset is None
    v_reset_val = 0.0 if v_reset is None else float(v_reset)

    if return_ms:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    _lif_kernel[grid](
        current_flat, v_prev_flat, spike_flat, v_next_flat,
        TOTAL=TOTAL, BLOCK=BLOCK,
        DECAY=decay, V_TH=v_threshold,
        V_RESET=v_reset_val, USE_SOFT_RESET=use_soft,
    )

    ms = 0.0
    if return_ms:
        end.record()
        torch.cuda.synchronize(device)
        ms = start.elapsed_time(end)

    spike = spike_flat.reshape(orig_shape)
    v_next = v_next_flat.reshape(orig_shape)

    return spike, v_next, ms
