"""
SparseFlow Ops/sparse_lif.py — SparseLIF nn.Module Wrapper

Stateful LIF neuron module backed by Kernels/lif.py.
Manages membrane potential state across timesteps, supports
multi-step (5D) input, and provides diagnostic hooks.

This is the STANDALONE LIF module — not fused with convolution.
For fused Conv+LIF, use Ops/sparse_fused_conv_lif.py.

Compatible with spikingjelly's LIFNode interface for from_dense().
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn


class SparseLIF(nn.Module):
    """
    Leaky Integrate-and-Fire neuron with Triton kernel acceleration.

    Maintains internal membrane potential state (self.v).
    Supports single-step [N, *] and multi-step [T, N, *] inputs.
    """

    def __init__(
        self,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float = None,
        step_mode: str = "s",
        return_ms: bool = False,
    ):
        super().__init__()
        self.tau = float(tau)
        self.decay = 1.0 / self.tau  # decay factor per timestep
        self.v_threshold = float(v_threshold)
        self.v_reset = v_reset  # None → soft reset
        self.step_mode = step_mode  # "s" = single-step, "m" = multi-step
        self.return_ms = return_ms

        self._triton_available = False
        try:
            import triton  # noqa: F401
            self._triton_available = True
        except Exception:
            pass

        # State
        self.register_buffer("v", None, persistent=False)

        self._last_sparse_ms = 0.0
        self._last_diag: Dict[str, Any] = {}

    @classmethod
    def from_sj(cls, lif_node, return_ms: bool = False, **kwargs):
        """
        Create from a spikingjelly LIFNode.

        Args:
            lif_node: spikingjelly.activation_based.neuron.LIFNode
        """
        tau = getattr(lif_node, "tau", 2.0)
        v_th = getattr(lif_node, "v_threshold", 1.0)
        v_reset = getattr(lif_node, "v_reset", None)
        step_mode = getattr(lif_node, "step_mode", "s")

        return cls(
            tau=float(tau),
            v_threshold=float(v_th),
            v_reset=float(v_reset) if v_reset is not None else None,
            step_mode=step_mode,
            return_ms=return_ms,
            **kwargs,
        )

    def reset(self):
        """Reset membrane potential to zero."""
        self.v = None

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        """
        Args:
            current: Input current.
                     step_mode="s": [N, *] single timestep
                     step_mode="m": [T, N, *] multiple timesteps

        Returns:
            spike: Same shape as current
        """
        if self.step_mode == "m" and current.ndim >= 3:
            return self._multi_step_forward(current)
        return self._single_step_forward(current)

    def _single_step_forward(self, current: torch.Tensor) -> torch.Tensor:
        if self.v is None:
            self.v = torch.zeros_like(current)

        if self._triton_available and current.is_cuda:
            from Kernels.lif import lif_forward

            spike, v_next, ms = lif_forward(
                current=current,
                v_prev=self.v,
                decay=self.decay,
                v_threshold=self.v_threshold,
                v_reset=self.v_reset,
                return_ms=self.return_ms,
            )
            self.v = v_next.detach()
            self._last_sparse_ms = ms
            return spike
        else:
            return self._python_lif(current)

    def _multi_step_forward(self, current: torch.Tensor) -> torch.Tensor:
        T = current.shape[0]
        spikes = []
        for t in range(T):
            spike = self._single_step_forward(current[t])
            spikes.append(spike)
        return torch.stack(spikes, dim=0)

    def _python_lif(self, current: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch fallback."""
        v_temp = self.v * self.decay + current
        spike = (v_temp >= self.v_threshold).float()

        if self.v_reset is None:
            v_next = v_temp - spike * self.v_threshold
        else:
            v_next = spike * self.v_reset + (1.0 - spike) * v_temp

        self.v = v_next.detach()
        return spike

    def extra_repr(self):
        reset_str = f"v_reset={self.v_reset}" if self.v_reset is not None else "soft_reset"
        return (
            f"tau={self.tau}, v_threshold={self.v_threshold}, "
            f"{reset_str}, step_mode={self.step_mode}"
        )