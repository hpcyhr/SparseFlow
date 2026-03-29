"""
SparseFlow Ops/sparse_if.py — SparseIF nn.Module Wrapper

Stateful Integrate-and-Fire neuron module backed by Kernels/ifnode.py.
Same pattern as SparseLIF but without the decay parameter.
"""

import sys
from pathlib import Path
from typing import Dict, Any

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn


class SparseIF(nn.Module):
    """
    Integrate-and-Fire neuron (no leak) with Triton kernel acceleration.

    Maintains internal membrane potential state (self.v).
    Supports single-step [N, *] and multi-step [T, N, *] inputs.
    """

    def __init__(
        self,
        v_threshold: float = 1.0,
        v_reset: float = None,
        step_mode: str = "s",
        return_ms: bool = False,
    ):
        super().__init__()
        self.v_threshold = float(v_threshold)
        self.v_reset = v_reset
        self.step_mode = step_mode
        self.return_ms = return_ms

        self._triton_available = False
        try:
            import triton  # noqa: F401
            self._triton_available = True
        except Exception:
            pass

        self.register_buffer("v", None, persistent=False)
        self._last_sparse_ms = 0.0
        self._last_diag: Dict[str, Any] = {}

    @classmethod
    def from_sj(cls, if_node, return_ms: bool = False, **kwargs):
        """Create from a spikingjelly IFNode."""
        v_th = getattr(if_node, "v_threshold", 1.0)
        v_reset = getattr(if_node, "v_reset", None)
        step_mode = getattr(if_node, "step_mode", "s")

        return cls(
            v_threshold=float(v_th),
            v_reset=float(v_reset) if v_reset is not None else None,
            step_mode=step_mode,
            return_ms=return_ms,
            **kwargs,
        )

    def reset(self):
        self.v = None

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        if self.step_mode == "m" and current.ndim >= 3:
            return self._multi_step_forward(current)
        return self._single_step_forward(current)

    def _single_step_forward(self, current: torch.Tensor) -> torch.Tensor:
        if self.v is None:
            self.v = torch.zeros_like(current)

        if self._triton_available and current.is_cuda:
            from Kernels.ifnode import if_forward

            spike, v_next, ms = if_forward(
                current=current,
                v_prev=self.v,
                v_threshold=self.v_threshold,
                v_reset=self.v_reset,
                return_ms=self.return_ms,
            )
            self.v = v_next.detach()
            self._last_sparse_ms = ms
            return spike
        else:
            return self._python_if(current)

    def _multi_step_forward(self, current: torch.Tensor) -> torch.Tensor:
        T = current.shape[0]
        spikes = []
        for t in range(T):
            spike = self._single_step_forward(current[t])
            spikes.append(spike)
        return torch.stack(spikes, dim=0)

    def _python_if(self, current: torch.Tensor) -> torch.Tensor:
        v_temp = self.v + current
        spike = (v_temp >= self.v_threshold).float()

        if self.v_reset is None:
            v_next = v_temp - spike * self.v_threshold
        else:
            v_next = spike * self.v_reset + (1.0 - spike) * v_temp

        self.v = v_next.detach()
        return spike

    def extra_repr(self):
        reset_str = f"v_reset={self.v_reset}" if self.v_reset is not None else "soft_reset"
        return f"v_threshold={self.v_threshold}, {reset_str}, step_mode={self.step_mode}"