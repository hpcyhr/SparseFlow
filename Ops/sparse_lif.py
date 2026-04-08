"""
Ops/sparse_lif.py

Standalone SparseLIF wrapper backed by Kernels/lif.py.

Note:
- This module supports single-step and multi-step tensors.
- Multi-step mode currently uses a Python loop over time steps.
- The kernel implements a fixed update form: v_next = v_prev * decay + current.
"""

import sys
import warnings
from pathlib import Path
from typing import Any, Dict

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn


class SparseLIF(nn.Module):
    def __init__(
        self,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float = None,
        step_mode: str = "s",
        decay_input: bool = False,
        return_ms: bool = False,
    ):
        super().__init__()
        self.tau = float(tau)
        self.decay = 1.0 / self.tau
        self.v_threshold = float(v_threshold)
        self.v_reset = v_reset
        self.step_mode = str(step_mode)
        self.decay_input = bool(decay_input)
        self.return_ms = bool(return_ms)

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
    def from_sj(cls, lif_node, return_ms: bool = False, **kwargs):
        tau = getattr(lif_node, "tau", 2.0)
        v_th = getattr(lif_node, "v_threshold", 1.0)
        v_reset = getattr(lif_node, "v_reset", None)
        step_mode = getattr(lif_node, "step_mode", "s")
        decay_input = bool(getattr(lif_node, "decay_input", False))
        if decay_input:
            warnings.warn(
                "[SparseFlow] SparseLIF does not fully reproduce spikingjelly decay_input=True dynamics; using fixed v*decay + I update.",
                UserWarning,
            )
        return cls(
            tau=float(tau),
            v_threshold=float(v_th),
            v_reset=float(v_reset) if v_reset is not None else None,
            step_mode=step_mode,
            decay_input=decay_input,
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
            self._last_sparse_ms = float(ms)
            return spike

        return self._python_lif(current)

    def _multi_step_forward(self, current: torch.Tensor) -> torch.Tensor:
        t = current.shape[0]
        spikes = []
        for idx in range(t):
            spikes.append(self._single_step_forward(current[idx]))
        return torch.stack(spikes, dim=0)

    def _python_lif(self, current: torch.Tensor) -> torch.Tensor:
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
            f"tau={self.tau}, v_threshold={self.v_threshold}, {reset_str}, "
            f"step_mode={self.step_mode}, decay_input={self.decay_input}"
        )

