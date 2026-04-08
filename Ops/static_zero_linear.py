import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class StaticZeroLinear(nn.Module):
    """
    Exact-zero linear backend.

    For x == 0, Linear(x) is exactly:
        y = bias
    (or zeros when bias is None).

    This module is a cheap drop-in replacement when a linear layer receives
    exact-zero input in measured runs.
    """

    def __init__(self, out_features: int, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.out_features = int(out_features)
        if bias is not None:
            self.register_buffer("bias_buf", bias.detach().float().clone())
        else:
            self.bias_buf = None
        self._forward_count = 0
        self.collect_diag = False
        self.profile_runtime = False
        self._last_diag: Dict[str, Any] = {}
        self._last_sparse_ms = 0.0
        self.backend_family = "exact_zero"
        self.diag_path = "staticzero_linear"
        self.fallback_reason = "exact_zero_shortcut"

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "StaticZeroLinear":
        return cls(out_features=linear.out_features, bias=linear.bias).to(linear.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            raise ValueError(f"StaticZeroLinear expects dim>=2 input, got {x.dim()}D")
        self._forward_count += 1

        out_shape = (*x.shape[:-1], self.out_features)
        y = torch.zeros(out_shape, dtype=torch.float32, device=x.device)
        if self.bias_buf is not None:
            b = self.bias_buf.to(device=x.device, dtype=torch.float32)
            view_shape = [1] * (y.dim() - 1) + [self.out_features]
            y = y + b.view(*view_shape)
        if self.collect_diag:
            self._last_diag = self.get_diag()
        return y

    def get_diag(self) -> Dict[str, Any]:
        return {
            "backend_family": "exact_zero",
            "zero_reason": "static_zero_input_linear",
            "diag_path": "staticzero_linear",
            "fallback_reason": self.fallback_reason,
            "has_bias": self.bias_buf is not None,
            "out_features": self.out_features,
            "forward_count": self._forward_count,
            "sparse_path_executed": False,
            "sparse_total_ms": 0.0,
        }

    def extra_repr(self) -> str:
        return f"out_features={self.out_features}, has_bias={self.bias_buf is not None}"
