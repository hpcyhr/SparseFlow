"""
SparseLinear — conv-style sparse operator backend for Linear/GEMM.

Key design:
  - Drop-in replacement for nn.Linear
  - No explicit for-T loop in multi-step mode
  - Fold all leading dims into an effective batch N_eff, apply Linear on last dim
  - Runtime dispatch: zero / sparse / dense
  - Buffer reuse + profiling like SparseConv2d
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Kernels.linear import sparse_linear_forward


@dataclass
class _ProfileStats:
    calls: int = 0
    zero_check_ms: float = 0.0
    policy_ms: float = 0.0
    reshape_ms: float = 0.0
    buffer_ms: float = 0.0
    sparse_kernel_ms: float = 0.0
    dense_fallback_ms: float = 0.0
    output_pack_ms: float = 0.0
    total_ms: float = 0.0

    zero_path_hits: int = 0
    sparse_path_hits: int = 0
    dense_path_hits: int = 0
    triton_supported_calls: int = 0

    last_path: str = "none"
    last_avg_active_ratio: float = -1.0
    ema_active_ratio: float = -1.0
    last_total_ms: float = 0.0


class SparseLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        threshold: float = 1e-6,
        return_ms: bool = False,
        dense_threshold: float = 0.25,
        warmup_steps: int = 8,
        calib_every: int = 32,
        ema_decay: float = 0.9,
        zero_streak_needed: int = 2,
        profile_runtime: bool = False,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.threshold = float(threshold)
        self.return_ms = bool(return_ms)

        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)

        # runtime / capability
        self._triton_available = False
        try:
            import triton  # noqa: F401
            self._triton_available = True
        except Exception:
            pass

        # cached transposed weight [Cin, Cout]
        self._w_t = None
        self._w_t_version = -1

        # reusable buffers
        self._counts_buf = None
        self._tile_cin_buf = None

        # last timings
        self._last_sparse_ms = 0.0
        self._last_dense_ms = 0.0

        # routing policy
        self._force_zero = False
        self._force_dense = False
        self._warmup_left = int(max(0, warmup_steps))
        self._calib_every = int(max(1, calib_every))
        self._ema_decay = float(ema_decay)
        self._dense_threshold = float(dense_threshold)
        self._zero_streak_needed = int(max(1, zero_streak_needed))
        self._zero_streak = 0
        self._forward_count = 0
        self._ema_active_ratio: Optional[float] = None
        self._last_avg_active_ratio = -1.0

        # profiling
        self.profile_runtime = bool(profile_runtime)
        self._profile = _ProfileStats()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def from_dense(
        cls,
        linear: nn.Linear,
        threshold: float = 1e-6,
        return_ms: bool = False,
        **kwargs,
    ) -> "SparseLinear":
        mod = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            threshold=threshold,
            return_ms=return_ms,
            **kwargs,
        )
        mod.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            mod.bias.data.copy_(linear.bias.data)
        return mod.to(linear.weight.device)

    # ------------------------------------------------------------------
    # public helpers
    # ------------------------------------------------------------------
    def set_runtime_profiling(self, enabled: bool = True):
        self.profile_runtime = bool(enabled)
        return self

    def reset_runtime_profile(self):
        self._profile = _ProfileStats()

    def get_runtime_profile(self) -> Dict[str, Any]:
        return asdict(self._profile)

    def get_runtime_profile_pretty(self) -> str:
        p = self._profile
        if p.calls == 0:
            return f"SparseLinear({self.in_features}->{self.out_features}) runtime profile: no calls"
        avg = lambda x: x / max(1, p.calls)
        return (
            f"SparseLinear({self.in_features}->{self.out_features})\n"
            f"  calls={p.calls}\n"
            f"  last_path={p.last_path}\n"
            f"  last_avg_active_ratio={p.last_avg_active_ratio:.6f}\n"
            f"  ema_active_ratio={p.ema_active_ratio:.6f}\n"
            f"  zero_path_hits={p.zero_path_hits}\n"
            f"  sparse_path_hits={p.sparse_path_hits}\n"
            f"  dense_path_hits={p.dense_path_hits}\n"
            f"  zero_check_ms={avg(p.zero_check_ms):.4f}\n"
            f"  policy_ms={avg(p.policy_ms):.4f}\n"
            f"  reshape_ms={avg(p.reshape_ms):.4f}\n"
            f"  buffer_ms={avg(p.buffer_ms):.4f}\n"
            f"  sparse_kernel_ms={avg(p.sparse_kernel_ms):.4f}\n"
            f"  dense_fallback_ms={avg(p.dense_fallback_ms):.4f}\n"
            f"  output_pack_ms={avg(p.output_pack_ms):.4f}\n"
            f"  total_ms={avg(p.total_ms):.4f}\n"
        )

    # ------------------------------------------------------------------
    # internal timing
    # ------------------------------------------------------------------
    def _stamp(self) -> float:
        if self.profile_runtime and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def _elapsed_ms(self, t0: float) -> float:
        if self.profile_runtime and torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    def _profile_add(self, field: str, ms: float):
        if self.profile_runtime:
            setattr(self._profile, field, getattr(self._profile, field) + ms)

    # ------------------------------------------------------------------
    # support / cache / buffers
    # ------------------------------------------------------------------
    def _supports_sparse_kernel(self) -> bool:
        return self._triton_available

    def _get_w_t(self) -> torch.Tensor:
        ver = self.weight._version
        if self._w_t is None or self._w_t_version != ver or self._w_t.device != self.weight.device:
            self._w_t = self.weight.data.half().t().contiguous()
            self._w_t_version = ver
        return self._w_t

    def _ensure_buffers(self, x2d: torch.Tensor):
        N, C_IN = x2d.shape
        if N >= 1024:
            block_m = 128
        elif N >= 256:
            block_m = 64
        else:
            block_m = 32

        n_tiles = (N + block_m - 1) // block_m

        if (
            self._counts_buf is None
            or self._counts_buf.numel() < n_tiles
            or self._counts_buf.device != x2d.device
        ):
            self._counts_buf = torch.empty(n_tiles, dtype=torch.int32, device=x2d.device)

        needed_tile_cin = n_tiles * C_IN
        if (
            self._tile_cin_buf is None
            or self._tile_cin_buf.numel() < needed_tile_cin
            or self._tile_cin_buf.device != x2d.device
        ):
            self._tile_cin_buf = torch.empty(needed_tile_cin, dtype=torch.int32, device=x2d.device)

        return self._counts_buf, self._tile_cin_buf

    # ------------------------------------------------------------------
    # policy
    # ------------------------------------------------------------------
    def _maybe_zero_fast_path(self, x2d: torch.Tensor) -> bool:
        return bool(torch.count_nonzero(x2d).item() == 0)

    def _estimate_active_ratio(self, x2d: torch.Tensor) -> float:
        nnz = torch.count_nonzero(x2d).item()
        return float(nnz) / float(max(1, x2d.numel()))

    def _should_collect_ratio(self) -> bool:
        if self._warmup_left > 0:
            return True
        return (self._forward_count % self._calib_every) == 0

    def _update_policy(self, avg_active_ratio: Optional[float]):
        if avg_active_ratio is None:
            return

        self._last_avg_active_ratio = float(avg_active_ratio)
        if self._ema_active_ratio is None:
            self._ema_active_ratio = float(avg_active_ratio)
        else:
            self._ema_active_ratio = (
                self._ema_decay * self._ema_active_ratio
                + (1.0 - self._ema_decay) * float(avg_active_ratio)
            )

        if avg_active_ratio == 0.0:
            self._zero_streak += 1
        else:
            self._zero_streak = 0

        self._force_zero = self._zero_streak >= self._zero_streak_needed
        self._force_dense = (self._ema_active_ratio is not None) and (
            self._ema_active_ratio > self._dense_threshold
        )

        if self._warmup_left > 0:
            self._warmup_left -= 1

        if self.profile_runtime:
            self._profile.last_avg_active_ratio = self._last_avg_active_ratio
            self._profile.ema_active_ratio = -1.0 if self._ema_active_ratio is None else float(self._ema_active_ratio)

    # ------------------------------------------------------------------
    # folded execution
    # ------------------------------------------------------------------
    def _dense_fallback(self, x2d: torch.Tensor) -> torch.Tensor:
        y = F.linear(x2d.float(), self.weight.float(), None if self.bias is None else self.bias.float())
        return y

    def _zero_output(self, x2d: torch.Tensor) -> torch.Tensor:
        y = torch.zeros(x2d.shape[0], self.out_features, dtype=torch.float32, device=x2d.device)
        if self.bias is not None:
            y += self.bias.detach().to(dtype=torch.float32, device=x2d.device).view(1, -1)
        self._last_sparse_ms = 0.0
        self._last_dense_ms = 0.0
        return y

    def _forward_folded_2d(self, x2d: torch.Tensor) -> torch.Tensor:
        if self.profile_runtime:
            self._profile.calls += 1
            t_total = self._stamp()
        else:
            t_total = 0.0

        # zero check
        if self.profile_runtime:
            t0 = self._stamp()
        is_all_zero = self._force_zero or self._maybe_zero_fast_path(x2d)
        if self.profile_runtime:
            self._profile_add("zero_check_ms", self._elapsed_ms(t0))

        if is_all_zero:
            if self.profile_runtime:
                self._profile.zero_path_hits += 1
                self._profile.last_path = "zero"
            y = self._zero_output(x2d)
            if self.profile_runtime:
                total_ms = self._elapsed_ms(t_total)
                self._profile.total_ms += total_ms
                self._profile.last_total_ms = total_ms
            return y

        # policy
        if self.profile_runtime:
            t0 = self._stamp()
        need_ratio = self._should_collect_ratio()
        avg_active_ratio = self._estimate_active_ratio(x2d) if need_ratio else self._last_avg_active_ratio
        self._update_policy(avg_active_ratio)
        use_sparse = self._supports_sparse_kernel() and not self._force_dense
        if self.profile_runtime:
            self._profile_add("policy_ms", self._elapsed_ms(t0))
            if self._supports_sparse_kernel():
                self._profile.triton_supported_calls += 1

        # sparse path
        if use_sparse:
            if self.profile_runtime:
                t0 = self._stamp()
            counts_buf, tile_cin_buf = self._ensure_buffers(x2d)
            if self.profile_runtime:
                self._profile_add("buffer_ms", self._elapsed_ms(t0))

            if self.profile_runtime:
                t0 = self._stamp()
            y, sparse_ms = sparse_linear_forward(
                x=x2d,
                weight=self.weight,
                bias=self.bias,
                threshold=self.threshold,
                w_t=self._get_w_t(),
                counts_buf=counts_buf,
                tile_cin_buf=tile_cin_buf,
                return_ms=self.return_ms,
            )
            self._last_sparse_ms = float(sparse_ms)
            self._last_dense_ms = 0.0
            if self.profile_runtime:
                # include full sparse kernel path wall time, not only CUDA event ms
                self._profile_add("sparse_kernel_ms", self._elapsed_ms(t0))
                self._profile.sparse_path_hits += 1
                self._profile.last_path = "sparse"
        else:
            # dense fallback
            if self.profile_runtime:
                t0 = self._stamp()
            y = self._dense_fallback(x2d)
            self._last_sparse_ms = 0.0
            self._last_dense_ms = 0.0
            if self.profile_runtime:
                self._profile_add("dense_fallback_ms", self._elapsed_ms(t0))
                self._profile.dense_path_hits += 1
                self._profile.last_path = "dense"

        if self.profile_runtime:
            total_ms = self._elapsed_ms(t_total)
            self._profile.total_ms += total_ms
            self._profile.last_total_ms = total_ms

        return y

    # ------------------------------------------------------------------
    # public forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Supported:
          - 2D: [B, Cin]
          - 3D: [T, B, Cin]
          - generic >=2D: [..., Cin]
        We always apply Linear on the last dimension and fold all prefix dims.
        """
        if x.dim() < 2:
            raise ValueError(f"Expected input dim >= 2, got shape {tuple(x.shape)}")
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dim == in_features ({self.in_features}), got {x.shape[-1]}"
            )

        if self.profile_runtime:
            t0 = self._stamp()

        prefix_shape = tuple(x.shape[:-1])
        x2d = x.reshape(-1, self.in_features)

        if self.profile_runtime:
            self._profile_add("reshape_ms", self._elapsed_ms(t0))

        y2d = self._forward_folded_2d(x2d)

        if self.profile_runtime:
            t0 = self._stamp()

        y = y2d.reshape(*prefix_shape, self.out_features)

        if self.profile_runtime:
            self._profile_add("output_pack_ms", self._elapsed_ms(t0))

        return y
