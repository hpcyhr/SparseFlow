import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Any, Dict

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class _ProfileStats:
    calls: int = 0
    zero_path_hits: int = 0
    sparse_path_hits: int = 0
    dense_path_hits: int = 0
    triton_supported_calls: int = 0

    total_ms: float = 0.0
    reshape_ms: float = 0.0
    policy_ms: float = 0.0
    buffer_ms: float = 0.0
    sparse_kernel_ms: float = 0.0
    dense_fallback_ms: float = 0.0
    output_pack_ms: float = 0.0

    last_path: str = "none"
    last_total_ms: float = 0.0
    last_reshape_ms: float = 0.0
    last_policy_ms: float = 0.0
    last_buffer_ms: float = 0.0
    last_sparse_kernel_ms: float = 0.0
    last_dense_fallback_ms: float = 0.0
    last_output_pack_ms: float = 0.0
    last_avg_active_ratio: float = -1.0


class SparseLinear(nn.Module):
    """
    Grouped tile-level Dynamic-K sparse Linear.

    Mirrors SparseConv2d style:
      - grouped bitmask metadata
      - three-stage prescan
      - optional runtime profiling
      - EMA-based dense fallback / zero promotion
      - inference_mode to disable periodic calibration syncs
      - backend metadata propagation from kernel entry
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        threshold: float = 1e-6,
        return_ms: bool = False,
        dense_threshold: float = 0.85,
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

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            bound = 1.0 / max(float(in_features), 1.0) ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

        self._last_sparse_ms = 0.0
        self._last_dense_ms = 0.0
        self._last_avg_active_ratio = -1.0

        self._triton_available = False
        try:
            import triton  # noqa: F401
            self._triton_available = True
        except Exception:
            self._triton_available = False
        self._warned_triton_runtime_error = False

        self._w_t = None
        self._w_t_version = -1
        self._ag_mask_buf = None
        self._tile_class_buf = None

        # policy state
        self._warmup_left = int(max(0, warmup_steps))
        self._calib_every = int(max(1, calib_every))
        self._ema_decay = float(ema_decay)
        self._dense_threshold = float(dense_threshold)
        self._zero_streak_needed = int(max(1, zero_streak_needed))
        self._zero_streak = 0
        self._forward_count = 0
        self._ema_active_ratio: Optional[float] = None
        self._force_zero = False
        self._force_dense = False
        self._inference_mode = False

        # diagnostics
        self.collect_diag = False
        self._last_diag: Dict[str, Any] = {}
        self.profile_runtime = bool(profile_runtime)
        self._profile = _ProfileStats()
        self.backend_family = "sparse_kernel"
        self.diag_path = "linear_three_stage"
        self.fallback_reason = ""
        self.meta_source = "measured"
        self.diag_source = "measured"
        self.support_status = "supported"
        self.score_family = "linear"
        self._runtime_dispatch_seen = 0
        self._runtime_dispatch_mismatch = 0

    @classmethod
    def from_dense(
        cls,
        linear: nn.Linear,
        threshold: float = 1e-6,
        return_ms: bool = False,
        **kwargs,
    ) -> "SparseLinear":
        sparse = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            threshold=threshold,
            return_ms=return_ms,
            **kwargs,
        )
        sparse = sparse.to(linear.weight.device)
        sparse.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            sparse.bias.data.copy_(linear.bias.data)
        return sparse

    def set_inference_mode(self, enabled: bool):
        """
        Disable periodic ratio calibration during timed runs.
        Mirrors SparseConv2d behavior.
        """
        self._inference_mode = bool(enabled)
        if enabled:
            self.collect_diag = False
            self.profile_runtime = False

    def _record_runtime_dispatch(self, runtime_backend: str):
        expected = str(
            getattr(
                self,
                "sf_dispatch_decision",
                getattr(self, "sf_backend_mode", ""),
            )
        ).strip().lower()
        if not expected:
            return
        runtime = runtime_backend.strip().lower()
        self._runtime_dispatch_seen += 1
        if expected != runtime:
            self._runtime_dispatch_mismatch += 1

    def get_profile_summary(self) -> str:
        p = self._profile
        if p.calls == 0:
            return "SparseLinear runtime profile: no calls"

        def avg(x: float) -> float:
            return x / max(1, p.calls)

        return (
            f"SparseLinear runtime profile\n"
            f"  calls={p.calls}, last_path={p.last_path}\n"
            f"  hits: zero={p.zero_path_hits}, sparse={p.sparse_path_hits}, dense={p.dense_path_hits}\n"
            f"  avg total={avg(p.total_ms):.4f} ms, last total={p.last_total_ms:.4f} ms\n"
            f"  avg reshape={avg(p.reshape_ms):.4f} ms\n"
            f"  avg buffer={avg(p.buffer_ms):.4f} ms\n"
            f"  avg sparse_kernel={avg(p.sparse_kernel_ms):.4f} ms\n"
            f"  avg dense_fallback={avg(p.dense_fallback_ms):.4f} ms\n"
            f"  last avg_active_ratio={p.last_avg_active_ratio:.6f}"
        )

    # ------------------------------------------------------------------
    # timing helpers
    # ------------------------------------------------------------------
    def _stamp(self) -> float:
        if self.profile_runtime and self.weight.is_cuda:
            torch.cuda.synchronize(self.weight.device)
        return time.perf_counter()

    def _elapsed_ms(self, t0: float) -> float:
        if self.profile_runtime and self.weight.is_cuda:
            torch.cuda.synchronize(self.weight.device)
        return (time.perf_counter() - t0) * 1000.0

    def _profile_add(self, field: str, ms: float):
        if self.profile_runtime:
            setattr(self._profile, field, getattr(self._profile, field) + ms)

    def _profile_set_last(self, field: str, ms: float):
        if self.profile_runtime:
            setattr(self._profile, field, ms)

    # ------------------------------------------------------------------
    # capability / layout helpers
    # ------------------------------------------------------------------
    def _supports_triton(self) -> bool:
        return self._triton_available

    def _supports_sparse(self) -> bool:
        return self.in_features > 0 and self.out_features > 0

    def _get_w_t(self) -> torch.Tensor:
        ver = self.weight._version
        if self._w_t is None or self._w_t_version != ver:
            # Kernels/linear.py expects weight transposed to [Cin, Cout]
            self._w_t = self.weight.data.half().t().contiguous()
            self._w_t_version = ver
        return self._w_t

    def _flatten_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        if x.dim() < 2:
            raise ValueError(f"Expected at least 2D input, got {x.dim()}D")

        # generic last-dim linear
        if x.shape[-1] == self.in_features:
            return x.reshape(-1, self.in_features), ("lastdim", x.shape[:-1])

        # optional compatibility for [T,B,C,H,W] flattened to in_features
        if x.dim() == 5 and (x.shape[2] * x.shape[3] * x.shape[4] == self.in_features):
            t, b, c, h, w = x.shape
            return x.reshape(t * b, c * h * w), ("tbchw", t, b)

        raise ValueError(
            f"Input shape {tuple(x.shape)} is incompatible with in_features={self.in_features}. "
            f"Need x.shape[-1] == in_features, or for 5D input C*H*W == in_features."
        )

    def _restore_output(self, y2d: torch.Tensor, meta: Tuple) -> torch.Tensor:
        mode = meta[0]
        if mode == "lastdim":
            prefix_shape = meta[1]
            return y2d.reshape(*prefix_shape, self.out_features)
        if mode == "tbchw":
            t, b = meta[1], meta[2]
            return y2d.reshape(t, b, self.out_features)
        raise RuntimeError(f"Unknown restore mode: {mode}")

    def _ensure_buffers(self, x2d: torch.Tensor):
        from Kernels.linear import _select_linear_block_m
        import triton

        n_rows = x2d.shape[0]
        block_m = _select_linear_block_m(n_rows)
        n_tiles = triton.cdiv(n_rows, block_m)

        if (
            self._ag_mask_buf is None
            or self._ag_mask_buf.numel() < n_tiles
            or self._ag_mask_buf.device != x2d.device
        ):
            self._ag_mask_buf = torch.empty(n_tiles, dtype=torch.int32, device=x2d.device)

        if (
            self._tile_class_buf is None
            or self._tile_class_buf.numel() < n_tiles
            or self._tile_class_buf.device != x2d.device
        ):
            self._tile_class_buf = torch.empty(n_tiles, dtype=torch.int32, device=x2d.device)

        return self._ag_mask_buf, self._tile_class_buf

    def _zero_output_2d(self, x2d: torch.Tensor) -> torch.Tensor:
        y = torch.zeros(
            x2d.shape[0], self.out_features,
            dtype=torch.float32, device=x2d.device
        )
        if self.bias is not None:
            y = y + self.bias.detach().float().view(1, -1)
        return y

    # ------------------------------------------------------------------
    # policy helpers
    # ------------------------------------------------------------------
    def _should_collect_ratio(self) -> bool:
        if self._inference_mode:
            return False
        if self._warmup_left > 0:
            return True
        return (self._forward_count % self._calib_every) == 0

    def _update_policy(self, avg_active_ratio: Optional[float]):
        if avg_active_ratio is None:
            return

        self._last_avg_active_ratio = avg_active_ratio
        if self._warmup_left > 0:
            self._warmup_left -= 1
            self._ema_active_ratio = avg_active_ratio
            return

        if self._ema_active_ratio is None:
            self._ema_active_ratio = avg_active_ratio
        else:
            self._ema_active_ratio = (
                self._ema_decay * self._ema_active_ratio
                + (1.0 - self._ema_decay) * avg_active_ratio
            )

        if avg_active_ratio == 0.0:
            self._zero_streak += 1
        else:
            self._zero_streak = 0
            self._force_zero = False

        if self._zero_streak >= self._zero_streak_needed:
            self._force_zero = True

        if self._ema_active_ratio is not None and self._ema_active_ratio > self._dense_threshold:
            self._force_dense = True
        else:
            self._force_dense = False

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.fallback_reason = ""
        if self.profile_runtime:
            self._profile.calls += 1
            t_total = self._stamp()

        if self.collect_diag:
            self._last_diag = {"sparse_path_executed": False}

        if self.profile_runtime:
            t0 = self._stamp()
        x2d, restore_meta = self._flatten_input(x)
        if self.profile_runtime:
            ms = self._elapsed_ms(t0)
            self._profile_add("reshape_ms", ms)
            self._profile_set_last("last_reshape_ms", ms)

        if self._force_zero:
            y2d = self._zero_output_2d(x2d)
            self._record_runtime_dispatch("staticzero")
            self.fallback_reason = "force_zero"
            if self.profile_runtime:
                self._profile.zero_path_hits += 1
                self._profile.last_path = "zero(force)"
                total_ms = self._elapsed_ms(t_total)
                self._profile_add("total_ms", total_ms)
                self._profile_set_last("last_total_ms", total_ms)
            return self._restore_output(y2d, restore_meta)

        use_triton = self._supports_triton() and x2d.is_cuda
        if self.profile_runtime and use_triton:
            self._profile.triton_supported_calls += 1

        if self.profile_runtime:
            t0 = self._stamp()
        need_ratio = use_triton and self._should_collect_ratio() and (not self._force_dense)
        if self.profile_runtime:
            ms = self._elapsed_ms(t0)
            self._profile_add("policy_ms", ms)
            self._profile_set_last("last_policy_ms", ms)

        self._forward_count += 1

        if use_triton and not self._force_dense and self._supports_sparse():
            y2d, avg_active_ratio, backend_kind = self._triton_forward(
                x2d, need_ratio=need_ratio
            )
            if backend_kind == "dense_fallback":
                path = "dense(fallback)"
                self._record_runtime_dispatch("dense")
                self.fallback_reason = "kernel_dense_fallback"
                if self.profile_runtime:
                    self._profile.dense_path_hits += 1
            elif backend_kind == "zero_tiles_only":
                path = "zero(tile_compaction)"
                self._record_runtime_dispatch("staticzero")
                if self.profile_runtime:
                    self._profile.zero_path_hits += 1
            else:
                path = "sparse"
                self._record_runtime_dispatch("sparse")
                if self.profile_runtime:
                    self._profile.sparse_path_hits += 1
        else:
            y2d = self._fallback_forward(x2d)
            avg_active_ratio = None
            path = "dense"
            self._record_runtime_dispatch("dense")
            self.fallback_reason = "module_dense_path"
            if self.profile_runtime:
                self._profile.dense_path_hits += 1

        if self.profile_runtime:
            self._profile.last_avg_active_ratio = self._last_avg_active_ratio

        if self.profile_runtime:
            t0 = self._stamp()
        self._update_policy(avg_active_ratio)
        if self.profile_runtime:
            ms = self._elapsed_ms(t0)
            self._profile_add("policy_ms", ms)
            self._profile_set_last("last_policy_ms", self._profile.last_policy_ms + ms)

        if self._force_zero and avg_active_ratio == 0.0:
            y2d = self._zero_output_2d(x2d)
            path = "zero(promoted)"
            self._record_runtime_dispatch("staticzero")
            self.fallback_reason = "zero_promoted"

        if self.profile_runtime:
            t0 = self._stamp()
        y = self._restore_output(y2d, restore_meta)
        if self.profile_runtime:
            ms = self._elapsed_ms(t0)
            self._profile_add("output_pack_ms", ms)
            self._profile_set_last("last_output_pack_ms", ms)
            self._profile.last_path = path
            total_ms = self._elapsed_ms(t_total)
            self._profile_add("total_ms", total_ms)
            self._profile_set_last("last_total_ms", total_ms)
        if self.collect_diag:
            self._last_diag["runtime_dispatch_seen"] = int(self._runtime_dispatch_seen)
            self._last_diag["runtime_dispatch_mismatch"] = int(self._runtime_dispatch_mismatch)
            self._last_diag["fallback_reason"] = str(self.fallback_reason)
        return y

    # ------------------------------------------------------------------
    def _triton_forward(self, x2d: torch.Tensor, need_ratio: bool = False):
        """
        Core Triton sparse linear path.

        Mirrors SparseConv2d convention:
          return (y, ms) + optional ratio + optional tile_stats + backend_meta
        """
        from Kernels.linear import sparse_linear_forward

        if self.profile_runtime:
            t0 = self._stamp()

        w_t = self._get_w_t()
        ag_mask_buf, tile_class_buf = self._ensure_buffers(x2d)
        x_f16 = x2d if (x2d.dtype == torch.float16 and x2d.is_contiguous()) else x2d.half().contiguous()

        if self.profile_runtime:
            ms = self._elapsed_ms(t0)
            self._profile_add("buffer_ms", ms)
            self._profile_set_last("last_buffer_ms", ms)

        collect_tiles = self.collect_diag
        want_ratio = need_ratio or collect_tiles
        want_tiles = collect_tiles

        try:
            result = sparse_linear_forward(
                x=x_f16,
                weight=self.weight,
                bias=self.bias,
                threshold=self.threshold,
                w_t=w_t,
                ag_mask_buf=ag_mask_buf,
                tile_class_buf=tile_class_buf,
                return_ms=self.return_ms,
                return_avg_active_ratio=want_ratio,
                return_tile_stats=want_tiles,
                return_backend_meta=True,
            )
        except Exception as err:
            # Robust fallback for Triton compile/autotune/runtime failures.
            if not self._warned_triton_runtime_error:
                warnings.warn(
                    "[SparseLinear] Triton sparse path failed, fallback to dense. "
                    f"error={type(err).__name__}: {err}"
                )
                self._warned_triton_runtime_error = True
            y2d = self._fallback_forward(x2d)
            if self.collect_diag:
                self._last_diag = {
                    "sparse_path_executed": False,
                    "backend": "dense_fallback",
                    "backend_reason": "triton_runtime_error",
                    "error_type": type(err).__name__,
                    "error_msg": str(err),
                    "backend_family": self.backend_family,
                    "diag_path": self.diag_path,
                    "fallback_reason": "triton_runtime_error",
                }
            return y2d, None, "dense_fallback"

        if not isinstance(result, tuple) or len(result) < 2:
            raise TypeError(f"sparse_linear_forward bad return: {type(result)}")

        idx = 0
        y2d = result[idx]
        idx += 1

        sparse_ms = result[idx]
        idx += 1

        avg_active_ratio = None
        if want_ratio and idx < len(result):
            avg_active_ratio = result[idx]
            idx += 1

        tile_stats = None
        if want_tiles and idx < len(result):
            tile_stats = result[idx]
            idx += 1

        backend_meta = result[idx] if idx < len(result) else {
            "backend": "sparse_triton",
            "reason": "no_meta",
        }

        backend_kind = backend_meta.get("backend", "sparse_triton") if backend_meta else "sparse_triton"

        if backend_kind == "dense_fallback":
            self._last_sparse_ms = 0.0
            self._last_dense_ms = float(sparse_ms)
            if self.profile_runtime:
                self._profile_add("dense_fallback_ms", self._last_dense_ms)
                self._profile_set_last("last_dense_fallback_ms", self._last_dense_ms)
                self._profile_set_last("last_sparse_kernel_ms", 0.0)
        else:
            self._last_sparse_ms = float(sparse_ms)
            self._last_dense_ms = 0.0
            if self.profile_runtime:
                self._profile_add("sparse_kernel_ms", self._last_sparse_ms)
                self._profile_set_last("last_sparse_kernel_ms", self._last_sparse_ms)
                self._profile_set_last("last_dense_fallback_ms", 0.0)

        if self.collect_diag:
            self._collect_tile_group_diag(
                x_f16,
                ag_mask_buf,
                tile_class_buf,
                sparse_ms,
                avg_active_ratio,
                tile_stats,
                backend_meta=backend_meta,
            )

        return y2d, avg_active_ratio, backend_kind

    def _collect_tile_group_diag(
        self,
        x2d: torch.Tensor,
        ag_mask_buf: torch.Tensor,
        tile_class_buf: torch.Tensor,
        sparse_ms: float,
        avg_active_ratio: Optional[float],
        tile_stats,
        backend_meta: Optional[Dict[str, Any]] = None,
    ):
        """
        Only called when collect_diag is True.
        May introduce GPU->CPU syncs, so do not enable during perf timing.
        """
        from Kernels.linear import (
            _select_linear_block_m,
            choose_group_size,
            _popcount_buf,
            TILE_ZERO,
            TILE_SPARSE,
            TILE_DENSEISH,
        )
        import triton

        n_rows, c_in = x2d.shape
        group_size = choose_group_size(c_in)
        num_groups = triton.cdiv(c_in, group_size)
        block_m = _select_linear_block_m(n_rows)
        n_tiles = triton.cdiv(n_rows, block_m)

        if tile_stats is not None:
            zt = tile_stats.get("zero_tiles", -1)
            st = tile_stats.get("sparse_tiles", -1)
            dt = tile_stats.get("denseish_tiles", -1)
        else:
            tc = tile_class_buf[:n_tiles].cpu()
            zt = int((tc == TILE_ZERO).sum().item())
            st = int((tc == TILE_SPARSE).sum().item())
            dt = int((tc == TILE_DENSEISH).sum().item())

        pc = _popcount_buf(ag_mask_buf, n_tiles)
        total_g = float(n_tiles * num_groups)
        active_g = float(pc.sum().item())
        agr = active_g / max(total_g, 1.0)

        self._last_diag = {
            "sparse_path_executed": True,
            "metadata_kind": "three_stage_grouped_linear_v3",
            "group_size": group_size,
            "num_groups": int(num_groups),
            "nonzero_group_count": active_g,
            "total_group_count": total_g,
            "active_group_ratio": agr,
            "tile_zero_count": int(zt),
            "total_tile_count": int(n_tiles),
            "tile_zero_ratio": zt / max(int(n_tiles), 1),
            "effective_k_ratio": agr,
            "sparse_compute_ms": float(sparse_ms),
            "sparse_total_ms": float(sparse_ms),
            "avg_active_ratio": float(avg_active_ratio) if avg_active_ratio is not None else -1.0,
            "zero_tiles": int(zt),
            "sparse_tiles": int(st),
            "denseish_tiles": int(dt),
            "prescan_mode": "three_stage_grouped_linear_v3",
            "kernel_type": "linear",
            "block_m": int(block_m),
        }

        if backend_meta is not None:
            self._last_diag["backend"] = backend_meta.get("backend", "unknown")
            self._last_diag["backend_reason"] = backend_meta.get("reason", "unknown")
        self._last_diag["backend_family"] = self.backend_family
        self._last_diag["diag_path"] = self.diag_path
        self._last_diag["fallback_reason"] = self.fallback_reason

        if tile_stats is not None:
            for key in (
                "stage1_zero_candidate",
                "stage1_denseish",
                "stage1_uncertain",
                "final_zero",
                "final_sparse",
                "final_denseish",
                "stage2_zero_refine_tiles",
                "stage2_uncertain_tiles",
            ):
                if key in tile_stats:
                    self._last_diag[key] = tile_stats[key]

    def _fallback_forward(self, x2d: torch.Tensor) -> torch.Tensor:
        if self.profile_runtime:
            t0 = self._stamp()

        # match SparseConv2d: fallback uses float path
        y = F.linear(x2d.float(), self.weight.float(), self.bias.float() if self.bias is not None else None).float()
        self._last_sparse_ms = 0.0

        if self.profile_runtime:
            self._last_dense_ms = self._elapsed_ms(t0)
            self._profile_add("dense_fallback_ms", self._last_dense_ms)
            self._profile_set_last("last_dense_fallback_ms", self._last_dense_ms)
            self._profile_set_last("last_sparse_kernel_ms", 0.0)
        else:
            self._last_dense_ms = 0.0

        return y

    def extra_repr(self) -> str:
        return (
            f"{self.in_features}, {self.out_features}, bias={self.bias is not None}, "
            f"sparse=True, dense_threshold={self._dense_threshold}, "
            f"warmup_left={self._warmup_left}, calib_every={self._calib_every}, "
            f"profile_runtime={self.profile_runtime}, inference_mode={self._inference_mode}"
        )
