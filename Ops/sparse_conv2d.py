"""
SparseConv2d — v25 Sync-gated + A/B tile launch + inference_mode

Changes from v24:
  [P0] _triton_forward only requests ratio during warmup/calibration,
       and passes this to sparse_conv2d_forward where syncs are gated.
  [P1] launch_all_tiles parameter — A/B switch for tile launch strategy.
       Mode A (False): existing active-tile-ID launch (1 sync from nonzero).
       Mode B (True): launch all tiles, zero tiles early-return (0 syncs).
  [P2] inference_mode flag — when set, _should_collect_ratio always returns
       False, preventing periodic calibration syncs during timed runs.
  Updated diagnostics metadata_kind: "three_stage_nhwc_v25"
"""

from Ops.static_zero_conv2d import make_zero_conv_output, ZERO_BACKEND_FAMILY
import torch.nn.functional as F
import torch.nn as nn
import torch
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import time
import math
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@dataclass
class _ProfileStats:
    calls: int = 0
    zero_path_hits: int = 0
    sparse_path_hits: int = 0
    dense_path_hits: int = 0
    triton_supported_calls: int = 0

    total_ms: float = 0.0
    policy_ms: float = 0.0
    reshape_ms: float = 0.0
    buffer_ms: float = 0.0
    sparse_kernel_ms: float = 0.0
    dense_fallback_ms: float = 0.0
    output_pack_ms: float = 0.0

    last_path: str = "none"
    last_total_ms: float = 0.0
    last_policy_ms: float = 0.0
    last_reshape_ms: float = 0.0
    last_buffer_ms: float = 0.0
    last_sparse_kernel_ms: float = 0.0
    last_dense_fallback_ms: float = 0.0
    last_output_pack_ms: float = 0.0
    last_avg_active_ratio: float = -1.0


class SparseConv2d(nn.Module):
    """SparseConv2d with configurable tile launch and inference mode.

    New parameters (v25):
        launch_all_tiles: bool
            Mode A (False, default): build active tile IDs, launch only active.
            Mode B (True): launch all tiles, zero tiles early-return in kernel.
            Mode B eliminates the nonzero() sync in _build_active_tile_ids.
            Use set_launch_all_tiles() or the constructor to configure.

    New methods (v25):
        set_inference_mode(enabled): disable periodic calibration for timing.
        set_launch_all_tiles(enabled): switch tile launch mode at runtime.
    """

    def __init__(
        self,
        in_channels, out_channels, kernel_size,
        stride=1, padding=0, dilation=1, groups=1, bias=True,
        block_size=None, threshold=1e-6, return_ms=False,
        dense_threshold: float = 0.85,
        warmup_steps: int = 8, calib_every: int = 32,
        ema_decay: float = 0.9, zero_streak_needed: int = 2,
        profile_runtime: bool = False,
        allow_sparse_1x1: bool = True,
        launch_all_tiles: bool = False,    # NEW [P1]: A/B switch
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.dilation = dilation
        self.groups = groups
        self.block_size = block_size
        self.threshold = threshold
        self.return_ms = return_ms
        self.allow_sparse_1x1 = bool(allow_sparse_1x1)
        self.launch_all_tiles = bool(launch_all_tiles)    # NEW [P1]

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups,
                                                *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # Internal state
        self._w_cl = None
        self._w_cl_version = -1
        self._ag_mask_buf = None
        self._tile_class_buf = None
        self._active_tile_ids_buf = None
        self._x_nhwc_buf = None
        self._x_nhwc_shape = None
        self._zero_template_cache = {}
        self._last_sparse_ms = 0.0
        self._last_dense_ms = 0.0
        self._last_avg_active_ratio = -1.0

        # Policy state
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
        self._inference_mode = False    # NEW [P2]: disable calibration for timing

        # Diagnostics (off by default)
        self.collect_diag = False
        self._last_diag = {}
        self.profile_runtime = profile_runtime
        self._profile = _ProfileStats()
        self.backend_family = "sparse_kernel"
        self.diag_path = "conv2d_three_stage"
        self.fallback_reason = ""
        self.meta_source = "measured"
        self.diag_source = "measured"
        self.support_status = "supported"
        self.score_family = "conv"
        self._runtime_dispatch_seen = 0
        self._runtime_dispatch_mismatch = 0

    # ----------------------------------------------------------------
    # NEW [P1]: runtime A/B switch
    # ----------------------------------------------------------------
    def set_launch_all_tiles(self, enabled: bool):
        """Switch between Mode A (active-tile-ID launch) and Mode B (launch-all)."""
        self.launch_all_tiles = bool(enabled)

    # ----------------------------------------------------------------
    # NEW [P2]: inference mode
    # ----------------------------------------------------------------
    def set_inference_mode(self, enabled: bool):
        """When enabled, _should_collect_ratio() always returns False.

        This prevents periodic calibration syncs during timed runs.
        Also clears collect_diag and profile_runtime for clean timing.
        """
        self._inference_mode = bool(enabled)
        if enabled:
            self.collect_diag = False
            self.profile_runtime = False

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    def _stamp(self):
        torch.cuda.synchronize(self.weight.device)
        return time.perf_counter()

    def _elapsed_ms(self, t0):
        torch.cuda.synchronize(self.weight.device)
        return (time.perf_counter() - t0) * 1000.0

    def _profile_add(self, key, val):
        setattr(self._profile, key, getattr(self._profile, key) + val)

    def _profile_set_last(self, key, val):
        setattr(self._profile, key, val)

    def get_profile_summary(self):
        p = self._profile
        if p.calls == 0:
            return "SparseConv2d runtime profile: no calls"
        def avg(x): return x / max(1, p.calls)
        return (
            f"SparseConv2d runtime profile\n"
            f"  calls={p.calls}, last_path={p.last_path}\n"
            f"  hits: zero={p.zero_path_hits}, sparse={p.sparse_path_hits}, "
            f"dense={p.dense_path_hits}, triton_ok={p.triton_supported_calls}\n"
            f"  avg: total={avg(p.total_ms):.3f}, reshape={avg(p.reshape_ms):.3f}, "
            f"buffer={avg(p.buffer_ms):.3f}, sparse_kernel={avg(p.sparse_kernel_ms):.3f}, "
            f"dense_fallback={avg(p.dense_fallback_ms):.3f}, output_pack={avg(p.output_pack_ms):.3f}\n"
            f"  last: total={p.last_total_ms:.3f}, path={p.last_path}, "
            f"active_ratio={p.last_avg_active_ratio:.4f}"
        )

    # ----------------------------------------------------------------
    # Supports / layout helpers
    # ----------------------------------------------------------------
    def _supports_triton(self):
        try:
            import triton  # noqa: F401
            return True
        except ImportError:
            return False

    def _supports_sparse(self):
        k = self.kernel_size
        s = self.stride
        p = self.padding
        if k == (1, 1):
            return self.allow_sparse_1x1 and s == (1, 1) and p == (0, 0)
        return (
            (k == (3, 3) and s == (1, 1) and p == (1, 1))
            or (k == (3, 3) and s == (2, 2) and p == (1, 1))
        )

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

    def _get_w_cl(self):
        ver = self.weight._version
        if self._w_cl is None or self._w_cl_version != ver:
            k = self.kernel_size[0]
            if k == 3:
                self._w_cl = self.weight.data.half().permute(0, 2, 3, 1).contiguous()
            else:
                self._w_cl = self.weight.data.half().reshape(
                    self.out_channels, self.in_channels).contiguous()
            self._w_cl_version = ver
        return self._w_cl

    def _ensure_buffers(self, x):
        from Kernels.conv2d import _select_tile_sizes
        import triton
        C_IN, H, W = x.shape[1], x.shape[2], x.shape[3]
        BH, BW = _select_tile_sizes(H, W)
        N_TILES = x.shape[0] * triton.cdiv(H, BH) * triton.cdiv(W, BW)
        if self._ag_mask_buf is None or self._ag_mask_buf.numel() < N_TILES:
            self._ag_mask_buf = torch.empty(N_TILES, dtype=torch.int32, device=x.device)
        if self._tile_class_buf is None or self._tile_class_buf.numel() < N_TILES:
            self._tile_class_buf = torch.empty(N_TILES, dtype=torch.int32, device=x.device)
        if self._active_tile_ids_buf is None or self._active_tile_ids_buf.numel() < N_TILES:
            self._active_tile_ids_buf = torch.empty(N_TILES, dtype=torch.int32, device=x.device)
        return self._ag_mask_buf, self._tile_class_buf, self._active_tile_ids_buf

    def _prepare_nhwc(self, x_f16):
        """NHWC from fp16 NCHW with buffer reuse."""
        target_shape = (x_f16.shape[0], x_f16.shape[2], x_f16.shape[3], x_f16.shape[1])
        if (self._x_nhwc_buf is not None
                and self._x_nhwc_buf.shape == target_shape
                and self._x_nhwc_buf.device == x_f16.device):
            self._x_nhwc_buf.copy_(x_f16.permute(0, 2, 3, 1))
            return self._x_nhwc_buf
        self._x_nhwc_buf = x_f16.permute(0, 2, 3, 1).contiguous()
        self._x_nhwc_shape = target_shape
        return self._x_nhwc_buf

    def _zero_output_4d(self, x):
        N, C_IN, H_IN, W_IN = x.shape
        k, s, p, d = self.kernel_size[0], self.stride[0], self.padding[0], self.dilation[0]
        H_OUT = (H_IN + 2 * p - d * (k - 1) - 1) // s + 1
        W_OUT = (W_IN + 2 * p - d * (k - 1) - 1) // s + 1
        y = torch.zeros(N, self.out_channels, H_OUT, W_OUT, dtype=torch.float32, device=x.device)
        if self.bias is not None:
            y = y + self.bias.detach().float().view(1, -1, 1, 1)
        return y

    def _zero_output_5d(self, x):
        T, B, C, H, W = x.shape
        y4d = self._zero_output_4d(x.reshape(T * B, C, H, W))
        _, Co, Ho, Wo = y4d.shape
        return y4d.reshape(T, B, Co, Ho, Wo)

    # ----------------------------------------------------------------
    # Policy
    # ----------------------------------------------------------------
    def _should_collect_ratio(self):
        """Return True only when calibration data is needed.

        [P2 FIX]: When _inference_mode is True, always returns False.
        This ensures zero syncs during timed inference.
        """
        if self._inference_mode:
            return False
        if self._warmup_left > 0:
            return True
        return (self._forward_count % self._calib_every) == 0

    def _update_policy(self, avg_active_ratio):
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
            self._ema_active_ratio = (self._ema_decay * self._ema_active_ratio
                                       + (1 - self._ema_decay) * avg_active_ratio)
        if avg_active_ratio == 0.0:
            self._zero_streak += 1
        else:
            self._zero_streak = 0
        if self._zero_streak >= self._zero_streak_needed:
            self._force_zero = True
        if self._ema_active_ratio is not None and self._ema_active_ratio > self._dense_threshold:
            self._force_dense = True
        else:
            self._force_dense = False

    @classmethod
    def from_dense(cls, conv: nn.Conv2d, block_size=None, threshold=1e-6,
                   return_ms=False, **kwargs):
        sparse = cls(
            in_channels=conv.in_channels, out_channels=conv.out_channels,
            kernel_size=conv.kernel_size, stride=conv.stride,
            padding=conv.padding, dilation=conv.dilation,
            groups=conv.groups, bias=conv.bias is not None,
            block_size=block_size, threshold=threshold,
            return_ms=return_ms, **kwargs,
        )
        sparse = sparse.to(conv.weight.device)
        sparse.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            sparse.bias.data.copy_(conv.bias.data)
        return sparse

    # ----------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        self.fallback_reason = ""
        if self.profile_runtime:
            self._profile.calls += 1
            t_total = self._stamp()
        if self.collect_diag:
            self._last_diag = {'sparse_path_executed': False}

        # Force-zero path (no sync — boolean flag)
        if self._force_zero:
            y = self._zero_output_5d(x) if x.dim() == 5 else self._zero_output_4d(x)
            self._last_sparse_ms = 0.0
            self._record_runtime_dispatch("staticzero")
            self.fallback_reason = "force_zero"
            if self.profile_runtime:
                self._profile.zero_path_hits += 1
                self._profile.last_path = "zero(force)"
                total_ms = self._elapsed_ms(t_total)
                self._profile_add("total_ms", total_ms)
                self._profile_set_last("last_total_ms", total_ms)
            return y

        # Reshape 5D → 4D
        reshaped = False
        if self.profile_runtime:
            t0 = self._stamp()
        if x.dim() == 5:
            T, B, C, H, W = x.shape
            x4d = x.reshape(T * B, C, H, W)
            reshaped = True
        elif x.dim() == 4:
            x4d = x
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")
        if self.profile_runtime:
            ms = self._elapsed_ms(t0)
            self._profile_add("reshape_ms", ms)
            self._profile_set_last("last_reshape_ms", ms)

        use_triton = self._supports_triton() and x4d.is_cuda
        if self.profile_runtime and use_triton:
            self._profile.triton_supported_calls += 1

        # Determine if ratio collection is needed this forward
        if self.profile_runtime:
            t0 = self._stamp()
        need_ratio = use_triton and self._should_collect_ratio() and (not self._force_dense)
        if self.profile_runtime:
            ms = self._elapsed_ms(t0)
            self._profile_add("policy_ms", ms)
            self._profile_set_last("last_policy_ms", ms)

        self._forward_count += 1

        # Dispatch
        if use_triton and not self._force_dense and self._supports_sparse():
            y4d, avg_active_ratio, backend_kind = self._triton_forward(x4d, need_ratio=need_ratio)
            if backend_kind == "dense_fallback":
                path = "dense(fallback)"
                self.fallback_reason = "kernel_dense_fallback"
                self._record_runtime_dispatch("dense")
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
            y4d = self._fallback_forward(x4d)
            avg_active_ratio = None
            path = "dense"
            self.fallback_reason = "module_dense_path"
            self._record_runtime_dispatch("dense")
            if self.profile_runtime:
                self._profile.dense_path_hits += 1

        if self.profile_runtime:
            self._profile.last_avg_active_ratio = self._last_avg_active_ratio

        # Policy update
        if self.profile_runtime:
            t0 = self._stamp()
        self._update_policy(avg_active_ratio)
        if self.profile_runtime:
            ms = self._elapsed_ms(t0)
            self._profile_add("policy_ms", ms)
            self._profile_set_last("last_policy_ms", self._profile.last_policy_ms + ms)

        if self._force_zero and avg_active_ratio == 0.0:
            y4d = self._zero_output_4d(x4d)
            path = "zero(promoted)"
            self.fallback_reason = "zero_promoted"
            self._record_runtime_dispatch("staticzero")

        # Reshape back 4D → 5D
        if reshaped:
            if self.profile_runtime:
                t0 = self._stamp()
            _, C_out, H_out, W_out = y4d.shape
            y = y4d.reshape(T, B, C_out, H_out, W_out)
            if self.profile_runtime:
                ms = self._elapsed_ms(t0)
                self._profile_add("output_pack_ms", ms)
                self._profile_set_last("last_output_pack_ms", ms)
        else:
            y = y4d
            if self.profile_runtime:
                self._profile_set_last("last_output_pack_ms", 0.0)

        if self.profile_runtime:
            self._profile.last_path = path
            total_ms = self._elapsed_ms(t_total)
            self._profile_add("total_ms", total_ms)
            self._profile_set_last("last_total_ms", total_ms)
        if self.collect_diag:
            self._last_diag["runtime_dispatch_seen"] = int(self._runtime_dispatch_seen)
            self._last_diag["runtime_dispatch_mismatch"] = int(self._runtime_dispatch_mismatch)
            self._last_diag["fallback_reason"] = str(self.fallback_reason)
        return y

    # ----------------------------------------------------------------
    def _triton_forward(self, x, need_ratio: bool = False):
        """Core Triton sparse convolution path.

        When need_ratio=False and collect_diag=False:
          → passes return_avg_active_ratio=False to kernel
          → kernel skips all .item() syncs (P0 fix in conv2d.py)
          → if launch_all_tiles=True: ZERO GPU→CPU syncs total
          → if launch_all_tiles=False: 1 sync (nonzero for active IDs)
        """
        from Kernels.conv2d import sparse_conv2d_forward

        if self.profile_runtime:
            t0 = self._stamp()

        ag_mask_buf, tile_class_buf, active_tile_ids_buf = self._ensure_buffers(x)
        w_cl = self._get_w_cl()

        # Single .half() → x_f16 NCHW; derive NHWC from it
        if x.dtype == torch.float16 and x.is_contiguous():
            x_f16 = x
        else:
            x_f16 = x.half().contiguous()
        x_nhwc = self._prepare_nhwc(x_f16)

        if self.profile_runtime:
            ms = self._elapsed_ms(t0)
            self._profile_add("buffer_ms", ms)
            self._profile_set_last("last_buffer_ms", ms)

        k = self.kernel_size[0]
        collect_tiles = self.collect_diag
        want_ratio = need_ratio or collect_tiles
        want_tiles = collect_tiles

        # [P0+P1] Pass flags and launch mode to kernel.
        # When want_ratio=False and want_tiles=False:
        #   kernel skips all .item() syncs.
        # When launch_all_tiles=True:
        #   kernel skips _build_active_tile_ids (nonzero sync).
        result = sparse_conv2d_forward(
            x=x_f16,
            weight=self.weight, bias=self.bias,
            kernel_size=k, stride=self.stride[0],
            padding=self.padding[0], dilation=self.dilation[0],
            groups=self.groups, threshold=self.threshold,
            w_cl=w_cl, ag_mask_buf=ag_mask_buf,
            tile_class_buf=tile_class_buf,
            return_ms=self.return_ms,
            return_avg_active_ratio=want_ratio,
            return_tile_stats=want_tiles,
            return_backend_meta=True,
            x_nhwc=x_nhwc,
            active_tile_ids_buf=active_tile_ids_buf,
            launch_all_tiles=self.launch_all_tiles,   # [P1] A/B switch
        )

        # --- Unpack result ---
        # Return arity: (y, ms) + optional (ratio,) + optional (tile_stats,) + optional (meta,)
        if not isinstance(result, tuple) or len(result) < 2:
            raise TypeError(f"sparse_conv2d_forward bad return: {type(result)}")

        idx = 0
        y = result[idx]; idx += 1
        sparse_ms = result[idx]; idx += 1

        avg_active_ratio = None
        if want_ratio and idx < len(result):
            avg_active_ratio = result[idx]; idx += 1

        tile_stats = None
        if want_tiles and idx < len(result):
            tile_stats = result[idx]; idx += 1

        backend_meta = result[idx] if idx < len(result) else {
            "backend": "sparse_triton", "reason": "no_meta"}

        # --- Update internal state ---
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
                x, ag_mask_buf, tile_class_buf,
                sparse_ms, avg_active_ratio, tile_stats,
                backend_meta=backend_meta,
            )
        return y, avg_active_ratio, backend_kind

    # ----------------------------------------------------------------
    def _collect_tile_group_diag(self, x, ag_mask_buf, tile_class_buf,
                                 sparse_ms, avg_active_ratio, tile_stats, backend_meta=None):
        """Collect diagnostics. Only called when self.collect_diag is True.
        NOTE: contains GPU→CPU syncs — must NOT be enabled during perf timing.
        """
        from Kernels.conv2d import (
            _select_tile_sizes, choose_group_size, _popcount_buf,
            TILE_ZERO, TILE_SPARSE, TILE_DENSEISH,
        )
        import triton
        N, C_IN, H, W = x.shape
        GROUP_SIZE_C = choose_group_size(C_IN)
        NUM_GROUPS = triton.cdiv(C_IN, GROUP_SIZE_C)
        BH, BW = _select_tile_sizes(H, W)
        N_TILES = N * triton.cdiv(H, BH) * triton.cdiv(W, BW)

        if tile_stats is not None:
            zt = tile_stats.get('zero_tiles', -1)
            st = tile_stats.get('sparse_tiles', -1)
            dt = tile_stats.get('denseish_tiles', -1)
        else:
            tc = tile_class_buf[:N_TILES].cpu()
            zt = int((tc == TILE_ZERO).sum().item())
            st = int((tc == TILE_SPARSE).sum().item())
            dt = int((tc == TILE_DENSEISH).sum().item())

        pc = _popcount_buf(ag_mask_buf, N_TILES)
        total_g = float(N_TILES * NUM_GROUPS)
        active_g = float(pc.sum().item())
        agr = active_g / max(total_g, 1.0)

        self._last_diag = {
            'sparse_path_executed': True,
            'metadata_kind': 'three_stage_nhwc_v25',
            'group_size': GROUP_SIZE_C, 'num_groups': NUM_GROUPS,
            'nonzero_group_count': active_g, 'total_group_count': total_g,
            'active_group_ratio': agr,
            'tile_zero_count': zt, 'total_tile_count': N_TILES,
            'tile_zero_ratio': zt / max(N_TILES, 1),
            'effective_k_ratio': agr,
            'sparse_compute_ms': sparse_ms, 'sparse_total_ms': sparse_ms,
            'zero_check_ms': -1.0, 'metadata_ms': -1.0,
            'zero_tiles': zt, 'sparse_tiles': st, 'denseish_tiles': dt,
            'prescan_mode': 'three_stage_nhwc_v25',
            'kernel_type': f"{self.kernel_size[0]}x{self.kernel_size[0]}/s{self.stride[0]}",
        }
        if backend_meta is not None:
            self._last_diag['backend'] = backend_meta.get('backend', 'unknown')
            self._last_diag['backend_reason'] = backend_meta.get('reason', 'unknown')
            if 'active_tiles' in backend_meta:
                self._last_diag['active_tiles'] = backend_meta['active_tiles']
            if 'total_tiles' in backend_meta:
                self._last_diag['launch_tile_count'] = backend_meta.get('launch_count', backend_meta['total_tiles'])
            if 'denseish_ratio_nonzero' in backend_meta:
                self._last_diag['denseish_ratio_nonzero'] = backend_meta['denseish_ratio_nonzero']
        self._last_diag['backend_family'] = self.backend_family
        self._last_diag['diag_path'] = self.diag_path
        self._last_diag['fallback_reason'] = self.fallback_reason
        if tile_stats is not None:
            for key in ('stage1_zero_candidate', 'stage1_denseish',
                        'stage1_uncertain', 'final_zero', 'final_sparse',
                        'final_denseish', 'stage2_zero_refine_tiles',
                        'stage2_uncertain_tiles'):
                if key in tile_stats:
                    self._last_diag[key] = tile_stats[key]

    # ----------------------------------------------------------------
    def _fallback_forward(self, x):
        if self.profile_runtime:
            t0 = self._stamp()
        y = F.conv2d(x, self.weight, self.bias,
                     self.stride, self.padding, self.dilation, self.groups)
        self._last_sparse_ms = 0.0
        if self.profile_runtime:
            self._last_dense_ms = self._elapsed_ms(t0)
            self._profile_add("dense_fallback_ms", self._last_dense_ms)
            self._profile_set_last("last_dense_fallback_ms", self._last_dense_ms)
            self._profile_set_last("last_sparse_kernel_ms", 0.0)
        else:
            self._last_dense_ms = 0.0
        return y

    def extra_repr(self):
        bs = self.block_size if self.block_size is not None else "auto"
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, block_size={bs}, "
            f"return_ms={self.return_ms}, sparse=True, metadata=three_stage_nhwc_v25, "
            f"dense_threshold={self._dense_threshold}, warmup_steps={self._warmup_left}, "
            f"calib_every={self._calib_every}, profile_runtime={self.profile_runtime}, "
            f"allow_sparse_1x1={self.allow_sparse_1x1}, "
            f"launch_all_tiles={self.launch_all_tiles}, "
            f"inference_mode={self._inference_mode}"
        )
