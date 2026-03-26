"""
SparseConv2d — v24 Three-Stage Prescan + Wider Dense-K + Reduced NHWC Overhead

Changes from v23:
  - [Priority C] _triton_forward creates x_f16 once, derives x_nhwc from it,
    passes both to sparse_conv2d_forward → eliminates redundant .half() inside kernel.
  - [Priority C] _prepare_nhwc_from_f16 accepts fp16 NCHW and only does permute.
  - Updated diagnostics: metadata_kind = "three_stage_nhwc_v24"
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
    zero_check_ms: float = 0.0
    policy_ms: float = 0.0
    reshape_ms: float = 0.0
    buffer_ms: float = 0.0
    sparse_kernel_ms: float = 0.0
    dense_fallback_ms: float = 0.0
    output_pack_ms: float = 0.0

    last_path: str = "none"
    last_total_ms: float = 0.0
    last_zero_check_ms: float = 0.0
    last_policy_ms: float = 0.0
    last_reshape_ms: float = 0.0
    last_buffer_ms: float = 0.0
    last_sparse_kernel_ms: float = 0.0
    last_dense_fallback_ms: float = 0.0
    last_output_pack_ms: float = 0.0
    last_avg_active_ratio: float = -1.0


class SparseConv2d(nn.Module):
    def __init__(
        self,
        in_channels, out_channels, kernel_size,
        stride=1, padding=0, dilation=1, groups=1, bias=True,
        block_size=None, threshold=1e-6, return_ms=False,
        dense_threshold: float = 0.85,
        warmup_steps: int = 8, calib_every: int = 32,
        ema_decay: float = 0.9, zero_streak_needed: int = 2,
        profile_runtime: bool = False,
        allow_sparse_1x1: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.block_size = block_size
        self.threshold = threshold
        self.return_ms = return_ms

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self._last_sparse_ms = 0.0
        self._last_dense_ms = 0.0
        self._triton_available = False
        try:
            import triton  # noqa: F401
            self._triton_available = True
        except ImportError:
            pass

        self._w_cl = None
        self._w_cl_version = -1
        self._ag_mask_buf = None
        self._tile_class_buf = None
        self._active_tile_ids_buf = None

        # [Priority C] NHWC buffer cache
        self._x_nhwc_buf = None
        self._x_nhwc_shape = None

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
        self._zero_template = None
        self._zero_template_key = None
        self.collect_diag = False
        self._last_diag = {}
        self.profile_runtime = profile_runtime
        self.allow_sparse_1x1 = bool(allow_sparse_1x1)
        self._profile = _ProfileStats()

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

    def get_profile_summary(self):
        p = self._profile
        if p.calls == 0:
            return "SparseConv2d runtime profile: no calls"
        def avg(x): return x / max(1, p.calls)
        return (
            f"SparseConv2d runtime profile\n"
            f"  calls={p.calls}, last_path={p.last_path}\n"
            f"  hits: zero={p.zero_path_hits}, sparse={p.sparse_path_hits}, dense={p.dense_path_hits}\n"
            f"  avg total={avg(p.total_ms):.4f} ms, last total={p.last_total_ms:.4f} ms\n"
            f"  avg zero_check={avg(p.zero_check_ms):.4f} ms\n"
            f"  avg buffer={avg(p.buffer_ms):.4f} ms\n"
            f"  avg sparse_kernel={avg(p.sparse_kernel_ms):.4f} ms\n"
            f"  avg dense_fallback={avg(p.dense_fallback_ms):.4f} ms\n"
            f"  last avg_active_ratio={p.last_avg_active_ratio:.6f}"
        )

    # ----------------------------------------------------------------
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

    def _profile_set_last(self, field: str, ms: float):
        if self.profile_runtime:
            setattr(self._profile, field, ms)

    # ----------------------------------------------------------------
    def _supports_triton(self) -> bool:
        if not self._triton_available:
            return False
        if self.dilation != (1, 1) or self.groups != 1:
            return False
        k, s, p = self.kernel_size, self.stride, self.padding
        # 1x1/downsample are split out of the unified sparse path by default.
        if k == (1, 1):
            return self.allow_sparse_1x1 and s == (1, 1) and p == (0, 0)
        return (
            (k == (3, 3) and s == (1, 1) and p == (1, 1))
            or (k == (3, 3) and s == (2, 2) and p == (1, 1))
        )

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
        """[Priority C] NHWC from fp16 NCHW with buffer reuse.

        Takes an already-fp16-contiguous NCHW tensor. Only does the permute
        (no redundant .half()). Reuses buffer if shape matches.
        """
        target_shape = (x_f16.shape[0], x_f16.shape[2], x_f16.shape[3], x_f16.shape[1])
        if (self._x_nhwc_buf is not None
                and self._x_nhwc_buf.shape == target_shape
                and self._x_nhwc_buf.device == x_f16.device):
            self._x_nhwc_buf.copy_(x_f16.permute(0, 2, 3, 1))
            return self._x_nhwc_buf
        self._x_nhwc_buf = x_f16.permute(0, 2, 3, 1).contiguous()
        self._x_nhwc_shape = target_shape
        return self._x_nhwc_buf

    def _maybe_zero_fast_path(self, x):
        return x.abs().max().item() <= self.threshold

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

    def _should_collect_ratio(self):
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
            self._ema_active_ratio = self._ema_decay * self._ema_active_ratio + (1 - self._ema_decay) * avg_active_ratio
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

    # ----------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        if self.profile_runtime:
            self._profile.calls += 1
            t_total = self._stamp()
        if self.collect_diag:
            self._last_diag = {'sparse_path_executed': False}

        if self._force_zero:
            y = self._zero_output_5d(x) if x.dim() == 5 else self._zero_output_4d(x)
            self._last_sparse_ms = 0.0
            if self.profile_runtime:
                self._profile.zero_path_hits += 1
                self._profile.last_path = "zero(force)"
                total_ms = self._elapsed_ms(t_total)
                self._profile_add("total_ms", total_ms)
                self._profile_set_last("last_total_ms", total_ms)
            return y

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

        if self.profile_runtime:
            t0 = self._stamp()
        avg_active_ratio = None
        need_ratio = use_triton and self._should_collect_ratio() and (not self._force_dense)
        if self.profile_runtime:
            ms = self._elapsed_ms(t0)
            self._profile_add("policy_ms", ms)
            self._profile_set_last("last_policy_ms", ms)

        self._forward_count += 1

        if use_triton and not self._force_dense:
            y4d, avg_active_ratio, backend_kind = self._triton_forward(x4d, need_ratio=need_ratio)
            if backend_kind == "dense_fallback":
                path = "dense(fallback)"
                if self.profile_runtime:
                    self._profile.dense_path_hits += 1
            elif backend_kind == "zero_tiles_only":
                path = "zero(tile_compaction)"
                if self.profile_runtime:
                    self._profile.zero_path_hits += 1
            else:
                path = "sparse"
                if self.profile_runtime:
                    self._profile.sparse_path_hits += 1
        else:
            y4d = self._fallback_forward(x4d)
            avg_active_ratio = None
            path = "dense"
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
            y4d = self._zero_output_4d(x4d)
            path = "zero(promoted)"

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
        return y

    # ----------------------------------------------------------------
    def _triton_forward(self, x, need_ratio: bool = False):
        from Kernels.conv2d import sparse_conv2d_forward

        if self.profile_runtime:
            t0 = self._stamp()

        ag_mask_buf, tile_class_buf, active_tile_ids_buf = self._ensure_buffers(x)
        w_cl = self._get_w_cl()

        # [Priority C] Single .half() → x_f16 NCHW; derive NHWC from it
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

        # Pass x_f16 as x (prescan uses NCHW), x_nhwc for compute
        if need_ratio or collect_tiles:
            result = sparse_conv2d_forward(
                x=x_f16,
                weight=self.weight, bias=self.bias,
                kernel_size=k, stride=self.stride[0],
                padding=self.padding[0], dilation=self.dilation[0],
                groups=self.groups, threshold=self.threshold,
                w_cl=w_cl, ag_mask_buf=ag_mask_buf,
                tile_class_buf=tile_class_buf,
                return_ms=self.return_ms,
                return_avg_active_ratio=True,
                return_tile_stats=collect_tiles,
                return_backend_meta=True,
                x_nhwc=x_nhwc,
                active_tile_ids_buf=active_tile_ids_buf,
            )
            if not isinstance(result, tuple):
                raise TypeError(f"sparse_conv2d_forward must return a tuple, got {type(result)!r}")
            if collect_tiles:
                if len(result) == 5:
                    y, sparse_ms, avg_active_ratio, tile_stats, backend_meta = result
                elif len(result) == 4:
                    # Some fallback paths do not emit tile_stats even when requested.
                    y, sparse_ms, avg_active_ratio, backend_meta = result
                    tile_stats = None
                else:
                    raise ValueError(
                        f"Unexpected sparse_conv2d_forward return arity with tile stats requested: {len(result)}"
                    )
            else:
                if len(result) == 4:
                    y, sparse_ms, avg_active_ratio, backend_meta = result
                    tile_stats = None
                elif len(result) == 3:
                    # Backward-compatible path when backend_meta is omitted.
                    y, sparse_ms, avg_active_ratio = result
                    backend_meta = {"backend": "sparse_triton", "reason": "no_backend_meta"}
                    tile_stats = None
                else:
                    raise ValueError(
                        f"Unexpected sparse_conv2d_forward return arity without tile stats: {len(result)}"
                    )
        else:
            out = sparse_conv2d_forward(
                x=x_f16,
                weight=self.weight, bias=self.bias,
                kernel_size=k, stride=self.stride[0],
                padding=self.padding[0], dilation=self.dilation[0],
                groups=self.groups, threshold=self.threshold,
                w_cl=w_cl, ag_mask_buf=ag_mask_buf,
                tile_class_buf=tile_class_buf,
                return_ms=self.return_ms,
                return_avg_active_ratio=False,
                return_tile_stats=False,
                return_backend_meta=True,
                x_nhwc=x_nhwc,
                active_tile_ids_buf=active_tile_ids_buf,
            )
            if not isinstance(out, tuple):
                raise TypeError(f"sparse_conv2d_forward must return a tuple, got {type(out)!r}")
            if len(out) == 3:
                y, sparse_ms, backend_meta = out
            elif len(out) == 2:
                y, sparse_ms = out
                backend_meta = {"backend": "sparse_triton", "reason": "no_backend_meta"}
            else:
                raise ValueError(
                    f"Unexpected sparse_conv2d_forward return arity without ratio/tile stats: {len(out)}"
                )
            avg_active_ratio = None
            tile_stats = None

        backend_kind = backend_meta.get("backend", "sparse_triton") if backend_meta is not None else "sparse_triton"
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
                self._last_diag['launch_tile_count'] = backend_meta['total_tiles'] if backend_meta.get('backend') == 'dense_fallback' else backend_meta['active_tiles']
            if 'denseish_ratio_nonzero' in backend_meta:
                self._last_diag['denseish_ratio_nonzero'] = backend_meta['denseish_ratio_nonzero']
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
            f"allow_sparse_1x1={self.allow_sparse_1x1}"
        )