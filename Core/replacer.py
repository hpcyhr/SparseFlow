"""
Core/replacer.py — Module replacement pass.

Walks a list of ReplacementTargets produced by Core/analyzer and swaps each
target module in the model for the corresponding SparseFlow wrapper
(SparseConv2d / SparseLinear / SparseAttention / …). Handles
StaticZero shortcuts for layers whose inputs are exact zero.

Round 4 cleanup (no semantic changes to non-fused paths):
  - Removed Conv+LIF fused replacement entirely. `FusedSparseConvLIF`,
    `fuse_conv_bn_eval`, `copy`, and the `_eligible_direct_fusion_conv_name`
    helper are gone.
  - Removed `fused_conv3x3_lif` / `fused_conv1x1_lif` / `fused_conv3x3s2_lif`
    entries from `_score_family_from_op`, from the `can_use_static_zero`
    op_type whitelist, and from `_create_sparse_module`.
  - Removed the `actually_fused` logic block and the two fused branches in
    `_log_replacement`.

Round 5.5a cleanup (no semantic changes):
  - Rewired the attention branch from `SparseAttentionBlock` (which had
    already been deleted from the repo in an earlier session, leaving a
    dangling `NameError` at runtime) to the consolidated `SparseAttention`
    class in `Ops/sparse_attention.py`. The 5-variant normalization map
    (`attention_matmul` → `attention_qkav`, `attention_proj_linear` →
    `attention_linear`) is preserved — those 5 variants still have distinct
    meaning inside `SparseAttention.from_dense(variant=...)`.

Round 5.5b (additive — Pool2d integration):
  - Added `maxpool2d` / `avgpool2d` branches in `_create_sparse_module`
    that call `SparseMaxPool2d.from_dense(...)` / `SparseAvgPool2d.from_dense(...)`.
    Pool modules are parameterless so the `.to(mod.weight.device)` tail
    used for conv branches is omitted.
  - `_score_family_from_op` now returns `"pool"` for pool op_types, matching
    the `self.score_family = "pool"` field that both pool Ops hardcode
    internally.
  - `_log_replacement` emits a dedicated pool format showing kernel / stride
    / input HxW.

Round 7 (breaking cleanup — back-compat shims removed):
  - Dropped the `enable_fused_conv_lif` kwarg from `replace()`. Callers
    that were passing it silently since Round 4 must be updated to omit
    it (Benchmark/bench_4test.py updated in the same round).
  - Dropped the synthetic `fused_count` position from the return tuple.
    `replace()` now returns a 4-tuple `(replaced, sparse_count,
    static_zero_count, dense_keep_count)` — bench code that read
    `summary[2]`/`[3]`/`[4]` has been reindexed accordingly.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch.nn as nn

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Core.analyzer import ReplacementTarget, display_block_info
from Ops.sparse_conv2d import SparseConv2d
from Ops.sparse_attention_block import SparseAttentionBlock
from Ops.sparse_maxpool2d import SparseMaxPool2d
from Ops.sparse_avgpool2d import SparseAvgPool2d
from Ops.static_zero_conv2d import StaticZeroConv2d
from Ops.static_zero_linear import StaticZeroLinear


_STATIC_ZERO_OP_TYPES = (
    "conv2d_3x3",
    "conv2d_1x1",
    "conv2d_3x3_s2",
    "depthwise_conv2d",
    "linear",
)

_ATTENTION_OP_TYPES = (
    "attention_qkav",
    "attention_linear",
    "attention_qkmix",
    "attention_matmul",
    "attention_proj_linear",
)


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


class ModuleReplacer:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Observability metadata
    # ------------------------------------------------------------------
    @staticmethod
    def _score_family_from_op(op_type: str) -> str:
        if op_type in (
            "conv2d_3x3", "conv2d_1x1", "conv2d_3x3_s2",
            "depthwise_conv2d", "conv1d", "conv3d",
        ):
            return "conv"
        if op_type == "linear":
            return "linear"
        if op_type in ("attention_linear", "attention_proj_linear"):
            return "attn_linear"
        if op_type in ("attention_qkav", "attention_qkmix", "attention_matmul"):
            return "attn_matmul"
        if op_type == "matmul":
            return "matmul"
        if op_type == "bmm":
            return "bmm"
        if op_type in ("maxpool2d", "avgpool2d"):
            return "pool"
        return "unknown"

    def _attach_observability(
        self,
        module: nn.Module,
        target: ReplacementTarget,
        backend_mode: str,
        backend_family: Optional[str] = None,
        fallback_reason: str = "",
    ) -> nn.Module:
        """Attach unified observability metadata for all replaced modules.

        Both `sf_backend_mode` and `sf_dispatch_decision` are set to the same
        value; keep both keys for compatibility with older runtime readers.
        """
        op_type = str(target.op_type)
        if backend_family is None:
            backend_family = "exact_zero" if backend_mode == "staticzero" else "sparse_kernel"

        score_family = self._score_family_from_op(op_type)
        setattr(module, "sf_layer_name", str(target.conv_name))
        setattr(module, "sf_operator_type", op_type)
        setattr(module, "sf_operator_family", score_family)
        setattr(module, "sf_spike_source", str(target.spike_name))
        setattr(module, "sf_backend_mode", str(backend_mode))
        setattr(module, "sf_dispatch_decision", str(backend_mode))
        setattr(module, "sf_backend_family", str(backend_family))
        setattr(module, "sf_reason_code", "")
        setattr(module, "sf_meta_source", "measured")
        setattr(module, "sf_diag_source", "measured")
        setattr(module, "sf_support_status", "supported")
        setattr(module, "sf_score_family", score_family)
        setattr(module, "sf_tile_source", "unknown")
        setattr(module, "sf_fallback_reason", str(fallback_reason))
        return module

    # ------------------------------------------------------------------
    # Driver loop
    # ------------------------------------------------------------------
    def replace(
        self,
        model: nn.Module,
        targets: List[ReplacementTarget],
        block_sizes: Optional[Dict[str, int]] = None,
        static_zero_layers: Optional[Set[str]] = None,
        disable_static_zero: bool = False,
        only_static_zero: bool = False,
        sparse_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, int, int, int]:
        """Replace all matched targets in `model` in place.

        Returns a 4-tuple
            (replaced, sparse_count, static_zero_count, dense_keep_count).
        The legacy 5-tuple shape with `fused_count` in position [2] was
        dropped in Round 7 after Benchmark/bench_4test.py was updated to
        the new indexing.
        """
        if static_zero_layers is None:
            static_zero_layers = set()
        if disable_static_zero:
            static_zero_layers = set()
        if sparse_kwargs is None:
            sparse_kwargs = {}

        replaced = 0
        sparse_count = 0
        static_zero_count = 0
        dense_keep_count = 0

        for target in targets:
            block = target.block_size
            if block_sizes and target.conv_name in block_sizes:
                block = block_sizes[target.conv_name]

            module_name = target.conv_name
            op = target.op_type

            # StaticZero shortcut
            can_use_static_zero = (
                (not disable_static_zero)
                and (module_name in static_zero_layers)
                and op in _STATIC_ZERO_OP_TYPES
            )
            if can_use_static_zero:
                if op == "linear":
                    new_module = StaticZeroLinear.from_linear(target.conv_module)
                    kind = "StaticZeroLinear"
                else:
                    new_module = StaticZeroConv2d.from_conv(target.conv_module)
                    kind = "StaticZeroConv2d"
                self._attach_observability(
                    new_module, target,
                    backend_mode="staticzero",
                    backend_family="exact_zero",
                )
                _set_module_by_name(model, module_name, new_module)
                replaced += 1
                static_zero_count += 1
                if self.verbose:
                    print(f"  [STATIC ] {module_name} ({op}) -> {kind}")
                continue

            if only_static_zero:
                dense_keep_count += 1
                if self.verbose:
                    print(f"  [KEEP   ] {module_name} ({op}) -> DenseKeep")
                continue

            new_module = self._create_sparse_module(
                target, block, sparse_kwargs=sparse_kwargs,
            )
            if new_module is None:
                dense_keep_count += 1
                if self.verbose:
                    print(f"  [KEEP   ] {module_name} ({op}) -> DenseKeep (unsupported)")
                continue

            self._attach_observability(
                new_module, target,
                backend_mode="sparse",
                backend_family="sparse_kernel",
            )
            _set_module_by_name(model, module_name, new_module)
            replaced += 1
            sparse_count += 1

            if self.verbose:
                self._log_replacement(target)

        return replaced, sparse_count, static_zero_count, dense_keep_count

    # ------------------------------------------------------------------
    # Per-op sparse module construction
    # ------------------------------------------------------------------
    def _create_sparse_module(
        self,
        target: ReplacementTarget,
        block: Optional[int],
        sparse_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[nn.Module]:
        if sparse_kwargs is None:
            sparse_kwargs = {}
        op = target.op_type
        mod = target.conv_module
        common_threshold = float(sparse_kwargs.get("threshold", 1e-6))
        common_return_ms = bool(sparse_kwargs.get("return_ms", False))
        common_collect_diag = bool(sparse_kwargs.get("collect_diag", False))
        common_profile_runtime = bool(sparse_kwargs.get("profile_runtime", False))

        def _apply_common(sparse):
            if hasattr(sparse, "return_ms"):
                sparse.return_ms = common_return_ms
            if hasattr(sparse, "collect_diag"):
                sparse.collect_diag = common_collect_diag
            if hasattr(sparse, "profile_runtime"):
                sparse.profile_runtime = common_profile_runtime
            return sparse

        # ---- conv2d 1x1 / 3x3 / 3x3_s2 ----
        if op in ("conv2d_3x3", "conv2d_1x1", "conv2d_3x3_s2"):
            sparse = SparseConv2d.from_dense(
                mod,
                block_size=block,
                threshold=common_threshold,
                return_ms=common_return_ms,
            )
            return _apply_common(sparse).to(mod.weight.device)

        # ---- depthwise conv2d ----
        if op == "depthwise_conv2d":
            from Ops.sparse_depthwise_conv2d import SparseDepthwiseConv2d
            sparse = SparseDepthwiseConv2d.from_dense(
                mod,
                threshold=common_threshold,
                return_ms=common_return_ms,
            )
            return _apply_common(sparse).to(mod.weight.device)

        # ---- maxpool2d ----
        if op == "maxpool2d":
            if not isinstance(mod, nn.MaxPool2d):
                return None
            if bool(getattr(mod, "return_indices", False)):
                return None
            if bool(getattr(mod, "ceil_mode", False)):
                return None
            sparse = SparseMaxPool2d.from_dense(
                mod,
                threshold=common_threshold,
                return_ms=common_return_ms,
            )
            return _apply_common(sparse)

        # ---- avgpool2d ----
        if op == "avgpool2d":
            if not isinstance(mod, nn.AvgPool2d):
                return None
            if bool(getattr(mod, "ceil_mode", False)):
                return None
            sparse = SparseAvgPool2d.from_dense(
                mod,
                threshold=common_threshold,
                return_ms=common_return_ms,
            )
            return _apply_common(sparse)

        # ---- conv1d ----
        if op == "conv1d":
            from Ops.sparse_conv1d import SparseConv1d
            if not isinstance(mod, nn.Conv1d):
                return None
            sparse = SparseConv1d.from_dense(
                mod,
                threshold=common_threshold,
                return_ms=common_return_ms,
            )
            return _apply_common(sparse).to(mod.weight.device)

        # ---- conv3d ----
        if op == "conv3d":
            from Ops.sparse_conv3d import SparseConv3d
            if not isinstance(mod, nn.Conv3d):
                return None
            sparse = SparseConv3d.from_dense(
                mod,
                threshold=common_threshold,
                return_ms=common_return_ms,
            )
            return _apply_common(sparse).to(mod.weight.device)

        # ---- linear ----
        if op == "linear":
            from Ops.sparse_linear import SparseLinear
            if not isinstance(mod, nn.Linear):
                return None
            sparse = SparseLinear.from_dense(
                mod,
                threshold=common_threshold,
                return_ms=common_return_ms,
            )
            return _apply_common(sparse).to(mod.weight.device)

        # ---- attention block (composite module) ----
        if op in _ATTENTION_OP_TYPES:
            variant = op
            if op == "attention_matmul":
                variant = "attention_qkav"
            elif op == "attention_proj_linear":
                variant = "attention_linear"
            sparse = SparseAttentionBlock.from_dense(
                mod,
                variant=variant,
                threshold=common_threshold,
                return_ms=common_return_ms,
                profile_runtime=common_profile_runtime,
            )
            return _apply_common(sparse)

        # ---- matmul ----
        if op == "matmul":
            from Ops.sparse_matmul import SparseMatmul
            sparse = SparseMatmul(
                threshold=common_threshold,
                return_ms=common_return_ms,
                profile_runtime=common_profile_runtime,
            )
            sparse.collect_diag = common_collect_diag
            return sparse

        # ---- bmm ----
        if op == "bmm":
            from Ops.sparse_bmm import SparseBMM
            sparse = SparseBMM(
                threshold=common_threshold,
                return_ms=common_return_ms,
                profile_runtime=common_profile_runtime,
            )
            sparse.collect_diag = common_collect_diag
            return sparse

        return None

    # ------------------------------------------------------------------
    # Verbose logging
    # ------------------------------------------------------------------
    def _log_replacement(self, target: ReplacementTarget):
        op = target.op_type
        mod = target.conv_module

        if op == "linear":
            print(
                f"  [REPLACE] {target.conv_name} "
                f"(linear {mod.in_features}->{mod.out_features}) "
                f"<- {target.spike_name}"
            )
            return

        if op in ("matmul", "bmm"):
            info = display_block_info(target)
            print(f"  [REPLACE] {target.conv_name} ({op}) <- {target.spike_name}, {info}")
            return

        if op in _ATTENTION_OP_TYPES:
            info = display_block_info(target)
            h = getattr(mod, "num_heads", "?")
            d = getattr(mod, "head_dim", "?")
            print(
                f"  [REPLACE] {target.conv_name} ({op}, heads={h}, head_dim={d}) "
                f"<- {target.spike_name}, {info}"
            )
            return

        if op == "conv1d":
            print(
                f"  [REPLACE] {target.conv_name} "
                f"(conv1d {mod.in_channels}->{mod.out_channels}, "
                f"k={mod.kernel_size}, s={mod.stride}, p={mod.padding}) "
                f"<- {target.spike_name}"
            )
            return

        if op == "conv3d":
            print(
                f"  [REPLACE] {target.conv_name} "
                f"(conv3d {mod.in_channels}->{mod.out_channels}, "
                f"k={mod.kernel_size}, s={mod.stride}, p={mod.padding}) "
                f"<- {target.spike_name}"
            )
            return

        if op == "depthwise_conv2d":
            info = display_block_info(target)
            print(f"  [REPLACE] {target.conv_name} (depthwise_conv2d) <- {target.spike_name}, {info}")
            return

        if op in ("maxpool2d", "avgpool2d"):
            kind = "MaxPool2d" if op == "maxpool2d" else "AvgPool2d"
            ks = getattr(mod, "kernel_size", "?")
            st = getattr(mod, "stride", "?")
            h, w = int(getattr(target, "input_h", 0) or 0), int(getattr(target, "input_w", 0) or 0)
            hw = f"{h}x{w}" if h > 0 and w > 0 else "?x?"
            print(
                f"  [REPLACE] {target.conv_name} ({kind} k={ks}, s={st}, in={hw}) "
                f"<- {target.spike_name}"
            )
            return

        info = display_block_info(target)
        print(f"  [REPLACE] {target.conv_name} ({op}) <- {target.spike_name}, {info}")
