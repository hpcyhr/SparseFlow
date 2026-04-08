import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from typing import Any, Dict, List, Optional, Set, Tuple
import copy
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from Core.analyzer import ReplacementTarget, display_block_info
from Ops.sparse_conv2d import SparseConv2d
from Ops.static_zero_conv2d import StaticZeroConv2d
from Ops.static_zero_linear import StaticZeroLinear
from Ops.sparse_fused_conv_lif import FusedSparseConvLIF
from Ops.sparse_attention_block import SparseAttentionBlock


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _eligible_direct_fusion_conv_name(conv_name: str) -> bool:
    leaf = conv_name.rsplit(".", 1)[-1]
    return leaf in ("conv1", "conv2")


class ModuleReplacer:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    @staticmethod
    def _score_family_from_op(op_type: str) -> str:
        if op_type in ("conv2d_3x3", "conv2d_1x1", "conv2d_3x3_s2", "fused_conv3x3_lif", "fused_conv1x1_lif", "fused_conv3x3s2_lif", "depthwise_conv2d", "conv1d", "conv3d"):
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
        return "unknown"

    def _attach_observability(
        self,
        module: nn.Module,
        target: ReplacementTarget,
        backend_mode: str,
        backend_family: Optional[str] = None,
        fallback_reason: str = "",
    ) -> nn.Module:
        """
        Attach unified observability metadata for all replaced modules.

        Note: keep both `sf_backend_mode` and `sf_dispatch_decision` for
        compatibility with older runtime readers.
        """
        op_type = str(target.op_type)
        if backend_family is None:
            backend_family = "exact_zero" if backend_mode == "staticzero" else "sparse_kernel"

        setattr(module, "sf_layer_name", str(target.conv_name))
        setattr(module, "sf_operator_type", op_type)
        setattr(module, "sf_operator_family", self._score_family_from_op(op_type))
        setattr(module, "sf_spike_source", str(target.spike_name))
        setattr(module, "sf_backend_mode", str(backend_mode))
        setattr(module, "sf_dispatch_decision", str(backend_mode))
        setattr(module, "sf_backend_family", str(backend_family))
        setattr(module, "sf_reason_code", "")
        setattr(module, "sf_meta_source", "measured")
        setattr(module, "sf_diag_source", "measured")
        setattr(module, "sf_support_status", "supported")
        setattr(module, "sf_score_family", self._score_family_from_op(op_type))
        setattr(module, "sf_tile_source", "unknown")
        setattr(module, "sf_fallback_reason", str(fallback_reason))
        return module

    def replace(
        self,
        model: nn.Module,
        targets: List[ReplacementTarget],
        block_sizes: Optional[Dict[str, int]] = None,
        static_zero_layers: Optional[Set[str]] = None,
        disable_static_zero: bool = False,
        only_static_zero: bool = False,
        enable_fused_conv_lif: bool = False,
        sparse_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, int, int, int, int]:
        if static_zero_layers is None:
            static_zero_layers = set()
        if disable_static_zero:
            static_zero_layers = set()
        if sparse_kwargs is None:
            sparse_kwargs = {}

        replaced = 0
        sparse_count = 0
        fused_count = 0
        static_zero_count = 0
        dense_keep_count = 0

        for target in targets:
            block = target.block_size
            if block_sizes and target.conv_name in block_sizes:
                block = block_sizes[target.conv_name]

            module_name = target.conv_name
            op = target.op_type

            can_use_static_zero = (
                (not disable_static_zero)
                and (module_name in static_zero_layers)
                and op
                in (
                    "conv2d_3x3",
                    "conv2d_1x1",
                    "conv2d_3x3_s2",
                    "depthwise_conv2d",
                    "fused_conv3x3_lif",
                    "fused_conv1x1_lif",
                    "fused_conv3x3s2_lif",
                    "linear",
                )
            )
            if can_use_static_zero:
                if op == "linear":
                    new_module = StaticZeroLinear.from_linear(target.conv_module)
                else:
                    new_module = StaticZeroConv2d.from_conv(target.conv_module)
                self._attach_observability(
                    new_module,
                    target,
                    backend_mode="staticzero",
                    backend_family="exact_zero",
                )
                _set_module_by_name(model, module_name, new_module)
                replaced += 1
                static_zero_count += 1
                if self.verbose:
                    if op == "linear":
                        print(f"  [STATIC ] {module_name} ({op}) -> StaticZeroLinear")
                    else:
                        print(f"  [STATIC ] {module_name} ({op}) -> StaticZeroConv2d")
                continue

            if only_static_zero:
                dense_keep_count += 1
                if self.verbose:
                    print(f"  [KEEP   ] {module_name} ({op}) -> DenseKeep")
                continue

            new_module = self._create_sparse_module(
                target,
                block,
                enable_fused_conv_lif=enable_fused_conv_lif,
                sparse_kwargs=sparse_kwargs,
            )
            if new_module is None:
                dense_keep_count += 1
                if self.verbose:
                    print(f"  [KEEP   ] {module_name} ({op}) -> DenseKeep (unsupported)")
                continue

            self._attach_observability(
                new_module,
                target,
                backend_mode="sparse",
                backend_family="fused_sparse" if isinstance(new_module, FusedSparseConvLIF) else "sparse_kernel",
            )
            _set_module_by_name(model, module_name, new_module)
            replaced += 1

            actually_fused = (
                enable_fused_conv_lif
                and
                op in ("fused_conv3x3_lif", "fused_conv1x1_lif", "fused_conv3x3s2_lif")
                and _eligible_direct_fusion_conv_name(module_name)
                and target.lif_module is not None
                and isinstance(new_module, FusedSparseConvLIF)
            )

            if actually_fused:
                if target.bn_name is not None:
                    _set_module_by_name(model, target.bn_name, nn.Identity())
                    if self.verbose:
                        print(f"  [FUSE-ID] {target.bn_name} -> nn.Identity (folded into {module_name})")
                if target.lif_name is not None:
                    _set_module_by_name(model, target.lif_name, nn.Identity())
                    if self.verbose:
                        print(f"  [FUSE-ID] {target.lif_name} -> nn.Identity (fused into {module_name})")
                fused_count += 1
            else:
                sparse_count += 1

            if self.verbose:
                self._log_replacement(target, actually_fused=actually_fused)

        return replaced, sparse_count, fused_count, static_zero_count, dense_keep_count

    def _create_sparse_module(
        self,
        target: ReplacementTarget,
        block: Optional[int],
        enable_fused_conv_lif: bool = False,
        sparse_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[nn.Module]:
        if sparse_kwargs is None:
            sparse_kwargs = {}
        op = target.op_type
        mod = target.conv_module
        common_threshold = sparse_kwargs.get("threshold", 1e-6)
        common_return_ms = bool(sparse_kwargs.get("return_ms", False))
        common_collect_diag = bool(sparse_kwargs.get("collect_diag", False))
        common_profile_runtime = bool(sparse_kwargs.get("profile_runtime", False))

        if op in ("conv2d_3x3", "conv2d_1x1", "conv2d_3x3_s2"):
            sparse = SparseConv2d.from_dense(
                mod,
                block_size=block,
                threshold=common_threshold,
                return_ms=common_return_ms,
            )
            if hasattr(sparse, "collect_diag"):
                sparse.collect_diag = common_collect_diag
            if hasattr(sparse, "profile_runtime"):
                sparse.profile_runtime = common_profile_runtime
            return sparse.to(mod.weight.device)

        if op == "depthwise_conv2d":
            from Ops.sparse_depthwise_conv2d import SparseDepthwiseConv2d
            sparse = SparseDepthwiseConv2d.from_dense(
                mod,
                threshold=common_threshold,
                return_ms=common_return_ms,
            )
            if hasattr(sparse, "collect_diag"):
                sparse.collect_diag = common_collect_diag
            return sparse.to(mod.weight.device)

        if op in ("fused_conv3x3_lif", "fused_conv1x1_lif", "fused_conv3x3s2_lif"):
            if not enable_fused_conv_lif:
                sparse = SparseConv2d.from_dense(
                    mod,
                    block_size=block,
                    threshold=common_threshold,
                    return_ms=common_return_ms,
                )
                if hasattr(sparse, "collect_diag"):
                    sparse.collect_diag = common_collect_diag
                if hasattr(sparse, "profile_runtime"):
                    sparse.profile_runtime = common_profile_runtime
                return sparse.to(mod.weight.device)
            if target.lif_module is None:
                sparse = SparseConv2d.from_dense(
                    mod,
                    block_size=block,
                    threshold=common_threshold,
                    return_ms=common_return_ms,
                )
                if hasattr(sparse, "collect_diag"):
                    sparse.collect_diag = common_collect_diag
                if hasattr(sparse, "profile_runtime"):
                    sparse.profile_runtime = common_profile_runtime
                return sparse.to(mod.weight.device)

            conv_for_fusion = mod
            if target.bn_name is not None and getattr(target, "bn_module", None) is not None:
                try:
                    conv_for_fusion = fuse_conv_bn_eval(
                        copy.deepcopy(mod).eval(),
                        copy.deepcopy(target.bn_module).eval(),
                    )
                except Exception:
                    conv_for_fusion = mod

            fused = FusedSparseConvLIF.from_conv_and_lif(
                conv_for_fusion,
                target.lif_module,
                block_size=block,
                threshold=common_threshold,
                return_ms=common_return_ms,
            )
            return fused.to(mod.weight.device)

        if op == "conv1d":
            from Ops.sparse_conv1d import SparseConv1d
            if not isinstance(mod, nn.Conv1d):
                return None
            sparse = SparseConv1d.from_dense(
                mod,
                threshold=common_threshold,
                return_ms=common_return_ms,
            )
            if hasattr(sparse, "collect_diag"):
                sparse.collect_diag = common_collect_diag
            return sparse.to(mod.weight.device)

        if op == "conv3d":
            from Ops.sparse_conv3d import SparseConv3d
            if not isinstance(mod, nn.Conv3d):
                return None
            sparse = SparseConv3d.from_dense(
                mod,
                threshold=common_threshold,
                return_ms=common_return_ms,
            )
            if hasattr(sparse, "collect_diag"):
                sparse.collect_diag = common_collect_diag
            return sparse.to(mod.weight.device)

        if op == "linear":
            from Ops.sparse_linear import SparseLinear
            if not isinstance(mod, nn.Linear):
                return None
            sparse = SparseLinear.from_dense(
                mod,
                threshold=common_threshold,
                return_ms=common_return_ms,
            )
            if hasattr(sparse, "collect_diag"):
                sparse.collect_diag = common_collect_diag
            if hasattr(sparse, "profile_runtime"):
                sparse.profile_runtime = common_profile_runtime
            return sparse.to(mod.weight.device)

        if op in ("attention_qkav", "attention_linear", "attention_qkmix", "attention_matmul", "attention_proj_linear"):
            variant = op
            if op == "attention_matmul":
                variant = "attention_qkav"
            elif op == "attention_proj_linear":
                variant = "attention_linear"
            sparse = SparseAttentionBlock.from_dense(
                mod,
                variant=variant,
                threshold=common_threshold,
            )
            if hasattr(sparse, "collect_diag"):
                sparse.collect_diag = common_collect_diag
            return sparse

        if op == "matmul":
            from Ops.sparse_matmul import SparseMatmul
            sparse = SparseMatmul(
                threshold=common_threshold,
                return_ms=common_return_ms,
                profile_runtime=common_profile_runtime,
            )
            sparse.collect_diag = common_collect_diag
            return sparse

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

    def _log_replacement(self, target: ReplacementTarget, actually_fused: bool = False):
        op = target.op_type
        mod = target.conv_module

        if op == "linear":
            in_f = mod.in_features
            out_f = mod.out_features
            print(f"  [REPLACE] {target.conv_name} (linear {in_f}->{out_f}) <- {target.spike_name}")
            return

        if op in ("matmul", "bmm"):
            info = display_block_info(target)
            print(f"  [REPLACE] {target.conv_name} ({op}) <- {target.spike_name}, {info}")
            return

        if op in ("attention_qkav", "attention_linear", "attention_qkmix", "attention_matmul", "attention_proj_linear"):
            info = display_block_info(target)
            h = getattr(mod, "num_heads", "?")
            d = getattr(mod, "head_dim", "?")
            print(f"  [REPLACE] {target.conv_name} ({op}, heads={h}, head_dim={d}) <- {target.spike_name}, {info}")
            return

        if "fused" in op and actually_fused:
            info = display_block_info(target)
            fused_with = target.lif_name or "?"
            extra = f", BN={target.bn_name}" if target.bn_name else ""
            print(f"  [FUSE   ] {target.conv_name} ({op}) + {fused_with}{extra} <- {target.spike_name}, {info}")
            return
        if "fused" in op and not actually_fused:
            info = display_block_info(target)
            print(f"  [REPLACE] {target.conv_name} ({op}, fusion=OFF) <- {target.spike_name}, {info}")
            return

        if op == "conv1d":
            print(
                f"  [REPLACE] {target.conv_name} (conv1d {mod.in_channels}->{mod.out_channels}, "
                f"k={mod.kernel_size}, s={mod.stride}, p={mod.padding}) <- {target.spike_name}"
            )
            return

        if op == "conv3d":
            print(
                f"  [REPLACE] {target.conv_name} (conv3d {mod.in_channels}->{mod.out_channels}, "
                f"k={mod.kernel_size}, s={mod.stride}, p={mod.padding}) <- {target.spike_name}"
            )
            return

        if op == "depthwise_conv2d":
            info = display_block_info(target)
            print(f"  [REPLACE] {target.conv_name} (depthwise_conv2d) <- {target.spike_name}, {info}")
            return

        info = display_block_info(target)
        print(f"  [REPLACE] {target.conv_name} ({op}) <- {target.spike_name}, {info}")
