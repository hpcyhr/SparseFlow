import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from typing import Dict, List, Optional, Set, Tuple
import copy
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from Core.analyzer import ReplacementTarget, display_block_info
from Ops.sparse_conv2d import SparseConv2d
from Ops.static_zero_conv2d import StaticZeroConv2d
from Ops.sparse_fused_conv_lif import FusedSparseConvLIF


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _eligible_direct_fusion_conv_name(conv_name: str) -> bool:
    leaf = conv_name.rsplit('.', 1)[-1]
    return leaf in ('conv1', 'conv2')


class ModuleReplacer:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def replace(
        self,
        model: nn.Module,
        targets: List[ReplacementTarget],
        block_sizes: Optional[Dict[str, int]] = None,
        static_zero_layers: Optional[Set[str]] = None,
        disable_static_zero: bool = False,
        only_static_zero: bool = False,
    ) -> Tuple[int, int, int, int, int]:
        if static_zero_layers is None:
            static_zero_layers = set()
        if disable_static_zero:
            static_zero_layers = set()

        replaced = 0
        sparse_count = 0
        fused_count = 0
        static_zero_count = 0
        dense_keep_count = 0

        for target in targets:
            block = target.block_size
            if block_sizes and target.conv_name in block_sizes:
                block = block_sizes[target.conv_name]

            conv_name = target.conv_name
            op = target.op_type

            can_use_static_zero = (
                (not disable_static_zero)
                and (conv_name in static_zero_layers)
                and op in (
                    'conv2d_3x3', 'conv2d_1x1', 'conv2d_3x3_s2',
                    'fused_conv3x3_lif', 'fused_conv1x1_lif', 'fused_conv3x3s2_lif',
                )
            )
            if can_use_static_zero:
                new_module = StaticZeroConv2d.from_conv(target.conv_module)
                _set_module_by_name(model, conv_name, new_module)
                replaced += 1
                static_zero_count += 1
                if target.bn_name is not None:
                    _set_module_by_name(model, target.bn_name, nn.Identity())
                if target.lif_name is not None:
                    _set_module_by_name(model, target.lif_name, nn.Identity())
                if self.verbose:
                    print(f'  [STATIC ] {conv_name} ({op}) -> StaticZeroConv2d')
                continue

            if only_static_zero:
                dense_keep_count += 1
                if self.verbose:
                    print(f'  [KEEP   ] {conv_name} ({op}) -> DenseKeep')
                continue

            new_module = self._create_sparse_module(target, block)
            if new_module is None:
                dense_keep_count += 1
                if self.verbose:
                    print(f'  [KEEP   ] {conv_name} ({op}) -> DenseKeep (unsupported)')
                continue

            _set_module_by_name(model, conv_name, new_module)
            replaced += 1

            actually_fused = (
                op in ('fused_conv3x3_lif', 'fused_conv1x1_lif', 'fused_conv3x3s2_lif')
                and _eligible_direct_fusion_conv_name(conv_name)
                and target.lif_module is not None
                and isinstance(new_module, FusedSparseConvLIF)
            )

            if actually_fused:
                if target.bn_name is not None:
                    _set_module_by_name(model, target.bn_name, nn.Identity())
                    if self.verbose:
                        print(f'  [FUSE-ID] {target.bn_name} -> nn.Identity (folded into {conv_name})')
                if target.lif_name is not None:
                    _set_module_by_name(model, target.lif_name, nn.Identity())
                    if self.verbose:
                        print(f'  [FUSE-ID] {target.lif_name} -> nn.Identity (fused into {conv_name})')
                fused_count += 1
            else:
                sparse_count += 1

            if self.verbose:
                self._log_replacement(target, actually_fused=actually_fused)

        return replaced, sparse_count, fused_count, static_zero_count, dense_keep_count

    def _create_sparse_module(self, target: ReplacementTarget, block: Optional[int]) -> Optional[nn.Module]:
        op = target.op_type

        if op in ('conv2d_3x3', 'conv2d_1x1', 'conv2d_3x3_s2'):
            sparse = SparseConv2d.from_dense(target.conv_module, block_size=block)
            return sparse.to(target.conv_module.weight.device)

        if op in ('fused_conv3x3_lif', 'fused_conv1x1_lif', 'fused_conv3x3s2_lif'):
            if target.lif_module is None:
                sparse = SparseConv2d.from_dense(target.conv_module, block_size=block)
                return sparse.to(target.conv_module.weight.device)

            conv_for_fusion = target.conv_module
            if target.bn_name is not None and getattr(target, 'bn_module', None) is not None:
                try:
                    conv_for_fusion = fuse_conv_bn_eval(copy.deepcopy(target.conv_module).eval(), copy.deepcopy(target.bn_module).eval())
                except Exception:
                    conv_for_fusion = target.conv_module

            fused = FusedSparseConvLIF.from_conv_and_lif(
                conv_for_fusion,
                target.lif_module,
                block_size=block,
            )
            return fused.to(target.conv_module.weight.device)

        if op == 'linear':
            from Ops.sparse_linear import SparseLinear
            if not isinstance(target.conv_module, nn.Linear):
                return None
            sparse = SparseLinear.from_dense(target.conv_module)
            return sparse.to(target.conv_module.weight.device)

        return None

    def _log_replacement(self, target: ReplacementTarget, actually_fused: bool = False):
        op = target.op_type
        if op == 'linear':
            in_f = target.conv_module.in_features
            out_f = target.conv_module.out_features
            print(f'  [REPLACE] {target.conv_name} (linear {in_f}->{out_f}) <- {target.spike_name}')
        elif 'fused' in op and actually_fused:
            info = display_block_info(target)
            fused_with = target.lif_name or '?'
            extra = f', BN={target.bn_name}' if target.bn_name else ''
            print(f'  [FUSE   ] {target.conv_name} ({op}) + {fused_with}{extra} <- {target.spike_name}, {info}')
        else:
            info = display_block_info(target)
            print(f'  [REPLACE] {target.conv_name} ({op}) <- {target.spike_name}, {info}')
