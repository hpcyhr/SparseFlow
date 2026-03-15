"""
自动算子替换（增强版）

支持：
  1. Standalone Conv2d → SparseConv2d
  2. Static-zero Conv2d → StaticZeroConv2d
  3. Fused Conv2d+LIF → FusedSparseConvLIF + Identity
  4. Linear → SparseLinear

新增控制：
  - disable_static_zero: 禁用 StaticZeroConv2d，即使层被判为全零也不允许替换为 StaticZero
  - only_static_zero: 仅替换全零层；非全零层保持 Dense（DenseKeep）
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from typing import Dict, List, Optional, Set, Tuple
import torch.nn as nn

from Core.analyzer import ReplacementTarget, display_block_info
from Ops.sparse_conv2d import SparseConv2d
from Ops.static_zero_conv2d import StaticZeroConv2d


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """Replace a submodule by dot-separated name."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


class ModuleReplacer:
    """
    replacer supporting:
      1. Standalone Conv2d → SparseConv2d / StaticZeroConv2d / DenseKeep
      2. Fused Conv2d+LIF → FusedSparseConvLIF + Identity
      3. Linear → SparseLinear
    """

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
        """
        Returns:
            replaced_total,
            sparse_count,
            fused_count,
            static_zero_count,
            dense_keep_count
        """
        if static_zero_layers is None:
            static_zero_layers = set()

        # 最后一层保险：只要禁用 static-zero，就在 replacer 内部清空
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

            # -------------------------------
            # 1) StaticZero 分支（最后闸门）
            # -------------------------------
            can_use_static_zero = (
                (not disable_static_zero)
                and (conv_name in static_zero_layers)
                and op in ("conv2d_3x3", "conv2d_1x1", "conv2d_3x3_s2")
            )

            if can_use_static_zero:
                new_module = StaticZeroConv2d.from_conv(target.conv_module)
                _set_module_by_name(model, conv_name, new_module)

                replaced += 1
                static_zero_count += 1

                if self.verbose:
                    print(f"  [STATIC ] {conv_name} ({op}) -> StaticZeroConv2d")
                continue

            # ---------------------------------------------------
            # 2) only_static_zero: 非全零层保持 Dense，不做替换
            # ---------------------------------------------------
            if only_static_zero:
                dense_keep_count += 1
                if self.verbose:
                    print(f"  [KEEP   ] {conv_name} ({op}) -> DenseKeep")
                continue

            # -------------------------------
            # 3) 正常 Sparse / Fused / Linear
            # -------------------------------
            new_module = self._create_sparse_module(target, block)
            if new_module is None:
                dense_keep_count += 1
                if self.verbose:
                    print(f"  [KEEP   ] {conv_name} ({op}) -> DenseKeep (unsupported)")
                continue

            _set_module_by_name(model, conv_name, new_module)
            replaced += 1

            if op in ("fused_conv3x3_lif", "fused_conv1x1_lif", "fused_conv3x3s2_lif"):
                if target.lif_name is not None:
                    _set_module_by_name(model, target.lif_name, nn.Identity())
                    if self.verbose:
                        print(f"  [FUSE-ID] {target.lif_name} -> nn.Identity "
                              f"(fused into {conv_name})")
                fused_count += 1
            else:
                sparse_count += 1

            if self.verbose:
                self._log_replacement(target)

        return replaced, sparse_count, fused_count, static_zero_count, dense_keep_count

    def _create_sparse_module(
        self,
        target: ReplacementTarget,
        block: Optional[int]
    ) -> Optional[nn.Module]:
        op = target.op_type

        # ── Standalone Conv2d ──
        if op in ("conv2d_3x3", "conv2d_1x1", "conv2d_3x3_s2"):
            sparse = SparseConv2d.from_dense(target.conv_module, block_size=block)
            return sparse.to(target.conv_module.weight.device)

        # ── Fused Conv2d + LIF ──
        if op in ("fused_conv3x3_lif", "fused_conv1x1_lif", "fused_conv3x3s2_lif"):
            from Ops.fused_sparse_conv_lif import FusedSparseConvLIF
            if target.lif_module is None:
                sparse = SparseConv2d.from_dense(target.conv_module, block_size=block)
                return sparse.to(target.conv_module.weight.device)
            fused = FusedSparseConvLIF.from_conv_and_lif(
                target.conv_module,
                target.lif_module,
                block_size=block
            )
            return fused.to(target.conv_module.weight.device)

        # ── Linear ──
        if op == "linear":
            from Ops.sparse_linear import SparseLinear
            if not isinstance(target.conv_module, nn.Linear):
                return None
            sparse = SparseLinear.from_dense(target.conv_module)
            return sparse.to(target.conv_module.weight.device)

        return None

    def _log_replacement(self, target: ReplacementTarget):
        op = target.op_type

        if op == "linear":
            in_f = target.conv_module.in_features
            out_f = target.conv_module.out_features
            print(f"  [REPLACE] {target.conv_name} (linear {in_f}->{out_f}) "
                  f"<- {target.spike_name}")

        elif "fused" in op:
            info = display_block_info(target)
            fused_with = target.lif_name or "?"
            print(f"  [FUSE   ] {target.conv_name} ({op}) + {fused_with} "
                  f"<- {target.spike_name}, {info}")

        else:
            info = display_block_info(target)
            print(f"  [REPLACE] {target.conv_name} ({op}) "
                  f"<- {target.spike_name}, {info}")