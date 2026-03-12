"""
自动算子替换

  - Fused replacement: Conv2d (fused target) → FusedSparseConvLIF,
    downstream LIFNode → nn.Identity (avoid double-activation)
  - Linear replacement: nn.Linear → SparseLinear (tile-level Dynamic-K)
  - All Conv2d standalone replacement preserved
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from typing import Dict, List, Optional
import torch.nn as nn

from Core.analyzer import ReplacementTarget, display_block_info
from Ops.sparse_conv2d import SparseConv2d


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
      1. Standalone Conv2d → SparseConv2d
      2. Fused Conv2d+LIF → FusedSparseConvLIF + Identity
      3. Linear → SparseLinear
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def replace(self, model: nn.Module, targets: List[ReplacementTarget],
                block_sizes: Optional[Dict[str, int]] = None) -> int:
        replaced = 0

        for target in targets:
            block = target.block_size
            if block_sizes and target.conv_name in block_sizes:
                block = block_sizes[target.conv_name]

            new_module = self._create_sparse_module(target, block)
            if new_module is None:
                continue

            _set_module_by_name(model, target.conv_name, new_module)
            replaced += 1

            # ── For fused targets: replace downstream LIFNode with Identity ──
            if target.op_type in ("fused_conv3x3_lif", "fused_conv1x1_lif"):
                if target.lif_name is not None:
                    _set_module_by_name(model, target.lif_name, nn.Identity())
                    if self.verbose:
                        print(f"  [FUSE-ID] {target.lif_name} → nn.Identity "
                              f"(fused into {target.conv_name})")

            if self.verbose:
                self._log_replacement(target)

        return replaced

    def _create_sparse_module(self, target: ReplacementTarget,
                              block: Optional[int]) -> Optional[nn.Module]:
        op = target.op_type

        # ── Standalone Conv2d ──
        if op in ("conv2d_3x3", "conv2d_1x1"):
            sparse = SparseConv2d.from_dense(target.conv_module, block_size=block)
            return sparse.to(target.conv_module.weight.device)

        # ── Fused Conv2d + LIF ──
        if op in ("fused_conv3x3_lif", "fused_conv1x1_lif"):
            from Ops.fused_sparse_conv_lif import FusedSparseConvLIF
            if target.lif_module is None:
                # Fallback to standalone if LIF not found
                sparse = SparseConv2d.from_dense(target.conv_module, block_size=block)
                return sparse.to(target.conv_module.weight.device)
            fused = FusedSparseConvLIF.from_conv_and_lif(
                target.conv_module, target.lif_module,
                block_size=block)
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
            print(f"  [REPLACE] {target.conv_name} (linear {in_f}→{out_f}) "
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