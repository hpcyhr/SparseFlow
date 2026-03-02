"""
自动算子替换 — v10.4 透传模式

核心改动：block_size=None 表示"由 kernel 动态决策"，不再表示"跳过"。
  - Gatekeeper（是否跳过）完全由 Analyzer 负责（min(H,W)<7 → 不生成 target）。
  - Replacer 对所有 target 无条件执行替换。
  - block_size=None 透传给 SparseConv2d → sparse_conv2d_forward → _select_block_sizes。
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
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


class ModuleReplacer:
    """将分析器识别的目标 Conv2d 替换为稀疏加速版本"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def replace(self, model: nn.Module, targets: List[ReplacementTarget],
                block_sizes: Optional[Dict[str, int]] = None) -> int:
        """
        执行替换，返回成功替换数。

        block_size 流向：
          target.block_size (通常 None)
            → block_sizes 手动覆盖（如有）
            → SparseConv2d.from_dense(block_size=...)
            → sparse_conv2d_forward(block_size=...)
            → _select_block_sizes(H, W, ...) 动态决策
        """
        replaced = 0

        for target in targets:
            # 手动覆盖优先，否则使用 target 的值（通常 None）
            block = target.block_size
            if block_sizes and target.conv_name in block_sizes:
                block = block_sizes[target.conv_name]

            new_module = self._create_sparse_module(target, block)
            if new_module is None:
                continue

            _set_module_by_name(model, target.conv_name, new_module)
            replaced += 1

            if self.verbose:
                info = display_block_info(target)
                print(f"  [REPLACE] {target.conv_name} ({target.op_type}) "
                      f"<- {target.spike_name}, {info}")

        return replaced

    def _create_sparse_module(self, target: ReplacementTarget,
                              block: Optional[int]) -> Optional[nn.Module]:
        conv = target.conv_module

        if target.op_type in ("conv2d_3x3", "conv2d_1x1"):
            # block=None → SparseConv2d 存储 None → kernel 动态决策
            sparse = SparseConv2d.from_dense(conv, block_size=block)
            device = conv.weight.device
            sparse = sparse.to(device)
            return sparse

        return None