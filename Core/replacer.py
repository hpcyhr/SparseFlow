"""
自动算子替换逻辑 — 将目标 Conv2d / Linear 替换为稀疏版本
"""

from typing import Dict, List, Optional
import torch.nn as nn

from sparseflow.core.analyzer import ReplacementTarget
from sparseflow.ops.sparse_conv2d import SparseConv2d


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """
    按 dot-separated name 替换 model 中的子模块。
    例如 name="layer1.0.conv1" 会替换 model.layer1[0].conv1
    """
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    setattr(parent, parts[-1], new_module)


class ModuleReplacer:
    """将分析器识别的目标算子替换为稀疏加速版本"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def replace(self, model: nn.Module, targets: List[ReplacementTarget],
                block_sizes: Optional[Dict[str, int]] = None) -> int:
        """
        执行替换，返回成功替换的算子数量。

        Args:
            model: 原模型（原地修改）
            targets: 分析器输出的替换目标列表
            block_sizes: 手动覆盖 block 大小 {layer_name: block_size}
        """
        replaced = 0

        for target in targets:
            block = target.block_size
            if block_sizes and target.conv_name in block_sizes:
                block = block_sizes[target.conv_name]

            if block is None:
                if self.verbose:
                    print(f"  [SKIP] {target.conv_name}: spatial too small, skipping")
                continue

            new_module = self._create_sparse_module(target, block)
            if new_module is None:
                continue

            _set_module_by_name(model, target.conv_name, new_module)
            replaced += 1

            if self.verbose:
                print(f"  [REPLACE] {target.conv_name} ({target.op_type}) "
                      f"<- {target.spike_name}, block={block}")

        return replaced

    def _create_sparse_module(self, target: ReplacementTarget, block: int
                              ) -> Optional[nn.Module]:
        """根据 target 类型创建对应的稀疏 Module"""
        conv = target.conv_module

        if target.op_type == "conv2d_3x3":
            return SparseConv2d.from_dense(conv, block_size=block)

        elif target.op_type == "conv2d_1x1":
            # TODO: 实现 1x1 稀疏卷积
            return SparseConv2d.from_dense(conv, block_size=block)

        elif target.op_type == "linear":
            # TODO: 实现稀疏 Linear
            return None

        return None