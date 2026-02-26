"""
网络拓扑分析器 — 识别脉冲后继算子，生成替换目标列表

策略：遍历 named_modules，找到每个 spike_op 后继的可替换算子（Conv2d / Linear 等）。
使用固定搜索窗口（向后 10 个模块），覆盖同 BasicBlock 内和跨 block 边界的情况。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type
import torch
import torch.nn as nn

from sparseflow.core.registry import SpikeOpRegistry
from sparseflow.utils.block_selector import select_block_size


@dataclass
class ReplacementTarget:
    """描述一个待替换的算子"""
    conv_name: str                   # Conv 层的 module name
    conv_module: nn.Module           # 原始 Conv2d / Linear module
    spike_name: str                  # 上游脉冲算子的 name
    op_type: str                     # "conv2d_3x3" | "conv2d_1x1" | "linear"
    block_size: Optional[int]        # 推荐的 block 大小（None 表示跳过）
    input_h: int = 0                 # 输入特征图高度
    input_w: int = 0                 # 输入特征图宽度


class NetworkAnalyzer:
    """
    分析 SNN 模型拓扑，找到所有 spike_op -> target_op 的替换目标。
    """

    # 搜索窗口大小：从 spike_op 向后最多搜索多少个模块
    SEARCH_WINDOW = 10

    def __init__(self, registry: SpikeOpRegistry):
        self.registry = registry

    def analyze(self, model: nn.Module, sample_input: Optional[torch.Tensor] = None
                ) -> List[ReplacementTarget]:
        """
        分析模型，返回所有可替换目标。

        Args:
            model: SNN 模型
            sample_input: 可选的样本输入 (T, B, C, H, W)，用于推断各层输入 shape。
                          如果不提供，则不做 shape 推断，block_size 需要手动指定。

        Returns:
            List[ReplacementTarget]
        """
        module_list = list(model.named_modules())

        # 如果提供了 sample_input，先跑一次推断各层输入 shape
        input_shapes = {}
        if sample_input is not None:
            input_shapes = self._infer_input_shapes(model, sample_input)

        targets = []
        visited_convs = set()

        for i, (name, module) in enumerate(module_list):
            if not self.registry.is_spike_op(module):
                continue

            # 从 spike_op 向后搜索可替换的算子
            for j in range(i + 1, min(i + self.SEARCH_WINDOW, len(module_list))):
                next_name, next_module = module_list[j]

                if next_name in visited_convs:
                    continue

                target = self._try_match(next_name, next_module, name, input_shapes)
                if target is not None:
                    targets.append(target)
                    visited_convs.add(next_name)
                    break  # 每个 spike_op 只匹配一个后继

        return targets

    def _try_match(self, conv_name: str, conv_module: nn.Module,
                   spike_name: str, input_shapes: Dict
                   ) -> Optional[ReplacementTarget]:
        """
        尝试将一个模块匹配为可替换目标。
        """
        if not isinstance(conv_module, nn.Conv2d):
            # TODO: 支持 nn.Linear 等
            return None

        k = conv_module.kernel_size
        s = conv_module.stride
        p = conv_module.padding
        g = conv_module.groups

        # 匹配 3×3 conv, stride=1, padding=1, groups=1 (标准卷积)
        if k == (3, 3) and s == (1, 1) and p == (1, 1) and g == 1:
            op_type = "conv2d_3x3"
        # 匹配 1×1 conv, stride=1, padding=0, groups=1
        elif k == (1, 1) and s == (1, 1) and p == (0, 0) and g == 1:
            op_type = "conv2d_1x1"
        else:
            return None

        # 推断 block size
        H, W = 0, 0
        ishape = input_shapes.get(conv_name)
        if ishape is not None:
            if len(ishape) == 5:
                H, W = ishape[3], ishape[4]
            elif len(ishape) == 4:
                H, W = ishape[2], ishape[3]

        block = select_block_size(H, W) if H > 0 else None

        return ReplacementTarget(
            conv_name=conv_name,
            conv_module=conv_module,
            spike_name=spike_name,
            op_type=op_type,
            block_size=block,
            input_h=H,
            input_w=W,
        )

    @staticmethod
    def _infer_input_shapes(model: nn.Module, sample_input: torch.Tensor) -> Dict[str, tuple]:
        """跑一次前向，通过 hook 记录每个模块的输入 shape"""
        input_shapes = {}
        hooks = []

        def make_hook(name):
            def hook(m, inp, out):
                if isinstance(inp, (tuple, list)) and len(inp) > 0:
                    x = inp[0]
                    if isinstance(x, torch.Tensor):
                        input_shapes[name] = tuple(x.shape)
            return hook

        for name, module in model.named_modules():
            hooks.append(module.register_forward_hook(make_hook(name)))

        # spikingjelly 需要 reset
        try:
            from spikingjelly.activation_based import functional as sj_func
            sj_func.reset_net(model)
        except ImportError:
            pass

        with torch.no_grad():
            _ = model(sample_input)

        for h in hooks:
            h.remove()

        return input_shapes