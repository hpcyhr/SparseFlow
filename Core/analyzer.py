"""
网络拓扑分析器 — v10.4 去中心化版

设计原则：Analyzer 不决定 block 策略。
  - ReplacementTarget.block_size 默认 None（延迟到 kernel 动态决策）。
  - display_block_info() 从 Kernels.conv2d._select_block_sizes 导入，
    保证显示值与实际 kernel 行为一致。
  - BFS 穿透所有非终端节点（仅 Conv2d / spike_op 阻断），
    完整覆盖 ResNet Shortcut 分叉。
  - Fallback 透明层集合与 FX 模式完全同步。
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import warnings
import torch
import torch.nn as nn

from Core.registry import SpikeOpRegistry


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class ReplacementTarget:
    """描述一个待替换的 Conv2d 算子"""
    conv_name: str
    conv_module: nn.Module
    spike_name: str
    op_type: str                     # "conv2d_3x3" | "conv2d_1x1"
    block_size: Optional[int] = None # None → kernel 动态决策
    input_h: int = 0
    input_w: int = 0


# ============================================================================
# 透明层（FX + Fallback 共用）
# ============================================================================

_TRANSPARENT_MODULES = (
    nn.Dropout, nn.Dropout2d, nn.Dropout3d,
    nn.Identity,
    nn.Flatten,
    nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d,
    nn.AvgPool2d, nn.MaxPool2d,
    nn.BatchNorm2d,
    nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.GELU, nn.SiLU,
    nn.Sequential,
)

_TRANSPARENT_FUNCTIONS = {
    'dropout', 'flatten', 'reshape', 'view', 'contiguous',
    'permute', 'transpose', 'unsqueeze', 'squeeze',
    'adaptive_avg_pool2d', 'adaptive_max_pool2d',
    'avg_pool2d', 'max_pool2d',
    'batch_norm',
    'relu', 'relu_', 'gelu', 'silu', 'leaky_relu',
    'add', 'iadd', 'mul', 'cat', 'getitem',
}

_TRANSPARENT_METHODS = {
    'view', 'reshape', 'contiguous', 'permute', 'transpose',
    'unsqueeze', 'squeeze', 'flatten', 'float', 'half',
    'mean',
    'add', 'add_', '__add__', '__iadd__',
    'mul', 'mul_', '__mul__',
}


def _is_conv2d_node(node, modules_dict: dict) -> bool:
    if node.op == 'call_module':
        mod = modules_dict.get(node.target)
        return isinstance(mod, nn.Conv2d)
    return False


def _is_spike_node(node, modules_dict: dict, registry: SpikeOpRegistry) -> bool:
    if node.op == 'call_module':
        mod = modules_dict.get(node.target)
        if mod is not None:
            return registry.is_spike_op(mod)
    return False


def _is_transparent_fallback(module: nn.Module) -> bool:
    """Fallback 模式透明层判断（与 FX _TRANSPARENT_MODULES 同步）"""
    return isinstance(module, _TRANSPARENT_MODULES)


# ============================================================================
# 日志显示工具 — 从 kernel 导入实际策略
# ============================================================================

def display_block_info(target: ReplacementTarget) -> str:
    """
    格式化显示 v10.4 动态 Block 信息。
    直接调用 kernel 的 _select_block_sizes 保证一致性。
    """
    H, W = target.input_h, target.input_w
    if H <= 0 or W <= 0:
        return "H=? BLOCK_M=?"

    try:
        from Kernels.conv2d import _select_block_sizes
        C_IN = target.conv_module.in_channels if hasattr(target.conv_module, 'in_channels') else 64
        C_OUT = target.conv_module.out_channels if hasattr(target.conv_module, 'out_channels') else 64
        k = 3 if target.op_type == "conv2d_3x3" else 1
        _, _, BLOCK_M, BLOCK_N, _ = _select_block_sizes(H, W, C_IN, C_OUT, k, 1)
        return f"H={H} BLOCK_M={BLOCK_M}"
    except ImportError:
        return f"H={H} BLOCK_M=?"


# ============================================================================
# 分析器
# ============================================================================

class NetworkAnalyzer:
    """
    分析 SNN 模型拓扑，找到所有 spike_op -> Conv2d 的替换目标。

    block_size 策略：完全延迟到 kernel。Analyzer 只做拓扑发现和
    gatekeeper（跳过 min(H,W) < 7 的层）。
    """

    FALLBACK_SEARCH_WINDOW = 15
    MAX_BFS_DEPTH = 20

    def __init__(self, registry: SpikeOpRegistry):
        self.registry = registry

    def analyze(self, model: nn.Module,
                sample_input: Optional[torch.Tensor] = None
                ) -> List[ReplacementTarget]:
        input_shapes = {}
        if sample_input is not None:
            input_shapes = self._infer_input_shapes(model, sample_input)

        try:
            targets = self._analyze_fx(model, input_shapes)
            if targets:
                return targets
        except Exception as e:
            warnings.warn(
                f"[SparseFlow] torch.fx symbolic trace 失败: {e}\n"
                f"  回退到基于 forward hook 的线性搜索模式。"
            )

        return self._analyze_fallback(model, input_shapes)

    # ── FX 模式 ──

    def _analyze_fx(self, model, input_shapes):
        graph_module = torch.fx.symbolic_trace(model)
        modules_dict = dict(graph_module.named_modules())

        spike_nodes = [
            n for n in graph_module.graph.nodes
            if _is_spike_node(n, modules_dict, self.registry)
        ]

        targets = []
        visited_convs: Set[str] = set()

        for spike_node in spike_nodes:
            spike_name = spike_node.target
            for conv_name in self._bfs_find_convs(spike_node, modules_dict, visited_convs):
                conv_module = modules_dict[conv_name]
                target = self._make_target(conv_name, conv_module, spike_name, input_shapes)
                if target is not None:
                    targets.append(target)
                    visited_convs.add(conv_name)

        return targets

    def _bfs_find_convs(self, spike_node, modules_dict, visited_convs):
        """
        BFS：仅 Conv2d 和 spike_op 阻断，其余一律穿透。

        这确保：
          spike → BN → conv3x3 (主路)          ✓ 发现
          spike → downsample.0(conv1x1) (旁路)  ✓ 发现
          spike → ... → add → BN → conv (下一层) ✓ 穿过 add 发现
        """
        found: List[str] = []
        queue: List[Tuple] = [(u, 0) for u in spike_node.users]
        seen: Set[int] = set()

        while queue:
            node, depth = queue.pop(0)

            if depth > self.MAX_BFS_DEPTH:
                continue
            nid = id(node)
            if nid in seen:
                continue
            seen.add(nid)

            if _is_conv2d_node(node, modules_dict):
                conv_name = node.target
                if conv_name not in visited_convs:
                    found.append(conv_name)
                continue  # Conv2d 是终点

            if _is_spike_node(node, modules_dict, self.registry):
                continue  # 另一个脉冲源 → 不穿透

            # 所有其他节点一律穿透
            for user in node.users:
                queue.append((user, depth + 1))

        return found

    # ── Fallback 模式 ──

    def _analyze_fallback(self, model, input_shapes):
        module_list = list(model.named_modules())
        targets = []
        visited_convs: Set[str] = set()

        for i, (name, module) in enumerate(module_list):
            if not self.registry.is_spike_op(module):
                continue

            for j in range(i + 1, min(i + self.FALLBACK_SEARCH_WINDOW, len(module_list))):
                next_name, next_module = module_list[j]

                if next_name in visited_convs:
                    continue

                if self.registry.is_spike_op(next_module):
                    break

                # 与 FX 同步的透明层判断
                if _is_transparent_fallback(next_module):
                    continue

                if isinstance(next_module, nn.Conv2d):
                    target = self._make_target(next_name, next_module, name, input_shapes)
                    if target is not None:
                        targets.append(target)
                        visited_convs.add(next_name)
                    continue

        return targets

    # ── 共用工具 ──

    def _make_target(self, conv_name, conv_module, spike_name, input_shapes):
        """
        构造 ReplacementTarget。block_size 始终为 None（延迟决策）。
        仅在 min(H,W) < 7 时返回 None（gatekeeper）。
        """
        if not isinstance(conv_module, nn.Conv2d):
            return None

        k = conv_module.kernel_size
        s = conv_module.stride
        p = conv_module.padding
        g = conv_module.groups

        if k == (3, 3) and s == (1, 1) and p == (1, 1) and g == 1:
            op_type = "conv2d_3x3"
        elif k == (1, 1) and s == (1, 1) and p == (0, 0) and g == 1:
            op_type = "conv2d_1x1"
        else:
            return None

        H, W = 0, 0
        ishape = input_shapes.get(conv_name)
        if ishape is not None:
            if len(ishape) == 5:
                H, W = ishape[3], ishape[4]
            elif len(ishape) == 4:
                H, W = ishape[2], ishape[3]

        # Gatekeeper
        if H > 0 and min(H, W) < 7:
            return None

        return ReplacementTarget(
            conv_name=conv_name,
            conv_module=conv_module,
            spike_name=spike_name,
            op_type=op_type,
            block_size=None,   # ← 核心改动：不预设，延迟到 kernel
            input_h=H,
            input_w=W,
        )

    @staticmethod
    def _infer_input_shapes(model, sample_input):
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