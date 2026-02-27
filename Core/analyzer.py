"""
网络拓扑分析器 — 基于 torch.fx 计算图搜索，识别脉冲后继 Conv2d

核心改进（相比旧版线性搜索）：
  1. 使用 torch.fx 符号追踪构建计算图
  2. 通过 node.users 递归查找脉冲源的所有下游 Conv2d
  3. 透明层穿透：Dropout / Flatten / Identity / Pooling / view / reshape 等不阻断搜索
  4. 支持分叉：同一脉冲源可指向多个后继 Conv2d（例如 ResNet 主路 + Shortcut）
  5. 仅替换 nn.Conv2d，不处理 nn.Linear / nn.BatchNorm2d

Fallback：若 fx.symbolic_trace 失败（spikingjelly 的有状态模块不可追踪），
         自动回退到基于 forward hook 的搜索策略。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Type
import warnings
import torch
import torch.nn as nn

from sparseflow.core.registry import SpikeOpRegistry
from sparseflow.utils.block_selector import select_block_size


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class ReplacementTarget:
    """描述一个待替换的 Conv2d 算子"""
    conv_name: str                   # Conv 层的 module name (dot-separated)
    conv_module: nn.Module           # 原始 nn.Conv2d module
    spike_name: str                  # 上游脉冲算子的 name
    op_type: str                     # "conv2d_3x3" | "conv2d_1x1"
    block_size: Optional[int]        # 推荐的 block 大小（None 表示跳过）
    input_h: int = 0                 # 输入特征图高度
    input_w: int = 0                 # 输入特征图宽度


# ============================================================================
# 透明层类型（搜索时穿透，不阻断递归）
# ============================================================================

# 这些层不改变数据的稀疏性质，搜索时应"穿透"继续向下
_TRANSPARENT_MODULES = (
    nn.Dropout, nn.Dropout2d, nn.Dropout3d,
    nn.Identity,
    nn.Flatten,
    nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d,
    nn.AvgPool2d, nn.MaxPool2d,
    nn.BatchNorm2d,      # BN 不改变零/非零结构，穿透
    nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.GELU, nn.SiLU,
)

# torch.fx 中的函数调用节点，也应穿透
_TRANSPARENT_FUNCTIONS = {
    'dropout', 'flatten', 'reshape', 'view', 'contiguous',
    'permute', 'transpose', 'unsqueeze', 'squeeze',
    'adaptive_avg_pool2d', 'adaptive_max_pool2d',
    'avg_pool2d', 'max_pool2d',
    'batch_norm',
    'relu', 'relu_', 'gelu', 'silu', 'leaky_relu',
}

# 方法调用也需穿透
_TRANSPARENT_METHODS = {
    'view', 'reshape', 'contiguous', 'permute', 'transpose',
    'unsqueeze', 'squeeze', 'flatten', 'float', 'half',
    'mean',  # global avg pool 可能用 .mean()
}


def _is_transparent_node(node, modules_dict: dict) -> bool:
    """判断一个 fx.Node 是否为透明层（应穿透继续搜索）"""
    if node.op == 'call_module':
        mod = modules_dict.get(node.target)
        if mod is not None:
            return isinstance(mod, _TRANSPARENT_MODULES)
    elif node.op == 'call_function':
        fname = getattr(node.target, '__name__', str(node.target))
        return fname in _TRANSPARENT_FUNCTIONS
    elif node.op == 'call_method':
        return node.target in _TRANSPARENT_METHODS
    return False


def _is_conv2d_node(node, modules_dict: dict) -> bool:
    """判断 fx.Node 是否为 nn.Conv2d"""
    if node.op == 'call_module':
        mod = modules_dict.get(node.target)
        return isinstance(mod, nn.Conv2d)
    return False


def _is_spike_node(node, modules_dict: dict, registry: SpikeOpRegistry) -> bool:
    """判断 fx.Node 是否为脉冲源"""
    if node.op == 'call_module':
        mod = modules_dict.get(node.target)
        if mod is not None:
            return registry.is_spike_op(mod)
    return False


# ============================================================================
# 基于 torch.fx 的计算图分析器
# ============================================================================

class NetworkAnalyzer:
    """
    分析 SNN 模型拓扑，找到所有 spike_op -> Conv2d 的替换目标。

    优先使用 torch.fx 计算图搜索；若 fx.symbolic_trace 失败，
    自动回退到基于 forward hook 的线性搜索。
    """

    # 回退模式的搜索窗口
    FALLBACK_SEARCH_WINDOW = 15

    # BFS 递归深度上限（防止无限循环）
    MAX_BFS_DEPTH = 20

    def __init__(self, registry: SpikeOpRegistry):
        self.registry = registry

    def analyze(self, model: nn.Module, sample_input: Optional[torch.Tensor] = None
                ) -> List[ReplacementTarget]:
        """
        分析模型，返回所有可替换的 Conv2d 目标。

        Args:
            model: SNN 模型
            sample_input: 可选的样本输入 (T, B, C, H, W)，用于推断各层输入 shape

        Returns:
            List[ReplacementTarget]
        """
        # 推断输入 shape（无论哪种搜索模式都需要）
        input_shapes = {}
        if sample_input is not None:
            input_shapes = self._infer_input_shapes(model, sample_input)

        # 优先尝试 torch.fx 计算图搜索
        try:
            targets = self._analyze_fx(model, input_shapes)
            if targets:
                return targets
            # fx 成功但没找到目标，可能是 trace 不完整，回退
        except Exception as e:
            warnings.warn(
                f"[SparseFlow] torch.fx symbolic trace 失败: {e}\n"
                f"  回退到基于 forward hook 的线性搜索模式。"
            )

        # Fallback: 基于 named_modules 的线性搜索
        return self._analyze_fallback(model, input_shapes)

    # ====================================================================
    # 方法一：torch.fx 计算图搜索
    # ====================================================================

    def _analyze_fx(self, model: nn.Module, input_shapes: Dict) -> List[ReplacementTarget]:
        """基于 torch.fx 的计算图搜索"""
        # symbolic_trace
        graph_module = torch.fx.symbolic_trace(model)
        modules_dict = dict(graph_module.named_modules())

        # 1. 定位所有脉冲源节点
        spike_nodes = []
        for node in graph_module.graph.nodes:
            if _is_spike_node(node, modules_dict, self.registry):
                spike_nodes.append(node)

        # 2. 对每个脉冲源，BFS 递归查找下游 Conv2d
        targets = []
        visited_convs: Set[str] = set()

        for spike_node in spike_nodes:
            spike_name = spike_node.target  # module name
            downstream_convs = self._bfs_find_convs(
                spike_node, modules_dict, visited_convs
            )

            for conv_name in downstream_convs:
                conv_module = modules_dict[conv_name]
                target = self._make_target(conv_name, conv_module, spike_name, input_shapes)
                if target is not None:
                    targets.append(target)
                    visited_convs.add(conv_name)

        return targets

    def _bfs_find_convs(self, spike_node, modules_dict: dict,
                        visited_convs: Set[str]) -> List[str]:
        """
        BFS 从脉冲源出发，穿透透明层，找到所有直接下游 Conv2d。

        支持分叉：spike → [BN → Conv_A, Downsample → Conv_B]
        """
        found_convs: List[str] = []
        queue: List[Tuple] = []  # (node, depth)
        seen: Set[str] = set()

        # 初始化：脉冲源的所有 user
        for user in spike_node.users:
            queue.append((user, 0))

        while queue:
            node, depth = queue.pop(0)

            # 防止无限循环
            if depth > self.MAX_BFS_DEPTH:
                continue

            node_id = f"{node.op}:{node.target}"
            if node_id in seen:
                continue
            seen.add(node_id)

            # 找到 Conv2d → 记录
            if _is_conv2d_node(node, modules_dict):
                conv_name = node.target
                if conv_name not in visited_convs:
                    found_convs.append(conv_name)
                continue  # 不再向下搜索（Conv2d 是终点）

            # 遇到另一个脉冲源 → 停止（不属于当前脉冲源的下游）
            if _is_spike_node(node, modules_dict, self.registry):
                continue

            # 透明层 → 穿透，继续向下搜索所有 user
            if _is_transparent_node(node, modules_dict):
                for user in node.users:
                    queue.append((user, depth + 1))
                continue

            # 其他节点（如 getattr、add 等）→ 也尝试穿透
            # 这是为了处理 ResNet 中的残差连接 (add 操作)
            if node.op in ('call_function', 'call_method'):
                # add, mul 等运算的输出可能还会流向下一个层
                for user in node.users:
                    queue.append((user, depth + 1))
            elif node.op == 'call_module':
                # 非透明的 module，但不是 Conv2d 也不是脉冲源 → 尝试穿透
                # （例如自定义模块、Sequential 容器等）
                for user in node.users:
                    queue.append((user, depth + 1))

        return found_convs

    # ====================================================================
    # 方法二：基于 forward hook 的线性搜索（Fallback）
    # ====================================================================

    def _analyze_fallback(self, model: nn.Module, input_shapes: Dict
                          ) -> List[ReplacementTarget]:
        """
        回退到基于 named_modules 的线性搜索。

        改进点（相比旧版）：
          - 搜索窗口扩大到 15
          - 同一脉冲源可匹配多个后继 Conv2d（支持分叉）
          - 穿透 BN / Dropout / Flatten / Pooling
        """
        module_list = list(model.named_modules())
        targets = []
        visited_convs: Set[str] = set()

        for i, (name, module) in enumerate(module_list):
            if not self.registry.is_spike_op(module):
                continue

            # 从 spike_op 向后搜索所有可替换的 Conv2d（不止一个）
            for j in range(i + 1, min(i + self.FALLBACK_SEARCH_WINDOW, len(module_list))):
                next_name, next_module = module_list[j]

                if next_name in visited_convs:
                    continue

                # 遇到另一个 spike op → 停止搜索
                if self.registry.is_spike_op(next_module):
                    break

                # 透明层 → 跳过，继续搜索
                if isinstance(next_module, _TRANSPARENT_MODULES):
                    continue

                # Conv2d → 尝试匹配
                if isinstance(next_module, nn.Conv2d):
                    target = self._make_target(next_name, next_module, name, input_shapes)
                    if target is not None:
                        targets.append(target)
                        visited_convs.add(next_name)
                    # 注意：不 break，继续搜索同一脉冲源的其他后继 Conv2d
                    continue

        return targets

    # ====================================================================
    # 公共工具
    # ====================================================================

    def _make_target(self, conv_name: str, conv_module: nn.Conv2d,
                     spike_name: str, input_shapes: Dict
                     ) -> Optional[ReplacementTarget]:
        """
        尝试将一个 Conv2d 构造为 ReplacementTarget。

        仅匹配：
          - 3×3 conv, stride=1, padding=1, groups=1
          - 1×1 conv, stride=1, padding=0, groups=1
        """
        if not isinstance(conv_module, nn.Conv2d):
            return None

        k = conv_module.kernel_size
        s = conv_module.stride
        p = conv_module.padding
        g = conv_module.groups

        # 匹配 3×3 标准卷积
        if k == (3, 3) and s == (1, 1) and p == (1, 1) and g == 1:
            op_type = "conv2d_3x3"
        # 匹配 1×1 点卷积
        elif k == (1, 1) and s == (1, 1) and p == (0, 0) and g == 1:
            op_type = "conv2d_1x1"
        else:
            return None

        # 推断特征图尺寸和 block size
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