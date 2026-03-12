"""
网络拓扑分析器

  - FusedTarget: new dataclass for Conv2d → [optional BN] → LIFNode patterns
  - Linear detection: nn.Linear after spike ops → marked as "linear" target
  - Look-ahead fusion: after finding a Conv2d target, peek forward to detect
    if the next non-transparent module is a LIFNode (with optional BN between)
  - ReplacementTarget.op_type extended: "fused_conv3x3_lif", "fused_conv1x1_lif", "linear"
  - ReplacementTarget.lif_name / lif_module: populated for fused targets
  - All v1 functionality preserved (BFS, fallback, gatekeeper, etc.)
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dataclasses import dataclass, field
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
    """
    Describes a replacement target.

    op_type values:
      - "conv2d_3x3"           : standalone sparse conv
      - "conv2d_1x1"           : standalone sparse conv
      - "fused_conv3x3_lif"    : fused conv3x3 + LIF
      - "fused_conv1x1_lif"    : fused conv1x1 + LIF
      - "linear"               : sparse linear
    """
    conv_name: str
    conv_module: nn.Module
    spike_name: str               # upstream spike source
    op_type: str
    block_size: Optional[int] = None
    input_h: int = 0
    input_w: int = 0
    # Fusion fields (populated only for fused_conv*_lif targets)
    lif_name: Optional[str] = None
    lif_module: Optional[nn.Module] = None
    bn_name: Optional[str] = None    # optional BN between conv and LIF


# ============================================================================
# 透明层
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
    'mean', 'add', 'add_', '__add__', '__iadd__',
    'mul', 'mul_', '__mul__',
}


def _is_conv2d_node(node, modules_dict: dict) -> bool:
    if node.op == 'call_module':
        mod = modules_dict.get(node.target)
        return isinstance(mod, nn.Conv2d)
    return False


def _is_linear_node(node, modules_dict: dict) -> bool:
    if node.op == 'call_module':
        mod = modules_dict.get(node.target)
        return isinstance(mod, nn.Linear)
    return False


def _is_spike_node(node, modules_dict: dict, registry: SpikeOpRegistry) -> bool:
    if node.op == 'call_module':
        mod = modules_dict.get(node.target)
        if mod is not None:
            return registry.is_spike_op(mod)
    return False


def _is_transparent_fallback(module: nn.Module) -> bool:
    return isinstance(module, _TRANSPARENT_MODULES)


# ============================================================================
# Block info display (from kernel)
# ============================================================================

def display_block_info(target: ReplacementTarget) -> str:
    H, W = target.input_h, target.input_w
    if H <= 0 or W <= 0:
        return "H=? BLOCK_M=?"
    try:
        from Kernels.conv2d import _select_block_sizes
        C_IN = getattr(target.conv_module, 'in_channels', 64)
        C_OUT = getattr(target.conv_module, 'out_channels', 64)
        k = 3 if '3x3' in target.op_type else 1
        _, _, BLOCK_M, BLOCK_N, _ = _select_block_sizes(H, W, C_IN, C_OUT, k, 1)
        return f"H={H} BLOCK_M={BLOCK_M}"
    except (ImportError, Exception):
        return f"H={H} BLOCK_M=?"


# ============================================================================
# Fusion look-ahead helper
# ============================================================================

def _look_ahead_for_lif(module_list, conv_idx, registry, max_distance=5):
    """
    Starting from conv_idx+1, look ahead up to max_distance modules
    for a LIFNode, allowing only BN or Identity between them.

    Returns:
        (lif_name, lif_module, bn_name_or_None) if fusion pattern found
        None otherwise
    """
    bn_name = None
    for j in range(conv_idx + 1, min(conv_idx + max_distance + 1, len(module_list))):
        name_j, mod_j = module_list[j]

        if isinstance(mod_j, nn.BatchNorm2d):
            bn_name = name_j  # BN is allowed in the fusion chain
            continue

        if isinstance(mod_j, nn.Identity):
            continue

        if registry.is_spike_op(mod_j):
            return (name_j, mod_j, bn_name)

        # Hit something else (another Conv, Linear, etc.) → no fusion
        break

    return None


# ============================================================================
# 分析器
# ============================================================================

class NetworkAnalyzer:
    """
    analyzer: finds Conv2d targets, Linear targets, and fusion patterns.

    Analysis pipeline:
      1. Infer input shapes via forward hooks
      2. Try FX symbolic trace → BFS from spike nodes
      3. Fallback to linear search if FX fails
      4. For each Conv2d target, look ahead for Conv→[BN]→LIF fusion
      5. Detect Linear layers after spike ops
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
        visited: Set[str] = set()

        for spike_node in spike_nodes:
            spike_name = spike_node.target

            # Find downstream Conv2d and Linear
            for target_name, target_type in self._bfs_find_targets(
                    spike_node, modules_dict, visited):
                mod = modules_dict[target_name]

                if target_type == 'conv2d':
                    target = self._make_conv_target(
                        target_name, mod, spike_name, input_shapes, model)
                    if target is not None:
                        targets.append(target)
                        visited.add(target_name)

                elif target_type == 'linear':
                    target = self._make_linear_target(
                        target_name, mod, spike_name, input_shapes)
                    if target is not None:
                        targets.append(target)
                        visited.add(target_name)

        return targets

    def _bfs_find_targets(self, spike_node, modules_dict, visited):
        """BFS: find Conv2d and Linear targets downstream of spike node."""
        found = []
        queue = [(u, 0) for u in spike_node.users]
        seen = set()

        while queue:
            node, depth = queue.pop(0)
            if depth > self.MAX_BFS_DEPTH:
                continue
            nid = id(node)
            if nid in seen:
                continue
            seen.add(nid)

            if _is_conv2d_node(node, modules_dict):
                name = node.target
                if name not in visited:
                    found.append((name, 'conv2d'))
                continue

            if _is_linear_node(node, modules_dict):
                name = node.target
                if name not in visited:
                    found.append((name, 'linear'))
                continue

            if _is_spike_node(node, modules_dict, self.registry):
                continue

            for user in node.users:
                queue.append((user, depth + 1))

        return found

    # ── Fallback 模式 ──

    def _analyze_fallback(self, model, input_shapes):
        module_list = list(model.named_modules())
        targets = []
        visited: Set[str] = set()

        for i, (name, module) in enumerate(module_list):
            if not self.registry.is_spike_op(module):
                continue

            for j in range(i + 1, min(i + self.FALLBACK_SEARCH_WINDOW, len(module_list))):
                next_name, next_module = module_list[j]

                if next_name in visited:
                    continue

                if self.registry.is_spike_op(next_module):
                    break

                if _is_transparent_fallback(next_module):
                    continue

                if isinstance(next_module, nn.Conv2d):
                    target = self._make_conv_target(
                        next_name, next_module, name, input_shapes, model)
                    if target is not None:
                        targets.append(target)
                        visited.add(next_name)
                    continue

                if isinstance(next_module, nn.Linear):
                    target = self._make_linear_target(
                        next_name, next_module, name, input_shapes)
                    if target is not None:
                        targets.append(target)
                        visited.add(next_name)
                    continue

        return targets

    # ── Target constructors ──

    def _make_conv_target(self, conv_name, conv_module, spike_name,
                          input_shapes, model=None):
        """Construct Conv2d target. Optionally detect fusion with downstream LIF."""
        if not isinstance(conv_module, nn.Conv2d):
            return None

        k = conv_module.kernel_size
        s = conv_module.stride
        p = conv_module.padding
        g = conv_module.groups

        if k == (3, 3) and s == (1, 1) and p == (1, 1) and g == 1:
            base_op = "conv2d_3x3"
        elif k == (1, 1) and s == (1, 1) and p == (0, 0) and g == 1:
            base_op = "conv2d_1x1"
        else:
            return None

        H, W = 0, 0
        ishape = input_shapes.get(conv_name)
        if ishape is not None:
            if len(ishape) == 5:
                H, W = ishape[3], ishape[4]
            elif len(ishape) == 4:
                H, W = ishape[2], ishape[3]

        if H > 0 and min(H, W) < 7:
            return None

        # ── Look-ahead for Conv → [BN] → LIF fusion ──
        lif_name = None
        lif_module = None
        bn_name = None
        op_type = base_op

        if model is not None:
            module_list = list(model.named_modules())
            conv_idx = None
            for idx, (mname, _) in enumerate(module_list):
                if mname == conv_name:
                    conv_idx = idx
                    break

            if conv_idx is not None:
                fusion = _look_ahead_for_lif(
                    module_list, conv_idx, self.registry, max_distance=5)
                if fusion is not None:
                    lif_name, lif_module, bn_name = fusion
                    if base_op == "conv2d_3x3":
                        op_type = "fused_conv3x3_lif"
                    else:
                        op_type = "fused_conv1x1_lif"

        return ReplacementTarget(
            conv_name=conv_name,
            conv_module=conv_module,
            spike_name=spike_name,
            op_type=op_type,
            block_size=None,
            input_h=H, input_w=W,
            lif_name=lif_name,
            lif_module=lif_module,
            bn_name=bn_name,
        )

    def _make_linear_target(self, linear_name, linear_module, spike_name,
                            input_shapes):
        """Construct Linear target."""
        if not isinstance(linear_module, nn.Linear):
            return None

        return ReplacementTarget(
            conv_name=linear_name,    # reuse field name for consistency
            conv_module=linear_module,
            spike_name=spike_name,
            op_type="linear",
            block_size=None,
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