import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dataclasses import dataclass
from typing import List, Optional, Set
import warnings
import torch
import torch.nn as nn

from Core.registry import SpikeOpRegistry


@dataclass
class ReplacementTarget:
    conv_name: str
    conv_module: nn.Module
    spike_name: str
    op_type: str
    block_size: Optional[int] = None
    input_h: int = 0
    input_w: int = 0
    lif_name: Optional[str] = None
    lif_module: Optional[nn.Module] = None
    bn_name: Optional[str] = None
    bn_module: Optional[nn.Module] = None


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


def _is_conv2d_node(node, modules_dict: dict) -> bool:
    return node.op == 'call_module' and isinstance(modules_dict.get(node.target), nn.Conv2d)


def _is_linear_node(node, modules_dict: dict) -> bool:
    return node.op == 'call_module' and isinstance(modules_dict.get(node.target), nn.Linear)


def _is_spike_node(node, modules_dict: dict, registry: SpikeOpRegistry) -> bool:
    if node.op != 'call_module':
        return False
    mod = modules_dict.get(node.target)
    return mod is not None and registry.is_spike_op(mod)


def _is_transparent_fallback(module: nn.Module) -> bool:
    return isinstance(module, _TRANSPARENT_MODULES)


def display_block_info(target: ReplacementTarget) -> str:
    H, W = target.input_h, target.input_w
    if H <= 0 or W <= 0:
        return 'H=? BLOCK_M=?'
    try:
        from Kernels.conv2d import _select_block_sizes
        C_IN = getattr(target.conv_module, 'in_channels', 64)
        C_OUT = getattr(target.conv_module, 'out_channels', 64)
        k = 3 if '3x3' in target.op_type else 1
        s = 2 if ('s2' in target.op_type) else 1
        _, _, BLOCK_M, _, _ = _select_block_sizes(H, W, C_IN, C_OUT, k, s)
        return f'H={H} BLOCK_M={BLOCK_M}'
    except Exception:
        return f'H={H} BLOCK_M=?'


def _eligible_direct_fusion_conv_name(conv_name: str) -> bool:
    # Conservative rule for SpikingJelly ResNet/Bottleneck:
    #   conv1 -> bn1 -> sn1 : safe to fuse
    #   conv2 -> bn2 -> sn2 : safe to fuse
    #   conv3 -> bn3 -> add -> sn3 : NOT safe to fuse
    # We therefore only allow conv1/conv2 local fusion here.
    leaf = conv_name.rsplit('.', 1)[-1]
    return leaf in ('conv1', 'conv2')


def _look_ahead_for_lif(module_list, conv_idx, registry, max_distance=5):
    conv_name, _ = module_list[conv_idx]
    if not _eligible_direct_fusion_conv_name(conv_name):
        return None

    bn_name = None
    bn_module = None
    for j in range(conv_idx + 1, min(conv_idx + max_distance + 1, len(module_list))):
        name_j, mod_j = module_list[j]
        if isinstance(mod_j, nn.BatchNorm2d):
            bn_name = name_j
            bn_module = mod_j
            continue
        if isinstance(mod_j, nn.Identity):
            continue
        if registry.is_spike_op(mod_j):
            return (name_j, mod_j, bn_name, bn_module)
        break
    return None


class NetworkAnalyzer:
    FALLBACK_SEARCH_WINDOW = 15
    MAX_BFS_DEPTH = 20

    def __init__(self, registry: SpikeOpRegistry):
        self.registry = registry

    def analyze(self, model: nn.Module, sample_input: Optional[torch.Tensor] = None) -> List[ReplacementTarget]:
        input_shapes = {}
        if sample_input is not None:
            input_shapes = self._infer_input_shapes(model, sample_input)
        try:
            targets = self._analyze_fx(model, input_shapes)
            if targets:
                return targets
        except Exception as e:
            warnings.warn(
                f'[SparseFlow] torch.fx symbolic trace 失败: {e}\n'
                f'  回退到基于 forward hook 的线性搜索模式。'
            )
        return self._analyze_fallback(model, input_shapes)

    def _analyze_fx(self, model, input_shapes):
        graph_module = torch.fx.symbolic_trace(model)
        modules_dict = dict(graph_module.named_modules())
        spike_nodes = [n for n in graph_module.graph.nodes if _is_spike_node(n, modules_dict, self.registry)]
        targets = []
        visited: Set[str] = set()
        for spike_node in spike_nodes:
            spike_name = spike_node.target
            for target_name, target_type in self._bfs_find_targets(spike_node, modules_dict, visited):
                mod = modules_dict[target_name]
                if target_type == 'conv2d':
                    target = self._make_conv_target(target_name, mod, spike_name, input_shapes, model)
                    if target is not None:
                        targets.append(target)
                        visited.add(target_name)
                elif target_type == 'linear':
                    target = self._make_linear_target(target_name, mod, spike_name, input_shapes)
                    if target is not None:
                        targets.append(target)
                        visited.add(target_name)
        return targets

    def _bfs_find_targets(self, spike_node, modules_dict, visited):
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
                    target = self._make_conv_target(next_name, next_module, name, input_shapes, model)
                    if target is not None:
                        targets.append(target)
                        visited.add(next_name)
                    continue
                if isinstance(next_module, nn.Linear):
                    target = self._make_linear_target(next_name, next_module, name, input_shapes)
                    if target is not None:
                        targets.append(target)
                        visited.add(next_name)
                    continue
        return targets

    def _make_conv_target(self, conv_name, conv_module, spike_name, input_shapes, model=None):
        if not isinstance(conv_module, nn.Conv2d):
            return None
        k = conv_module.kernel_size
        s = conv_module.stride
        p = conv_module.padding
        g = conv_module.groups
        if k == (3, 3) and s == (1, 1) and p == (1, 1) and g == 1:
            base_op = 'conv2d_3x3'
        elif k == (3, 3) and s == (2, 2) and p == (1, 1) and g == 1:
            base_op = 'conv2d_3x3_s2'
        elif k == (1, 1) and s == (1, 1) and p == (0, 0) and g == 1:
            base_op = 'conv2d_1x1'
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

        lif_name = None
        lif_module = None
        bn_name = None
        bn_module = None
        op_type = base_op
        if model is not None:
            module_list = list(model.named_modules())
            conv_idx = None
            for idx, (mname, _) in enumerate(module_list):
                if mname == conv_name:
                    conv_idx = idx
                    break
            if conv_idx is not None:
                fusion = _look_ahead_for_lif(module_list, conv_idx, self.registry, max_distance=5)
                if fusion is not None and _eligible_direct_fusion_conv_name(conv_name):
                    lif_name, lif_module, bn_name, bn_module = fusion
                    if base_op == 'conv2d_3x3_s2':
                        op_type = 'fused_conv3x3s2_lif'
                    elif base_op == 'conv2d_3x3':
                        op_type = 'fused_conv3x3_lif'
                    else:
                        op_type = 'fused_conv1x1_lif'
        return ReplacementTarget(
            conv_name=conv_name,
            conv_module=conv_module,
            spike_name=spike_name,
            op_type=op_type,
            block_size=None,
            input_h=H,
            input_w=W,
            lif_name=lif_name,
            lif_module=lif_module,
            bn_name=bn_name,
            bn_module=bn_module,
        )

    def _make_linear_target(self, linear_name, linear_module, spike_name, input_shapes):
        if not isinstance(linear_module, nn.Linear):
            return None
        return ReplacementTarget(
            conv_name=linear_name,
            conv_module=linear_module,
            spike_name=spike_name,
            op_type='linear',
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
