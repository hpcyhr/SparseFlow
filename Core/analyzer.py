"""
Core/analyzer.py — Replacement-target discovery.

Finds modules that follow a spike source (LIF / IF / …) and are eligible
for sparse replacement. Runs in three modes in order, falling back on
failure:

    1. torch.fx symbolic trace  → BFS from spike nodes to the next
       conv1d / conv2d / conv3d / linear / attention consumer
    2. Runtime-order scan       → forward-hook event trace under a sample
       input, then pair each spike event with its next target event
    3. Module-order fallback    → linear scan over named_modules()

Attention blocks are discovered by a separate pass because they're
composite `nn.Module`s (q/k/v/proj + attn_lif) rather than leaf ops.

Round 4 cleanup (no semantic changes):
  - Removed Conv+LIF fusion detection (`_look_ahead_for_lif`,
    `_eligible_direct_fusion_conv_name`, fused op_type rewriting inside
    `_make_conv2d_target`). Fused operators are out of scope for the
    current codebase.
  - Removed `lif_name`, `lif_module`, `bn_name`, `bn_module` fields from
    `ReplacementTarget`. Bench code that defensively assigned these via
    `hasattr(t, "lif_name")` continues to work unchanged.
  - Removed fused op_type entries from `display_block_info`.

Round 6 cleanup (pool replacement disabled):
  - `nn.MaxPool2d` and `nn.AvgPool2d` are treated as transparent again and
    are no longer emitted as replacement targets by any discovery path.
  - This keeps pool layers on their native dense implementation and removes
    them from SparseFlow's analyzer/replacer/dispatch optimization chain.
  - The sparse pool operator implementations remain in the repository, but
    the framework no longer routes models into them.
"""

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn

from Core.registry import SpikeOpRegistry


@dataclass
class ReplacementTarget:
    """Record describing one replaceable module.

    `conv_name` / `conv_module` are historical names — the target may be
    any op type (conv1d/2d/3d, linear, attention block); the field names
    are kept for backward compatibility with existing downstream code.
    """

    conv_name: str
    conv_module: nn.Module
    spike_name: str
    op_type: str

    block_size: Optional[int] = None
    input_h: int = 0
    input_w: int = 0
    input_d: int = 0

    # Optional extension payload for non-conv ops (e.g. attention blocks
    # carry num_heads / head_dim here).
    extra: Optional[dict] = None


_TRANSPARENT_MODULES = (
    nn.Dropout, nn.Dropout2d, nn.Dropout3d,
    nn.Identity,
    nn.Flatten,
    nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
    nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
    nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
    nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.GELU, nn.SiLU,
    nn.Sequential,
)
# Pool layers are intentionally transparent. SparseFlow currently optimizes
# conv / linear / attention paths only; 2d pool replacement is disabled at
# the framework level because it regressed benchmark latency.


# =============================================================================
# Node / module predicates
# =============================================================================

def _is_conv1d_node(node, modules_dict: dict) -> bool:
    return node.op == "call_module" and isinstance(modules_dict.get(node.target), nn.Conv1d)


def _is_conv2d_node(node, modules_dict: dict) -> bool:
    return node.op == "call_module" and isinstance(modules_dict.get(node.target), nn.Conv2d)


def _is_conv3d_node(node, modules_dict: dict) -> bool:
    return node.op == "call_module" and isinstance(modules_dict.get(node.target), nn.Conv3d)


def _is_linear_node(node, modules_dict: dict) -> bool:
    return node.op == "call_module" and isinstance(modules_dict.get(node.target), nn.Linear)


def _is_maxpool2d_node(node, modules_dict: dict) -> bool:
    return node.op == "call_module" and isinstance(modules_dict.get(node.target), nn.MaxPool2d)


def _is_avgpool2d_node(node, modules_dict: dict) -> bool:
    return node.op == "call_module" and isinstance(modules_dict.get(node.target), nn.AvgPool2d)


def _is_attention_like_module(module: nn.Module) -> bool:
    """Heuristic: an attention block carries Linear q/k/v/proj + attn_lif."""
    if module is None:
        return False
    for attr in ("q", "k", "v", "proj"):
        if not hasattr(module, attr):
            return False
        if not isinstance(getattr(module, attr), nn.Linear):
            return False
    if not hasattr(module, "attn_lif"):
        return False
    return True


def _is_attention_node(node, modules_dict: dict) -> bool:
    if node.op != "call_module":
        return False
    return _is_attention_like_module(modules_dict.get(node.target))


def _infer_attention_variant(module: nn.Module) -> str:
    cls = module.__class__.__name__.lower()
    if "qkmix" in cls:
        return "attention_qkmix"
    if hasattr(module, "scale"):
        return "attention_qkav"
    return "attention_linear"


def _is_spike_node(node, modules_dict: dict, registry: SpikeOpRegistry) -> bool:
    if node.op != "call_module":
        return False
    mod = modules_dict.get(node.target)
    return mod is not None and registry.is_spike_op(mod)


def _is_transparent_fallback(module: nn.Module) -> bool:
    return isinstance(module, _TRANSPARENT_MODULES)


def _as_pair(x) -> Tuple[int, int]:
    if isinstance(x, (tuple, list)):
        if len(x) == 1:
            return int(x[0]), int(x[0])
        return int(x[0]), int(x[1])
    return int(x), int(x)


# =============================================================================
# Display helper
# =============================================================================

def display_block_info(target: ReplacementTarget) -> str:
    """Compact one-line summary of a target's spatial shape and tile size."""
    op = target.op_type
    if op in ("conv2d_3x3", "conv2d_3x3_s2", "conv2d_1x1"):
        h, w = target.input_h, target.input_w
        if h <= 0 or w <= 0:
            return "H=? BLOCK_M=?"
        try:
            from Kernels.conv2d import _select_block_sizes

            c_in = getattr(target.conv_module, "in_channels", 64)
            c_out = getattr(target.conv_module, "out_channels", 64)
            k = 3 if "3x3" in op else 1
            s = 2 if "s2" in op else 1
            _, _, block_m, _, _ = _select_block_sizes(h, w, c_in, c_out, k, s)
            return f"H={h} BLOCK_M={block_m}"
        except Exception:
            return f"H={h} BLOCK_M=?"

    if op == "depthwise_conv2d":
        h, w = target.input_h, target.input_w
        return f"DW H={h if h > 0 else '?'} W={w if w > 0 else '?'}"

    if op == "conv1d":
        l = target.input_w
        return f"L={l if l > 0 else '?'}"

    if op == "conv3d":
        d, h, w = target.input_d, target.input_h, target.input_w
        d_s = str(d) if d > 0 else "?"
        h_s = str(h) if h > 0 else "?"
        w_s = str(w) if w > 0 else "?"
        return f"D={d_s} H={h_s} W={w_s}"

    if op in ("attention_qkav", "attention_linear", "attention_qkmix"):
        n, c = target.input_h, target.input_w
        n_s = str(n) if n > 0 else "?"
        c_s = str(c) if c > 0 else "?"
        return f"N={n_s} C={c_s}"

    if op in ("matmul", "bmm"):
        m, k = target.input_h, target.input_w
        m_s = str(m) if m > 0 else "?"
        k_s = str(k) if k > 0 else "?"
        return f"M={m_s} K={k_s}"

    return ""


# =============================================================================
# NetworkAnalyzer
# =============================================================================

class NetworkAnalyzer:
    FALLBACK_SEARCH_WINDOW = 15
    MAX_BFS_DEPTH = 20

    def __init__(self, registry: SpikeOpRegistry):
        self.registry = registry

    def analyze(
        self,
        model: nn.Module,
        sample_input: Optional[torch.Tensor] = None,
    ) -> List[ReplacementTarget]:
        input_shapes = {}
        if sample_input is not None:
            input_shapes = self._infer_input_shapes(model, sample_input)

        # Always collect attention-like block targets as an extra pass.
        attention_targets = self._analyze_attention_modules(model, input_shapes)
        attention_names = {t.conv_name for t in attention_targets}

        fx_error = None
        try:
            targets = self._analyze_fx(model, input_shapes)
            if targets:
                merged = self._merge_targets(targets, attention_targets, attention_names)
                if merged:
                    return merged
        except Exception as err:
            fx_error = err
            warnings.warn(
                f"[SparseFlow] torch.fx symbolic trace failed: {err}\n"
                "  Falling back to runtime-order scan mode."
            )

        if sample_input is not None:
            try:
                rt_targets = self._analyze_runtime_fallback(model, input_shapes, sample_input)
                if rt_targets:
                    merged = self._merge_targets(rt_targets, attention_targets, attention_names)
                    if merged:
                        return merged
            except Exception as err:
                warnings.warn(
                    f"[SparseFlow] runtime-order fallback failed: {err}\n"
                    "  Falling back to linear module-order scan mode."
                )
        elif fx_error is None:
            warnings.warn(
                "[SparseFlow] torch.fx produced no targets and no sample input was provided; "
                "falling back to linear module-order scan mode."
            )

        fb_targets = self._analyze_fallback(model, input_shapes)
        merged = self._merge_targets(fb_targets, attention_targets, attention_names)
        return merged

    # ----------------------------------------------------------------
    # Target merging
    # ----------------------------------------------------------------

    @staticmethod
    def _is_inside_attention_target(module_name: str, attention_names: Set[str]) -> bool:
        for attn_name in attention_names:
            if module_name.startswith(attn_name + "."):
                return True
        return False

    def _merge_targets(
        self,
        base_targets: List[ReplacementTarget],
        attention_targets: List[ReplacementTarget],
        attention_names: Set[str],
    ) -> List[ReplacementTarget]:
        """Merge base targets with attention block targets.

        Rule: if a base target lives inside an attention block, the parent
        attention replacement takes priority and the child is dropped.
        """
        merged: List[ReplacementTarget] = []
        seen: Set[str] = set()

        for t in base_targets:
            name = t.conv_name
            if self._is_inside_attention_target(name, attention_names):
                continue
            if name in seen:
                continue
            merged.append(t)
            seen.add(name)

        for t in attention_targets:
            if t.conv_name in seen:
                continue
            merged.append(t)
            seen.add(t.conv_name)

        return merged

    # ----------------------------------------------------------------
    # Attention-block pass
    # ----------------------------------------------------------------

    def _analyze_attention_modules(self, model: nn.Module, input_shapes: dict) -> List[ReplacementTarget]:
        targets: List[ReplacementTarget] = []
        for name, module in model.named_modules():
            if not _is_attention_like_module(module):
                continue
            target = self._make_attention_target(
                name=name,
                module=module,
                spike_name=(f"{name}.attn_lif" if hasattr(module, "attn_lif") else name),
                input_shapes=input_shapes,
            )
            if target is not None:
                targets.append(target)
        return targets

    # ----------------------------------------------------------------
    # Mode 1: torch.fx
    # ----------------------------------------------------------------

    def _analyze_fx(self, model: nn.Module, input_shapes: dict) -> List[ReplacementTarget]:
        graph_module = torch.fx.symbolic_trace(model)
        modules_dict = dict(graph_module.named_modules())
        spike_nodes = [
            n for n in graph_module.graph.nodes
            if _is_spike_node(n, modules_dict, self.registry)
        ]

        targets: List[ReplacementTarget] = []
        visited: Set[str] = set()
        for spike_node in spike_nodes:
            spike_name = spike_node.target
            for target_name, target_type in self._bfs_find_targets(spike_node, modules_dict, visited):
                mod = modules_dict[target_name]
                target = self._make_target(
                    target_name,
                    mod,
                    target_type,
                    spike_name,
                    input_shapes,
                )
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

            if _is_conv1d_node(node, modules_dict):
                name = node.target
                if name not in visited:
                    found.append((name, "conv1d"))
                continue
            if _is_conv2d_node(node, modules_dict):
                name = node.target
                if name not in visited:
                    found.append((name, "conv2d"))
                continue
            if _is_conv3d_node(node, modules_dict):
                name = node.target
                if name not in visited:
                    found.append((name, "conv3d"))
                continue
            if _is_linear_node(node, modules_dict):
                name = node.target
                if name not in visited:
                    found.append((name, "linear"))
                continue
            if _is_attention_node(node, modules_dict):
                name = node.target
                if name not in visited:
                    found.append((name, "attention"))
                continue
            if _is_spike_node(node, modules_dict, self.registry):
                # Don't cross another spike boundary.
                continue

            for user in node.users:
                queue.append((user, depth + 1))
        return found

    # ----------------------------------------------------------------
    # Mode 2: runtime-order fallback
    # ----------------------------------------------------------------

    def _analyze_runtime_fallback(
        self,
        model: nn.Module,
        input_shapes: dict,
        sample_input: torch.Tensor,
    ) -> List[ReplacementTarget]:
        modules_dict = dict(model.named_modules())
        events = []
        hooks = []

        def _target_type_from_module(mod: nn.Module) -> Optional[str]:
            if isinstance(mod, nn.Conv1d):
                return "conv1d"
            if isinstance(mod, nn.Conv2d):
                return "conv2d"
            if isinstance(mod, nn.Conv3d):
                return "conv3d"
            if isinstance(mod, nn.Linear):
                return "linear"
            if _is_attention_like_module(mod):
                return "attention"
            return None

        def make_hook(name: str, kind: str):
            def hook(_m, _inp, _out):
                events.append((kind, name))
            return hook

        for name, module in model.named_modules():
            if self.registry.is_spike_op(module):
                hooks.append(module.register_forward_hook(make_hook(name, "spike")))
            elif _target_type_from_module(module) is not None:
                hooks.append(module.register_forward_hook(make_hook(name, "target")))

        try:
            try:
                from spikingjelly.activation_based import functional as sj_func
                sj_func.reset_net(model)
            except ImportError:
                pass
            with torch.no_grad():
                _ = model(sample_input)
        finally:
            for h in hooks:
                h.remove()

        targets: List[ReplacementTarget] = []
        visited: Set[str] = set()
        last_spike_name: Optional[str] = None
        for kind, name in events:
            if kind == "spike":
                last_spike_name = name
                continue
            if kind != "target":
                continue
            if name in visited or last_spike_name is None:
                continue

            module = modules_dict.get(name)
            if module is None:
                continue
            target_type = _target_type_from_module(module)
            if target_type is None:
                continue

            target = self._make_target(
                name,
                module,
                target_type,
                last_spike_name,
                input_shapes,
            )
            if target is not None:
                targets.append(target)
                visited.add(name)

        return targets

    # ----------------------------------------------------------------
    # Mode 3: linear module-order fallback
    # ----------------------------------------------------------------

    def _analyze_fallback(self, model: nn.Module, input_shapes: dict) -> List[ReplacementTarget]:
        module_list = list(model.named_modules())
        targets: List[ReplacementTarget] = []
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

                target_type = None
                if isinstance(next_module, nn.Conv1d):
                    target_type = "conv1d"
                elif isinstance(next_module, nn.Conv2d):
                    target_type = "conv2d"
                elif isinstance(next_module, nn.Conv3d):
                    target_type = "conv3d"
                elif isinstance(next_module, nn.Linear):
                    target_type = "linear"
                elif _is_attention_like_module(next_module):
                    target_type = "attention"
                else:
                    continue

                target = self._make_target(
                    next_name,
                    next_module,
                    target_type,
                    name,
                    input_shapes,
                )
                if target is not None:
                    targets.append(target)
                    visited.add(next_name)

        return targets

    # ----------------------------------------------------------------
    # Target builders
    # ----------------------------------------------------------------

    def _make_target(
        self,
        module_name: str,
        module: nn.Module,
        target_type: str,
        spike_name: str,
        input_shapes: dict,
    ) -> Optional[ReplacementTarget]:
        if target_type == "conv1d":
            return self._make_conv1d_target(module_name, module, spike_name, input_shapes)
        if target_type == "conv2d":
            return self._make_conv2d_target(module_name, module, spike_name, input_shapes)
        if target_type == "conv3d":
            return self._make_conv3d_target(module_name, module, spike_name, input_shapes)
        if target_type == "linear":
            return self._make_linear_target(module_name, module, spike_name)
        if target_type == "attention":
            return self._make_attention_target(module_name, module, spike_name, input_shapes)
        return None

    def _make_conv2d_target(
        self,
        conv_name: str,
        conv_module: nn.Conv2d,
        spike_name: str,
        input_shapes: dict,
    ) -> Optional[ReplacementTarget]:
        if not isinstance(conv_module, nn.Conv2d):
            return None

        k = _as_pair(conv_module.kernel_size)
        s = _as_pair(conv_module.stride)
        p = _as_pair(conv_module.padding)
        g = int(conv_module.groups)

        # Depthwise special case.
        if (
            g == conv_module.in_channels
            and conv_module.out_channels == conv_module.in_channels
            and k[0] == k[1]
            and s[0] == s[1]
            and p[0] == p[1]
        ):
            h, w = self._extract_hw(input_shapes.get(conv_name))
            return ReplacementTarget(
                conv_name=conv_name,
                conv_module=conv_module,
                spike_name=spike_name,
                op_type="depthwise_conv2d",
                block_size=None,
                input_h=h,
                input_w=w,
            )

        # Grouped (non-depthwise) conv2d is not supported by the current
        # sparse kernel.
        if g != 1:
            return None

        if k == (3, 3) and s == (1, 1) and p == (1, 1):
            op_type = "conv2d_3x3"
        elif k == (3, 3) and s == (2, 2) and p == (1, 1):
            op_type = "conv2d_3x3_s2"
        elif k == (1, 1) and s == (1, 1) and p == (0, 0):
            op_type = "conv2d_1x1"
        else:
            return None

        h, w = self._extract_hw(input_shapes.get(conv_name))
        # Small feature maps: per-tile overhead dominates; skip sparse path.
        if h > 0 and w > 0 and min(h, w) < 7:
            return None

        return ReplacementTarget(
            conv_name=conv_name,
            conv_module=conv_module,
            spike_name=spike_name,
            op_type=op_type,
            block_size=None,
            input_h=h,
            input_w=w,
        )

    def _make_conv1d_target(
        self,
        name: str,
        module: nn.Conv1d,
        spike_name: str,
        input_shapes: dict,
    ) -> Optional[ReplacementTarget]:
        if not isinstance(module, nn.Conv1d):
            return None
        l = self._extract_l(input_shapes.get(name))
        return ReplacementTarget(
            conv_name=name,
            conv_module=module,
            spike_name=spike_name,
            op_type="conv1d",
            block_size=None,
            input_w=l,
        )

    def _make_conv3d_target(
        self,
        name: str,
        module: nn.Conv3d,
        spike_name: str,
        input_shapes: dict,
    ) -> Optional[ReplacementTarget]:
        if not isinstance(module, nn.Conv3d):
            return None
        d, h, w = self._extract_dhw(input_shapes.get(name))
        return ReplacementTarget(
            conv_name=name,
            conv_module=module,
            spike_name=spike_name,
            op_type="conv3d",
            block_size=None,
            input_d=d,
            input_h=h,
            input_w=w,
        )

    def _make_linear_target(
        self,
        name: str,
        module: nn.Linear,
        spike_name: str,
    ) -> Optional[ReplacementTarget]:
        if not isinstance(module, nn.Linear):
            return None
        return ReplacementTarget(
            conv_name=name,
            conv_module=module,
            spike_name=spike_name,
            op_type="linear",
            block_size=None,
        )

    def _make_attention_target(
        self,
        name: str,
        module: nn.Module,
        spike_name: str,
        input_shapes: dict,
    ) -> Optional[ReplacementTarget]:
        if not _is_attention_like_module(module):
            return None

        ishape = input_shapes.get(name)
        n, c = self._extract_nc(ishape)
        op_type = _infer_attention_variant(module)
        num_heads = int(getattr(module, "num_heads", 1))
        head_dim = int(getattr(
            module, "head_dim",
            max(1, c // max(1, num_heads)),
        ))
        extra = {"num_heads": num_heads, "head_dim": head_dim}

        return ReplacementTarget(
            conv_name=name,
            conv_module=module,
            spike_name=spike_name,
            op_type=op_type,
            block_size=None,
            input_h=n,   # token length
            input_w=c,   # channel dim
            extra=extra,
        )

    def _make_maxpool2d_target(
        self,
        name: str,
        module: nn.MaxPool2d,
        spike_name: str,
        input_shapes: dict,
    ) -> Optional[ReplacementTarget]:
        return None

    def _make_avgpool2d_target(
        self,
        name: str,
        module: nn.AvgPool2d,
        spike_name: str,
        input_shapes: dict,
    ) -> Optional[ReplacementTarget]:
        return None

    # ----------------------------------------------------------------
    # Shape extractors
    # ----------------------------------------------------------------

    @staticmethod
    def _extract_hw(ishape) -> Tuple[int, int]:
        if ishape is None:
            return 0, 0
        if len(ishape) >= 2:
            return int(ishape[-2]), int(ishape[-1])
        return 0, 0

    @staticmethod
    def _extract_l(ishape) -> int:
        if ishape is None:
            return 0
        if len(ishape) >= 1:
            return int(ishape[-1])
        return 0

    @staticmethod
    def _extract_dhw(ishape) -> Tuple[int, int, int]:
        if ishape is None:
            return 0, 0, 0
        if len(ishape) >= 3:
            return int(ishape[-3]), int(ishape[-2]), int(ishape[-1])
        return 0, 0, 0

    @staticmethod
    def _extract_nc(ishape) -> Tuple[int, int]:
        """Attention-block input shape: [T, B, N, C] or [B, N, C]."""
        if ishape is None:
            return 0, 0
        if len(ishape) >= 2:
            return int(ishape[-2]), int(ishape[-1])
        return 0, 0

    # ----------------------------------------------------------------
    # Shape inference under a sample input
    # ----------------------------------------------------------------

    @staticmethod
    def _infer_input_shapes(model: nn.Module, sample_input: torch.Tensor) -> Dict[str, tuple]:
        input_shapes: Dict[str, tuple] = {}
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
