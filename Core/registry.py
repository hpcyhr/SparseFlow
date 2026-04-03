"""
Core/registry.py

Registry for:
1) spike source ops (LIF / IF / PLIF...)
2) module target ops that SparseFlow can replace.
"""

from typing import Tuple, Type
import torch.nn as nn

_DEFAULT_SPIKE_OPS = None


def _get_default_spike_ops():
    global _DEFAULT_SPIKE_OPS
    if _DEFAULT_SPIKE_OPS is not None:
        return _DEFAULT_SPIKE_OPS

    ops = []
    try:
        from spikingjelly.activation_based import neuron as sj_neuron
        ops.extend(
            [
                sj_neuron.LIFNode,
                sj_neuron.IFNode,
                sj_neuron.ParametricLIFNode,
            ]
        )
    except ImportError:
        pass

    _DEFAULT_SPIKE_OPS = tuple(ops)
    return _DEFAULT_SPIKE_OPS


def _is_attention_like_module(module: nn.Module) -> bool:
    """
    Structural check for spike-attention blocks in external transformer models.

    We intentionally use a capability-based check rather than class-name matching
    to keep this registry stable across model files.
    """
    if module is None:
        return False

    for attr in ("q", "k", "v", "proj"):
        if not hasattr(module, attr):
            return False
        if not isinstance(getattr(module, attr), nn.Linear):
            return False

    # spike-attention blocks in this repo have an attention spike node
    if not hasattr(module, "attn_lif"):
        return False

    return True


class SpikeOpRegistry:
    """
    Spike op + replaceable target op registry.

    Supported replacement target families:
    - nn.Conv1d
    - nn.Conv2d (including depthwise subset)
    - nn.Conv3d
    - nn.Linear
    - attention-like blocks (structural match, see is_target_attention_like)
    """

    def __init__(self):
        self._spike_op_types: list = []
        self._target_op_types: list = [nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear]

    def register(self, op_type: Type[nn.Module]):
        if op_type not in self._spike_op_types:
            self._spike_op_types.append(op_type)
        return self

    def register_many(self, op_types):
        for op_t in op_types:
            self.register(op_t)
        return self

    def register_target(self, op_type: Type[nn.Module]):
        if op_type not in self._target_op_types:
            self._target_op_types.append(op_type)
        return self

    def is_spike_op(self, module: nn.Module) -> bool:
        return isinstance(module, tuple(self._spike_op_types))

    def is_target_conv1d(self, module: nn.Module) -> bool:
        return isinstance(module, nn.Conv1d)

    def is_target_conv2d(self, module: nn.Module) -> bool:
        return isinstance(module, nn.Conv2d)

    def is_target_conv3d(self, module: nn.Module) -> bool:
        return isinstance(module, nn.Conv3d)

    def is_target_linear(self, module: nn.Module) -> bool:
        return isinstance(module, nn.Linear)

    def is_target_depthwise_conv2d(self, module: nn.Module) -> bool:
        if not isinstance(module, nn.Conv2d):
            return False
        return (
            module.groups == module.in_channels
            and module.out_channels == module.in_channels
        )

    def is_target_attention_like(self, module: nn.Module) -> bool:
        return _is_attention_like_module(module)

    @property
    def spike_op_types(self) -> Tuple:
        return tuple(self._spike_op_types)

    @property
    def target_op_types(self) -> Tuple:
        return tuple(self._target_op_types)

    @classmethod
    def default(cls) -> "SpikeOpRegistry":
        registry = cls()
        default_ops = _get_default_spike_ops()
        if default_ops:
            registry.register_many(default_ops)
        return registry
