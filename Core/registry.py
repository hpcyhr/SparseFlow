"""
脉冲算子注册表 — 识别 LIF / IF 等产生脉冲输出的算子类型
"""

from typing import Tuple, Type
import torch.nn as nn

# 延迟导入，避免强依赖 spikingjelly
_DEFAULT_SPIKE_OPS = None


def _get_default_spike_ops():
    global _DEFAULT_SPIKE_OPS
    if _DEFAULT_SPIKE_OPS is not None:
        return _DEFAULT_SPIKE_OPS

    ops = []
    try:
        from spikingjelly.activation_based import neuron as sj_neuron
        ops.extend([
            sj_neuron.LIFNode,
            sj_neuron.IFNode,
            sj_neuron.ParametricLIFNode,
        ])
    except ImportError:
        pass

    _DEFAULT_SPIKE_OPS = tuple(ops)
    return _DEFAULT_SPIKE_OPS


class SpikeOpRegistry:
    """
    管理已知的脉冲输出算子类型。
    用于判断某个 nn.Module 是否为 spike 源。
    """

    def __init__(self):
        self._spike_op_types: list = []

    def register(self, op_type: Type[nn.Module]):
        if op_type not in self._spike_op_types:
            self._spike_op_types.append(op_type)
        return self

    def register_many(self, op_types):
        for t in op_types:
            self.register(t)
        return self

    def is_spike_op(self, module: nn.Module) -> bool:
        return isinstance(module, tuple(self._spike_op_types))

    @property
    def spike_op_types(self) -> Tuple:
        return tuple(self._spike_op_types)

    @classmethod
    def default(cls) -> "SpikeOpRegistry":
        registry = cls()
        default_ops = _get_default_spike_ops()
        if default_ops:
            registry.register_many(default_ops)
        return registry