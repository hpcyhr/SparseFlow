"""
SparseFlow — 即插即用的 SNN 稀疏推理加速库

Usage:
    import sparseflow
    model = sparseflow.optimize(model)
"""

from sparseflow.core.registry import SpikeOpRegistry
from sparseflow.core.analyzer import NetworkAnalyzer
from sparseflow.core.replacer import ModuleReplacer

__version__ = "0.1.0"


def optimize(model, block_sizes=None, verbose=True):
    """
    一行代码将 SNN 模型中脉冲后继算子替换为稀疏加速版本。

    Args:
        model: nn.Module, 包含 LIF/IF 等脉冲神经元的 SNN 模型
        block_sizes: dict or None, 手动指定 block 大小 {layer_name: block_size}
                     为 None 时自动选择
        verbose: bool, 是否打印替换日志

    Returns:
        model: 替换后的模型（原地修改）
    """
    registry = SpikeOpRegistry.default()
    analyzer = NetworkAnalyzer(registry)
    replacer = ModuleReplacer(verbose=verbose)

    # Step 1: 分析网络拓扑，找到所有 spike_op -> target_op 的映射
    targets = analyzer.analyze(model)

    if verbose:
        print(f"[SparseFlow] Found {len(targets)} replaceable operators")

    # Step 2: 逐个替换
    replaced = replacer.replace(model, targets, block_sizes=block_sizes)

    if verbose:
        print(f"[SparseFlow] Replaced {replaced} operators")

    return model