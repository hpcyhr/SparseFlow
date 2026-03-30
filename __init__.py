"""
SparseFlow package entry.

Usage:
    import sparseflow
    model = sparseflow.optimize(model, sample_input=sample_input)
"""

from Core.registry import SpikeOpRegistry
from Core.analyzer import NetworkAnalyzer
from Core.replacer import ModuleReplacer

__version__ = "0.1.0"


def optimize(model, sample_input=None, block_sizes=None, verbose=True):
    """
    Optimize a spiking model in-place by replacing supported dense modules
    with SparseFlow operators.

    Args:
        model: nn.Module containing spike operators.
        sample_input: optional tensor used by analyzer to infer per-layer input shapes.
        block_sizes: optional dict {layer_name: block_size} for conv2d kernels.
        verbose: whether to print analysis and replacement logs.

    Returns:
        The same model object, modified in-place.
    """
    registry = SpikeOpRegistry.default()
    analyzer = NetworkAnalyzer(registry)
    replacer = ModuleReplacer(verbose=verbose)

    targets = analyzer.analyze(model, sample_input=sample_input)
    if verbose:
        print(f"[SparseFlow] Found {len(targets)} replaceable operators")

    replaced, sparse_count, fused_count, static_zero_count, dense_keep_count = replacer.replace(
        model,
        targets,
        block_sizes=block_sizes,
    )
    if verbose:
        print(
            "[SparseFlow] Replacement summary: "
            f"total={replaced}, sparse={sparse_count}, fused={fused_count}, "
            f"static_zero={static_zero_count}, dense_keep={dense_keep_count}"
        )

    return model
