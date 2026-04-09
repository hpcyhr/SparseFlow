"""
SparseFlow package entry.

Usage:
    import sparseflow
    model = sparseflow.optimize(model, sample_input=sample_input)
"""

from Core.analyzer import NetworkAnalyzer
from Core.registry import SpikeOpRegistry
from Core.replacer import ModuleReplacer

__version__ = "0.3.0"


def optimize(
    model,
    sample_input=None,
    block_sizes=None,
    verbose=True,
    threshold=1e-6,
    return_ms=False,
    collect_diag=False,
    profile_runtime=False,
):
    """Optimize a spiking model in-place by replacing supported dense modules
    with SparseFlow operators.

    Args:
        model: nn.Module containing spike operators.
        sample_input: optional tensor used by analyzer to infer per-layer input shapes.
        block_sizes: optional dict {layer_name: block_size} for conv2d kernels.
        verbose: whether to print analysis and replacement logs.
        threshold: prescan activity threshold passed to sparse operator wrappers.
        return_ms: whether wrappers should return kernel timing internally.
        collect_diag: whether wrappers should collect diagnostic metadata.
        profile_runtime: whether wrappers should track runtime profiling stats.

    Returns:
        The same model object, modified in-place.

    Version history:
      - 0.3.0 (Round 7): removed deprecated `fuse_conv_lif` kwarg. Any caller
        that was still passing `fuse_conv_lif=...` will raise a TypeError;
        the argument had been a silent no-op since Round 4 when Conv+LIF
        fused replacement was deleted from the codebase.
    """
    registry = SpikeOpRegistry.default()
    analyzer = NetworkAnalyzer(registry)
    replacer = ModuleReplacer(verbose=verbose)

    targets = analyzer.analyze(model, sample_input=sample_input)
    if verbose:
        print(f"[SparseFlow] Found {len(targets)} replaceable operators")

    replaced, sparse_count, static_zero_count, dense_keep_count = replacer.replace(
        model,
        targets,
        block_sizes=block_sizes,
        sparse_kwargs={
            "threshold": float(threshold),
            "return_ms": bool(return_ms),
            "collect_diag": bool(collect_diag),
            "profile_runtime": bool(profile_runtime),
        },
    )
    if verbose:
        print(
            "[SparseFlow] Replacement summary: "
            f"total={replaced}, sparse={sparse_count}, "
            f"static_zero={static_zero_count}, dense_keep={dense_keep_count}"
        )

    return model