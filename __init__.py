"""
SparseFlow package entry.

Usage:
    import sparseflow
    model = sparseflow.optimize(model, sample_input=sample_input)
"""

from Core.registry import SpikeOpRegistry
from Core.analyzer import NetworkAnalyzer
from Core.replacer import ModuleReplacer

__version__ = "0.2.0"


def optimize(
    model,
    sample_input=None,
    block_sizes=None,
    verbose=True,
    threshold=1e-6,
    return_ms=False,
    collect_diag=False,
    profile_runtime=False,
    fuse_conv_lif=False,
):
    """
    Optimize a spiking model in-place by replacing supported dense modules
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
        fuse_conv_lif: enable Conv+LIF fused replacement where applicable.

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
        enable_fused_conv_lif=bool(fuse_conv_lif),
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
            f"total={replaced}, sparse={sparse_count}, fused={fused_count}, "
            f"static_zero={static_zero_count}, dense_keep={dense_keep_count}"
        )

    return model
