"""
Lightweight hook-based profiler for SparseFlow sparse operators.

This utility is intentionally minimal and is mainly used for quick debugging/
inspection rather than benchmark-grade reporting.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from Utils.config import NUMERIC_EPS


class SparseProfiler:
    """Collect per-layer sparsity and sparse-kernel timing via forward hooks."""

    def __init__(self):
        self.stats: Dict[str, Dict[str, Any]] = {}
        self._hooks = []

    def attach(self, model: nn.Module):
        """Attach hooks to all SparseConv2d modules in `model`."""
        try:
            from Ops.sparse_conv2d import SparseConv2d
        except Exception:
            # Optional compatibility path for pip-installed package layout.
            from sparseflow.ops.sparse_conv2d import SparseConv2d  # type: ignore

        for name, module in model.named_modules():
            if isinstance(module, SparseConv2d):
                self.stats[name] = {
                    "total_zeros": 0,
                    "total_elems": 0,
                    "total_sparse_ms": 0.0,
                    "call_count": 0,
                }
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook(module, inp, _out):
            x = inp[0]
            if x.dim() == 5:
                t, b, c, h, w = x.shape
                x = x.reshape(t * b, c, h, w)
            st = self.stats[name]
            st["total_zeros"] += (x.abs() <= NUMERIC_EPS).sum().item()
            st["total_elems"] += x.numel()
            st["call_count"] += 1
            if hasattr(module, "_last_sparse_ms"):
                st["total_sparse_ms"] += float(module._last_sparse_ms)

        return hook

    def detach(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def report(self) -> str:
        lines = []
        lines.append(f"{'Layer':<40} {'Sparsity':>9} {'Calls':>6} {'Time(ms)':>10}")
        lines.append("-" * 70)
        for name, st in self.stats.items():
            if st["total_elems"] == 0:
                continue
            sparsity = st["total_zeros"] / st["total_elems"] * 100.0
            lines.append(
                f"{name:<40} {sparsity:>8.2f}% {st['call_count']:>6} "
                f"{st['total_sparse_ms']:>9.2f}"
            )
        return "\n".join(lines)
