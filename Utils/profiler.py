"""
性能分析工具 — 延迟、吞吐、稀疏率统计
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class SparseProfiler:
    """
    Hook-based 性能分析器，统计每个 SparseConv2d 的：
    - 稀疏率 (sparsity)
    - Stage-2 延迟 (sparse_ms)
    - 跳过的 block 比例
    """

    def __init__(self):
        self.stats: Dict[str, Dict[str, Any]] = {}
        self._hooks = []

    def attach(self, model: nn.Module):
        """挂载 hook 到模型中所有 SparseConv2d"""
        from sparseflow.ops.sparse_conv2d import SparseConv2d

        for name, module in model.named_modules():
            if isinstance(module, SparseConv2d):
                self.stats[name] = {
                    "total_zeros": 0,
                    "total_elems": 0,
                    "total_sparse_ms": 0.0,
                    "call_count": 0,
                }
                h = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(h)

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            x = inp[0]
            if x.dim() == 5:
                T, B, C, H, W = x.shape
                x = x.reshape(T * B, C, H, W)
            st = self.stats[name]
            st["total_zeros"] += (x.abs() <= 1e-6).sum().item()
            st["total_elems"] += x.numel()
            st["call_count"] += 1
            # sparse_ms 由 SparseConv2d 在 forward 中记录到 module._last_sparse_ms
            if hasattr(module, "_last_sparse_ms"):
                st["total_sparse_ms"] += module._last_sparse_ms
        return hook

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def report(self) -> str:
        lines = []
        lines.append(f"{'Layer':<40} {'Sparsity':>9} {'Calls':>6} {'Time(ms)':>10}")
        lines.append("-" * 70)
        for name, st in self.stats.items():
            if st["total_elems"] == 0:
                continue
            sparsity = st["total_zeros"] / st["total_elems"] * 100
            lines.append(
                f"{name:<40} {sparsity:>8.2f}% {st['call_count']:>6} "
                f"{st['total_sparse_ms']:>9.2f}"
            )
        return "\n".join(lines)