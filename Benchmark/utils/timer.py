from __future__ import annotations

import time
from typing import Callable, Dict, List

import torch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class BenchmarkTimer:
    """GPU-aware benchmark timer with strict warmup/measure protocol."""

    def __init__(self, device: torch.device):
        self.device = torch.device(device)

    def run(
        self,
        fn: Callable[[], torch.Tensor],
        warmup: int = 20,
        iters: int = 100,
    ) -> Dict[str, object]:
        warmup = max(int(warmup), 0)
        iters = max(int(iters), 1)

        for _ in range(warmup):
            fn()
        _sync(self.device)

        if self.device.type == "cuda":
            return self._run_cuda(fn, iters)
        return self._run_cpu(fn, iters)

    def _run_cuda(self, fn: Callable[[], torch.Tensor], iters: int) -> Dict[str, object]:
        times_ms: List[float] = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            _sync(self.device)
            start.record()
            fn()
            end.record()
            _sync(self.device)
            times_ms.append(float(start.elapsed_time(end)))
        avg_ms = float(sum(times_ms) / len(times_ms))
        return {"avg_ms": avg_ms, "times_ms": times_ms}

    def _run_cpu(self, fn: Callable[[], torch.Tensor], iters: int) -> Dict[str, object]:
        times_ms: List[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)
        avg_ms = float(sum(times_ms) / len(times_ms))
        return {"avg_ms": avg_ms, "times_ms": times_ms}
