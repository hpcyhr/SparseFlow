from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

try:
    from benchmark.ops import (
        run_attention_suite,
        run_bmm_suite,
        run_conv_suite,
        run_linear_suite,
        run_matmul_suite,
    )
    from benchmark.results.exporter import (
        OPERATOR_PRINT_ORDER,
        aggregate_results,
        build_export_payload,
    )
    from benchmark.utils.sparsity import DEFAULT_SPARSITY_REGIMES
    from benchmark.utils.timer import BenchmarkTimer
except ModuleNotFoundError:
    from ops import (  # type: ignore
        run_attention_suite,
        run_bmm_suite,
        run_conv_suite,
        run_linear_suite,
        run_matmul_suite,
    )
    from results.exporter import (  # type: ignore
        OPERATOR_PRINT_ORDER,
        aggregate_results,
        build_export_payload,
    )
    from utils.sparsity import DEFAULT_SPARSITY_REGIMES  # type: ignore
    from utils.timer import BenchmarkTimer  # type: ignore


@dataclass
class BenchmarkConfig:
    device: torch.device
    dtype: torch.dtype
    warmup: int
    iters: int
    scales: Sequence[str]
    sparsity_modes: Sequence[str]
    sparsity_regimes: Sequence[Tuple[str, float]]
    operators: Sequence[str]
    seed: int
    structured_tile: Tuple[int, int]

    def to_meta(self) -> Dict:
        return {
            "device": str(self.device),
            "dtype": str(self.dtype),
            "warmup": int(self.warmup),
            "iters": int(self.iters),
            "scales": list(self.scales),
            "sparsity_modes": list(self.sparsity_modes),
            "sparsity_regimes": [
                {"name": name, "level": float(level)} for name, level in self.sparsity_regimes
            ],
            "operators": list(self.operators),
            "seed": int(self.seed),
            "structured_tile": [int(self.structured_tile[0]), int(self.structured_tile[1])],
        }


def run_operator_benchmarks(config: BenchmarkConfig) -> Dict:
    timer = BenchmarkTimer(config.device)
    case_results: List[Dict] = []
    start_ts = time.time()

    op_set = {op.strip().lower() for op in config.operators}

    run_index = 0
    for scale in config.scales:
        for regime_name, sparsity_level in config.sparsity_regimes:
            for sparsity_mode in config.sparsity_modes:
                seed_base = int(config.seed + run_index * 1000)
                run_index += 1

                if "conv" in op_set:
                    case_results.extend(
                        run_conv_suite(
                            timer=timer,
                            device=config.device,
                            dtype=config.dtype,
                            scale=scale,
                            sparsity_regime=regime_name,
                            sparsity_level=sparsity_level,
                            sparsity_mode=sparsity_mode,
                            warmup=config.warmup,
                            iters=config.iters,
                            seed=seed_base + 11,
                            structured_tile=config.structured_tile,
                        )
                    )

                if "linear" in op_set:
                    case_results.extend(
                        run_linear_suite(
                            timer=timer,
                            device=config.device,
                            dtype=config.dtype,
                            scale=scale,
                            sparsity_regime=regime_name,
                            sparsity_level=sparsity_level,
                            sparsity_mode=sparsity_mode,
                            warmup=config.warmup,
                            iters=config.iters,
                            seed=seed_base + 23,
                            structured_tile=config.structured_tile,
                        )
                    )

                if "matmul" in op_set:
                    case_results.extend(
                        run_matmul_suite(
                            timer=timer,
                            device=config.device,
                            dtype=config.dtype,
                            scale=scale,
                            sparsity_regime=regime_name,
                            sparsity_level=sparsity_level,
                            sparsity_mode=sparsity_mode,
                            warmup=config.warmup,
                            iters=config.iters,
                            seed=seed_base + 37,
                            structured_tile=config.structured_tile,
                        )
                    )

                if "bmm" in op_set:
                    case_results.extend(
                        run_bmm_suite(
                            timer=timer,
                            device=config.device,
                            dtype=config.dtype,
                            scale=scale,
                            sparsity_regime=regime_name,
                            sparsity_level=sparsity_level,
                            sparsity_mode=sparsity_mode,
                            warmup=config.warmup,
                            iters=config.iters,
                            seed=seed_base + 53,
                            structured_tile=config.structured_tile,
                        )
                    )

                if "attention" in op_set:
                    case_results.extend(
                        run_attention_suite(
                            timer=timer,
                            device=config.device,
                            dtype=config.dtype,
                            scale=scale,
                            sparsity_regime=regime_name,
                            sparsity_level=sparsity_level,
                            sparsity_mode=sparsity_mode,
                            warmup=config.warmup,
                            iters=config.iters,
                            seed=seed_base + 71,
                            structured_tile=config.structured_tile,
                        )
                    )

    aggregated = aggregate_results(case_results)
    meta = config.to_meta()
    meta["elapsed_seconds"] = float(time.time() - start_ts)
    payload = build_export_payload(meta=meta, aggregated=aggregated)
    return payload


def print_console_summary(payload: Dict) -> None:
    print("\n" + "=" * 88)
    print("SparseFlow Operator Benchmark Summary")
    print("=" * 88)

    available_ops = [op for op in OPERATOR_PRINT_ORDER if op in payload]
    if not available_ops:
        print("No operator results were produced.")
        return

    for op_name in available_ops:
        info = payload[op_name]
        print(f"Operator: {op_name}")
        print(f"  Avg Speedup: {float(info['speedup_avg']):.4f}x")
        print(f"  Avg Latency Dense: {float(info['dense_latency_avg_ms']):.4f} ms")
        print(f"  Avg Latency Sparse: {float(info['sparse_latency_avg_ms']):.4f} ms")
        print(
            f"  Cases: total={int(info['num_cases'])}, ok={int(info['num_ok'])}, "
            f"failed={int(info['num_failed'])}"
        )
        print()


def default_sparsity_regimes() -> List[Tuple[str, float]]:
    return [(k, float(v)) for k, v in DEFAULT_SPARSITY_REGIMES.items()]
