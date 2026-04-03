from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Support both:
# 1) python Benchmark/run_benchmark.py (cwd anywhere)
# 2) cd Benchmark && python run_benchmark.py
for p in (SCRIPT_DIR, PROJECT_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

try:
    from benchmark.bench_ops import (  # type: ignore
        BenchmarkConfig,
        default_sparsity_regimes,
        print_console_summary,
        run_operator_benchmarks,
    )
    from benchmark.results.exporter import save_json  # type: ignore
except ModuleNotFoundError:
    from bench_ops import (  # type: ignore
        BenchmarkConfig,
        default_sparsity_regimes,
        print_console_summary,
        run_operator_benchmarks,
    )
    from results.exporter import save_json  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research-grade SparseFlow operator benchmark (dense vs sparse kernels)."
    )
    parser.add_argument("--device", type=str, default="cuda", help="Benchmark device, e.g. cuda or cuda:0")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Tensor dtype for benchmark inputs/weights.",
    )
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations per case.")
    parser.add_argument("--iters", type=int, default=100, help="Measured iterations per case.")
    parser.add_argument(
        "--operators",
        type=str,
        default="conv,linear,matmul,bmm,attention",
        help="Comma-separated operator families.",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="small,medium,large",
        help="Comma-separated scales.",
    )
    parser.add_argument(
        "--sparsity-modes",
        type=str,
        default="bernoulli,structured",
        help="Comma-separated sparsity modes: bernoulli, structured.",
    )
    parser.add_argument(
        "--sparsity-levels",
        type=str,
        default="0.1,0.5,0.8",
        help="Comma-separated sparsity levels in [0.0, 0.9].",
    )
    parser.add_argument(
        "--structured-tile",
        type=str,
        default="16,16",
        help="Structured sparsity tile as H,W (e.g., 16,16).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument(
        "--output",
        type=str,
        default="results/operator_benchmark_results.json",
        help="JSON output file path.",
    )
    return parser.parse_args()


def parse_csv_list(text: str) -> List[str]:
    return [s.strip() for s in str(text).split(",") if s.strip()]


def parse_levels(text: str) -> List[float]:
    levels = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    out: List[float] = []
    for lv in levels:
        lv = max(0.0, min(0.9, float(lv)))
        out.append(lv)
    if not out:
        raise ValueError("No valid sparsity level provided.")
    return out


def levels_to_regimes(levels: Sequence[float]) -> List[Tuple[str, float]]:
    default = default_sparsity_regimes()
    default_levels = [round(v, 4) for _, v in default]
    input_levels = [round(float(v), 4) for v in levels]
    if input_levels == default_levels:
        return default
    return [(f"level_{lv:.2f}", float(lv)) for lv in levels]


def parse_dtype(dtype_str: str) -> torch.dtype:
    dtype_str = dtype_str.strip().lower()
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def parse_tile(tile_text: str) -> Tuple[int, int]:
    vals = [int(v.strip()) for v in str(tile_text).split(",") if v.strip()]
    if len(vals) == 1:
        return (max(vals[0], 1), max(vals[0], 1))
    if len(vals) >= 2:
        return (max(vals[0], 1), max(vals[1], 1))
    return (16, 16)


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required for this benchmark, but no CUDA device is available.")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    operators = parse_csv_list(args.operators)
    scales = parse_csv_list(args.scales)
    sparsity_modes = parse_csv_list(args.sparsity_modes)
    levels = parse_levels(args.sparsity_levels)
    regimes = levels_to_regimes(levels)
    dtype = parse_dtype(args.dtype)
    tile = parse_tile(args.structured_tile)

    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    cfg = BenchmarkConfig(
        device=device,
        dtype=dtype,
        warmup=int(args.warmup),
        iters=int(args.iters),
        scales=scales,
        sparsity_modes=sparsity_modes,
        sparsity_regimes=regimes,
        operators=operators,
        seed=int(args.seed),
        structured_tile=tile,
    )

    payload = run_operator_benchmarks(cfg)
    print_console_summary(payload)
    out_path = save_json(payload, args.output)
    print(f"JSON results exported to: {out_path}")


if __name__ == "__main__":
    main()

