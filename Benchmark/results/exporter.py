from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Dict, List


OPERATOR_PRINT_ORDER = [
    "conv_3x3",
    "conv_1x1",
    "linear",
    "matmul",
    "bmm",
    "attention",
]


def aggregate_results(case_results: List[Dict]) -> Dict[str, Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for case in case_results:
        key = str(case.get("operator", "unknown"))
        grouped.setdefault(key, []).append(case)

    aggregated: Dict[str, Dict] = {}
    for op_name, cases in grouped.items():
        ok_cases = [c for c in cases if c.get("status") == "ok"]
        failed_cases = [c for c in cases if c.get("status") != "ok"]

        if ok_cases:
            speedup_avg = mean(float(c["speedup"]) for c in ok_cases)
            dense_avg = mean(float(c["dense_latency_ms"]) for c in ok_cases)
            sparse_avg = mean(float(c["sparse_latency_ms"]) for c in ok_cases)
            cosine_avg = mean(float(c["cosine_similarity"]) for c in ok_cases)
            max_abs_avg = mean(float(c["max_abs_error"]) for c in ok_cases)
            max_abs_max = max(float(c["max_abs_error"]) for c in ok_cases)
        else:
            speedup_avg = 0.0
            dense_avg = 0.0
            sparse_avg = 0.0
            cosine_avg = 0.0
            max_abs_avg = 0.0
            max_abs_max = 0.0

        aggregated[op_name] = {
            "speedup_avg": speedup_avg,
            "dense_latency_avg_ms": dense_avg,
            "sparse_latency_avg_ms": sparse_avg,
            "cosine_similarity_avg": cosine_avg,
            "max_abs_error_avg": max_abs_avg,
            "max_abs_error_max": max_abs_max,
            "num_cases": len(cases),
            "num_ok": len(ok_cases),
            "num_failed": len(failed_cases),
            "cases": cases,
        }

    return aggregated


def build_export_payload(meta: Dict, aggregated: Dict[str, Dict]) -> Dict:
    payload: Dict = {"_meta": meta}
    ordered_names = [n for n in OPERATOR_PRINT_ORDER if n in aggregated]
    for op_name in ordered_names:
        payload[op_name] = aggregated[op_name]
    for op_name in sorted(aggregated.keys()):
        if op_name not in payload:
            payload[op_name] = aggregated[op_name]
    return payload


def save_json(payload: Dict, output_path: str) -> Path:
    out_path = Path(output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path

