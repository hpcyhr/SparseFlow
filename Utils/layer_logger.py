"""
Utils/layer_logger.py

Canonical structured per-layer observability logger for SparseFlow experiments.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


SCHEMA_FIELDS = [
    "run_id",
    "model",
    "dataset",
    "T",
    "batch_idx",
    "layer_name",
    "module_name",
    "operator_type",
    "operator_family",
    "mode_used",
    "backend_mode",
    "backend_family",
    "dispatch_decision",
    "reason_code",
    "support_status",
    "meta_source",
    "diag_source",
    "tile_source",
    "fallback_reason",
    "replaced",
    "sparse_path_executed",
    "input_shape",
    "output_shape",
    "element_sparsity",
    "nonzero_group_count",
    "total_group_count",
    "active_group_ratio",
    "tile_zero_count",
    "total_tile_count",
    "tile_zero_ratio",
    "effective_k_ratio",
    "runtime_ms",
    "zero_check_ms",
    "metadata_ms",
    "sparse_compute_ms",
    "sparse_total_ms",
]


@dataclass
class LayerRecord:
    run_id: str = ""
    model: str = ""
    dataset: str = ""
    T: int = 0
    batch_idx: int = -1
    layer_name: str = ""
    module_name: str = ""
    operator_type: str = ""
    operator_family: str = ""
    mode_used: str = ""
    backend_mode: str = ""
    backend_family: str = ""
    dispatch_decision: str = ""
    reason_code: str = ""
    support_status: str = ""
    meta_source: str = ""
    diag_source: str = ""
    tile_source: str = ""
    fallback_reason: str = ""
    replaced: bool = False
    sparse_path_executed: bool = False
    input_shape: str = ""
    output_shape: str = ""
    element_sparsity: float = -1.0
    nonzero_group_count: float = -1.0
    total_group_count: float = -1.0
    active_group_ratio: float = -1.0
    tile_zero_count: float = -1.0
    total_tile_count: float = -1.0
    tile_zero_ratio: float = -1.0
    effective_k_ratio: float = -1.0
    runtime_ms: float = -1.0
    zero_check_ms: float = -1.0
    metadata_ms: float = -1.0
    sparse_compute_ms: float = -1.0
    sparse_total_ms: float = -1.0


class LayerLogger:
    def __init__(self, run_id: str = "", model: str = "", dataset: str = "", T: int = 0):
        self.defaults = dict(run_id=run_id, model=model, dataset=dataset, T=T)
        self.records: List[LayerRecord] = []

    def log_layer(self, **kwargs) -> LayerRecord:
        merged = {**self.defaults, **kwargs}
        rec = LayerRecord(
            **{k: v for k, v in merged.items() if k in LayerRecord.__dataclass_fields__}
        )
        self.records.append(rec)
        return rec

    def log_static_zero(self, layer_name, batch_idx: int = -1, input_shape=None):
        return self.log_layer(
            batch_idx=batch_idx,
            layer_name=layer_name,
            module_name=layer_name,
            operator_family="conv",
            mode_used="staticzero",
            backend_mode="staticzero",
            backend_family="exact_zero",
            dispatch_decision="staticzero",
            reason_code="exact_zero",
            support_status="exact_zero_shortcut",
            meta_source="shortcut",
            diag_source="shortcut",
            tile_source="shortcut",
            replaced=True,
            sparse_path_executed=False,
            input_shape=str(input_shape) if input_shape is not None else "",
            element_sparsity=1.0,
            active_group_ratio=0.0,
            tile_zero_ratio=1.0,
        )

    def log_dense(self, layer_name, batch_idx: int = -1, input_shape=None, element_sparsity: float = -1.0):
        return self.log_layer(
            batch_idx=batch_idx,
            layer_name=layer_name,
            module_name=layer_name,
            mode_used="dense",
            backend_mode="dense",
            dispatch_decision="dense",
            replaced=False,
            sparse_path_executed=False,
            input_shape=str(input_shape) if input_shape is not None else "",
            element_sparsity=element_sparsity,
        )

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [asdict(r) for r in self.records]

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dicts(), f, indent=2, ensure_ascii=False)

    def save_csv(self, path: str):
        rows = self.to_dicts()
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SCHEMA_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def print_summary(self):
        if not self.records:
            print("  (no records)")
            return
        print(
            f"\n  {'Layer':<40} {'Mode':<12} {'Backend':<10} {'AGR':>6} "
            f"{'TileZR':>7} {'SparseMs':>9} {'Reason':>14}"
        )
        print(f"  {'-' * 116}")
        for r in self.records:
            short = r.layer_name if len(r.layer_name) <= 39 else "..." + r.layer_name[-36:]
            agr = f"{r.active_group_ratio:.3f}" if r.active_group_ratio >= 0 else "n/a"
            tzr = f"{r.tile_zero_ratio:.3f}" if r.tile_zero_ratio >= 0 else "n/a"
            sms = f"{r.sparse_total_ms:.3f}" if r.sparse_total_ms >= 0 else "n/a"
            reason = r.reason_code or "n/a"
            print(
                f"  {short:<40} {r.mode_used:<12} {r.backend_mode:<10} "
                f"{agr:>6} {tzr:>7} {sms:>9} {reason:>14}"
            )

