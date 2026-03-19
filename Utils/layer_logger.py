"""
Utils/layer_logger.py — Structured per-layer sparsity & timing logger.

Usage:
    logger = LayerLogger(run_id="resnet18_T4", model="resnet18", dataset="cifar10", T=4)
    logger.log_layer(layer_name="layer1.0.conv2", mode_used="sparseconv", ...)
    logger.save_json("diag.json")
    logger.save_csv("diag.csv")
    logger.print_summary()
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


SCHEMA_FIELDS = [
    "run_id", "model", "dataset", "T", "batch_idx", "layer_name",
    "mode_used", "replaced", "sparse_path_executed",
    "input_shape", "output_shape",
    "element_sparsity",
    "nonzero_group_count", "total_group_count", "active_group_ratio",
    "tile_zero_count", "total_tile_count", "tile_zero_ratio",
    "effective_k_ratio",
    "zero_check_ms", "metadata_ms", "sparse_compute_ms", "sparse_total_ms",
]


@dataclass
class LayerRecord:
    run_id: str = ""
    model: str = ""
    dataset: str = ""
    T: int = 0
    batch_idx: int = -1
    layer_name: str = ""

    mode_used: str = ""
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

    zero_check_ms: float = -1.0
    metadata_ms: float = -1.0
    sparse_compute_ms: float = -1.0
    sparse_total_ms: float = -1.0


class LayerLogger:

    def __init__(self, run_id="", model="", dataset="", T=0):
        self.defaults = dict(run_id=run_id, model=model, dataset=dataset, T=T)
        self.records: List[LayerRecord] = []

    def log_layer(self, **kwargs) -> LayerRecord:
        merged = {**self.defaults, **kwargs}
        rec = LayerRecord(
            **{k: v for k, v in merged.items() if k in LayerRecord.__dataclass_fields__}
        )
        self.records.append(rec)
        return rec

    def log_static_zero(self, layer_name, batch_idx=-1, input_shape=None):
        return self.log_layer(
            batch_idx=batch_idx, layer_name=layer_name,
            mode_used="staticzero", replaced=True, sparse_path_executed=False,
            input_shape=str(input_shape) if input_shape else "",
            element_sparsity=1.0, active_group_ratio=0.0, tile_zero_ratio=1.0,
        )

    def log_dense(self, layer_name, batch_idx=-1, input_shape=None, element_sparsity=-1.0):
        return self.log_layer(
            batch_idx=batch_idx, layer_name=layer_name,
            mode_used="dense", replaced=False, sparse_path_executed=False,
            input_shape=str(input_shape) if input_shape else "",
            element_sparsity=element_sparsity,
        )

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [asdict(r) for r in self.records]

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dicts(), f, indent=2)

    def save_csv(self, path: str):
        rows = self.to_dicts()
        if not rows:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SCHEMA_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def print_summary(self):
        if not self.records:
            print("  (no records)")
            return
        print(f"\n  {'Layer':<40} {'Mode':<12} {'ElemSp':>7} {'AGR':>6} "
              f"{'TileZR':>7} {'SparseMs':>9} {'Path':>8}")
        print(f"  {'-'*95}")
        for r in self.records:
            short = r.layer_name if len(r.layer_name) <= 39 else "..." + r.layer_name[-36:]
            es = f"{r.element_sparsity*100:.1f}%" if r.element_sparsity >= 0 else "n/a"
            agr = f"{r.active_group_ratio:.3f}" if r.active_group_ratio >= 0 else "n/a"
            tzr = f"{r.tile_zero_ratio:.3f}" if r.tile_zero_ratio >= 0 else "n/a"
            sms = f"{r.sparse_total_ms:.3f}" if r.sparse_total_ms >= 0 else "n/a"
            path = "yes" if r.sparse_path_executed else "no"
            print(f"  {short:<40} {r.mode_used:<12} {es:>7} {agr:>6} "
                  f"{tzr:>7} {sms:>9} {path:>8}")