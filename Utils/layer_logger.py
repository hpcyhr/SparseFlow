"""
Utils/layer_logger.py - Structured per-layer observability logger.

This logger is the canonical structured output path for SparseFlow experiments.
It keeps backward-compatible fields while adding standardized dispatch/runtime
provenance fields used by Core/Ops/Benchmark.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


# Legacy fields stay first to preserve compatibility with old CSV tooling.
SCHEMA_FIELDS = [
    "run_id",
    "model",
    "dataset",
    "T",
    "batch_idx",
    "layer_name",
    "mode_used",
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
    "zero_check_ms",
    "metadata_ms",
    "sparse_compute_ms",
    "sparse_total_ms",
    # Standardized observability extensions.
    "operator_family",
    "operator_type",
    "backend_mode",
    "backend_family",
    "diag_path",
    "dispatch_decision",
    "reason_code",
    "support_status",
    "meta_source",
    "diag_source",
    "tile_source",
    "score_family",
    "fallback_reason",
    "decision_confidence",
    "dense_fallback_ms",
    "runtime_total_ms",
    "active_tiles",
    "launch_tile_count",
    "group_size",
    "num_groups",
]


@dataclass
class LayerRecord:
    # Run identity
    run_id: str = ""
    model: str = ""
    dataset: str = ""
    T: int = 0
    batch_idx: int = -1
    layer_name: str = ""

    # Legacy mode fields
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

    # Standardized observability fields
    operator_family: str = "unknown"
    operator_type: str = "unknown"
    backend_mode: str = ""
    backend_family: str = "unknown"
    diag_path: str = "unknown"
    dispatch_decision: str = ""
    reason_code: str = ""
    support_status: str = "supported"
    meta_source: str = "measured"
    diag_source: str = "measured"
    tile_source: str = "unknown"
    score_family: str = "unknown"
    fallback_reason: str = ""
    decision_confidence: float = -1.0

    # Runtime extras
    dense_fallback_ms: float = -1.0
    runtime_total_ms: float = -1.0
    active_tiles: float = -1.0
    launch_tile_count: float = -1.0
    group_size: float = -1.0
    num_groups: float = -1.0


class LayerLogger:
    def __init__(self, run_id: str = "", model: str = "", dataset: str = "", T: int = 0):
        self.defaults = dict(run_id=run_id, model=model, dataset=dataset, T=T)
        self.records: List[LayerRecord] = []

    def log_layer(self, **kwargs) -> LayerRecord:
        merged = {**self.defaults, **kwargs}
        record = LayerRecord(
            **{k: v for k, v in merged.items() if k in LayerRecord.__dataclass_fields__}
        )
        self.records.append(record)
        return record

    def log_from_diag(
        self,
        *,
        layer_name: str,
        diag: Dict[str, Any],
        batch_idx: int = -1,
        mode_used: str = "",
        replaced: bool = True,
        operator_family: str = "unknown",
        operator_type: str = "unknown",
        dispatch_decision: str = "",
        reason_code: str = "",
        support_status: str = "supported",
        meta_source: str = "measured",
        diag_source: str = "measured",
        tile_source: str = "unknown",
        score_family: str = "unknown",
        fallback_reason: str = "",
        decision_confidence: float = -1.0,
    ) -> LayerRecord:
        """Convenience helper: log a layer by combining structured args + diag dict."""
        return self.log_layer(
            batch_idx=batch_idx,
            layer_name=layer_name,
            mode_used=mode_used,
            replaced=replaced,
            sparse_path_executed=bool(diag.get("sparse_path_executed", False)),
            operator_family=operator_family,
            operator_type=operator_type,
            backend_mode=str(diag.get("backend", "")),
            backend_family=str(diag.get("backend_family", "unknown")),
            diag_path=str(diag.get("diag_path", diag.get("_diag_path", "unknown"))),
            dispatch_decision=dispatch_decision,
            reason_code=reason_code,
            support_status=support_status,
            meta_source=meta_source,
            diag_source=diag_source,
            tile_source=tile_source,
            score_family=score_family,
            fallback_reason=fallback_reason,
            decision_confidence=decision_confidence,
            nonzero_group_count=float(diag.get("nonzero_group_count", -1.0)),
            total_group_count=float(diag.get("total_group_count", -1.0)),
            active_group_ratio=float(diag.get("active_group_ratio", -1.0)),
            tile_zero_count=float(diag.get("tile_zero_count", -1.0)),
            total_tile_count=float(diag.get("total_tile_count", -1.0)),
            tile_zero_ratio=float(diag.get("tile_zero_ratio", -1.0)),
            effective_k_ratio=float(diag.get("effective_k_ratio", -1.0)),
            zero_check_ms=float(diag.get("zero_check_ms", -1.0)),
            metadata_ms=float(diag.get("metadata_ms", -1.0)),
            sparse_compute_ms=float(diag.get("sparse_compute_ms", -1.0)),
            sparse_total_ms=float(diag.get("sparse_total_ms", -1.0)),
            dense_fallback_ms=float(diag.get("dense_fallback_ms", -1.0)),
            runtime_total_ms=float(diag.get("runtime_total_ms", -1.0)),
            active_tiles=float(diag.get("active_tiles", -1.0)),
            launch_tile_count=float(diag.get("launch_tile_count", -1.0)),
            group_size=float(diag.get("group_size", -1.0)),
            num_groups=float(diag.get("num_groups", -1.0)),
        )

    def log_static_zero(self, layer_name: str, batch_idx: int = -1, input_shape=None) -> LayerRecord:
        return self.log_layer(
            batch_idx=batch_idx,
            layer_name=layer_name,
            mode_used="staticzero",
            backend_mode="staticzero",
            backend_family="exact_zero",
            diag_path="staticzero",
            replaced=True,
            sparse_path_executed=False,
            input_shape=str(input_shape) if input_shape is not None else "",
            element_sparsity=1.0,
            active_group_ratio=0.0,
            tile_zero_ratio=1.0,
        )

    def log_dense(
        self,
        layer_name: str,
        batch_idx: int = -1,
        input_shape=None,
        element_sparsity: float = -1.0,
    ) -> LayerRecord:
        return self.log_layer(
            batch_idx=batch_idx,
            layer_name=layer_name,
            mode_used="dense",
            backend_mode="dense",
            backend_family="dense_torch",
            diag_path="dense",
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

        # Include unknown extension keys if call sites add future fields.
        all_fields = list(SCHEMA_FIELDS)
        for row in rows:
            for key in row.keys():
                if key not in all_fields:
                    all_fields.append(key)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def print_summary(self):
        if not self.records:
            print("  (no records)")
            return

        print(
            f"\n  {'Layer':<40} {'Mode':<12} {'Backend':<14} {'AGR':>6} {'TileZR':>7} "
            f"{'SparseMs':>9} {'Reason':<14}"
        )
        print(f"  {'-' * 120}")
        for r in self.records:
            short = r.layer_name if len(r.layer_name) <= 39 else "..." + r.layer_name[-36:]
            agr = f"{r.active_group_ratio:.3f}" if r.active_group_ratio >= 0 else "n/a"
            tzr = f"{r.tile_zero_ratio:.3f}" if r.tile_zero_ratio >= 0 else "n/a"
            sms = f"{r.sparse_total_ms:.3f}" if r.sparse_total_ms >= 0 else "n/a"
            reason = r.reason_code or r.fallback_reason or "n/a"
            print(
                f"  {short:<40} {r.mode_used:<12} {r.backend_family:<14} "
                f"{agr:>6} {tzr:>7} {sms:>9} {reason:<14}"
            )
