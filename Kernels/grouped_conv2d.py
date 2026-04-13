"""
SparseFlow Kernels/grouped_conv2d.py

Grouped Conv2d unified orchestration layer.

This file aligns grouped/depthwise Conv2d with the same SparseFlow method
used by conv1d/conv2d/conv3d/linear:
  1. validate / support check
  2. tile / group config per sub-problem
  3. metadata build first
  4. optional stats / dense fallback
  5. active-only or all-tiles launch
  6. structured backend_meta / tile_stats return

Implementation strategy:
  - grouped conv is treated as a composition of `groups` independent Conv2d
    sparse sub-problems, each executed by the main conv2d sparse kernel
  - depthwise conv is the special case groups == C_in == C_out
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from Utils.config import PRESCAN_ACTIVITY_EPS, SPARSE_DENSE_RATIO_THRESHOLD
from Utils.sparse_helpers import choose_group_size
from Kernels.conv2d import sparse_conv2d_forward

FALLBACK_RATIO = SPARSE_DENSE_RATIO_THRESHOLD


def _pair(v) -> Tuple[int, int]:
    if isinstance(v, (tuple, list)):
        if len(v) == 1:
            return int(v[0]), int(v[0])
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _is_supported_sparse_pattern(
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
) -> bool:
    return (
        dilation == (1, 1)
        and (
            (kernel_size == (1, 1) and stride == (1, 1) and padding == (0, 0))
            or (kernel_size == (3, 3) and stride == (1, 1) and padding == (1, 1))
            or (kernel_size == (3, 3) and stride == (2, 2) and padding == (1, 1))
        )
    )


def _normalize_group_list(bufs: Optional[Sequence[Any]], groups: int) -> List[Any]:
    if bufs is None:
        return [None] * groups
    out = list(bufs)
    if len(out) < groups:
        out.extend([None] * (groups - len(out)))
    return out[:groups]


def _unpack_subgroup_result(
    result: Tuple[Any, ...],
    *,
    want_ratio: bool,
    want_tiles: bool,
) -> Tuple[torch.Tensor, float, Optional[float], Optional[Dict[str, Any]], Dict[str, Any]]:
    idx = 0
    y = result[idx]
    idx += 1
    ms = float(result[idx])
    idx += 1

    avg_active_ratio = None
    if want_ratio and idx < len(result):
        avg_active_ratio = result[idx]
        idx += 1

    tile_stats = None
    if want_tiles and idx < len(result):
        tile_stats = result[idx]
        idx += 1

    backend_meta = result[idx] if idx < len(result) and isinstance(result[idx], dict) else {}
    return y, ms, avg_active_ratio, tile_stats, backend_meta


def sparse_grouped_conv2d_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    kernel_size=3,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
    threshold: float = PRESCAN_ACTIVITY_EPS,
    w_cl_groups: Optional[Sequence[torch.Tensor]] = None,
    ag_mask_bufs: Optional[Sequence[torch.Tensor]] = None,
    tile_class_bufs: Optional[Sequence[torch.Tensor]] = None,
    active_tile_ids_bufs: Optional[Sequence[torch.Tensor]] = None,
    return_ms: bool = False,
    fallback_ratio: float = FALLBACK_RATIO,
    return_avg_active_ratio: bool = False,
    return_tile_stats: bool = False,
    return_backend_meta: bool = False,
    launch_all_tiles: bool = False,
):
    import torch.nn.functional as Fn

    N, C_IN, H_IN, W_IN = x.shape
    C_OUT = int(weight.shape[0])
    device = x.device

    k = _pair(kernel_size)
    s = _pair(stride)
    p = _pair(padding)
    d = _pair(dilation)

    need_stats = return_tile_stats or return_avg_active_ratio

    def _finalize_return(y, ms, avg_active_ratio_val=None, tile_stats_val=None, backend_meta_val=None):
        ret = (y, ms)
        if return_avg_active_ratio:
            ret = ret + (avg_active_ratio_val,)
        if return_tile_stats:
            ret = ret + (tile_stats_val,)
        if return_backend_meta:
            ret = ret + (backend_meta_val,)
        return ret

    def _dense_fallback(
        reason: str = "dense_fallback",
        avg_active_ratio_val: float = 1.0,
        tile_stats_val: Optional[Dict[str, Any]] = None,
        backend_meta_extra: Optional[Dict[str, Any]] = None,
    ):
        dense_ms = 0.0
        if return_ms:
            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            se.record()

        y = Fn.conv2d(
            x,
            weight,
            bias,
            stride=s,
            padding=p,
            dilation=d,
            groups=groups,
        )

        if return_ms:
            ee.record()
            torch.cuda.synchronize(device)
            dense_ms = se.elapsed_time(ee)

        backend_meta = {"backend": "dense_fallback", "reason": reason}
        if backend_meta_extra:
            backend_meta.update(backend_meta_extra)
        return _finalize_return(y, dense_ms, avg_active_ratio_val, tile_stats_val, backend_meta)

    if groups == 1:
        return sparse_conv2d_forward(
            x=x,
            weight=weight,
            bias=bias,
            kernel_size=k[0],
            stride=s[0],
            padding=p[0],
            dilation=d[0],
            groups=1,
            threshold=threshold,
            return_ms=return_ms,
            fallback_ratio=fallback_ratio,
            return_avg_active_ratio=return_avg_active_ratio,
            return_tile_stats=return_tile_stats,
            return_backend_meta=return_backend_meta,
            launch_all_tiles=launch_all_tiles,
        )

    if x.ndim != 4 or weight.ndim != 4:
        return _dense_fallback(reason="invalid_tensor_rank")
    if groups <= 0 or C_IN % groups != 0 or C_OUT % groups != 0:
        return _dense_fallback(reason="invalid_groups")
    if not _is_supported_sparse_pattern(k, s, p, d):
        return _dense_fallback(reason="unsupported_pattern")

    H_OUT = (H_IN + 2 * p[0] - d[0] * (k[0] - 1) - 1) // s[0] + 1
    W_OUT = (W_IN + 2 * p[1] - d[1] * (k[1] - 1) - 1) // s[1] + 1
    if H_OUT <= 0 or W_OUT <= 0:
        return _dense_fallback(reason="invalid_output_shape")

    cin_per_group = C_IN // groups
    cout_per_group = C_OUT // groups
    subgroup_group_size = choose_group_size(cin_per_group)
    subgroup_num_groups = (cin_per_group + subgroup_group_size - 1) // subgroup_group_size

    w_cl_groups = _normalize_group_list(w_cl_groups, groups)
    ag_mask_bufs = _normalize_group_list(ag_mask_bufs, groups)
    tile_class_bufs = _normalize_group_list(tile_class_bufs, groups)
    active_tile_ids_bufs = _normalize_group_list(active_tile_ids_bufs, groups)

    y = torch.empty(N, C_OUT, H_OUT, W_OUT, dtype=x.dtype, device=device)

    total_tiles = 0
    launch_count = 0
    active_tiles = 0
    zero_tiles = 0
    sparse_tiles = 0
    denseish_tiles = 0
    denseish_nonzero_tiles = 0
    active_group_slots = 0.0
    total_group_slots = 0.0
    subgroup_backend_kinds: List[str] = []
    subgroup_backend_reasons: List[str] = []
    sparse_group_count = 0
    dense_group_count = 0
    zero_group_count = 0

    if return_ms:
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()

    for group_idx in range(groups):
        cin_start = group_idx * cin_per_group
        cout_start = group_idx * cout_per_group
        cin_end = cin_start + cin_per_group
        cout_end = cout_start + cout_per_group

        x_g = x[:, cin_start:cin_end, :, :].contiguous()
        w_g = weight[cout_start:cout_end, :, :, :].contiguous()
        b_g = bias[cout_start:cout_end].contiguous() if bias is not None else None

        result = sparse_conv2d_forward(
            x=x_g,
            weight=w_g,
            bias=b_g,
            kernel_size=k[0],
            stride=s[0],
            padding=p[0],
            dilation=d[0],
            groups=1,
            threshold=threshold,
            w_cl=w_cl_groups[group_idx],
            ag_mask_buf=ag_mask_bufs[group_idx],
            tile_class_buf=tile_class_bufs[group_idx],
            return_ms=False,
            fallback_ratio=fallback_ratio,
            return_avg_active_ratio=need_stats,
            return_tile_stats=return_tile_stats,
            return_backend_meta=True,
            active_tile_ids_buf=active_tile_ids_bufs[group_idx],
            launch_all_tiles=launch_all_tiles,
        )

        y_g, _, avg_ratio_g, tile_stats_g, backend_meta_g = _unpack_subgroup_result(
            result,
            want_ratio=need_stats,
            want_tiles=return_tile_stats,
        )
        y[:, cout_start:cout_end, :, :] = y_g

        backend_kind_g = str(backend_meta_g.get("backend", "sparse_triton"))
        backend_reason_g = str(backend_meta_g.get("reason", ""))
        subgroup_backend_kinds.append(backend_kind_g)
        subgroup_backend_reasons.append(backend_reason_g)
        if backend_kind_g == "dense_fallback":
            dense_group_count += 1
        elif backend_kind_g == "zero_tiles_only":
            zero_group_count += 1
        else:
            sparse_group_count += 1

        total_tiles_g = int(backend_meta_g.get("total_tiles", 0))
        total_tiles += total_tiles_g
        launch_count += int(backend_meta_g.get("launch_count", total_tiles_g if launch_all_tiles else 0))

        avg_ratio_val_g = None
        if avg_ratio_g is not None:
            avg_ratio_val_g = float(avg_ratio_g)
        elif tile_stats_g is not None and tile_stats_g.get("avg_active_group_ratio") is not None:
            avg_ratio_val_g = float(tile_stats_g["avg_active_group_ratio"])

        if total_tiles_g > 0:
            total_group_slots += float(total_tiles_g * subgroup_num_groups)
            if avg_ratio_val_g is not None:
                active_group_slots += avg_ratio_val_g * float(total_tiles_g * subgroup_num_groups)

        if tile_stats_g is not None:
            zero_tiles += int(tile_stats_g.get("zero_tiles", 0))
            sparse_tiles += int(tile_stats_g.get("sparse_tiles", 0))
            denseish_tiles += int(tile_stats_g.get("denseish_tiles", 0))
            active_tiles += int(
                tile_stats_g.get(
                    "active_tiles",
                    int(tile_stats_g.get("sparse_tiles", 0)) + int(tile_stats_g.get("denseish_tiles", 0)),
                )
            )
            denseish_nonzero_tiles += int(tile_stats_g.get("denseish_tiles", 0))

    total_ms = 0.0
    if return_ms:
        end_ev.record()
        torch.cuda.synchronize(device)
        total_ms = start_ev.elapsed_time(end_ev)

    avg_active_ratio = None
    if need_stats and total_group_slots > 0:
        avg_active_ratio = active_group_slots / max(total_group_slots, 1.0)

    tile_stats = None
    if return_tile_stats:
        total_nonzero = sparse_tiles + denseish_tiles
        tile_stats = {
            "zero_tiles": int(zero_tiles),
            "sparse_tiles": int(sparse_tiles),
            "denseish_tiles": int(denseish_tiles),
            "total_tiles": int(total_tiles),
            "active_tiles": int(active_tiles),
            "active_tile_ratio": float(active_tiles) / max(float(total_tiles), 1.0),
            "denseish_ratio_nonzero": float(denseish_nonzero_tiles) / max(float(total_nonzero), 1.0),
            "avg_active_group_ratio": float(avg_active_ratio) if avg_active_ratio is not None else -1.0,
            "prescan_mode": "groupwise_conv2d_v1",
            "group_size_c": int(subgroup_group_size),
            "num_groups": int(subgroup_num_groups),
            "conv_groups": int(groups),
            "subgroup_backend_kinds": list(subgroup_backend_kinds),
        }

    if dense_group_count == groups:
        backend = "dense_fallback"
        reason = "all_subgroups_dense_fallback"
    elif zero_group_count == groups:
        backend = "zero_tiles_only"
        reason = "all_subgroups_zero_tiles_only"
    else:
        backend = "sparse_triton"
        reason = "grouped_conv2d_unified_v1"

    backend_meta = {
        "backend": backend,
        "reason": reason,
        "total_tiles": int(total_tiles),
        "launch_count": int(launch_count),
        "launch_mode": "all_tiles" if launch_all_tiles else "active_only",
        "active_tiles": int(active_tiles),
        "groups": int(groups),
        "group_size_c": int(subgroup_group_size),
        "num_groups": int(subgroup_num_groups),
        "subgroup_backend_kinds": list(subgroup_backend_kinds),
        "subgroup_backend_reasons": list(subgroup_backend_reasons),
    }
    if avg_active_ratio is not None:
        backend_meta["avg_active_group_ratio"] = float(avg_active_ratio)

    return _finalize_return(y, total_ms, avg_active_ratio, tile_stats, backend_meta)
