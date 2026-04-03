from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from benchmark.ops.common import finalize_failed_case, finalize_ok_case, make_case_base
    from benchmark.utils.sparsity import make_sparse_input
    from benchmark.utils.timer import BenchmarkTimer
except ModuleNotFoundError:
    from ops.common import finalize_failed_case, finalize_ok_case, make_case_base  # type: ignore
    from utils.sparsity import make_sparse_input  # type: ignore
    from utils.timer import BenchmarkTimer  # type: ignore


CONV_SCALES = {
    "small": {"batch": 8, "channels": 32, "height": 28, "width": 28},
    "medium": {"batch": 16, "channels": 64, "height": 56, "width": 56},
    "large": {"batch": 32, "channels": 128, "height": 112, "width": 112},
}


def run_conv_suite(
    timer: BenchmarkTimer,
    device: torch.device,
    dtype: torch.dtype,
    scale: str,
    sparsity_regime: str,
    sparsity_level: float,
    sparsity_mode: str,
    warmup: int,
    iters: int,
    seed: int,
    structured_tile: Sequence[int],
) -> List[Dict]:
    if scale not in CONV_SCALES:
        raise ValueError(f"Unknown Conv scale: {scale}")

    base = CONV_SCALES[scale]
    b = int(base["batch"])
    c = int(base["channels"])
    h = int(base["height"])
    w = int(base["width"])

    variants = [
        ("conv_3x3", "3x3_s1_p1", 3, 1, 1),
        ("conv_3x3", "3x3_s2_p1", 3, 2, 1),
        ("conv_1x1", "1x1_s1_p0", 1, 1, 0),
    ]

    results: List[Dict] = []
    for idx, (operator, variant, kernel, stride, padding) in enumerate(variants):
        case_seed = int(seed + idx * 17)
        shape_meta = {
            "input": [b, c, h, w],
            "weight": [c, c, kernel, kernel],
            "kernel_size": kernel,
            "stride": stride,
            "padding": padding,
        }
        case = make_case_base(
            operator=operator,
            variant=variant,
            scale=scale,
            sparsity_regime=sparsity_regime,
            sparsity_level=sparsity_level,
            sparsity_mode=sparsity_mode,
            seed=case_seed,
            shape_meta=shape_meta,
            dtype=dtype,
        )
        try:
            result = _run_one_conv_case(
                case=case,
                timer=timer,
                device=device,
                dtype=dtype,
                b=b,
                c=c,
                h=h,
                w=w,
                kernel=kernel,
                stride=stride,
                padding=padding,
                warmup=warmup,
                iters=iters,
                sparsity_level=sparsity_level,
                sparsity_mode=sparsity_mode,
                seed=case_seed,
                structured_tile=structured_tile,
            )
        except Exception as exc:
            result = finalize_failed_case(case, exc)
        results.append(result)
    return results


def _run_one_conv_case(
    case: Dict,
    timer: BenchmarkTimer,
    device: torch.device,
    dtype: torch.dtype,
    b: int,
    c: int,
    h: int,
    w: int,
    kernel: int,
    stride: int,
    padding: int,
    warmup: int,
    iters: int,
    sparsity_level: float,
    sparsity_mode: str,
    seed: int,
    structured_tile: Sequence[int],
) -> Dict:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    x = make_sparse_input(
        shape=(b, c, h, w),
        device=device,
        dtype=dtype,
        sparsity_level=sparsity_level,
        sparsity_mode=sparsity_mode,
        structured_tile=structured_tile,
        seed=seed + 1,
    )

    dense_conv = nn.Conv2d(
        in_channels=c,
        out_channels=c,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        bias=True,
    ).to(device=device, dtype=dtype).eval()

    sparse_conv = _build_sparse_conv_module(dense_conv)
    sparse_conv = sparse_conv.to(device=device).eval()

    with torch.no_grad():
        dense_ref = dense_conv(x)
        sparse_ref = sparse_conv(x)

    def dense_fn() -> torch.Tensor:
        with torch.no_grad():
            return dense_conv(x)

    def sparse_fn() -> torch.Tensor:
        with torch.no_grad():
            return sparse_conv(x)

    dense_t = timer.run(dense_fn, warmup=warmup, iters=iters)
    sparse_t = timer.run(sparse_fn, warmup=warmup, iters=iters)
    return finalize_ok_case(
        case=case,
        dense_latency_ms=float(dense_t["avg_ms"]),
        sparse_latency_ms=float(sparse_t["avg_ms"]),
        dense_out=dense_ref,
        sparse_out=sparse_ref,
    )


def _build_sparse_conv_module(dense_conv: nn.Conv2d) -> nn.Module:
    try:
        from Ops.sparse_conv2d import SparseConv2d
    except Exception as exc:
        raise RuntimeError(
            "TODO hook: SparseFlow SparseConv2d is not available in Ops/sparse_conv2d.py"
        ) from exc

    sparse_conv = SparseConv2d.from_dense(dense_conv, return_ms=False)
    if hasattr(sparse_conv, "set_inference_mode"):
        sparse_conv.set_inference_mode(True)
    if hasattr(sparse_conv, "collect_diag"):
        sparse_conv.collect_diag = False
    return sparse_conv


def dense_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
) -> torch.Tensor:
    return F.conv2d(x, weight, bias=bias, stride=stride, padding=padding)


def sparse_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    threshold: float = 1e-6,
) -> torch.Tensor:
    """SparseFlow compatibility hook: sparse_conv2d(x, weight, ...)."""
    try:
        from Ops.sparse_conv2d import SparseConv2d
    except Exception as exc:
        raise RuntimeError(
            "TODO hook: SparseFlow SparseConv2d is not available in Ops/sparse_conv2d.py"
        ) from exc

    out_channels, in_channels, kh, kw = weight.shape
    conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(kh, kw),
        stride=stride,
        padding=padding,
        bias=bias is not None,
    ).to(device=x.device, dtype=weight.dtype)
    with torch.no_grad():
        conv.weight.copy_(weight)
        if bias is not None and conv.bias is not None:
            conv.bias.copy_(bias)
    sparse = SparseConv2d.from_dense(conv, threshold=threshold, return_ms=False).to(x.device).eval()
    if hasattr(sparse, "set_inference_mode"):
        sparse.set_inference_mode(True)
    return sparse(x)
