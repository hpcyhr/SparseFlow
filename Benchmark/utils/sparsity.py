from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch


DEFAULT_SPARSITY_REGIMES = {
    "low": 0.10,
    "medium": 0.50,
    "high": 0.80,
}


def clamp_sparsity_level(level: float) -> float:
    return max(0.0, min(0.9, float(level)))


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def apply_sparsity(
    x: torch.Tensor,
    sparsity_level: float,
    mode: str = "bernoulli",
    structured_tile: Sequence[int] = (16, 16),
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    level = clamp_sparsity_level(sparsity_level)
    if level <= 0:
        return x

    mode = str(mode).lower()
    if mode == "bernoulli":
        return _apply_bernoulli_sparsity(x, level, generator=generator)
    if mode == "structured":
        return _apply_structured_sparsity(
            x,
            level,
            structured_tile=structured_tile,
            generator=generator,
        )
    raise ValueError(f"Unsupported sparsity mode: {mode}")


def make_sparse_input(
    shape: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
    sparsity_level: float,
    sparsity_mode: str,
    structured_tile: Sequence[int],
    seed: int,
) -> torch.Tensor:
    generator = make_generator(device, seed)
    x = torch.randn(tuple(shape), device=device, dtype=dtype, generator=generator)
    x = apply_sparsity(
        x,
        sparsity_level=sparsity_level,
        mode=sparsity_mode,
        structured_tile=structured_tile,
        generator=generator,
    )
    return x.contiguous()


def _apply_bernoulli_sparsity(
    x: torch.Tensor,
    sparsity_level: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    keep_prob = 1.0 - sparsity_level
    mask = torch.rand(
        x.shape,
        device=x.device,
        generator=generator,
        dtype=torch.float32,
    ) < keep_prob
    return x * mask.to(dtype=x.dtype)


def _apply_structured_sparsity(
    x: torch.Tensor,
    sparsity_level: float,
    structured_tile: Sequence[int],
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    tile_h = int(structured_tile[0]) if len(structured_tile) >= 1 else 16
    tile_w = int(structured_tile[1]) if len(structured_tile) >= 2 else tile_h
    tile_h = max(tile_h, 1)
    tile_w = max(tile_w, 1)

    if x.ndim == 1:
        return _apply_bernoulli_sparsity(x, sparsity_level, generator=generator)

    if x.ndim == 2:
        return _structured_last2d(
            x,
            sparsity_level=sparsity_level,
            tile_h=min(tile_h, x.shape[-2]),
            tile_w=min(tile_w, x.shape[-1]),
            generator=generator,
        )

    prefix = x.shape[:-2]
    h, w = x.shape[-2], x.shape[-1]
    x2d = x.reshape(-1, h, w)
    y2d = _structured_last2d(
        x2d,
        sparsity_level=sparsity_level,
        tile_h=min(tile_h, h),
        tile_w=min(tile_w, w),
        generator=generator,
    )
    return y2d.reshape(*prefix, h, w)


def _structured_last2d(
    x: torch.Tensor,
    sparsity_level: float,
    tile_h: int,
    tile_w: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    # x: [..., H, W]
    *prefix, h, w = x.shape
    n_th = int(math.ceil(h / tile_h))
    n_tw = int(math.ceil(w / tile_w))
    keep_prob = 1.0 - sparsity_level

    tile_keep = torch.rand(
        (*prefix, n_th, n_tw),
        device=x.device,
        generator=generator,
        dtype=torch.float32,
    ) < keep_prob
    tile_keep = tile_keep.to(dtype=x.dtype)

    expanded = (
        tile_keep.repeat_interleave(tile_h, dim=-2)
        .repeat_interleave(tile_w, dim=-1)
    )
    expanded = expanded[..., :h, :w]
    return x * expanded
