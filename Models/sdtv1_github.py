"""
SDT-v1 style SNN Transformer models adapted for SparseFlow benchmark.

Reference:
  https://github.com/BICLab/Spike-Driven-Transformer

Adaptation notes:
- Keep dependencies minimal (no timm).
- Keep spike-friendly module naming (q/k/v/proj/fc1/fc2, *_lif).
- Use linear-complexity spike attention for stable ImageNet benchmark runs.
- Accept both [B, C, H, W] and [T, B, C, H, W] inputs.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, Type

import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron as sj_neuron


def _to_2tuple(x) -> Tuple[int, int]:
    if isinstance(x, (tuple, list)):
        if len(x) == 1:
            return int(x[0]), int(x[0])
        return int(x[0]), int(x[1])
    v = int(x)
    return v, v


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02):
    if hasattr(nn.init, "trunc_normal_"):
        nn.init.trunc_normal_(tensor, std=std)
    else:
        nn.init.normal_(tensor, std=std)


def _build_spike_node(
    spiking_neuron: Optional[Type[nn.Module]],
    v_threshold: float,
) -> nn.Module:
    neuron_cls = spiking_neuron or sj_neuron.LIFNode
    candidates = [
        {"tau": 2.0, "detach_reset": True, "v_threshold": v_threshold},
        {"tau": 2.0, "detach_reset": True},
        {"v_threshold": v_threshold},
        {},
    ]
    for kwargs in candidates:
        try:
            return neuron_cls(**kwargs)
        except TypeError:
            continue
    return neuron_cls()


class ConvPatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
        spiking_neuron: Optional[Type[nn.Module]],
        v_threshold: float,
    ):
        super().__init__()
        hidden = max(embed_dim // 2, 32)
        self.pre_conv = nn.Conv2d(in_channels, hidden, kernel_size=3, stride=1, padding=1, bias=False)
        self.pre_bn = nn.BatchNorm2d(hidden)
        self.pre_lif = _build_spike_node(spiking_neuron, v_threshold)

        self.proj = nn.Conv2d(hidden, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dim)
        self.proj_lif = _build_spike_node(spiking_neuron, v_threshold)

        self.rpe_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, groups=embed_dim, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dim)
        self.rpe_lif = _build_spike_node(spiking_neuron, v_threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, C, H, W]
        t, b, _, _, _ = x.shape
        y = x.flatten(0, 1).contiguous()

        y = self.pre_conv(y)
        y = self.pre_bn(y).reshape(t, b, y.shape[1], y.shape[2], y.shape[3]).contiguous()
        y = self.pre_lif(y).flatten(0, 1).contiguous()

        y = self.proj(y)
        h, w = y.shape[-2], y.shape[-1]
        y = self.proj_bn(y).reshape(t, b, y.shape[1], h, w).contiguous()
        y = self.proj_lif(y)

        y_feat = y
        y = y.flatten(0, 1).contiguous()
        y = self.rpe_conv(y)
        y = self.rpe_bn(y).reshape(t, b, y.shape[1], h, w).contiguous()
        y = self.rpe_lif(y)
        y = y + y_feat

        # [T, B, C, H, W] -> [T, B, N, C]
        y = y.flatten(-2).transpose(-1, -2).contiguous()
        return y


class SpikeMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        spiking_neuron: Optional[Type[nn.Module]],
        v_threshold: float,
    ):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc1_bn = nn.BatchNorm1d(hidden_dim)
        self.fc1_lif = _build_spike_node(spiking_neuron, v_threshold)

        self.fc2 = nn.Linear(hidden_dim, dim)
        self.fc2_bn = nn.BatchNorm1d(dim)
        self.fc2_lif = _build_spike_node(spiking_neuron, v_threshold)

        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, N, C]
        t, b, n, _ = x.shape

        y = self.fc1(x.flatten(0, 1))
        y = self.fc1_bn(y.transpose(-1, -2)).transpose(-1, -2)
        y = y.reshape(t, b, n, self.hidden_dim).contiguous()
        y = self.fc1_lif(y)

        y = self.fc2(y.flatten(0, 1))
        y = self.fc2_bn(y.transpose(-1, -2)).transpose(-1, -2)
        y = y.reshape(t, b, n, self.dim).contiguous()
        y = self.fc2_lif(y)
        return y


class SpikeLinearAttention(nn.Module):
    """
    Linear-complexity variant:
      KV = K^T V  -> [D, D]
      Y  = Q KV   -> [N, D]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        spiking_neuron: Optional[Type[nn.Module]],
        v_threshold: float,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = _build_spike_node(spiking_neuron, v_threshold)

        self.k = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = _build_spike_node(spiking_neuron, v_threshold)

        self.v = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = _build_spike_node(spiking_neuron, v_threshold)

        self.attn_lif = _build_spike_node(spiking_neuron, v_threshold)

        self.proj = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = _build_spike_node(spiking_neuron, v_threshold)

    def _linear_bn_lif(
        self,
        x: torch.Tensor,
        linear: nn.Linear,
        bn: nn.BatchNorm1d,
        lif: nn.Module,
    ) -> torch.Tensor:
        t, b, n, c = x.shape
        y = linear(x.flatten(0, 1))
        y = bn(y.transpose(-1, -2)).transpose(-1, -2).reshape(t, b, n, c).contiguous()
        y = lif(y)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, N, C]
        t, b, n, c = x.shape
        n_float = float(max(1, n))

        q = self._linear_bn_lif(x, self.q, self.q_bn, self.q_lif)
        k = self._linear_bn_lif(x, self.k, self.k_bn, self.k_lif)
        v = self._linear_bn_lif(x, self.v, self.v_bn, self.v_lif)

        q = q.reshape(t, b, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4).contiguous()
        k = k.reshape(t, b, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4).contiguous()
        v = v.reshape(t, b, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4).contiguous()

        kv = torch.matmul(k.transpose(-2, -1), v) / n_float
        y = torch.matmul(q, kv)
        y = y.transpose(2, 3).reshape(t, b, n, c).contiguous()
        y = self.attn_lif(y)

        y = self.proj(y.flatten(0, 1))
        y = self.proj_bn(y.transpose(-1, -2)).transpose(-1, -2).reshape(t, b, n, c).contiguous()
        y = self.proj_lif(y)
        return y


class SDTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        spiking_neuron: Optional[Type[nn.Module]],
        v_threshold: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpikeLinearAttention(dim, num_heads, spiking_neuron, v_threshold)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SpikeMLP(dim, int(dim * mlp_ratio), spiking_neuron, v_threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SDTV1Github(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 384,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        depth: int = 8,
        patch_size: int = 16,
        T: int = 4,
        spiking_neuron: Optional[Type[nn.Module]] = None,
        v_threshold: float = 1.0,
    ):
        super().__init__()
        self.img_size = _to_2tuple(img_size)
        self.num_classes = int(num_classes)
        self.T = int(T)

        self.patch_embed = ConvPatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=int(patch_size),
            spiking_neuron=spiking_neuron,
            v_threshold=v_threshold,
        )
        self.blocks = nn.ModuleList(
            [
                SDTBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    spiking_neuron=spiking_neuron,
                    v_threshold=v_threshold,
                )
                for _ in range(depth)
            ]
        )
        self.pre_head_lif = _build_spike_node(spiking_neuron, v_threshold)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if getattr(m, "weight", None) is not None:
                nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        y = self.patch_embed(x)
        for blk in self.blocks:
            y = blk(y)
        return y.mean(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        elif x.dim() != 5:
            raise ValueError(f"Expected 4D or 5D input, got shape={tuple(x.shape)}")

        y = self.forward_features(x)
        y = self.pre_head_lif(y)
        y = self.head(y)
        return y


def sdtv1_cifar_tiny(
    pretrained: bool = False,
    progress: bool = True,
    spiking_neuron: Optional[Type[nn.Module]] = None,
    v_threshold: float = 1.0,
    num_classes: int = 10,
    T: int = 4,
    **kwargs,
) -> nn.Module:
    del pretrained, progress, kwargs
    return SDTV1Github(
        img_size=(32, 32),
        in_channels=3,
        num_classes=num_classes,
        embed_dim=256,
        num_heads=8,
        mlp_ratio=4.0,
        depth=6,
        patch_size=4,
        T=T,
        spiking_neuron=spiking_neuron,
        v_threshold=v_threshold,
    )


def sdtv1_cifar_small(
    pretrained: bool = False,
    progress: bool = True,
    spiking_neuron: Optional[Type[nn.Module]] = None,
    v_threshold: float = 1.0,
    num_classes: int = 10,
    T: int = 4,
    **kwargs,
) -> nn.Module:
    del pretrained, progress, kwargs
    return SDTV1Github(
        img_size=(32, 32),
        in_channels=3,
        num_classes=num_classes,
        embed_dim=384,
        num_heads=12,
        mlp_ratio=4.0,
        depth=8,
        patch_size=4,
        T=T,
        spiking_neuron=spiking_neuron,
        v_threshold=v_threshold,
    )


def sdtv1_imagenet_tiny(
    pretrained: bool = False,
    progress: bool = True,
    spiking_neuron: Optional[Type[nn.Module]] = None,
    v_threshold: float = 1.0,
    num_classes: int = 1000,
    T: int = 4,
    **kwargs,
) -> nn.Module:
    del pretrained, progress, kwargs
    return SDTV1Github(
        img_size=(224, 224),
        in_channels=3,
        num_classes=num_classes,
        embed_dim=384,
        num_heads=8,
        mlp_ratio=4.0,
        depth=8,
        patch_size=16,
        T=T,
        spiking_neuron=spiking_neuron,
        v_threshold=v_threshold,
    )


def sdtv1_imagenet_small(
    pretrained: bool = False,
    progress: bool = True,
    spiking_neuron: Optional[Type[nn.Module]] = None,
    v_threshold: float = 1.0,
    num_classes: int = 1000,
    T: int = 4,
    **kwargs,
) -> nn.Module:
    del pretrained, progress, kwargs
    return SDTV1Github(
        img_size=(224, 224),
        in_channels=3,
        num_classes=num_classes,
        embed_dim=512,
        num_heads=8,
        mlp_ratio=4.0,
        depth=10,
        patch_size=16,
        T=T,
        spiking_neuron=spiking_neuron,
        v_threshold=v_threshold,
    )


MODEL_BUILDERS: Dict[str, Callable[..., nn.Module]] = {
    "sdtv1_cifar_tiny": sdtv1_cifar_tiny,
    "sdtv1_cifar_small": sdtv1_cifar_small,
    "sdtv1_imagenet_tiny": sdtv1_imagenet_tiny,
    "sdtv1_imagenet_small": sdtv1_imagenet_small,
}

