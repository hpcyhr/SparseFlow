"""
Spikformer-style SNN Transformer models adapted from:
  https://github.com/ZK-Zhou/spikformer (cifar10/model.py)

Adaptation notes:
- Remove timm dependency to keep benchmark setup lightweight.
- Keep the core Spike-Linear/Spike-Attention/Spike-MLP structure.
- Accept multi-step input directly in shape [T, B, C, H, W].
- Provide builder functions compatible with Benchmark/bench_4test.py.
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


class MLP(nn.Module):
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

        y = self.fc1(x.flatten(0, 1))  # [TB, N, hidden]
        y = self.fc1_bn(y.transpose(-1, -2)).transpose(-1, -2)
        y = y.reshape(t, b, n, self.hidden_dim).contiguous()
        y = self.fc1_lif(y)

        y = self.fc2(y.flatten(0, 1))  # [TB, N, C]
        y = self.fc2_bn(y.transpose(-1, -2)).transpose(-1, -2)
        y = y.reshape(t, b, n, self.dim).contiguous()
        y = self.fc2_lif(y)
        return y


class SSA(nn.Module):
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
        self.scale = self.head_dim ** -0.5

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
        y = linear(x.flatten(0, 1))  # [TB, N, C]
        y = bn(y.transpose(-1, -2)).transpose(-1, -2).reshape(t, b, n, c).contiguous()
        y = lif(y)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, N, C]
        t, b, n, c = x.shape

        q = self._linear_bn_lif(x, self.q, self.q_bn, self.q_lif)
        k = self._linear_bn_lif(x, self.k, self.k_bn, self.k_lif)
        v = self._linear_bn_lif(x, self.v, self.v_bn, self.v_lif)

        q = q.reshape(t, b, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4).contiguous()
        k = k.reshape(t, b, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4).contiguous()
        v = v.reshape(t, b, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4).contiguous()

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        y = torch.matmul(attn, v)  # [T, B, heads, N, head_dim]
        y = y.transpose(2, 3).reshape(t, b, n, c).contiguous()
        y = self.attn_lif(y)

        y = self.proj(y.flatten(0, 1))
        y = self.proj_bn(y.transpose(-1, -2)).transpose(-1, -2).reshape(t, b, n, c).contiguous()
        y = self.proj_lif(y)
        return y


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        spiking_neuron: Optional[Type[nn.Module]],
        v_threshold: float,
    ):
        super().__init__()
        self.attn = SSA(dim, num_heads, spiking_neuron, v_threshold)
        self.mlp = MLP(dim, int(dim * mlp_ratio), spiking_neuron, v_threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SPS(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        spiking_neuron: Optional[Type[nn.Module]],
        v_threshold: float,
    ):
        super().__init__()
        c1 = embed_dim // 8
        c2 = embed_dim // 4
        c3 = embed_dim // 2
        c4 = embed_dim

        self.conv0 = nn.Conv2d(in_channels, c1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(c1)
        self.lif0 = _build_spike_node(spiking_neuron, v_threshold)

        self.conv1 = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.lif1 = _build_spike_node(spiking_neuron, v_threshold)

        self.conv2 = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c3)
        self.lif2 = _build_spike_node(spiking_neuron, v_threshold)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(c4)
        self.lif3 = _build_spike_node(spiking_neuron, v_threshold)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.rpe_conv = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(c4)
        self.rpe_lif = _build_spike_node(spiking_neuron, v_threshold)

    def _conv_bn_lif(
        self,
        x_flat: torch.Tensor,
        conv: nn.Conv2d,
        bn: nn.BatchNorm2d,
        lif: nn.Module,
        t: int,
        b: int,
    ) -> torch.Tensor:
        y = conv(x_flat)  # [TB, C, H, W]
        h, w = y.shape[-2], y.shape[-1]
        y = bn(y).reshape(t, b, -1, h, w).contiguous()
        y = lif(y)
        return y.flatten(0, 1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, C, H, W]
        t, b, _, _, _ = x.shape
        y = x.flatten(0, 1).contiguous()

        y = self._conv_bn_lif(y, self.conv0, self.bn0, self.lif0, t, b)
        y = self._conv_bn_lif(y, self.conv1, self.bn1, self.lif1, t, b)
        y = self._conv_bn_lif(y, self.conv2, self.bn2, self.lif2, t, b)
        y = self.pool2(y)
        y = self._conv_bn_lif(y, self.conv3, self.bn3, self.lif3, t, b)
        y = self.pool3(y)

        h, w = y.shape[-2], y.shape[-1]
        y_feat = y.reshape(t, b, -1, h, w).contiguous()
        y = self.rpe_conv(y)
        y = self.rpe_bn(y).reshape(t, b, -1, h, w).contiguous()
        y = self.rpe_lif(y)
        y = y + y_feat

        # [T, B, C, H, W] -> [T, B, N, C]
        y = y.flatten(-2).transpose(-1, -2).contiguous()
        return y


class SpikformerGithub(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int] = (32, 32),
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        depth: int = 4,
        T: int = 4,
        spiking_neuron: Optional[Type[nn.Module]] = None,
        v_threshold: float = 1.0,
    ):
        super().__init__()
        self.img_size = _to_2tuple(img_size)
        self.num_classes = int(num_classes)
        self.T = int(T)

        self.patch_embed = SPS(
            in_channels=in_channels,
            embed_dim=embed_dim,
            spiking_neuron=spiking_neuron,
            v_threshold=v_threshold,
        )
        self.blocks = nn.ModuleList(
            [
                Block(
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
        # x: [T, B, C, H, W]
        y = self.patch_embed(x)
        for blk in self.blocks:
            y = blk(y)
        # token average -> [T, B, C]
        return y.mean(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compatible with both [B,C,H,W] and [T,B,C,H,W]
        if x.dim() == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        elif x.dim() != 5:
            raise ValueError(f"Expected 4D or 5D input, got shape={tuple(x.shape)}")

        y = self.forward_features(x)
        y = self.pre_head_lif(y)
        y = self.head(y)
        # [T, B, num_classes]
        return y


def spikformer_cifar_tiny(
    pretrained: bool = False,
    progress: bool = True,
    spiking_neuron: Optional[Type[nn.Module]] = None,
    v_threshold: float = 1.0,
    num_classes: int = 10,
    T: int = 4,
    **kwargs,
) -> nn.Module:
    del pretrained, progress, kwargs
    return SpikformerGithub(
        img_size=(32, 32),
        in_channels=3,
        num_classes=num_classes,
        embed_dim=192,
        num_heads=3,
        mlp_ratio=4.0,
        depth=4,
        T=T,
        spiking_neuron=spiking_neuron,
        v_threshold=v_threshold,
    )


def spikformer_cifar_small(
    pretrained: bool = False,
    progress: bool = True,
    spiking_neuron: Optional[Type[nn.Module]] = None,
    v_threshold: float = 1.0,
    num_classes: int = 10,
    T: int = 4,
    **kwargs,
) -> nn.Module:
    del pretrained, progress, kwargs
    return SpikformerGithub(
        img_size=(32, 32),
        in_channels=3,
        num_classes=num_classes,
        embed_dim=256,
        num_heads=8,
        mlp_ratio=4.0,
        depth=6,
        T=T,
        spiking_neuron=spiking_neuron,
        v_threshold=v_threshold,
    )


MODEL_BUILDERS: Dict[str, Callable[..., nn.Module]] = {
    "spikformer_cifar_tiny": spikformer_cifar_tiny,
    "spikformer_cifar_small": spikformer_cifar_small,
}

