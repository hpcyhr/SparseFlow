import argparse
from typing import Iterable, Dict, Any, List

import os
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from spikingjelly.activation_based.model import spiking_resnet
from spikingjelly.activation_based import functional as sj_func
from spikingjelly.activation_based import neuron as sj_neuron
from spikingjelly.activation_based import surrogate as sj_surrogate


# ========= 模型构建 =========

def build_spiking_resnet50(
    pretrained: bool = False,
    step_mode: str = "m",
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = spiking_resnet.spiking_resnet50(
        pretrained=pretrained,
        spiking_neuron=sj_neuron.LIFNode,
        surrogate_function=sj_surrogate.Sigmoid(),
        detach_reset=True,
    )

    sj_func.set_step_mode(model, step_mode=step_mode)
    model.to(device)
    return model, device


# ========= 数据集 / Poisson 编码 =========

def build_dataset(name: str, data_root: str, image_size: int):
    if name == "cifar10":
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        ds = torchvision.datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transform,
        )
        return ds
    else:
        raise ValueError(f"Unsupported dataset: {name}")


def encode_poisson(images: torch.Tensor, T: int, H: int, W: int, device: torch.device):
    images = images.to(device)

    if images.shape[-2] != H or images.shape[-1] != W:
        images = F.interpolate(images, size=(H, W), mode="bilinear", align_corners=False)

    rate = images.clamp(0.0, 1.0)
    rate_seq = rate.unsqueeze(0).repeat(T, 1, 1, 1, 1)

    input_data = torch.bernoulli(rate_seq)
    return input_data


class LayerBlockStats:
    def __init__(self, block_sizes: Iterable[int], threshold: float = 1e-6):
        self.block_sizes = list(block_sizes)
        self.threshold = threshold
        self.stats: Dict[str, Dict[str, Any]] = {}

    def update(self, name: str, spikes: torch.Tensor):
        if spikes.ndim != 5:
            return

        T_, B_, C_, H_, W_ = spikes.shape
        flat = spikes.view(T_ * B_ * C_, H_, W_)

        bin_spikes = (flat.abs() > self.threshold)

        active = bin_spikes.sum().item()
        total = bin_spikes.numel()

        layer_stat = self.stats.setdefault(
            name,
            {"active": 0.0, "total": 0.0, "block": {}},
        )
        layer_stat["active"] += active
        layer_stat["total"] += total

        for b in self.block_sizes:
            pad_h = (b - H_ % b) % b
            pad_w = (b - W_ % b) % b

            x = F.pad(bin_spikes, (0, pad_w, 0, pad_h))
            N, Hp, Wp = x.shape
            bh, bw = Hp // b, Wp // b

            blocks = x.view(N, bh, b, bw, b).permute(0, 1, 3, 2, 4)
            block_sums = blocks.reshape(-1, b, b).sum(dim=(1, 2))

            zero_blocks = (block_sums == 0).sum().item()
            total_blocks = block_sums.numel()

            b_stat = layer_stat["block"].setdefault(
                b, {"zero": 0.0, "total_blocks": 0.0}
            )
            b_stat["zero"] += zero_blocks
            b_stat["total_blocks"] += total_blocks

    def finalize_per_layer(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for name, st in self.stats.items():
            if st["total"] == 0:
                continue
            overall_sparsity = 1.0 - (st["active"] / st["total"])

            for b, bst in st["block"].items():
                if bst["total_blocks"] == 0:
                    continue
                zero_ratio = bst["zero"] / bst["total_blocks"]
                rows.append(
                    {
                        "layer": name,
                        "block_size": b,
                        "overall_sparsity": overall_sparsity,
                        "zero_block_ratio": zero_ratio,
                    }
                )
        return rows

    @staticmethod
    def summarize_by_block(rows: List[Dict[str, Any]]):
        summary: Dict[int, Dict[str, float]] = {}
        blocks = sorted({r["block_size"] for r in rows})
        for b in blocks:
            s_list = [r["overall_sparsity"] for r in rows if r["block_size"] == b]
            z_list = [r["zero_block_ratio"] for r in rows if r["block_size"] == b]
            if not s_list:
                continue

            ms = sum(s_list) / len(s_list)
            mz = sum(z_list) / len(z_list)

            summary[b] = {
                "mean_overall_sparsity": ms,
                "mean_zero_block_ratio": mz,
            }
        return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spiking-ResNet50 稀疏度批量统计（全 testloader，单一 T/B 配置）"
    )
    parser.add_argument("--T", type=int, default=8, help="时间步 T")
    parser.add_argument("--H", type=int, default=224, help="输入高度 H")
    parser.add_argument("--W", type=int, default=224, help="输入宽度 W")
    parser.add_argument("--batch_size", type=int, default=64, help="DataLoader batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument(
        "--blocks",
        type=int,
        nargs="+",
        default=[16, 32],
        help="Block size 列表，例如：--blocks 16 32",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-6,
        help="激活判断阈值 |x| <= threshold 视为 0",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备字符串，例如 'cuda', 'cuda:0', 'cpu'；默认自动选择",
    )
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="是否加载预训练 ResNet50 权重",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10"],
        help="目前支持 cifar10",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="../../data",
        help="数据根目录",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="ALL_resnet50_cifar10_sparsity_summary.csv",
        help="输出的总 summary CSV 文件名（会追加写入）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model, dev = build_spiking_resnet50(
        pretrained=args.pretrain,
        step_mode="m",
        device=args.device,
    )
    print(f"[INFO] 模型构建完成（ResNet50），device = {dev}")

    dataset = build_dataset(
        name=args.dataset,
        data_root=args.data_root,
        image_size=args.H,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"[INFO] 测试集大小: {len(dataset)}, batch_size={args.batch_size}")

    stats = LayerBlockStats(block_sizes=args.blocks, threshold=args.threshold)

    def get_hook(name: str):
        def hook(m, i, o):
            stats.update(name, o.detach().cpu())
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, sj_neuron.BaseNode):
            hooks.append(module.register_forward_hook(get_hook(name)))

    model.eval()
    total_batches = len(loader)
    total_input_spikes = 0.0

    with torch.no_grad():
        for bi, (images, labels) in enumerate(loader):
            input_data = encode_poisson(
                images=images,
                T=args.T,
                H=args.H,
                W=args.W,
                device=dev,
            )
            total_input_spikes += input_data.sum().item()

            sj_func.reset_net(model)
            _ = model(input_data.to(dev))

            if (bi + 1) % 10 == 0 or (bi + 1) == total_batches:
                print(f"[INFO] 已处理 batch {bi+1}/{total_batches}")

    print(f"[DEBUG] 全测试集输入 spike 总数: {total_input_spikes:.0f}")

    for h in hooks:
        h.remove()

    rows = stats.finalize_per_layer()
    summary = LayerBlockStats.summarize_by_block(rows)

    print("\n========== Summary over layers (per block size) ==========")
    for b, s in summary.items():
        print(
            f"Block {b:>3}: "
            f"mean sparsity = {s['mean_overall_sparsity']*100:.2f}%, "
            f"mean zero-block = {s['mean_zero_block_ratio']*100:.2f}%"
        )

    if args.output_csv is not None and args.output_csv != "":
        header_needed = not os.path.exists(args.output_csv)
        with open(args.output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if header_needed:
                writer.writerow(
                    ["T", "B", "H", "W", "BlockSize",
                     "AvgOverallSparsity", "AvgZeroBlockRatio"]
                )
            for b, s in sorted(summary.items()):
                writer.writerow([
                    args.T,
                    args.batch_size,
                    args.H,
                    args.W,
                    b,
                    f"{s['mean_overall_sparsity'] * 100:.4f}%",
                    f"{s['mean_zero_block_ratio'] * 100:.4f}%",
                ])

        print(f"[INFO] 已将 summary 追加写入 CSV: {args.output_csv}")


if __name__ == "__main__":
    main()