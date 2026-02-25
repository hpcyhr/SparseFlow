import argparse
from typing import Iterable, Dict, Any, List, Tuple

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


# ========= 部分一：构建 Spiking ResNet-18 =========

def build_spiking_resnet18(
    pretrained: bool = False,
    step_mode: str = "m",
    device: str = None,
):
    """
    构建一个基于 spikingjelly.activation_based 的 SNN ResNet-18 模型。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 构建 SNN ResNet-18
    model = spiking_resnet.spiking_resnet18(
        pretrained=pretrained,
        spiking_neuron=sj_neuron.LIFNode,
        surrogate_function=sj_surrogate.Sigmoid(),
        detach_reset=True,
    )

    # 设置多步模式：输入 [T, B, C, H, W]
    sj_func.set_step_mode(model, step_mode=step_mode)

    model.to(device)
    return model, device


def make_dummy_input(
    T: int = 4,
    B: int = 1,
    C: int = 3,
    H: int = 224,
    W: int = 224,
    device: str = None,
):
    """
    生成一个 (T, B, C, H, W) 的随机输入。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(T, B, C, H, W, device=device)
    return x


# ========= 部分一拓展：数据集相关 =========

def build_dataset(name: str, data_root: str, image_size: int):
    """
    构建一个简单的图像数据集，目前支持:
      - 'cifar10': 使用 torchvision.datasets.CIFAR10
      - 'fake': 不加载数据集（占位，仍然用随机输入）
    """
    if name == "cifar10":
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),   # 为了激活分析，简单 ToTensor 即可
        ])
        ds = torchvision.datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transform,
        )
        return ds
    elif name == "fake":
        # 占位：仍然用随机输入
        return None
    else:
        raise ValueError(f"Unsupported dataset: {name}")


def make_input_from_dataset(
    dataset,
    T: int,
    B: int,
    H: int,
    W: int,
    device: str,
):
    """
    从数据集中取一个 batch 的图像，resize 到 HxW，
    然后做 Poisson/Bernoulli rate 编码，得到 (T, B, C, H, W) 的 0/1 spike 输入。
    """
    loader = DataLoader(dataset, batch_size=B, shuffle=False)
    images, labels = next(iter(loader))   # images: (B, C, H0, W0)

    images = images.to(device)

    # 若数据集图片大小与目标不一致，则插值到目标大小
    if images.shape[-2] != H or images.shape[-1] != W:
        images = F.interpolate(images, size=(H, W), mode="bilinear", align_corners=False)

    # CIFAR ToTensor 后范围在 [0,1]，我们直接当作发放率
    rate = images.clamp(0.0, 1.0)  # (B, C, H, W)

    # 扩展到时间维度: 先变成 (T, B, C, H, W)
    rate_seq = rate.unsqueeze(0).repeat(T, 1, 1, 1, 1)

    # 按 rate 采样 Bernoulli → 得到 0/1 spike train
    input_data = torch.bernoulli(rate_seq)

    # 简单 debug：看一下输入 spike 总数
    total_spikes = input_data.sum().item()
    print(f"[DEBUG] 输入 spike 总数: {total_spikes:.0f}")

    return input_data, labels


# ========= 部分二：Task 2 稀疏度分析函数 =========

def run_task_2_analysis(
    model: nn.Module,
    input_data: torch.Tensor,
    block_sizes: Iterable[int] = (16, 32),
    threshold: float = 1e-6,
    device: str = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor]]:
    """
    对 SNN 中所有 BaseNode (如 LIFNode) 的输出进行时空稀疏度分析，
    统计不同 block size 下 zero-block 比例。
    """
    if isinstance(block_sizes, int):
        block_sizes = [block_sizes]
    else:
        block_sizes = list(block_sizes)

    results: List[Dict[str, Any]] = []
    layer_spikes: Dict[str, torch.Tensor] = {}

    model.eval()

    # 设备选择
    if device is None:
        if input_data.is_cuda:
            device = input_data.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 重置网络状态（非常重要）
    sj_func.reset_net(model)

    # 把输入搬到对应 device
    input_data = input_data.to(device)

    # 定义 hook：抓每个 spiking neuron 的输出 (T, B, C, H, W)
    def get_hook(name: str):
        def hook(m, i, o):
            layer_spikes[name] = o.detach().cpu().float()
        return hook

    # 注册 hook 到所有 BaseNode 子类
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, sj_neuron.BaseNode):
            hooks.append(module.register_forward_hook(get_hook(name)))

    # 前向传播一次
    with torch.no_grad():
        _ = model(input_data)

    # 清理 hook
    for h in hooks:
        h.remove()

    # 打印表头
    print(f"{'Layer Name':<30} | {'Total Sparsity':<15} | {'Block Size':<10} | {'Zero Block %'}")
    print("-" * 80)

    # 遍历每一层的 spikes
    for name, spikes in layer_spikes.items():
        if spikes.ndim != 5:
            continue

        # 形状信息
        T_, B_, C_, H_, W_ = spikes.shape

        # DEBUG：每层总 spike 数（按阈值判断）
        total_spikes = (spikes.abs() > threshold).float().sum().item()
        print(f"[DEBUG] layer {name}: spike_count = {total_spikes:.0f}, shape = {tuple(spikes.shape)}")

        # 展平成 (N, H, W)，只看空间
        flat_spikes = spikes.view(T_ * B_ * C_, H_, W_)

        # 绝对值大于 threshold 视为激活
        bin_spikes = (flat_spikes.abs() > threshold).float()

        # 整体稀疏度
        overall_sparsity = 1.0 - bin_spikes.mean().item()

        for b in block_sizes:
            # padding 到 block 的倍数
            pad_h = (b - H_ % b) % b
            pad_w = (b - W_ % b) % b

            x = F.pad(bin_spikes, (0, pad_w, 0, pad_h))  # (N, H', W')
            N, Hp, Wp = x.shape
            bh, bw = Hp // b, Wp // b

            # 切成 block (N, bh, bw, b, b)
            blocks = x.view(N, bh, b, bw, b).permute(0, 1, 3, 2, 4)

            # 每个 block 求和，判断是否全零
            block_sums = blocks.reshape(-1, b, b).sum(dim=(1, 2))
            zero_blocks = (block_sums == 0).float().mean().item()

            print(
                f"{name[:30]:<30} | "
                f"{overall_sparsity:<15.2%} | "
                f"{b:<10} | "
                f"{zero_blocks:.2%}"
            )

            results.append(
                {
                    "layer": name,
                    "block_size": b,
                    "overall_sparsity": overall_sparsity,
                    "zero_block_ratio": zero_blocks,
                    "height": H_,
                    "width": W_,
                }
            )

    return results, layer_spikes


# ========= 部分三：主函数 =========

def parse_args():
    parser = argparse.ArgumentParser(
        description="Spiking-ResNet18 Task2 稀疏度分析脚本"
    )
    parser.add_argument("--T", type=int, default=4, help="时间步 T")
    parser.add_argument("--B", type=int, default=1, help="Batch size")
    parser.add_argument("--H", type=int, default=224, help="输入高度 H")
    parser.add_argument("--W", type=int, default=224, help="输入宽度 W")
    parser.add_argument(
        "--blocks",
        type=int,
        nargs="+",
        default=[16, 32],
        help="Block size 列表，例如：--blocks 16 32 64",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-6,
        help="判断 0 激活的阈值 |x| <= threshold 视为 0",
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
        help="是否加载预训练权重（默认不加载）",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "fake"],
        help="使用的输入数据：'cifar10' 使用真实 CIFAR-10 图像，'fake' 使用随机输入",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="../../data",
        help="数据集根目录（用于 torchvision.datasets）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 构建模型
    model, dev = build_spiking_resnet18(
        pretrained=args.pretrain,
        step_mode="m",
        device=args.device,
    )
    print(f"[INFO] 模型构建完成，device = {dev}")

    # 2. 准备输入：根据 dataset 决定是用真实图像还是随机输入
    if args.dataset == "fake":
        input_data = make_dummy_input(
            T=args.T,
            B=args.B,
            C=3,
            H=args.H,
            W=args.W,
            device=dev,
        )
        print(f"[INFO] 使用随机输入，形状: {input_data.shape}")
    else:
        dataset = build_dataset(
            name=args.dataset,
            data_root=args.data_root,
            image_size=args.H,   # 这里假设 H=W
        )
        input_data, labels = make_input_from_dataset(
            dataset=dataset,
            T=args.T,
            B=args.B,
            H=args.H,
            W=args.W,
            device=dev,
        )
        print(f"[INFO] 使用数据集 {args.dataset} 构造 Poisson spike 输入，形状: {input_data.shape}")

    # 3. 运行 Task 2 分析
    results, layer_spikes = run_task_2_analysis(
        model=model,
        input_data=input_data,
        block_sizes=args.blocks,
        threshold=args.threshold,
        device=dev,
    )

    # 4. 给一个简单 summary
    print("\n========== Summary (per block size) ==========")
    by_block = {}
    for r in results:
        b = r["block_size"]
        by_block.setdefault(b, []).append(r)

    for b, rs in by_block.items():
        avg_zero = sum(x["zero_block_ratio"] for x in rs) / len(rs)
        avg_sparse = sum(x["overall_sparsity"] for x in rs) / len(rs)
        print(
            f"Block {b:>3}: Avg overall sparsity = {avg_sparse:.2%}, "
            f"Avg zero-block ratio = {avg_zero:.2%} (over {len(rs)} layers)"
        )

    print("\n[INFO] 分析完成～")


if __name__ == "__main__":
    main()