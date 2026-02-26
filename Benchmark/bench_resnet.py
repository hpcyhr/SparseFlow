"""
SparseFlow Benchmark — Spiking-ResNet on CIFAR-10/100

支持的网络：resnet34, resnet50, resnet101, resnet152
支持的数据集：cifar10, cifar100
比较对象：SparseFlow sparse kernel vs Triton dense kernel vs cuDNN (F.conv2d)

用法：
    cd ~/SparseFlow
    python Benchmark/bench_resnet.py --model resnet34 --dataset cifar10
    python Benchmark/bench_resnet.py --model resnet50 --dataset cifar100 --T 32
    python Benchmark/bench_resnet.py --model resnet34 --dataset cifar10 --gpu 2

    # 批量跑所有组合
    bash Benchmark/run_all.sh
"""

import sys
from pathlib import Path

# 项目根目录
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from spikingjelly.activation_based.model import spiking_resnet
from spikingjelly.activation_based import functional as sj_func
from spikingjelly.activation_based import neuron as sj_neuron

from Kernels.conv2d import (
    prescan_kernel, sparse_conv3x3_weighted_kernel, dense_conv3x3_kernel,
    sparse_conv2d_forward,
)
from Kernels.linear import sparse_linear_forward
from Kernels.batchnorm2d import sparse_batchnorm2d_forward
from Ops.sparse_conv2d import SparseConv2d
from Ops.sparse_linear import SparseLinear
from Ops.sparse_batchnorm2d import SparseBatchNorm2d

import triton

# =============================================================================
# 注册表
# =============================================================================
SPIKE_OUTPUT_OPS = (
    sj_neuron.LIFNode,
    sj_neuron.IFNode,
    sj_neuron.ParametricLIFNode,
)

# =============================================================================
# Block 大小选择
# =============================================================================
def select_block_size(H, W):
    spatial = min(H, W)
    if spatial >= 56:
        return 16
    elif spatial >= 14:
        return 8
    else:
        return None

# =============================================================================
# 全局 device（在 main 中设定）
# =============================================================================
DEVICE = None

def sync():
    """同步当前设备"""
    torch.cuda.synchronize(DEVICE)

def make_event():
    """在当前设备上创建 CUDA event"""
    return torch.cuda.Event(enable_timing=True)

# =============================================================================
# Triton dense / sparse / cuDNN benchmark 函数
# =============================================================================

def run_sparse_conv(feat, block):
    """SparseFlow 稀疏卷积 benchmark（使用 prescan + 跳零逻辑，box filter）"""
    feat = feat.contiguous()
    N, C, H, W = feat.shape
    GRID_H = triton.cdiv(H, block)
    GRID_W = triton.cdiv(W, block)
    total = N * C * GRID_H * GRID_W

    flags = torch.empty(total, dtype=torch.int32, device=feat.device)
    prescan_kernel[(total,)](
        feat, flags, N, C, H, W, GRID_H, GRID_W,
        BLOCK_H=block, BLOCK_W=block, THRESHOLD=1e-6,
    )
    sync()

    nz_idx = flags.nonzero(as_tuple=False).squeeze(1).int()
    num_nz = nz_idx.numel()
    y = torch.zeros_like(feat)
    if num_nz == 0:
        return y, 0.0

    start, end = make_event(), make_event()
    start.record()
    dense_conv3x3_kernel[(num_nz,)](
        feat, y, N, C, H, W, GRID_H, GRID_W,
        BLOCK_H=block, BLOCK_W=block,
    )
    end.record()
    sync()
    return y, start.elapsed_time(end)


def run_dense_conv(feat, block):
    """稠密 Triton 卷积 benchmark (box filter)"""
    feat = feat.contiguous()
    N, C, H, W = feat.shape
    GRID_H = triton.cdiv(H, block)
    GRID_W = triton.cdiv(W, block)
    total = N * C * GRID_H * GRID_W
    y = torch.empty_like(feat)
    start, end = make_event(), make_event()
    start.record()
    dense_conv3x3_kernel[(total,)](
        feat, y, N, C, H, W, GRID_H, GRID_W,
        BLOCK_H=block, BLOCK_W=block,
    )
    end.record()
    sync()
    return y, start.elapsed_time(end)


def run_cudnn_conv(feat, module):
    """cuDNN conv 基准"""
    start, end = make_event(), make_event()
    start.record()
    with torch.no_grad():
        y = F.conv2d(feat, module.weight, module.bias,
                     module.stride, module.padding, module.dilation, module.groups)
    end.record()
    sync()
    return y, start.elapsed_time(end)


def run_sparse_linear_bench(feat_2d, module):
    """稀疏 Linear benchmark"""
    start, end = make_event(), make_event()
    start.record()
    y, _ = sparse_linear_forward(feat_2d.contiguous(), module.weight, module.bias)
    end.record()
    sync()
    return start.elapsed_time(end)


def run_dense_linear_bench(feat_2d, module):
    """稠密 Linear benchmark"""
    start, end = make_event(), make_event()
    start.record()
    with torch.no_grad():
        y = F.linear(feat_2d, module.weight, module.bias)
    end.record()
    sync()
    return start.elapsed_time(end)


def run_sparse_bn_bench(feat_4d, module):
    """稀疏 BN benchmark"""
    _, ms = sparse_batchnorm2d_forward(feat_4d, module)
    return ms


def run_dense_bn_bench(feat_4d, module):
    """稠密 BN benchmark"""
    start, end = make_event(), make_event()
    start.record()
    with torch.no_grad():
        y = F.batch_norm(feat_4d, module.running_mean, module.running_var,
                         module.weight, module.bias, False, 0.0, module.eps)
    end.record()
    sync()
    return start.elapsed_time(end)


# =============================================================================
# 网络分析
# =============================================================================

class LayerInfo:
    def __init__(self, name, module, layer_type, block_size=None, H=0, W=0):
        self.name = name
        self.module = module
        self.layer_type = layer_type  # "conv2d" | "linear" | "batchnorm2d"
        self.block_size = block_size
        self.H, self.W = H, W
        self.total_dense_ms = 0.0
        self.total_sparse_ms = 0.0
        self.total_cudnn_ms = 0.0
        self.total_zeros = 0
        self.total_elems = 0


def analyze_network(model, sample_input, device):
    """分析网络，识别脉冲后继的 Conv / BN / Linear 层"""
    input_shapes = {}
    hooks = []

    def make_hook(name):
        def hook(m, inp, out):
            if isinstance(inp, (tuple, list)) and len(inp) > 0:
                x = inp[0]
                if isinstance(x, torch.Tensor):
                    input_shapes[name] = tuple(x.shape)
        return hook

    module_list = list(model.named_modules())
    for name, module in module_list:
        hooks.append(module.register_forward_hook(make_hook(name)))
    sj_func.reset_net(model)
    with torch.no_grad():
        _ = model(sample_input)
    for h in hooks:
        h.remove()

    layer_infos = {}
    visited = set()

    for i, (name, module) in enumerate(module_list):
        if not isinstance(module, SPIKE_OUTPUT_OPS):
            continue

        for j in range(i + 1, min(i + 10, len(module_list))):
            next_name, next_module = module_list[j]
            if next_name in visited:
                continue

            ishape = input_shapes.get(next_name)
            if ishape is None:
                continue

            # --- Conv2d ---
            if isinstance(next_module, nn.Conv2d):
                k = next_module.kernel_size
                s = next_module.stride
                p = next_module.padding
                if k == (3, 3) and s == (1, 1) and p == (1, 1):
                    if len(ishape) == 5:
                        H, W = ishape[3], ishape[4]
                    elif len(ishape) == 4:
                        H, W = ishape[2], ishape[3]
                    else:
                        continue
                    block = select_block_size(H, W)
                    if block is None:
                        visited.add(next_name)
                        break
                    layer_infos[next_name] = LayerInfo(
                        next_name, next_module, "conv2d", block, H, W)
                    visited.add(next_name)
                    print(f"  [CONV  ] LIF={name:<30} -> {next_name:<30} H={H} Block={block}")
                    break

            # --- BatchNorm2d ---
            elif isinstance(next_module, nn.BatchNorm2d):
                if len(ishape) == 5:
                    H, W = ishape[3], ishape[4]
                elif len(ishape) == 4:
                    H, W = ishape[2], ishape[3]
                else:
                    continue
                layer_infos[next_name] = LayerInfo(
                    next_name, next_module, "batchnorm2d", None, H, W)
                visited.add(next_name)
                print(f"  [BN    ] LIF={name:<30} -> {next_name:<30} H={H}")
                continue

            # --- Linear ---
            elif isinstance(next_module, nn.Linear):
                layer_infos[next_name] = LayerInfo(
                    next_name, next_module, "linear")
                visited.add(next_name)
                print(f"  [LINEAR] LIF={name:<30} -> {next_name:<30}")
                break

    return layer_infos


# =============================================================================
# 模型构建
# =============================================================================

MODEL_BUILDERS = {
    "resnet34": spiking_resnet.spiking_resnet34,
    "resnet50": spiking_resnet.spiking_resnet50,
    "resnet101": spiking_resnet.spiking_resnet101,
    "resnet152": spiking_resnet.spiking_resnet152,
}

def build_model(model_name, device, v_threshold=0.5):
    builder = MODEL_BUILDERS[model_name]
    model = builder(
        pretrained=True,
        spiking_neuron=sj_neuron.LIFNode,
        surrogate_function=sj_neuron.LIFNode().surrogate_function,
        detach_reset=True,
    )
    for m in model.modules():
        if isinstance(m, sj_neuron.LIFNode):
            m.v_threshold = v_threshold
    sj_func.set_step_mode(model, step_mode="m")
    model.to(device).eval()
    return model


def build_dataset(dataset_name, data_root):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if dataset_name == "cifar10":
        ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    elif dataset_name == "cifar100":
        ds = datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return ds


# =============================================================================
# 主流程
# =============================================================================

def main():
    global DEVICE

    parser = argparse.ArgumentParser(description="SparseFlow Benchmark")
    parser.add_argument("--model", type=str, default="resnet34",
                        choices=list(MODEL_BUILDERS.keys()))
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100"])
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--v_threshold", type=float, default=0.5)
    parser.add_argument("--power", type=float, default=250.0, help="GPU TDP (W)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU device id, -1=auto select (most free memory)")
    args = parser.parse_args()

    # ---- 自动选择空闲显存最多的 GPU ----
    if args.gpu >= 0:
        gpu_id = args.gpu
    else:
        gpu_id = 0
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            max_free = 0
            for i in range(num_gpus):
                free, total = torch.cuda.mem_get_info(i)
                print(f"  GPU {i}: {free / 1024**3:.1f} GB free / {total / 1024**3:.1f} GB total")
                if free > max_free:
                    max_free = free
                    gpu_id = i
        print(f"  → 自动选择 GPU {gpu_id} (空闲显存最大)")

    DEVICE = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(DEVICE)
    device = DEVICE
    title = f"Spiking-{args.model.upper()} on {args.dataset.upper()}"

    # ---- 构建模型 ----
    print(f"[1/4] 构建 {title} (on {device}) ...")
    model = build_model(args.model, device, args.v_threshold)

    # ---- 数据集 ----
    print(f"[2/4] 加载 {args.dataset} 测试集 (root={args.data_root}) ...")
    ds = build_dataset(args.dataset, args.data_root)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=8, pin_memory=True)
    print(f"  测试集共 {len(ds)} 张，{len(loader)} 个 batch")

    # ---- 分析网络 ----
    print(f"[3/4] 分析网络，识别脉冲后继算子 ...")
    imgs_s, _ = next(iter(loader))
    sample_input = torch.bernoulli(
        imgs_s[:4].to(device).unsqueeze(0).repeat(args.T, 1, 1, 1, 1).clamp(0, 1)
    )
    layer_infos = analyze_network(model, sample_input, device)

    if not layer_infos:
        print("未找到符合条件的目标层，退出。")
        return
    print(f"\n共找到 {len(layer_infos)} 个目标层\n")

    # ---- Hook 捕获特征图 ----
    captured = {}

    def make_capture_hook(name):
        def hook(m, inp, out):
            x = inp[0]
            if isinstance(x, torch.Tensor):
                x = x.detach()
                if x.dim() == 5:
                    T, B, C, H, W = x.shape
                    x = x.reshape(T * B, C, H, W)
                # 确保在正确的 GPU 设备上，contiguous，独立副本
                captured[name] = x.to(device).contiguous().clone()
        return hook

    hook_handles = []
    for name, info in layer_infos.items():
        h = info.module.register_forward_hook(make_capture_hook(name))
        hook_handles.append(h)

    # ---- 预热 ----
    print(f"[4/4] 开始基准测试 (warmup={args.warmup}) ...")
    for i, (imgs, _) in enumerate(loader):
        if i >= args.warmup:
            break
        inp = torch.bernoulli(
            imgs.to(device).unsqueeze(0).repeat(args.T, 1, 1, 1, 1).clamp(0, 1)
        )
        sj_func.reset_net(model)
        with torch.no_grad():
            _ = model(inp)
    print(f"  预热 {args.warmup} batch 完成")

    # ---- 正式测试 ----
    for batch_idx, (imgs, _) in enumerate(loader):
        inp = torch.bernoulli(
            imgs.to(device).unsqueeze(0).repeat(args.T, 1, 1, 1, 1).clamp(0, 1)
        )
        sj_func.reset_net(model)
        with torch.no_grad():
            _ = model(inp)

        for name, info in layer_infos.items():
            feat = captured.get(name)
            if feat is None:
                continue

            if info.layer_type == "conv2d":
                block = info.block_size
                info.total_zeros += (feat.abs() <= 1e-6).sum().item()
                info.total_elems += feat.numel()
                _, dense_ms = run_dense_conv(feat, block)
                _, sparse_ms = run_sparse_conv(feat, block)
                _, cudnn_ms = run_cudnn_conv(feat, info.module)
                info.total_dense_ms += dense_ms
                info.total_sparse_ms += sparse_ms
                info.total_cudnn_ms += cudnn_ms

            elif info.layer_type == "batchnorm2d":
                info.total_zeros += (feat.abs().sum(dim=1) <= 1e-6).sum().item()
                info.total_elems += feat.shape[0] * feat.shape[2] * feat.shape[3]
                info.total_dense_ms += run_dense_bn_bench(feat, info.module)
                info.total_sparse_ms += run_sparse_bn_bench(feat, info.module)
                info.total_cudnn_ms += run_dense_bn_bench(feat, info.module)

            elif info.layer_type == "linear":
                feat_2d = feat.reshape(feat.shape[0], -1) if feat.dim() > 2 else feat
                info.total_zeros += (feat_2d.abs().sum(dim=1) <= 1e-6).sum().item()
                info.total_elems += feat_2d.shape[0]
                info.total_dense_ms += run_dense_linear_bench(feat_2d, info.module)
                info.total_sparse_ms += run_sparse_linear_bench(feat_2d, info.module)
                info.total_cudnn_ms += run_dense_linear_bench(feat_2d, info.module)

        if (batch_idx + 1) % 50 == 0:
            pct = (batch_idx + 1) / len(loader) * 100
            print(f"  [{pct:5.1f}%] batch {batch_idx+1}/{len(loader)}")

    for h in hook_handles:
        h.remove()

    # ---- 输出报告 ----
    print("\n" + "=" * 90)
    print(f"{'SparseFlow Benchmark — ' + title:^90}")
    print(f"{'T=' + str(args.T) + '  BS=' + str(args.batch_size) + '  Power=' + str(args.power) + 'W':^90}")
    print("=" * 90)
    print(f"{'Layer':<35} {'Type':<6} {'Blk':>4} {'H':>4} "
          f"{'Sparsity':>9} {'vs Triton':>10} {'vs cuDNN':>9}")
    print("-" * 90)

    total_d_j = 0.0
    total_s_j = 0.0

    for name, info in layer_infos.items():
        if info.total_elems == 0:
            continue
        sparsity = info.total_zeros / info.total_elems * 100
        speedup = info.total_dense_ms / max(info.total_sparse_ms, 1e-9)
        cudnn_speedup = info.total_cudnn_ms / max(info.total_sparse_ms, 1e-9)

        ed = (info.total_dense_ms / 1000.0) * args.power
        es = (info.total_sparse_ms / 1000.0) * args.power
        total_d_j += ed
        total_s_j += es

        sname = name if len(name) <= 34 else "..." + name[-31:]
        blk = str(info.block_size) if info.block_size else "-"
        h = str(info.H) if info.H else "-"
        print(f"{sname:<35} {info.layer_type:<6} {blk:>4} {h:>4} "
              f"{sparsity:>8.2f}% {speedup:>9.2f}x {cudnn_speedup:>8.2f}x")

    print("-" * 90)
    all_d = sum(i.total_dense_ms for i in layer_infos.values())
    all_s = sum(i.total_sparse_ms for i in layer_infos.values())
    all_c = sum(i.total_cudnn_ms for i in layer_infos.values())
    print(f"{'[TOTAL]':<35} {'':>6} {'':>4} {'':>4} {'':>9} "
          f"{all_d / max(all_s, 1e-9):>9.2f}x {all_c / max(all_s, 1e-9):>8.2f}x")
    print(f"\n  Dense  Energy : {total_d_j:.4f} J")
    print(f"  Sparse Energy : {total_s_j:.4f} J")
    print(f"  Energy Saving : {(1 - total_s_j / max(total_d_j, 1e-9)) * 100:.2f}%")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()