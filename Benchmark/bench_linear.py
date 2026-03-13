"""
SparseFlow Benchmark — Sparse Linear on Spiking-ResNet

比较对象：SparseFlow sparse Linear (tile-level Dynamic-K) vs PyTorch F.linear
验证内容：
  1. 逐层算子验证：Sparse Linear 输出 vs F.linear 输出的数值误差
  2. 性能测试：稀疏率、延迟、加速比

用法：
    cd ~/SparseFlow
    python Benchmark/bench_linear.py --model resnet34 --dataset cifar10
    python Benchmark/bench_linear.py --model resnet50 --dataset cifar100 --T 32
    python Benchmark/bench_linear.py --model resnet50 --dataset cifar10 --batch_size 64

说明：
    ResNet 中 Linear 层只有最后的分类头 (fc)，但 LIF 输出的脉冲数据
    展平后具有高稀疏性，是验证 tile-level Dynamic-K 效果的理想场景。
"""

import sys
from pathlib import Path

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

from Kernels.linear import sparse_linear_forward

# =============================================================================
# 注册表
# =============================================================================
SPIKE_OUTPUT_OPS = (
    sj_neuron.LIFNode,
    sj_neuron.IFNode,
    sj_neuron.ParametricLIFNode,
)

DEVICE = None

def sync():
    torch.cuda.synchronize(DEVICE)

def make_event():
    return torch.cuda.Event(enable_timing=True)

# =============================================================================
# Benchmark 函数
# =============================================================================

def run_sparse_linear(feat, linear_module):
    """SparseFlow 稀疏 Linear benchmark"""
    feat = feat.contiguous()
    y, sparse_ms = sparse_linear_forward(
        x=feat,
        weight=linear_module.weight.contiguous(),
        bias=linear_module.bias,
        threshold=1e-6,
        return_ms=True,
    )
    return y, sparse_ms


def run_dense_linear(feat, linear_module):
    """PyTorch F.linear 基准"""
    start, end = make_event(), make_event()
    start.record()
    with torch.no_grad():
        y = F.linear(feat, linear_module.weight, linear_module.bias)
    end.record()
    sync()
    return y, start.elapsed_time(end)


# =============================================================================
# 逐层数值验证
# =============================================================================

def verify_layer(feat, linear_module, layer_name):
    """验证单层：Sparse 输出 vs F.linear 输出的数值差异"""
    feat = feat.contiguous()

    with torch.no_grad():
        y_dense = F.linear(feat, linear_module.weight, linear_module.bias)

    y_sparse, _ = sparse_linear_forward(
        x=feat,
        weight=linear_module.weight.contiguous(),
        bias=linear_module.bias,
        threshold=1e-6,
    )

    diff = (y_sparse - y_dense).float()
    y_ref = y_dense.float()

    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()

    denom = y_ref.abs().clamp(min=1.0)
    max_rel = (diff.abs() / denom).max().item()

    flat_s = y_sparse.flatten().float()
    flat_c = y_dense.flatten().float()
    norm_s = flat_s.norm()
    norm_c = flat_c.norm()
    if norm_s < 1e-8 and norm_c < 1e-8:
        cos = 1.0
    elif norm_s < 1e-8 or norm_c < 1e-8:
        cos = 0.0
    else:
        cos = F.cosine_similarity(flat_s.unsqueeze(0), flat_c.unsqueeze(0)).item()

    return max_abs, mean_abs, max_rel, cos


# =============================================================================
# 网络分析：识别 LIF 后继的 Linear 层
# =============================================================================

class LayerInfo:
    def __init__(self, name, module, H=0, W=0):
        self.name = name
        self.module = module
        self.H, self.W = H, W
        self.total_sparse_ms = 0.0
        self.total_dense_ms = 0.0
        self.total_zeros = 0
        self.total_elems = 0
        self.verify_max_abs = 0.0
        self.verify_mean_abs = 0.0
        self.verify_max_rel = 0.0
        self.verify_cos_sum = 0.0
        self.verify_count = 0


def analyze_network(model, sample_input, device):
    """分析网络，识别脉冲后继的 Linear 层"""
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

    TRANSPARENT_TYPES = (nn.BatchNorm2d, nn.BatchNorm1d,
                         nn.Dropout, nn.Dropout2d,
                         nn.Identity, nn.Flatten,
                         nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d,
                         nn.ReLU, nn.ReLU6, nn.LeakyReLU)

    for i, (name, module) in enumerate(module_list):
        if not isinstance(module, SPIKE_OUTPUT_OPS):
            continue

        for j in range(i + 1, min(i + 20, len(module_list))):
            next_name, next_module = module_list[j]
            if next_name in visited:
                continue
            if isinstance(next_module, SPIKE_OUTPUT_OPS):
                break
            if isinstance(next_module, TRANSPARENT_TYPES):
                continue

            if isinstance(next_module, nn.Linear):
                layer_infos[next_name] = LayerInfo(next_name, next_module)
                visited.add(next_name)
                in_f = next_module.in_features
                out_f = next_module.out_features
                print(f"  [LINEAR] LIF={name:<30} -> {next_name:<30} "
                      f"({in_f}→{out_f})")
                continue

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

    parser = argparse.ArgumentParser(description="SparseFlow Benchmark — Sparse Linear")
    parser.add_argument("--model", type=str, default="resnet34",
                        choices=list(MODEL_BUILDERS.keys()))
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100"])
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--v_threshold", type=float, default=0.5)
    parser.add_argument("--power", type=float, default=250.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--gpu", type=int, default=-1)
    args = parser.parse_args()

    # ---- GPU 选择 ----
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
        print(f"  → 自动选择 GPU {gpu_id}")

    DEVICE = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(DEVICE)
    device = DEVICE
    title = f"Spiking-{args.model.upper()} Linear on {args.dataset.upper()}"

    # ---- 构建模型 ----
    print(f"\n[1/5] 构建 {title} (on {device}) ...")
    model = build_model(args.model, device, args.v_threshold)

    # ---- 数据集 ----
    print(f"[2/5] 加载 {args.dataset} 测试集 ...")
    ds = build_dataset(args.dataset, args.data_root)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=8, pin_memory=True)
    print(f"  测试集共 {len(ds)} 张，{len(loader)} 个 batch")

    # ---- 分析网络 ----
    print(f"[3/5] 分析网络，识别脉冲后继 Linear 层 ...")
    imgs_s, _ = next(iter(loader))
    sample_input = torch.bernoulli(
        imgs_s[:4].to(device).unsqueeze(0).repeat(args.T, 1, 1, 1, 1).clamp(0, 1)
    )
    layer_infos = analyze_network(model, sample_input, device)

    if not layer_infos:
        print("  未找到脉冲后继的 Linear 层，退出。")
        return
    print(f"\n  共找到 {len(layer_infos)} 个目标 Linear 层\n")

    # ---- Hook 捕获 Linear 输入特征 ----
    captured = {}

    def make_capture_hook(name):
        def hook(m, inp, out):
            x = inp[0]
            if isinstance(x, torch.Tensor):
                x = x.detach()
                # Linear 输入可能是 5D/3D/2D，统一 flatten 到 2D
                if x.dim() == 5:
                    T, B, C, H, W = x.shape
                    x = x.reshape(T * B, C * H * W)
                elif x.dim() == 3:
                    T, B, C = x.shape
                    x = x.reshape(T * B, C)
                captured[name] = x.to(device).contiguous().clone()
        return hook

    hook_handles = []
    for name, info in layer_infos.items():
        h = info.module.register_forward_hook(make_capture_hook(name))
        hook_handles.append(h)

    # ---- 逐层正确性验证 ----
    print(f"[4/5] 逐层算子正确性验证 ...")
    verify_batches = 3
    verify_count = 0
    for imgs, _ in loader:
        if verify_count >= verify_batches:
            break
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
            max_abs, mean_abs, max_rel, cos = verify_layer(feat, info.module, name)
            info.verify_max_abs = max(info.verify_max_abs, max_abs)
            info.verify_mean_abs += mean_abs
            info.verify_max_rel = max(info.verify_max_rel, max_rel)
            info.verify_cos_sum += cos
            info.verify_count += 1

        verify_count += 1

    print(f"\n  {'Layer':<35} {'max_abs':>10} {'mean_abs':>10} {'max_rel':>10} {'cosine':>12} {'Status':>8}")
    print(f"  {'-'*87}")
    for name, info in layer_infos.items():
        if info.verify_count == 0:
            continue
        avg_mean = info.verify_mean_abs / info.verify_count
        avg_cos = info.verify_cos_sum / info.verify_count
        ok = info.verify_max_abs < 0.1 and avg_cos > 0.999
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {name:<35} {info.verify_max_abs:>10.6f} {avg_mean:>10.6f} "
              f"{info.verify_max_rel:>10.6f} {avg_cos:>12.8f} {status:>8}")

    # ---- 预热 + 性能测试 ----
    print(f"\n[5/5] 性能基准测试 (warmup={args.warmup}) ...")
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

            info.total_zeros += (feat.abs() <= 1e-6).sum().item()
            info.total_elems += feat.numel()

            _, sparse_ms = run_sparse_linear(feat, info.module)
            _, dense_ms = run_dense_linear(feat, info.module)
            info.total_sparse_ms += sparse_ms
            info.total_dense_ms += dense_ms

        if (batch_idx + 1) % 100 == 0:
            pct = (batch_idx + 1) / len(loader) * 100
            print(f"  [{pct:5.1f}%] batch {batch_idx+1}/{len(loader)}")

    for h in hook_handles:
        h.remove()

    # ---- 输出性能报告 ----
    print(f"\n{'='*90}")
    print(f"{'SparseFlow Benchmark — ' + title:^90}")
    print(f"{'T=' + str(args.T) + '  BS=' + str(args.batch_size) + '  Power=' + str(args.power) + 'W':^90}")
    print(f"{'='*90}")
    print(f"{'Layer':<35} {'In':>6} {'Out':>6} "
          f"{'Sparsity':>9} {'Sparse(ms)':>11} {'Dense(ms)':>10} {'Speedup':>9}")
    print("-" * 90)

    for name, info in layer_infos.items():
        if info.total_elems == 0:
            continue
        sparsity = info.total_zeros / info.total_elems * 100
        in_f = info.module.in_features
        out_f = info.module.out_features

        if info.total_sparse_ms < 1e-6:
            speedup_str = "      inf"
        else:
            speedup = info.total_dense_ms / info.total_sparse_ms
            speedup_str = f"{speedup:>8.2f}x"

        sname = name if len(name) <= 34 else "..." + name[-31:]
        print(f"{sname:<35} {in_f:>6} {out_f:>6} "
              f"{sparsity:>8.2f}% {info.total_sparse_ms:>10.2f} "
              f"{info.total_dense_ms:>10.2f} {speedup_str}")

    print("-" * 90)
    all_s = sum(i.total_sparse_ms for i in layer_infos.values())
    all_d = sum(i.total_dense_ms for i in layer_infos.values())
    if all_s > 1e-6:
        print(f"{'[TOTAL]':<35} {'':>6} {'':>6} {'':>9} "
              f"{all_s:>10.2f} {all_d:>10.2f} {all_d/all_s:>8.2f}x")

    es = (all_s / 1000.0) * args.power
    ed = (all_d / 1000.0) * args.power
    print(f"\n  F.linear Energy : {ed:.4f} J")
    print(f"  Sparse   Energy : {es:.4f} J")
    saving = (1 - es / max(ed, 1e-9)) * 100
    print(f"  Energy Saving   : {saving:.2f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()