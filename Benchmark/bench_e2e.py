"""
SparseFlow End-to-End Benchmark — Table 1 数据采集

测量真实 model(x) 端到端推理延迟、能耗估算、分类准确率。
用于论文 Table 1 数据采集。

三种模式：
  1. cuDNN baseline：原始 spikingjelly 模型，所有层走 F.conv2d
  2. SparseFlow：所有目标 Conv2d → SparseConv2d
  3. SparseFlow + Fused：目标 Conv2d+LIF → FusedSparseConvLIF（--fused）

用法：
    cd ~/SparseFlow

    # 非 Fused（Conv2d → SparseConv2d）
    python Benchmark/bench_e2e.py --model resnet18 --dataset cifar10 --T 4
    python Benchmark/bench_e2e.py --model resnet50 --dataset cifar100 --T 16

    # Fused（Conv2d+LIF → FusedSparseConvLIF）
    python Benchmark/bench_e2e.py --model resnet34 --dataset cifar10 --T 8 --fused

    # 指定 GPU
    python Benchmark/bench_e2e.py --model resnet50 --dataset cifar100 --T 16 --gpu 2

输出：
    - 逐层稀疏率统计
    - cuDNN / SparseFlow 端到端延迟 (ms/batch)
    - 加速比
    - 能耗估算 (J)
    - 整网数值一致性（cosine similarity, 分类一致率）
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import copy
import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from spikingjelly.activation_based.model import spiking_resnet
from spikingjelly.activation_based import functional as sj_func
from spikingjelly.activation_based import neuron as sj_neuron

from Ops.sparse_conv2d import SparseConv2d
from Ops.sparse_fused_conv_lif import FusedSparseConvLIF


# =============================================================================
# 全局
# =============================================================================
DEVICE = None
SPIKE_OPS = (sj_neuron.LIFNode, sj_neuron.IFNode, sj_neuron.ParametricLIFNode)

TRANSPARENT_TYPES = (
    nn.BatchNorm2d, nn.BatchNorm1d,
    nn.Dropout, nn.Dropout2d,
    nn.Identity, nn.Flatten,
    nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d,
    nn.ReLU, nn.ReLU6, nn.LeakyReLU,
)


def sync():
    torch.cuda.synchronize(DEVICE)


def make_event():
    return torch.cuda.Event(enable_timing=True)


# =============================================================================
# 模型构建
# =============================================================================

MODEL_BUILDERS = {
    "resnet18": spiking_resnet.spiking_resnet18,
    "resnet34": spiking_resnet.spiking_resnet34,
    "resnet50": spiking_resnet.spiking_resnet50,
    "resnet101": spiking_resnet.spiking_resnet101,
    "resnet152": spiking_resnet.spiking_resnet152,
}


def build_model(model_name, device, v_threshold=1.0):
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
        ds = datasets.CIFAR10(root=data_root, train=False,
                              download=True, transform=transform)
    elif dataset_name == "cifar100":
        ds = datasets.CIFAR100(root=data_root, train=False,
                               download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return ds


# =============================================================================
# 工具函数
# =============================================================================

def _set_module_by_name(model, name, new_module):
    """按 dot-separated name 替换子模块"""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def make_spike_input(imgs, T, device):
    """将图像转为脉冲序列 [T, B, C, H, W]"""
    return torch.bernoulli(
        imgs.to(device).unsqueeze(0).repeat(T, 1, 1, 1, 1).clamp(0, 1)
    )


# =============================================================================
# 网络分析 — 找到所有 Spike → Conv2d 目标
# =============================================================================

def analyze_targets(model, sample_input, device, fused=False):
    """
    分析网络拓扑，返回替换目标列表。

    fused=False: 返回 conv_targets (Conv2d → SparseConv2d)
    fused=True:  返回 fused_targets (Conv2d+LIF → FusedSparseConvLIF)

    Returns:
        targets: list of dict, 每个 dict 包含:
            name: Conv2d 的 module name
            module: Conv2d 实例
            kernel_size: (3,3) 或 (1,1)
            H, W: 输入特征图尺寸
            lif_name: (fused only) 下游 LIFNode 的 name
            lif_module: (fused only) LIFNode 实例
    """
    # 1. 用 hook 收集每层输入 shape
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

    # 2. 从 Spike 节点出发，向后搜索目标
    targets = []
    visited = set()

    for i, (name, module) in enumerate(module_list):
        if not isinstance(module, SPIKE_OPS):
            continue

        for j in range(i + 1, min(i + 15, len(module_list))):
            next_name, next_module = module_list[j]
            if next_name in visited:
                continue
            if isinstance(next_module, SPIKE_OPS):
                break  # 遇到下一个 spike 源，停止
            if isinstance(next_module, TRANSPARENT_TYPES):
                continue
            if not isinstance(next_module, nn.Conv2d):
                continue

            # 检查是否是支持的 Conv2d
            k = next_module.kernel_size
            s = next_module.stride
            p = next_module.padding
            g = next_module.groups

            supported = False
            if k == (3, 3) and s == (1, 1) and p == (1, 1) and g == 1:
                supported = True
            elif k == (1, 1) and s == (1, 1) and p == (0, 0) and g == 1:
                supported = True
            if not supported:
                visited.add(next_name)
                continue

            # 获取 feature map 尺寸
            ishape = input_shapes.get(next_name)
            if ishape is None:
                continue
            if len(ishape) == 5:
                H, W = ishape[3], ishape[4]
            elif len(ishape) == 4:
                H, W = ishape[2], ishape[3]
            else:
                continue
            if min(H, W) < 7:
                visited.add(next_name)
                continue

            target = {
                "name": next_name,
                "module": next_module,
                "kernel_size": k,
                "H": H, "W": W,
            }

            # Fused 模式：检查 Conv 下游是否有 LIF
            if fused:
                lif_name, lif_module = None, None
                for jj in range(j + 1, min(j + 6, len(module_list))):
                    jj_name, jj_mod = module_list[jj]
                    if isinstance(jj_mod, (nn.BatchNorm2d, nn.Identity)):
                        continue
                    if isinstance(jj_mod, SPIKE_OPS):
                        lif_name = jj_name
                        lif_module = jj_mod
                    break  # 不是 BN/Identity/LIF → 非 fusion 模式
                target["lif_name"] = lif_name
                target["lif_module"] = lif_module

            targets.append(target)
            visited.add(next_name)

    return targets


# =============================================================================
# 模型替换
# =============================================================================

def replace_model(model, targets, fused=False):
    """
    在 model 上执行算子替换。

    fused=False: Conv2d → SparseConv2d
    fused=True:  Conv2d+LIF → FusedSparseConvLIF, LIF → Identity
    """
    replaced = 0
    fused_count = 0
    sparse_count = 0

    for target in targets:
        conv_name = target["name"]
        conv_module = target["module"]

        if fused and target.get("lif_name") is not None:
            # Fused 替换
            fused_module = FusedSparseConvLIF.from_conv_and_lif(
                conv_module, target["lif_module"])
            _set_module_by_name(model, conv_name, fused_module)
            _set_module_by_name(model, target["lif_name"], nn.Identity())
            fused_count += 1
            replaced += 1
        else:
            # 标准 SparseConv2d 替换
            sparse_conv = SparseConv2d.from_dense(conv_module)
            _set_module_by_name(model, conv_name, sparse_conv)
            sparse_count += 1
            replaced += 1

    return replaced, sparse_count, fused_count


# =============================================================================
# 稀疏率统计
# =============================================================================

def measure_sparsity(model, targets, loader, device, T, num_batches=10):
    """
    对目标层的输入特征图统计稀疏率。
    使用 forward hook 捕获每个目标 Conv2d 的输入。
    """
    sparsity_data = {t["name"]: {"zeros": 0, "total": 0} for t in targets}
    captured = {}

    def make_hook(name):
        def hook(m, inp, out):
            x = inp[0]
            if isinstance(x, torch.Tensor):
                x = x.detach()
                if x.dim() == 5:
                    T_, B, C, H, W = x.shape
                    x = x.reshape(T_ * B, C, H, W)
                captured[name] = x
        return hook

    hook_handles = []
    for target in targets:
        m = target["module"]
        hook_handles.append(m.register_forward_hook(make_hook(target["name"])))

    batch_count = 0
    for imgs, _ in loader:
        if batch_count >= num_batches:
            break
        inp = make_spike_input(imgs, T, device)
        sj_func.reset_net(model)
        with torch.no_grad():
            _ = model(inp)

        for name, feat in captured.items():
            if name in sparsity_data:
                sparsity_data[name]["zeros"] += (feat.abs() <= 1e-6).sum().item()
                sparsity_data[name]["total"] += feat.numel()
        captured.clear()
        batch_count += 1

    for h in hook_handles:
        h.remove()

    return sparsity_data


# =============================================================================
# 端到端延迟测量
# =============================================================================

def measure_e2e_latency(model, loader, device, T, warmup=10, max_batches=None):
    """
    测量真实 model(x) 的端到端推理延迟。

    Returns:
        avg_ms: 平均每 batch 延迟 (ms)
        total_ms: 总延迟 (ms)
        num_batches: 测量的 batch 数
    """
    # Warmup
    warmup_count = 0
    for imgs, _ in loader:
        if warmup_count >= warmup:
            break
        inp = make_spike_input(imgs, T, device)
        sj_func.reset_net(model)
        with torch.no_grad():
            _ = model(inp)
        warmup_count += 1

    sync()

    # 正式测量
    total_ms = 0.0
    num_batches = 0
    for imgs, _ in loader:
        if max_batches is not None and num_batches >= max_batches:
            break

        inp = make_spike_input(imgs, T, device)

        sj_func.reset_net(model)

        start_evt = make_event()
        end_evt = make_event()
        start_evt.record()

        with torch.no_grad():
            _ = model(inp)

        end_evt.record()
        sync()
        total_ms += start_evt.elapsed_time(end_evt)
        num_batches += 1

    avg_ms = total_ms / max(num_batches, 1)
    return avg_ms, total_ms, num_batches


# =============================================================================
# 数值一致性验证
# =============================================================================

def verify_consistency(model_baseline, model_sparse, loader, device, T,
                       num_batches=5):
    """
    对比 baseline 和 sparse 模型在相同输入上的输出。

    Returns:
        avg_cosine: 平均余弦相似度
        avg_pred_agree: 平均分类一致率
        max_abs_err: 最大绝对误差
    """
    cosine_sum = 0.0
    agree_sum = 0.0
    global_max_abs = 0.0
    count = 0

    for imgs, _ in loader:
        if count >= num_batches:
            break

        inp = make_spike_input(imgs, T, device)

        # Baseline
        sj_func.reset_net(model_baseline)
        with torch.no_grad():
            out_base = model_baseline(inp)

        # Sparse
        sj_func.reset_net(model_sparse)
        with torch.no_grad():
            out_sparse = model_sparse(inp)

        # 按时间步平均（如果是 3D）
        if out_base.dim() == 3:
            out_base = out_base.mean(dim=0)
        if out_sparse.dim() == 3:
            out_sparse = out_sparse.mean(dim=0)

        diff = (out_sparse - out_base).float()
        max_abs = diff.abs().max().item()
        global_max_abs = max(global_max_abs, max_abs)

        flat_s = out_sparse.flatten().float()
        flat_b = out_base.flatten().float()
        if flat_s.norm() < 1e-8 and flat_b.norm() < 1e-8:
            cos = 1.0
        elif flat_s.norm() < 1e-8 or flat_b.norm() < 1e-8:
            cos = 0.0
        else:
            cos = F.cosine_similarity(
                flat_s.unsqueeze(0), flat_b.unsqueeze(0)).item()
        cosine_sum += cos

        pred_base = out_base.argmax(dim=-1)
        pred_sparse = out_sparse.argmax(dim=-1)
        agree = (pred_base == pred_sparse).float().mean().item()
        agree_sum += agree

        count += 1

    n = max(count, 1)
    return cosine_sum / n, agree_sum / n, global_max_abs


# =============================================================================
# 主流程
# =============================================================================

def main():
    global DEVICE

    parser = argparse.ArgumentParser(
        description="SparseFlow E2E Benchmark — Table 1 数据采集")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=list(MODEL_BUILDERS.keys()))
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100"])
    parser.add_argument("--T", type=int, default=4,
                        help="SNN 时间步数")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fused", action="store_true",
                        help="启用 FusedSparseConvLIF（Conv+LIF 融合）")
    parser.add_argument("--v_threshold", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=2.0,
                        help="LIF 膜时间常数")
    parser.add_argument("--power", type=float, default=250.0,
                        help="GPU TDP (W), 用于能耗估算")
    parser.add_argument("--warmup", type=int, default=10,
                        help="预热 batch 数")
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU ID, -1=自动选择空闲最大的")
    parser.add_argument("--verify_batches", type=int, default=10,
                        help="数值一致性验证 batch 数")
    parser.add_argument("--sparsity_batches", type=int, default=20,
                        help="稀疏率统计 batch 数")
    parser.add_argument("--save_json", type=str, default="",
                        help="结果保存为 JSON 文件（留空则不保存）")
    args = parser.parse_args()

    mode_str = "Fused" if args.fused else "SparseConv"

    # ─── GPU 选择 ───
    if args.gpu >= 0:
        gpu_id = args.gpu
    else:
        gpu_id = 0
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            max_free = 0
            for i in range(num_gpus):
                free, total = torch.cuda.mem_get_info(i)
                if free > max_free:
                    max_free = free
                    gpu_id = i

    DEVICE = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(DEVICE)
    device = DEVICE

    title = (f"Spiking-{args.model.upper()} | {args.dataset.upper()} | "
             f"T={args.T} | {mode_str}")

    print(f"\n{'='*80}")
    print(f"{'SparseFlow E2E Benchmark':^80}")
    print(f"{title:^80}")
    print(f"{'='*80}")
    print(f"  GPU:          {gpu_id} ({torch.cuda.get_device_name(gpu_id)})")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Mode:         {mode_str}")
    print(f"  Power (TDP):  {args.power} W")
    print()

    # ─── 1. 构建模型 ───
    print(f"[1/6] 构建 Spiking-{args.model} ...")
    model_baseline = build_model(args.model, device, args.v_threshold)

    # ─── 2. 加载数据集 ───
    print(f"[2/6] 加载 {args.dataset} 测试集 ...")
    ds = build_dataset(args.dataset, args.data_root)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    num_classes = 10 if args.dataset == "cifar10" else 100
    print(f"  测试集: {len(ds)} 张, {len(loader)} batches, "
          f"{num_classes} classes")

    # ─── 3. 分析网络 & 构建 sparse 模型 ───
    print(f"[3/6] 分析网络拓扑，识别替换目标 ...")
    imgs_s, _ = next(iter(loader))
    sample_input = make_spike_input(imgs_s[:4], args.T, device)
    targets = analyze_targets(model_baseline, sample_input, device,
                              fused=args.fused)

    if not targets:
        print("  未找到可替换的目标层，退出。")
        return

    target_summary = []
    for t in targets:
        k_str = f"{t['kernel_size'][0]}x{t['kernel_size'][1]}"
        fuse_str = ""
        if args.fused and t.get("lif_name"):
            fuse_str = f" → FUSE(LIF={t['lif_name']})"
        target_summary.append(
            f"  {t['name']:<40} k={k_str} H={t['H']}{fuse_str}")
    print(f"  共找到 {len(targets)} 个目标层:")
    for s in target_summary:
        print(s)

    # 深拷贝 baseline → sparse
    model_sparse = copy.deepcopy(model_baseline)
    replaced, sparse_n, fused_n = replace_model(
        model_sparse, targets, fused=args.fused)
    print(f"\n  替换完成: {sparse_n} SparseConv2d + {fused_n} FusedConvLIF "
          f"= {replaced} total")

    # ─── 4. 稀疏率统计 ───
    print(f"\n[4/6] 统计输入特征图稀疏率 "
          f"({args.sparsity_batches} batches) ...")
    sparsity_data = measure_sparsity(
        model_baseline, targets, loader, device, args.T,
        num_batches=args.sparsity_batches)

    print(f"\n  {'Layer':<40} {'Sparsity':>10}")
    print(f"  {'-'*52}")
    sparsity_values = []
    for t in targets:
        name = t["name"]
        sd = sparsity_data[name]
        if sd["total"] > 0:
            sp = sd["zeros"] / sd["total"] * 100
        else:
            sp = 0.0
        sparsity_values.append(sp)
        short_name = name if len(name) <= 39 else "..." + name[-36:]
        print(f"  {short_name:<40} {sp:>9.2f}%")
    avg_sparsity = sum(sparsity_values) / max(len(sparsity_values), 1)
    print(f"  {'[平均]':<40} {avg_sparsity:>9.2f}%")

    # ─── 5. 端到端延迟测量 ───
    print(f"\n[5/6] 端到端延迟测量 (warmup={args.warmup}) ...")

    # cuDNN baseline
    print(f"  测量 cuDNN baseline ...")
    cudnn_avg, cudnn_total, cudnn_n = measure_e2e_latency(
        model_baseline, loader, device, args.T, warmup=args.warmup)
    print(f"    cuDNN:      {cudnn_avg:.2f} ms/batch  "
          f"(total={cudnn_total:.1f} ms, {cudnn_n} batches)")

    # SparseFlow
    print(f"  测量 {mode_str} ...")
    sparse_avg, sparse_total, sparse_n_batches = measure_e2e_latency(
        model_sparse, loader, device, args.T, warmup=args.warmup)
    print(f"    {mode_str}: {sparse_avg:.2f} ms/batch  "
          f"(total={sparse_total:.1f} ms, {sparse_n_batches} batches)")

    # 加速比
    speedup = cudnn_avg / sparse_avg if sparse_avg > 1e-6 else float("inf")
    print(f"\n  Speedup ({mode_str} vs cuDNN): {speedup:.3f}x")

    # 能耗
    cudnn_energy = (cudnn_total / 1000.0) * args.power
    sparse_energy = (sparse_total / 1000.0) * args.power
    energy_saving = (1 - sparse_energy / max(cudnn_energy, 1e-9)) * 100
    print(f"  Energy cuDNN:      {cudnn_energy:.4f} J")
    print(f"  Energy {mode_str}: {sparse_energy:.4f} J")
    print(f"  Energy saving:     {energy_saving:.2f}%")

    # ─── 6. 数值一致性 ───
    print(f"\n[6/6] 数值一致性验证 ({args.verify_batches} batches) ...")
    avg_cos, avg_agree, max_abs = verify_consistency(
        model_baseline, model_sparse, loader, device, args.T,
        num_batches=args.verify_batches)
    print(f"  Cosine similarity:  {avg_cos:.8f}")
    print(f"  Pred agreement:     {avg_agree*100:.2f}%")
    print(f"  Max absolute error: {max_abs:.6f}")

    consistency_ok = avg_cos > 0.999 and max_abs < 0.1
    print(f"  Consistency:        {'PASS' if consistency_ok else 'FAIL'}")

    # ─── 汇总 ───
    print(f"\n{'='*80}")
    print(f"{'SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"  Model:          Spiking-{args.model}")
    print(f"  Dataset:        {args.dataset}")
    print(f"  T:              {args.T}")
    print(f"  Mode:           {mode_str}")
    print(f"  Avg Sparsity:   {avg_sparsity:.2f}%")
    print(f"  cuDNN:          {cudnn_avg:.2f} ms/batch")
    print(f"  {mode_str + ':':<16}{sparse_avg:.2f} ms/batch")
    print(f"  Speedup:        {speedup:.3f}x")
    print(f"  Energy saving:  {energy_saving:.2f}%")
    print(f"  Cosine sim:     {avg_cos:.8f}")
    print(f"  Pred agreement: {avg_agree*100:.2f}%")
    print(f"  Consistency:    {'PASS' if consistency_ok else 'FAIL'}")
    print(f"{'='*80}\n")

    # ─── JSON 输出 ───
    results = {
        "model": args.model,
        "dataset": args.dataset,
        "T": args.T,
        "fused": args.fused,
        "mode": mode_str,
        "batch_size": args.batch_size,
        "gpu": torch.cuda.get_device_name(gpu_id),
        "avg_sparsity_pct": round(avg_sparsity, 2),
        "cudnn_ms_per_batch": round(cudnn_avg, 2),
        "sparse_ms_per_batch": round(sparse_avg, 2),
        "speedup": round(speedup, 4),
        "cudnn_energy_j": round(cudnn_energy, 4),
        "sparse_energy_j": round(sparse_energy, 4),
        "energy_saving_pct": round(energy_saving, 2),
        "cosine_sim": round(avg_cos, 8),
        "pred_agreement_pct": round(avg_agree * 100, 2),
        "max_abs_err": round(max_abs, 6),
        "consistency": "PASS" if consistency_ok else "FAIL",
        "num_targets": len(targets),
        "num_replaced": replaced,
        "num_sparse_conv": sparse_n,
        "num_fused": fused_n,
    }

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  结果已保存到: {args.save_json}")
    else:
        # 默认保存
        fname = (f"results_{args.model}_{args.dataset}_T{args.T}"
                 f"{'_fused' if args.fused else ''}.json")
        with open(fname, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  结果已保存到: {fname}")


if __name__ == "__main__":
    main()