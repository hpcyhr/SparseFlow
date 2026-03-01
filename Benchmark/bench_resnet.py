"""
SparseFlow Benchmark — Spiking-ResNet on CIFAR-10/100

比较对象：SparseFlow sparse kernel (带权重) vs cuDNN (F.conv2d)
验证内容：
  1. 逐层算子验证：每个替换的 Conv2d 的 Sparse 输出 vs cuDNN 输出的数值误差
  2. 整网验证：替换后的模型 vs 原始模型在小批量数据上的推理结果一致性

用法：
    cd ~/SparseFlow
    python Benchmark/bench_resnet.py --model resnet34 --dataset cifar10
    python Benchmark/bench_resnet.py --model resnet50 --dataset cifar100 --T 32
    python Benchmark/bench_resnet.py --model resnet34 --dataset cifar10 --gpu 2
"""

import sys
from pathlib import Path

# 项目根目录
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from spikingjelly.activation_based.model import spiking_resnet
from spikingjelly.activation_based import functional as sj_func
from spikingjelly.activation_based import neuron as sj_neuron

from Kernels.conv2d import sparse_conv2d_forward
from Ops.sparse_conv2d import SparseConv2d

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
    if spatial >= 32:
        return 16
    elif spatial >= 16:
        return 8
    elif spatial >= 8:
        return 4
    else:
        return None

# =============================================================================
# 全局 device（在 main 中设定）
# =============================================================================
DEVICE = None

def sync():
    torch.cuda.synchronize(DEVICE)

def make_event():
    return torch.cuda.Event(enable_timing=True)

# =============================================================================
# Benchmark 函数 — 使用真实权重
# =============================================================================

def run_sparse_conv_weighted(feat, conv_module, block):
    """
    SparseFlow 稀疏卷积 benchmark — 使用真实卷积权重。
    调用 sparse_conv2d_forward (prescan + weighted sparse kernel)
    """
    feat = feat.contiguous()
    k = conv_module.kernel_size[0]  # 3 or 1

    y, sparse_ms = sparse_conv2d_forward(
        x=feat,
        weight=conv_module.weight.contiguous(),
        bias=conv_module.bias,
        block_size=block,
        kernel_size=k,
        threshold=1e-6,
    )
    return y, sparse_ms


def run_cudnn_conv(feat, module):
    """cuDNN conv 基准（F.conv2d）"""
    start, end = make_event(), make_event()
    start.record()
    with torch.no_grad():
        y = F.conv2d(feat, module.weight, module.bias,
                     module.stride, module.padding, module.dilation, module.groups)
    end.record()
    sync()
    return y, start.elapsed_time(end)


# =============================================================================
# 逐层数值验证
# =============================================================================

def verify_layer(feat, conv_module, block, layer_name):
    """
    验证单层：Sparse 输出 vs cuDNN 输出的数值差异。

    Returns:
        max_abs_err: 最大绝对误差
        mean_abs_err: 平均绝对误差
        max_rel_err: 最大相对误差 (排除接近零的元素)
        cosine_sim: 余弦相似度
    """
    feat = feat.contiguous()

    # cuDNN 参考
    with torch.no_grad():
        y_cudnn = F.conv2d(feat, conv_module.weight, conv_module.bias,
                           conv_module.stride, conv_module.padding,
                           conv_module.dilation, conv_module.groups)

    # Sparse 计算
    k = conv_module.kernel_size[0]
    y_sparse, _ = sparse_conv2d_forward(
        x=feat,
        weight=conv_module.weight.contiguous(),
        bias=conv_module.bias,
        block_size=block,
        kernel_size=k,
        threshold=1e-6,
    )

    diff = (y_sparse - y_cudnn).float()
    y_ref = y_cudnn.float()

    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()

    # 相对误差（使用 max(|y_ref|, 1.0) 作为分母，避免除以小值导致的巨大相对误差）
    denom = y_ref.abs().clamp(min=1.0)
    max_rel = (diff.abs() / denom).max().item()

    # 余弦相似度（处理全零情况）
    flat_s = y_sparse.flatten().float()
    flat_c = y_cudnn.flatten().float()
    norm_s = flat_s.norm()
    norm_c = flat_c.norm()
    if norm_s < 1e-8 and norm_c < 1e-8:
        # 两者都接近零向量 → 完全一致
        cos = 1.0
    elif norm_s < 1e-8 or norm_c < 1e-8:
        # 一个为零另一个不为零 → 完全不一致
        cos = 0.0
    else:
        cos = F.cosine_similarity(flat_s.unsqueeze(0), flat_c.unsqueeze(0)).item()

    return max_abs, mean_abs, max_rel, cos


# =============================================================================
# 网络分析 (仅 Conv2d)
# =============================================================================

class LayerInfo:
    def __init__(self, name, module, layer_type, block_size=None, H=0, W=0):
        self.name = name
        self.module = module
        self.layer_type = layer_type
        self.block_size = block_size
        self.H, self.W = H, W
        self.total_sparse_ms = 0.0
        self.total_cudnn_ms = 0.0
        self.total_zeros = 0
        self.total_elems = 0
        # 数值验证统计
        self.verify_max_abs = 0.0
        self.verify_mean_abs = 0.0
        self.verify_max_rel = 0.0
        self.verify_cos_sum = 0.0
        self.verify_count = 0


def analyze_network(model, sample_input, device):
    """分析网络，识别脉冲后继的 Conv2d 层"""
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

    # 透明层：搜索时穿透，不阻断
    TRANSPARENT_TYPES = (nn.BatchNorm2d, nn.Dropout, nn.Dropout2d,
                         nn.Identity, nn.Flatten,
                         nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d,
                         nn.ReLU, nn.ReLU6, nn.LeakyReLU)

    for i, (name, module) in enumerate(module_list):
        if not isinstance(module, SPIKE_OUTPUT_OPS):
            continue

        for j in range(i + 1, min(i + 15, len(module_list))):
            next_name, next_module = module_list[j]
            if next_name in visited:
                continue

            if isinstance(next_module, SPIKE_OUTPUT_OPS):
                break

            if isinstance(next_module, TRANSPARENT_TYPES):
                continue

            if isinstance(next_module, nn.Linear):
                continue

            ishape = input_shapes.get(next_name)
            if ishape is None:
                continue

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
                        continue
                    layer_infos[next_name] = LayerInfo(
                        next_name, next_module, "conv2d", block, H, W)
                    visited.add(next_name)
                    print(f"  [CONV  ] LIF={name:<30} -> {next_name:<30} H={H} Block={block}")
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
# 整网推理验证
# =============================================================================

def _set_module_by_name(model, name, new_module):
    """按 dot-separated name 替换 model 中的子模块"""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def verify_end_to_end(model, layer_infos, loader, device, T, num_batches=5):
    """
    整网推理验证：
      1. 深拷贝一份原始模型 (cuDNN baseline)
      2. 在原模型上替换所有目标 Conv2d 为 SparseConv2d
      3. 对比两个模型在相同输入上的输出 logits

    Returns:
        results: list of dict，每个 batch 的验证统计
    """
    print(f"\n{'='*90}")
    print(f"{'整网推理验证 (End-to-End Correctness)':^90}")
    print(f"{'='*90}")

    # 深拷贝原始模型作为 baseline
    model_baseline = copy.deepcopy(model)

    # 在原模型上替换 Conv2d → SparseConv2d
    model_sparse = model  # 原地替换
    replaced = 0
    for name, info in layer_infos.items():
        if info.layer_type == "conv2d" and info.block_size is not None:
            sparse_conv = SparseConv2d.from_dense(
                info.module, block_size=info.block_size)
            _set_module_by_name(model_sparse, name, sparse_conv)
            replaced += 1

    print(f"  替换了 {replaced} 个 Conv2d → SparseConv2d")
    print(f"  验证 {num_batches} 个 batch ...\n")

    results = []
    batch_count = 0

    for imgs, labels in loader:
        if batch_count >= num_batches:
            break

        inp = torch.bernoulli(
            imgs.to(device).unsqueeze(0).repeat(T, 1, 1, 1, 1).clamp(0, 1)
        )

        # Baseline (cuDNN)
        sj_func.reset_net(model_baseline)
        with torch.no_grad():
            logits_baseline = model_baseline(inp)

        # Sparse
        sj_func.reset_net(model_sparse)
        with torch.no_grad():
            logits_sparse = model_sparse(inp)

        # 比较 logits
        diff = (logits_sparse - logits_baseline).float()
        ref = logits_baseline.float()

        max_abs = diff.abs().max().item()
        mean_abs = diff.abs().mean().item()

        # 相对误差
        denom = ref.abs().clamp(min=1.0)
        max_rel = (diff.abs() / denom).max().item()

        # 余弦相似度
        flat_s = logits_sparse.flatten().float()
        flat_c = logits_baseline.flatten().float()
        norm_s = flat_s.norm()
        norm_c = flat_c.norm()
        if norm_s < 1e-8 and norm_c < 1e-8:
            cos = 1.0
        elif norm_s < 1e-8 or norm_c < 1e-8:
            cos = 0.0
        else:
            cos = F.cosine_similarity(flat_s.unsqueeze(0), flat_c.unsqueeze(0)).item()

        # 分类一致性
        pred_base = logits_baseline.argmax(dim=-1)
        pred_sparse = logits_sparse.argmax(dim=-1)
        agree = (pred_base == pred_sparse).float().mean().item()

        r = {
            "batch": batch_count,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "max_rel": max_rel,
            "cosine": cos,
            "pred_agree": agree,
        }
        results.append(r)

        status = "✓ PASS" if max_abs < 0.05 and cos > 0.999 else "✗ FAIL"
        print(f"  Batch {batch_count}: max_abs={max_abs:.6f}  mean_abs={mean_abs:.6f}  "
              f"max_rel={max_rel:.6f}  cos={cos:.8f}  pred_agree={agree*100:.1f}%  {status}")

        batch_count += 1

    # 汇总
    if results:
        avg_max_abs = sum(r["max_abs"] for r in results) / len(results)
        avg_cos = sum(r["cosine"] for r in results) / len(results)
        avg_agree = sum(r["pred_agree"] for r in results) / len(results)
        all_pass = all(r["max_abs"] < 0.05 and r["cosine"] > 0.999 for r in results)

        print(f"\n  {'─'*70}")
        print(f"  汇总: avg_max_abs={avg_max_abs:.6f}  avg_cos={avg_cos:.8f}  "
              f"avg_pred_agree={avg_agree*100:.1f}%")
        print(f"  整网验证结果: {'✓ 全部通过' if all_pass else '✗ 存在误差过大的 batch'}")
    print(f"{'='*90}\n")

    return results


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
    parser.add_argument("--verify_batches", type=int, default=5,
                        help="整网验证使用的 batch 数")
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
    print(f"[1/6] 构建 {title} (on {device}) ...")
    model = build_model(args.model, device, args.v_threshold)

    # ---- 数据集 ----
    print(f"[2/6] 加载 {args.dataset} 测试集 (root={args.data_root}) ...")
    ds = build_dataset(args.dataset, args.data_root)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=8, pin_memory=True)
    print(f"  测试集共 {len(ds)} 张，{len(loader)} 个 batch")

    # ---- 分析网络 ----
    print(f"[3/6] 分析网络，识别脉冲后继 Conv2d ...")
    imgs_s, _ = next(iter(loader))
    sample_input = torch.bernoulli(
        imgs_s[:4].to(device).unsqueeze(0).repeat(args.T, 1, 1, 1, 1).clamp(0, 1)
    )
    layer_infos = analyze_network(model, sample_input, device)

    if not layer_infos:
        print("未找到符合条件的目标层，退出。")
        return
    print(f"\n共找到 {len(layer_infos)} 个目标 Conv2d 层\n")

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
                elif x.dim() == 3:
                    T, B, C = x.shape
                    x = x.reshape(T * B, C)
                captured[name] = x.to(device).contiguous().clone()
        return hook

    hook_handles = []
    for name, info in layer_infos.items():
        h = info.module.register_forward_hook(make_capture_hook(name))
        hook_handles.append(h)

    # ---- 逐层算子验证 ----
    print(f"[4/6] 逐层算子正确性验证 ...")
    # 跑 3 个 batch 做验证
    verify_batches_per_layer = 3
    verify_count = 0
    for imgs, _ in loader:
        if verify_count >= verify_batches_per_layer:
            break
        inp = torch.bernoulli(
            imgs.to(device).unsqueeze(0).repeat(args.T, 1, 1, 1, 1).clamp(0, 1)
        )
        sj_func.reset_net(model)
        with torch.no_grad():
            _ = model(inp)

        for name, info in layer_infos.items():
            feat = captured.get(name)
            if feat is None or info.layer_type != "conv2d":
                continue

            max_abs, mean_abs, max_rel, cos = verify_layer(
                feat, info.module, info.block_size, name)

            info.verify_max_abs = max(info.verify_max_abs, max_abs)
            info.verify_mean_abs += mean_abs
            info.verify_max_rel = max(info.verify_max_rel, max_rel)
            info.verify_cos_sum += cos
            info.verify_count += 1

        verify_count += 1

    # 输出逐层验证报告
    print(f"\n  {'Layer':<35} {'max_abs':>10} {'mean_abs':>10} {'max_rel':>10} {'cosine':>12} {'Status':>8}")
    print(f"  {'-'*87}")
    all_layer_pass = True
    for name, info in layer_infos.items():
        if info.verify_count == 0:
            continue
        avg_mean = info.verify_mean_abs / info.verify_count
        avg_cos = info.verify_cos_sum / info.verify_count
        sname = name if len(name) <= 34 else "..." + name[-31:]

        # 判定标准: max_abs < 0.05 (fp16 精度), cosine > 0.999
        ok = info.verify_max_abs < 0.05 and avg_cos > 0.999
        status = "✓ PASS" if ok else "✗ FAIL"
        if not ok:
            all_layer_pass = False

        print(f"  {sname:<35} {info.verify_max_abs:>10.6f} {avg_mean:>10.6f} "
              f"{info.verify_max_rel:>10.6f} {avg_cos:>12.8f} {status:>8}")

    print(f"\n  逐层验证结果: {'✓ 全部通过' if all_layer_pass else '✗ 存在不通过的层'}\n")

    # ---- 预热 ----
    print(f"[5/6] 性能基准测试 (warmup={args.warmup}) ...")
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

    # ---- 正式性能测试 ----
    for batch_idx, (imgs, _) in enumerate(loader):
        inp = torch.bernoulli(
            imgs.to(device).unsqueeze(0).repeat(args.T, 1, 1, 1, 1).clamp(0, 1)
        )
        sj_func.reset_net(model)
        with torch.no_grad():
            _ = model(inp)

        for name, info in layer_infos.items():
            feat = captured.get(name)
            if feat is None or info.layer_type != "conv2d":
                continue

            block = info.block_size

            # 稀疏度统计
            info.total_zeros += (feat.abs() <= 1e-6).sum().item()
            info.total_elems += feat.numel()

            # 带权重的稀疏卷积 benchmark
            _, sparse_ms = run_sparse_conv_weighted(feat, info.module, block)
            _, cudnn_ms = run_cudnn_conv(feat, info.module)
            info.total_sparse_ms += sparse_ms
            info.total_cudnn_ms += cudnn_ms

        if (batch_idx + 1) % 50 == 0:
            pct = (batch_idx + 1) / len(loader) * 100
            print(f"  [{pct:5.1f}%] batch {batch_idx+1}/{len(loader)}")

    for h in hook_handles:
        h.remove()

    # ---- 输出性能报告 ----
    print(f"\n{'='*90}")
    print(f"{'SparseFlow Benchmark — ' + title:^90}")
    print(f"{'T=' + str(args.T) + '  BS=' + str(args.batch_size) + '  Power=' + str(args.power) + 'W':^90}")
    print(f"{'='*90}")
    print(f"{'Layer':<35} {'Blk':>4} {'H':>4} "
          f"{'Sparsity':>9} {'Sparse(ms)':>11} {'cuDNN(ms)':>10} {'Speedup':>9}")
    print("-" * 90)

    total_sparse_j = 0.0
    total_cudnn_j = 0.0

    for name, info in layer_infos.items():
        if info.total_elems == 0:
            continue
        sparsity = info.total_zeros / info.total_elems * 100

        if info.total_sparse_ms < 1e-6:
            speedup_str = "      inf"
        else:
            speedup = info.total_cudnn_ms / info.total_sparse_ms
            speedup_str = f"{speedup:>8.2f}x"

        es = (info.total_sparse_ms / 1000.0) * args.power
        ec = (info.total_cudnn_ms / 1000.0) * args.power
        total_sparse_j += es
        total_cudnn_j += ec

        sname = name if len(name) <= 34 else "..." + name[-31:]
        blk = str(info.block_size) if info.block_size else "-"
        h = str(info.H) if info.H else "-"
        print(f"{sname:<35} {blk:>4} {h:>4} "
              f"{sparsity:>8.2f}% {info.total_sparse_ms:>10.2f} "
              f"{info.total_cudnn_ms:>10.2f} {speedup_str}")

    print("-" * 90)
    all_s = sum(i.total_sparse_ms for i in layer_infos.values())
    all_c = sum(i.total_cudnn_ms for i in layer_infos.values())
    if all_s > 1e-6:
        print(f"{'[TOTAL]':<35} {'':>4} {'':>4} {'':>9} "
              f"{all_s:>10.2f} {all_c:>10.2f} {all_c/all_s:>8.2f}x")
    print(f"\n  cuDNN  Energy : {total_cudnn_j:.4f} J")
    print(f"  Sparse Energy : {total_sparse_j:.4f} J")
    saving = (1 - total_sparse_j / max(total_cudnn_j, 1e-9)) * 100
    print(f"  Energy Saving : {saving:.2f}%")
    print("=" * 90)

    # ---- 整网推理验证 ----
    print(f"\n[6/6] 整网推理一致性验证 ...")
    # 重建原始模型用于整网验证
    model_fresh = build_model(args.model, device, args.v_threshold)
    layer_infos_fresh = analyze_network(model_fresh, sample_input, device)
    verify_end_to_end(model_fresh, layer_infos_fresh, loader, device,
                      args.T, num_batches=args.verify_batches)


if __name__ == "__main__":
    main()