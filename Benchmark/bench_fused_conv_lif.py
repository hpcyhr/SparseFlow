"""
SparseFlow Benchmark — Fused Sparse Conv + LIF on Spiking-ResNet

比较对象：
  - Fused:    SparseFlow fused_sparse_conv_lif_forward (Conv+LIF 一次 kernel 完成)
  - Separate: SparseFlow sparse_conv2d_forward + Python LIF (两次 kernel + 中间 VRAM)

验证内容：
  1. 数值正确性：Fused 输出 (spike, V_next) vs Separate 输出的误差
  2. 性能测试：Fused vs Separate 的延迟对比

用法：
    cd ~/SparseFlow
    python Benchmark/bench_fused_conv_lif.py --model resnet34 --dataset cifar10
    python Benchmark/bench_fused_conv_lif.py --model resnet50 --dataset cifar100 --T 32
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from spikingjelly.activation_based.model import spiking_resnet
from spikingjelly.activation_based import functional as sj_func
from spikingjelly.activation_based import neuron as sj_neuron

from Kernels.conv2d import sparse_conv2d_forward
from Kernels.fused_conv_lif import fused_sparse_conv_lif_forward

# =============================================================================
# 注册表
# =============================================================================
SPIKE_OUTPUT_OPS = (
    sj_neuron.LIFNode,
    sj_neuron.IFNode,
    sj_neuron.ParametricLIFNode,
)

DEVICE = None
DEFAULT_TAU = 2.0
DEFAULT_V_TH = 1.0

def sync():
    torch.cuda.synchronize(DEVICE)

def make_event():
    return torch.cuda.Event(enable_timing=True)

# =============================================================================
# Benchmark 函数
# =============================================================================

def run_fused(feat, conv_module, v_prev, decay, v_th):
    """Fused sparse Conv + LIF：一次 kernel 完成"""
    feat = feat.contiguous()
    k = conv_module.kernel_size[0]

    spike, v_next, fused_ms = fused_sparse_conv_lif_forward(
        x=feat,
        weight=conv_module.weight.contiguous(),
        bias=conv_module.bias,
        v_prev=v_prev,
        kernel_size=k,
        decay=decay,
        v_threshold=v_th,
        threshold=1e-6,
        return_ms=True,
    )
    return spike, v_next, fused_ms


def run_separate(feat, conv_module, v_prev, decay, v_th):
    """Separate: sparse Conv → VRAM write → Python LIF"""
    feat = feat.contiguous()
    k = conv_module.kernel_size[0]

    start, end = make_event(), make_event()
    start.record()

    # Step 1: sparse conv
    y_conv, _ = sparse_conv2d_forward(
        x=feat,
        weight=conv_module.weight.contiguous(),
        bias=conv_module.bias,
        kernel_size=k,
        threshold=1e-6,
        return_ms=False,
    )

    # Step 2: LIF dynamics (Python, on VRAM)
    v_temp = v_prev * decay + y_conv.float()
    spike = (v_temp > v_th).float()
    v_next = v_temp * (1.0 - spike)

    end.record()
    sync()
    separate_ms = start.elapsed_time(end)

    return spike, v_next, separate_ms


def run_cudnn_separate(feat, conv_module, v_prev, decay, v_th):
    """Baseline: cuDNN F.conv2d → Python LIF"""
    start, end = make_event(), make_event()
    start.record()

    with torch.no_grad():
        y_conv = F.conv2d(feat, conv_module.weight, conv_module.bias,
                          conv_module.stride, conv_module.padding,
                          conv_module.dilation, conv_module.groups)

    v_temp = v_prev * decay + y_conv.float()
    spike = (v_temp > v_th).float()
    v_next = v_temp * (1.0 - spike)

    end.record()
    sync()
    return spike, v_next, start.elapsed_time(end)


# =============================================================================
# 数值验证
# =============================================================================

def verify_fused(feat, conv_module, v_prev, decay, v_th, layer_name):
    """验证 Fused 输出 vs Separate 输出的数值差异"""
    # Separate baseline
    k = conv_module.kernel_size[0]
    with torch.no_grad():
        y_conv = F.conv2d(feat, conv_module.weight, conv_module.bias,
                          conv_module.stride, conv_module.padding,
                          conv_module.dilation, conv_module.groups)
    v_temp_ref = v_prev * decay + y_conv.float()
    spike_ref = (v_temp_ref > v_th).float()
    v_next_ref = v_temp_ref * (1.0 - spike_ref)

    # Fused
    spike_fused, v_next_fused, _ = fused_sparse_conv_lif_forward(
        x=feat.contiguous(),
        weight=conv_module.weight.contiguous(),
        bias=conv_module.bias,
        v_prev=v_prev,
        kernel_size=k,
        decay=decay,
        v_threshold=v_th,
        threshold=1e-6,
    )

    # Compare spikes
    spike_agree = (spike_fused == spike_ref).float().mean().item()

    # Compare V_next
    diff_v = (v_next_fused - v_next_ref).float()
    max_abs_v = diff_v.abs().max().item()
    mean_abs_v = diff_v.abs().mean().item()

    flat_f = v_next_fused.flatten().float()
    flat_r = v_next_ref.flatten().float()
    if flat_f.norm() < 1e-8 and flat_r.norm() < 1e-8:
        cos = 1.0
    elif flat_f.norm() < 1e-8 or flat_r.norm() < 1e-8:
        cos = 0.0
    else:
        cos = F.cosine_similarity(flat_f.unsqueeze(0), flat_r.unsqueeze(0)).item()

    return spike_agree, max_abs_v, mean_abs_v, cos


# =============================================================================
# 网络分析：识别 LIF 前驱的 Conv2d（fusion 候选）
# =============================================================================

class LayerInfo:
    def __init__(self, name, module, lif_name, lif_module, H=0, W=0):
        self.name = name
        self.module = module
        self.lif_name = lif_name
        self.lif_module = lif_module
        self.H, self.W = H, W
        self.total_fused_ms = 0.0
        self.total_separate_ms = 0.0
        self.total_cudnn_ms = 0.0
        self.total_zeros = 0
        self.total_elems = 0
        # 验证统计
        self.verify_spike_agree = 0.0
        self.verify_max_abs = 0.0
        self.verify_cos_sum = 0.0
        self.verify_count = 0


def analyze_network(model, sample_input, device):
    """
    分析网络，找出 Conv2d → [optional BN] → LIFNode 的 fusion 模式。
    只匹配 3×3 stride=1 pad=1 和 1×1 stride=1 pad=0 的 Conv2d。
    """
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
        if not isinstance(module, nn.Conv2d):
            continue
        if name in visited:
            continue

        # 检查 Conv2d 是否是我们支持的类型
        k = module.kernel_size
        s = module.stride
        p = module.padding
        g = module.groups
        supported = False
        if k == (3, 3) and s == (1, 1) and p == (1, 1) and g == 1:
            supported = True
        elif k == (1, 1) and s == (1, 1) and p == (0, 0) and g == 1:
            supported = True
        if not supported:
            continue

        # 获取 feature map 尺寸
        ishape = input_shapes.get(name)
        if ishape is None:
            continue
        if len(ishape) == 5:
            H, W = ishape[3], ishape[4]
        elif len(ishape) == 4:
            H, W = ishape[2], ishape[3]
        else:
            continue
        if min(H, W) < 7:
            continue

        # Look-ahead: Conv → [optional BN] → LIFNode?
        lif_name, lif_module = None, None
        for j in range(i + 1, min(i + 6, len(module_list))):
            jname, jmod = module_list[j]
            if isinstance(jmod, nn.BatchNorm2d):
                continue
            if isinstance(jmod, nn.Identity):
                continue
            if isinstance(jmod, SPIKE_OUTPUT_OPS):
                lif_name = jname
                lif_module = jmod
                break
            break  # 不是 BN/Identity/LIF → 不是 fusion 模式

        if lif_name is not None:
            layer_infos[name] = LayerInfo(name, module, lif_name, lif_module, H, W)
            visited.add(name)
            k_str = f"{k[0]}x{k[1]}"
            print(f"  [FUSION] Conv={name:<30} -> LIF={lif_name:<25} "
                  f"k={k_str} H={H}")

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

    parser = argparse.ArgumentParser(
        description="SparseFlow Benchmark — Fused Sparse Conv + LIF")
    parser.add_argument("--model", type=str, default="resnet34",
                        choices=list(MODEL_BUILDERS.keys()))
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100"])
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--v_threshold", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=2.0, help="LIF time constant")
    parser.add_argument("--power", type=float, default=250.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--gpu", type=int, default=-1)
    args = parser.parse_args()

    decay = math.exp(-1.0 / args.tau)
    v_th = args.v_threshold

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
    title = f"Fused Conv+LIF — Spiking-{args.model.upper()} on {args.dataset.upper()}"

    # ---- 构建模型 ----
    print(f"\n[1/5] 构建 {title} (on {device}) ...")
    print(f"  LIF params: tau={args.tau}, decay={decay:.4f}, v_th={v_th}")
    model = build_model(args.model, device, args.v_threshold)

    # ---- 数据集 ----
    print(f"[2/5] 加载 {args.dataset} 测试集 ...")
    ds = build_dataset(args.dataset, args.data_root)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    print(f"  测试集共 {len(ds)} 张，{len(loader)} 个 batch")

    # ---- 分析网络 ----
    print(f"[3/5] 分析网络，识别 Conv → LIF fusion 模式 ...")
    imgs_s, _ = next(iter(loader))
    sample_input = torch.bernoulli(
        imgs_s[:4].to(device).unsqueeze(0).repeat(args.T, 1, 1, 1, 1).clamp(0, 1)
    )
    layer_infos = analyze_network(model, sample_input, device)

    if not layer_infos:
        print("  未找到 Conv → LIF fusion 模式，退出。")
        return
    print(f"\n  共找到 {len(layer_infos)} 个 fusion 候选层\n")

    # ---- Hook 捕获 Conv 输入特征 ----
    captured = {}

    def make_capture_hook(name):
        def hook(m, inp, out):
            x = inp[0]
            if isinstance(x, torch.Tensor):
                x = x.detach()
                if x.dim() == 5:
                    T, B, C, H, W = x.shape
                    x = x.reshape(T * B, C, H, W)
                captured[name] = x.to(device).contiguous().clone()
        return hook

    hook_handles = []
    for name, info in layer_infos.items():
        h = info.module.register_forward_hook(make_capture_hook(name))
        hook_handles.append(h)

    # ---- 正确性验证 ----
    print(f"[4/5] Fused vs Separate 正确性验证 ...")
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

            N, C_OUT, H, W = feat.shape[0], info.module.out_channels, info.H, info.W
            v_prev = torch.zeros(N, C_OUT, H, W, dtype=torch.float32, device=device)

            spike_agree, max_abs, mean_abs, cos = verify_fused(
                feat, info.module, v_prev, decay, v_th, name)

            info.verify_spike_agree += spike_agree
            info.verify_max_abs = max(info.verify_max_abs, max_abs)
            info.verify_cos_sum += cos
            info.verify_count += 1

        verify_count += 1

    print(f"\n  {'Layer':<35} {'spike_agree':>12} {'V_max_abs':>10} {'V_cosine':>10} {'Status':>8}")
    print(f"  {'-'*77}")
    for name, info in layer_infos.items():
        if info.verify_count == 0:
            continue
        avg_agree = info.verify_spike_agree / info.verify_count
        avg_cos = info.verify_cos_sum / info.verify_count
        ok = avg_agree > 0.99 and avg_cos > 0.999
        status = "✓ PASS" if ok else "✗ FAIL"
        sname = name if len(name) <= 34 else "..." + name[-31:]
        print(f"  {sname:<35} {avg_agree*100:>11.2f}% {info.verify_max_abs:>10.6f} "
              f"{avg_cos:>10.6f} {status:>8}")

    # ---- 预热 + 性能测试 ----
    print(f"\n[5/5] 性能基准测试 (warmup={args.warmup}) ...")

    # 预热 Stage A: 模型 forward（填充 captured）
    for i, (imgs, _) in enumerate(loader):
        if i >= args.warmup:
            break
        inp = torch.bernoulli(
            imgs.to(device).unsqueeze(0).repeat(args.T, 1, 1, 1, 1).clamp(0, 1)
        )
        sj_func.reset_net(model)
        with torch.no_grad():
            _ = model(inp)
    print(f"  模型预热 {args.warmup} batch 完成")

    # 预热 Stage B: 触发 Triton kernel 编译（避免 perf loop 中首次编译导致 segfault）
    print(f"  Triton kernel 编译预热 ...")
    for name, info in layer_infos.items():
        feat = captured.get(name)
        if feat is None:
            continue
        N, C_OUT = feat.shape[0], info.module.out_channels
        H, W = info.H, info.W
        v_prev_warmup = torch.zeros(N, C_OUT, H, W, dtype=torch.float32, device=device)
        # 触发 fused kernel autotune
        run_fused(feat, info.module, v_prev_warmup, decay, v_th)
        # 触发 separate sparse kernel autotune
        run_separate(feat, info.module, v_prev_warmup, decay, v_th)
        del v_prev_warmup
    torch.cuda.empty_cache()
    print(f"  Triton kernel 编译预热完成")

    # 预分配 v_prev buffer（复用，避免每次 torch.zeros）
    v_prev_cache = {}
    for name, info in layer_infos.items():
        feat = captured.get(name)
        if feat is None:
            continue
        N, C_OUT = feat.shape[0], info.module.out_channels
        H, W = info.H, info.W
        v_prev_cache[name] = torch.zeros(N, C_OUT, H, W, dtype=torch.float32, device=device)

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

            N_actual = feat.shape[0]
            v_prev = v_prev_cache[name][:N_actual]
            v_prev.zero_()

            _, _, fused_ms = run_fused(feat, info.module, v_prev, decay, v_th)
            _, _, separate_ms = run_separate(feat, info.module, v_prev, decay, v_th)
            _, _, cudnn_ms = run_cudnn_separate(feat, info.module, v_prev, decay, v_th)
            info.total_fused_ms += fused_ms
            info.total_separate_ms += separate_ms
            info.total_cudnn_ms += cudnn_ms

        if (batch_idx + 1) % 50 == 0:
            pct = (batch_idx + 1) / len(loader) * 100
            print(f"  [{pct:5.1f}%] batch {batch_idx+1}/{len(loader)}")

    for h in hook_handles:
        h.remove()

    # ---- 输出性能报告 ----
    print(f"\n{'='*100}")
    print(f"{'SparseFlow Benchmark — ' + title:^100}")
    print(f"{'T=' + str(args.T) + '  BS=' + str(args.batch_size) + '  tau=' + str(args.tau):^100}")
    print(f"{'='*100}")
    print(f"{'Layer':<30} {'H':>4} {'Sparsity':>9} "
          f"{'Fused(ms)':>10} {'Sparse(ms)':>11} {'cuDNN(ms)':>10} "
          f"{'vs Sparse':>10} {'vs cuDNN':>9}")
    print("-" * 100)

    for name, info in layer_infos.items():
        if info.total_elems == 0:
            continue
        sparsity = info.total_zeros / info.total_elems * 100

        vs_sparse = (info.total_separate_ms / info.total_fused_ms
                     if info.total_fused_ms > 1e-6 else float('inf'))
        vs_cudnn = (info.total_cudnn_ms / info.total_fused_ms
                    if info.total_fused_ms > 1e-6 else float('inf'))

        sname = name if len(name) <= 29 else "..." + name[-26:]
        print(f"{sname:<30} {info.H:>4} {sparsity:>8.2f}% "
              f"{info.total_fused_ms:>10.2f} {info.total_separate_ms:>10.2f} "
              f"{info.total_cudnn_ms:>10.2f} "
              f"{vs_sparse:>9.2f}x {vs_cudnn:>8.2f}x")

    print("-" * 100)
    all_f = sum(i.total_fused_ms for i in layer_infos.values())
    all_s = sum(i.total_separate_ms for i in layer_infos.values())
    all_c = sum(i.total_cudnn_ms for i in layer_infos.values())
    if all_f > 1e-6:
        print(f"{'[TOTAL]':<30} {'':>4} {'':>9} "
              f"{all_f:>10.2f} {all_s:>10.2f} {all_c:>10.2f} "
              f"{all_s/all_f:>9.2f}x {all_c/all_f:>8.2f}x")

    ef = (all_f / 1000.0) * args.power
    ec = (all_c / 1000.0) * args.power
    print(f"\n  cuDNN+LIF Energy : {ec:.4f} J")
    print(f"  Fused    Energy  : {ef:.4f} J")
    saving = (1 - ef / max(ec, 1e-9)) * 100
    print(f"  Energy Saving    : {saving:.2f}%")
    print("=" * 100)


if __name__ == "__main__":
    main()