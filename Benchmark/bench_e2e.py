import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import copy
import json
import math
from collections import OrderedDict, defaultdict

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
from Ops.static_zero_conv2d import StaticZeroConv2d


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

MIN_SPATIAL_SIZE = 8


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


def build_dataset(dataset_name, data_root, spike_mode="normalized_bernoulli"):
    normalize = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )

    if spike_mode == "normalized_bernoulli":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
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
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _get_module_by_name(model, name):
    parts = name.split(".")
    cur = model
    for part in parts:
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def make_spike_input(imgs, T, device, spike_mode="normalized_bernoulli"):
    imgs = imgs.to(device)

    if spike_mode == "normalized_bernoulli":
        rates = imgs.clamp(0, 1)
        return torch.bernoulli(
            rates.unsqueeze(0).repeat(T, 1, 1, 1, 1)
        )

    elif spike_mode == "raw_bernoulli":
        rates = imgs.clamp(0, 1)
        return torch.bernoulli(
            rates.unsqueeze(0).repeat(T, 1, 1, 1, 1)
        )

    elif spike_mode == "raw_repeat":
        return imgs.unsqueeze(0).repeat(T, 1, 1, 1, 1)

    else:
        raise ValueError(f"Unknown spike_mode: {spike_mode}")


def classify_target_type(info):
    k = info["kernel_size"]
    s = info["stride"]
    p = info["padding"]
    g = info["groups"]

    if k == (1, 1) and s == (1, 1) and p == (0, 0) and g == 1:
        return "1x1/s1"
    if k == (3, 3) and s == (1, 1) and p == (1, 1) and g == 1:
        return "3x3/s1"
    if k == (3, 3) and s == (2, 2) and p == (1, 1) and g == 1:
        return "3x3/s2"
    return "unsupported"


def route_label(info, static_zero_layers, disable_static_zero, only_static_zero):
    name = info["name"]
    kind = classify_target_type(info)

    if (not disable_static_zero) and (name in static_zero_layers):
        return f"StaticZeroConv2d[{kind}]"

    if only_static_zero:
        return f"DenseKeep[{kind}]"

    return f"SparseConv2d[{kind}]"


# =============================================================================
# 网络分析
# =============================================================================

def analyze_targets(model, sample_input, device, fused=False):
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

    targets = []
    skipped = []
    visited = set()

    for i, (name, module) in enumerate(module_list):
        if not isinstance(module, SPIKE_OPS):
            continue

        for j in range(i + 1, min(i + 15, len(module_list))):
            next_name, next_module = module_list[j]
            if next_name in visited:
                continue
            if isinstance(next_module, SPIKE_OPS):
                break
            if isinstance(next_module, TRANSPARENT_TYPES):
                continue
            if not isinstance(next_module, nn.Conv2d):
                continue

            k = next_module.kernel_size
            s = next_module.stride
            p = next_module.padding
            g = next_module.groups
            c_in = next_module.in_channels
            c_out = next_module.out_channels

            ishape = input_shapes.get(next_name)
            H, W = 0, 0
            if ishape is not None:
                if len(ishape) == 5:
                    H, W = ishape[3], ishape[4]
                elif len(ishape) == 4:
                    H, W = ishape[2], ishape[3]

            info = {
                "name": next_name,
                "module": next_module,
                "kernel_size": k,
                "stride": s,
                "padding": p,
                "groups": g,
                "in_channels": c_in,
                "out_channels": c_out,
                "H": H,
                "W": W,
                "input_shape": ishape,
            }

            if ishape is None or H == 0 or W == 0:
                info["reason"] = "no_input_shape"
                skipped.append(info)
                visited.add(next_name)
                continue

            if min(H, W) < MIN_SPATIAL_SIZE:
                info["reason"] = "small_feature_map"
                skipped.append(info)
                visited.add(next_name)
                continue

            kind = classify_target_type(info)
            if kind == "unsupported":
                info["reason"] = f"unsupported_config_k{k[0]}s{s[0]}p{p[0]}g{g}"
                skipped.append(info)
                visited.add(next_name)
                continue

            info["reason"] = "supported"
            info["kind"] = kind

            if fused:
                lif_name, lif_module = None, None
                for jj in range(j + 1, min(j + 6, len(module_list))):
                    jj_name, jj_mod = module_list[jj]
                    if isinstance(jj_mod, (nn.BatchNorm2d, nn.Identity)):
                        continue
                    if isinstance(jj_mod, SPIKE_OPS):
                        lif_name = jj_name
                        lif_module = jj_mod
                    break
                info["lif_name"] = lif_name
                info["lif_module"] = lif_module

            targets.append(info)
            visited.add(next_name)

    return targets, skipped


def print_analysis_report(targets, skipped, fused=False):
    all_candidates = targets + skipped
    all_candidates.sort(key=lambda x: x["name"])

    total_conv = len(all_candidates)
    supported_n = len(targets)
    skipped_n = len(skipped)

    print(f"\n  ┌{'─'*120}┐")
    print(f"  │{'候选层分析报告':^118}│")
    print(f"  ├{'─'*120}┤")
    header = (f"  │ {'Layer':<36} {'C_in':>5} {'C_out':>5} "
              f"{'Kernel':>6} {'Stride':>6} {'Pad':>4} {'Grp':>4} "
              f"{'H':>4} {'W':>4} {'Status':<10} {'Reason':<22} │")
    print(header)
    print(f"  ├{'─'*120}┤")

    for info in all_candidates:
        name = info["name"]
        short = name if len(name) <= 35 else "..." + name[-32:]
        k = info["kernel_size"]
        s = info["stride"]
        p = info["padding"]
        g = info["groups"]
        c_in = info["in_channels"]
        c_out = info["out_channels"]
        H = info["H"]
        W = info["W"]
        reason = info["reason"]

        if reason == "supported":
            status = "REPLACE"
            fuse_info = ""
            if fused and info.get("lif_name"):
                fuse_info = "+LIF"
            status = f"REPLACE{fuse_info}"
        else:
            status = "SKIP"

        k_str = f"{k[0]}x{k[1]}"
        s_str = f"{s[0]},{s[1]}"
        p_str = f"{p[0]},{p[1]}" if isinstance(p, tuple) else str(p)

        line = (f"  │ {short:<36} {c_in:>5} {c_out:>5} "
                f"{k_str:>6} {s_str:>6} {p_str:>4} {g:>4} "
                f"{H:>4} {W:>4} {status:<10} {reason:<22} │")
        print(line)

    print(f"  └{'─'*120}┘")
    print(f"\n  汇总: total_conv={total_conv}  "
          f"supported={supported_n}  skipped={skipped_n}  "
          f"replaced={supported_n}")

    if skipped:
        reason_counts = {}
        for s in skipped:
            r = s["reason"]
            reason_counts[r] = reason_counts.get(r, 0) + 1
        parts = [f"{r}={c}" for r, c in sorted(reason_counts.items())]
        print(f"  跳过原因: {', '.join(parts)}")


# =============================================================================
# 稀疏率统计
# =============================================================================

def measure_sparsity(model, targets, loader, device, T, num_batches=10,
                     spike_mode="normalized_bernoulli"):
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
        name = target["name"]
        m = _get_module_by_name(model, name)
        hook_handles.append(m.register_forward_hook(make_hook(name)))

    batch_count = 0
    for imgs, _ in loader:
        if batch_count >= num_batches:
            break
        inp = make_spike_input(imgs, T, device, spike_mode=spike_mode)
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


def collect_static_zero_layers(targets, sparsity_data):
    zero_layers = set()
    for t in targets:
        name = t["name"]
        sd = sparsity_data[name]
        if sd["total"] > 0 and sd["zeros"] == sd["total"]:
            zero_layers.add(name)
    return zero_layers


# =============================================================================
# stage/logits 诊断
# =============================================================================

def measure_stage_spike_rates(model, loader, device, T, num_batches=10,
                              spike_mode="normalized_bernoulli"):
    stage_names = [
        "sn1",
        "layer1.0.sn2",
        "layer1.2.sn3",
        "layer2.0.sn2",
        "layer2.7.sn3",
        "layer3.0.sn2",
        "layer3.35.sn3",
        "layer4.0.sn2",
    ]

    modules = dict(model.named_modules())
    stats = OrderedDict()
    hooks = []

    for name in stage_names:
        if name in modules:
            stats[name] = {"nz": 0, "total": 0}

            def make_hook(layer_name):
                def hook(m, inp, out):
                    if isinstance(out, torch.Tensor):
                        y = out.detach()
                        stats[layer_name]["nz"] += (y.abs() > 1e-6).sum().item()
                        stats[layer_name]["total"] += y.numel()
                return hook

            hooks.append(modules[name].register_forward_hook(make_hook(name)))

    count = 0
    for imgs, _ in loader:
        if count >= num_batches:
            break
        inp = make_spike_input(imgs, T, device, spike_mode=spike_mode)
        sj_func.reset_net(model)
        with torch.no_grad():
            _ = model(inp)
        count += 1

    for h in hooks:
        h.remove()

    rates = OrderedDict()
    for name, st in stats.items():
        rates[name] = 100.0 * st["nz"] / max(st["total"], 1)
    return rates


def print_stage_spike_rates(rates):
    print(f"\n  {'Stage':<20} {'Spike Rate':>12}")
    print(f"  {'-'*36}")
    for name, val in rates.items():
        print(f"  {name:<20} {val:>11.4f}%")


def inspect_logits(model, loader, device, T, num_batches=5,
                   spike_mode="normalized_bernoulli"):
    mean_abs_sum = 0.0
    std_sum = 0.0
    max_abs_sum = 0.0
    count = 0

    for imgs, _ in loader:
        if count >= num_batches:
            break

        inp = make_spike_input(imgs, T, device, spike_mode=spike_mode)
        sj_func.reset_net(model)
        with torch.no_grad():
            out = model(inp)

        if out.dim() == 3:
            out = out.mean(dim=0)

        mean_abs_sum += out.abs().mean().item()
        std_sum += out.std().item()
        max_abs_sum += out.abs().max().item()
        count += 1

    n = max(count, 1)
    return {
        "mean_abs": mean_abs_sum / n,
        "std": std_sum / n,
        "max_abs": max_abs_sum / n,
    }


# =============================================================================
# 模型替换（唯一真相来源）
# =============================================================================

def replace_model(
    model,
    targets,
    fused=False,
    static_zero_layers=None,
    only_static_zero=False,
):
    if static_zero_layers is None:
        static_zero_layers = set()

    replaced = 0
    sparse_count = 0
    fused_count = 0
    static_zero_count = 0
    dense_keep_count = 0

    for target in targets:
        conv_name = target["name"]
        conv_module = target["module"]

        # 1) StaticZero 最优先，但只吃最终传进来的 static_zero_layers
        if conv_name in static_zero_layers:
            zero_conv = StaticZeroConv2d.from_conv(conv_module)
            _set_module_by_name(model, conv_name, zero_conv)
            static_zero_count += 1
            replaced += 1
            continue

        # 2) only_static_zero: 非全零层保持 dense
        if only_static_zero:
            dense_keep_count += 1
            replaced += 1
            continue

        # 3) fused 路径
        if fused and target.get("lif_name") is not None:
            fused_module = FusedSparseConvLIF.from_conv_and_lif(
                conv_module, target["lif_module"])
            _set_module_by_name(model, conv_name, fused_module)
            _set_module_by_name(model, target["lif_name"], nn.Identity())
            fused_count += 1
            replaced += 1
            continue

        # 4) 默认 SparseConv2d
        sparse_conv = SparseConv2d.from_dense(conv_module)
        _set_module_by_name(model, conv_name, sparse_conv)
        sparse_count += 1
        replaced += 1

    return replaced, sparse_count, fused_count, static_zero_count, dense_keep_count


# =============================================================================
# 端到端延迟测量
# =============================================================================

def measure_e2e_latency(model, loader, device, T, warmup=10, max_batches=None,
                        spike_mode="normalized_bernoulli"):
    warmup_count = 0
    for imgs, _ in loader:
        if warmup_count >= warmup:
            break
        inp = make_spike_input(imgs, T, device, spike_mode=spike_mode)
        sj_func.reset_net(model)
        with torch.no_grad():
            _ = model(inp)
        warmup_count += 1

    sync()

    total_ms = 0.0
    num_batches = 0
    for imgs, _ in loader:
        if max_batches is not None and num_batches >= max_batches:
            break

        inp = make_spike_input(imgs, T, device, spike_mode=spike_mode)
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
# Per-layer Timing
# =============================================================================

def measure_layer_timing(model, loader, device, T, target_names,
                         warmup=5, num_batches=20,
                         spike_mode="normalized_bernoulli"):
    name_to_module = {}
    for name, module in model.named_modules():
        if name in target_names:
            name_to_module[name] = module

    timing = {name: {"total_ms": 0.0, "count": 0} for name in target_names}
    pre_events = {}
    post_events = {}
    hook_handles = []

    def make_pre_hook(layer_name):
        def hook(m, inp):
            evt = torch.cuda.Event(enable_timing=True)
            evt.record()
            pre_events[layer_name] = evt
        return hook

    def make_post_hook(layer_name):
        def hook(m, inp, out):
            evt = torch.cuda.Event(enable_timing=True)
            evt.record()
            post_events[layer_name] = evt
        return hook

    for name in target_names:
        if name in name_to_module:
            mod = name_to_module[name]
            hook_handles.append(mod.register_forward_pre_hook(make_pre_hook(name)))
            hook_handles.append(mod.register_forward_hook(make_post_hook(name)))

    warmup_count = 0
    for imgs, _ in loader:
        if warmup_count >= warmup:
            break
        inp = make_spike_input(imgs, T, device, spike_mode=spike_mode)
        sj_func.reset_net(model)
        with torch.no_grad():
            _ = model(inp)
        warmup_count += 1
    sync()

    batch_count = 0
    for imgs, _ in loader:
        if batch_count >= num_batches:
            break

        pre_events.clear()
        post_events.clear()

        inp = make_spike_input(imgs, T, device, spike_mode=spike_mode)
        sj_func.reset_net(model)
        with torch.no_grad():
            _ = model(inp)
        sync()

        for name in target_names:
            if name in pre_events and name in post_events:
                ms = pre_events[name].elapsed_time(post_events[name])
                timing[name]["total_ms"] += ms
                timing[name]["count"] += 1

        batch_count += 1

    for h in hook_handles:
        h.remove()

    result = {}
    for name in target_names:
        t = timing[name]
        result[name] = t["total_ms"] / t["count"] if t["count"] > 0 else 0.0
    return result


def print_layer_profile(baseline_timing, sparse_timing, targets, e2e_ms=None):
    target_names = [t["name"] for t in targets]

    print(f"\n  {'Layer':<40} {'Base(ms)':>9} {'Sparse(ms)':>11} {'Speedup':>8}")
    print(f"  {'-'*72}")

    total_base = 0.0
    total_sparse = 0.0

    bucket = defaultdict(lambda: {"count": 0, "base": 0.0, "sparse": 0.0})
    target_map = {t["name"]: t for t in targets}

    for name in target_names:
        b = baseline_timing.get(name, 0.0)
        s = sparse_timing.get(name, 0.0)
        total_base += b
        total_sparse += s

        sp = f"{b / s:.3f}x" if s > 1e-6 else "inf"
        short = name if len(name) <= 39 else "..." + name[-36:]
        print(f"  {short:<40} {b:>9.3f} {s:>11.3f} {sp:>8}")

        kind = classify_target_type(target_map[name])
        bucket[kind]["count"] += 1
        bucket[kind]["base"] += b
        bucket[kind]["sparse"] += s

    print(f"  {'-'*72}")
    sp = f"{total_base / total_sparse:.3f}x" if total_sparse > 1e-6 else "inf"
    print(f"  {'[REPLACED TOTAL]':<40} {total_base:>9.3f} {total_sparse:>11.3f} {sp:>8}")

    if e2e_ms is not None and e2e_ms > 1e-6:
        ratio = 100.0 * total_base / e2e_ms
        print(f"\n  replaced layers 占 baseline e2e: {ratio:.2f}%")

    print(f"\n  {'Bucket':<12} {'Count':>6} {'Base(ms)':>10} {'Sparse(ms)':>11} {'Speedup':>8}")
    print(f"  {'-'*58}")
    for kind in ["1x1/s1", "3x3/s1", "3x3/s2"]:
        c = bucket[kind]["count"]
        b = bucket[kind]["base"]
        s = bucket[kind]["sparse"]
        sp = f"{b / s:.3f}x" if s > 1e-6 else "inf"
        print(f"  {kind:<12} {c:>6} {b:>10.3f} {s:>11.3f} {sp:>8}")


# =============================================================================
# 数值一致性验证
# =============================================================================

def verify_consistency(model_baseline, model_sparse, loader, device, T,
                       num_batches=5, spike_mode="normalized_bernoulli"):
    cosine_sum = 0.0
    agree_sum = 0.0
    global_max_abs = 0.0
    count = 0

    for imgs, _ in loader:
        if count >= num_batches:
            break

        inp = make_spike_input(imgs, T, device, spike_mode=spike_mode)

        sj_func.reset_net(model_baseline)
        with torch.no_grad():
            out_base = model_baseline(inp)

        sj_func.reset_net(model_sparse)
        with torch.no_grad():
            out_sparse = model_sparse(inp)

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
                flat_s.unsqueeze(0), flat_b.unsqueeze(0)
            ).item()
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
        description="SparseFlow E2E Benchmark — unified static-zero / sparse evaluation")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=list(MODEL_BUILDERS.keys()))
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100"])
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--v_threshold", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--power", type=float, default=250.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--verify_batches", type=int, default=10)
    parser.add_argument("--sparsity_batches", type=int, default=20)
    parser.add_argument("--save_json", type=str, default="")
    parser.add_argument("--layer_profile", action="store_true")
    parser.add_argument("--profile_batches", type=int, default=20)

    parser.add_argument(
        "--spike_mode",
        type=str,
        default="normalized_bernoulli",
        choices=["normalized_bernoulli", "raw_bernoulli", "raw_repeat"],
    )
    parser.add_argument("--inspect_spikes", action="store_true")
    parser.add_argument("--inspect_logits", action="store_true")

    parser.add_argument("--disable_static_zero", action="store_true")
    parser.add_argument("--only_static_zero", action="store_true")

    args = parser.parse_args()

    if args.disable_static_zero and args.only_static_zero:
        raise ValueError("--disable_static_zero 和 --only_static_zero 不能同时开启。")

    mode_str = "Fused" if args.fused else "SparseConv"

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
    print(f"  Layer profile:{args.layer_profile}")
    print(f"  Spike mode:   {args.spike_mode}")
    print(f"  Disable SZ:   {args.disable_static_zero}")
    print(f"  Only SZ:      {args.only_static_zero}")
    print()

    # 1. 模型
    print(f"[1/6] 构建 Spiking-{args.model} ...")
    model_baseline = build_model(args.model, device, args.v_threshold)

    # 2. 数据
    print(f"[2/6] 加载 {args.dataset} 测试集 ...")
    ds = build_dataset(args.dataset, args.data_root, spike_mode=args.spike_mode)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    num_classes = 10 if args.dataset == "cifar10" else 100
    print(f"  测试集: {len(ds)} 张, {len(loader)} batches, {num_classes} classes")

    # 3. 分析
    print(f"[3/6] 分析网络拓扑，识别替换目标 ...")
    imgs_s, _ = next(iter(loader))
    sample_input = make_spike_input(
        imgs_s[:4], args.T, device, spike_mode=args.spike_mode
    )
    targets, skipped = analyze_targets(model_baseline, sample_input, device,
                                       fused=args.fused)

    print_analysis_report(targets, skipped, fused=args.fused)

    if not targets:
        print("\n  未找到可替换的目标层，退出。")
        return

    # 4. 稀疏率统计
    print(f"\n[4/6] 统计输入特征图稀疏率 ({args.sparsity_batches} batches) ...")
    sparsity_data = measure_sparsity(
        model_baseline, targets, loader, device, args.T,
        num_batches=args.sparsity_batches,
        spike_mode=args.spike_mode
    )

    zero_layers = collect_static_zero_layers(targets, sparsity_data)

    # 这里是唯一真相来源：最终 static_zero_layers 只在这里决策
    if args.disable_static_zero:
        print(f"\n  检测到 {len(zero_layers)} 个全零输入层，但 --disable_static_zero 已启用，"
              f"这些层不会替换为 StaticZeroConv2d。")
        static_zero_layers = set()
    elif args.only_static_zero:
        print(f"\n  检测到 {len(zero_layers)} 个全零输入层，将仅替换这些层为 StaticZeroConv2d；"
              f"其余层保持 Dense。")
        static_zero_layers = set(zero_layers)
    else:
        print(f"\n  检测到 {len(zero_layers)} 个全零输入层，将替换为 StaticZeroConv2d。")
        static_zero_layers = set(zero_layers)

    print(f"\n  {'Layer':<40} {'Sparsity':>10} {'Route':>26}")
    print(f"  {'-'*84}")
    sparsity_values = []
    for t in targets:
        name = t["name"]
        sd = sparsity_data[name]
        sp = sd["zeros"] / sd["total"] * 100 if sd["total"] > 0 else 0.0
        sparsity_values.append(sp)
        route = route_label(
            t,
            static_zero_layers=static_zero_layers,
            disable_static_zero=args.disable_static_zero,
            only_static_zero=args.only_static_zero,
        )
        short_name = name if len(name) <= 39 else "..." + name[-36:]
        print(f"  {short_name:<40} {sp:>9.2f}% {route:>26}")

    avg_sparsity = sum(sparsity_values) / max(len(sparsity_values), 1)
    print(f"  {'[平均]':<40} {avg_sparsity:>9.2f}%")

    if args.inspect_spikes:
        print(f"\n[diag] 统计 stage-wise spike rate ...")
        stage_rates = measure_stage_spike_rates(
            model_baseline, loader, device, args.T,
            num_batches=min(args.sparsity_batches, 10),
            spike_mode=args.spike_mode
        )
        print_stage_spike_rates(stage_rates)
    else:
        stage_rates = {}

    if args.inspect_logits:
        print(f"\n[diag] 检查 baseline logits 统计 ...")
        logits_stats = inspect_logits(
            model_baseline, loader, device, args.T,
            num_batches=5,
            spike_mode=args.spike_mode
        )
        print(f"  mean(|logits|): {logits_stats['mean_abs']:.6f}")
        print(f"  std(logits):    {logits_stats['std']:.6f}")
        print(f"  max(|logits|):  {logits_stats['max_abs']:.6f}")
    else:
        logits_stats = {}

    # 构建 sparse model
    model_sparse = copy.deepcopy(model_baseline)
    replaced, sparse_n, fused_n, static_zero_n, dense_keep_n = replace_model(
        model_sparse,
        targets,
        fused=args.fused,
        static_zero_layers=static_zero_layers,
        only_static_zero=args.only_static_zero,
    )

    if args.disable_static_zero and static_zero_n != 0:
        raise RuntimeError(
            f"--disable_static_zero 已启用，但实际仍生成了 {static_zero_n} 个 StaticZeroConv2d。"
        )

    print(f"\n  替换完成: {sparse_n} SparseConv2d + "
          f"{fused_n} FusedConvLIF + "
          f"{static_zero_n} StaticZeroConv2d + "
          f"{dense_keep_n} DenseKeep = {replaced} total")

    # 5. e2e
    print(f"\n[5/6] 端到端延迟测量 (warmup={args.warmup}) ...")

    print(f"  测量 cuDNN baseline ...")
    cudnn_avg, cudnn_total, cudnn_n = measure_e2e_latency(
        model_baseline, loader, device, args.T,
        warmup=args.warmup,
        spike_mode=args.spike_mode
    )
    print(f"    cuDNN:      {cudnn_avg:.2f} ms/batch  "
          f"(total={cudnn_total:.1f} ms, {cudnn_n} batches)")

    print(f"  测量 {mode_str} ...")
    sparse_avg, sparse_total, sparse_n_batches = measure_e2e_latency(
        model_sparse, loader, device, args.T,
        warmup=args.warmup,
        spike_mode=args.spike_mode
    )
    print(f"    {mode_str}: {sparse_avg:.2f} ms/batch  "
          f"(total={sparse_total:.1f} ms, {sparse_n_batches} batches)")

    speedup = cudnn_avg / sparse_avg if sparse_avg > 1e-6 else float("inf")
    print(f"\n  Speedup ({mode_str} vs cuDNN): {speedup:.3f}x")

    cudnn_energy = (cudnn_total / 1000.0) * args.power
    sparse_energy = (sparse_total / 1000.0) * args.power
    energy_saving = (1 - sparse_energy / max(cudnn_energy, 1e-9)) * 100
    print(f"  Energy cuDNN:      {cudnn_energy:.4f} J")
    print(f"  Energy {mode_str}: {sparse_energy:.4f} J")
    print(f"  Energy saving:     {energy_saving:.2f}%")

    # 5.5 layer profile
    layer_profile_data = {}
    if args.layer_profile:
        print(f"\n  [Layer Profile] 逐层耗时对比 ({args.profile_batches} batches) ...")
        target_names = [t["name"] for t in targets]

        print(f"    测量 baseline 逐层耗时 ...")
        baseline_timing = measure_layer_timing(
            model_baseline, loader, device, args.T,
            target_names, warmup=5, num_batches=args.profile_batches,
            spike_mode=args.spike_mode
        )

        print(f"    测量 {mode_str} 逐层耗时 ...")
        sparse_timing = measure_layer_timing(
            model_sparse, loader, device, args.T,
            target_names, warmup=5, num_batches=args.profile_batches,
            spike_mode=args.spike_mode
        )

        print_layer_profile(
            baseline_timing, sparse_timing, targets,
            e2e_ms=cudnn_avg
        )
        layer_profile_data = {
            name: {
                "baseline_ms": round(baseline_timing.get(name, 0.0), 4),
                "sparse_ms": round(sparse_timing.get(name, 0.0), 4),
            }
            for name in target_names
        }

    # 6. consistency
    print(f"\n[6/6] 数值一致性验证 ({args.verify_batches} batches) ...")
    avg_cos, avg_agree, max_abs = verify_consistency(
        model_baseline, model_sparse, loader, device, args.T,
        num_batches=args.verify_batches,
        spike_mode=args.spike_mode
    )
    print(f"  Cosine similarity:  {avg_cos:.8f}")
    print(f"  Pred agreement:     {avg_agree*100:.2f}%")
    print(f"  Max absolute error: {max_abs:.6f}")

    consistency_ok = avg_cos > 0.999 and max_abs < 0.1
    print(f"  Consistency:        {'PASS' if consistency_ok else 'FAIL'}")

    # summary
    print(f"\n{'='*80}")
    print(f"{'SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"  Model:          Spiking-{args.model}")
    print(f"  Dataset:        {args.dataset}")
    print(f"  T:              {args.T}")
    print(f"  Mode:           {mode_str}")
    print(f"  Spike mode:     {args.spike_mode}")
    print(f"  Targets:        {len(targets)} replaced, {len(skipped)} skipped")
    print(f"  Avg Sparsity:   {avg_sparsity:.2f}%")
    print(f"  StaticZero:     {static_zero_n}")
    print(f"  DenseKeep:      {dense_keep_n}")
    print(f"  cuDNN:          {cudnn_avg:.2f} ms/batch")
    print(f"  {mode_str + ':':<16}{sparse_avg:.2f} ms/batch")
    print(f"  Speedup:        {speedup:.3f}x")
    print(f"  Energy saving:  {energy_saving:.2f}%")
    print(f"  Cosine sim:     {avg_cos:.8f}")
    print(f"  Pred agreement: {avg_agree*100:.2f}%")
    print(f"  Consistency:    {'PASS' if consistency_ok else 'FAIL'}")
    print(f"{'='*80}\n")

    results = {
        "model": args.model,
        "dataset": args.dataset,
        "T": args.T,
        "fused": args.fused,
        "mode": mode_str,
        "spike_mode": args.spike_mode,
        "disable_static_zero": args.disable_static_zero,
        "only_static_zero": args.only_static_zero,
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
        "num_skipped": len(skipped),
        "num_replaced": replaced,
        "num_sparse_conv": sparse_n,
        "num_fused": fused_n,
        "num_static_zero": static_zero_n,
        "num_dense_keep": dense_keep_n,
        "static_zero_layers": sorted(list(static_zero_layers)),
        "stage_spike_rates": stage_rates,
        "logits_stats": logits_stats,
    }

    if layer_profile_data:
        results["layer_profile"] = layer_profile_data

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  结果已保存到: {args.save_json}")
    else:
        fname = (f"results_{args.model}_{args.dataset}_T{args.T}"
                 f"{'_fused' if args.fused else ''}.json")
        with open(fname, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  结果已保存到: {fname}")


if __name__ == "__main__":
    main()