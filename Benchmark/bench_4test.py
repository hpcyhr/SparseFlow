import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import copy
import json
import math
from collections import Counter, OrderedDict, defaultdict
import os
from PIL import Image
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from inspect import signature

from spikingjelly.activation_based.model import spiking_resnet, spiking_vgg, sew_resnet
from spikingjelly.activation_based import functional as sj_func
from spikingjelly.activation_based import neuron as sj_neuron

from Ops.sparse_conv2d import SparseConv2d
from Ops.sparse_fused_conv_lif import FusedSparseConvLIF
from Ops.static_zero_conv2d import StaticZeroConv2d, make_synthetic_zero_diag
from Utils.layer_logger import LayerLogger
from Utils.dispatch_model import dispatch_all_layers, decisions_to_sets
from Utils.timing_utils import prepare_for_timing, set_launch_mode, count_sync_state
from Core.registry import SpikeOpRegistry
from Core.analyzer import NetworkAnalyzer
from Core.replacer import ModuleReplacer


DEVICE = None
SPIKE_OPS = (sj_neuron.LIFNode, sj_neuron.IFNode, sj_neuron.ParametricLIFNode)

TRANSPARENT_TYPES = (
    nn.BatchNorm2d, nn.BatchNorm1d,
    nn.Dropout, nn.Dropout2d,
    nn.Identity, nn.Flatten,
    nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d,
    nn.ReLU, nn.ReLU6, nn.LeakyReLU,
)

MIN_SPATIAL_SIZE = 4


def sync():
    torch.cuda.synchronize(DEVICE)


def make_event():
    return torch.cuda.Event(enable_timing=True)


def _discover_model_builders():
    modules = [
        ("resnet", spiking_resnet, "spiking_"),
        ("vgg", spiking_vgg, "spiking_"),
        ("sew_resnet", sew_resnet, "sew_"),
    ]
    builders = OrderedDict()
    metadata = {}
    for family, mod, prefix in modules:
        for attr_name in dir(mod):
            if not attr_name.startswith(prefix):
                continue
            fn = getattr(mod, attr_name)
            if not callable(fn):
                continue
            builders[attr_name] = fn
            metadata[attr_name] = {"family": family, "module": mod}
    return builders, metadata


MODEL_BUILDERS, MODEL_META = _discover_model_builders()


def print_supported_models():
    print("\n鏀寔鐨勬ā鍨?")
    for name in MODEL_BUILDERS:
        print(f"  {name}")
    print()


def _filter_builder_kwargs(builder_fn, candidate_kwargs):
    try:
        sig = signature(builder_fn)
    except (ValueError, TypeError):
        return candidate_kwargs
    params = sig.parameters
    accepts_var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return {k: v for k, v in candidate_kwargs.items() if v is not None}
    filtered = {}
    for k, v in candidate_kwargs.items():
        if v is None:
            continue
        if k in params:
            filtered[k] = v
    return filtered


def set_random_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reinitialize_model_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif hasattr(m, "reset_parameters") and not isinstance(m, (sj_neuron.LIFNode, sj_neuron.IFNode, sj_neuron.ParametricLIFNode, nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
            try:
                m.reset_parameters()
            except Exception:
                pass


def build_model(model_name, device, v_threshold=1.0, weight_init="random", sew_cnf=None,
                num_classes=None, seed=None):
    if weight_init not in ("random", "pretrained"):
        raise ValueError(f"Unknown weight_init: {weight_init}")

    # Auto-default cnf='ADD' for SEW-ResNet models when not explicitly specified
    if sew_cnf is None and model_name.startswith("sew_"):
        sew_cnf = "ADD"

    builder = MODEL_BUILDERS[model_name]
    use_pretrained = (weight_init == "pretrained")
    builder_kwargs = {
        "pretrained": use_pretrained,
        "progress": True,
        "spiking_neuron": sj_neuron.LIFNode,
        "v_threshold": v_threshold,
        "num_classes": num_classes,
        "cnf": sew_cnf,
    }
    filtered = _filter_builder_kwargs(builder, builder_kwargs)
    set_random_seed(seed)
    model = builder(**filtered)
    if weight_init == "random":
        reinitialize_model_weights(model)
    model.to(device).eval()
    sj_func.set_step_mode(model, 'm')
    return model

class FlatImageFolderDataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        exts=(".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"),
    ):
        self.root = root
        self.transform = transform
        self.samples = []

        if not os.path.isdir(root):
            raise RuntimeError(f"Dataset directory does not exist: {root}")

        for name in sorted(os.listdir(root)):
            if name.endswith(exts):
                self.samples.append(os.path.join(root, name))

        if not self.samples:
            raise RuntimeError(f"No image files found in: {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # 杩欓噷鍙仛鎺ㄧ悊鎬ц兘娴嬭瘯锛屾墍浠ユ爣绛剧粰 dummy=0 鍗冲彲
        return img, 0

def build_dataset(dataset_name, data_root, spike_mode="normalized_bernoulli"):
    if dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        ds = datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transform,
        )

    elif dataset_name == "cifar100":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        ds = datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=transform,
        )

    elif dataset_name == "imagenet_val_flat":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        ds = FlatImageFolderDataset(
            root=data_root,
            transform=transform,
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return ds


def _set_module_by_name(model, name, new_module):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


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
    if spike_mode in ("normalized_bernoulli", "raw_bernoulli"):
        rates = imgs.clamp(0, 1)
        return torch.bernoulli(rates.unsqueeze(0).repeat(T, 1, 1, 1, 1))
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
    if k == (1, 1) and s == (2, 2) and p == (0, 0) and g == 1:
        return "1x1/s2"
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


def analyze_targets(model, sample_input, device, fused=False, min_spatial_size=MIN_SPATIAL_SIZE):
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
                "name": next_name, "module": next_module,
                "kernel_size": k, "stride": s, "padding": p,
                "groups": g, "in_channels": c_in, "out_channels": c_out,
                "H": H, "W": W, "input_shape": ishape,
            }

            if ishape is None or H == 0 or W == 0:
                info["reason"] = "no_input_shape"
                skipped.append(info); visited.add(next_name); continue
            if min(H, W) < min_spatial_size:
                info["reason"] = "small_feature_map"
                skipped.append(info); visited.add(next_name); continue

            kind = classify_target_type(info)
            if kind == "unsupported":
                info["reason"] = f"unsupported_config_k{k[0]}s{s[0]}p{p[0]}g{g}"
                skipped.append(info); visited.add(next_name); continue

            info["reason"] = "supported"
            info["kind"] = kind
            targets.append(info)
            visited.add(next_name)

    return targets, skipped


def print_analysis_report(targets, skipped, fused=False):
    all_candidates = targets + skipped
    total_conv = len(all_candidates)
    supported_n = len(targets)
    skipped_n = len(skipped)

    print(f"\n  {'=' * 120}")
    print(f"  {'Candidate Layer Analysis':^118}")
    print(f"  {'=' * 120}")
    header = (
        f"  {'Layer':<36} {'C_in':>5} {'C_out':>5} "
        f"{'Kernel':>6} {'Stride':>6} {'Pad':>4} {'Grp':>4} "
        f"{'H':>4} {'W':>4} {'Status':<10} {'Reason':<22}"
    )
    print(header)
    print(f"  {'-' * 120}")

    for info in all_candidates:
        name = info["name"]
        short = name if len(name) <= 35 else "..." + name[-32:]
        k, s, p, g = info["kernel_size"], info["stride"], info["padding"], info["groups"]
        reason = info["reason"]
        status = "REPLACE" if reason == "supported" else "SKIP"
        k_str = f"{k[0]}x{k[1]}"
        s_str = f"{s[0]},{s[1]}"
        p_str = f"{p[0]},{p[1]}" if isinstance(p, tuple) else str(p)
        line = (
            f"  {short:<36} {info['in_channels']:>5} {info['out_channels']:>5} "
            f"{k_str:>6} {s_str:>6} {p_str:>4} {g:>4} "
            f"{info['H']:>4} {info['W']:>4} {status:<10} {reason:<22}"
        )
        print(line)

    print(f"  {'=' * 120}")
    print(f"\n  Summary: total_conv={total_conv}  supported={supported_n}  skipped={skipped_n}")
    if skipped:
        reason_counts = {}
        for s in skipped:
            r = s["reason"]
            reason_counts[r] = reason_counts.get(r, 0) + 1
        parts = [f"{r}={c}" for r, c in sorted(reason_counts.items())]
        print(f"  Skip reasons: {', '.join(parts)}")


def measure_sparsity(model, targets, loader, device, T, num_batches=10,
                     spike_mode="normalized_bernoulli"):
    sparsity_data = {t["name"]: {"zeros": 0, "total": 0} for t in targets}

    def make_hook(name):
        def hook(m, inp, out):
            x = inp[0]
            if not isinstance(x, torch.Tensor):
                return
            with torch.no_grad():
                x = x.detach()
                if x.dim() == 5:
                    T_, B, C, H, W = x.shape
                    x = x.reshape(T_ * B, C, H, W)
                sparsity_data[name]["zeros"] += (x.abs() <= 1e-6).sum().item()
                sparsity_data[name]["total"] += x.numel()
                del x
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


def measure_group_sparsity(model, targets, loader, device, T, num_batches=5,
                           spike_mode="normalized_bernoulli"):
    for _, mod in model.named_modules():
        if isinstance(mod, SparseConv2d):
            mod.collect_diag = True

    batch_count = 0
    for imgs, _ in loader:
        if batch_count >= num_batches:
            break
        inp = make_spike_input(imgs, T, device, spike_mode=spike_mode)
        sj_func.reset_net(model)
        with torch.no_grad():
            _ = model(inp)
        batch_count += 1

    target_names = {t["name"] for t in targets}
    group_data = {}
    for name, mod in model.named_modules():
        if isinstance(mod, SparseConv2d) and name in target_names:
            diag = getattr(mod, '_last_diag', {})
            group_data[name] = {
                'active_group_ratio': diag.get('active_group_ratio', -1.0),
                'tile_zero_ratio': diag.get('tile_zero_ratio', -1.0),
                'total_group_count': diag.get('total_group_count', -1.0),
                'nonzero_group_count': diag.get('nonzero_group_count', -1.0),
                'tile_zero_count': diag.get('tile_zero_count', -1.0),
                'total_tile_count': diag.get('total_tile_count', -1.0),
                'effective_k_ratio': diag.get('effective_k_ratio', -1.0),
                'sparse_compute_ms': diag.get('sparse_compute_ms', -1.0),
                'sparse_total_ms': diag.get('sparse_total_ms', -1.0),
                # v19 two-stage prescan fields
                'stage1_zero_tiles': diag.get('stage1_zero_tiles', -1),
                'stage2_tiles': diag.get('stage2_tiles', -1),
                'zero_tiles': diag.get('zero_tiles', -1),
                'sparse_tiles': diag.get('sparse_tiles', -1),
                'denseish_tiles': diag.get('denseish_tiles', -1),
                'prescan_mode': diag.get('prescan_mode', 'unknown'),
                'metadata_kind': diag.get('metadata_kind', 'unknown'),
                'kernel_type': diag.get('kernel_type', 'unknown'),
                # v22 zero-candidate fields
                'stage1_zero_candidate': diag.get('stage1_zero_candidate', -1),
                'stage1_denseish': diag.get('stage1_denseish', -1),
                'stage1_uncertain': diag.get('stage1_uncertain', -1),
            }

    for _, mod in model.named_modules():
        if isinstance(mod, SparseConv2d):
            mod.collect_diag = False
    return group_data


def should_use_sparse_selective(target, group_data, min_cin=128, max_agr=0.5, no_1x1=False):
    name = target["name"]
    c_in = target.get("in_channels", 0)
    kind = classify_target_type(target)
    if no_1x1 and kind == "1x1/s1":
        return False
    if c_in < min_cin:
        return False
    gd = group_data.get(name, {})
    agr = gd.get('active_group_ratio', 1.0)
    if agr < 0:
        agr = 1.0
    if agr > max_agr:
        return False
    return True


def pick_stage_probe_names(model, max_probes=8):
    spike_names = [name for name, module in model.named_modules() if isinstance(module, SPIKE_OPS)]
    if not spike_names:
        return []
    if len(spike_names) <= max_probes:
        return spike_names
    chosen = []
    last_idx = -1
    for i in range(max_probes):
        idx = round(i * (len(spike_names) - 1) / max(max_probes - 1, 1))
        if idx == last_idx:
            continue
        chosen.append(spike_names[idx])
        last_idx = idx
    return chosen


def measure_stage_spike_rates(model, loader, device, T, num_batches=10,
                              spike_mode="normalized_bernoulli", stage_names=None):
    if stage_names is None:
        stage_names = pick_stage_probe_names(model)
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
    mean_abs_sum, std_sum, max_abs_sum, count = 0.0, 0.0, 0.0, 0
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
    return {"mean_abs": mean_abs_sum / n, "std": std_sum / n, "max_abs": max_abs_sum / n}


def replace_model(model, targets, fused=False, static_zero_layers=None,
                  only_static_zero=False, selective_sparse_set=None):
    if static_zero_layers is None:
        static_zero_layers = set()

    replaced = sparse_count = fused_count = static_zero_count = dense_keep_count = 0

    for target in targets:
        conv_name = target["name"]
        conv_module = target["module"]

        if conv_name in static_zero_layers:
            zero_conv = StaticZeroConv2d.from_conv(conv_module)
            _set_module_by_name(model, conv_name, zero_conv)
            static_zero_count += 1; replaced += 1; continue

        if only_static_zero:
            dense_keep_count += 1; replaced += 1; continue

        if selective_sparse_set is not None and conv_name not in selective_sparse_set:
            dense_keep_count += 1; replaced += 1; continue

        if fused and target.get("lif_name") is not None:
            fused_module = FusedSparseConvLIF.from_conv_and_lif(
                conv_module, target["lif_module"])
            _set_module_by_name(model, conv_name, fused_module)
            _set_module_by_name(model, target["lif_name"], nn.Identity())
            fused_count += 1; replaced += 1; continue

        sparse_conv = SparseConv2d.from_dense(conv_module)
        _set_module_by_name(model, conv_name, sparse_conv)
        sparse_count += 1; replaced += 1

    return replaced, sparse_count, fused_count, static_zero_count, dense_keep_count


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

    total_ms = 0.0; num_batches = 0
    for imgs, _ in loader:
        if max_batches is not None and num_batches >= max_batches:
            break
        inp = make_spike_input(imgs, T, device, spike_mode=spike_mode)
        sj_func.reset_net(model)
        sync()
        start_evt = make_event(); end_evt = make_event()
        start_evt.record()
        with torch.no_grad():
            _ = model(inp)
        end_evt.record()
        sync()
        total_ms += start_evt.elapsed_time(end_evt)
        num_batches += 1

    avg_ms = total_ms / max(num_batches, 1)
    return avg_ms, total_ms, num_batches


def measure_layer_timing(model, loader, device, T, target_names,
                         warmup=5, num_batches=20, spike_mode="normalized_bernoulli"):
    name_to_module = {}
    for name, module in model.named_modules():
        if name in target_names:
            name_to_module[name] = module

    timing = {name: {"total_ms": 0.0, "count": 0} for name in target_names}
    pre_events = {}; post_events = {}
    hook_handles = []

    def make_pre_hook(layer_name):
        def hook(m, inp):
            evt = torch.cuda.Event(enable_timing=True); evt.record()
            pre_events[layer_name] = evt
        return hook
    def make_post_hook(layer_name):
        def hook(m, inp, out):
            evt = torch.cuda.Event(enable_timing=True); evt.record()
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
        pre_events.clear(); post_events.clear()
        inp = make_spike_input(imgs, T, device, spike_mode=spike_mode)
        sj_func.reset_net(model)
        with torch.no_grad():
            _ = model(inp)
        sync()
        for name in target_names:
            if name in pre_events and name in post_events:
                ms = pre_events[name].elapsed_time(post_events[name])
                timing[name]["total_ms"] += ms; timing[name]["count"] += 1
        batch_count += 1

    for h in hook_handles:
        h.remove()
    result = {}
    for name in target_names:
        t = timing[name]
        result[name] = t["total_ms"] / t["count"] if t["count"] > 0 else 0.0
    return result


def print_layer_profile(baseline_timing, sparse_timing, targets, e2e_baseline_ms=None,
                        e2e_sparse_ms=None, label="Sparse"):
    """Print per-layer timing comparison between baseline and a sparse variant.

    Args:
        baseline_timing: dict layer_name -> avg ms (dense baseline)
        sparse_timing:   dict layer_name -> avg ms (sparse variant)
        targets:         list of target info dicts
        e2e_baseline_ms: optional end-to-end baseline latency for percentage calc
        e2e_sparse_ms:   optional end-to-end sparse latency for percentage calc
        label:           label string for the sparse variant column header
    """
    target_names = [t["name"] for t in targets]

    col_sparse = f"{label}(ms)"
    print(f"\n  {'Layer':<40} {'Base(ms)':>9} {col_sparse:>14} {'Speedup':>8}")
    print(f"  {'-'*75}")
    total_base = 0.0; total_sparse = 0.0
    bucket = defaultdict(lambda: {"count": 0, "base": 0.0, "sparse": 0.0})
    target_map = {t["name"]: t for t in targets}

    for name in target_names:
        b = baseline_timing.get(name, 0.0); s = sparse_timing.get(name, 0.0)
        total_base += b; total_sparse += s
        sp = f"{b / s:.3f}x" if s > 1e-6 else "inf"
        short = name if len(name) <= 39 else "..." + name[-36:]
        print(f"  {short:<40} {b:>9.3f} {s:>14.3f} {sp:>8}")
        kind = classify_target_type(target_map[name])
        bucket[kind]["count"] += 1; bucket[kind]["base"] += b; bucket[kind]["sparse"] += s

    print(f"  {'-'*75}")
    sp = f"{total_base / total_sparse:.3f}x" if total_sparse > 1e-6 else "inf"
    print(f"  {'[REPLACED TOTAL]':<40} {total_base:>9.3f} {total_sparse:>14.3f} {sp:>8}")

    if e2e_baseline_ms is not None and e2e_baseline_ms > 1e-6:
        print(f"\n  鏇挎崲灞傚崰 baseline e2e: {100.0 * total_base / e2e_baseline_ms:.2f}%")
    if e2e_sparse_ms is not None and e2e_sparse_ms > 1e-6:
        print(f"  鏇挎崲灞傚崰 {label} e2e:   {100.0 * total_sparse / e2e_sparse_ms:.2f}%")

    saved_ms = total_base - total_sparse
    print(f"  鏇挎崲灞傝妭鐪? {saved_ms:.3f} ms/batch")

    print(f"\n  {'Bucket':<12} {'Count':>6} {'Base(ms)':>10} {col_sparse:>14} {'Speedup':>8}")
    print(f"  {'-'*58}")
    for kind in ["1x1/s1", "1x1/s2", "3x3/s1", "3x3/s2"]:
        c = bucket[kind]["count"]; b = bucket[kind]["base"]; s = bucket[kind]["sparse"]
        if c == 0:
            continue
        sp = f"{b / s:.3f}x" if s > 1e-6 else "inf"
        print(f"  {kind:<12} {c:>6} {b:>10.3f} {s:>14.3f} {sp:>8}")


def verify_consistency(model_baseline, model_sparse, loader, device, T,
                       num_batches=5, spike_mode="normalized_bernoulli"):
    cosine_sum = agree_sum = 0.0; global_max_abs = 0.0; count = 0
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
        flat_s = out_sparse.flatten().float(); flat_b = out_base.flatten().float()
        if flat_s.norm() < 1e-8 and flat_b.norm() < 1e-8:
            cos = 1.0
        elif flat_s.norm() < 1e-8 or flat_b.norm() < 1e-8:
            cos = 0.0
        else:
            cos = F.cosine_similarity(flat_s.unsqueeze(0), flat_b.unsqueeze(0)).item()
        cosine_sum += cos
        pred_base = out_base.argmax(dim=-1); pred_sparse = out_sparse.argmax(dim=-1)
        agree_sum += (pred_base == pred_sparse).float().mean().item()
        count += 1
    n = max(count, 1)
    return cosine_sum / n, agree_sum / n, global_max_abs


def route_label_from_backend(info, backend):
    kind = classify_target_type(info)
    if backend == "staticzero":
        return f"StaticZeroConv2d[{kind}]"
    if backend == "dense":
        return f"DenseKeep[{kind}]"
    return f"SparseConv2d[{kind}]"


def print_fourway_route_report(targets, sparsity_data, hybrid_decisions=None):
    print(f"\n  {'Layer':<40} {'Sparsity':>10} {'StaticZeroOnly':>24} {'SparseOnly':>24} {'Hybrid':>24}")
    print(f"  {'-'*132}")
    sparsity_values = []
    for t in targets:
        name = t["name"]
        sd = sparsity_data[name]
        sp = sd["zeros"] / sd["total"] * 100 if sd["total"] > 0 else 0.0
        sparsity_values.append(sp)
        sz_route = route_label_from_backend(t, "staticzero") if sp >= 99.9999 else route_label_from_backend(t, "dense")
        sp_route = route_label_from_backend(t, "sparse")
        hy_backend = hybrid_decisions[name].backend if hybrid_decisions and name in hybrid_decisions else "sparse"
        hy_route = route_label_from_backend(t, hy_backend)
        short = name if len(name) <= 39 else "..." + name[-36:]
        print(f"  {short:<40} {sp:>9.2f}% {sz_route:>24} {sp_route:>24} {hy_route:>24}")
    avg_sparsity = sum(sparsity_values) / max(len(sparsity_values), 1)
    print(f"  {'[骞冲潎]':<40} {avg_sparsity:>9.2f}%")
    return avg_sparsity


def print_dispatch_decision_report(targets, group_sparsity_data, dispatch_decisions):
    print(f"\n  {'Layer':<40} {'AGR':>8} {'TileZR':>8} {'Decision':>12} {'Score':>9} {'Reason':>28}")
    print(f"  {'-'*112}")
    for t in targets:
        name = t["name"]
        gd = group_sparsity_data.get(name, {})
        dec = dispatch_decisions.get(name)
        agr = gd.get('active_group_ratio', dec.agr if dec is not None else -1.0)
        tzr = gd.get('tile_zero_ratio', dec.tzr if dec is not None else -1.0)
        score = dec.score_sparse if dec is not None else 0.0
        reason = dec.reason if dec is not None else 'n/a'
        backend = dec.backend if dec is not None else 'n/a'
        short = name if len(name) <= 39 else "..." + name[-36:]
        agr_s = f"{agr:.4f}" if agr >= 0 else "n/a"
        tzr_s = f"{tzr:.4f}" if tzr >= 0 else "n/a"
        score_s = f"{score:.4f}" if dec is not None else "n/a"
        if len(reason) > 28:
            reason = reason[:25] + '...'
        print(f"  {short:<40} {agr_s:>8} {tzr_s:>8} {backend:>12} {score_s:>9} {reason:>28}")


def measure_mode(model, loader, device, T, warmup, spike_mode, power, label):
    avg_ms, total_ms, num_batches = measure_e2e_latency(
        model, loader, device, T, warmup=warmup, spike_mode=spike_mode)
    energy_j = (total_ms / 1000.0) * power
    return {"label": label, "avg_ms": avg_ms, "total_ms": total_ms,
            "num_batches": num_batches, "energy_j": energy_j}


def print_mode_result(res):
    print(f"    {res['label']:<24}{res['avg_ms']:.2f} ms/batch  (total={res['total_ms']:.1f} ms, {res['num_batches']} batches)")


def build_fourway_models(model_baseline, targets, hybrid_static_zero_layers, hybrid_sparse_set=None):
    if hybrid_static_zero_layers is None:
        hybrid_static_zero_layers = set()
    if hybrid_sparse_set is None:
        hybrid_sparse_set = set()

    model_static_zero_only = copy.deepcopy(model_baseline)
    _, sz_sp, _, sz_sz, sz_dk = replace_model(
        model_static_zero_only, targets, fused=False,
        static_zero_layers=set(hybrid_static_zero_layers), only_static_zero=True)

    model_sparse_only = copy.deepcopy(model_baseline)
    _, so_sp, _, so_sz, so_dk = replace_model(
        model_sparse_only, targets, fused=False, static_zero_layers=set(),
        only_static_zero=False, selective_sparse_set=None)

    model_hybrid = copy.deepcopy(model_baseline)
    _, hy_sp, _, hy_sz, hy_dk = replace_model(
        model_hybrid, targets, fused=False,
        static_zero_layers=set(hybrid_static_zero_layers), only_static_zero=False,
        selective_sparse_set=set(hybrid_sparse_set))

    counts = {
        "static_zero_only": {"num_sparse_conv": sz_sp, "num_static_zero": sz_sz, "num_dense_keep": sz_dk},
        "sparse_only": {"num_sparse_conv": so_sp, "num_static_zero": so_sz, "num_dense_keep": so_dk},
        "hybrid": {"num_sparse_conv": hy_sp, "num_static_zero": hy_sz, "num_dense_keep": hy_dk},
    }
    return model_static_zero_only, model_sparse_only, model_hybrid, counts


def _print_core_target_report(targets):
    print(f"\n  {'Layer':<50} {'Type':<22} {'Spike Source':<38}")
    print(f"  {'-'*114}")
    for t in targets:
        layer = t.conv_name if len(t.conv_name) <= 49 else "..." + t.conv_name[-46:]
        spike = t.spike_name if len(t.spike_name) <= 37 else "..." + t.spike_name[-34:]
        print(f"  {layer:<50} {t.op_type:<22} {spike:<38}")

    op_counter = Counter(t.op_type for t in targets)
    print(f"\n  [Core-AllOps] replaceable layers: {len(targets)}")
    for op_name, count in sorted(op_counter.items()):
        print(f"    - {op_name}: {count}")


def run_core_all_ops_benchmark(args, model_baseline, loader, device, gpu_id):
    print(f"[3/6] Analyze replaceable targets via Core (all operator types) ...")
    imgs_s, _ = next(iter(loader))
    sample_input = make_spike_input(imgs_s[:4], args.T, device, spike_mode=args.spike_mode)

    registry = SpikeOpRegistry.default()
    analyzer = NetworkAnalyzer(registry)
    targets = analyzer.analyze(model_baseline, sample_input=sample_input)

    if not targets:
        print("\n  No replaceable layers found by Core analyzer.")
        return
    _print_core_target_report(targets)

    print(f"\n[4/6] Build replaced model (Core replacer, all ops enabled) ...")
    model_replaced = copy.deepcopy(model_baseline)
    replacer = ModuleReplacer(verbose=True)
    replaced, sparse_count, fused_count, static_zero_count, dense_keep_count = replacer.replace(
        model_replaced,
        targets,
        block_sizes=None,
        static_zero_layers=set(),
        disable_static_zero=True,
        only_static_zero=False,
    )
    print(
        f"\n  Replacement done: total={replaced}, sparse={sparse_count}, "
        f"fused={fused_count}, static_zero={static_zero_count}, dense_keep={dense_keep_count}"
    )

    print(f"\n[5/6] End-to-end latency (warmup={args.warmup}) ...")
    dense_avg, dense_total, dense_n = measure_e2e_latency(
        model_baseline,
        loader,
        device,
        args.T,
        warmup=args.warmup,
        spike_mode=args.spike_mode,
    )
    repl_avg, repl_total, repl_n = measure_e2e_latency(
        model_replaced,
        loader,
        device,
        args.T,
        warmup=args.warmup,
        spike_mode=args.spike_mode,
    )
    speedup = dense_avg / repl_avg if repl_avg > 1e-6 else float("inf")
    dense_energy = (dense_total / 1000.0) * args.power
    repl_energy = (repl_total / 1000.0) * args.power
    energy_saving = (1 - repl_energy / max(dense_energy, 1e-9)) * 100.0

    print(f"  Dense:     {dense_avg:.2f} ms/batch (total={dense_total:.1f} ms, {dense_n} batches)")
    print(f"  Replaced:  {repl_avg:.2f} ms/batch (total={repl_total:.1f} ms, {repl_n} batches)")
    print(f"  Speedup:   {speedup:.3f}x")
    print(f"  Energy save (est.): {energy_saving:.2f}%")

    print(f"\n[6/6] Consistency check ({args.verify_batches} batches) ...")
    avg_cos, avg_agree, max_abs = verify_consistency(
        model_baseline,
        model_replaced,
        loader,
        device,
        args.T,
        num_batches=args.verify_batches,
        spike_mode=args.spike_mode,
    )
    consistency_ok = avg_cos > 0.999 and max_abs < 0.1
    print(f"  Cosine similarity:  {avg_cos:.8f}")
    print(f"  Pred agreement:     {avg_agree * 100:.2f}%")
    print(f"  Max absolute error: {max_abs:.6f}")
    print(f"  Consistency:        {'PASS' if consistency_ok else 'FAIL'}")

    op_counter = Counter(t.op_type for t in targets)
    results = {
        "script_mode": "core_all_ops",
        "model": args.model,
        "dataset": args.dataset,
        "T": args.T,
        "batch_size": args.batch_size,
        "gpu": torch.cuda.get_device_name(gpu_id),
        "weight_init": args.weight_init,
        "seed": args.seed,
        "spike_mode": args.spike_mode,
        "num_targets": len(targets),
        "op_type_counts": dict(sorted(op_counter.items())),
        "replace_summary": {
            "total": replaced,
            "sparse": sparse_count,
            "fused": fused_count,
            "static_zero": static_zero_count,
            "dense_keep": dense_keep_count,
        },
        "dense": {
            "ms_per_batch": round(dense_avg, 4),
            "total_ms": round(dense_total, 4),
            "energy_j": round(dense_energy, 6),
        },
        "replaced": {
            "ms_per_batch": round(repl_avg, 4),
            "total_ms": round(repl_total, 4),
            "energy_j": round(repl_energy, 6),
            "speedup": round(speedup, 6),
            "energy_saving_pct": round(energy_saving, 4),
        },
        "consistency": {
            "cosine_sim": round(avg_cos, 8),
            "pred_agreement_pct": round(avg_agree * 100.0, 4),
            "max_abs_err": round(max_abs, 6),
            "status": "PASS" if consistency_ok else "FAIL",
        },
    }

    out_path = (
        args.out_json
        or args.save_json
        or f"results_core_all_ops_{args.model}_{args.dataset}_T{args.T}_bs{args.batch_size}_{args.weight_init}.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Result JSON saved: {out_path}")


def main():
    global DEVICE

    parser = argparse.ArgumentParser(
        description="SparseFlow four-way E2E benchmark v21-conservative-dispatch")
    parser.add_argument("--model", type=str, default="spiking_resnet18",
                        choices=list(MODEL_BUILDERS.keys()))
    parser.add_argument("--dataset",type=str,default="cifar10",
                        choices=["cifar10", "cifar100", "imagenet_val_flat"],)
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--v_threshold", type=float, default=1.0)
    parser.add_argument("--weight_init", type=str, default="random",
                        choices=["random", "pretrained"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sew_cnf", type=str, default=None,
                        help="SEW-ResNet connect function: ADD, AND, IAND. "
                             "Auto-defaults to ADD for sew_* models.")
    parser.add_argument("--list_models", action="store_true")
    parser.add_argument("--power", type=float, default=250.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--verify_batches", type=int, default=10)
    parser.add_argument("--sparsity_batches", type=int, default=20)
    parser.add_argument("--out_json", type=str, default="", help="Output JSON path")
    parser.add_argument("--save_json", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--spike_mode", type=str, default="normalized_bernoulli",
                        choices=["normalized_bernoulli", "raw_bernoulli", "raw_repeat"])
    parser.add_argument("--collect_diag", action="store_true")
    parser.add_argument("--diag_json", type=str, default="")
    parser.add_argument("--diag_csv", type=str, default="")
    parser.add_argument("--selective_sparse", action="store_true")
    parser.add_argument("--min_cin_for_sparse", type=int, default=128)
    parser.add_argument("--max_agr_for_sparse", type=float, default=0.5)
    parser.add_argument("--no_sparse_1x1", action="store_true")
    parser.add_argument("--min_spatial_size", type=int, default=4)
    # 鈹€鈹€ NEW: layer-level profiling 鈹€鈹€
    parser.add_argument("--layer_profile", action="store_true",
                        help="Measure per-layer timing for baseline vs each sparse variant")
    parser.add_argument("--layer_profile_warmup", type=int, default=5,
                        help="Warmup batches for layer profiling")
    parser.add_argument("--layer_profile_batches", type=int, default=20,
                        help="Measurement batches for layer profiling")


    # --- v25: sync-gating and A/B tile launch ---
    parser.add_argument("--launch_all_tiles", action="store_true",
                        help="Mode B: launch all tiles, zero tiles early-return. "
                             "Default Mode A: build active tile IDs.")
    parser.add_argument("--inference_mode", action="store_true",
                        help="Disable periodic calibration syncs during timing.")
    parser.add_argument("--ab_compare", action="store_true",
                        help="Run both Mode A and Mode B and report comparison.")
    parser.add_argument(
        "--replace_all_ops",
        action="store_true",
        help=(
            "Use Core analyzer/replacer to benchmark all currently supported operator "
            "types (conv1d/conv2d/depthwise/conv3d/linear). "
            "When enabled, run simplified dense-vs-replaced pipeline."
        ),
    )

    args = parser.parse_args()

    if args.list_models:
        print_supported_models(); return

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
                    max_free = free; gpu_id = i

    DEVICE = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(DEVICE)
    device = DEVICE

    title = f"{args.model} | {args.dataset.upper()} | T={args.T} | Four-Way v21-conservative-dispatch"
    print(f"\n{'='*80}")
    print(f"{'SparseFlow Four-Way Benchmark v21-conservative-dispatch':^80}")
    print(f"{title:^80}")
    print(f"{'='*80}")
    print(f"  GPU:          {gpu_id} ({torch.cuda.get_device_name(gpu_id)})")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Spike mode:   {args.spike_mode}")
    print(f"  Weight init:  {args.weight_init}")
    print(f"  Seed:         {args.seed}")
    print(f"  Power (TDP):  {args.power} W")
    print(f"  Min spatial:  {args.min_spatial_size}x{args.min_spatial_size}")
    print(f"  Layer profile: {'ON' if args.layer_profile else 'OFF'}")
    print()

    print(f"[1/7] 鏋勫缓 {args.model} ...")
    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "cifar100":
        num_classes = 100
    elif args.dataset == "imagenet_val_flat":
        num_classes = 1000
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    model_baseline = build_model(
        args.model, device, args.v_threshold,
        weight_init=args.weight_init, sew_cnf=args.sew_cnf,
        num_classes=num_classes, seed=args.seed)

    print(f"[2/7] 鍔犺浇 {args.dataset} 娴嬭瘯闆?...")
    ds = build_dataset(args.dataset, args.data_root, spike_mode=args.spike_mode)
    # loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print(f"  娴嬭瘯闆? {len(ds)} 寮? {len(loader)} batches, {num_classes} classes")

    if args.replace_all_ops:
        run_core_all_ops_benchmark(args, model_baseline, loader, device, gpu_id)
        return

    print(f"[3/7] 鍒嗘瀽缃戠粶鎷撴墤 ...")
    imgs_s, _ = next(iter(loader))
    sample_input = make_spike_input(imgs_s[:4], args.T, device, spike_mode=args.spike_mode)
    targets, skipped = analyze_targets(
        model_baseline, sample_input, device, fused=False,
        min_spatial_size=args.min_spatial_size)
    print_analysis_report(targets, skipped, fused=False)
    if not targets:
        print("\n  No replaceable target layers found, exiting.")
        return

    print(f"\n[4/7] 缁熻绋€鐤忕巼涓庤皟搴︾壒寰?({args.sparsity_batches} batches) ...")
    sparsity_data = measure_sparsity(
        model_baseline, targets, loader, device, args.T,
        num_batches=args.sparsity_batches, spike_mode=args.spike_mode)
    zero_layers = collect_static_zero_layers(targets, sparsity_data)
    print(f"\n  Detected {len(zero_layers)} exact-zero-input layers.")

    print(f"\n  [Dispatch] Measuring group/tile sparsity for hybrid routing ...")
    _tmp_model = copy.deepcopy(model_baseline)
    replace_model(_tmp_model, targets, fused=False, static_zero_layers=set(), only_static_zero=False)
    group_sparsity_data = measure_group_sparsity(
        _tmp_model, targets, loader, device, args.T,
        num_batches=min(args.sparsity_batches, 5), spike_mode=args.spike_mode)
    del _tmp_model

    for _t in targets:
        _n = _t["name"]
        if _n in zero_layers:
            _gd_existing = group_sparsity_data.get(_n, {})
            group_sparsity_data[_n] = make_synthetic_zero_diag(
                layer_name=_n, source='staticzero',
                total_group_count=_gd_existing.get('total_group_count', -1.0),
                total_tile_count=_gd_existing.get('total_tile_count', -1.0))

    for _t in targets:
        _n = _t["name"]
        if _n in zero_layers:
            continue
        _gd = group_sparsity_data.get(_n, {})
        if _gd.get('active_group_ratio', -1.0) >= 0:
            continue
        _sd = sparsity_data.get(_n, {"zeros": 0, "total": 1})
        _elem_sp = _sd["zeros"] / max(_sd["total"], 1)
        if _elem_sp >= 0.9999:
            _gd_existing = group_sparsity_data.get(_n, {})
            group_sparsity_data[_n] = make_synthetic_zero_diag(
                layer_name=_n, source='zero_fastpath',
                total_group_count=_gd_existing.get('total_group_count', -1.0),
                total_tile_count=_gd_existing.get('total_tile_count', -1.0))

    legacy_sparse_candidate_set = None
    if args.selective_sparse:
        legacy_sparse_candidate_set = set()
        for t in targets:
            name = t["name"]
            if name in zero_layers:
                continue
            if should_use_sparse_selective(
                t, group_sparsity_data,
                min_cin=args.min_cin_for_sparse,
                max_agr=args.max_agr_for_sparse,
                no_1x1=args.no_sparse_1x1):
                legacy_sparse_candidate_set.add(name)
        print(f"\n  [Legacy selective filter] {len(legacy_sparse_candidate_set)}/{len(targets)} layers allowed for sparse")

    dispatch_decisions = dispatch_all_layers(targets, group_sparsity_data, zero_layers=set(zero_layers))

    hybrid_static_zero_layers, hybrid_sparse_set = decisions_to_sets(dispatch_decisions)

    # Safety net: StaticZero must remain exact-zero only. If a non-exact layer somehow
    # slips through as staticzero, remap it to sparse instead of producing wrong zeros.
    invalid_static_zero_layers = set(hybrid_static_zero_layers) - set(zero_layers)
    if invalid_static_zero_layers:
        for _name in sorted(invalid_static_zero_layers):
            _dec = dispatch_decisions[_name]
            _dec.backend = 'sparse'
            _dec.reason = 'bench_safety_remap_nonexact_staticzero_to_sparse'
        hybrid_static_zero_layers -= invalid_static_zero_layers
        hybrid_sparse_set |= invalid_static_zero_layers

    if legacy_sparse_candidate_set is not None:
        hybrid_sparse_set &= legacy_sparse_candidate_set
        for _name, _dec in dispatch_decisions.items():
            if _dec.backend == 'sparse' and _name not in hybrid_sparse_set:
                _dec.backend = 'dense'
                _dec.reason = 'legacy_selective_filter_keep_dense'

    # Re-sync sets from final decisions after all remaps / legacy filters.
    hybrid_static_zero_layers, hybrid_sparse_set = decisions_to_sets(dispatch_decisions)

    avg_sparsity = print_fourway_route_report(targets, sparsity_data, hybrid_decisions=dispatch_decisions)

    print(f"\n  {'Layer':<40} {'AGR':>8} {'TileZR':>8} {'Zero':>6} {'Sparse':>7} {'Dense':>7} {'Source':>10}")
    print(f"  {'-'*90}")
    for t in targets:
        name = t["name"]
        gd = group_sparsity_data.get(name, {})
        agr = gd.get('active_group_ratio', -1.0)
        tzr = gd.get('tile_zero_ratio', -1.0)
        zt = gd.get('zero_tiles', -1)
        st = gd.get('sparse_tiles', -1)
        dt = gd.get('denseish_tiles', -1)
        _synth = gd.get('_synthetic', False)
        _src = gd.get('_diag_path', 'measured') if _synth else ('measured' if agr >= 0 else 'unavail')
        short = name if len(name) <= 39 else "..." + name[-36:]
        agr_s = f"{agr:.4f}" if agr >= 0 else "n/a"
        tzr_s = f"{tzr:.4f}" if tzr >= 0 else "n/a"
        zt_s = str(zt) if zt >= 0 else "n/a"
        st_s = str(st) if st >= 0 else "n/a"
        dt_s = str(dt) if dt >= 0 else "n/a"
        print(f"  {short:<40} {agr_s:>8} {tzr_s:>8} {zt_s:>6} {st_s:>7} {dt_s:>7} {_src:>10}")

    print_dispatch_decision_report(targets, group_sparsity_data, dispatch_decisions)

    model_static_zero_only, model_sparse_only, model_hybrid, route_counts = build_fourway_models(
        model_baseline, targets, hybrid_static_zero_layers, hybrid_sparse_set=hybrid_sparse_set)
    print(f"\n  鏇挎崲瀹屾垚:")
    for mode_name, rc in route_counts.items():
        print(f"    {mode_name}: {rc['num_sparse_conv']} Sparse + {rc['num_static_zero']} StaticZero + {rc['num_dense_keep']} DenseKeep")


    # 鈹€鈹€ v25: timing preparation 鈹€鈹€
    if args.inference_mode:
        n_so = prepare_for_timing(model_sparse_only)
        n_hy = prepare_for_timing(model_hybrid)
        print(f"  [v25] inference_mode: {n_so} sparse_only + {n_hy} hybrid modules configured")

    if args.launch_all_tiles:
        set_launch_mode(model_sparse_only, launch_all=True)
        set_launch_mode(model_hybrid, launch_all=True)
        print(f"  [v25] launch_all_tiles=True (Mode B)")

    _so_state = count_sync_state(model_sparse_only)
    _hy_state = count_sync_state(model_hybrid)
    print(f"  [v25] sparse_only sync state: {_so_state}")
    print(f"  [v25] hybrid sync state: {_hy_state}")


    print(f"\n[5/7] 绔埌绔欢杩?(warmup={args.warmup}) ...")
    dense_res = measure_mode(model_baseline, loader, device, args.T, args.warmup, args.spike_mode, args.power, "Dense cuDNN")
    sz_res = measure_mode(model_static_zero_only, loader, device, args.T, args.warmup, args.spike_mode, args.power, "StaticZero only")
    so_res = measure_mode(model_sparse_only, loader, device, args.T, args.warmup, args.spike_mode, args.power, "SparseConv only")
    hy_res = measure_mode(model_hybrid, loader, device, args.T, args.warmup, args.spike_mode, args.power, "SparseConv + StaticZero")
    for r in [dense_res, sz_res, so_res, hy_res]:
        print_mode_result(r)

    sz_speedup = dense_res["avg_ms"] / sz_res["avg_ms"] if sz_res["avg_ms"] > 1e-6 else float("inf")
    so_speedup = dense_res["avg_ms"] / so_res["avg_ms"] if so_res["avg_ms"] > 1e-6 else float("inf")
    hy_speedup = dense_res["avg_ms"] / hy_res["avg_ms"] if hy_res["avg_ms"] > 1e-6 else float("inf")
    sz_esave = (1 - sz_res["energy_j"] / max(dense_res["energy_j"], 1e-9)) * 100
    so_esave = (1 - so_res["energy_j"] / max(dense_res["energy_j"], 1e-9)) * 100
    hy_esave = (1 - hy_res["energy_j"] / max(dense_res["energy_j"], 1e-9)) * 100

    print(f"\n  Speedup SZ-only: {sz_speedup:.3f}x  Sparse-only: {so_speedup:.3f}x  Hybrid: {hy_speedup:.3f}x")
    print(f"  Energy  SZ-only: {sz_esave:.2f}%  Sparse-only: {so_esave:.2f}%  Hybrid: {hy_esave:.2f}%")

    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    # [6/7] 閫愬眰鍔犻€熷垎鏋?(Layer-level profiling)
    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    layer_profile_data = {}
    if args.layer_profile:
        lp_warmup = args.layer_profile_warmup
        lp_batches = args.layer_profile_batches
        target_names = [t["name"] for t in targets]

        print(f"\n[6/7] 閫愬眰鍔犻€熷垎鏋?(warmup={lp_warmup}, batches={lp_batches}) ...")

        # 6a) Baseline (Dense cuDNN) per-layer timing
        print(f"  Measuring baseline (Dense cuDNN) per-layer timing ...")
        baseline_layer_timing = measure_layer_timing(
            model_baseline, loader, device, args.T, target_names,
            warmup=lp_warmup, num_batches=lp_batches, spike_mode=args.spike_mode)

        # 6b) SparseConv-only per-layer timing
        print(f"  Measuring SparseConv-only per-layer timing ...")
        # For sparse models, the replaced layer names still exist but the module
        # type is now SparseConv2d / StaticZeroConv2d 鈥?hooks work the same way.
        sparse_only_layer_timing = measure_layer_timing(
            model_sparse_only, loader, device, args.T, target_names,
            warmup=lp_warmup, num_batches=lp_batches, spike_mode=args.spike_mode)

        # 6c) Hybrid per-layer timing
        print(f"  Measuring Hybrid per-layer timing ...")
        hybrid_layer_timing = measure_layer_timing(
            model_hybrid, loader, device, args.T, target_names,
            warmup=lp_warmup, num_batches=lp_batches, spike_mode=args.spike_mode)

        # Print SparseConv-only layer profile
        print(f"\n  {'=' * 77}")
        print(f"  {'Per-layer Compare: Dense cuDNN vs SparseConv-only':^77}")
        print(f"  {'=' * 77}")
        print_layer_profile(baseline_layer_timing, sparse_only_layer_timing, targets,
                            e2e_baseline_ms=dense_res["avg_ms"],
                            e2e_sparse_ms=so_res["avg_ms"],
                            label="SparseOnly")

        # Print Hybrid layer profile
        print(f"\n  {'=' * 77}")
        print(f"  {'Per-layer Compare: Dense cuDNN vs Hybrid (Sparse+StaticZero)':^77}")
        print(f"  {'=' * 77}")
        print_layer_profile(baseline_layer_timing, hybrid_layer_timing, targets,
                            e2e_baseline_ms=dense_res["avg_ms"],
                            e2e_sparse_ms=hy_res["avg_ms"],
                            label="Hybrid")

        # Store for JSON output
        layer_profile_data = {
            "baseline_layer_ms": {k: round(v, 4) for k, v in baseline_layer_timing.items()},
            "sparse_only_layer_ms": {k: round(v, 4) for k, v in sparse_only_layer_timing.items()},
            "hybrid_layer_ms": {k: round(v, 4) for k, v in hybrid_layer_timing.items()},
            "per_layer_speedup_sparse_only": {},
            "per_layer_speedup_hybrid": {},
        }
        for name in target_names:
            b = baseline_layer_timing.get(name, 0.0)
            s_so = sparse_only_layer_timing.get(name, 0.0)
            s_hy = hybrid_layer_timing.get(name, 0.0)
            layer_profile_data["per_layer_speedup_sparse_only"][name] = (
                round(b / s_so, 4) if s_so > 1e-6 else float("inf"))
            layer_profile_data["per_layer_speedup_hybrid"][name] = (
                round(b / s_hy, 4) if s_hy > 1e-6 else float("inf"))
    else:
        print(f"\n[6/7] 閫愬眰鍔犻€熷垎鏋?... SKIPPED (use --layer_profile to enable)")

    print(f"\n[7/7] 涓€鑷存€ч獙璇?({args.verify_batches} batches) ...")
    sz_cos, sz_agr, sz_mabs = verify_consistency(model_baseline, model_static_zero_only, loader, device, args.T, num_batches=args.verify_batches, spike_mode=args.spike_mode)
    so_cos, so_agr, so_mabs = verify_consistency(model_baseline, model_sparse_only, loader, device, args.T, num_batches=args.verify_batches, spike_mode=args.spike_mode)
    hy_cos, hy_agr, hy_mabs = verify_consistency(model_baseline, model_hybrid, loader, device, args.T, num_batches=args.verify_batches, spike_mode=args.spike_mode)
    sz_ok = sz_cos > 0.999 and sz_mabs < 0.1
    so_ok = so_cos > 0.999 and so_mabs < 0.1
    hy_ok = hy_cos > 0.999 and hy_mabs < 0.1
    print(f"  SZ-only:  cos={sz_cos:.8f}  agree={sz_agr*100:.2f}%  max_abs={sz_mabs:.6f}  {'PASS' if sz_ok else 'FAIL'}")
    print(f"  Sparse:   cos={so_cos:.8f}  agree={so_agr*100:.2f}%  max_abs={so_mabs:.6f}  {'PASS' if so_ok else 'FAIL'}")
    print(f"  Hybrid:   cos={hy_cos:.8f}  agree={hy_agr*100:.2f}%  max_abs={hy_mabs:.6f}  {'PASS' if hy_ok else 'FAIL'}")

    print(f"\n{'='*96}")
    print(f"{'FOUR-WAY SUMMARY v21-conservative-dispatch':^96}")
    print(f"{'='*96}")
    print(f"  Model: {args.model}  Dataset: {args.dataset}  T={args.T}  Spike: {args.spike_mode}")
    print(f"  Avg Sparsity: {avg_sparsity:.2f}%  Targets: {len(targets)}  Skipped: {len(skipped)}")
    print(f"\n  {'Mode':<28} {'Latency(ms)':>12} {'Speedup':>10} {'EnergySave':>12} {'Consistency':>12}")
    print(f"  {'-'*80}")
    print(f"  {'Dense cuDNN':<28} {dense_res['avg_ms']:>12.2f} {'1.000x':>10} {'0.00%':>12} {'REF':>12}")
    print(f"  {'StaticZero only':<28} {sz_res['avg_ms']:>12.2f} {sz_speedup:>9.3f}x {sz_esave:>11.2f}% {('PASS' if sz_ok else 'FAIL'):>12}")
    print(f"  {'SparseConv only':<28} {so_res['avg_ms']:>12.2f} {so_speedup:>9.3f}x {so_esave:>11.2f}% {('PASS' if so_ok else 'FAIL'):>12}")
    print(f"  {'SparseConv + StaticZero':<28} {hy_res['avg_ms']:>12.2f} {hy_speedup:>9.3f}x {hy_esave:>11.2f}% {('PASS' if hy_ok else 'FAIL'):>12}")
    print(f"{'='*96}\n")


    # 鈹€鈹€ v25: A/B tile launch comparison 鈹€鈹€
    if args.ab_compare:
        print(f"\n[A/B] Tile launch mode comparison ...")
        prepare_for_timing(model_sparse_only)
        prepare_for_timing(model_hybrid)

        set_launch_mode(model_sparse_only, launch_all=False)
        set_launch_mode(model_hybrid, launch_all=False)
        so_a = measure_mode(model_sparse_only, loader, device, args.T, args.warmup, args.spike_mode, args.power, "Sparse ModeA")
        hy_a = measure_mode(model_hybrid, loader, device, args.T, args.warmup, args.spike_mode, args.power, "Hybrid ModeA")

        set_launch_mode(model_sparse_only, launch_all=True)
        set_launch_mode(model_hybrid, launch_all=True)
        so_b = measure_mode(model_sparse_only, loader, device, args.T, args.warmup, args.spike_mode, args.power, "Sparse ModeB")
        hy_b = measure_mode(model_hybrid, loader, device, args.T, args.warmup, args.spike_mode, args.power, "Hybrid ModeB")

        print(f"\n  {'Config':<28} {'Latency(ms)':>12}")
        print(f"  {'-'*44}")
        for r in [so_a, so_b, hy_a, hy_b]:
            print_mode_result(r)
        so_diff = so_a["avg_ms"] - so_b["avg_ms"]
        hy_diff = hy_a["avg_ms"] - hy_b["avg_ms"]
        print(f"\n  Sparse: ModeA - ModeB = {so_diff:+.2f} ms  (positive = ModeB faster)")
        print(f"  Hybrid: ModeA - ModeB = {hy_diff:+.2f} ms  (positive = ModeB faster)")

    results = {
        "script_version": "v21-conservative-dispatch",
        "model": args.model, "dataset": args.dataset, "T": args.T,
        "batch_size": args.batch_size, "gpu": torch.cuda.get_device_name(gpu_id),
        "spike_mode": args.spike_mode, "weight_init": args.weight_init, "seed": args.seed,
        "avg_sparsity_pct": round(avg_sparsity, 2),
        "num_targets": len(targets), "num_skipped": len(skipped),
        "num_zero_layers": len(zero_layers),
        "static_zero_layers": sorted(list(zero_layers)),
        "route_counts": route_counts,
        "hybrid_static_zero_layers": sorted(list(hybrid_static_zero_layers)),
        "hybrid_sparse_layers": sorted(list(hybrid_sparse_set)),
        "hybrid_dense_keep_layers": sorted([t["name"] for t in targets if t["name"] not in hybrid_static_zero_layers and t["name"] not in hybrid_sparse_set]),
        "dispatch_decisions": {name: dec.to_dict() for name, dec in dispatch_decisions.items()},
        "dispatch_invalid_staticzero_remapped_layers": sorted(list(invalid_static_zero_layers)) if "invalid_static_zero_layers" in locals() else [],
        "dense": {"ms_per_batch": round(dense_res["avg_ms"], 2), "energy_j": round(dense_res["energy_j"], 4)},
        "static_zero_only": {"ms_per_batch": round(sz_res["avg_ms"], 2), "speedup": round(sz_speedup, 4), "energy_saving_pct": round(sz_esave, 2), "cosine_sim": round(sz_cos, 8), "consistency": "PASS" if sz_ok else "FAIL"},
        "sparse_only": {"ms_per_batch": round(so_res["avg_ms"], 2), "speedup": round(so_speedup, 4), "energy_saving_pct": round(so_esave, 2), "cosine_sim": round(so_cos, 8), "consistency": "PASS" if so_ok else "FAIL"},
        "hybrid": {"ms_per_batch": round(hy_res["avg_ms"], 2), "speedup": round(hy_speedup, 4), "energy_saving_pct": round(hy_esave, 2), "cosine_sim": round(hy_cos, 8), "consistency": "PASS" if hy_ok else "FAIL"},
    }
    # 鈹€鈹€ NEW: include layer profile data in JSON 鈹€鈹€
    if layer_profile_data:
        results["layer_profile"] = layer_profile_data

    if group_sparsity_data:
        results["group_sparsity"] = group_sparsity_data

    if args.collect_diag and (args.diag_json or args.diag_csv):
        run_id = f"{args.model}_{args.dataset}_T{args.T}_fourway_v21_conservative_dispatch"
        layer_logger = LayerLogger(run_id=run_id, model=args.model, dataset=args.dataset, T=args.T)
        for t in targets:
            name = t["name"]
            sd = sparsity_data.get(name, {"zeros": 0, "total": 1})
            elem_sp = sd["zeros"] / max(sd["total"], 1)
            gd = group_sparsity_data.get(name, {})
            ishape = t.get("input_shape")
            _is_sz = (name in zero_layers)
            _is_dk = (name not in hybrid_sparse_set and not _is_sz)

            if _is_sz:
                layer_logger.log_static_zero(layer_name=name, input_shape=ishape)
            elif _is_dk:
                layer_logger.log_dense(layer_name=name, input_shape=ishape, element_sparsity=elem_sp)
            else:
                _has_diag = gd.get('active_group_ratio', -1) >= 0
                layer_logger.log_layer(
                    layer_name=name, mode_used="sparseconv", replaced=True,
                    sparse_path_executed=_has_diag,
                    input_shape=str(ishape) if ishape else "",
                    element_sparsity=elem_sp,
                    nonzero_group_count=gd.get('nonzero_group_count', -1.0),
                    total_group_count=gd.get('total_group_count', -1.0),
                    active_group_ratio=gd.get('active_group_ratio', -1.0),
                    tile_zero_count=gd.get('tile_zero_count', -1.0),
                    total_tile_count=gd.get('total_tile_count', -1.0),
                    tile_zero_ratio=gd.get('tile_zero_ratio', -1.0),
                    effective_k_ratio=gd.get('effective_k_ratio', -1.0),
                    sparse_compute_ms=gd.get('sparse_compute_ms', -1.0),
                    sparse_total_ms=gd.get('sparse_total_ms', -1.0))
        if args.diag_json:
            layer_logger.save_json(args.diag_json)
            print(f"  璇婃柇 JSON: {args.diag_json}")
        if args.diag_csv:
            layer_logger.save_csv(args.diag_csv)
            print(f"  璇婃柇 CSV: {args.diag_csv}")
        layer_logger.print_summary()

    out_path = args.out_json or args.save_json or f"results_fourway_{args.model}_{args.dataset}_T{args.T}_bs{args.batch_size}_{args.weight_init}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  缁撴灉: {out_path}")


if __name__ == "__main__":
    main()
