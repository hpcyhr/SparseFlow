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
try:
    from Models.spikformer_github import MODEL_BUILDERS as SPIKFORMER_MODEL_BUILDERS
except Exception:
    SPIKFORMER_MODEL_BUILDERS = {}

try:
    from Models.sdtv1_github import MODEL_BUILDERS as SDTV1_MODEL_BUILDERS
except Exception:
    SDTV1_MODEL_BUILDERS = {}

try:
    from Models.qkformer_github import MODEL_BUILDERS as QKFORMER_MODEL_BUILDERS
except Exception:
    QKFORMER_MODEL_BUILDERS = {}


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
BENCH_VERSION = "v29-observability-cleanup"


_EXTERNAL_MODEL_SOURCES = [
    ("Models.spikformer_github", SPIKFORMER_MODEL_BUILDERS),
    ("Models.sdtv1_github", SDTV1_MODEL_BUILDERS),
    ("Models.qkformer_github", QKFORMER_MODEL_BUILDERS),
]


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

    for module_name, model_builders in _EXTERNAL_MODEL_SOURCES:
        for name, fn in model_builders.items():
            builders[name] = fn
            metadata[name] = {"family": "external_transformer", "module": module_name}
    return builders, metadata


MODEL_BUILDERS, MODEL_META = _discover_model_builders()


def print_supported_models():
    print("\nSupported models:")
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
                T=None,
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
    # Only external transformer builders consume T at model-construction time.
    meta = MODEL_META.get(model_name, {})
    if meta.get("family") == "external_transformer":
        builder_kwargs["T"] = T
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

        # Benchmark-only dataset: label is a dummy value.
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
        print(f"\n  Replaced layers / baseline e2e: {100.0 * total_base / e2e_baseline_ms:.2f}%")
    if e2e_sparse_ms is not None and e2e_sparse_ms > 1e-6:
        print(f"  Replaced layers / {label} e2e:   {100.0 * total_sparse / e2e_sparse_ms:.2f}%")

    saved_ms = total_base - total_sparse
    print(f"  Replaced-layer saved latency: {saved_ms:.3f} ms/batch")

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
    print(f"  {'[AVG]':<40} {avg_sparsity:>9.2f}%")
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


def _extract_runtime_dispatch_disagreement(model):
    summary = {
        "modules_with_runtime_dispatch": 0,
        "modules_with_observed_dispatch": 0,
        "total_seen": 0,
        "total_mismatch": 0,
        "mismatch_rate": 0.0,
    }
    per_module = {}
    for name, mod in model.named_modules():
        seen = int(getattr(mod, "_runtime_dispatch_seen", 0))
        mismatch = int(getattr(mod, "_runtime_dispatch_mismatch", 0))
        if hasattr(mod, "_runtime_dispatch_seen") or hasattr(mod, "_runtime_dispatch_mismatch"):
            summary["modules_with_runtime_dispatch"] += 1
        if seen > 0:
            summary["modules_with_observed_dispatch"] += 1
            summary["total_seen"] += seen
            summary["total_mismatch"] += mismatch
            per_module[name] = {
                "seen": seen,
                "mismatch": mismatch,
                "mismatch_rate": float(mismatch / max(seen, 1)),
            }
    if summary["total_seen"] > 0:
        summary["mismatch_rate"] = float(summary["total_mismatch"] / summary["total_seen"])
    return summary, per_module


def _empty_runtime_hint_summary():
    return {
        "requested_dense_layers": 0,
        "requested_zero_layers": 0,
        "forced_dense": 0,
        "forced_zero": 0,
        "missing_module": 0,
        "not_hintable": 0,
    }


def _apply_runtime_backend_hints(
    model,
    force_dense_layers=None,
    force_zero_layers=None,
    verbose=True,
    label="",
):
    """Apply runtime backend hints to sparse modules.

    This does NOT change the replacement topology; it only pre-sets sparse module
    runtime policy flags (`_force_dense` / `_force_zero`) where supported.
    """
    dense_layers = set(force_dense_layers or [])
    zero_layers = set(force_zero_layers or [])
    # Zero hint has priority if overlap exists.
    dense_layers -= zero_layers

    summary = _empty_runtime_hint_summary()
    summary["requested_dense_layers"] = len(dense_layers)
    summary["requested_zero_layers"] = len(zero_layers)

    modules = dict(model.named_modules())
    for name in sorted(dense_layers | zero_layers):
        mod = modules.get(name)
        if mod is None:
            summary["missing_module"] += 1
            continue

        want_zero = name in zero_layers
        want_dense = name in dense_layers and not want_zero
        touched = False

        if hasattr(mod, "_force_zero"):
            mod._force_zero = bool(want_zero)
            touched = True
        if hasattr(mod, "_force_dense"):
            mod._force_dense = bool(want_dense and not want_zero)
            touched = True
        if hasattr(mod, "_warmup_left"):
            mod._warmup_left = 0
            touched = True
        if hasattr(mod, "_ema_active_ratio"):
            if want_zero:
                mod._ema_active_ratio = 0.0
            elif want_dense:
                thr = float(getattr(mod, "_dense_threshold", 0.85))
                mod._ema_active_ratio = thr + 1e-3
            touched = True

        if not touched:
            summary["not_hintable"] += 1
            continue

        if want_zero:
            summary["forced_zero"] += 1
        elif want_dense:
            summary["forced_dense"] += 1

    if verbose:
        prefix = f"  [{label}] " if label else "  "
        print(
            prefix
            + "runtime hints: "
            + f"force_dense={summary['forced_dense']}, "
            + f"force_zero={summary['forced_zero']}, "
            + f"missing={summary['missing_module']}, "
            + f"not_hintable={summary['not_hintable']}"
        )
    return summary


def build_fourway_models(
    model_baseline,
    targets,
    hybrid_static_zero_layers,
    hybrid_sparse_set=None,
    sparse_only_force_dense_layers=None,
    sparse_only_force_zero_layers=None,
):
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
    so_hint = _apply_runtime_backend_hints(
        model_sparse_only,
        force_dense_layers=sparse_only_force_dense_layers,
        force_zero_layers=sparse_only_force_zero_layers,
        verbose=True,
        label="Sparse-only",
    )

    model_hybrid = copy.deepcopy(model_baseline)
    _, hy_sp, _, hy_sz, hy_dk = replace_model(
        model_hybrid, targets, fused=False,
        static_zero_layers=set(hybrid_static_zero_layers), only_static_zero=False,
        selective_sparse_set=set(hybrid_sparse_set))

    counts = {
        "static_zero_only": {"num_sparse_conv": sz_sp, "num_static_zero": sz_sz, "num_dense_keep": sz_dk},
        "sparse_only": {
            "num_sparse_conv": so_sp,
            "num_static_zero": so_sz,
            "num_dense_keep": so_dk,
            "runtime_force_dense": so_hint["forced_dense"],
            "runtime_force_zero": so_hint["forced_zero"],
        },
        "hybrid": {"num_sparse_conv": hy_sp, "num_static_zero": hy_sz, "num_dense_keep": hy_dk},
    }
    return model_static_zero_only, model_sparse_only, model_hybrid, counts, so_hint


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


def _normalize_core_targets_for_fuse_mode(targets, fuse_conv_lif):
    """
    Analyzer may still mark conv targets as fused_conv* based on topology.
    The current Core replacer no longer constructs fused conv+LIF modules, so
    bench_4test always normalizes those tags back to plain conv2d_* targets.
    """

    remap = {
        "fused_conv3x3_lif": "conv2d_3x3",
        "fused_conv3x3s2_lif": "conv2d_3x3_s2",
        "fused_conv1x1_lif": "conv2d_1x1",
    }

    changed = 0
    for t in targets:
        old_op = getattr(t, "op_type", "")
        new_op = remap.get(old_op, old_op)
        if new_op == old_op:
            continue
        t.op_type = new_op
        # Clear fused-only payload to avoid accidental downstream confusion.
        if hasattr(t, "lif_name"):
            t.lif_name = None
        if hasattr(t, "lif_module"):
            t.lif_module = None
        if hasattr(t, "bn_name"):
            t.bn_name = None
        if hasattr(t, "bn_module"):
            t.bn_module = None
        changed += 1
    return changed


_CORE_STATICZERO_ELIGIBLE_OPS = {
    "conv2d_3x3",
    "conv2d_1x1",
    "conv2d_3x3_s2",
    "depthwise_conv2d",
    "fused_conv3x3_lif",
    "fused_conv1x1_lif",
    "fused_conv3x3s2_lif",
    "linear",
}


def _is_core_staticzero_eligible(target):
    return getattr(target, "op_type", "") in _CORE_STATICZERO_ELIGIBLE_OPS


def _core_target_name(target):
    return getattr(target, "conv_name", "")


def _core_target_op_type(target):
    return getattr(target, "op_type", "unknown")


def _safe_speedup(base_ms, opt_ms):
    if opt_ms <= 1e-9:
        return float("inf")
    return base_ms / opt_ms


def _fmt_speedup(x):
    if x is None:
        return "n/a"
    if isinstance(x, float) and math.isnan(x):
        return "n/a"
    if not math.isfinite(x):
        return "inf"
    return f"{x:.3f}x"


def _select_microbench_input(sample_input, max_batch=2):
    if not isinstance(sample_input, torch.Tensor):
        return sample_input
    if sample_input.dim() >= 2:
        keep = max(1, min(int(max_batch), int(sample_input.shape[1])))
        if keep != int(sample_input.shape[1]):
            return sample_input[:, :keep].contiguous()
    return sample_input


def _clone_benchmark_arg(arg):
    if isinstance(arg, torch.Tensor):
        return arg.clone()
    if isinstance(arg, tuple):
        return tuple(_clone_benchmark_arg(x) for x in arg)
    if isinstance(arg, list):
        return [_clone_benchmark_arg(x) for x in arg]
    return arg


def _capture_module_input(model, sample_input, target_name):
    captured = {}
    module = _get_module_by_name(model, target_name)

    def hook(_, inp):
        if "args" in captured:
            return
        args = []
        for item in inp:
            if isinstance(item, torch.Tensor):
                args.append(item.detach().contiguous())
            else:
                args.append(item)
        captured["args"] = tuple(args)

    handle = module.register_forward_pre_hook(hook)
    try:
        sj_func.reset_net(model)
        with torch.no_grad():
            _ = model(sample_input)
    finally:
        handle.remove()
    return captured.get("args")


def _prepare_module_for_microbench(module):
    prepare_for_timing(module)
    for _, submodule in module.named_modules():
        if hasattr(submodule, "collect_diag"):
            submodule.collect_diag = False
        if hasattr(submodule, "profile_runtime"):
            submodule.profile_runtime = False
        if hasattr(submodule, "return_ms"):
            submodule.return_ms = False


def _snapshot_module_runtime_flags(module):
    snapshot = []
    for _, submodule in module.named_modules():
        state = {}
        for attr in ("collect_diag", "profile_runtime", "return_ms", "_inference_mode"):
            if hasattr(submodule, attr):
                state[attr] = getattr(submodule, attr)
        snapshot.append((submodule, state))
    return snapshot


def _restore_module_runtime_flags(snapshot):
    for submodule, state in snapshot:
        if "_inference_mode" in state:
            if hasattr(submodule, "set_inference_mode"):
                submodule.set_inference_mode(bool(state["_inference_mode"]))
            else:
                submodule._inference_mode = bool(state["_inference_mode"])
        for attr in ("collect_diag", "profile_runtime", "return_ms"):
            if attr in state:
                setattr(submodule, attr, state[attr])


def _measure_isolated_module_latency(module, input_args, warmup=8, iters=30):
    warmup = max(0, int(warmup))
    iters = max(1, int(iters))
    snapshot = _snapshot_module_runtime_flags(module)
    _prepare_module_for_microbench(module)

    try:
        for _ in range(warmup):
            args = _clone_benchmark_arg(input_args)
            sj_func.reset_net(module)
            with torch.no_grad():
                _ = module(*args)
        sync()

        total_ms = 0.0
        for _ in range(iters):
            args = _clone_benchmark_arg(input_args)
            sj_func.reset_net(module)
            sync()
            start_evt = make_event()
            end_evt = make_event()
            start_evt.record()
            with torch.no_grad():
                _ = module(*args)
            end_evt.record()
            sync()
            total_ms += start_evt.elapsed_time(end_evt)
        return total_ms / max(iters, 1)
    finally:
        _restore_module_runtime_flags(snapshot)


def _build_replaced_speedup_summary(targets, baseline_timing, opt_timing, name_getter, op_getter):
    per_op = defaultdict(
        lambda: {
            "count": 0,
            "base_sum_ms": 0.0,
            "opt_sum_ms": 0.0,
            "layer_speedup_sum": 0.0,
            "layer_speedup_count": 0,
        }
    )
    all_stat = {
        "count": 0,
        "base_sum_ms": 0.0,
        "opt_sum_ms": 0.0,
        "layer_speedup_sum": 0.0,
        "layer_speedup_count": 0,
    }

    for t in targets:
        name = name_getter(t)
        op = op_getter(t)
        b = float(baseline_timing.get(name, 0.0))
        s = float(opt_timing.get(name, 0.0))

        st = per_op[op]
        st["count"] += 1
        st["base_sum_ms"] += b
        st["opt_sum_ms"] += s

        all_stat["count"] += 1
        all_stat["base_sum_ms"] += b
        all_stat["opt_sum_ms"] += s

        if s > 1e-6:
            sp = b / s
            st["layer_speedup_sum"] += sp
            st["layer_speedup_count"] += 1
            all_stat["layer_speedup_sum"] += sp
            all_stat["layer_speedup_count"] += 1

    def _finalize(st):
        if st["count"] == 0:
            weighted = None
        else:
            weighted = _safe_speedup(st["base_sum_ms"], st["opt_sum_ms"])
        mean_layer = (
            st["layer_speedup_sum"] / st["layer_speedup_count"]
            if st["layer_speedup_count"] > 0
            else None
        )
        return {
            "count": int(st["count"]),
            "base_sum_ms": round(st["base_sum_ms"], 6),
            "opt_sum_ms": round(st["opt_sum_ms"], 6),
            "weighted_speedup": (
                round(weighted, 6) if weighted is not None and math.isfinite(weighted)
                else (float("inf") if weighted is not None else None)
            ),
            "mean_layer_speedup": (
                round(mean_layer, 6) if mean_layer is not None and math.isfinite(mean_layer)
                else (float("inf") if mean_layer is not None else None)
            ),
            "valid_layer_speedup_count": int(st["layer_speedup_count"]),
        }

    by_op = {k: _finalize(v) for k, v in sorted(per_op.items(), key=lambda kv: kv[0])}
    return {
        "all_replaced": _finalize(all_stat),
        "by_op_type": by_op,
    }


def _build_core_replaced_speedup_summary(targets, baseline_timing, opt_timing):
    return _build_replaced_speedup_summary(
        targets,
        baseline_timing,
        opt_timing,
        name_getter=_core_target_name,
        op_getter=_core_target_op_type,
    )


def _build_conv_replaced_speedup_summary(targets, baseline_timing, opt_timing):
    return _build_replaced_speedup_summary(
        targets,
        baseline_timing,
        opt_timing,
        name_getter=lambda t: t["name"],
        op_getter=classify_target_type,
    )


def _print_replaced_speedup_summary(summary, mode_label):
    print(f"\n  [Replaced-Operator Avg Speedup] {mode_label}")
    print(
        f"  {'OpType':<24} {'Count':>6} {'Base(ms)':>10} "
        f"{'Mode(ms)':>10} {'W-Speedup':>10} {'AvgLayer':>10}"
    )
    print(f"  {'-'*76}")
    for op_name, st in summary["by_op_type"].items():
        print(
            f"  {op_name:<24} {st['count']:>6} {st['base_sum_ms']:>10.3f} "
            f"{st['opt_sum_ms']:>10.3f} {_fmt_speedup(st['weighted_speedup']):>10} "
            f"{_fmt_speedup(st['mean_layer_speedup']):>10}"
        )

    all_st = summary["all_replaced"]
    print(f"  {'-'*76}")
    print(
        f"  {'ALL_REPLACED':<24} {all_st['count']:>6} {all_st['base_sum_ms']:>10.3f} "
        f"{all_st['opt_sum_ms']:>10.3f} {_fmt_speedup(all_st['weighted_speedup']):>10} "
        f"{_fmt_speedup(all_st['mean_layer_speedup']):>10}"
    )
    print(
        "  Note: W-Speedup=ΣBase/ΣMode, AvgLayer=mean(Base_i/Mode_i) over valid layers."
    )


def _print_core_replaced_speedup_summary(summary, mode_label):
    _print_replaced_speedup_summary(summary, mode_label)


def measure_replaced_operator_speedup(
    baseline_model,
    mode_models,
    targets,
    replaced_names_by_mode,
    sample_input,
    name_getter,
    op_getter,
    warmup=8,
    iters=30,
    max_batch=2,
):
    micro_input = _select_microbench_input(sample_input, max_batch=max_batch)
    target_map = OrderedDict()
    for target in targets:
        name = name_getter(target)
        if name:
            target_map[name] = target

    union_names = []
    seen = set()
    for names in replaced_names_by_mode.values():
        for name in sorted(set(names)):
            if name in target_map and name not in seen:
                union_names.append(name)
                seen.add(name)

    baseline_timing = {}
    mode_timings = {label: {} for label in mode_models}
    skipped_layers = []

    for name in union_names:
        try:
            input_args = _capture_module_input(baseline_model, micro_input, name)
        except Exception as exc:
            skipped_layers.append({"layer": name, "stage": "capture", "error": str(exc)})
            continue
        if input_args is None:
            skipped_layers.append({"layer": name, "stage": "capture", "error": "no_input_captured"})
            continue

        try:
            baseline_module = _get_module_by_name(baseline_model, name)
            baseline_timing[name] = _measure_isolated_module_latency(
                baseline_module,
                input_args,
                warmup=warmup,
                iters=iters,
            )
        except Exception as exc:
            skipped_layers.append({"layer": name, "stage": "baseline", "error": str(exc)})
            continue

        for label, model in mode_models.items():
            if name not in replaced_names_by_mode.get(label, set()):
                continue
            try:
                mode_module = _get_module_by_name(model, name)
                mode_timings[label][name] = _measure_isolated_module_latency(
                    mode_module,
                    input_args,
                    warmup=warmup,
                    iters=iters,
                )
            except Exception as exc:
                skipped_layers.append({"layer": name, "stage": label, "error": str(exc)})

    summaries = {}
    for label, names in replaced_names_by_mode.items():
        valid_targets = [
            target_map[name]
            for name in sorted(set(names))
            if name in target_map and name in baseline_timing and name in mode_timings.get(label, {})
        ]
        summaries[label] = _build_replaced_speedup_summary(
            valid_targets,
            baseline_timing,
            mode_timings.get(label, {}),
            name_getter=name_getter,
            op_getter=op_getter,
        )

    meta = {
        "sample_batch_size": (
            int(micro_input.shape[1])
            if isinstance(micro_input, torch.Tensor) and micro_input.dim() >= 2
            else None
        ),
        "warmup": int(max(0, warmup)),
        "iters": int(max(1, iters)),
        "skipped_layers": skipped_layers,
    }
    return summaries, meta


def measure_sparsity_core(
    model,
    targets,
    loader,
    device,
    T,
    num_batches=10,
    spike_mode="normalized_bernoulli",
):
    target_names = [_core_target_name(t) for t in targets]
    sparsity_data = {name: {"zeros": 0, "total": 0} for name in target_names}

    def make_hook(name):
        def hook(m, inp, out):
            if not isinstance(inp, (tuple, list)) or len(inp) == 0:
                return
            x = inp[0]
            if not isinstance(x, torch.Tensor):
                return
            with torch.no_grad():
                x = x.detach()
                sparsity_data[name]["zeros"] += (x.abs() <= 1e-6).sum().item()
                sparsity_data[name]["total"] += x.numel()
        return hook

    hook_handles = []
    for name in target_names:
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


def collect_core_static_zero_layers(targets, sparsity_data):
    zero_layers = set()
    for t in targets:
        name = _core_target_name(t)
        if not name or not _is_core_staticzero_eligible(t):
            continue
        sd = sparsity_data.get(name, {"zeros": 0, "total": 0})
        if sd["total"] > 0 and sd["zeros"] == sd["total"]:
            zero_layers.add(name)
    return zero_layers


def measure_group_sparsity_core(
    model,
    targets,
    loader,
    device,
    T,
    num_batches=5,
    spike_mode="normalized_bernoulli",
):
    target_names = {_core_target_name(t) for t in targets}
    for name, mod in model.named_modules():
        if name in target_names and hasattr(mod, "collect_diag"):
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

    group_data = {}
    for name, mod in model.named_modules():
        if name not in target_names:
            continue
        if not hasattr(mod, "_last_diag"):
            continue
        diag = getattr(mod, "_last_diag", {}) or {}
        group_data[name] = {
            "active_group_ratio": diag.get("active_group_ratio", -1.0),
            "tile_zero_ratio": diag.get("tile_zero_ratio", -1.0),
            "total_group_count": diag.get("total_group_count", -1.0),
            "nonzero_group_count": diag.get("nonzero_group_count", -1.0),
            "tile_zero_count": diag.get("tile_zero_count", -1.0),
            "total_tile_count": diag.get("total_tile_count", -1.0),
            "effective_k_ratio": diag.get("effective_k_ratio", -1.0),
            "sparse_compute_ms": diag.get("sparse_compute_ms", -1.0),
            "sparse_total_ms": diag.get("sparse_total_ms", -1.0),
            "stage1_zero_tiles": diag.get("stage1_zero_tiles", -1),
            "stage2_tiles": diag.get("stage2_tiles", -1),
            "zero_tiles": diag.get("zero_tiles", -1),
            "sparse_tiles": diag.get("sparse_tiles", -1),
            "denseish_tiles": diag.get("denseish_tiles", -1),
            "prescan_mode": diag.get("prescan_mode", "unknown"),
            "metadata_kind": diag.get("metadata_kind", "unknown"),
            "kernel_type": diag.get("kernel_type", "unknown"),
            "stage1_zero_candidate": diag.get("stage1_zero_candidate", -1),
            "stage1_denseish": diag.get("stage1_denseish", -1),
            "stage1_uncertain": diag.get("stage1_uncertain", -1),
            "block_m": diag.get("block_m", -1),
        }

    for name, mod in model.named_modules():
        if name in target_names and hasattr(mod, "collect_diag"):
            mod.collect_diag = False
    return group_data


def run_core_all_ops_benchmark(args, model_baseline, loader, device, gpu_id):
    print(f"[3/7] Analyze replaceable targets via Core (all operator types) ...")
    imgs_s, _ = next(iter(loader))
    sample_input = make_spike_input(imgs_s[:4], args.T, device, spike_mode=args.spike_mode)

    registry = SpikeOpRegistry.default()
    analyzer = NetworkAnalyzer(registry)
    targets = analyzer.analyze(model_baseline, sample_input=sample_input)

    if not targets:
        print("\n  No replaceable layers found by Core analyzer.")
        return

    normalized_cnt = _normalize_core_targets_for_fuse_mode(
        targets, fuse_conv_lif=args.fuse_conv_lif
    )
    if normalized_cnt > 0:
        if args.fuse_conv_lif:
            print(
                f"  [Core-AllOps] fuse_conv_lif=ON but fused replacer path is unavailable; "
                f"normalized {normalized_cnt} fused_conv* targets to plain conv2d_*"
            )
        else:
            print(
                f"  [Core-AllOps] normalized {normalized_cnt} fused_conv* targets to plain conv2d_*"
            )

    if args.core_only_linear:
        linear_targets = [t for t in targets if t.op_type == "linear"]
        print(
            f"  [Core-AllOps] linear-only filter enabled: "
            f"{len(linear_targets)}/{len(targets)} targets kept"
        )
        targets = linear_targets
        if not targets:
            print("  No linear targets found after filtering.")
            return

    _print_core_target_report(targets)
    op_counter = Counter(t.op_type for t in targets)

    print(
        f"\n[4/7] Measure static-zero candidates ({args.sparsity_batches} batches) ..."
    )
    sparsity_data = measure_sparsity_core(
        model_baseline,
        targets,
        loader,
        device,
        args.T,
        num_batches=args.sparsity_batches,
        spike_mode=args.spike_mode,
    )
    zero_layers = collect_core_static_zero_layers(targets, sparsity_data)
    if args.spike_mode in ("normalized_bernoulli", "raw_bernoulli") and zero_layers:
        print(
            "  [Core-AllOps] stochastic spike mode detected; "
            "re-validating static-zero layers on a second pass for robustness ..."
        )
        sparsity_data_recheck = measure_sparsity_core(
            model_baseline,
            targets,
            loader,
            device,
            args.T,
            num_batches=args.sparsity_batches,
            spike_mode=args.spike_mode,
        )
        zero_layers_recheck = collect_core_static_zero_layers(targets, sparsity_data_recheck)
        unstable_zero_layers = set(zero_layers) - set(zero_layers_recheck)
        if unstable_zero_layers:
            print(
                f"  [Core-AllOps] removed {len(unstable_zero_layers)} unstable static-zero layer(s): "
                + ", ".join(sorted(unstable_zero_layers))
            )
        zero_layers = set(zero_layers) & set(zero_layers_recheck)

    eligible_staticzero = sum(1 for t in targets if _is_core_staticzero_eligible(t))
    print(
        f"  StaticZero eligible targets: {eligible_staticzero}/{len(targets)}, "
        f"exact-zero detected: {len(zero_layers)}"
    )

    print(
        f"\n[5/7] Measure group/tile diagnostics + build 4 benchmark modes "
        f"(Core replacer, all ops enabled, "
        f"fuse_conv_lif={'ON' if args.fuse_conv_lif else 'OFF'}) ..."
    )
    replacer = ModuleReplacer(verbose=True)
    sparse_create_kwargs = {
        "threshold": float(args.prescan_threshold),
        "return_ms": False,
        "collect_diag": False,
        "profile_runtime": False,
    }

    print("  - Measuring sparse diagnostics for hybrid dispatch ...")
    model_diag = copy.deepcopy(model_baseline)
    # Build sparse-replaced model only for collecting per-layer diagnostics.
    replacer.replace(
        model_diag,
        targets,
        block_sizes=None,
        static_zero_layers=set(),
        disable_static_zero=True,
        only_static_zero=False,
        sparse_kwargs=sparse_create_kwargs,
    )
    group_sparsity_data = measure_group_sparsity_core(
        model_diag,
        targets,
        loader,
        device,
        args.T,
        num_batches=min(args.sparsity_batches, 5),
        spike_mode=args.spike_mode,
    )
    del model_diag

    # Synthetic exact-zero diagnostics for static-zero layers.
    for t in targets:
        name = _core_target_name(t)
        if name in zero_layers:
            existing = group_sparsity_data.get(name, {})
            group_sparsity_data[name] = make_synthetic_zero_diag(
                layer_name=name,
                source="staticzero",
                total_group_count=existing.get("total_group_count", -1.0),
                total_tile_count=existing.get("total_tile_count", -1.0),
            )

    # If diagnostics are unavailable but element sparsity is effectively 100%,
    # treat as zero-fastpath source to avoid false dense routing.
    for t in targets:
        name = _core_target_name(t)
        if name in zero_layers:
            continue
        gd = group_sparsity_data.get(name, {})
        if gd.get("active_group_ratio", -1.0) >= 0:
            continue
        sd = sparsity_data.get(name, {"zeros": 0, "total": 1})
        elem_sp = sd["zeros"] / max(sd["total"], 1)
        if elem_sp >= 0.9999:
            existing = group_sparsity_data.get(name, {})
            group_sparsity_data[name] = make_synthetic_zero_diag(
                layer_name=name,
                source="zero_fastpath",
                total_group_count=existing.get("total_group_count", -1.0),
                total_tile_count=existing.get("total_tile_count", -1.0),
            )

    dispatch_decisions = dispatch_all_layers(
        targets,
        group_sparsity_data,
        zero_layers=set(zero_layers),
    )
    hybrid_static_zero_layers, hybrid_sparse_set = decisions_to_sets(dispatch_decisions)

    # Safety net: StaticZero must remain exact-zero only.
    invalid_static_zero_layers = set(hybrid_static_zero_layers) - set(zero_layers)
    if invalid_static_zero_layers:
        for _name in sorted(invalid_static_zero_layers):
            _dec = dispatch_decisions[_name]
            _dec.backend = "sparse"
            _dec.reason = "core_safety_remap_nonexact_staticzero_to_sparse"
        hybrid_static_zero_layers -= invalid_static_zero_layers
        hybrid_sparse_set |= invalid_static_zero_layers

    # Keep a compact dispatch report for Core path.
    print(
        f"  - Hybrid dispatch sets: "
        f"StaticZero={len(hybrid_static_zero_layers)}, Sparse={len(hybrid_sparse_set)}, "
        f"DenseKeep={len(targets) - len(hybrid_static_zero_layers) - len(hybrid_sparse_set)}"
    )
    report_targets = [{"name": _core_target_name(t)} for t in targets]
    print_dispatch_decision_report(report_targets, group_sparsity_data, dispatch_decisions)

    print("  - Building StaticZero only ...")
    model_staticzero = copy.deepcopy(model_baseline)
    sz_summary = replacer.replace(
        model_staticzero,
        targets,
        block_sizes=None,
        static_zero_layers=set(zero_layers),
        disable_static_zero=False,
        only_static_zero=True,
        sparse_kwargs=sparse_create_kwargs,
    )

    print("  - Building SparseKernel only ...")
    model_sparse = copy.deepcopy(model_baseline)
    so_summary = replacer.replace(
        model_sparse,
        targets,
        block_sizes=None,
        static_zero_layers=set(),
        disable_static_zero=True,
        only_static_zero=False,
        sparse_kwargs=sparse_create_kwargs,
    )
    core_target_names = {_core_target_name(t) for t in targets}
    sparse_only_force_dense_layers = set()
    if args.sparse_only_dispatch_dense:
        sparse_only_force_dense_layers = {
            n for n in core_target_names
            if n not in hybrid_sparse_set and n not in zero_layers
        }
    sparse_only_force_zero_layers = set(zero_layers) if args.sparse_only_promote_zero else set()
    sparse_only_hint_summary = _apply_runtime_backend_hints(
        model_sparse,
        force_dense_layers=sparse_only_force_dense_layers,
        force_zero_layers=sparse_only_force_zero_layers,
        verbose=True,
        label="Sparse-only",
    )

    print("  - Building SparseKernel + StaticZero ...")
    model_hybrid = copy.deepcopy(model_baseline)
    hybrid_target_names = set(hybrid_static_zero_layers) | set(hybrid_sparse_set)
    hybrid_targets = [t for t in targets if _core_target_name(t) in hybrid_target_names]
    hy_summary = replacer.replace(
        model_hybrid,
        hybrid_targets,
        block_sizes=None,
        static_zero_layers=set(hybrid_static_zero_layers),
        disable_static_zero=False,
        only_static_zero=False,
        sparse_kwargs=sparse_create_kwargs,
    )

    total_targets = len(targets)

    route_counts = {
        "static_zero_only": {
            "total": total_targets,
            "sparse": sz_summary[1],
            "fused": 0,
            "static_zero": sz_summary[2],
            "dense_keep": sz_summary[3],
        },
        "sparse_only": {
            "total": total_targets,
            "sparse": so_summary[1],
            "fused": 0,
            "static_zero": so_summary[2],
            "dense_keep": so_summary[3],
            "runtime_force_dense": sparse_only_hint_summary["forced_dense"],
            "runtime_force_zero": sparse_only_hint_summary["forced_zero"],
        },
        "hybrid": {
            "total": total_targets,
            "sparse": hy_summary[1],
            "fused": 0,
            "static_zero": hy_summary[2],
            "dense_keep": hy_summary[3],
        },
    }
    print("\n  Replacement summary:")
    for mode_name, c in route_counts.items():
        print(
            f"    {mode_name}: total={c['total']}, sparse={c['sparse']}, "
            f"fused={c['fused']}, static_zero={c['static_zero']}, dense_keep={c['dense_keep']}"
        )

    if args.inference_mode:
        n_so = prepare_for_timing(model_sparse)
        n_hy = prepare_for_timing(model_hybrid)
        print(f"  [Timing] inference_mode enabled: sparse_only={n_so}, hybrid={n_hy}")

    if args.launch_all_tiles:
        n_so_launch = set_launch_mode(model_sparse, launch_all=True)
        n_hy_launch = set_launch_mode(model_hybrid, launch_all=True)
        print(f"  [Timing] launch_all_tiles enabled: sparse_only={n_so_launch}, hybrid={n_hy_launch}")

    if args.inference_mode or args.launch_all_tiles:
        print(f"  [Timing] sparse_only sync state: {count_sync_state(model_sparse)}")
        print(f"  [Timing] hybrid sync state: {count_sync_state(model_hybrid)}")

    print(f"\n[6/7] End-to-end latency (warmup={args.warmup}) ...")
    dense_res = measure_mode(model_baseline, loader, device, args.T, args.warmup, args.spike_mode, args.power, "Dense cuDNN")
    sz_res = measure_mode(model_staticzero, loader, device, args.T, args.warmup, args.spike_mode, args.power, "StaticZero only")
    so_res = measure_mode(model_sparse, loader, device, args.T, args.warmup, args.spike_mode, args.power, "SparseKernel only")
    hy_res = measure_mode(model_hybrid, loader, device, args.T, args.warmup, args.spike_mode, args.power, "SparseKernel + StaticZero")
    for r in [dense_res, sz_res, so_res, hy_res]:
        print_mode_result(r)

    sz_speedup = dense_res["avg_ms"] / sz_res["avg_ms"] if sz_res["avg_ms"] > 1e-6 else float("inf")
    so_speedup = dense_res["avg_ms"] / so_res["avg_ms"] if so_res["avg_ms"] > 1e-6 else float("inf")
    hy_speedup = dense_res["avg_ms"] / hy_res["avg_ms"] if hy_res["avg_ms"] > 1e-6 else float("inf")
    sz_esave = (1 - sz_res["energy_j"] / max(dense_res["energy_j"], 1e-9)) * 100.0
    so_esave = (1 - so_res["energy_j"] / max(dense_res["energy_j"], 1e-9)) * 100.0
    hy_esave = (1 - hy_res["energy_j"] / max(dense_res["energy_j"], 1e-9)) * 100.0

    print(f"\n  Speedup StaticZero-only: {sz_speedup:.3f}x")
    print(f"  Speedup Sparse-only:     {so_speedup:.3f}x")
    print(f"  Speedup Hybrid:          {hy_speedup:.3f}x")

    so_dispatch_summary, so_dispatch_per_module = _extract_runtime_dispatch_disagreement(model_sparse)
    hy_dispatch_summary, hy_dispatch_per_module = _extract_runtime_dispatch_disagreement(model_hybrid)
    if so_dispatch_summary["modules_with_runtime_dispatch"] > 0 or hy_dispatch_summary["modules_with_runtime_dispatch"] > 0:
        print(
            "  Runtime dispatch disagreement "
            f"(Sparse-only): {so_dispatch_summary['total_mismatch']}/{so_dispatch_summary['total_seen']} "
            f"({so_dispatch_summary['mismatch_rate'] * 100.0:.2f}%)"
        )
        print(
            "  Runtime dispatch disagreement "
            f"(Hybrid):      {hy_dispatch_summary['total_mismatch']}/{hy_dispatch_summary['total_seen']} "
            f"({hy_dispatch_summary['mismatch_rate'] * 100.0:.2f}%)"
        )

    layer_profile_data = {}
    replaced_operator_speedup = {}
    replaced_operator_speedup_meta = {}

    replaced_names_by_mode = {
        "static_zero_only": set(zero_layers),
        "sparse_only": set(core_target_names),
        "hybrid": set(hybrid_target_names),
    }
    print(
        f"\n  [ReplacedOpBench] lightweight isolated-module benchmark "
        f"(batch<={args.replaced_op_batch}, warmup={args.replaced_op_warmup}, "
        f"iters={args.replaced_op_iters}) ..."
    )
    replaced_operator_speedup, replaced_operator_speedup_meta = measure_replaced_operator_speedup(
        baseline_model=model_baseline,
        mode_models={
            "static_zero_only": model_staticzero,
            "sparse_only": model_sparse,
            "hybrid": model_hybrid,
        },
        targets=targets,
        replaced_names_by_mode=replaced_names_by_mode,
        sample_input=sample_input,
        name_getter=_core_target_name,
        op_getter=_core_target_op_type,
        warmup=args.replaced_op_warmup,
        iters=args.replaced_op_iters,
        max_batch=args.replaced_op_batch,
    )
    replaced_operator_speedup["meta"] = replaced_operator_speedup_meta
    _print_core_replaced_speedup_summary(
        replaced_operator_speedup["static_zero_only"], "StaticZero only"
    )
    _print_core_replaced_speedup_summary(
        replaced_operator_speedup["sparse_only"], "SparseKernel only"
    )
    _print_core_replaced_speedup_summary(
        replaced_operator_speedup["hybrid"], "SparseKernel + StaticZero"
    )
    if replaced_operator_speedup_meta["skipped_layers"]:
        print(
            f"  [ReplacedOpBench] skipped {len(replaced_operator_speedup_meta['skipped_layers'])} "
            "layer/mode measurement(s); see JSON for details."
        )

    if args.layer_profile:
        lp_warmup = args.layer_profile_warmup
        lp_batches = args.layer_profile_batches
        target_names = [_core_target_name(t) for t in targets]

        print(
            f"\n  [LayerProfile] profiling replaced operators "
            f"(warmup={lp_warmup}, batches={lp_batches}) ..."
        )
        baseline_layer_timing = measure_layer_timing(
            model_baseline,
            loader,
            device,
            args.T,
            target_names,
            warmup=lp_warmup,
            num_batches=lp_batches,
            spike_mode=args.spike_mode,
        )
        staticzero_layer_timing = measure_layer_timing(
            model_staticzero,
            loader,
            device,
            args.T,
            target_names,
            warmup=lp_warmup,
            num_batches=lp_batches,
            spike_mode=args.spike_mode,
        )
        sparse_only_layer_timing = measure_layer_timing(
            model_sparse,
            loader,
            device,
            args.T,
            target_names,
            warmup=lp_warmup,
            num_batches=lp_batches,
            spike_mode=args.spike_mode,
        )
        hybrid_layer_timing = measure_layer_timing(
            model_hybrid,
            loader,
            device,
            args.T,
            target_names,
            warmup=lp_warmup,
            num_batches=lp_batches,
            spike_mode=args.spike_mode,
        )

        layer_profile_data = {
            "baseline_layer_ms": {k: round(v, 6) for k, v in baseline_layer_timing.items()},
            "static_zero_only_layer_ms": {k: round(v, 6) for k, v in staticzero_layer_timing.items()},
            "sparse_only_layer_ms": {k: round(v, 6) for k, v in sparse_only_layer_timing.items()},
            "hybrid_layer_ms": {k: round(v, 6) for k, v in hybrid_layer_timing.items()},
            "per_layer_speedup_static_zero_only": {},
            "per_layer_speedup_sparse_only": {},
            "per_layer_speedup_hybrid": {},
        }
        for name in target_names:
            b = baseline_layer_timing.get(name, 0.0)
            s_sz = staticzero_layer_timing.get(name, 0.0)
            s_so = sparse_only_layer_timing.get(name, 0.0)
            s_hy = hybrid_layer_timing.get(name, 0.0)
            layer_profile_data["per_layer_speedup_static_zero_only"][name] = (
                round(b / s_sz, 6) if s_sz > 1e-6 else float("inf")
            )
            layer_profile_data["per_layer_speedup_sparse_only"][name] = (
                round(b / s_so, 6) if s_so > 1e-6 else float("inf")
            )
            layer_profile_data["per_layer_speedup_hybrid"][name] = (
                round(b / s_hy, 6) if s_hy > 1e-6 else float("inf")
            )
    else:
        print("  [LayerProfile] skipped heavy hook-based profiling (--layer_profile=OFF).")

    print(f"\n[7/7] Consistency check ({args.verify_batches} batches) ...")
    sz_cos, sz_agr, sz_mabs = verify_consistency(
        model_baseline, model_staticzero, loader, device, args.T,
        num_batches=args.verify_batches, spike_mode=args.spike_mode
    )
    so_cos, so_agr, so_mabs = verify_consistency(
        model_baseline, model_sparse, loader, device, args.T,
        num_batches=args.verify_batches, spike_mode=args.spike_mode
    )
    hy_cos, hy_agr, hy_mabs = verify_consistency(
        model_baseline, model_hybrid, loader, device, args.T,
        num_batches=args.verify_batches, spike_mode=args.spike_mode
    )
    sz_ok = sz_cos > 0.999 and sz_mabs < 0.1
    so_ok = so_cos > 0.999 and so_mabs < 0.1
    hy_ok = hy_cos > 0.999 and hy_mabs < 0.1
    print(f"  StaticZero-only: cos={sz_cos:.8f}  agree={sz_agr*100:.2f}%  max_abs={sz_mabs:.6f}  {'PASS' if sz_ok else 'FAIL'}")
    print(f"  Sparse-only:     cos={so_cos:.8f}  agree={so_agr*100:.2f}%  max_abs={so_mabs:.6f}  {'PASS' if so_ok else 'FAIL'}")
    print(f"  Hybrid:          cos={hy_cos:.8f}  agree={hy_agr*100:.2f}%  max_abs={hy_mabs:.6f}  {'PASS' if hy_ok else 'FAIL'}")

    print(f"\n{'='*96}")
    print(f"{'FOUR-WAY SUMMARY (Core All Ops)':^96}")
    print(f"{'='*96}")
    print(f"  Model: {args.model}  Dataset: {args.dataset}  T={args.T}  Spike: {args.spike_mode}")
    print(f"  Targets: {len(targets)}  StaticZero exact-zero layers: {len(zero_layers)}")
    print(f"  Fuse Conv+LIF: {'ON' if args.fuse_conv_lif else 'OFF'}")
    print(f"\n  {'Mode':<28} {'Latency(ms)':>12} {'Speedup':>10} {'EnergySave':>12} {'Consistency':>12}")
    print(f"  {'-'*80}")
    print(f"  {'Dense cuDNN':<28} {dense_res['avg_ms']:>12.2f} {'1.000x':>10} {'0.00%':>12} {'REF':>12}")
    print(f"  {'StaticZero only':<28} {sz_res['avg_ms']:>12.2f} {sz_speedup:>9.3f}x {sz_esave:>11.2f}% {('PASS' if sz_ok else 'FAIL'):>12}")
    print(f"  {'SparseKernel only':<28} {so_res['avg_ms']:>12.2f} {so_speedup:>9.3f}x {so_esave:>11.2f}% {('PASS' if so_ok else 'FAIL'):>12}")
    print(f"  {'SparseKernel + StaticZero':<28} {hy_res['avg_ms']:>12.2f} {hy_speedup:>9.3f}x {hy_esave:>11.2f}% {('PASS' if hy_ok else 'FAIL'):>12}")
    if replaced_operator_speedup:
        so_all = replaced_operator_speedup["sparse_only"]["all_replaced"]
        hy_all = replaced_operator_speedup["hybrid"]["all_replaced"]
        print(
            f"\n  Replaced-op avg speedup (Sparse-only): "
            f"W={_fmt_speedup(so_all['weighted_speedup'])}, "
            f"AvgLayer={_fmt_speedup(so_all['mean_layer_speedup'])}"
        )
        print(
            f"  Replaced-op avg speedup (Hybrid):      "
            f"W={_fmt_speedup(hy_all['weighted_speedup'])}, "
            f"AvgLayer={_fmt_speedup(hy_all['mean_layer_speedup'])}"
        )
    print(f"{'='*96}\n")

    results = {
        "script_mode": "core_all_ops_four_way",
        "model": args.model,
        "dataset": args.dataset,
        "T": args.T,
        "batch_size": args.batch_size,
        "gpu": torch.cuda.get_device_name(gpu_id),
        "weight_init": args.weight_init,
        "seed": args.seed,
        "spike_mode": args.spike_mode,
        "prescan_threshold": float(args.prescan_threshold),
        "fuse_conv_lif": bool(args.fuse_conv_lif),
        "core_only_linear": bool(args.core_only_linear),
        "num_targets": len(targets),
        "op_type_counts": dict(sorted(op_counter.items())),
        "num_staticzero_eligible": eligible_staticzero,
        "num_staticzero_exact_zero": len(zero_layers),
        "static_zero_layers": sorted(list(zero_layers)),
        "hybrid_static_zero_layers": sorted(list(hybrid_static_zero_layers)),
        "hybrid_sparse_layers": sorted(list(hybrid_sparse_set)),
        "hybrid_dense_keep_layers": sorted(
            [
                _core_target_name(t)
                for t in targets
                if _core_target_name(t) not in hybrid_static_zero_layers
                and _core_target_name(t) not in hybrid_sparse_set
            ]
        ),
        "dispatch_decisions": {name: dec.to_dict() for name, dec in dispatch_decisions.items()},
        "dispatch_invalid_staticzero_remapped_layers": sorted(list(invalid_static_zero_layers)),
        "stochastic_unstable_staticzero_removed_layers": (
            sorted(list(unstable_zero_layers)) if "unstable_zero_layers" in locals() else []
        ),
        "route_counts": route_counts,
        "sparse_only_runtime_hints": {
            "dispatch_dense_enabled": bool(args.sparse_only_dispatch_dense),
            "promote_zero_enabled": bool(args.sparse_only_promote_zero),
            **sparse_only_hint_summary,
        },
        "runtime_dispatch_disagreement": {
            "sparse_only": so_dispatch_summary,
            "hybrid": hy_dispatch_summary,
        },
        "runtime_dispatch_disagreement_per_module": {
            "sparse_only": so_dispatch_per_module,
            "hybrid": hy_dispatch_per_module,
        },
        "dense": {
            "ms_per_batch": round(dense_res["avg_ms"], 4),
            "total_ms": round(dense_res["total_ms"], 4),
            "energy_j": round(dense_res["energy_j"], 6),
        },
        "static_zero_only": {
            "ms_per_batch": round(sz_res["avg_ms"], 4),
            "total_ms": round(sz_res["total_ms"], 4),
            "energy_j": round(sz_res["energy_j"], 6),
            "speedup": round(sz_speedup, 6),
            "energy_saving_pct": round(sz_esave, 4),
            "cosine_sim": round(sz_cos, 8),
            "pred_agreement_pct": round(sz_agr * 100.0, 4),
            "max_abs_err": round(sz_mabs, 6),
            "consistency": "PASS" if sz_ok else "FAIL",
        },
        "sparse_only": {
            "ms_per_batch": round(so_res["avg_ms"], 4),
            "total_ms": round(so_res["total_ms"], 4),
            "energy_j": round(so_res["energy_j"], 6),
            "speedup": round(so_speedup, 6),
            "energy_saving_pct": round(so_esave, 4),
            "cosine_sim": round(so_cos, 8),
            "pred_agreement_pct": round(so_agr * 100.0, 4),
            "max_abs_err": round(so_mabs, 6),
            "runtime_force_dense": int(sparse_only_hint_summary.get("forced_dense", 0)),
            "runtime_force_zero": int(sparse_only_hint_summary.get("forced_zero", 0)),
            "consistency": "PASS" if so_ok else "FAIL",
        },
        "hybrid": {
            "ms_per_batch": round(hy_res["avg_ms"], 4),
            "total_ms": round(hy_res["total_ms"], 4),
            "energy_j": round(hy_res["energy_j"], 6),
            "speedup": round(hy_speedup, 6),
            "energy_saving_pct": round(hy_esave, 4),
            "cosine_sim": round(hy_cos, 8),
            "pred_agreement_pct": round(hy_agr * 100.0, 4),
            "max_abs_err": round(hy_mabs, 6),
            "consistency": "PASS" if hy_ok else "FAIL",
        },
    }
    if layer_profile_data:
        results["layer_profile"] = layer_profile_data
    if replaced_operator_speedup:
        results["replaced_operator_speedup"] = replaced_operator_speedup
    if group_sparsity_data:
        results["group_sparsity"] = group_sparsity_data

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
        description=f"SparseFlow four-way E2E benchmark {BENCH_VERSION}")
    parser.add_argument("--model", type=str, default="spiking_resnet18",
                        choices=list(MODEL_BUILDERS.keys()))
    parser.add_argument("--dataset",type=str,default="cifar10",
                        choices=["cifar10", "cifar100", "imagenet_val_flat"],)
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--v_threshold", type=float, default=1.0)
    parser.add_argument(
        "--prescan_threshold",
        type=float,
        default=1e-6,
        help="Activity threshold passed to sparse operator wrappers (prescan epsilon).",
    )
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
    # NEW: layer-level profiling
    parser.add_argument("--layer_profile", action="store_true",
                        help="Measure per-layer timing for baseline vs each sparse variant")
    parser.add_argument("--layer_profile_warmup", type=int, default=5,
                        help="Warmup batches for layer profiling")
    parser.add_argument("--layer_profile_batches", type=int, default=20,
                        help="Measurement batches for layer profiling")
    parser.add_argument("--replaced_op_warmup", type=int, default=8,
                        help="Warmup iterations for lightweight replaced-operator microbenchmark")
    parser.add_argument("--replaced_op_iters", type=int, default=30,
                        help="Measured iterations for lightweight replaced-operator microbenchmark")
    parser.add_argument("--replaced_op_batch", type=int, default=2,
                        help="Max representative batch size for lightweight replaced-operator microbenchmark")


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
            "When enabled, run full four-way pipeline (Dense / StaticZero / Sparse / Hybrid)."
        ),
    )
    parser.add_argument(
        "--core_only_linear",
        action="store_true",
        help="When --replace_all_ops is enabled, replace only nn.Linear targets.",
    )
    parser.add_argument(
        "--fuse_conv_lif",
        action="store_true",
        help="Enable Conv2d+LIF fused replacement in Core pipeline (default: disabled).",
    )
    parser.add_argument(
        "--sparse_only_dispatch_dense",
        dest="sparse_only_dispatch_dense",
        action="store_true",
        default=True,
        help=(
            "In Sparse-only mode, force dense runtime path for layers that dispatch "
            "marks as dense (keeps sparse modules but avoids low-gain sparse execution)."
        ),
    )
    parser.add_argument(
        "--no_sparse_only_dispatch_dense",
        dest="sparse_only_dispatch_dense",
        action="store_false",
        help="Disable dispatch-driven dense runtime hints in Sparse-only mode.",
    )
    parser.add_argument(
        "--sparse_only_promote_zero",
        action="store_true",
        help=(
            "In Sparse-only mode, also force zero-runtime path for exact-zero layers. "
            "Default OFF to keep Sparse-only separate from StaticZero semantics."
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

    title = f"{args.model} | {args.dataset.upper()} | T={args.T} | Four-Way {BENCH_VERSION}"
    print(f"\n{'='*80}")
    print(f"{('SparseFlow Four-Way Benchmark ' + BENCH_VERSION):^80}")
    print(f"{title:^80}")
    print(f"{'='*80}")
    print(f"  GPU:          {gpu_id} ({torch.cuda.get_device_name(gpu_id)})")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Spike mode:   {args.spike_mode}")
    print(f"  Weight init:  {args.weight_init}")
    print(f"  Seed:         {args.seed}")
    print(f"  Power (TDP):  {args.power} W")
    print(f"  Prescan eps:  {args.prescan_threshold}")
    print(f"  Min spatial:  {args.min_spatial_size}x{args.min_spatial_size}")
    print(f"  Layer profile: {'ON' if args.layer_profile else 'OFF'}")
    print(
        f"  Replaced-op bench: batch<={args.replaced_op_batch}, "
        f"warmup={args.replaced_op_warmup}, iters={args.replaced_op_iters}"
    )
    print(f"  Sparse-only dispatch-dense hint: {'ON' if args.sparse_only_dispatch_dense else 'OFF'}")
    print(f"  Sparse-only promote-zero hint:  {'ON' if args.sparse_only_promote_zero else 'OFF'}")
    print()

    print(f"[1/7] Building {args.model} ...")
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
        T=args.T,
        num_classes=num_classes, seed=args.seed)

    print(f"[2/7] Loading {args.dataset} test set ...")
    ds = build_dataset(args.dataset, args.data_root, spike_mode=args.spike_mode)
    # loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print(f"  Test set: {len(ds)} samples, {len(loader)} batches, {num_classes} classes")

    if args.replace_all_ops:
        run_core_all_ops_benchmark(args, model_baseline, loader, device, gpu_id)
        return

    print(f"[3/7] Analyzing network topology ...")
    imgs_s, _ = next(iter(loader))
    sample_input = make_spike_input(imgs_s[:4], args.T, device, spike_mode=args.spike_mode)
    targets, skipped = analyze_targets(
        model_baseline, sample_input, device, fused=False,
        min_spatial_size=args.min_spatial_size)
    print_analysis_report(targets, skipped, fused=False)
    if not targets:
        print("\n  No replaceable target layers found, exiting.")
        return

    print(f"\n[4/7] Measuring sparsity and dispatch features ({args.sparsity_batches} batches) ...")
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

    sparse_only_force_dense_layers = set()
    if args.sparse_only_dispatch_dense:
        sparse_only_force_dense_layers = {
            t["name"] for t in targets
            if t["name"] not in hybrid_sparse_set and t["name"] not in zero_layers
        }
    sparse_only_force_zero_layers = set(zero_layers) if args.sparse_only_promote_zero else set()

    model_static_zero_only, model_sparse_only, model_hybrid, route_counts, sparse_only_hint_summary = build_fourway_models(
        model_baseline,
        targets,
        hybrid_static_zero_layers,
        hybrid_sparse_set=hybrid_sparse_set,
        sparse_only_force_dense_layers=sparse_only_force_dense_layers,
        sparse_only_force_zero_layers=sparse_only_force_zero_layers,
    )
    print(f"\n  Replacement finished:")
    for mode_name, rc in route_counts.items():
        print(f"    {mode_name}: {rc['num_sparse_conv']} Sparse + {rc['num_static_zero']} StaticZero + {rc['num_dense_keep']} DenseKeep")


    # v25: timing preparation
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


    print(f"\n[5/7] End-to-end latency (warmup={args.warmup}) ...")
    dense_res = measure_mode(model_baseline, loader, device, args.T, args.warmup, args.spike_mode, args.power, "Dense cuDNN")
    sz_res = measure_mode(model_static_zero_only, loader, device, args.T, args.warmup, args.spike_mode, args.power, "StaticZero only")
    so_res = measure_mode(model_sparse_only, loader, device, args.T, args.warmup, args.spike_mode, args.power, "SparseKernel only")
    hy_res = measure_mode(model_hybrid, loader, device, args.T, args.warmup, args.spike_mode, args.power, "SparseKernel + StaticZero")
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

    # [6/7] Lightweight replaced-op benchmark + optional heavy layer profile
    layer_profile_data = {}
    replaced_operator_speedup = {}
    replaced_operator_speedup_meta = {}

    replaced_names_by_mode = {
        "static_zero_only": set(zero_layers),
        "sparse_only": {t["name"] for t in targets},
        "hybrid": set(hybrid_static_zero_layers) | set(hybrid_sparse_set),
    }
    print(
        f"\n[6/7] Replaced-operator microbenchmark "
        f"(batch<={args.replaced_op_batch}, warmup={args.replaced_op_warmup}, "
        f"iters={args.replaced_op_iters}) ..."
    )
    replaced_operator_speedup, replaced_operator_speedup_meta = measure_replaced_operator_speedup(
        baseline_model=model_baseline,
        mode_models={
            "static_zero_only": model_static_zero_only,
            "sparse_only": model_sparse_only,
            "hybrid": model_hybrid,
        },
        targets=targets,
        replaced_names_by_mode=replaced_names_by_mode,
        sample_input=sample_input,
        name_getter=lambda t: t["name"],
        op_getter=classify_target_type,
        warmup=args.replaced_op_warmup,
        iters=args.replaced_op_iters,
        max_batch=args.replaced_op_batch,
    )
    replaced_operator_speedup["meta"] = replaced_operator_speedup_meta
    _print_replaced_speedup_summary(
        replaced_operator_speedup["static_zero_only"], "StaticZero only"
    )
    _print_replaced_speedup_summary(
        replaced_operator_speedup["sparse_only"], "SparseKernel only"
    )
    _print_replaced_speedup_summary(
        replaced_operator_speedup["hybrid"], "SparseKernel + StaticZero"
    )
    if replaced_operator_speedup_meta["skipped_layers"]:
        print(
            f"  [ReplacedOpBench] skipped {len(replaced_operator_speedup_meta['skipped_layers'])} "
            "layer/mode measurement(s); see JSON for details."
        )

    if args.layer_profile:
        lp_warmup = args.layer_profile_warmup
        lp_batches = args.layer_profile_batches
        target_names = [t["name"] for t in targets]

        print(f"\n  [LayerProfile] heavy hook-based profiling (warmup={lp_warmup}, batches={lp_batches}) ...")

        # 6a) Baseline (Dense cuDNN) per-layer timing
        print(f"  Measuring baseline (Dense cuDNN) per-layer timing ...")
        baseline_layer_timing = measure_layer_timing(
            model_baseline, loader, device, args.T, target_names,
            warmup=lp_warmup, num_batches=lp_batches, spike_mode=args.spike_mode)

        # 6b) SparseKernel-only per-layer timing
        print(f"  Measuring SparseKernel-only per-layer timing ...")
        # For sparse models, the replaced layer names still exist but the module
        # Type is now SparseConv2d / StaticZeroConv2d, hooks work the same way.
        sparse_only_layer_timing = measure_layer_timing(
            model_sparse_only, loader, device, args.T, target_names,
            warmup=lp_warmup, num_batches=lp_batches, spike_mode=args.spike_mode)

        # 6c) Hybrid per-layer timing
        print(f"  Measuring Hybrid per-layer timing ...")
        hybrid_layer_timing = measure_layer_timing(
            model_hybrid, loader, device, args.T, target_names,
            warmup=lp_warmup, num_batches=lp_batches, spike_mode=args.spike_mode)

        # Print SparseKernel-only layer profile
        print(f"\n  {'=' * 77}")
        print(f"  {'Per-layer Compare: Dense cuDNN vs SparseKernel-only':^77}")
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
        print("  [LayerProfile] skipped heavy hook-based profiling (--layer_profile=OFF).")

    print(f"\n[7/7] Consistency check ({args.verify_batches} batches) ...")
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
    print(f"{('FOUR-WAY SUMMARY ' + BENCH_VERSION):^96}")
    print(f"{'='*96}")
    print(f"  Model: {args.model}  Dataset: {args.dataset}  T={args.T}  Spike: {args.spike_mode}")
    print(f"  Avg Sparsity: {avg_sparsity:.2f}%  Targets: {len(targets)}  Skipped: {len(skipped)}")
    print(f"\n  {'Mode':<28} {'Latency(ms)':>12} {'Speedup':>10} {'EnergySave':>12} {'Consistency':>12}")
    print(f"  {'-'*80}")
    print(f"  {'Dense cuDNN':<28} {dense_res['avg_ms']:>12.2f} {'1.000x':>10} {'0.00%':>12} {'REF':>12}")
    print(f"  {'StaticZero only':<28} {sz_res['avg_ms']:>12.2f} {sz_speedup:>9.3f}x {sz_esave:>11.2f}% {('PASS' if sz_ok else 'FAIL'):>12}")
    print(f"  {'SparseKernel only':<28} {so_res['avg_ms']:>12.2f} {so_speedup:>9.3f}x {so_esave:>11.2f}% {('PASS' if so_ok else 'FAIL'):>12}")
    print(f"  {'SparseKernel + StaticZero':<28} {hy_res['avg_ms']:>12.2f} {hy_speedup:>9.3f}x {hy_esave:>11.2f}% {('PASS' if hy_ok else 'FAIL'):>12}")
    if replaced_operator_speedup:
        so_all = replaced_operator_speedup["sparse_only"]["all_replaced"]
        hy_all = replaced_operator_speedup["hybrid"]["all_replaced"]
        print(
            f"\n  Replaced-op avg speedup (Sparse-only): "
            f"W={_fmt_speedup(so_all['weighted_speedup'])}, "
            f"AvgLayer={_fmt_speedup(so_all['mean_layer_speedup'])}"
        )
        print(
            f"  Replaced-op avg speedup (Hybrid):      "
            f"W={_fmt_speedup(hy_all['weighted_speedup'])}, "
            f"AvgLayer={_fmt_speedup(hy_all['mean_layer_speedup'])}"
        )
    print(f"{'='*96}\n")


    # v25: A/B tile launch comparison
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
        "script_version": BENCH_VERSION,
        "model": args.model, "dataset": args.dataset, "T": args.T,
        "batch_size": args.batch_size, "gpu": torch.cuda.get_device_name(gpu_id),
        "spike_mode": args.spike_mode, "weight_init": args.weight_init, "seed": args.seed,
        "avg_sparsity_pct": round(avg_sparsity, 2),
        "num_targets": len(targets), "num_skipped": len(skipped),
        "num_zero_layers": len(zero_layers),
        "static_zero_layers": sorted(list(zero_layers)),
        "route_counts": route_counts,
        "sparse_only_runtime_hints": {
            "dispatch_dense_enabled": bool(args.sparse_only_dispatch_dense),
            "promote_zero_enabled": bool(args.sparse_only_promote_zero),
            **sparse_only_hint_summary,
        },
        "hybrid_static_zero_layers": sorted(list(hybrid_static_zero_layers)),
        "hybrid_sparse_layers": sorted(list(hybrid_sparse_set)),
        "hybrid_dense_keep_layers": sorted([t["name"] for t in targets if t["name"] not in hybrid_static_zero_layers and t["name"] not in hybrid_sparse_set]),
        "dispatch_decisions": {name: dec.to_dict() for name, dec in dispatch_decisions.items()},
        "dispatch_invalid_staticzero_remapped_layers": sorted(list(invalid_static_zero_layers)) if "invalid_static_zero_layers" in locals() else [],
        "dense": {"ms_per_batch": round(dense_res["avg_ms"], 2), "energy_j": round(dense_res["energy_j"], 4)},
        "static_zero_only": {"ms_per_batch": round(sz_res["avg_ms"], 2), "speedup": round(sz_speedup, 4), "energy_saving_pct": round(sz_esave, 2), "cosine_sim": round(sz_cos, 8), "consistency": "PASS" if sz_ok else "FAIL"},
        "sparse_only": {
            "ms_per_batch": round(so_res["avg_ms"], 2),
            "speedup": round(so_speedup, 4),
            "energy_saving_pct": round(so_esave, 2),
            "cosine_sim": round(so_cos, 8),
            "runtime_force_dense": int(sparse_only_hint_summary.get("forced_dense", 0)),
            "runtime_force_zero": int(sparse_only_hint_summary.get("forced_zero", 0)),
            "consistency": "PASS" if so_ok else "FAIL",
        },
        "hybrid": {"ms_per_batch": round(hy_res["avg_ms"], 2), "speedup": round(hy_speedup, 4), "energy_saving_pct": round(hy_esave, 2), "cosine_sim": round(hy_cos, 8), "consistency": "PASS" if hy_ok else "FAIL"},
    }
    # NEW: include layer profile data in JSON
    if layer_profile_data:
        results["layer_profile"] = layer_profile_data
    if replaced_operator_speedup:
        results["replaced_operator_speedup"] = replaced_operator_speedup

    if group_sparsity_data:
        results["group_sparsity"] = group_sparsity_data

    if args.collect_diag and (args.diag_json or args.diag_csv):
        run_id = f"{args.model}_{args.dataset}_T{args.T}_fourway_{BENCH_VERSION.replace('-', '_')}"
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
                    layer_name=name, mode_used="sparsekernel", replaced=True,
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
            print(f"  Diagnostic JSON: {args.diag_json}")
        if args.diag_csv:
            layer_logger.save_csv(args.diag_csv)
            print(f"  Diagnostic CSV: {args.diag_csv}")
        layer_logger.print_summary()

    out_path = args.out_json or args.save_json or f"results_fourway_{args.model}_{args.dataset}_T{args.T}_bs{args.batch_size}_{args.weight_init}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Result: {out_path}")


if __name__ == "__main__":
    main()
