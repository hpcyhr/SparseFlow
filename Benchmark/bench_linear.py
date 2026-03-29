import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import copy
import json
import math
import time
from collections import OrderedDict
import os
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from spikingjelly.activation_based import functional as sj_func
from spikingjelly.activation_based import neuron as sj_neuron

from Ops.sparse_linear import SparseLinear


DEVICE = None
SPIKE_OPS = (sj_neuron.LIFNode, sj_neuron.IFNode, sj_neuron.ParametricLIFNode)


# =========================================================
# Utils
# =========================================================

def sync():
    torch.cuda.synchronize(DEVICE)


def make_event():
    return torch.cuda.Event(enable_timing=True)


def set_random_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        return img, 0


def build_dataset(dataset_name, data_root):
    if dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        num_classes = 10

    elif dataset_name == "cifar100":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        ds = datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform)
        num_classes = 100

    elif dataset_name == "imagenet_val_flat":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        ds = FlatImageFolderDataset(root=data_root, transform=transform)
        num_classes = 1000

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return ds, num_classes


def make_spike_input(imgs, T, device, spike_mode="normalized_bernoulli"):
    imgs = imgs.to(device)
    if spike_mode in ("normalized_bernoulli", "raw_bernoulli"):
        rates = imgs.clamp(0, 1)
        return torch.bernoulli(rates.unsqueeze(0).repeat(T, 1, 1, 1, 1))
    elif spike_mode == "raw_repeat":
        return imgs.unsqueeze(0).repeat(T, 1, 1, 1, 1)
    else:
        raise ValueError(f"Unknown spike_mode: {spike_mode}")


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


# =========================================================
# Linear-dominant spiking network
# =========================================================

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=256):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        patch_dim = in_chans * patch_size * patch_size
        self.proj = nn.Linear(patch_dim, embed_dim)

    def forward(self, x):
        # x: [T, B, C, H, W]
        T, B, C, H, W = x.shape
        P = self.patch_size
        gh = H // P
        gw = W // P

        x = x.reshape(T, B, C, gh, P, gw, P)
        x = x.permute(0, 1, 3, 5, 2, 4, 6).contiguous()   # [T, B, gh, gw, C, P, P]
        x = x.reshape(T, B, gh * gw, C * P * P)           # [T, B, N, patch_dim]
        x = self.proj(x)                                  # [T, B, N, embed_dim]
        return x


class SpikingMLPBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, v_threshold=1.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.lif1 = sj_neuron.LIFNode(v_threshold=v_threshold)

        self.fc2 = nn.Linear(hidden_dim, dim)
        self.lif2 = sj_neuron.LIFNode(v_threshold=v_threshold)

    def forward(self, x):
        # x: [T, B, N, C]
        x = self.fc1(x)
        x = self.lif1(x)
        x = self.fc2(x)
        x = self.lif2(x)
        return x


class SpikingPatchMLP(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=256,
        depth=4,
        mlp_ratio=4.0,
        v_threshold=1.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.patch_lif = sj_neuron.LIFNode(v_threshold=v_threshold)

        self.blocks = nn.ModuleList([
            SpikingMLPBlock(embed_dim, mlp_ratio=mlp_ratio, v_threshold=v_threshold)
            for _ in range(depth)
        ])

        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: [T, B, C, H, W]
        x = self.patch_embed(x)       # [T, B, N, C]
        x = self.patch_lif(x)

        for blk in self.blocks:
            x = blk(x)

        # token average pool
        x = x.mean(dim=2)             # [T, B, C]
        x = self.head(x)              # [T, B, num_classes]
        return x


def build_model(
    dataset_name,
    num_classes,
    device,
    v_threshold=1.0,
    embed_dim=256,
    depth=4,
    mlp_ratio=4.0,
    patch_size=4,
    seed=42,
):
    if dataset_name in ("cifar10", "cifar100"):
        img_size = 32
    elif dataset_name == "imagenet_val_flat":
        img_size = 224
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    set_random_seed(seed)

    model = SpikingPatchMLP(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        mlp_ratio=mlp_ratio,
        v_threshold=v_threshold,
    ).to(device).eval()

    sj_func.set_step_mode(model, "m")
    return model


# =========================================================
# Analyze / replace Linear
# =========================================================

def analyze_linear_targets(model, sample_input):
    input_shapes = {}
    hooks = []

    def make_hook(name):
        def hook(m, inp, out):
            if isinstance(inp, (tuple, list)) and len(inp) > 0:
                x = inp[0]
                if isinstance(x, torch.Tensor):
                    input_shapes[name] = tuple(x.shape)
        return hook

    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(make_hook(name)))

    sj_func.reset_net(model)
    with torch.no_grad():
        _ = model(sample_input)

    for h in hooks:
        h.remove()

    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        ishape = input_shapes.get(name, None)
        if ishape is None:
            continue

        info = {
            "name": name,
            "module": module,
            "in_features": module.in_features,
            "out_features": module.out_features,
            "input_shape": ishape,
        }
        targets.append(info)

    return targets


def print_analysis_report(targets):
    print(f"\n  ┌{'─'*108}┐")
    print(f"  │{'Linear 候选层分析报告':^106}│")
    print(f"  ├{'─'*108}┤")
    print(f"  │ {'Layer':<42} {'InFeat':>8} {'OutFeat':>8} {'InputShape':<40} │")
    print(f"  ├{'─'*108}┤")
    for t in targets:
        name = t["name"]
        short = name if len(name) <= 41 else "..." + name[-38:]
        ishape = str(t["input_shape"])
        if len(ishape) > 40:
            ishape = ishape[:37] + "..."
        print(f"  │ {short:<42} {t['in_features']:>8} {t['out_features']:>8} {ishape:<40} │")
    print(f"  └{'─'*108}┘")
    print(f"\n  汇总: total_linear={len(targets)}")


def replace_linears(model, targets):
    replaced = 0
    for t in targets:
        name = t["name"]
        dense_linear = t["module"]
        sparse_linear = SparseLinear.from_dense(dense_linear)
        _set_module_by_name(model, name, sparse_linear)
        replaced += 1
    return replaced


# =========================================================
# Sparsity / diag
# =========================================================

def measure_linear_input_sparsity(model, targets, loader, device, T, num_batches=10,
                                  spike_mode="normalized_bernoulli"):
    sparsity_data = {t["name"]: {"zeros": 0, "total": 0} for t in targets}

    def make_hook(name):
        def hook(m, inp, out):
            x = inp[0]
            if not isinstance(x, torch.Tensor):
                return
            with torch.no_grad():
                xd = x.detach()
                sparsity_data[name]["zeros"] += (xd.abs() <= 1e-6).sum().item()
                sparsity_data[name]["total"] += xd.numel()
        return hook

    handles = []
    for t in targets:
        name = t["name"]
        mod = _get_module_by_name(model, name)
        handles.append(mod.register_forward_hook(make_hook(name)))

    count = 0
    for imgs, _ in loader:
        if count >= num_batches:
            break
        inp = make_spike_input(imgs, T, device, spike_mode=spike_mode)
        sj_func.reset_net(model)
        with torch.no_grad():
            _ = model(inp)
        count += 1

    for h in handles:
        h.remove()

    return sparsity_data


def measure_group_sparsity_linear(model, targets, loader, device, T, num_batches=5,
                                  spike_mode="normalized_bernoulli"):
    for _, mod in model.named_modules():
        if isinstance(mod, SparseLinear):
            mod.collect_diag = True

    count = 0
    for imgs, _ in loader:
        if count >= num_batches:
            break
        inp = make_spike_input(imgs, T, device, spike_mode=spike_mode)
        sj_func.reset_net(model)
        with torch.no_grad():
            _ = model(inp)
        count += 1

    target_names = {t["name"] for t in targets}
    group_data = {}
    for name, mod in model.named_modules():
        if isinstance(mod, SparseLinear) and name in target_names:
            diag = getattr(mod, "_last_diag", {})
            group_data[name] = {
                "active_group_ratio": diag.get("active_group_ratio", -1.0),
                "tile_zero_ratio": diag.get("tile_zero_ratio", -1.0),
                "nonzero_group_count": diag.get("nonzero_group_count", -1.0),
                "total_group_count": diag.get("total_group_count", -1.0),
                "tile_zero_count": diag.get("tile_zero_count", -1.0),
                "total_tile_count": diag.get("total_tile_count", -1.0),
                "sparse_compute_ms": diag.get("sparse_compute_ms", -1.0),
                "avg_active_ratio": diag.get("avg_active_ratio", -1.0),
                "zero_tiles": diag.get("zero_tiles", -1),
                "sparse_tiles": diag.get("sparse_tiles", -1),
                "denseish_tiles": diag.get("denseish_tiles", -1),
                "group_size": diag.get("group_size", -1),
                "num_groups": diag.get("num_groups", -1),
            }

    for _, mod in model.named_modules():
        if isinstance(mod, SparseLinear):
            mod.collect_diag = False

    return group_data


def print_route_report(targets, sparsity_data, group_sparsity_data):
    print(f"\n  {'Layer':<42} {'ElemSparse':>11} {'AGR':>8} {'TileZR':>8} {'Zero':>6} {'Sparse':>7} {'Dense':>7}")
    print(f"  {'-'*100}")
    avg_sparse = 0.0
    for t in targets:
        name = t["name"]
        sd = sparsity_data[name]
        sp = sd["zeros"] / max(sd["total"], 1) * 100.0
        avg_sparse += sp
        gd = group_sparsity_data.get(name, {})
        agr = gd.get("active_group_ratio", -1.0)
        tzr = gd.get("tile_zero_ratio", -1.0)
        zt = gd.get("zero_tiles", -1)
        st = gd.get("sparse_tiles", -1)
        dt = gd.get("denseish_tiles", -1)
        short = name if len(name) <= 41 else "..." + name[-38:]
        agr_s = f"{agr:.4f}" if agr >= 0 else "n/a"
        tzr_s = f"{tzr:.4f}" if tzr >= 0 else "n/a"
        print(f"  {short:<42} {sp:>10.2f}% {agr_s:>8} {tzr_s:>8} {str(zt):>6} {str(st):>7} {str(dt):>7}")

    avg_sparse /= max(len(targets), 1)
    print(f"\n  平均输入元素稀疏率: {avg_sparse:.2f}%")
    return avg_sparse


# =========================================================
# Timing / verification
# =========================================================

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
        sync()
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


def measure_mode(model, loader, device, T, warmup, spike_mode, power, label):
    avg_ms, total_ms, num_batches = measure_e2e_latency(
        model, loader, device, T, warmup=warmup, spike_mode=spike_mode
    )
    energy_j = (total_ms / 1000.0) * power
    return {
        "label": label,
        "avg_ms": avg_ms,
        "total_ms": total_ms,
        "num_batches": num_batches,
        "energy_j": energy_j,
    }


def print_mode_result(res):
    print(f"    {res['label']:<20}{res['avg_ms']:.2f} ms/batch  (total={res['total_ms']:.1f} ms, {res['num_batches']} batches)")


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
            cos = F.cosine_similarity(flat_s.unsqueeze(0), flat_b.unsqueeze(0)).item()

        cosine_sum += cos

        pred_base = out_base.argmax(dim=-1)
        pred_sparse = out_sparse.argmax(dim=-1)
        agree_sum += (pred_base == pred_sparse).float().mean().item()
        count += 1

    n = max(count, 1)
    return cosine_sum / n, agree_sum / n, global_max_abs


def measure_layer_timing(model, loader, device, T, target_names,
                         warmup=5, num_batches=20, spike_mode="normalized_bernoulli"):
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


def print_layer_profile(baseline_timing, sparse_timing, targets, e2e_baseline_ms=None, e2e_sparse_ms=None):
    print(f"\n  {'Layer':<42} {'Dense(ms)':>10} {'Sparse(ms)':>11} {'Speedup':>9}")
    print(f"  {'-'*78}")
    total_base = 0.0
    total_sparse = 0.0

    for t in targets:
        name = t["name"]
        b = baseline_timing.get(name, 0.0)
        s = sparse_timing.get(name, 0.0)
        total_base += b
        total_sparse += s
        sp = f"{b / s:.3f}x" if s > 1e-6 else "inf"
        short = name if len(name) <= 41 else "..." + name[-38:]
        print(f"  {short:<42} {b:>10.3f} {s:>11.3f} {sp:>9}")

    print(f"  {'-'*78}")
    sp = f"{total_base / total_sparse:.3f}x" if total_sparse > 1e-6 else "inf"
    print(f"  {'[LINEAR TOTAL]':<42} {total_base:>10.3f} {total_sparse:>11.3f} {sp:>9}")

    if e2e_baseline_ms is not None and e2e_baseline_ms > 1e-6:
        print(f"\n  Linear layers 占 Dense e2e:  {100.0 * total_base / e2e_baseline_ms:.2f}%")
    if e2e_sparse_ms is not None and e2e_sparse_ms > 1e-6:
        print(f"  Linear layers 占 Sparse e2e: {100.0 * total_sparse / e2e_sparse_ms:.2f}%")

    print(f"  Linear layers 节省: {total_base - total_sparse:.3f} ms/batch")


# =========================================================
# Main
# =========================================================

def main():
    global DEVICE

    parser = argparse.ArgumentParser(description="SparseFlow SparseLinear benchmark")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100", "imagenet_val_flat"])
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--v_threshold", type=float, default=1.0)
    parser.add_argument("--spike_mode", type=str, default="normalized_bernoulli",
                        choices=["normalized_bernoulli", "raw_bernoulli", "raw_repeat"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--power", type=float, default=250.0)

    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--patch_size", type=int, default=4)

    parser.add_argument("--verify_batches", type=int, default=10)
    parser.add_argument("--sparsity_batches", type=int, default=10)

    parser.add_argument("--layer_profile", action="store_true")
    parser.add_argument("--layer_profile_warmup", type=int, default=5)
    parser.add_argument("--layer_profile_batches", type=int, default=20)

    parser.add_argument("--out_json", type=str, default="")
    args = parser.parse_args()

    DEVICE = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(DEVICE)
    device = DEVICE

    print(f"\n{'='*88}")
    print(f"{'SparseFlow SparseLinear Benchmark':^88}")
    print(f"{'='*88}")
    print(f"  GPU:         {args.gpu} ({torch.cuda.get_device_name(args.gpu)})")
    print(f"  Dataset:     {args.dataset}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  T:           {args.T}")
    print(f"  Spike mode:  {args.spike_mode}")
    print(f"  Embed dim:   {args.embed_dim}")
    print(f"  Depth:       {args.depth}")
    print(f"  MLP ratio:   {args.mlp_ratio}")
    print(f"  Patch size:  {args.patch_size}")
    print(f"  Layer prof:  {'ON' if args.layer_profile else 'OFF'}")
    print()

    print("[1/6] 加载数据集 ...")
    ds, num_classes = build_dataset(args.dataset, args.data_root)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print(f"  测试集: {len(ds)} 张, {len(loader)} batches, {num_classes} classes")

    print("[2/6] 构建 linear-dominant SpikingPatchMLP ...")
    model_dense = build_model(
        dataset_name=args.dataset,
        num_classes=num_classes,
        device=device,
        v_threshold=args.v_threshold,
        embed_dim=args.embed_dim,
        depth=args.depth,
        mlp_ratio=args.mlp_ratio,
        patch_size=args.patch_size,
        seed=args.seed,
    )

    imgs_s, _ = next(iter(loader))
    sample_input = make_spike_input(imgs_s[:4], args.T, device, spike_mode=args.spike_mode)

    print("[3/6] 分析 Linear 候选层 ...")
    targets = analyze_linear_targets(model_dense, sample_input)
    print_analysis_report(targets)
    if not targets:
        print("未找到可替换的 Linear 层，退出。")
        return

    print(f"[4/6] 收集 Linear 输入稀疏率 / grouped diag ({args.sparsity_batches} batches) ...")
    sparsity_data = measure_linear_input_sparsity(
        model_dense, targets, loader, device, args.T,
        num_batches=args.sparsity_batches, spike_mode=args.spike_mode
    )

    model_sparse = copy.deepcopy(model_dense)
    replaced = replace_linears(model_sparse, targets)
    print(f"  已替换 {replaced} 个 Linear -> SparseLinear")

    group_sparsity_data = measure_group_sparsity_linear(
        model_sparse, targets, loader, device, args.T,
        num_batches=min(args.sparsity_batches, 5), spike_mode=args.spike_mode
    )

    avg_sparsity = print_route_report(targets, sparsity_data, group_sparsity_data)

    print(f"\n[5/6] 端到端延迟 (warmup={args.warmup}) ...")
    dense_res = measure_mode(model_dense, loader, device, args.T, args.warmup, args.spike_mode, args.power, "Dense Linear")
    sparse_res = measure_mode(model_sparse, loader, device, args.T, args.warmup, args.spike_mode, args.power, "SparseLinear")
    print_mode_result(dense_res)
    print_mode_result(sparse_res)

    speedup = dense_res["avg_ms"] / sparse_res["avg_ms"] if sparse_res["avg_ms"] > 1e-6 else float("inf")
    esave = (1 - sparse_res["energy_j"] / max(dense_res["energy_j"], 1e-9)) * 100.0

    if args.layer_profile:
        print(f"\n  [Layer profile] Measuring per-layer timing ...")
        target_names = [t["name"] for t in targets]
        dense_layer_timing = measure_layer_timing(
            model_dense, loader, device, args.T, target_names,
            warmup=args.layer_profile_warmup,
            num_batches=args.layer_profile_batches,
            spike_mode=args.spike_mode
        )
        sparse_layer_timing = measure_layer_timing(
            model_sparse, loader, device, args.T, target_names,
            warmup=args.layer_profile_warmup,
            num_batches=args.layer_profile_batches,
            spike_mode=args.spike_mode
        )
        print_layer_profile(
            dense_layer_timing, sparse_layer_timing, targets,
            e2e_baseline_ms=dense_res["avg_ms"],
            e2e_sparse_ms=sparse_res["avg_ms"]
        )
    else:
        dense_layer_timing = {}
        sparse_layer_timing = {}

    print(f"\n[6/6] 一致性验证 ({args.verify_batches} batches) ...")
    cos, agree, max_abs = verify_consistency(
        model_dense, model_sparse, loader, device, args.T,
        num_batches=args.verify_batches, spike_mode=args.spike_mode
    )
    ok = cos > 0.999 and max_abs < 0.1
    print(f"  SparseLinear: cos={cos:.8f}  agree={agree*100:.2f}%  max_abs={max_abs:.6f}  {'PASS' if ok else 'FAIL'}")

    print(f"\n{'='*96}")
    print(f"{'SPARSELINEAR SUMMARY':^96}")
    print(f"{'='*96}")
    print(f"  Dataset: {args.dataset}  T={args.T}  Spike: {args.spike_mode}")
    print(f"  Avg Input Sparsity: {avg_sparsity:.2f}%  Targets: {len(targets)}")
    print(f"\n  {'Mode':<24} {'Latency(ms)':>12} {'Speedup':>10} {'EnergySave':>12} {'Consistency':>12}")
    print(f"  {'-'*80}")
    print(f"  {'Dense Linear':<24} {dense_res['avg_ms']:>12.2f} {'1.000x':>10} {'0.00%':>12} {'REF':>12}")
    print(f"  {'SparseLinear':<24} {sparse_res['avg_ms']:>12.2f} {speedup:>9.3f}x {esave:>11.2f}% {('PASS' if ok else 'FAIL'):>12}")
    print(f"{'='*96}\n")

    results = {
        "script_version": "bench_sparse_linear_v1",
        "dataset": args.dataset,
        "T": args.T,
        "batch_size": args.batch_size,
        "gpu": torch.cuda.get_device_name(args.gpu),
        "spike_mode": args.spike_mode,
        "seed": args.seed,
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "mlp_ratio": args.mlp_ratio,
        "patch_size": args.patch_size,
        "num_targets": len(targets),
        "avg_input_sparsity_pct": round(avg_sparsity, 2),
        "dense": {
            "ms_per_batch": round(dense_res["avg_ms"], 2),
            "energy_j": round(dense_res["energy_j"], 4),
        },
        "sparse": {
            "ms_per_batch": round(sparse_res["avg_ms"], 2),
            "speedup": round(speedup, 4),
            "energy_saving_pct": round(esave, 2),
            "cosine_sim": round(cos, 8),
            "pred_agreement": round(agree, 8),
            "max_abs": round(max_abs, 8),
            "consistency": "PASS" if ok else "FAIL",
        },
        "targets": [
            {
                "name": t["name"],
                "in_features": t["in_features"],
                "out_features": t["out_features"],
                "input_shape": str(t["input_shape"]),
            }
            for t in targets
        ],
        "group_sparsity": group_sparsity_data,
    }

    if args.layer_profile:
        results["layer_profile"] = {
            "dense_layer_ms": {k: round(v, 4) for k, v in dense_layer_timing.items()},
            "sparse_layer_ms": {k: round(v, 4) for k, v in sparse_layer_timing.items()},
            "per_layer_speedup": {
                k: (round(dense_layer_timing[k] / sparse_layer_timing[k], 4)
                    if sparse_layer_timing.get(k, 0.0) > 1e-6 else float("inf"))
                for k in dense_layer_timing
            }
        }

    out_path = args.out_json or f"results_sparse_linear_{args.dataset}_T{args.T}_bs{args.batch_size}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  结果: {out_path}")


if __name__ == "__main__":
    main()