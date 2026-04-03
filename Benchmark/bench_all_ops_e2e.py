import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import copy
import json
import os
from collections import Counter, OrderedDict
from inspect import signature

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from spikingjelly.activation_based import functional as sj_func
from spikingjelly.activation_based import neuron as sj_neuron
from spikingjelly.activation_based.model import sew_resnet, spiking_resnet, spiking_vgg

from Core.analyzer import NetworkAnalyzer
from Core.registry import SpikeOpRegistry
from Core.replacer import ModuleReplacer
from Models.spikformer_github import MODEL_BUILDERS as SPIKFORMER_MODEL_BUILDERS
from Models.sdtv1_github import MODEL_BUILDERS as SDTV1_MODEL_BUILDERS
from Models.qkformer_github import MODEL_BUILDERS as QKFORMER_MODEL_BUILDERS


_EXTERNAL_MODEL_SOURCES = [
    SPIKFORMER_MODEL_BUILDERS,
    SDTV1_MODEL_BUILDERS,
    QKFORMER_MODEL_BUILDERS,
]
EXTERNAL_MODEL_BUILDERS = OrderedDict()
for _src in _EXTERNAL_MODEL_SOURCES:
    EXTERNAL_MODEL_BUILDERS.update(_src)


def _discover_model_builders():
    modules = [
        ("resnet", spiking_resnet, "spiking_"),
        ("vgg", spiking_vgg, "spiking_"),
        ("sew_resnet", sew_resnet, "sew_"),
    ]
    builders = OrderedDict()
    for _, mod, prefix in modules:
        for attr_name in dir(mod):
            if not attr_name.startswith(prefix):
                continue
            fn = getattr(mod, attr_name)
            if callable(fn):
                builders[attr_name] = fn
    for source in _EXTERNAL_MODEL_SOURCES:
        for name, fn in source.items():
            builders[name] = fn
    return builders


MODEL_BUILDERS = _discover_model_builders()


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


def build_model(model_name, device, v_threshold=1.0, weight_init="random", sew_cnf=None, T=None, num_classes=None, seed=42):
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
    # Only external transformer builders consume T at construction time.
    if model_name in EXTERNAL_MODEL_BUILDERS:
        builder_kwargs["T"] = T
    filtered = _filter_builder_kwargs(builder, builder_kwargs)
    set_random_seed(seed)
    model = builder(**filtered)
    model.to(device).eval()
    sj_func.set_step_mode(model, "m")
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
        return img, 0


def build_dataset(dataset_name, data_root):
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset_name == "cifar10":
        return datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform), 10
    if dataset_name == "cifar100":
        return datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform), 100
    if dataset_name == "imagenet_val_flat":
        imagenet_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        return FlatImageFolderDataset(root=data_root, transform=imagenet_transform), 1000
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def make_spike_input(imgs, T, device, spike_mode="normalized_bernoulli"):
    imgs = imgs.to(device)
    if spike_mode in ("normalized_bernoulli", "raw_bernoulli"):
        rates = imgs.clamp(0, 1)
        return torch.bernoulli(rates.unsqueeze(0).repeat(T, 1, 1, 1, 1))
    if spike_mode == "raw_repeat":
        return imgs.unsqueeze(0).repeat(T, 1, 1, 1, 1)
    raise ValueError(f"Unknown spike_mode: {spike_mode}")


def measure_e2e_latency(model, loader, device, T, warmup=10, max_batches=None, spike_mode="normalized_bernoulli"):
    for i, (imgs, _) in enumerate(loader):
        if i >= warmup:
            break
        inp = make_spike_input(imgs, T, device, spike_mode=spike_mode)
        sj_func.reset_net(model)
        with torch.no_grad():
            _ = model(inp)
    torch.cuda.synchronize(device)

    total_ms = 0.0
    num_batches = 0
    for imgs, _ in loader:
        if max_batches is not None and num_batches >= max_batches:
            break
        inp = make_spike_input(imgs, T, device, spike_mode=spike_mode)
        sj_func.reset_net(model)
        torch.cuda.synchronize(device)
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        with torch.no_grad():
            _ = model(inp)
        end_evt.record()
        torch.cuda.synchronize(device)
        total_ms += start_evt.elapsed_time(end_evt)
        num_batches += 1
    avg_ms = total_ms / max(num_batches, 1)
    return avg_ms, total_ms, num_batches


def verify_consistency(model_dense, model_replaced, loader, device, T, num_batches=5, spike_mode="normalized_bernoulli"):
    cosine_sum = 0.0
    agree_sum = 0.0
    global_max_abs = 0.0
    count = 0
    for imgs, _ in loader:
        if count >= num_batches:
            break
        inp = make_spike_input(imgs, T, device, spike_mode=spike_mode)

        sj_func.reset_net(model_dense)
        with torch.no_grad():
            out_dense = model_dense(inp)

        sj_func.reset_net(model_replaced)
        with torch.no_grad():
            out_replaced = model_replaced(inp)

        if out_dense.dim() == 3:
            out_dense = out_dense.mean(dim=0)
        if out_replaced.dim() == 3:
            out_replaced = out_replaced.mean(dim=0)

        diff = (out_replaced - out_dense).float()
        global_max_abs = max(global_max_abs, diff.abs().max().item())

        flat_a = out_replaced.flatten().float()
        flat_b = out_dense.flatten().float()
        if flat_a.norm() < 1e-8 and flat_b.norm() < 1e-8:
            cos = 1.0
        elif flat_a.norm() < 1e-8 or flat_b.norm() < 1e-8:
            cos = 0.0
        else:
            cos = F.cosine_similarity(flat_a.unsqueeze(0), flat_b.unsqueeze(0)).item()
        cosine_sum += cos

        pred_dense = out_dense.argmax(dim=-1)
        pred_replaced = out_replaced.argmax(dim=-1)
        agree_sum += (pred_dense == pred_replaced).float().mean().item()
        count += 1

    n = max(count, 1)
    return cosine_sum / n, agree_sum / n, global_max_abs


def main():
    parser = argparse.ArgumentParser(description="SparseFlow Core all-ops E2E benchmark")
    parser.add_argument("--model", type=str, default="spiking_resnet18", choices=list(MODEL_BUILDERS.keys()))
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "imagenet_val_flat"])
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--v_threshold", type=float, default=1.0)
    parser.add_argument("--weight_init", type=str, default="random", choices=["random", "pretrained"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sew_cnf", type=str, default=None)
    parser.add_argument("--power", type=float, default=250.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max_batches", type=int, default=0, help="0 means all test batches")
    parser.add_argument("--verify_batches", type=int, default=10)
    parser.add_argument("--core_only_linear", action="store_true",
                        help="Replace only nn.Linear targets after Core analysis.")
    parser.add_argument("--fuse_conv_lif", action="store_true",
                        help="Enable Conv2d+LIF fused replacement in Core pipeline (default: disabled).")
    parser.add_argument("--spike_mode", type=str, default="normalized_bernoulli",
                        choices=["normalized_bernoulli", "raw_bernoulli", "raw_repeat"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--out_json", type=str, default="")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    set_random_seed(args.seed)

    ds, num_classes = build_dataset(args.dataset, args.data_root)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model_dense = build_model(
        model_name=args.model,
        device=device,
        v_threshold=args.v_threshold,
        weight_init=args.weight_init,
        sew_cnf=args.sew_cnf,
        T=args.T,
        num_classes=num_classes,
        seed=args.seed,
    )

    imgs_s, _ = next(iter(loader))
    sample_input = make_spike_input(imgs_s[:4], args.T, device, spike_mode=args.spike_mode)

    print(f"{'=' * 90}")
    print(f"{'SparseFlow Core All-Ops E2E Benchmark':^90}")
    print(f"{'=' * 90}")
    print(f"  Device:      {device} ({torch.cuda.get_device_name(device)})")
    print(f"  Model:       {args.model}")
    print(f"  Dataset:     {args.dataset}")
    print(f"  T:           {args.T}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Fuse Conv+LIF: {'ON' if args.fuse_conv_lif else 'OFF'}")
    print()

    registry = SpikeOpRegistry.default()
    analyzer = NetworkAnalyzer(registry)
    targets = analyzer.analyze(model_dense, sample_input=sample_input)
    if not targets:
        print("No replaceable layers found.")
        return

    if args.core_only_linear:
        linear_targets = [t for t in targets if t.op_type == "linear"]
        print(f"  Linear-only filter: {len(linear_targets)}/{len(targets)} targets kept")
        targets = linear_targets
        if not targets:
            print("No linear targets found after filtering.")
            return

    op_counter = Counter(t.op_type for t in targets)
    print(f"  Replaceable layers: {len(targets)}")
    for op_name, count in sorted(op_counter.items()):
        print(f"    - {op_name}: {count}")
    print()

    model_replaced = copy.deepcopy(model_dense)
    replacer = ModuleReplacer(verbose=True)
    replace_summary = replacer.replace(
        model_replaced,
        targets,
        block_sizes=None,
        static_zero_layers=set(),
        disable_static_zero=True,
        only_static_zero=False,
        enable_fused_conv_lif=args.fuse_conv_lif,
    )
    replaced, sparse_count, fused_count, static_zero_count, dense_keep_count = replace_summary

    max_batches = args.max_batches if args.max_batches > 0 else None
    dense_avg, dense_total, dense_n = measure_e2e_latency(
        model_dense, loader, device, args.T, warmup=args.warmup, max_batches=max_batches, spike_mode=args.spike_mode
    )
    repl_avg, repl_total, repl_n = measure_e2e_latency(
        model_replaced, loader, device, args.T, warmup=args.warmup, max_batches=max_batches, spike_mode=args.spike_mode
    )
    speedup = dense_avg / repl_avg if repl_avg > 1e-6 else float("inf")

    dense_energy = (dense_total / 1000.0) * args.power
    repl_energy = (repl_total / 1000.0) * args.power
    energy_saving = (1.0 - repl_energy / max(dense_energy, 1e-9)) * 100.0

    avg_cos, avg_agree, max_abs = verify_consistency(
        model_dense,
        model_replaced,
        loader,
        device,
        args.T,
        num_batches=args.verify_batches,
        spike_mode=args.spike_mode,
    )
    consistency_ok = avg_cos > 0.999 and max_abs < 0.1

    print(f"\n  Dense:     {dense_avg:.2f} ms/batch (total={dense_total:.1f} ms, {dense_n} batches)")
    print(f"  Replaced:  {repl_avg:.2f} ms/batch (total={repl_total:.1f} ms, {repl_n} batches)")
    print(f"  Speedup:   {speedup:.3f}x")
    print(f"  Energy save (estimated): {energy_saving:.2f}%")
    print(f"  Consistency: cos={avg_cos:.8f}, agree={avg_agree * 100:.2f}%, max_abs={max_abs:.6f} -> {'PASS' if consistency_ok else 'FAIL'}")

    results = {
        "script_mode": "core_all_ops_e2e",
        "model": args.model,
        "dataset": args.dataset,
        "T": args.T,
        "batch_size": args.batch_size,
        "gpu": torch.cuda.get_device_name(device),
        "weight_init": args.weight_init,
        "seed": args.seed,
        "spike_mode": args.spike_mode,
        "fuse_conv_lif": bool(args.fuse_conv_lif),
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

    out_path = args.out_json or (
        f"results_core_all_ops_e2e_{args.model}_{args.dataset}_T{args.T}_bs{args.batch_size}_{args.weight_init}.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Result JSON saved: {out_path}")


if __name__ == "__main__":
    main()
