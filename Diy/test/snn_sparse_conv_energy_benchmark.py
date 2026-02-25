# 该脚本实现了SNN卷积稀疏计算的性能和能耗基准测试，核心思路是：
# 1. 选取ResNet-18中一个具有适度稀疏性的层作为测试对象。
# 2. 实现两个Triton卷积核：一个是标准的3x3卷积，另一个是块稀疏优化的3x3卷积。
# 3. 对选定层的输出特征图进行基准测试，比较两种卷积核的执行时间，并估算能耗。
# 4. 输出详细的性能和能耗分析结果。   


import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from spikingjelly.activation_based.model import spiking_resnet
from spikingjelly.activation_based import functional as sj_func
from spikingjelly.activation_based import neuron as sj_neuron
from spikingjelly.activation_based import surrogate as sj_surrogate
import triton
import triton.language as tl

# ==============================
# Triton Kernels (保持不变)
# ==============================
@triton.jit
def dense_conv3x3_kernel(x_ptr, y_ptr, B, C, H, W, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    pid_w, pid_h, pid_bc = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b, c = pid_bc // C, pid_bc % C
    offs_h, offs_w = pid_h * BLOCK_H + tl.arange(0, BLOCK_H), pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    hh, ww = offs_h[:, None], offs_w[None, :]
    mask = (hh < H) & (ww < W)
    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    for kh in range(-1, 2):
        for kw in range(-1, 2):
            h_idx, w_idx = hh + kh, ww + kw
            m = mask & (h_idx >= 0) & (h_idx < H) & (w_idx >= 0) & (w_idx < W)
            x_val = tl.load(x_ptr + ((b*C+c)*H + h_idx)*W + w_idx, mask=m, other=0.0)
            acc += x_val
    tl.store(y_ptr + ((b*C+c)*H + hh)*W + ww, acc, mask=mask)

@triton.jit
def blocksparse_conv3x3_kernel(x_ptr, y_ptr, B, C, H, W, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    pid_w, pid_h, pid_bc = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b, c = pid_bc // C, pid_bc % C
    offs_h, offs_w = pid_h * BLOCK_H + tl.arange(0, BLOCK_H), pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    hh, ww = offs_h[:, None], offs_w[None, :]
    mask = (hh < H) & (ww < W)
    
    x_chunk = tl.load(x_ptr + ((b*C+c)*H + hh)*W + ww, mask=mask, other=0.0)
    if tl.max(tl.abs(x_chunk)) <= 1e-6:
        return

    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    for kh in range(-1, 2):
        for kw in range(-1, 2):
            h_idx, w_idx = hh + kh, ww + kw
            m = mask & (h_idx >= 0) & (h_idx < H) & (w_idx >= 0) & (w_idx < W)
            acc += tl.load(x_ptr + ((b*C+c)*H + h_idx)*W + w_idx, mask=m, other=0.0)
    tl.store(y_ptr + ((b*C+c)*H + hh)*W + ww, acc, mask=mask)

def run_kernel_once(feat, block, mode="dense"):
    B, C, H, W = feat.shape
    y = torch.empty_like(feat)
    grid = (triton.cdiv(W, block), triton.cdiv(H, block), B * C)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    if mode == "dense":
        dense_conv3x3_kernel[grid](feat, y, B, C, H, W, BLOCK_H=block, BLOCK_W=block)
    else:
        blocksparse_conv3x3_kernel[grid](feat, y, B, C, H, W, BLOCK_H=block, BLOCK_W=block)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

# ==================
# 主流程
# ==================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block", type=int, default=16)
    parser.add_argument("--v_threshold", type=float, default=0.2)
    parser.add_argument("--power", type=float, default=250.0, help="GPU平均功耗(W)")
    args = parser.parse_args()

    device = torch.device("cuda")
    
    model = spiking_resnet.spiking_resnet18(pretrained=True, spiking_neuron=sj_neuron.LIFNode)
    for m in model.modules():
        if isinstance(m, sj_neuron.LIFNode):
            m.v_threshold = args.v_threshold
    sj_func.set_step_mode(model, step_mode="m")
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = datasets.CIFAR10(root="../../data", train=False, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    total_samples = len(ds)

    # 选层逻辑 (同前)
    imgs_first, _ = next(iter(loader))
    input_first = torch.bernoulli(imgs_first.to(device).unsqueeze(0).repeat(args.T, 1, 1, 1, 1).clamp(0, 1))
    layer_output = {}
    def get_hook(name): return lambda m, i, o: layer_output.update({name: o.detach()})
    hooks = [m.register_forward_hook(get_hook(n)) for n, m in model.named_modules() if isinstance(m, sj_neuron.BaseNode)]
    sj_func.reset_net(model)
    with torch.no_grad(): _ = model(input_first)
    for h in hooks: h.remove()

    chosen_name = None
    for stage in ["layer3", "layer2", "layer4"]:
        cands = [n for n in layer_output.keys() if n.startswith(stage)]
        for name in cands:
            fr = (layer_output[name].abs() > 1e-6).float().mean().item()
            if 0.001 < fr < 0.2:
                chosen_name = name; break
        if chosen_name: break

    # Benchmark
    target_data = {"out": None}
    def target_hook(m, i, o): target_data["out"] = o.detach()
    for n, m in model.named_modules():
        if n == chosen_name: m.register_forward_hook(target_hook)

    total_dense_time, total_sparse_time, total_zeros, total_elems = 0.0, 0.0, 0.0, 0.0
    processed_samples = 0

    print(f"\n[START] 开始全量能耗测试: T={args.T}, Block={args.block}, BatchSize={args.batch_size}")

    for imgs, _ in loader:
        B_cur = imgs.size(0)
        input_seq = torch.bernoulli(imgs.to(device).unsqueeze(0).repeat(args.T, 1, 1, 1, 1).clamp(0, 1))
        sj_func.reset_net(model)
        with torch.no_grad(): _ = model(input_seq)
        
        feat = target_data["out"].abs().mean(dim=0)
        total_zeros += (feat <= 1e-6).sum().item()
        total_elems += feat.numel()

        total_dense_time += run_kernel_once(feat, args.block, "dense")
        total_sparse_time += run_kernel_once(feat, args.block, "sparse")
        processed_samples += B_cur

    # --- 能耗计算逻辑 ---
    # 能量 (Joule) = 功率 (Watt) * 时间 (Second)
    energy_dense = (total_dense_time / 1000.0) * args.power
    energy_sparse = (total_sparse_time / 1000.0) * args.power
    # 注意：实际上Sparse算子执行时GPU核心利用率下降，功耗会更低，这里采用保守的时间比估算
    
    print("\n" + "="*50)
    print(f"{'Performance & Energy Summary':^50}")
    print("-" * 50)
    print(f"{'Target Layer':<25} | {chosen_name}")
    print(f"{'Average Sparsity (%)':<25} | {total_zeros/total_elems*100:.2f}%")
    print(f"{'Speedup (Time)':<25} | {total_dense_time/total_sparse_time:.2f}x")
    print("-" * 50)
    print(f"{'Est. Dense Energy':<25} | {energy_dense:.4f} J")
    print(f"{'Est. Sparse Energy':<25} | {energy_sparse:.4f} J")
    print(f"{'Energy Saving Ratio':<25} | {energy_dense/energy_sparse:.2f}x")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()

# 测试命令： python snn_sparse_conv_energy_benchmark.py --block 16 --T 32 --batch_size 32