# SNN稀疏卷积算子库 Demo — CIFAR-10版
# 针对 Spiking-ResNet34 + CIFAR-10 测试集
#
# 运行：
#   python snn_sparse_conv_resnet34_v2.py --T 32 --batch_size 32
#   python snn_sparse_conv_resnet34_v2.py --T 16 --batch_size 32

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from spikingjelly.activation_based.model import spiking_resnet
from spikingjelly.activation_based import functional as sj_func
from spikingjelly.activation_based import neuron as sj_neuron
import triton
import triton.language as tl

# =============================================================================
# 0. 注册表：已知产生脉冲输出的算子类型
# =============================================================================
SPIKE_OUTPUT_OPS = (
    sj_neuron.LIFNode,
    sj_neuron.IFNode,
    sj_neuron.ParametricLIFNode,
)

# =============================================================================
# 1. Block 大小选择策略
#    H/W >= 56  -> Block=16  (layer1: 56x56, layer2: 28x28)
#    14<=H/W<56 -> Block=8   (layer3: 14x14)
#    H/W <= 7   -> None      (layer4: 7x7，跳过)
# =============================================================================
def select_block_size(H: int, W: int):
    spatial = min(H, W)
    if spatial >= 56:
        return 16
    elif spatial >= 14:
        return 8
    else:
        return None


# =============================================================================
# 2. Triton Kernels
# =============================================================================

@triton.jit
def prescan_kernel(
    x_ptr, flags_ptr,
    N, C, H, W,
    GRID_H, GRID_W,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Stage-1: 扫描每个block，标记是否非零"""
    pid = tl.program_id(0)
    total = N * C * GRID_H * GRID_W
    if pid >= total:
        return

    gw = pid % GRID_W
    tmp = pid // GRID_W
    gh = tmp % GRID_H
    tmp = tmp // GRID_H
    c  = tmp % C
    n  = tmp // C

    offs_h = gh * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = gw * BLOCK_W + tl.arange(0, BLOCK_W)
    hh = offs_h[:, None]
    ww = offs_w[None, :]
    mask = (hh < H) & (ww < W)

    val = tl.load(x_ptr + ((n * C + c) * H + hh) * W + ww, mask=mask, other=0.0)
    is_nz = tl.max(tl.abs(val)) > 1e-6
    tl.store(flags_ptr + pid, is_nz.to(tl.int32))


@triton.jit
def sparse_conv3x3_kernel(
    x_ptr, y_ptr,
    idx_ptr, num_nz,
    N, C, H, W,
    GRID_H, GRID_W,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Stage-2: 只对非零block做3x3卷积"""
    pid = tl.program_id(0)
    if pid >= num_nz:
        return

    orig = tl.load(idx_ptr + pid)
    gw = orig % GRID_W
    tmp = orig // GRID_W
    gh = tmp % GRID_H
    tmp = tmp // GRID_H
    c  = tmp % C
    n  = tmp // C

    offs_h = gh * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = gw * BLOCK_W + tl.arange(0, BLOCK_W)
    hh = offs_h[:, None]
    ww = offs_w[None, :]
    mask = (hh < H) & (ww < W)

    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    for kh in range(-1, 2):
        for kw in range(-1, 2):
            h_idx = hh + kh
            w_idx = ww + kw
            m = mask & (h_idx >= 0) & (h_idx < H) & (w_idx >= 0) & (w_idx < W)
            acc += tl.load(
                x_ptr + ((n * C + c) * H + h_idx) * W + w_idx,
                mask=m, other=0.0
            )
    tl.store(y_ptr + ((n * C + c) * H + hh) * W + ww, acc, mask=mask)


@triton.jit
def dense_conv3x3_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    GRID_H, GRID_W,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """基准对照：稠密3x3卷积，1D grid避免z维超限"""
    pid = tl.program_id(0)
    total = N * C * GRID_H * GRID_W
    if pid >= total:
        return

    gw = pid % GRID_W
    tmp = pid // GRID_W
    gh = tmp % GRID_H
    tmp = tmp // GRID_H
    c  = tmp % C
    n  = tmp // C

    offs_h = gh * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = gw * BLOCK_W + tl.arange(0, BLOCK_W)
    hh = offs_h[:, None]
    ww = offs_w[None, :]
    mask = (hh < H) & (ww < W)

    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    for kh in range(-1, 2):
        for kw in range(-1, 2):
            h_idx = hh + kh
            w_idx = ww + kw
            m = mask & (h_idx >= 0) & (h_idx < H) & (w_idx >= 0) & (w_idx < W)
            acc += tl.load(
                x_ptr + ((n * C + c) * H + h_idx) * W + w_idx,
                mask=m, other=0.0
            )
    tl.store(y_ptr + ((n * C + c) * H + hh) * W + ww, acc, mask=mask)


# =============================================================================
# 3. Python 封装
# =============================================================================

def run_sparse_conv(feat: torch.Tensor, block: int):
    """两阶段稀疏卷积，feat: [T*B, C, H, W]，返回 (output, stage2_ms)"""
    N, C, H, W = feat.shape
    GRID_H = triton.cdiv(H, block)
    GRID_W = triton.cdiv(W, block)
    total  = N * C * GRID_H * GRID_W

    # Stage-1: prescan（不计入计时）
    flags = torch.empty(total, dtype=torch.int32, device=feat.device)
    prescan_kernel[(total,)](
        feat, flags, N, C, H, W, GRID_H, GRID_W,
        BLOCK_H=block, BLOCK_W=block
    )
    torch.cuda.synchronize()

    nz_idx = flags.nonzero(as_tuple=False).squeeze(1).int()
    num_nz = nz_idx.numel()
    y = torch.zeros_like(feat)
    if num_nz == 0:
        return y, 0.0

    # Stage-2: 只对非零block做卷积（计时）
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    sparse_conv3x3_kernel[(num_nz,)](
        feat, y, nz_idx, num_nz,
        N, C, H, W, GRID_H, GRID_W,
        BLOCK_H=block, BLOCK_W=block
    )
    end.record()
    torch.cuda.synchronize()
    return y, start.elapsed_time(end)


def run_dense_conv(feat: torch.Tensor, block: int):
    """稠密卷积基准，feat: [T*B, C, H, W]，返回 (output, ms)"""
    N, C, H, W = feat.shape
    GRID_H = triton.cdiv(H, block)
    GRID_W = triton.cdiv(W, block)
    total  = N * C * GRID_H * GRID_W
    y = torch.empty_like(feat)
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    dense_conv3x3_kernel[(total,)](
        feat, y, N, C, H, W, GRID_H, GRID_W,
        BLOCK_H=block, BLOCK_W=block
    )
    end.record()
    torch.cuda.synchronize()
    return y, start.elapsed_time(end)


def run_cudnn_conv(feat: torch.Tensor, module: torch.nn.Conv2d):
    """cuDNN conv基准，用F.conv2d直接调用，绕过spikingjelly的shape检查"""
    import torch.nn.functional as F
    weight = module.weight
    bias   = module.bias
    stride  = module.stride
    padding = module.padding
    dilation = module.dilation
    groups   = module.groups
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        y = F.conv2d(feat, weight, bias, stride, padding, dilation, groups)
    end.record()
    torch.cuda.synchronize()
    return y, start.elapsed_time(end)


# =============================================================================
# 4. 网络分析：识别脉冲后继Conv层
#    策略：在同一BasicBlock内，每个LIF节点往后找最近的
#          3x3/stride=1/padding=1 Conv2d
# =============================================================================

class ConvLayerInfo:
    def __init__(self, name, module, block_size, H, W):
        self.name        = name
        self.module      = module
        self.block_size  = block_size
        self.H, self.W   = H, W
        self.total_dense_ms  = 0.0
        self.total_sparse_ms = 0.0
        self.total_cudnn_ms  = 0.0
        self.total_zeros     = 0
        self.total_elems     = 0


def analyze_network(model, sample_input, device):
    # 先跑一次，记录每个Conv的输入shape
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

    conv_infos    = {}
    visited_convs = set()

    for i, (name, module) in enumerate(module_list):
        if not isinstance(module, SPIKE_OUTPUT_OPS):
            continue

        # 搜索窗口：向后最多找10个模块
        # sn1 -> 同block的conv2（距离约2步）
        # sn2 -> 下一个block的conv1（距离约3步，跨block边界）
        # 不限制parent，用固定窗口覆盖跨block情况
        for j in range(i + 1, min(i + 10, len(module_list))):
            next_name, next_module = module_list[j]

            if not isinstance(next_module, nn.Conv2d):
                continue
            if not (next_module.kernel_size == (3, 3) and
                    next_module.stride      == (1, 1) and
                    next_module.padding     == (1, 1)):
                continue
            # 已注册则跳过继续找（不break）
            if next_name in visited_convs:
                continue

            ishape = input_shapes.get(next_name)
            if ishape is None:
                continue
            if len(ishape) == 5:
                H, W = ishape[3], ishape[4]
            elif len(ishape) == 4:
                H, W = ishape[2], ishape[3]
            else:
                continue

            block = select_block_size(H, W)
            if block is None:
                print(f"  [SKIP ] {next_name}: H={H},W={W} <= 7，跳过")
                visited_convs.add(next_name)
                break

            conv_infos[next_name] = ConvLayerInfo(next_name, next_module, block, H, W)
            visited_convs.add(next_name)
            print(f"  [TARGET] LIF={name:<35} -> Conv={next_name:<35} H={H},W={W},Block={block}")
            break

    return conv_infos


# =============================================================================
# 5. 主流程
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T",           type=int,   default=16)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--v_threshold", type=float, default=0.5)
    parser.add_argument("--power",       type=float, default=250.0, help="GPU平均功耗(W)")
    parser.add_argument("--warmup",      type=int,   default=5)
    parser.add_argument("--data_root",   type=str,   default="../../data",
                        help="数据根目录，CIFAR-10会自动下载到此处")
    args = parser.parse_args()

    device = torch.device("cuda")

    # ---- 构建模型 ----
    print("[1/4] 构建 Spiking-ResNet34 ...")
    model = spiking_resnet.spiking_resnet34(
        pretrained=True,
        spiking_neuron=sj_neuron.LIFNode,
        surrogate_function=sj_neuron.LIFNode().surrogate_function,
        detach_reset=True,
    )
    for m in model.modules():
        if isinstance(m, sj_neuron.LIFNode):
            m.v_threshold = args.v_threshold
    sj_func.set_step_mode(model, step_mode="m")
    model.to(device).eval()

    # ---- 数据集 ----
    # CIFAR-10: 10类，测试集10000张，原始32x32，resize到224x224
    print(f"[2/4] 加载 CIFAR-10 测试集 (root={args.data_root}) ...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.CIFAR10(
        root=args.data_root, train=False, download=True, transform=transform
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=8, pin_memory=True)
    print(f"  测试集共 {len(ds)} 张，{len(loader)} 个 batch")

    # ---- 分析网络 ----
    print("[3/4] 分析网络，识别脉冲后继 Conv 层 ...")
    imgs_s, _ = next(iter(loader))
    sample_input = torch.bernoulli(
        imgs_s[:4].to(device).unsqueeze(0).repeat(args.T, 1, 1, 1, 1).clamp(0, 1)
    )
    conv_infos = analyze_network(model, sample_input, device)

    if not conv_infos:
        print("未找到符合条件的目标层，退出。")
        return
    print(f"\n共找到 {len(conv_infos)} 个目标 Conv 层\n")

    # ---- 注册 hook 捕获 Conv 输入的脉冲特征图 ----
    captured = {}
    def make_capture_hook(name):
        def hook(m, inp, out):
            x = inp[0].detach()
            if x.dim() == 5:
                T, B, C, H, W = x.shape
                x = x.reshape(T * B, C, H, W)
            captured[name] = x
        return hook

    hook_handles = []
    for name, info in conv_infos.items():
        h = info.module.register_forward_hook(make_capture_hook(name))
        hook_handles.append(h)

    # ---- 预热 ----
    print("[4/4] 开始基准测试 ...")
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

    # ---- 正式测试 ----
    for batch_idx, (imgs, _) in enumerate(loader):
        inp = torch.bernoulli(
            imgs.to(device).unsqueeze(0).repeat(args.T, 1, 1, 1, 1).clamp(0, 1)
        )
        sj_func.reset_net(model)
        with torch.no_grad():
            _ = model(inp)

        for name, info in conv_infos.items():
            feat = captured.get(name)
            if feat is None:
                continue
            block = info.block_size

            info.total_zeros += (feat.abs() <= 1e-6).sum().item()
            info.total_elems += feat.numel()

            _, dense_ms  = run_dense_conv(feat, block)
            _, sparse_ms = run_sparse_conv(feat, block)
            _, cudnn_ms  = run_cudnn_conv(feat, info.module)
            info.total_dense_ms  += dense_ms
            info.total_sparse_ms += sparse_ms
            info.total_cudnn_ms  += cudnn_ms

        if (batch_idx + 1) % 50 == 0:
            pct = (batch_idx + 1) / len(loader) * 100
            print(f"  [{pct:5.1f}%] batch {batch_idx+1}/{len(loader)}")

    for h in hook_handles:
        h.remove()

    # ---- 输出报告 ----
    print("\n" + "=" * 80)
    print(f"{'SNN Sparse Conv — Spiking-ResNet34 on CIFAR-10':^70}")
    print(f"{'T=' + str(args.T) + '  BS=' + str(args.batch_size) + '  Power=' + str(args.power) + 'W':^70}")
    print("=" * 70)
    print(f"{'Layer':<38} {'Blk':>4} {'H':>4} {'Sparsity':>9} {'vs Triton':>10} {'vs cuDNN':>9}")
    print("-" * 80)

    total_d_j = 0.0
    total_s_j = 0.0

    for name, info in conv_infos.items():
        if info.total_elems == 0:
            continue
        sparsity = info.total_zeros / info.total_elems * 100
        speedup  = info.total_dense_ms / max(info.total_sparse_ms, 1e-9)
        ed       = (info.total_dense_ms  / 1000.0) * args.power
        es       = (info.total_sparse_ms / 1000.0) * args.power
        esave    = ed / max(es, 1e-9)
        total_d_j += ed
        total_s_j += es

        cudnn_speedup = info.total_cudnn_ms / max(info.total_sparse_ms, 1e-9)
        sname = name if len(name) <= 37 else "..." + name[-34:]
        print(f"{sname:<38} {info.block_size:>4} {info.H:>4} "
              f"{sparsity:>8.2f}% {speedup:>9.2f}x {cudnn_speedup:>8.2f}x")

    print("-" * 80)
    all_d = sum(i.total_dense_ms  for i in conv_infos.values())
    all_s = sum(i.total_sparse_ms for i in conv_infos.values())
    all_c = sum(i.total_cudnn_ms  for i in conv_infos.values())
    print(f"{'[TOTAL]':<38} {'':>4} {'':>4} {'':>9} "
          f"{all_d/max(all_s,1e-9):>9.2f}x {all_c/max(all_s,1e-9):>8.2f}x")
    print(f"\n  Dense  Energy : {total_d_j:.4f} J")
    print(f"  Sparse Energy : {total_s_j:.4f} J")
    print(f"  Energy Saving : {(1 - total_s_j/max(total_d_j,1e-9))*100:.2f}%")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
    #python snn_sparse_conv_resnet34_cifar10.py --T 32 --batch_size 32
    #python snn_sparse_conv_resnet34_cifar10.py --T 16 --batch_size 32