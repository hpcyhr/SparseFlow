# 诊断脚本：打印 Spiking-ResNet34 的模块结构，找出 LIF 和 Conv 的位置关系

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from spikingjelly.activation_based.model import spiking_resnet
from spikingjelly.activation_based import functional as sj_func
from spikingjelly.activation_based import neuron as sj_neuron

SPIKE_OUTPUT_OPS = (
    sj_neuron.LIFNode,
    sj_neuron.IFNode,
    sj_neuron.ParametricLIFNode,
)

device = torch.device("cuda")

model = spiking_resnet.spiking_resnet34(
    pretrained=True,
    spiking_neuron=sj_neuron.LIFNode,
    surrogate_function=sj_neuron.LIFNode().surrogate_function,
    detach_reset=True,
)
sj_func.set_step_mode(model, step_mode="m")
model.to(device).eval()

# ---- 打印所有模块名和类型 ----
print("=" * 80)
print("完整模块列表 (name -> type)")
print("=" * 80)
module_list = list(model.named_modules())
for name, module in module_list:
    print(f"  {name:<55} {type(module).__name__}")

# ---- 单独打印 LIF 层及其前后各3个模块 ----
print("\n" + "=" * 80)
print("LIF 层及其上下文（前后各3个模块）")
print("=" * 80)
for i, (name, module) in enumerate(module_list):
    if isinstance(module, SPIKE_OUTPUT_OPS):
        start = max(0, i - 3)
        end   = min(len(module_list), i + 4)
        print(f"\n  >>> LIF @ index {i}: {name}")
        for j in range(start, end):
            marker = " **" if j == i else "   "
            n, m = module_list[j]
            print(f"  [{j:03d}]{marker} {n:<55} {type(m).__name__}")

# ---- 用一个 batch 跑一遍，记录各模块输出的 shape ----
print("\n" + "=" * 80)
print("各模块输出 shape（用4张图推一次）")
print("=" * 80)

T = 4
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
ds = datasets.CIFAR10(root="../../data", train=False, download=True, transform=transform)
loader = DataLoader(ds, batch_size=4, shuffle=False)
imgs, _ = next(iter(loader))
sample_input = torch.bernoulli(
    imgs.to(device).unsqueeze(0).repeat(T, 1, 1, 1, 1).clamp(0, 1)
)

output_shapes = {}
hooks = []
def make_hook(name):
    def hook(m, inp, out):
        if isinstance(out, torch.Tensor):
            output_shapes[name] = tuple(out.shape)
        # 同时记录输入 shape
        if isinstance(inp, (tuple, list)) and len(inp) > 0 and isinstance(inp[0], torch.Tensor):
            output_shapes[name + "__input"] = tuple(inp[0].shape)
    return hook

for name, module in model.named_modules():
    hooks.append(module.register_forward_hook(make_hook(name)))

sj_func.reset_net(model)
with torch.no_grad():
    _ = model(sample_input)
for h in hooks:
    h.remove()

# 只打印 LIF 和 Conv2d 层的 shape
for name, module in module_list:
    if isinstance(module, (SPIKE_OUTPUT_OPS + (nn.Conv2d,))):
        out_shape = output_shapes.get(name, "N/A")
        inp_shape = output_shapes.get(name + "__input", "N/A")
        print(f"  {name:<50} in={inp_shape}  out={out_shape}")