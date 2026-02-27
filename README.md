# SparseFlow — Project Instructions

## 项目定位

SparseFlow 是一个即插即用的 **SNN 稀疏推理加速库**，利用脉冲神经网络（SNN）中 LIF 神经元输出的天然高稀疏性，通过跳过全零 block 的计算来获得实质性的性能和能耗收益。

用户只需一行代码即可将已有 SNN 网络替换为稀疏加速版本：

```python
import sparseflow
model = sparseflow.optimize(model)  # 自动识别并替换所有支持的算子
```

---

## 核心 Idea

SNN 网络的 LIF 神经元输出脉冲数据，具有天然的高稀疏性（大量元素为 0）。传统稠密算子（包括 cuDNN）对全零 block 仍然执行完整计算，浪费算力。通过对脉冲数据进行分 block 预筛选，跳过全零 block 的计算，可以获得实质性的性能和能耗收益。

---

## 方法设计

### 两阶段 Triton Kernel

- **Stage-1 prescan**：轻量扫描所有 block，生成非零 block 索引列表（不计入主计算时间）
- **Stage-2 sparse conv**：只对非零 block 执行卷积（如 3×3），零 block 完全跳过

### 自动化算子替换框架

1. **注册表识别** (`Core/registry.py`)：识别 LIF / IF / ParametricLIF 等脉冲输出算子，延迟导入 spikingjelly 避免强依赖
2. **网络拓扑分析** (`Core/analyzer.py`)：
   - **主路径**：使用 `torch.fx` 符号追踪构建计算图，通过 `node.users` BFS 递归查找下游 Conv2d
   - **透明层穿透**：Dropout / Flatten / Identity / Pooling / BN / reshape 等不阻断搜索
   - **分叉支持**：同一脉冲源可指向多个后继 Conv2d（ResNet 主路 + Shortcut）
   - **Fallback**：fx 追踪失败时自动回退到基于 forward hook 的线性搜索
3. **收益评估与动态分块** (`Utils/block_selector.py`)：
   - `H ≥ 32` → Block = 16（大型图，效率最高）
   - `16 ≤ H < 32` → Block = 8（中型图）
   - `8 ≤ H < 16` → Block = 4（小型图，优化下界）
   - `H < 8` → 不替换（微型图，Gating 开销过大）
4. **算子替换** (`Core/replacer.py`)：按 dot-separated module name 原地替换 Conv2d 为 SparseConv2d

### 仅替换 Conv2d 的设计决策

基于 Benchmark 实测数据：
- **Conv2d**：稀疏加速效果显著（vs cuDNN 13~90x），是主要优化目标
- **BatchNorm2d**：逐元素运算，原生 cuDNN/ATen 已高度优化，自定义 Triton BN 反而慢 3x，**不替换**
- **Linear**：FC 层通常在网络出口，特征已 flatten，PyTorch BLAS 库极其高效，**不替换**

---

## 算子支持

| 算子 | 说明 | 状态 |
|------|------|------|
| Conv2d 3×3 | 标准卷积，最常见 | ✅ Triton kernel + nn.Module 封装 |
| Conv2d 1×1 | ResNet 瓶颈层、降采样 | ✅ Triton kernel + nn.Module 封装 |
| Linear | 全连接 | ❌ 不替换（BLAS 已最优） |
| BatchNorm2d | 批归一化 | ❌ 不替换（cuDNN 已最优） |

---

## 项目结构

```
SparseFlow/
├── Core/                            # 自动化算子替换框架
│   ├── __init__.py
│   ├── registry.py                  #   脉冲算子注册表（延迟导入 spikingjelly）
│   ├── analyzer.py                  #   torch.fx 计算图分析 + fallback 线性搜索
│   └── replacer.py                  #   模块替换（仅 Conv2d）
│
├── Kernels/                         # Triton GPU kernels
│   ├── conv2d.py                    #   ✅ prescan + sparse_conv3x3 + sparse_conv1x1
│   ├── linear.py                    #   稀疏 Linear kernel（benchmark 用）
│   └── batchnorm2d.py               #   稀疏 BN（benchmark 用）
│
├── Ops/                             # nn.Module 封装层
│   ├── sparse_conv2d.py             #   ✅ SparseConv2d
│   ├── sparse_linear.py             #   SparseLinear（benchmark 用）
│   └── sparse_batchnorm2d.py        #   SparseBatchNorm2d（benchmark 用）
│
├── Utils/
│   └── block_selector.py            #   收益评估与动态分块
│
├── Benchmark/
│   ├── bench_resnet.py              #   ✅ 通用 benchmark: resnet34/50/101/152 × cifar10/100
│   ├── test_correctness.py          #   ✅ 正确性验证
│   └── run_all.sh                   #   批量运行脚本
│
├── Diy/                             # 早期实验代码 (legacy)
├── Instructions.md                  # 本文件
└── README.md
```

### 三层架构

| 层级 | 目录 | 职责 |
|------|------|------|
| **Kernel 层** | `Kernels/` | Triton GPU kernel，执行实际计算 |
| **算子封装层** | `Ops/` | 将 kernel 包装为 `nn.Module`，管理 weight/bias、处理输入 shape、提供 fallback |
| **框架层** | `Core/` | 自动识别网络中的替换目标，调用 Ops 层完成替换 |

### 导入约定

项目**不作为 pip 包安装**，所有跨目录引用使用 `sys.path` + 相对路径：

```python
import sys
from pathlib import Path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Kernels.conv2d import sparse_conv2d_forward
from Core.registry import SpikeOpRegistry
```

---

## 关键实现细节

### Core/analyzer.py — 计算图搜索

```
优先路径: torch.fx.symbolic_trace(model)
  → 遍历 graph.nodes 找到所有 spike 源
  → BFS 递归 node.users，穿透透明层
  → 找到所有下游 Conv2d（支持分叉）

Fallback 路径（fx 失败时）:
  → 基于 named_modules 线性搜索
  → 搜索窗口 = 15，穿透 BN/Dropout/Pooling
  → 同一脉冲源可匹配多个后继 Conv2d
```

### Triton Kernel 设计 (`Kernels/conv2d.py`)

- **prescan_kernel**: 通用预扫描，按 (N, C_in, GRID_H, GRID_W) 展开
- **sparse_conv3x3_weighted_kernel**: scatter 模式，`tl.atomic_add` 累加
- **sparse_conv1x1_weighted_kernel**: 同 scatter 模式，无 3×3 循环
- **dense_conv3x3_kernel**: 基准对照用的稠密 box filter

### SparseConv2d (`Ops/sparse_conv2d.py`)

- `from_dense(conv, block_size)`: 从 nn.Conv2d 拷贝权重
- 自动处理 5D (T, N, C, H, W) 输入
- Triton 不可用时 fallback 到 `F.conv2d`
- `_last_sparse_ms` 属性供 profiler 读取

### 多 GPU 支持

所有 `torch.cuda.synchronize()` 显式传入 `device` 参数：
```python
torch.cuda.synchronize(x.device)  # 不要用裸 synchronize()
```

Benchmark 自动选择空闲显存最大的 GPU：
```bash
python Benchmark/bench_resnet.py --model resnet50 --dataset cifar10
# 或手动指定：
python Benchmark/bench_resnet.py --model resnet50 --dataset cifar10 --gpu 2
```

---

## Benchmark 实测结果 (Spiking-ResNet50, CIFAR-10, T=16)

| 层 | H | Block | Sparsity | vs cuDNN |
|----|---|-------|----------|----------|
| layer1.0.conv2 | 56 | 16 | 98.5% | 13.1x |
| layer1.1.conv2 | 56 | 16 | 98.9% | 15.6x |
| layer1.2.conv2 | 56 | 16 | 99.4% | 35.4x |
| layer2.1.conv2 | 28 | 8 | 100.0% | 72.2x |
| layer2.2.conv2 | 28 | 8 | 99.9% | 69.6x |
| layer3.1.conv2 | 14 | 8 | 99.7% | 38.3x |

---

## 协作约定

- 所有 kernel 使用 Triton 编写，遵循两阶段（prescan + sparse compute）模式
- 每个算子需提供 nn.Module 封装（放在 `Ops/`）
- 新算子开发流程：kernel (`Kernels/`) → nn.Module (`Ops/`) → 注册到 analyzer → 正确性测试 → benchmark
- 代码注释中英文均可
- kernel 中 `tl.constexpr` 参数在 JIT 时确定，运行时参数用普通参数
- 所有 `torch.cuda.synchronize()` 必须带 device 参数