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

### 两阶段 Triton Kernel（Conv2d / Linear）

- **Stage-1 prescan**：轻量扫描所有 block/行，生成非零索引列表（不计入主计算时间）
- **Stage-2 sparse compute**：只对非零 block/行 执行计算，零部分完全跳过

### BatchNorm2d 稀疏优化

- 预计算全零位置的 BN 输出常数 `bias_c = (-mean_c / sqrt(var_c + eps)) * gamma_c + beta_c`
- 非零位置正常 BN → 零位置直接赋值常数（用 `torch.where` 向量化实现）

### 自动化算子替换框架

1. **注册表识别** (`Core/registry.py`)：识别 LIF / IF / ParametricLIF 等脉冲输出算子，延迟导入 spikingjelly 避免强依赖
2. **网络拓扑分析** (`Core/analyzer.py`)：遍历 named_modules，以固定搜索窗口（10 个模块）向后查找 spike_op 的后继 Conv2d / BN / Linear，通过 forward hook 推断各层输入 shape
3. **自动 block 大小选择** (`Utils/block_selector.py`)：根据特征图尺寸自动选择
   - `H ≥ 56` → Block = 16
   - `14 ≤ H < 56` → Block = 8
   - `H ≤ 7` → 跳过（不使用稀疏加速）
4. **算子替换** (`Core/replacer.py`)：按 dot-separated module name 原地替换

---

## 算子支持优先级

### P0 — 核心算子
| 算子 | 说明 | 状态 |
|------|------|------|
| Conv2d 3×3 | 标准卷积，最常见 | ✅ Triton kernel + nn.Module 已完成 |
| Conv2d 1×1 | ResNet 瓶颈层、降采样 | ✅ Triton kernel + nn.Module 已完成 |

### P1 — SNN 高频算子
| 算子 | 说明 | 状态 |
|------|------|------|
| Linear | 全连接，分类头 / QKV 投影 | ✅ Triton kernel + nn.Module 已完成 |
| BatchNorm2d | 脉冲输入全零位置用预计算常数替代 | ✅ PyTorch 向量化 + nn.Module 已完成 |

### P2 — 扩展网络结构覆盖
| 算子 | 说明 | 状态 |
|------|------|------|
| Conv2d depthwise | MobileNet 类网络 | 🔜 placeholder 已创建 |
| MultiheadAttention | Spiking Transformer 核心算子 | 🔜 placeholder 已创建 |
| ConvTranspose2d | 生成网络、分割网络 | 🔜 待开发 |

---

## 项目目录结构

```
SparseFlow/
├── Kernels/                         # Triton GPU kernels
│   ├── conv2d.py                    #   ✅ prescan + sparse_conv3x3 + sparse_conv1x1（带权重）
│   ├── linear.py                    #   ✅ linear_prescan + sparse_linear（按行跳零）
│   ├── batchnorm2d.py               #   ✅ 预计算零位置常数 + torch.where 向量化
│   ├── depthwise.py                 #   🔜 placeholder
│   └── attention.py                 #   🔜 placeholder
│
├── Ops/                             # nn.Module 封装层
│   ├── sparse_conv2d.py             #   ✅ SparseConv2d: from_dense(), 5D, fallback
│   ├── sparse_linear.py             #   ✅ SparseLinear: from_dense(), 3D/5D, fallback
│   ├── sparse_batchnorm2d.py        #   ✅ SparseBatchNorm2d: from_dense(), 5D, 缓存常数
│   └── sparse_attention.py          #   🔜 placeholder
│
├── Core/                            # 自动化算子替换框架
│   ├── registry.py                  #   脉冲算子注册表
│   ├── analyzer.py                  #   网络拓扑分析
│   └── replacer.py                  #   模块替换
│
├── Utils/
│   ├── block_selector.py            #   自动 block 大小选择
│   └── profiler.py                  #   性能分析
│
├── Benchmark/                       # 性能测试
│   ├── bench_resnet.py              #   ✅ 通用 benchmark: resnet34/50/101/152 × cifar10/100
│   ├── test_correctness.py          #   ✅ 正确性验证: Conv2d + Linear + BN
│   └── run_all.sh                   #   ✅ 批量运行脚本
│
├── Diy/                             # 早期实验代码
│   └── test/                        #   resnet34 单独 benchmark (legacy)
│
├── Instructions.md                  # 本文件
└── README.md                        # 项目说明
```

### 三层架构说明

| 层级 | 目录 | 职责 |
|------|------|------|
| **Kernel 层** | `Kernels/` | Triton GPU kernel / PyTorch 向量化实现，执行实际计算 |
| **算子封装层** | `Ops/` | 将 kernel 包装为 `nn.Module`，管理 weight/bias、处理输入 shape、提供 fallback |
| **框架层** | `Core/` | 自动识别网络中的替换目标，调用 Ops 层完成替换 |

### 导入约定

项目**不作为 pip 包安装**，所有跨目录引用使用 `sys.path` + 相对路径：

```python
# 示例：Ops/sparse_conv2d.py 引用 Kernels/conv2d.py
import sys
from pathlib import Path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Kernels.conv2d import sparse_conv2d_forward
```

```python
# 示例：Benchmark/bench_resnet.py 引用 Ops 和 Kernels
from Ops.sparse_conv2d import SparseConv2d
from Ops.sparse_linear import SparseLinear
from Kernels.conv2d import prescan_kernel, dense_conv3x3_kernel
```

---

## 关键实现细节

### Conv2d Triton Kernel (`Kernels/conv2d.py`)

- **prescan_kernel**: 按 (N, C_in, GRID_H, GRID_W) 展开，每个 program 检查一个 block 是否全零
- **sparse_conv3x3_weighted_kernel**: scatter 模式 — 每个非零 block 计算对所有 C_OUT 通道的贡献，`tl.atomic_add` 累加
- **sparse_conv1x1_weighted_kernel**: 同 scatter 模式，无 3×3 循环
- **dense_conv3x3_kernel**: 基准对照用的稠密 box filter

### Linear Triton Kernel (`Kernels/linear.py`)

- **linear_prescan_kernel**: 按行扫描 [N, C_in]，标记全零行
- **sparse_linear_kernel**: 2D grid (num_nz, cdiv(C_OUT, BLOCK_C))，每个非零行做一段输出的 dot product
- 输出初始化为 0，零行不参与计算

### BatchNorm2d (`Kernels/batchnorm2d.py`)

- 不使用 Triton kernel，BN 推理是逐元素操作，PyTorch 向量化更高效
- 预计算 `zero_bias_c = (-mean_c * inv_std_c) * gamma_c + beta_c`
- 用 `torch.where(spatial_is_zero, zero_bias, normal_bn_output)` 完成

### SparseConv2d (`Ops/sparse_conv2d.py`)

- `from_dense(conv, block_size)`: 从 nn.Conv2d 拷贝权重
- 自动处理 5D (T, N, C, H, W) 输入
- Triton 不可用时 fallback 到 `F.conv2d`

### SparseLinear (`Ops/sparse_linear.py`)

- `from_dense(linear)`: 从 nn.Linear 拷贝权重
- 支持 2D/3D/5D 输入
- Triton 不可用时 fallback 到 `F.linear`

### SparseBatchNorm2d (`Ops/sparse_batchnorm2d.py`)

- `from_dense(bn)`: 从 nn.BatchNorm2d 拷贝所有参数和 running stats
- 仅推理模式使用稀疏优化，训练模式走标准 BN
- `_zero_bias` 缓存预计算常数，避免重复计算

---

## Benchmark 设计

### bench_resnet.py

通用测试脚本，支持所有组合：

```bash
python Benchmark/bench_resnet.py --model resnet34  --dataset cifar10
python Benchmark/bench_resnet.py --model resnet50  --dataset cifar100
python Benchmark/bench_resnet.py --model resnet101 --dataset cifar10  --T 32
python Benchmark/bench_resnet.py --model resnet152 --dataset cifar100 --T 16 --batch_size 16
```

测试流程：
1. 构建 Spiking-ResNet (spikingjelly pretrained)
2. 分析网络拓扑，识别脉冲后继的 Conv2d / BN / Linear 层
3. 注册 hook 捕获脉冲特征图
4. 对每个目标层分别跑 SparseFlow / Triton dense / cuDNN，统计延迟
5. 输出逐层报告：稀疏率、vs Triton 加速比、vs cuDNN 加速比、能耗节省

### run_all.sh

批量运行所有 8 种组合 (4 模型 × 2 数据集)：

```bash
cd ~/SparseFlow
bash Benchmark/run_all.sh          # 默认 T=16, BS=32
T=32 BS=16 bash Benchmark/run_all.sh  # 自定义参数
```

---

## 技术栈

- **语言**：Python + Triton（GPU kernel）
- **框架依赖**：PyTorch 2.0+, Triton 2.0+
- **SNN 框架**：spikingjelly（延迟导入，非强依赖）
- **目标硬件**：NVIDIA GPU（Triton 支持的架构）

---

## 当前进度

- [x] Conv2d 3×3 稀疏 Triton kernel（带真实权重，scatter + atomic_add）
- [x] Conv2d 1×1 稀疏 Triton kernel（带真实权重）
- [x] Linear 稀疏 Triton kernel（按行 prescan + sparse matmul）
- [x] BatchNorm2d 稀疏优化（预计算常数 + torch.where）
- [x] SparseConv2d / SparseLinear / SparseBatchNorm2d nn.Module 封装
- [x] Core/ 框架（registry、analyzer、replacer）
- [x] `sparseflow.optimize()` 顶层 API
- [x] Utils（block_selector、profiler）
- [x] 数值正确性测试（Conv2d + Linear + BN，含 5D 输入）
- [x] 通用 Benchmark 脚本（resnet34/50/101/152 × cifar10/100）
- [x] 批量运行脚本 run_all.sh
- [x] ResNet18/34/50/101 稀疏度分析实验（CIFAR-10/100）
- [ ] GPU 上数值正确性全部验证通过
- [ ] 性能 benchmark 完整结果收集
- [ ] Depthwise Conv kernel (P2)
- [ ] MultiheadAttention kernel (P2)

---

## 协作约定

- 所有 kernel 使用 Triton 编写（BN 除外，BN 推理用 PyTorch 向量化更高效）
- 遵循两阶段（prescan + sparse compute）模式
- 每个算子需提供 nn.Module 封装（放在 `Ops/`），接口与 PyTorch 原生算子兼容
- 新算子开发流程：kernel 实现 (`Kernels/`) → nn.Module 封装 (`Ops/`) → 正确性测试 (`Benchmark/test_correctness.py`) → 性能 benchmark → 更新 Instructions.md
- 跨目录导入使用 `sys.path.insert(0, PROJECT_ROOT)` + 绝对模块名（如 `from Kernels.conv2d import ...`）
- 代码注释使用中文或英文均可，API 文档使用英文
- kernel 中的 `tl.constexpr` 参数必须在 JIT 编译时确定，运行时参数使用普通参数