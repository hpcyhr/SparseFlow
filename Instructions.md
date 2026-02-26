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

1. **注册表识别**：识别 LIF 等脉冲输出算子
2. **自动 block 大小选择**：根据特征图尺寸自动选择
   - `H ≥ 56` → Block = 16
   - `14 ≤ H < 56` → Block = 8
   - `H ≤ 7` → 跳过（不使用稀疏加速）
3. **自动算子匹配**：在全局 module 列表中匹配 LIF 后继 Conv 层，通过 hook 捕获脉冲特征图

---

## 算子支持优先级

### P0 — 已有基础
| 算子 | 说明 | 状态 |
|------|------|------|
| Conv2d 3×3 | 标准卷积，最常见 | ✅ 已验证 |
| Conv2d 1×1 | ResNet 瓶颈层、降采样 | 待开发 |

### P1 — SNN 高频算子
| 算子 | 说明 | 状态 |
|------|------|------|
| Linear | 全连接，Transformer QKV 投影等 | 待开发 |
| BatchNorm2d | 输入为脉冲时可跳过全零样本 | 待开发 |

### P2 — 扩展网络结构覆盖
| 算子 | 说明 | 状态 |
|------|------|------|
| Conv2d depthwise | MobileNet 类网络 | 待开发 |
| MultiheadAttention | Spiking Transformer 核心算子 | 待开发 |
| ConvTranspose2d | 生成网络、分割网络 | 待开发 |

---

## 库架构

```
sparseflow/
├── core/
│   ├── registry.py          # 脉冲算子注册表（识别 LIF 等脉冲输出算子）
│   ├── analyzer.py          # 网络拓扑分析，识别脉冲后继算子
│   └── replacer.py          # 自动算子替换逻辑
├── kernels/
│   ├── conv2d.py            # 稀疏 Conv2d Triton kernel（3×3 已有）
│   ├── linear.py            # 稀疏 Linear kernel
│   ├── depthwise.py         # 稀疏 Depthwise Conv kernel
│   └── attention.py         # 稀疏 Attention kernel
├── ops/
│   ├── sparse_conv2d.py     # 封装为 nn.Module，可直接替换 torch.nn.Conv2d
│   ├── sparse_linear.py     # 封装为 nn.Module，可直接替换 torch.nn.Linear
│   └── sparse_attention.py  # 封装为 nn.Module
├── utils/
│   ├── profiler.py          # 性能分析工具（延迟、吞吐、稀疏率统计）
│   └── block_selector.py    # 自动 block 大小选择策略
└── benchmark/
    ├── resnet34.py          # ResNet34 端到端 benchmark（已有）
    └── ...
```

---

## 技术栈

- **语言**：Python + Triton（GPU kernel）
- **框架依赖**：PyTorch
- **目标硬件**：NVIDIA GPU（Triton 支持的架构）
- **核心工具**：Triton JIT 编译器

---

## 当前进度

- [x] 3×3 Conv2d 稀疏 kernel 开发完成并验证
- [x] ResNet34 端到端 benchmark 搭建完成
- [ ] 1×1 Conv2d kernel
- [ ] core/ 框架（registry、analyzer、replacer）
- [ ] Linear kernel
- [ ] 完整的 `sparseflow.optimize()` API

---

## 协作约定

- 所有 kernel 使用 Triton 编写，遵循两阶段（prescan + sparse compute）模式
- 每个算子需提供 nn.Module 封装，接口与 PyTorch 原生算子兼容
- 新算子开发流程：kernel 实现 → nn.Module 封装 → 注册到 registry → benchmark 验证
- 代码注释使用中文或英文均可，API 文档使用英文