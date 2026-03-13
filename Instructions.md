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

SNN 网络的 LIF 神经元输出脉冲数据，具有天然的高稀疏性（大量元素为 0）。传统稠密算子（包括 cuDNN）对全零 block 仍然执行完整计算，浪费算力。通过对脉冲数据进行 per-tile 通道级预筛选，跳过全零通道的计算，可以获得实质性的性能和能耗收益。

---

## 方法设计

### Conv2d 稀疏加速 — Per-Tile 通道稀疏 (v15.1)

三阶段流水线，全部在 GPU 上执行，零 Host-Device 同步：

1. **Stage-1 Step 1 (`prescan_count_kernel`)**: 对每个输出 Tile，统计其 **感受野扩展区域** 内的活跃输入通道数。3×3 卷积的感受野 = (BH+2×PAD) × (BW+2×PAD)，展平后 pad 到 power-of-2 (`RF_SIZE` constexpr) 以满足 Triton `tl.max`/`tl.reduce` 要求。Grid: `(N_TILES,)`
2. **Stage-1 Step 2 (`torch.cumsum`)**: 对 `tile_counts` 做 GPU 上的前缀和 → `tile_ptr`（CSR 行指针），不读回 CPU
3. **Stage-1 Step 3 (`prescan_write_kernel`)**: 将活跃通道索引写入 `tile_cin_buf[]`。预分配 worst-case buffer (`N_TILES × C_IN`)，消除动态分配和 sync
4. **Stage-2 (`sparse_conv3x3/1x1_pertile_kernel`)**: Per-Tile 稀疏 GEMM — 每个 Tile 从 `tile_ptr`/`tile_cin_buf` 读取自己的活跃通道列表，通过间接索引加载原始 Channel-Last 权重 `[C_OUT, K, K, C_IN]`，用 `tl.dot` 利用 Tensor Core 累加。K 维使用 `while k_start < active_K` 真正动态循环，零空转迭代

关键设计决策：

- **Channel-Last 权重布局 `[C_OUT, K, K, C_IN]`**：通道维度做间接索引跳跃时，最内层 `C_IN` 连续，保证合并访存。缓存为 `self._w_cl`，权重更新时自动失效
- **`triton.autotune` 16 种配置**：搜索 BLOCK_N∈{64,128} × BLOCK_K∈{32,64} × num_warps∈{4,8} × 两种 tile shape（8×8, 8×16），keyed on `(C_IN, C_OUT, H, W, GH, GW)` 独立调优缓存
- **Prescan↔Stage-2 BH/BW 一致性**：通过按 tile shape 分裂为独立 kernel（`_8x8` / `_8x16` 后缀）保证两阶段使用相同的空间划分
- **零 Host-Device 同步**：唯一的不可避免的 sync（读 `total_active` 做 buffer 分配）通过预分配 worst-case buffer 消除

### Fused Conv + LIF Kernel

消除 Conv 输出到 VRAM 的中间写入，在 Triton kernel 的 epilogue 中直接完成 LIF 神经元动力学：

```
I = conv_output + bias
V_temp = V_prev × decay + I
Spike = 1.0 if V_temp > V_threshold else 0.0
V_next = V_temp × (1.0 - Spike)   # soft reset
```

每个输出元素的内存 IO：读 x(sparse)、w、bias、V_prev → 写 Spike、V_next。跳过 Conv 中间结果的 VRAM 写入。

### 仅替换 Conv2d 的设计决策

基于 Benchmark 实测数据：

- **Conv2d**：稀疏加速效果显著，是主要优化目标
- **BatchNorm2d**：逐元素运算，原生 cuDNN/ATen 已高度优化，自定义 Triton BN 反而更慢，**不替换**
- **Linear**：FC 层通常在网络出口，特征已 flatten，PyTorch BLAS 库极其高效，**不替换**

### 自动化算子替换框架

1. **注册表识别** (`Core/registry.py`)：识别 LIF / IF / ParametricLIF 等脉冲输出算子，延迟导入 spikingjelly 避免强依赖
2. **网络拓扑分析** (`Core/analyzer.py`)：
   - **主路径**：使用 `torch.fx` 符号追踪构建计算图，通过 `node.users` BFS 递归查找下游 Conv2d
   - **透明层穿透**：Dropout / Flatten / Identity / Pooling / BN / reshape / ReLU 等不阻断搜索
   - **分叉支持**：同一脉冲源可指向多个后继 Conv2d（ResNet 主路 + Shortcut）
   - **Fusion 检测**：Conv2d → [optional BN] → LIFNode 模式自动识别为 `fused_conv*_lif` 类型
   - **Fallback**：fx 追踪失败时自动回退到基于 forward hook 的线性搜索
   - **`ReplacementTarget` 数据结构**：`op_type` 可为 `conv2d_3x3`、`conv2d_1x1`、`fused_conv3x3_lif`、`fused_conv1x1_lif`
3. **收益评估与动态分块** (`Utils/block_selector.py`)：根据特征图尺寸自动选择
   - `H ≥ 32` → Block = 16（大型图）
   - `16 ≤ H < 32` → Block = 8（中型图）
   - `8 ≤ H < 16` → Block = 4（小型图）
   - `H < 8` → 不替换（微型图，Gating 开销过大）
4. **算子替换** (`Core/replacer.py`)：按 dot-separated module name 原地替换 Conv2d 为 SparseConv2d（或 FusedSparseConvLIF）

---

## 算子支持

| 算子 | 说明 | 状态 |
|------|------|------|
| Conv2d 3×3 | 标准卷积，最常见 | ✅ Triton per-tile CSR kernel + autotune |
| Conv2d 1×1 | ResNet 瓶颈层、降采样 | ✅ Triton per-tile CSR kernel + autotune |
| Fused Conv3×3+LIF | Conv+LIF 一体化 kernel | ✅ 消除中间 VRAM 写入 |
| Fused Conv1×1+LIF | 同上，1×1 版本 | ✅ 消除中间 VRAM 写入 |
| Linear | 全连接 | ✅ Triton kernel 已实现（benchmark 用，不在 optimize() 中替换） |
| BatchNorm2d | 批归一化 | ✅ 向量化实现已完成（benchmark 用，不在 optimize() 中替换） |
| Conv2d depthwise | MobileNet 类网络 | 🔜 待开发 |
| MultiheadAttention | Spiking Transformer 核心算子 | 🔜 待开发 |

---

## 项目目录结构

```
SparseFlow/
├── Core/                            # 自动化算子替换框架
│   ├── __init__.py
│   ├── registry.py                  #   脉冲算子注册表（延迟导入 spikingjelly）
│   ├── analyzer.py                  #   torch.fx 计算图分析 + fallback 线性搜索 + fusion 检测
│   └── replacer.py                  #   模块替换（仅 Conv2d / FusedConvLIF）
│
├── Kernels/                         # Triton GPU kernels
│   ├── conv2d.py                    #   ✅ v15.1 per-tile CSR: prescan(count+cumsum+write) + autotune Stage-2
│   ├── fused_conv_lif.py            #   ✅ Fused sparse Conv + LIF kernel（复用 conv2d.py prescan 基础设施）
│   ├── linear.py                    #   ✅ tile-level Dynamic-K 稀疏 Linear
│   └── batchnorm2d.py               #   ✅ 预计算零位置常数 + torch.where 向量化
│
├── Ops/                             # nn.Module 封装层
│   ├── sparse_conv2d.py             #   ✅ SparseConv2d: from_dense(), 5D, w_cl 缓存, buffer 预分配, fallback
│   ├── sparse_fused_conv_lif.py     #   ✅ FusedSparseConvLIF: Conv+LIF 融合模块，按时间步迭代
│   ├── sparse_linear.py             #   ✅ SparseLinear: from_dense(), 3D/5D, fallback
│   └── sparse_batchnorm2d.py        #   ✅ SparseBatchNorm2d: from_dense(), 预计算常数缓存
│
├── Utils/
│   ├── block_selector.py            #   ✅ 动态 block 大小选择 (H≥32→16, H≥16→8, H≥8→4, H<8→skip)
│   └── profiler.py                  #   ✅ Hook-based 稀疏率/延迟统计
│
├── Benchmark/
│   ├── bench_resnet.py              #   ✅ Spiking-ResNet34/50/101/152 × CIFAR-10/100 逐层 + 整网验证
│   ├── bench_mobilenet.py           #   ✅ Spiking-MobileNetV2（ReLU6→LIF）× CIFAR-10/100
│   ├── bench_fused_conv_lif.py      #   ✅ Fused vs Separate 延迟/正确性对比
│   ├── bench_linear.py              #   ✅ Sparse Linear 逐层 benchmark
│   ├── bench_cusparse.py            #   ✅ cuSPARSE baseline（im2col + CSR + SpMM）
│   ├── test_correctness.py          #   ✅ 单元正确性验证（Conv2d + Linear + BN，含 5D 输入）
│   └── run_all.sh                   #   ✅ 批量运行脚本（4 模型 × 2 数据集）
│
├── Diy/                             # 独立实验脚本
│   └── ...
│
├── README.md                        # 项目介绍
└── Instructions.md                  # 本文件
```

### 三层架构说明

| 层级 | 目录 | 职责 |
|------|------|------|
| **Kernel 层** | `Kernels/` | 纯 Triton GPU kernel，接受 tensor 指针和标量参数，执行实际计算 |
| **算子封装层** | `Ops/` | 将 kernel 包装为 `nn.Module`，管理 weight/bias、处理输入 shape、缓存、fallback |
| **框架层** | `Core/` | 自动识别网络中的替换目标，调用 Ops 层完成替换 |

数据流：

```
sparseflow.optimize(model)
    → Core/registry.py    识别哪些 module 是脉冲源
    → Core/analyzer.py    torch.fx BFS 找到脉冲源的后继 Conv2d（含 fusion 检测），推断 input shape
    → Core/replacer.py    调用 SparseConv2d.from_dense() 或 FusedSparseConvLIF 创建模块，原地替换

model(x)  # 推理时
    → Ops/SparseConv2d.forward(x)
        → Kernels/conv2d.py::prescan_count_kernel     Stage-1 Step 1: 统计活跃通道
        → torch.cumsum (GPU)                            Stage-1 Step 2: CSR 前缀和
        → Kernels/conv2d.py::prescan_write_kernel      Stage-1 Step 3: 写入活跃通道索引
        → Kernels/conv2d.py::sparse_conv3x3_pertile    Stage-2: 只计算活跃通道（Tensor Core）
```

---

## 关键实现细节

### Triton Kernel 设计 (`Kernels/conv2d.py`, v15.1)

- **prescan_count_kernel**: 每个 Tile 的感受野扩展区域（3×3 卷积时 +1 pixel each side），展平为 1D 后 pad 到 power-of-2 RF_SIZE，用 `tl.max(tl.abs(patch))` 判断通道是否活跃
- **prescan_write_kernel**: 写入 CSR 格式的 `tile_cin_buf`，使用 `tl.cumsum` 做 intra-tile compaction
- **sparse_conv3x3_pertile_kernel**: Per-Tile 稀疏 GEMM，`while k_start < active_K` 动态循环，间接索引加载 Channel-Last 权重，`tl.dot` 累加。输出 tile 通过 `m_mask` 和 `n_mask` 保证不越界写入。按 tile shape 分裂为 `_8x8` / `_8x16` 两个版本
- **sparse_conv1x1_pertile_kernel**: 类似但无 3×3 空间滤波循环
- **Legacy kernels**: `prescan_kernel` 和 `dense_conv3x3_kernel` 保留用于向后兼容

### Tile 大小策略 (`_select_tile_sizes`)

| 条件 | BH × BW | BLOCK_M | 典型场景 |
|------|---------|---------|---------|
| H×W ≥ 3136 (H≈56) | 8 × 16 | 128 | ResNet layer1 |
| 其他 | 8 × 8 | 64 | ResNet layer2~4 |

### SparseConv2d (`Ops/sparse_conv2d.py`)

- `from_dense(conv, block_size, threshold, return_ms)`: 从现有 nn.Conv2d 一键转换
- `_get_w_cl()`: Channel-Last 权重缓存（`self._w_cl`），通过 `weight.data_ptr()` 检测失效
- `_ensure_buffers(x)`: 预分配 `counts_buf` 和 `tile_cin_buf`，按需扩容但不缩容
- 自动处理 5D (T, N, C, H, W) 输入（spikingjelly 多时间步格式）
- Triton 不可用时 fallback 到 `F.conv2d`
- `_last_sparse_ms` 属性供 profiler 读取

### FusedSparseConvLIF (`Ops/sparse_fused_conv_lif.py`)

- 替换 Conv2d → [optional BN] → LIFNode 的整个 pattern
- 5D 输入 (T, B, C, H, W) 按时间步 T 迭代，因为 LIF 状态 V 依赖前一时间步
- LIF 参数：tau (膜时间常数)、v_threshold (发放阈值)、soft reset
- 复用 `conv2d.py` 的 `_build_tile_csr` 和 prescan 基础设施

### NetworkAnalyzer (`Core/analyzer.py`)

- **主路径**：`torch.fx.symbolic_trace(model)` → 遍历 Graph nodes → BFS 从 spike_node 出发搜索下游 Conv2d/Linear
- **透明层穿透**：定义 `_TRANSPARENT_MODULES`（Dropout, Identity, Flatten, Pool, BN, ReLU, Sequential）和 `_TRANSPARENT_FUNCTIONS`/`_TRANSPARENT_METHODS` 用于 fx graph 中的函数节点
- **Fusion 检测**：找到 Conv2d 后 look-ahead 检查后续是否为 LIFNode（允许中间有 BN）
- **匹配条件**：Conv2d 3×3 (stride=1, padding=1, groups=1) 或 Conv2d 1×1 (stride=1, padding=0, groups=1)
- **Fallback**：fx 追踪失败时回退到基于 `named_modules()` 的线性搜索（搜索窗口 10 个模块）

---

## Benchmark 体系

### bench_resnet.py

主基准测试。完整流程：

1. 构建 Spiking-ResNet (spikingjelly pretrained)
2. torch.fx 分析网络拓扑，识别所有替换目标
3. 逐层正确性验证（Sparse vs cuDNN 数值误差）
4. 注册 hook 捕获脉冲特征图
5. 逐层性能测试：SparseFlow vs cuDNN 延迟
6. 整网推理一致性验证（deepcopy 替换前后对比 logits 和分类结果）

```bash
cd ~/SparseFlow
python Benchmark/bench_resnet.py --model resnet50 --dataset cifar100 --T 16
```

### bench_fused_conv_lif.py

Fused Conv+LIF 专项基准：

- Fused (一次 kernel) vs Separate (sparse conv + Python LIF) vs cuDNN+LIF
- 数值正确性验证（Spike 和 V_next 的误差）
- 延迟对比和能耗估算

### bench_mobilenet.py

Spiking-MobileNetV2 基准（torchvision 预训练 → ReLU6→LIF 替换）。注意 MobileNetV2 的 3×3 conv 几乎全是 depthwise (groups=C_in)，当前 SparseFlow 仅加速 groups=1 的标准 Conv2d。

### bench_linear.py

Sparse Linear 逐层基准：验证 tile-level Dynamic-K 对 FC 层的加速效果。

### bench_cusparse.py

cuSPARSE 独立基线：im2col + CSR 构建 + SpMM，覆盖典型 ResNet50 层配置。

### run_all.sh

批量运行所有组合：

```bash
cd ~/SparseFlow
bash Benchmark/run_all.sh                  # 默认 T=16, BS=32
T=32 BS=16 bash Benchmark/run_all.sh       # 自定义参数
```

---

## 技术栈

- **语言**：Python + Triton（GPU kernel）
- **框架依赖**：PyTorch 2.0+, Triton 2.0+
- **SNN 框架**：spikingjelly（延迟导入，非强依赖）
- **目标硬件**：NVIDIA GPU（A100 及以上，Triton 支持的架构）

---

## 关键技术约束与经验

### Triton 编译约束

- `range()` 的参数必须是 `tl.constexpr`，运行时变量不可用于 `range()`
- `while` 循环支持运行时边界，编译为真正的条件分支 — 这是实现动态 K 迭代的关键
- `tl.max` / `tl.reduce` 要求 power-of-two 大小 — 感受野张量需要 pad 到 `RF_SIZE`
- `tl.dot(A, B)` 要求 K 维度匹配且为 constexpr BLOCK_K — 通过 `k_mask` 处理 tail
- `continue` 不被 Triton 2.x 支持 — 用 `if condition:` 包裹计算体
- `tl.constexpr` 参数与 autotune Config 联动：BLOCK_M/N/K/H/W 通过 Config 注入

### 性能关键路径

- **CPU-GPU 同步是首要敌人**：`.item()`、`.nonzero()` 在 Python 侧、需要 sync 的动态分配都会摧毁性能。每个 sync 点必须审计并消除或通过预分配绕过
- **大特征图（56×56）是最难的场景**：全局通道稀疏很难消除通道，per-tile 稀疏至关重要，但 Python 侧开销也容易主导
- **BLOCK_N 大小影响 Tensor Core 利用率**：大特征图需要 BLOCK_N≥64 才能充分利用 Tensor Core
- **Prescan↔Stage-2 的 BH/BW 必须一致**：CSR tile 结构依赖空间 tile 粒度

### 开发原则

- **迭代式、版本标记的开发**：每步先验证正确性再调优性能
- **Benchmark 驱动**：vs cuDNN 的加速比是主要成功指标，按层配置跟踪
- **深度根因分析**：定位到具体代码路径（Python 元数据预处理 vs GPU kernel 性能）再设计解决方案
- **正确性与性能分离**：先做对，再做快

---

## 当前进度

- [x] Conv2d 3×3/1×1 per-tile CSR 稀疏 Triton kernel (v15.1, autotune)
- [x] Fused Conv+LIF Triton kernel（消除中间 VRAM 写入）
- [x] Channel-Last 权重布局 + 缓存
- [x] 零 Host-Device 同步的 hot path（worst-case buffer 预分配）
- [x] `while` 循环动态 K 迭代（零空转）
- [x] `triton.autotune` 16 种配置
- [x] SparseConv2d / FusedSparseConvLIF nn.Module 封装
- [x] Linear / BatchNorm2d Triton kernel（benchmark 用）
- [x] Core/ 框架（registry + torch.fx analyzer + replacer）
- [x] `sparseflow.optimize()` 顶层 API
- [x] Utils（block_selector、profiler）
- [x] 完整 Benchmark 体系（ResNet / MobileNet / Fused / Linear / cuSPARSE / 正确性）
- [x] 批量运行脚本 run_all.sh
- [ ] 持续 benchmark 调优（特别是 56×56 大特征图场景）
- [ ] Autotune 配置空间进一步精炼
- [ ] Depthwise Conv kernel（MobileNet 3×3 加速）
- [ ] MultiheadAttention kernel（Spiking Transformer）
- [ ] pip 可安装包 (setup.py / pyproject.toml)

---

## 协作约定

- 所有 Conv kernel 使用 Triton 编写，遵循三阶段（prescan_count + cumsum + prescan_write → sparse compute）模式
- 每个算子需提供 nn.Module 封装（放在 `Ops/`），接口与 PyTorch 原生算子兼容
- 新算子开发流程：kernel 实现 (`Kernels/`) → nn.Module 封装 (`Ops/`) → 注册到 analyzer + replacer → 正确性测试 (`Benchmark/test_correctness.py`) → 性能 benchmark → 更新 Instructions.md
- 跨目录导入使用 `sys.path.insert(0, PROJECT_ROOT)` + 绝对模块名（如 `from Kernels.conv2d import ...`）
- 代码注释使用中文或英文均可，API 文档使用英文
- kernel 中的 `tl.constexpr` 参数必须在 JIT 编译时确定，运行时参数使用普通参数或 `while` 循环
- 性能修改前必须有 benchmark 数据支撑，修改后必须验证正确性未退化