核心 Idea

SNN 网络的 LIF 神经元输出脉冲数据，具有天然的高稀疏性（大量元素为 0）。传统稠密算子（包括 cuDNN）对全零 block 仍然执行完整计算，浪费算力。通过对脉冲数据进行 per-tile 局部活跃子空间识别，并仅对活跃子空间执行规约，可以获得实质性的性能和能耗收益。

方法设计
实现版本说明

SparseFlow 的方法核心始终是 per-tile 动态规约子空间执行。
历史上，项目曾使用基于 per-channel CSR 的精确 prescan 实现（v15.1）；当前主线代码已演进为基于 grouped active metadata 的紧凑实现，以降低 Stage-1 元数据构建成本并改善 Stage-2 的执行规律性。本文档以下内容如未特别说明，均以当前 grouped 实现为准。

Conv2d 稀疏加速 — Per-Tile Grouped Dynamic-K（当前实现）

当前主线实现采用 per-tile grouped active metadata，而非早期版本中的精确 per-channel CSR。整体执行仍保持两阶段思想：先在每个输出 Tile 上识别局部活跃规约子空间，再仅对活跃子空间执行规约。

Stage-1: Active-Group Metadata Construction

对每个输出 Tile，系统不再构建精确的 active channel list，而是将输入通道维按固定大小 GROUP_SIZE 分组（当前实现中通常为 16），并为每个 Tile 构建：

ag_count[tile]：该 Tile 上活跃 group 的数量

ag_list[tile, i]：该 Tile 上活跃 group 的紧凑索引列表

活跃性判断基于该 Tile 对应输入感受野扩展区域内的局部非零检测。
与早期 v15.1 的 prescan_count + cumsum + prescan_write 三阶段 per-channel CSR 构建相比，当前 grouped 版本牺牲了部分精细度，但显著降低了 Stage-1 的元数据构建复杂度，并改善了 Stage-2 的执行规整性。

Stage-2: Per-Tile Grouped Sparse Reduction

对于每个 Tile，Stage-2 不再遍历全局规约维，也不再依赖 per-channel CSR 索引，而是：

读取该 Tile 的 ag_count / ag_list

仅遍历活跃 group

在每个活跃 group 内，对连续的一段输入通道执行 dense accumulation

使用 tl.dot 在组内完成 Tensor Core 友好的局部规约

因此，当前实现的稀疏执行粒度为：

Tile 级局部稀疏

Group 级动态规约

Group 内稠密计算

数据布局与缓存

Channel-Last 权重布局 [C_OUT, K, K, C_IN]：在 group 内连续通道访问时保证更好的访存局部性

权重缓存为 self._w_cl

Ops 层预分配并复用：

ag_count_buf

ag_list_buf

Runtime Fallback

当前实现包含简单的运行时收益保护机制：
当某层/某批次的平均活跃 group 比例过高时，直接 fallback 到原生 dense conv2d 路径，以避免在低稀疏收益场景下强行执行 sparse path。

当前实现与 v15.1 的关系

v15.1：per-channel CSR / 三阶段 prescan（count + cumsum + write）

当前实现：grouped active-list / compact active metadata / active-group-only iteration

也就是说，当前代码主线已经从“精确通道索引构建”演进为“紧凑 group 索引构建”。

Fused Conv + LIF Kernel（当前实现）

Fused 版本在稀疏卷积规约完成后，直接在 kernel epilogue 中完成 LIF 神经元动力学更新，从而避免 Conv 中间结果写回 VRAM 后再由独立 LIF 模块读取。

核心形式仍可写为：

I = conv_output + bias
V_temp = V_prev × decay + I
Spike = 1.0 if V_temp >= V_threshold else 0.0
V_next = soft-reset / hard-reset

其中：

当 v_reset is None 时，采用 soft reset

否则采用 hard reset

当前模块语义

当前 FusedSparseConvLIF 不再只是一个“函数式 Conv+LIF 封装”，而是一个 带内部膜电位状态的 stateful module：

内部维护 self.v

支持单步输入 [B, C, H, W]

支持多步输入 [T, B, C, H, W]

多步模式下按时间步迭代，逐步更新膜电位状态

forward() 只返回 spike tensor，状态 v_next 保存在模块内部

这使得 fused 模块在网络替换后，可以直接承担原 Conv2d + LIFNode 整体的时序语义。

当前 fused 实现与 conv2d 的关系

当前 fused kernel 与普通 sparse conv 一样，复用 active-group metadata 作为 Stage-1 基础设施，而不再复用旧版 _build_tile_csr / per-channel CSR 路径。

仅替换 Conv2d 的设计决策

基于 Benchmark 实测数据：

Conv2d：稀疏加速效果显著，是主要优化目标

BatchNorm2d：逐元素运算，原生 cuDNN/ATen 已高度优化，自定义 Triton BN 反而更慢，不替换

Linear：FC 层通常在网络出口，特征已 flatten，PyTorch BLAS 库极其高效，不替换

自动化算子替换框架

注册表识别 (Core/registry.py)：识别 LIF / IF / ParametricLIF 等脉冲输出算子，延迟导入 spikingjelly 避免强依赖

网络拓扑分析 (Core/analyzer.py)：

主路径：使用 torch.fx 符号追踪构建计算图，通过 node.users BFS 递归查找下游 Conv2d

透明层穿透：Dropout / Flatten / Identity / Pooling / BN / reshape / ReLU 等不阻断搜索

分叉支持：同一脉冲源可指向多个后继 Conv2d（ResNet 主路 + Shortcut）

Fusion 检测：Conv2d → [optional BN] → LIFNode 模式自动识别为 fused_conv*_lif 类型

Fallback：fx 追踪失败时自动回退到基于 forward hook 的线性搜索

ReplacementTarget 数据结构：op_type 可为 conv2d_3x3、conv2d_1x1、fused_conv3x3_lif、fused_conv1x1_lif

收益评估与动态分块 (Utils/block_selector.py)：根据特征图尺寸自动选择

H ≥ 32 → Block = 16（大型图）

16 ≤ H < 32 → Block = 8（中型图）

8 ≤ H < 16 → Block = 4（小型图）

H < 8 → 不替换（微型图，Gating 开销过大）

算子替换 (Core/replacer.py)：

标准路径：将 Conv2d 替换为 SparseConv2d

Fused 路径：将 Conv2d → [optional BN] → LIFNode 替换为 FusedSparseConvLIF，并使 fused 模块承担原 LIF 的时序状态更新语义

算子支持
算子	说明	状态
Conv2d 3×3	当前主线为 per-tile grouped sparse reduction	✅ Triton grouped active-list kernel + autotune
Conv2d 1×1	当前支持，但主 benchmark 仍主要聚焦 3×3	✅ Triton / dense fallback
Fused Conv3×3+LIF	grouped sparse conv + kernel epilogue LIF + stateful module	✅ 消除中间 VRAM 写入，支持多步输入
Fused Conv1×1+LIF	同上，1×1 版本	✅ 支持，但当前主要关注 3×3 主路径
Linear	全连接	✅ Triton kernel 已实现（benchmark 用，不在 optimize() 中替换）
BatchNorm2d	批归一化	✅ 向量化实现已完成（benchmark 用，不在 optimize() 中替换）
Conv2d depthwise	MobileNet 类网络	🔜 待开发
MultiheadAttention	Spiking Transformer 核心算子	🔜 待开发
项目目录结构
SparseFlow/
├── Core/                            # 自动化算子替换框架
│   ├── __init__.py
│   ├── registry.py                  #   脉冲算子注册表（延迟导入 spikingjelly）
│   ├── analyzer.py                  #   torch.fx 计算图分析 + fallback 线性搜索 + fusion 检测
│   └── replacer.py                  #   模块替换（仅 Conv2d / FusedConvLIF）
│
├── Kernels/                         # Triton GPU kernels
│   ├── conv2d.py                    #   ✅ 当前主线：per-tile grouped active metadata + active-group sparse reduction
│   ├── fused_conv_lif.py            #   ✅ Fused sparse Conv + LIF kernel（复用 grouped active metadata 基础设施）
│   ├── linear.py                    #   ✅ tile-level Dynamic-K 稀疏 Linear
│   └── batchnorm2d.py               #   ✅ 预计算零位置常数 + torch.where 向量化
│
├── Ops/                             # nn.Module 封装层
│   ├── sparse_conv2d.py             #   ✅ SparseConv2d: grouped metadata buffer 预分配, w_cl 缓存, 5D 支持, fallback
│   ├── sparse_fused_conv_lif.py     #   ✅ FusedSparseConvLIF: stateful Conv+LIF 融合模块，支持多步输入与内部膜电位状态
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
三层架构说明
层级	目录	职责
Kernel 层	Kernels/	纯 Triton GPU kernel，接受 tensor 指针和标量参数，执行实际计算
算子封装层	Ops/	将 kernel 包装为 nn.Module，管理 weight/bias、处理输入 shape、缓存、fallback
框架层	Core/	自动识别网络中的替换目标，调用 Ops 层完成替换

数据流：

sparseflow.optimize(model)
    → Core/registry.py    识别哪些 module 是脉冲源
    → Core/analyzer.py    torch.fx BFS 找到脉冲源的后继 Conv2d（含 fusion 检测），推断 input shape
    → Core/replacer.py    调用 SparseConv2d.from_dense() 或 FusedSparseConvLIF 创建模块，原地替换

model(x)  # 推理时
    → Ops/SparseConv2d.forward(x)
        → Kernels/conv2d.py::prescan_active_groups_kernel
            Stage-1: 为每个输出 Tile 构建 active-group metadata（ag_count / ag_list）
        → Runtime fallback check
            若活跃比例过高，则直接走 dense conv2d
        → Kernels/conv2d.py::sparse_conv3x3_ag_kernel
            Stage-2: 只遍历活跃 groups，在组内执行 dense accumulation（Tensor Core）

model(x)  # fused 推理时
    → Ops/FusedSparseConvLIF.forward(x)
        → 多步输入按时间步迭代
        → Kernels/fused_conv_lif.py::fused_ag_conv3x3_lif
            grouped sparse conv + LIF epilogue
        → 模块内部维护膜电位状态 self.v
关键实现细节
Triton Kernel 设计（Kernels/conv2d.py, 当前实现）

prescan_active_groups_kernel：对每个 Tile 的感受野扩展区域做 grouped prescan，输出紧凑 active-group metadata：

ag_count[tile]

ag_list[tile, i]

sparse_conv3x3_ag_kernel_*：Stage-2 grouped sparse reduction。每个 Tile：

读取自己的 ag_count / ag_list

仅遍历活跃 groups

在每个 group 内对连续输入通道执行 dense accumulation

使用 tl.dot 利用 Tensor Core

Fallback 机制：在 Python 侧基于平均 active-group ratio 做 layer-level dense fallback

按 tile shape 分裂 kernel：当前保留 _8x8 / _8x16 两个版本

Legacy kernels：旧版 prescan_kernel / dense_conv3x3_kernel 仍可保留用于兼容或实验，但不再是主线方法表述

Tile 大小策略 (_select_tile_sizes)
条件	BH × BW	BLOCK_M	典型场景
H×W ≥ 3136 (H≈56)	8 × 16	128	ResNet layer1
其他	8 × 8	64	ResNet layer2~4
SparseConv2d (Ops/sparse_conv2d.py)

from_dense(conv, block_size, threshold, return_ms): 从现有 nn.Conv2d 一键转换

_get_w_cl(): Channel-Last 权重缓存（self._w_cl）

_ensure_buffers(x): 预分配 grouped metadata buffer：

ag_count_buf

ag_list_buf

自动处理 5D (T, N, C, H, W) 输入（spikingjelly 多时间步格式）

Triton 不可用或 fallback 触发时回退到 F.conv2d

_last_sparse_ms 属性供 profiler 读取

FusedSparseConvLIF (Ops/sparse_fused_conv_lif.py)

替换 Conv2d → [optional BN] → LIFNode 的整个 pattern

作为 stateful fused module 工作，内部维护膜电位状态 self.v

支持 4D 单步输入 [B, C, H, W]

支持 5D 多步输入 [T, B, C, H, W]，按时间步迭代

forward() 只返回 spike tensor，膜电位状态在模块内部更新

LIF 参数：tau、v_threshold、v_reset、decay_input

复用 conv2d.py 的 active-group metadata 基础设施，而非旧版 _build_tile_csr

NetworkAnalyzer (Core/analyzer.py)

主路径：torch.fx.symbolic_trace(model) → 遍历 Graph nodes → BFS 从 spike_node 出发搜索下游 Conv2d/Linear

透明层穿透：定义 _TRANSPARENT_MODULES（Dropout, Identity, Flatten, Pool, BN, ReLU, Sequential）和 _TRANSPARENT_FUNCTIONS/_TRANSPARENT_METHODS 用于 fx graph 中的函数节点

Fusion 检测：找到 Conv2d 后 look-ahead 检查后续是否为 LIFNode（允许中间有 BN）

匹配条件：Conv2d 3×3 (stride=1, padding=1, groups=1) 或 Conv2d 1×1 (stride=1, padding=0, groups=1)

Fallback：fx 追踪失败时回退到基于 named_modules() 的线性搜索（搜索窗口 10 个模块）

Benchmark 体系
bench_resnet.py

主基准测试。完整流程：

构建 Spiking-ResNet (spikingjelly pretrained)

torch.fx 分析网络拓扑，识别所有替换目标

逐层正确性验证（Sparse vs cuDNN 数值误差）

注册 hook 捕获脉冲特征图

逐层性能测试：SparseFlow vs cuDNN 延迟

整网推理一致性验证（deepcopy 替换前后对比 logits 和分类结果）

cd ~/SparseFlow
python Benchmark/bench_resnet.py --model resnet50 --dataset cifar100 --T 16
bench_fused_conv_lif.py

Fused Conv+LIF 专项基准：

Fused (一次 kernel) vs Separate (sparse conv + Python LIF) vs cuDNN+LIF

数值正确性验证（Spike 和 V_next 的误差）

延迟对比和能耗估算

bench_mobilenet.py

Spiking-MobileNetV2 基准（torchvision 预训练 → ReLU6→LIF 替换）。注意 MobileNetV2 的 3×3 conv 几乎全是 depthwise (groups=C_in)，当前 SparseFlow 仅加速 groups=1 的标准 Conv2d。

bench_linear.py

Sparse Linear 逐层基准：验证 tile-level Dynamic-K 对 FC 层的加速效果。

bench_cusparse.py

cuSPARSE 独立基线：im2col + CSR 构建 + SpMM，覆盖典型 ResNet50 层配置。

run_all.sh

批量运行所有组合：

cd ~/SparseFlow
bash Benchmark/run_all.sh                  # 默认 T=16, BS=32
T=32 BS=16 bash Benchmark/run_all.sh       # 自定义参数
技术栈

语言：Python + Triton（GPU kernel）

框架依赖：PyTorch 2.0+, Triton 2.0+

SNN 框架：spikingjelly（延迟导入，非强依赖）

目标硬件：NVIDIA GPU（A100 及以上，Triton 支持的架构）

关键技术约束与经验
Triton 编译约束

range() 的参数必须是 tl.constexpr，运行时变量不可用于 range()

while 循环支持运行时边界，编译为真正的条件分支 — 这是实现动态 K 迭代的关键

tl.max / tl.reduce 要求 power-of-two 大小 — 感受野张量需要 pad 到 RF_SIZE

tl.dot(A, B) 要求 K 维度匹配且为 constexpr BLOCK_K — 通过 k_mask 处理 tail

continue 不被 Triton 2.x 支持 — 用 if condition: 包裹计算体

tl.constexpr 参数与 autotune Config 联动：BLOCK_M/N/K/H/W 通过 Config 注入

性能关键路径

CPU-GPU 同步是首要敌人：.item()、.nonzero() 在 Python 侧、需要 sync 的动态分配都会摧毁性能。每个 sync 点必须审计并消除或通过预分配绕过

大特征图（56×56）是最难的场景：全局通道稀疏很难消除通道，per-tile 稀疏至关重要，但 Python 侧开销也容易主导

BLOCK_N 大小影响 Tensor Core 利用率：大特征图需要 BLOCK_N≥64 才能充分利用 Tensor Core

Prescan↔Stage-2 的 BH/BW 必须一致：稀疏 tile 结构依赖空间 tile 粒度

开发原则

迭代式、版本标记的开发：每步先验证正确性再调优性能

Benchmark 驱动：vs cuDNN 的加速比是主要成功指标，按层配置跟踪

深度根因分析：定位到具体代码路径（Python 元数据预处理 vs GPU kernel 性能）再设计解决方案

正确性与性能分离：先做对，再做快

当前进度

 Conv2d 3×3 grouped per-tile sparse Triton kernel（compact active-group metadata, autotune）

 Fused Conv+LIF Triton kernel（grouped sparse reduction + LIF epilogue）

 Channel-Last 权重布局 + 缓存

 grouped metadata buffer 预分配（ag_count_buf / ag_list_buf）

 active-group-only iteration（仅遍历活跃 groups）

 triton.autotune 配置搜索

 SparseConv2d / FusedSparseConvLIF nn.Module 封装

 Linear / BatchNorm2d Triton kernel（benchmark 用）

 Core/ 框架（registry + torch.fx analyzer + replacer）

 sparseflow.optimize() 顶层 API

 Utils（block_selector、profiler）

 完整 Benchmark 体系（ResNet / MobileNet / Fused / Linear / cuSPARSE / 正确性）

 批量运行脚本 run_all.sh

 持续 benchmark 调优（特别是 56×56 大特征图场景）

 Autotune 配置空间进一步精炼

 Depthwise Conv kernel（MobileNet 3×3 加速）

 MultiheadAttention kernel（Spiking Transformer）

 pip 可安装包 (setup.py / pyproject.toml)

协作约定

所有 Conv kernel 使用 Triton 编写，遵循 “Stage-1 metadata construction → Stage-2 sparse compute” 模式

每个算子需提供 nn.Module 封装（放在 Ops/），接口与 PyTorch 原生算子兼容

新算子开发流程：kernel 实现 (Kernels/) → nn.Module 封装 (Ops/) → 注册到 analyzer + replacer → 正确性测试 (Benchmark/test_correctness.py) → 性能 benchmark → 更新 Instructions.md

跨目录导入使用 sys.path.insert(0, PROJECT_ROOT) + 绝对模块名（如 from Kernels.conv2d import ...）

代码注释使用中文或英文均可，API 文档使用英文

kernel 中的 tl.constexpr 参数必须在 JIT 编译时确定，运行时参数使用普通参数或 while 循环

性能修改前必须有 benchmark 数据支撑，修改后必须验证正确性未退化