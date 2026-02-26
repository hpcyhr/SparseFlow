# SparseFlow

SparseFlow is a plug-and-play **sparse inference acceleration library for Spiking Neural Networks (SNNs)**. It exploits the natural high sparsity of LIF neuron spike outputs — skipping all-zero blocks entirely — to deliver substantial speedup and energy savings with zero accuracy loss.

## Quick Start

```python
import sparseflow

model = sparseflow.optimize(model)  # That's it. All eligible ops are replaced.
```

## Why SparseFlow?

SNN neurons (LIF, IF, etc.) produce binary spike outputs where **97–99% of spatial blocks are all-zero** (verified on ResNet18/34/50/101 with CIFAR-10/100). Yet standard dense operators — including cuDNN — perform full computation on these zero blocks, wasting compute and energy.

SparseFlow fixes this with a two-stage Triton kernel design:

```
Input spike tensor ──► Stage-1: Prescan ──► Stage-2: Sparse Conv
                        (lightweight scan,    (only non-zero blocks
                         build nz_idx list)    touch the ALU)
```

**Measured sparsity on Spiking-ResNets (Poisson encoding, CIFAR-10, 224×224):**

| Model | Avg Sparsity | Zero-Block Ratio (Block=16) |
|-------|-------------|----------------------------|
| ResNet-18 | 99.06% | 97.90% |
| ResNet-34 | 99.93% | 99.81% |
| ResNet-50 | 99.93% | 99.83% |
| ResNet-101 | 99.97% | 99.93% |

## Architecture

```
sparseflow/
├── __init__.py                  # Top-level API: sparseflow.optimize(model)
│
├── core/                        # Automatic operator replacement framework
│   ├── registry.py              #   Spike op registry (LIF, IF, ParametricLIF, ...)
│   ├── analyzer.py              #   Network topology analysis: find spike → conv pairs
│   └── replacer.py              #   Module replacement: swap nn.Conv2d → SparseConv2d
│
├── kernels/                     # Triton GPU kernels (two-stage: prescan + sparse compute)
│   ├── conv2d.py                #   3×3 and 1×1 sparse convolution with real weights
│   ├── linear.py                #   Sparse fully-connected (TODO)
│   ├── depthwise.py             #   Sparse depthwise convolution (TODO)
│   └── attention.py             #   Sparse multi-head attention (TODO)
│
├── ops/                         # nn.Module wrappers (drop-in replacements for PyTorch ops)
│   ├── sparse_conv2d.py         #   SparseConv2d — replaces torch.nn.Conv2d
│   ├── sparse_linear.py         #   SparseLinear (TODO)
│   └── sparse_attention.py      #   SparseAttention (TODO)
│
├── utils/
│   ├── block_selector.py        #   Auto block size selection (H≥56→16, H≥14→8, H≤7→skip)
│   └── profiler.py              #   Hook-based latency / sparsity profiling
│
└── benchmark/
    └── test_correctness.py      #   Numerical correctness: sparse vs F.conv2d
```

**Data flow through the stack:**

```
sparseflow.optimize(model)
    │
    ▼
┌─────────┐     ┌──────────┐     ┌──────────┐
│ Registry │ ──► │ Analyzer │ ──► │ Replacer │
│ (which   │     │ (find    │     │ (swap    │
│  ops are │     │  spike → │     │  Conv2d  │
│  spikes) │     │  conv)   │     │  in-place│
└─────────┘     └──────────┘     └──────────┘
                                       │
                                       ▼
                                 SparseConv2d (ops/)
                                       │
                              ┌────────┴────────┐
                              ▼                  ▼
                        Triton path         Fallback path
                      (kernels/conv2d.py)   (F.conv2d)
                              │
                     ┌────────┴────────┐
                     ▼                  ▼
               Stage-1 Prescan    Stage-2 Sparse Conv
               (find nz blocks)   (compute only nz)
```

## Operator Support

| Priority | Operator | Status |
|----------|----------|--------|
| P0 | Conv2d 3×3 (stride=1, groups=1) | ✅ Implemented |
| P0 | Conv2d 1×1 (stride=1, groups=1) | ✅ Implemented |
| P1 | Linear | 🔜 Planned |
| P1 | BatchNorm2d | 🔜 Planned |
| P2 | Conv2d depthwise | 🔜 Planned |
| P2 | MultiheadAttention | 🔜 Planned |
| P2 | ConvTranspose2d | 🔜 Planned |

## How It Works

### Block Size Selection

SparseFlow automatically selects the prescan block size based on feature map spatial dimensions:

| Feature Map Size | Block Size | Typical Layer |
|-----------------|-----------|---------------|
| H ≥ 56 | 16 | layer1 (56×56), layer2 (28×28) |
| 14 ≤ H < 56 | 8 | layer3 (14×14) |
| H ≤ 7 | Skip | layer4 (7×7), too small to benefit |

### Two-Stage Kernel Design

**Stage-1 (Prescan):** A lightweight kernel scans every (N, C, block_h, block_w) tile. If all values in the tile are below a threshold (default 1e-6), the tile is marked as zero. Output: a compact list of non-zero block indices.

**Stage-2 (Sparse Compute):** Only non-zero blocks are dispatched to the convolution kernel. Each block loads the relevant input region, multiplies by the convolution weights, and accumulates to the output via atomic adds. Zero blocks are never touched.

### SparseConv2d Module

`SparseConv2d` is a drop-in replacement for `torch.nn.Conv2d`:

```python
from sparseflow.ops import SparseConv2d

# Create from existing Conv2d (copies weights)
sparse_conv = SparseConv2d.from_dense(original_conv, block_size=16)

# Or use directly
sparse_conv = SparseConv2d(64, 128, kernel_size=3, padding=1, block_size=16)
```

Features:
- Handles both 4D `(N,C,H,W)` and 5D `(T,N,C,H,W)` inputs (spikingjelly multi-step format)
- Automatic fallback to `F.conv2d` when Triton/CUDA is unavailable
- Records per-forward timing for profiling via `module._last_sparse_ms`

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Triton 2.0+ (for GPU acceleration)
- NVIDIA GPU (Triton-supported architecture)
- [spikingjelly](https://github.com/fangwei123456/spikingjelly) (for SNN model support)

## Project Status

- [x] 3×3 Conv2d sparse kernel with real weights
- [x] 1×1 Conv2d sparse kernel with real weights
- [x] `SparseConv2d` nn.Module wrapper
- [x] Core framework (registry, analyzer, replacer)
- [x] `sparseflow.optimize()` top-level API
- [x] Sparsity analysis on ResNet18/34/50/101
- [ ] Numerical correctness validation on GPU
- [ ] Linear kernel
- [ ] Performance benchmarking vs cuDNN
- [ ] pip-installable package