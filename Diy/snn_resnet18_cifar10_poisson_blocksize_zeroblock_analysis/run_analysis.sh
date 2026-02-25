#!/bin/bash

# 可以在这里改默认参数
T=16
B=64
H=224
W=224
DEVICE="cpu"          # 默认用 CPU，避免 CUDA nvrtc 那个坑
BLOCKS="16 32"        # 对应 --blocks 16 32
THRESHOLD=1e-6
PRETRAIN_FLAG="--pretrain"      # 开启预训练权重
DATASET="cifar10"
DATA_ROOT="../../data"

echo "=== 一键运行 Spiking-ResNet18 Task2 稀疏度分析 ==="
echo "T=${T}, B=${B}, Input=${H}x${W}, Blocks=${BLOCKS}, Device=${DEVICE}, Dataset=${DATASET}"
echo

python3 resnet18_snn_analysis.py \
  --T "${T}" \
  --B "${B}" \
  --H "${H}" \
  --W "${W}" \
  --device "${DEVICE}" \
  --blocks ${BLOCKS} \
  --threshold "${THRESHOLD}" \
  --dataset "${DATASET}" \
  --data_root "${DATA_ROOT}" \
  ${PRETRAIN_FLAG}