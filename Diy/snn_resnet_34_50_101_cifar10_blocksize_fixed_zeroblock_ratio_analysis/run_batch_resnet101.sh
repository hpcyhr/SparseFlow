#!/bin/bash

T_LIST=(8 16 32 64)
BATCH_LIST=(32 64)

H=224
W=224
NUM_WORKERS=4
DEVICE="cuda:1"
BLOCKS="16 32"
THRESHOLD=1e-6
DATASET="cifar10"
DATA_ROOT="../../data"
PRETRAIN_FLAG="--pretrain"

OUTPUT_CSV="ALL_resnet101_${DATASET}_sparsity_summary.csv"

rm -f "${OUTPUT_CSV}"

echo "=== sweep Spiking-ResNet101 稀疏度（全 testloader） ==="
echo "T in {${T_LIST[@]}}, B in {${BATCH_LIST[@]}}"
echo "结果 CSV: ${OUTPUT_CSV}"
echo

for T in "${T_LIST[@]}"; do
  for BATCH_SIZE in "${BATCH_LIST[@]}"; do
    echo "------------------------------------------------------------"
    echo "▶️ 运行：T=${T}, B=${BATCH_SIZE}"
    echo "------------------------------------------------------------"

    python3 resnet101_snn_batch_sparsity.py \
      --T "${T}" \
      --H "${H}" \
      --W "${W}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --device "${DEVICE}" \
      --blocks ${BLOCKS} \
      --threshold "${THRESHOLD}" \
      --dataset "${DATASET}" \
      --data_root "${DATA_ROOT}" \
      --output_csv "${OUTPUT_CSV}" \
      ${PRETRAIN_FLAG}

    echo
  done
done

echo "=== sweep 完成，最终 CSV 在：${OUTPUT_CSV} ==="