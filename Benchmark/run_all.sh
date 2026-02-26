#!/bin/bash
# SparseFlow — 批量运行所有 benchmark 组合
#
# 用法:
#   cd ~/SparseFlow
#   bash Benchmark/run_all.sh
#
# 可选参数:
#   T=32 BS=16 bash Benchmark/run_all.sh

T=${T:-16}
BS=${BS:-32}
DATA_ROOT=${DATA_ROOT:-"../data"}
GPU=${GPU:-0}

MODELS=("resnet34" "resnet50" "resnet101" "resnet152")
DATASETS=("cifar10" "cifar100")

echo "=============================================="
echo " SparseFlow — Batch Benchmark"
echo " T=$T  BS=$BS  DATA=$DATA_ROOT"
echo "=============================================="

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo ""
        echo ">>> Running: $model + $dataset"
        echo "----------------------------------------------"
        python Benchmark/bench_resnet.py \
            --model "$model" \
            --dataset "$dataset" \
            --T "$T" \
            --batch_size "$BS" \
            --data_root "$DATA_ROOT" \
            --gpu "$GPU" \
            2>&1 | tee "Benchmark/results_${model}_${dataset}_T${T}.log"
        echo ""
    done
done

echo "=============================================="
echo " All benchmarks completed."
echo " Logs saved to Benchmark/results_*.log"
echo "=============================================="