#!/bin/bash

# 任务1：GPU0 → v=1.0
python bench_4test.py \
--model spiking_resnet101 \
--dataset imagenet_val_flat \
--T 8 \
--batch_size 32 \
--gpu 0 \
--weight_init random \
--seed 42 \
--spike_mode normalized_bernoulli \
--replace_all_ops \
--layer_profile \
--layer_profile_warmup 10 \
--layer_profile_batches 157 \
--warmup 10 \
--verify_batches 157 \
--sparsity_batches 157 \
--data_root /home/yhr/SparseFlow/data/imagenet_val_flat \
--v_threshold 1.0 \
--out_json resnet101-t8-b32-1.0.json > run1.0.log 2>&1 &

# 任务2：GPU1 → v=0.75
python bench_4test.py \
--model spiking_resnet101 \
--dataset imagenet_val_flat \
--T 8 \
--batch_size 32 \
--gpu 1 \
--weight_init random \
--seed 42 \
--spike_mode normalized_bernoulli \
--replace_all_ops \
--layer_profile \
--layer_profile_warmup 10 \
--layer_profile_batches 157 \
--warmup 10 \
--verify_batches 157 \
--sparsity_batches 157 \
--data_root /home/yhr/SparseFlow/data/imagenet_val_flat \
--v_threshold 0.75 \
--out_json resnet101-t8-b32-0.75.json > run0.75.log 2>&1 &

# 任务3：GPU2 → v=0.5
python bench_4test.py \
--model spiking_resnet101 \
--dataset imagenet_val_flat \
--T 8 \
--batch_size 32 \
--gpu 2 \
--weight_init random \
--seed 42 \
--spike_mode normalized_bernoulli \
--replace_all_ops \
--layer_profile \
--layer_profile_warmup 10 \
--layer_profile_batches 157 \
--warmup 10 \
--verify_batches 157 \
--sparsity_batches 157 \
--data_root /home/yhr/SparseFlow/data/imagenet_val_flat \
--v_threshold 0.5 \
--out_json resnet101-t8-b32-0.5.json > run0.5.log 2>&1 &

# 任务4：GPU3 → v=0.25
python bench_4test.py \
--model spiking_resnet101 \
--dataset imagenet_val_flat \
--T 8 \
--batch_size 32 \
--gpu 3 \
--weight_init random \
--seed 42 \
--spike_mode normalized_bernoulli \
--replace_all_ops \
--layer_profile \
--layer_profile_warmup 10 \
--layer_profile_batches 157 \
--warmup 10 \
--verify_batches 157 \
--sparsity_batches 157 \
--data_root /home/yhr/SparseFlow/data/imagenet_val_flat \
--v_threshold 0.25 \
--out_json resnet101-t8-b32-0.25.json > run0.25.log 2>&1 &

echo "4个任务已全部后台启动！"
echo "查看运行状态：watch -n 1 nvidia-smi"