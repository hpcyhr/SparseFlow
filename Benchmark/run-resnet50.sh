#!/bin/bash

# 任务1：GPU0 → v=1.0
python bench_4test.py \
--model spiking_resnet50 \
--dataset imagenet_val_flat \
--T 8 \
--batch_size 32 \
--gpu 0 \
--weight_init random \
--seed 42 \
--spike_mode normalized_bernoulli \
--replace_all_ops \
--data_root /home/yhr/SparseFlow/data/imagenet_val_flat \
--inference_mode \
--v_threshold 1.0 \
--out_json resnet50-t8-b32-1.0.json &

# 任务2：GPU1 → v=0.75
python bench_4test.py \
--model spiking_resnet50 \
--dataset imagenet_val_flat \
--T 8 \
--batch_size 32 \
--gpu 1 \
--weight_init random \
--seed 42 \
--spike_mode normalized_bernoulli \
--replace_all_ops \
--data_root /home/yhr/SparseFlow/data/imagenet_val_flat \
--inference_mode \
--v_threshold 0.75 \
--out_json resnet50-t8-b32-0.75.json &

# 任务3：GPU2 → v=0.5
python bench_4test.py \
--model spiking_resnet50 \
--dataset imagenet_val_flat \
--T 8 \
--batch_size 32 \
--gpu 2 \
--weight_init random \
--seed 42 \
--spike_mode normalized_bernoulli \
--replace_all_ops \
--data_root /home/yhr/SparseFlow/data/imagenet_val_flat \
--inference_mode \
--v_threshold 0.5 \
--out_json resnet50-t8-b32-0.5.json &

# 任务4：GPU3 → v=0.25
python bench_4test.py \
--model spiking_resnet50 \
--dataset imagenet_val_flat \
--T 8 \
--batch_size 32 \
--gpu 3 \
--weight_init random \
--seed 42 \
--spike_mode normalized_bernoulli \
--replace_all_ops \
--data_root /home/yhr/SparseFlow/data/imagenet_val_flat \
--inference_mode \
--v_threshold 0.25 \
--out_json resnet50-t8-b32-0.25.json &

echo "4个任务已全部后台启动！"
echo "查看运行状态：watch -n 1 nvidia-smi"


