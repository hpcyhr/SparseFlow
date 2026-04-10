#!/bin/bash

# 1. 硬件预设
sudo nvidia-smi -pm 1
sudo nvidia-smi -i 3 -c EXCLUSIVE_PROCESS

# 阈值列表
thresholds=("1.0" "0.75" "0.5" "0.25")

for v in "${thresholds[@]}"
do
    echo "正在测试 v_threshold = $v ..."
    
    # 注意：这里去掉了末尾的 &，让它顺序执行
    # 日志文件按阈值区分，防止覆盖
    sudo nice -n -20 ionice -c 1 -n 0 python bench_4test.py \
    --model spiking_resnet152 \
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
    --v_threshold $v \
    --out_json "resnet152-t8-b32-v$v.json" > "log_v$v.log" 2>&1
    
    echo "v_threshold = $v 测试完成。"
done

# 3. 任务全部结束后恢复默认
sudo nvidia-smi -i 3 -c DEFAULT