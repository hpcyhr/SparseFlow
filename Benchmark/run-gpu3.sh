sudo nohup nice -n -20 ionice -c 1 -n 0 python bench_4test.py \
--model spiking_resnet34 \
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
--out_json resnet34-t8-b32-v1.0.json > experiment.log 2>&1 &
# 找到刚才启动的 python 进程 PID 并将其设置为不可杀
PID=$! && sudo echo -1000 > /proc/$PID/oom_score_adj && echo "Process $PID is now unkillable."
sudo nvidia-smi -i 0 -c DEFAULT


sudo nohup nice -n -20 ionice -c 1 -n 0 python bench_4test.py \
--model spiking_resnet34 \
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
--v_threshold 0.75 \
--out_json resnet34-t8-b32-v0.75.json > experiment.log 2>&1 &
# 找到刚才启动的 python 进程 PID 并将其设置为不可杀
PID=$! && sudo echo -1000 > /proc/$PID/oom_score_adj && echo "Process $PID is now unkillable."
sudo nvidia-smi -i 0 -c DEFAULT

sudo nohup nice -n -20 ionice -c 1 -n 0 python bench_4test.py \
--model spiking_resnet34 \
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
--v_threshold 0.5 \
--out_json resnet34-t8-b32-v0.5.json > experiment.log 2>&1 &
# 找到刚才启动的 python 进程 PID 并将其设置为不可杀
PID=$! && sudo echo -1000 > /proc/$PID/oom_score_adj && echo "Process $PID is now unkillable."
sudo nvidia-smi -i 0 -c DEFAULT

sudo nohup nice -n -20 ionice -c 1 -n 0 python bench_4test.py \
--model spiking_resnet34 \
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
--v_threshold 0.25 \
--out_json resnet34-t8-b32-v0.25.json > experiment.log 2>&1 &
# 找到刚才启动的 python 进程 PID 并将其设置为不可杀
PID=$! && sudo echo -1000 > /proc/$PID/oom_score_adj && echo "Process $PID is now unkillable."
sudo nvidia-smi -i 0 -c DEFAULT