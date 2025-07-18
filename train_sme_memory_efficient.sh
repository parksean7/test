#!/bin/bash

# Memory-efficient SME training script
# Reduces memory usage through various optimizations

echo "Starting memory-efficient SME training..."

# Set memory-efficient CUDA settings
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Run training with memory optimizations
python train_sme.py \
    --batch-size 1 \
    --num-epochs 50 \
    --lr 3e-4 \
    --sens_chans 2 \
    --sens_pools 2 \
    --num_adjacent 3 \
    --loss-type mse \
    --enable-checkpointing \
    --data-path-train /root/Data/train/ \
    --data-path-val /root/Data/val/ \
    --net-name sme_model_modified \
    --report-interval 50 \
    --target-key kspace \
    --max-key max_value \
    --patience 10 \
    --min-delta 1e-6

echo "SME training completed!"