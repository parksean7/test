#!/bin/bash

# SME training script with per-coil processing (PromptMR+ Strategy C)
echo "Starting SME training with per-coil processing..."

# Set memory-efficient CUDA settings
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Run SME training with per-coil strategy
python train_sme.py \
    --batch-size 1 \
    --num-epochs 50 \
    --lr 1e-4 \
    --sens-chans 32 \
    --sens-pools 4 \
    --num-adjacent 5 \
    --use-prompts \
    --loss-type mse \
    --net-name sme_model_per_coil \
    --report-interval 10 \
    --data-path-train /root/Data/train/ \
    --data-path-val /root/Data/val/ \
    --input-key kspace \
    --target-key image_label \
    --max-key max \
    --gradient-accumulation-steps 2

echo "SME training (per-coil) completed!"