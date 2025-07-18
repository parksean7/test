#!/bin/bash

# PromptMR+ without attention blocks training script
# Original cascade structure but without problematic PromptBlocks

echo "Starting PromptMR+ training WITHOUT attention blocks..."

# Set memory-efficient CUDA settings
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Run training with original structure but no attention blocks
python train_reconstructor.py \
    --batch-size 1 \
    --num-epochs 20 \
    --lr 3e-4 \
    --cascade 5 \
    --chans 1 \
    --use_checkpointing \
    --loss-type ssim \
    --sme-model-path ../result/sme_model_memory_efficient/checkpoints/best_sme_model.pt \
    --net-name promptmr_plus_no_attention \
    --report-interval 20 \
    --data-path-train /root/Data/train/ \
    --data-path-val /root/Data/val/ \
    --target-key kspace

echo "PromptMR+ training (no attention) completed!"