#!/bin/bash

# PromptMR+ reconstructor training script optimized for >0.98 SSIM
echo "Starting PromptMR+ reconstructor training (SSIM optimized)..."

# Set memory-efficient CUDA settings
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Run reconstructor training with all PromptMR+ features enabled
python train_reconstructor.py \
    --batch-size 1 \
    --num-epochs 30 \
    --lr 1e-4 \
    --cascade 5 \
    --chans 48 \
    --num-adjacent 5 \
    --use_prompts \
    --use_adaptive_input \
    --use_history_features \
    --use_checkpointing \
    --loss-type ssim \
    --sme-model-path ../result/sme_model_per_coil/checkpoints/best_sme_model.pt \
    --net-name promptmr_plus_ssim_optimized \
    --report-interval 10 \
    --data-path-train /root/Data/train/ \
    --data-path-val /root/Data/val/ \
    --input-key kspace \
    --target-key image_label \
    --max-key max \
    --accumulation-steps 4

echo "PromptMR+ reconstructor training completed!"