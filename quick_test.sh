#!/bin/bash

# Quick test training pipeline (fewer epochs for testing)
echo "Starting quick test training pipeline..."

# Set memory-efficient CUDA settings
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Quick SME training (5 epochs)
echo "Testing SME training..."
python train_sme.py \
    --batch-size 1 \
    --num-epochs 5 \
    --lr 1e-4 \
    --sens-chans 16 \
    --sens-pools 3 \
    --num-adjacent 5 \
    --loss-type mse \
    --net-name sme_model_test \
    --report-interval 2 \
    --data-path-train /root/Data/train/ \
    --data-path-val /root/Data/val/

if [ $? -eq 0 ]; then
    echo "✅ SME test completed!"
    
    # Quick reconstructor training (3 epochs)
    echo "Testing reconstructor training..."
    python train_reconstructor.py \
        --batch-size 1 \
        --num-epochs 3 \
        --lr 1e-4 \
        --cascade 3 \
        --chans 16 \
        --num-adjacent 5 \
        --use_prompts \
        --loss-type ssim \
        --sme-model-path ../result/sme_model_test/checkpoints/best_sme_model.pt \
        --net-name reconstructor_test \
        --report-interval 2 \
        --data-path-train /root/Data/train/ \
        --data-path-val /root/Data/val/ \
        --target-key image_label
    
    if [ $? -eq 0 ]; then
        echo "✅ Complete test pipeline successful!"
        echo "You can now run the full training pipeline."
    else
        echo "❌ Reconstructor test failed!"
    fi
else
    echo "❌ SME test failed!"
fi