import torch
import argparse
import shutil
import os, sys
import time
from pathlib import Path
import numpy as np

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

from utils.learning.reconstructor_train_part import train_reconstructor
from utils.common.utils import seed_fix


def parse():
    parser = argparse.ArgumentParser(description='Train PromptMR-plus Reconstructor',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=30, help='Number of epochs for reconstructor training')
    parser.add_argument('-l', '--lr', type=float, default=1e-4, help='Learning rate for reconstructor')
    parser.add_argument('-r', '--report-interval', type=int, default=10, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='promptmr_plus_reconstructor', help='Name of reconstructor network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/root/Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/root/Data/val/', help='Directory of validation data')
    
    # FIXED: Add missing num_adjacent parameter
    parser.add_argument('--num-adjacent', type=int, default=5, help='Number of adjacent slices for SME input')
    
    # FIXED: Reconstructor specific parameters optimized for SSIM
    parser.add_argument('--cascade', type=int, default=5, help='Number of cascades')
    parser.add_argument('--chans', type=int, default=48, help='Number of channels for reconstruction U-Net (48+ for SSIM)')
    parser.add_argument('--use_prompts', action='store_true', help='Use prompt learning in reconstructor')
    parser.add_argument('--use_adaptive_input', action='store_true', help='Use adaptive input buffering')
    parser.add_argument('--use_history_features', action='store_true', help='Use history feature tracking')
    parser.add_argument('--learnable_dc', action='store_true', help='Use learnable data consistency weights')
    parser.add_argument('--use_checkpointing', action='store_true', help='Use gradient checkpointing for memory efficiency')
    
    # FIXED: SME model path (should point to per-coil trained model)
    parser.add_argument('--sme-model-path', type=Path, default='../result/sme_model_per_coil/checkpoints/best_sme_model.pt', 
                       help='Path to pre-trained SME model (per-coil)')
    
    # Data parameters
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')  # FIXED: should be image
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')
    
    # Loss parameters
    parser.add_argument('--loss-type', type=str, default='ssim', choices=['ssim', 'l1', 'combined'], 
                       help='Loss type for reconstruction training')
    
    # Memory optimization
    parser.add_argument('--accumulation-steps', type=int, default=4, help='Gradient accumulation steps')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    
    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)

    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    args.val_dir = '../result' / args.net_name / 'reconstructions_val'
    args.main_dir = '../result' / args.net_name / __file__
    args.val_loss_dir = '../result' / args.net_name

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("STAGE 2: TRAINING PROMPTMR-PLUS RECONSTRUCTOR")
    print("="*60)
    print(f"SME Model Path: {args.sme_model_path}")
    print(f"Cascades: {args.cascade}")
    print(f"Channels: {args.chans}")
    print(f"Use Prompts: {args.use_prompts}")
    print(f"Use Adaptive Input: {args.use_adaptive_input}")
    print(f"Use History Features: {args.use_history_features}")
    print(f"Learnable DC: {args.learnable_dc}")
    print(f"Use Checkpointing: {args.use_checkpointing}")
    print(f"Loss Type: {args.loss_type}")
    print(f"Target Key: {args.target_key}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning Rate: {args.lr}")
    print("="*60)

    train_reconstructor(args)