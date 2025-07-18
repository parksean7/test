import torch
import argparse
import shutil
import os, sys
import time
from pathlib import Path
import numpy as np

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

from utils.learning.sme_train_part import train_sme
from utils.common.utils import seed_fix


def parse():
    parser = argparse.ArgumentParser(description='Train SME Model for PromptMR-plus',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size (keep at 1 for memory efficiency)')
    parser.add_argument('-e', '--num-epochs', type=int, default=50, help='Number of epochs for SME training')
    parser.add_argument('-l', '--lr', type=float, default=3e-4, help='Learning rate for SME')
    parser.add_argument('-r', '--report-interval', type=int, default=100, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='sme_model', help='Name of SME network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/root/Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/root/Data/val/', help='Directory of validation data')
    
    # SME specific parameters
    parser.add_argument('--sens_chans', type=int, default=3, help='Number of channels for SME U-Net')
    parser.add_argument('--sens_pools', type=int, default=3, help='Number of pooling layers for SME U-Net')
    parser.add_argument('--num_adjacent', type=int, default=5, help='Number of adjacent slices (2a+1)')
    parser.add_argument('--use_prompts', action='store_true', help='Use prompt learning in SME')
    
    # Data parameters
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')
    
    # Loss parameters
    parser.add_argument('--loss-type', type=str, default='mse', choices=['mse', 'l1'], help='Loss type for SME training')
    
    # Memory optimization parameters
    parser.add_argument('--enable-checkpointing', action='store_true', help='Enable gradient checkpointing for memory efficiency')
    parser.add_argument('--reduce-precision', action='store_true', help='Use reduced precision for memory efficiency')
    
    # Early stopping parameters
    parser.add_argument('--patience', type=int, default=10, help='Number of epochs to wait before early stopping')
    parser.add_argument('--min-delta', type=float, default=1e-6, help='Minimum change in validation loss to qualify as improvement')

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
    print("STAGE 1: TRAINING SENSITIVITY MAP ESTIMATION (SME) MODEL")
    print("="*60)
    print(f"Sensitivity Channels: {args.sens_chans}")
    print(f"Sensitivity Pools: {args.sens_pools}")
    print(f"Adjacent Slices: {args.num_adjacent}")
    print(f"Use Prompts: {args.use_prompts}")
    print(f"Loss Type: {args.loss_type}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning Rate: {args.lr}")
    print("="*60)

    train_sme(args)