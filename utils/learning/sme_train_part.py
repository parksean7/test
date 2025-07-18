"""
Updated SME training functions with proper loss computation
"""

import shutil
import numpy as np
import torch
import torch.nn as nn
import time
from pathlib import Path
import copy

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions
from utils.model.sme_model import SensitivityModel

import os


def train_sme_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    """Train SME model for one epoch"""
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, _, sme_input = data
        
        # Move to device
        device = next(model.parameters()).device
        mask = mask.to(device, non_blocking=True)
        kspace = kspace.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        maximum = maximum.to(device, non_blocking=True)
        sme_input = sme_input.to(device, non_blocking=True)
        
        # SME input is [B, 1, num_adj, H, W], we need [B, num_adj, H, W]
        sme_input = sme_input.squeeze(1)
        
        # Get sensitivity maps prediction
        sens_maps_pred = model(sme_input, mask.squeeze(-1))  # [B, coils, H, W, 2]
        
        # Compute "ground truth" sensitivity maps from fully sampled center
        # This is a simplified version - in practice you might have better ground truth
        sens_maps_gt = estimate_sensitivity_maps(kspace, mask, args.num_adjacent)
        sens_maps_gt = sens_maps_gt.to(device)
        
        # Compute loss
        if loss_type == 'mse':
            loss = nn.functional.mse_loss(sens_maps_pred, sens_maps_gt)
        else:
            # L1 loss
            loss = nn.functional.l1_loss(sens_maps_pred, sens_maps_gt)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'SME Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate_sme(args, model, data_loader):
    """Validate SME model"""
    model.eval()
    total_loss = 0.
    total_samples = 0
    
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, maximum, _, _, sme_input = data
            
            device = next(model.parameters()).device
            mask = mask.to(device, non_blocking=True)
            kspace = kspace.to(device, non_blocking=True)
            sme_input = sme_input.to(device, non_blocking=True)
            
            # Process input
            sme_input = sme_input.squeeze(1)
            
            # Get predictions
            sens_maps_pred = model(sme_input, mask.squeeze(-1))
            
            # Get ground truth
            sens_maps_gt = estimate_sensitivity_maps(kspace, mask, args.num_adjacent)
            sens_maps_gt = sens_maps_gt.to(device)
            
            # Compute loss
            loss = nn.functional.mse_loss(sens_maps_pred, sens_maps_gt)
            
            total_loss += loss.item() * mask.shape[0]
            total_samples += mask.shape[0]
    
    return total_loss / total_samples


def estimate_sensitivity_maps(kspace, mask, num_adjacent=5):
    """
    Estimate sensitivity maps from k-space data
    This is a simplified version - you may want to use more sophisticated methods
    """
    b, total_coils, h, w, _ = kspace.shape
    coils_per_slice = total_coils // num_adjacent
    
    # Use center k-space for estimation (ACS region)
    center_size = 24  # Size of auto-calibration region
    center_start = h // 2 - center_size // 2
    center_end = h // 2 + center_size // 2
    
    # Extract center k-space
    center_kspace = kspace[:, :, center_start:center_end, center_start:center_end, :]
    
    # Convert to image space
    if center_kspace.shape[-1] == 2:
        center_kspace_complex = torch.view_as_complex(center_kspace)
    else:
        center_kspace_complex = center_kspace
    
    # IFFT to get low-res images
    center_img = torch.fft.ifftshift(
        torch.fft.ifft2(
            torch.fft.fftshift(center_kspace_complex, dim=[-2, -1]),
            dim=[-2, -1],
            norm='ortho'
        ),
        dim=[-2, -1]
    )
    
    # Get center slice coils
    center_idx = num_adjacent // 2
    start_idx = center_idx * coils_per_slice
    end_idx = (center_idx + 1) * coils_per_slice
    center_coil_imgs = center_img[:, start_idx:end_idx]
    
    # Compute RSS for normalization
    rss = torch.sqrt(torch.sum(torch.abs(center_coil_imgs)**2, dim=1, keepdim=True))
    
    # Normalize to get sensitivity maps
    sens_maps = center_coil_imgs / (rss + 1e-8)
    
    # Resize to full resolution
    sens_maps_full = torch.nn.functional.interpolate(
        torch.view_as_real(sens_maps).permute(0, 1, 4, 2, 3).reshape(b, -1, center_size, center_size),
        size=(h, w),
        mode='bilinear',
        align_corners=False
    )
    
    # Reshape back
    sens_maps_full = sens_maps_full.reshape(b, coils_per_slice, 2, h, w).permute(0, 1, 3, 4, 2)
    
    return sens_maps_full


def save_sme_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    """Save SME model checkpoint"""
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'sme_model.pt'
    )
    
    if is_new_best:
        shutil.copyfile(exp_dir / 'sme_model.pt', exp_dir / 'best_sme_model.pt')


def train_sme(args):
    """Main SME training function"""
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        print('Current cuda device: ', torch.cuda.current_device())
    else:
        print('Using CPU device')

    # Create model
    model = SensitivityModel(
        chans=args.sens_chans,
        num_pools=args.sens_pools,
        num_adjacent=args.num_adjacent,
        use_prompts=args.use_prompts if hasattr(args, 'use_prompts') else True
    )
    model.to(device=device)
    
    # Loss function
    if args.loss_type == 'mse':
        loss_type = 'mse'
    else:
        loss_type = 'l1'
    
    # Data loaders
    train_loader, val_loader, display_loader = create_data_loaders(data_path=None, args=args)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.01
    )
    
    # Training variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    patience = args.patience if hasattr(args, 'patience') else 10
    early_stopped = False
    
    # Training loop
    for epoch in range(args.num_epochs):
        print(f'\n=== SME Epoch {epoch + 1}/{args.num_epochs} ===')
        
        # Train
        train_loss, train_time = train_sme_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        
        # Validate
        val_loss = validate_sme(args, model, val_loader)
        
        # Check for improvement
        is_new_best = val_loss < best_val_loss - args.min_delta
        if is_new_best:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Learning rate step
        scheduler.step()
        
        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"No improvement for {patience} consecutive epochs")
            print(f"Best validation loss: {best_val_loss:.4g}")
            early_stopped = True

        # Save model
        save_sme_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        
        # Print stats
        print(
            f'SME Epoch = [{epoch + 1:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s '
            f'EarlyStopping = [{epochs_without_improvement}/{patience}]',
        )
        
        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@SME NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        
        if early_stopped:
            break
    
    print("="*60)
    print("SME TRAINING COMPLETED!")
    print(f"Best validation loss: {best_val_loss:.4g}")
    print("SME model saved to:", args.exp_dir / 'best_sme_model.pt')
    print("="*60)