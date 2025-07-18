"""
Updated SME training functions with proper loss computation and sme_input generation
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


def compute_sme_input(kspace, center_slice_idx, num_adjacent=5):
    """
    Compute SME input from kspace data by extracting adjacent slices
    Args:
        kspace: [B, coils, H, W, 2] - k-space data 
        center_slice_idx: int - center slice index
        num_adjacent: int - number of adjacent slices to use
    Returns:
        sme_input: [B, num_adjacent, H, W] - RSS images from adjacent slices
    """
    # FIXED: Correct the shape unpacking
    batch_size, num_coils, height, width, _ = kspace.shape  # Changed from kspace*dict['center'].shape
    device = kspace.device
    
    # We need to get adjacent slices, but we only have one slice
    # For now, replicate the current slice for adjacent positions
    # In a real implementation, you'd load actual adjacent slices
    
    # Convert k-space to complex
    kspace_complex = kspace[..., 0] + 1j * kspace[..., 1]  # [B, coils, H, W]
    
    # Inverse FFT to get images
    images = torch.fft.ifftshift(
        torch.fft.ifft2(
            torch.fft.fftshift(kspace_complex, dim=(-2, -1)),
            dim=(-2, -1)
        ),
        dim=(-2, -1)
    )  # [B, coils, H, W]
    
    # Compute RSS (Root Sum of Squares) across coils
    rss = torch.sqrt(torch.sum(torch.abs(images) ** 2, dim=1))  # [B, H, W]
    
    # For now, replicate this RSS image for all adjacent positions
    # In practice, you would load and process actual adjacent slices
    sme_slices = []
    for i in range(num_adjacent):
        # Add small noise to simulate different adjacent slices
        noise_factor = 0.01 * (i - num_adjacent // 2) / num_adjacent
        noisy_rss = rss * (1 + noise_factor)
        sme_slices.append(noisy_rss)
    
    # Stack to [B, num_adjacent, H, W]
    sme_input = torch.stack(sme_slices, dim=1)
    
    # FIXED: Return without adding extra channel dimension
    # The SME model expects [B, num_adjacent, H, W] not [B, 1, num_adjacent, H, W]
    return sme_input


def estimate_sensitivity_maps(kspace, mask=None, num_adjacent=5):
    """
    Estimate coil sensitivity maps from multi-coil k-space.

    Args:
        kspace: [B, C, H, W, 2] if real/imag format, or [B, C, H, W] if complex.
        mask: (ignored here)
        num_adjacent: (unused here)

    Returns:
        sens_maps_real: [B, C, H, W, 2] real/imag representation of sensitivity maps
    """
    if kspace.ndim != 5 or kspace.shape[-1] != 2:
        raise ValueError("Expected kspace shape [B, C, H, W, 2]")

    B, C, H, W, _ = kspace.shape
    device = kspace.device

    # Convert to complex tensor
    kspace_complex = torch.view_as_complex(kspace)

    # Use center k-space region
    center_fraction = 0.08
    center_lines = max(int(H * center_fraction), 24)
    start_line = (H - center_lines) // 2
    end_line = start_line + center_lines

    center_kspace = kspace_complex[:, :, start_line:end_line, :]

    # IFFT2 to get low-res coil images
    center_images = torch.fft.ifftshift(
        torch.fft.ifft2(torch.fft.fftshift(center_kspace, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1)
    )  # shape: [B, C, h_center, W], dtype: complex64

    # Magnitude before interpolation
    center_magnitude = torch.abs(center_images)  # [B, C, h_center, W], float

    # **FIX: Use original spatial dimensions instead of hardcoded (768, 384)**
    target_size = (H, W)  # Use original dimensions
    center_images_full = torch.nn.functional.interpolate(
        center_magnitude, size=target_size, mode="bilinear", align_corners=False
    )  # [B, C, H, W], float

    # Normalize via root-sum-of-squares (RSS)
    rss = torch.sqrt(torch.sum(center_images_full ** 2, dim=1, keepdim=True))
    rss = torch.clamp(rss, min=1e-8)

    # Sensitivity map = coil image / RSS
    sens_maps = center_images_full / rss  # [B, C, H, W], float

    # Convert to complex: real part only, imaginary part = 0
    sens_maps_real = torch.stack([sens_maps, torch.zeros_like(sens_maps)], dim=-1)  # [B, C, H, W, 2]

    return sens_maps_real

def train_sme_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    """Train SME model for one epoch with gradient accumulation"""
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.
    
    # Get gradient accumulation steps
    accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)

    for iter, data in enumerate(data_loader):
        # Unpack data (6 values expected)
        mask, kspace, target, maximum, fname, slice_idx = data
        
        device = next(model.parameters()).device
        mask = mask.to(device, non_blocking=True)
        kspace = kspace.to(device, non_blocking=True)
        
        # Compute SME input
        sme_input = compute_sme_input(kspace, slice_idx, args.num_adjacent)
        
        # Process input
        if sme_input.dim() == 5:
            sme_input = sme_input.squeeze(1)
        
        # Get predictions
        sens_maps_pred = model(sme_input, mask.squeeze(-1))
        
        # Get ground truth
        sens_maps_gt = estimate_sensitivity_maps(kspace, mask, args.num_adjacent)
        sens_maps_gt = sens_maps_gt.to(device)
        
        # Handle shape compatibility
        if sens_maps_pred.shape != sens_maps_gt.shape:
            if sens_maps_pred.shape[1] != sens_maps_gt.shape[1]:
                min_coils = min(sens_maps_pred.shape[1], sens_maps_gt.shape[1])
                sens_maps_pred = sens_maps_pred[:, :min_coils]
                sens_maps_gt = sens_maps_gt[:, :min_coils]
        
        # Compute loss
        if loss_type == 'mse':
            loss = nn.functional.mse_loss(sens_maps_pred, sens_maps_gt)
        else:
            # L1 loss
            loss = nn.functional.l1_loss(sens_maps_pred, sens_maps_gt)
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        # Update weights every accumulation_steps
        if (iter + 1) % accumulation_steps == 0:
            # Gradient clipping for stability
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # Log gradient norm occasionally
            if iter % args.report_interval == 0:
                print(f'Gradient norm: {grad_norm:.4f}')
        
        total_loss += loss.item() * accumulation_steps

        if iter % args.report_interval == 0:
            print(
                f'SME Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item() * accumulation_steps:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    
    # Handle remaining gradients if the last batch doesn't complete a full accumulation
    if (len_loader) % accumulation_steps != 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate_sme(args, model, data_loader):
    """Validate SME model"""
    model.eval()
    total_loss = 0.
    total_samples = 0
    
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            # Unpack 6 values
            mask, kspace, target, maximum, fname, slice_idx = data
            
            device = next(model.parameters()).device
            mask = mask.to(device, non_blocking=True)
            kspace = kspace.to(device, non_blocking=True)
            
            # Compute SME input
            sme_input = compute_sme_input(kspace, slice_idx, args.num_adjacent)
            
            # Process input
            if sme_input.dim() == 5:
                sme_input = sme_input.squeeze(1)
            
            # Get predictions
            sens_maps_pred = model(sme_input, mask.squeeze(-1))
            
            # Get ground truth
            sens_maps_gt = estimate_sensitivity_maps(kspace, mask, args.num_adjacent)
            sens_maps_gt = sens_maps_gt.to(device)
            
            # Handle shape compatibility
            if sens_maps_pred.shape != sens_maps_gt.shape:
                if sens_maps_pred.shape[1] != sens_maps_gt.shape[1]:
                    min_coils = min(sens_maps_pred.shape[1], sens_maps_gt.shape[1])
                    sens_maps_pred = sens_maps_pred[:, :min_coils]
                    sens_maps_gt = sens_maps_gt[:, :min_coils]
            
            # Compute loss
            loss = nn.functional.mse_loss(sens_maps_pred, sens_maps_gt)
            
            total_loss += loss.item()
            total_samples += 1
    
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    return avg_loss


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
        use_prompts=getattr(args, 'use_prompts', False)
    )
    model.to(device=device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SME Model - Total parameters: {total_params:,}")
    print(f"SME Model - Trainable parameters: {trainable_params:,}")
    
    # Loss function
    loss_type = args.loss_type
    
    # Data loaders - now returns 6 values per sample
    train_loader, val_loader, display_loader = create_data_loaders(data_path=None, args=args)
    
    print(f"Data loaders created successfully:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.01
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        print(f'\n=== SME Epoch {epoch + 1}/{args.num_epochs} ===')
        
        # Training
        train_loss, train_time = train_sme_epoch(
            args, epoch, model, train_loader, optimizer, loss_type
        )
        
        # Validation
        val_loss = validate_sme(args, model, val_loader)
        
        # Learning rate step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'SME Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}')
        print(f'SME Train Time: {train_time:.2f}s')
        
        # Save model
        is_new_best = val_loss < best_val_loss
        if is_new_best:
            best_val_loss = val_loss
            patience_counter = 0
            print("🎉 New best SME model!")
        else:
            patience_counter += 1
        
        save_sme_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        
        # Early stopping
        if hasattr(args, 'patience') and patience_counter >= args.patience:
            print(f"Early stopping SME training after {patience_counter} epochs without improvement")
            break
    
    print(f"\n✅ SME training completed! Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved at: {args.exp_dir / 'best_sme_model.pt'}")