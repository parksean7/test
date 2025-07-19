"""
FIXED: Reconstructor training functions with proper multi-coil SME integration
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
from utils.common.utils import save_reconstructions, ssim_loss
from utils.model.sme_model import SensitivityModel
from utils.model.promptmr_reconstructor import PromptMRPlusReconstructor, ReconstructionLoss

import os


def compute_sme_input_per_coil(kspace, coil_idx, slice_idx, num_adjacent=5):
    """
    FIXED: Extract adjacent slices for a SINGLE coil (same as SME training)
    """
    batch_size, num_coils, height, width, _ = kspace.shape
    device = kspace.device
    
    # Extract single coil k-space
    single_coil_kspace = kspace[:, coil_idx]  # [B, H, W, 2]
    
    # Convert to complex
    kspace_complex = single_coil_kspace[..., 0] + 1j * single_coil_kspace[..., 1]  # [B, H, W]
    
    # For training: simulate adjacent slices by adding controlled noise
    adjacent_slices = []
    for adj_idx in range(num_adjacent):
        # Simulate adjacent slice with slight variation
        noise_factor = 0.02 * (adj_idx - num_adjacent // 2) / num_adjacent
        adj_kspace = kspace_complex * (1 + noise_factor)
        
        # IFFT to image domain
        adj_image = torch.fft.ifftshift(
            torch.fft.ifft2(torch.fft.fftshift(adj_kspace, dim=(-2, -1))),
            dim=(-2, -1)
        )
        
        # Take magnitude for SME input
        adj_magnitude = torch.abs(adj_image)  # [B, H, W]
        adjacent_slices.append(adj_magnitude)
    
    # Stack adjacent slices
    sme_input = torch.stack(adjacent_slices, dim=1)  # [B, num_adjacent, H, W]
    
    return sme_input


def generate_all_coil_sensitivity_maps(sme_model, kspace, mask, args):
    """
    FIXED: Generate sensitivity maps for ALL coils using the trained SME
    Args:
        sme_model: trained SME model (frozen)
        kspace: [B, coils, H, W, 2] - multi-coil k-space
        mask: [B, H, W] - sampling mask
        args: training arguments
    Returns:
        sens_maps: [B, coils, H, W, 2] - sensitivity maps for all coils
    """
    batch_size, num_coils, H, W, _ = kspace.shape
    device = kspace.device
    
    # Initialize output tensor
    sens_maps = torch.zeros_like(kspace)  # [B, coils, H, W, 2]
    
    with torch.no_grad():
        for coil_idx in range(num_coils):
            # Generate SME input for this coil
            sme_input = compute_sme_input_per_coil(kspace, coil_idx, slice_idx=0, num_adjacent=args.num_adjacent)
            # sme_input: [B, num_adjacent, H, W]
            
            # Ensure correct shape
            if sme_input.dim() == 5:
                sme_input = sme_input.squeeze(1)
            
            # Get sensitivity map for this coil
            coil_sens_map = sme_model(sme_input, mask.squeeze(-1))
            # coil_sens_map: [B, H, W, 2]
            
            # Store in output tensor
            sens_maps[:, coil_idx] = coil_sens_map
    
    return sens_maps  # [B, coils, H, W, 2]


def train_reconstructor_epoch(args, epoch, sme_model, reconstructor_model, data_loader, optimizer, loss_type):
    """
    FIXED: Train reconstructor for one epoch with proper multi-coil SME integration
    """
    sme_model.eval()  # SME model is frozen
    reconstructor_model.train()
    
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    # For gradient accumulation
    accumulation_steps = getattr(args, 'accumulation_steps', 4)

    for iter, data in enumerate(data_loader):
        # Handle data unpacking
        if len(data) == 7:
            mask, kspace, target, maximum, fnames, slices, sme_input = data
        else:
            mask, kspace, target, maximum, fnames, slices = data

        # Move to device
        device = next(sme_model.parameters()).device
        mask = mask.to(device, non_blocking=True)
        kspace = kspace.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        maximum = maximum.to(device, non_blocking=True)

        # FIXED: Generate sensitivity maps for ALL coils using frozen SME
        sens_maps = generate_all_coil_sensitivity_maps(sme_model, kspace, mask, args)
        # sens_maps: [B, coils, H, W, 2] - matches kspace coil dimension
        
        # Verify shapes match
        B_k, C_k, H_k, W_k, _ = kspace.shape
        B_s, C_s, H_s, W_s, _ = sens_maps.shape
        
        assert C_k == C_s, f"Coil mismatch: kspace {C_k} vs sens_maps {C_s}"
        assert H_k == H_s and W_k == W_s, f"Spatial mismatch: kspace {(H_k,W_k)} vs sens_maps {(H_s,W_s)}"

        # Forward pass through reconstructor
        output = reconstructor_model(kspace, mask, sens_maps)
        # output: [B, H, W] - reconstructed image
        
        # Compute reconstruction loss
        loss = loss_type(output, target, maximum)
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        # Update weights every accumulation_steps
        if (iter + 1) % accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(reconstructor_model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            if iter % args.report_interval == 0:
                print(f'Gradient norm: {grad_norm:.4f}')
        
        total_loss += loss.item() * accumulation_steps

        if iter % args.report_interval == 0:
            print(
                f'Recon Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item() * accumulation_steps:.4g} '
                f'Coils = {C_k} Spatial = {(H_k,W_k)} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()

    # Handle remaining gradients
    if (iter + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate_reconstructor(args, sme_model, reconstructor_model, data_loader):
    """
    FIXED: Validate reconstructor with proper multi-coil SME integration
    """
    sme_model.eval()
    reconstructor_model.eval()
    
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    total_ssim = 0.
    total_samples = 0
    
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            if len(data) == 7:
                mask, kspace, target, maximum, fnames, slices, sme_input = data
            else:
                mask, kspace, target, maximum, fnames, slices = data
            
            # Move to device
            device = next(sme_model.parameters()).device
            mask = mask.to(device, non_blocking=True)
            kspace = kspace.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            maximum = maximum.to(device, non_blocking=True)

            # Generate sensitivity maps for all coils
            sens_maps = generate_all_coil_sensitivity_maps(sme_model, kspace, mask, args)
            
            # Reconstruct
            output = reconstructor_model(kspace, mask, sens_maps)
            
            # Store reconstructions
            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].cpu().numpy()
            
            # Compute SSIM
            output_norm = output / maximum.view(-1, 1, 1)
            target_norm = target / maximum.view(-1, 1, 1)
            
            # Add channel dimension for SSIM if needed
            if output_norm.dim() == 3:
                output_norm = output_norm.unsqueeze(1)
                target_norm = target_norm.unsqueeze(1)
            
            # Compute SSIM (using 1 - ssim_loss to get actual SSIM)
            ssim_val = ssim_loss(output_norm, target_norm, maximum.view(-1, 1, 1))
            total_ssim += (1 - ssim_val.item()) * output.shape[0]
            total_samples += output.shape[0]
    
    # Stack reconstructions
    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    
    avg_ssim = total_ssim / total_samples
    
    return reconstructions, targets, avg_ssim


def load_sme_model(sme_model_path, device):
    """
    FIXED: Load pre-trained SME model with per-coil architecture
    """
    print(f"Loading pre-trained SME model from: {sme_model_path}")
    
    if not sme_model_path.exists():
        raise FileNotFoundError(f"SME model not found at {sme_model_path}")
    
    checkpoint = torch.load(sme_model_path, map_location=device)
    sme_args = checkpoint['args']
    
    # Create SME model with per-coil architecture
    sme_model = SensitivityModel(
        in_chans=sme_args.num_adjacent,  # 5 adjacent slices
        out_chans=2,  # Complex sensitivity map
        chans=sme_args.sens_chans,
        num_pools=sme_args.sens_pools,
        use_prompts=getattr(sme_args, 'use_prompts', True)
    )
    
    # Load weights
    sme_model.load_state_dict(checkpoint['model'], strict=False)
    sme_model.to(device)
    
    # Freeze parameters
    for param in sme_model.parameters():
        param.requires_grad = False
    
    print(f"✅ SME model loaded from epoch {checkpoint['epoch']}")
    print(f"✅ SME processes per-coil with {sme_args.num_adjacent} adjacent slices")
    
    return sme_model


def train_reconstructor(args):
    """
    FIXED: Main reconstructor training function with proper SME integration
    """
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        print('Current cuda device: ', torch.cuda.current_device())
    else:
        print('Using CPU device')

    # Load pre-trained SME model (frozen)
    sme_model = load_sme_model(args.sme_model_path, device)
    
    # FIXED: Extract num_adjacent from SME model if not provided
    if not hasattr(args, 'num_adjacent'):
        sme_checkpoint = torch.load(args.sme_model_path, map_location='cpu')
        args.num_adjacent = sme_checkpoint['args'].num_adjacent
        print(f"Using num_adjacent={args.num_adjacent} from SME model")
    
    # Create reconstructor model with OPTIMIZED settings for SSIM
    reconstructor_model = PromptMRPlusReconstructor(
        num_cascades=args.cascade,
        chans=args.chans,  # Use 48+ for best SSIM
        num_adj_slices=args.num_adjacent,
        use_prompts=getattr(args, 'use_prompts', True),  # Enable for SSIM
        use_adaptive_input=getattr(args, 'use_adaptive_input', True),  # Enable for SSIM
        use_history_features=getattr(args, 'use_history_features', True),  # Enable for SSIM
        learnable_dc=getattr(args, 'learnable_dc', False),
        use_checkpointing=getattr(args, 'use_checkpointing', False)
    )
    reconstructor_model.to(device=device)

    # Print model info
    sme_params = sum(p.numel() for p in sme_model.parameters())
    recon_params = sum(p.numel() for p in reconstructor_model.parameters())
    trainable_params = sum(p.numel() for p in reconstructor_model.parameters() if p.requires_grad)
    
    print(f"SME Model Parameters (frozen): {sme_params:,}")
    print(f"Reconstructor Parameters: {recon_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Total Parameters: {sme_params + recon_params:,}")

    # Loss function
    loss_type = ReconstructionLoss(loss_type=args.loss_type)
    
    # Data loaders
    train_loader, val_loader, display_loader = create_data_loaders(data_path=None, args=args)
    
    # FIXED: Optimizer with proper settings for SSIM optimization
    optimizer = torch.optim.AdamW(
        reconstructor_model.parameters(), 
        lr=args.lr, 
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.num_epochs, 
        eta_min=args.lr * 0.01
    )
    
    # Training variables
    best_val_loss = float('inf')
    best_val_ssim = 0.0
    
    # Training loop
    for epoch in range(args.num_epochs):
        print(f'\n=== Reconstructor Epoch {epoch + 1}/{args.num_epochs} ===')
        
        # Train
        train_loss, train_time = train_reconstructor_epoch(
            args, epoch, sme_model, reconstructor_model, 
            train_loader, optimizer, loss_type
        )
        
        # Validate
        reconstructions, targets, val_ssim = validate_reconstructor(
            args, sme_model, reconstructor_model, val_loader
        )
        
        # Convert SSIM to loss for consistency
        val_loss = 1 - val_ssim
        
        # Check for improvement
        is_new_best = val_ssim > best_val_ssim
        if is_new_best:
            best_val_ssim = val_ssim
            best_val_loss = val_loss
            
            # Save reconstructions
            save_reconstructions(reconstructions, args.val_dir, targets)
        
        # Learning rate step
        scheduler.step()
        
        # Save model
        save_reconstructor_model(
            args, args.exp_dir, epoch + 1, 
            reconstructor_model, optimizer, best_val_loss, is_new_best
        )
        
        # Print stats
        print(
            f'Recon Epoch = [{epoch + 1:4d}/{args.num_epochs:4d}] '
            f'TrainLoss = {train_loss:.4g} ValLoss = {val_loss:.4g} '
            f'ValSSIM = {val_ssim:.4f} TrainTime = {train_time:.4f}s'
        )
        
        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    
    print("="*60)
    print("RECONSTRUCTOR TRAINING COMPLETED!")
    print(f"Best validation SSIM: {best_val_ssim:.4f}")
    print("Model saved to:", args.exp_dir / 'best_reconstructor_model.pt')
    print("="*60)
    
    return best_val_ssim


def save_reconstructor_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    """Save reconstructor model checkpoint"""
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir,
            'model_type': 'promptmr_plus_full'
        },
        f=exp_dir / 'reconstructor_model.pt'
    )
    
    if is_new_best:
        shutil.copyfile(exp_dir / 'reconstructor_model.pt', exp_dir / 'best_reconstructor_model.pt')