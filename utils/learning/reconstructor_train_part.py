"""
Updated reconstructor training functions with proper integration
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


def train_reconstructor_epoch(args, epoch, sme_model, reconstructor_model, data_loader, optimizer, loss_type):
    """Train reconstructor for one epoch"""
    sme_model.eval()  # SME model is frozen
    reconstructor_model.train()
    
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.
    
    # For gradient accumulation
    accumulation_steps = getattr(args, 'accumulation_steps', 4)

    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, _, sme_input = data
        
        # Move to device
        device = next(sme_model.parameters()).device
        mask = mask.to(device, non_blocking=True)
        kspace = kspace.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        maximum = maximum.to(device, non_blocking=True)
        sme_input = sme_input.to(device, non_blocking=True)

        # Get sensitivity maps from frozen SME model
        sme_input = sme_input.squeeze(1)  # [B, 1, num_adj, H, W] -> [B, num_adj, H, W]
        with torch.no_grad():
            sens_maps = sme_model(sme_input, mask.squeeze(-1))

        # Forward pass through reconstructor
        output = reconstructor_model(kspace, mask, sens_maps)
        
        # Compute reconstruction loss
        loss = loss_type(output, target, maximum)
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        # Update weights every accumulation_steps
        if (iter + 1) % accumulation_steps == 0:
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(reconstructor_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # Log gradient norm
            if iter % args.report_interval == 0:
                print(f'Gradient norm: {grad_norm:.4f}')
        
        total_loss += loss.item() * accumulation_steps

        if iter % args.report_interval == 0:
            print(
                f'Recon Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item() * accumulation_steps:.4g} '
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
    """Validate reconstructor"""
    sme_model.eval()
    reconstructor_model.eval()
    
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    total_loss = 0.
    total_ssim = 0.
    total_samples = 0
    
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, maximum, fnames, slices, sme_input = data
            
            # Move to device
            device = next(sme_model.parameters()).device
            mask = mask.to(device, non_blocking=True)
            kspace = kspace.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            maximum = maximum.to(device, non_blocking=True)
            sme_input = sme_input.to(device, non_blocking=True)
            
            # Get sensitivity maps
            sme_input = sme_input.squeeze(1)
            sens_maps = sme_model(sme_input, mask.squeeze(-1))
            
            # Reconstruct
            output = reconstructor_model(kspace, mask, sens_maps)
            
            # Store reconstructions
            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].cpu().numpy()
            
            # Compute metrics
            # Normalize for SSIM
            output_norm = output / maximum.view(-1, 1, 1)
            target_norm = target / maximum.view(-1, 1, 1)
            
            # Add channel dimension for SSIM if needed
            if output_norm.dim() == 3:
                output_norm = output_norm.unsqueeze(1)
                target_norm = target_norm.unsqueeze(1)
            
            # Compute SSIM
            ssim_val = ssim_loss(output_norm, target_norm, maximum.view(-1, 1, 1))
            total_ssim += (1 - ssim_val.item()) * output.shape[0]  # Convert loss back to SSIM
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


def save_reconstructor_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    """Save reconstructor model"""
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


def load_sme_model(sme_model_path, device):
    """Load pre-trained SME model"""
    print(f"Loading pre-trained SME model from: {sme_model_path}")
    
    if not sme_model_path.exists():
        raise FileNotFoundError(f"SME model not found at {sme_model_path}")
    
    checkpoint = torch.load(sme_model_path, map_location=device)
    sme_args = checkpoint['args']
    
    # Create SME model
    sme_model = SensitivityModel(
        chans=sme_args.sens_chans,
        num_pools=sme_args.sens_pools,
        num_adjacent=sme_args.num_adjacent,
        use_prompts=getattr(sme_args, 'use_prompts', True)
    )
    
    # Load weights
    sme_model.load_state_dict(checkpoint['model'], strict=False)
    sme_model.to(device)
    
    # Freeze parameters
    for param in sme_model.parameters():
        param.requires_grad = False
    
    print(f"✅ SME model loaded from epoch {checkpoint['epoch']}")
    
    return sme_model


def train_reconstructor(args):
    """Main reconstructor training function"""
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        print('Current cuda device: ', torch.cuda.current_device())
    else:
        print('Using CPU device')

    # Load pre-trained SME model
    sme_model = load_sme_model(args.sme_model_path, device)
    
    # Create reconstructor model
    reconstructor_model = PromptMRPlusReconstructor(
        num_cascades=args.cascade,
        chans=args.chans,
        num_adj_slices=getattr(args, 'num_adjacent', 5),
        use_prompts=args.use_prompts if hasattr(args, 'use_prompts') else False,
        use_adaptive_input=args.use_adaptive_input if hasattr(args, 'use_adaptive_input') else False,
        use_history_features=args.use_history_features if hasattr(args, 'use_history_features') else False,
        learnable_dc=args.learnable_dc if hasattr(args, 'learnable_dc') else False,
        use_checkpointing=args.use_checkpointing if hasattr(args, 'use_checkpointing') else False
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
    
    # Optimizer
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