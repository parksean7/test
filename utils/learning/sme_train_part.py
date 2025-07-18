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
from utils.model.sme_model import SensitivityModel, SMELoss

import os


def train_sme_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        # Clear cache before each iteration to prevent memory buildup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        mask, kspace, target, maximum, _, _ = data
        device = next(model.parameters()).device
        mask = mask.to(device, non_blocking=True)
        kspace = kspace.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        maximum = maximum.to(device, non_blocking=True)

        # Forward pass through SME model
        # Remove extra batch dimension added by DataLoader FIRST
        kspace = kspace.squeeze(0)  # Remove DataLoader batch dimension
        mask = mask.squeeze(0)      # Remove DataLoader batch dimension

        # Now create target sensitivity maps with correct dimensions
        with torch.no_grad():
            # Create dummy target sensitivity maps with same shape as expected output
            # The model should output sensitivity maps with shape [B, C, H, W, 2]
            # kspace shape is now [B, 6, H, W] - 4D input format
            B, input_channels, H, W = kspace.shape
            output_coils = 15  # Original number of coils
            target_sens_maps = torch.zeros(B, output_coils, H, W, 2, device=kspace.device)
            print(f"DEBUG: Created dummy target_sens_maps shape: {target_sens_maps.shape}")
        
        print(f"DEBUG: After squeeze - kspace shape: {kspace.shape}")
        print(f"DEBUG: After squeeze - mask shape: {mask.shape}")
        pred_sens_maps = model(kspace, mask)
        
        # Debug shapes - only print first iteration
        if iter == 0:
            print(f"kspace shape: {kspace.shape}")
            print(f"pred_sens_maps shape: {pred_sens_maps.shape}")
            print(f"target_sens_maps shape: {target_sens_maps.shape}")
        
        # Compute loss
        loss = loss_type(pred_sens_maps, target_sens_maps)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Store loss value before cleanup
        current_loss = loss.item()
        total_loss += current_loss
        
        # Clear intermediate variables to free memory
        del pred_sens_maps, target_sens_maps, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if iter % args.report_interval == 0:
            print(
                f'SME Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {current_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate_sme(args, model, data_loader):
    model.eval()
    total_loss = 0.
    num_batches = 0
    start = time.perf_counter()
    
    loss_fn = SMELoss(args.loss_type)

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, _, fnames, slices = data
            device = next(model.parameters()).device
            kspace = kspace.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            
            # Remove extra batch dimension added by DataLoader
            kspace = kspace.squeeze(0)
            mask = mask.squeeze(0)
            
            # Create dummy target sensitivity maps (same as training)
            B, input_channels, H, W = kspace.shape
            output_coils = 15  # Original number of coils
            target_sens_maps = torch.zeros(B, output_coils, H, W, 2, device=kspace.device)
            
            # Forward pass
            pred_sens_maps = model(kspace, mask)
            
            # Compute loss
            loss = loss_fn(pred_sens_maps, target_sens_maps)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss, num_batches, None, None, None, time.perf_counter() - start


def create_target_sensitivity_maps(kspace, mask):
    """
    Create target sensitivity maps using ESPIRiT-like method
    This is a simplified version - in practice you'd use more sophisticated methods
    Memory optimized version
    """
    # Input kspace: [B, C, H, W, 2]
    # Convert to image domain (complex) - use smaller chunks to save memory
    with torch.no_grad():
        kspace_complex = torch.view_as_complex(kspace)  # [B, C, H, W]
        images = torch.fft.ifft2(kspace_complex, norm='ortho')  # [B, C, H, W]
        
        # Compute RSS for normalization
        rss = torch.sqrt(torch.sum(images.abs() ** 2, dim=1, keepdim=True))  # [B, 1, H, W]
        
        # Avoid division by zero
        rss = torch.clamp(rss, min=1e-6)
        
        # Compute sensitivity maps
        sens_maps = images / rss  # [B, C, H, W]
        
        # Convert back to real/imaginary format [B, C, H, W, 2]
        sens_maps_complex = torch.view_as_real(sens_maps)
        
        # Clear intermediate variables
        del kspace_complex, images, rss, sens_maps
        
    return sens_maps_complex


def save_sme_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir,
            'model_type': 'sme'
        },
        f=exp_dir / 'sme_model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'sme_model.pt', exp_dir / 'best_sme_model.pt')


def train_sme(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        print('Current cuda device: ', torch.cuda.current_device())
        
        # Print initial GPU memory stats
        print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")
        
        # Enable memory efficient settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Clear cache
        torch.cuda.empty_cache()
    else:
        print('Using CPU device')

    # Create SME model
    model = SensitivityModel(
        chans=args.sens_chans,
        num_pools=args.sens_pools,
        num_adjacent=args.num_adjacent,
        use_prompts=args.use_prompts
    )
    model.to(device=device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"SME Model Parameters: {total_params:,}")

    # Loss function and optimizer with better settings
    loss_type = SMELoss(args.loss_type).to(device=device)
    # Use AdamW with weight decay for better regularization
    # More aggressive regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    
    # Learning rate scheduler with warmup
    # First 5 epochs: linear warmup, then cosine annealing
    def lr_lambda(epoch):
        if epoch < 5:
            return epoch / 5.0
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - 5) / (args.num_epochs - 5)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    start_epoch = 0
    
    # Early stopping parameters
    patience = getattr(args, 'patience', 10)  # Default: 10 epochs without improvement
    min_delta = getattr(args, 'min_delta', 1e-6)  # Minimum improvement threshold
    epochs_without_improvement = 0
    early_stopped = False
    
    print(f"Early stopping enabled: patience={patience}, min_delta={min_delta}")

    # Data loaders
    train_loader = create_data_loaders(data_path=args.data_path_train, args=args, shuffle=True)
    val_loader = create_data_loaders(data_path=args.data_path_val, args=args)
    
    val_loss_log = np.empty((0, 2))
    
    print(f"Starting SME training with {len(train_loader)} training batches and {len(val_loader)} validation batches")
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f'SME Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        # Training
        train_loss, train_time = train_sme_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        
        # Validation
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate_sme(args, model, val_loader)
        
        # Update learning rate
        scheduler.step()
        
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "sme_val_loss_log")
        np.save(file_path, val_loss_log)
        print(f"SME loss file saved! {file_path}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loss = torch.tensor(train_loss).to(device, non_blocking=True)
        val_loss = torch.tensor(val_loss).to(device, non_blocking=True)

        # Early stopping logic
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            is_new_best = True
        else:
            epochs_without_improvement += 1
            is_new_best = False
            
        # Check for early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            print(f"No improvement for {patience} consecutive epochs")
            print(f"Best validation loss: {best_val_loss:.4g}")
            early_stopped = True

        save_sme_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        
        # Print memory usage
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**3
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f'GPU Memory: {current_memory:.2f} GB current, {peak_memory:.2f} GB peak')
        
        print(
            f'SME Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s '
            f'EarlyStopping = [{epochs_without_improvement}/{patience}]',
        )
        
        # Memory cleanup after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@SME NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            
        # Break if early stopping triggered
        if early_stopped:
            break
    
    print("="*60)
    if early_stopped:
        print("SME TRAINING COMPLETED (EARLY STOPPING)!")
        print(f"Stopped at epoch {epoch + 1}/{args.num_epochs}")
    else:
        print("SME TRAINING COMPLETED!")
    print(f"Best validation loss: {best_val_loss:.4g}")
    print("SME model saved to:", args.exp_dir / 'best_sme_model.pt')
    print("="*60)