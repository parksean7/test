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
    sme_model.eval()  # SME model is frozen
    reconstructor_model.train()
    
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.
    accumulation_steps = 4  # Accumulate gradients over 4 iterations

    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, _, sme_input = data
        device = next(sme_model.parameters()).device
        mask = mask.to(device, non_blocking=True)
        kspace = kspace.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        maximum = maximum.to(device, non_blocking=True)

        # Get sensitivity maps from frozen SME model
        # Use the sme_input which is already in the correct format [B, 6, H, W]
        sme_input = sme_input.to(device, non_blocking=True)
        image_input = sme_input.squeeze(1)  # [B, 1, 6, H, W] -> [B, 6, H, W]
        with torch.no_grad():
            sens_maps = sme_model(image_input, mask)

        # Forward pass through reconstructor
        output = reconstructor_model(kspace, mask, sens_maps)
        
        # Debug: print ranges for first few iterations
        if iter < 3:
            print(f"Debug iter {iter}: output_range=[{output.min():.6f}, {output.max():.6f}], target_range=[{target.min():.6f}, {target.max():.6f}]")
            print(f"Debug iter {iter}: sens_maps_range=[{sens_maps.min():.6f}, {sens_maps.max():.6f}]")
        
        # Compute reconstruction loss
        loss = loss_type(output, target, maximum)
        
        # Scale the loss to account for accumulation
        loss = loss / accumulation_steps
        loss.backward()  # Accumulate gradients
        
        # Moderate gradient clipping for PromptMR+ without attention blocks
        grad_norm = torch.nn.utils.clip_grad_norm_(reconstructor_model.parameters(), max_norm=0.02)
        
        # Update weights every accumulation_steps iterations
        if (iter + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # Set grads to None for memory
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Recon Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'GradNorm = {grad_norm:.4f} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()

    # Ensure any remaining gradients are applied at the end of the epoch
    if (iter + 1) % accumulation_steps != 0:
        optimizer.step()
    
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate_reconstructor(args, sme_model, reconstructor_model, data_loader):
    sme_model.eval()
    reconstructor_model.eval()
    
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, _, fnames, slices, sme_input = data
            device = next(sme_model.parameters()).device
            kspace = kspace.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            
            # Get sensitivity maps from SME model
            # Use the sme_input which is already in the correct format [B, 6, H, W]
            sme_input = sme_input.to(device, non_blocking=True)
            image_input = sme_input.squeeze(1)  # [B, 1, 6, H, W] -> [B, 6, H, W]
            sens_maps = sme_model(image_input, mask)
            
            # Forward pass through reconstructor
            output = reconstructor_model(kspace, mask, sens_maps)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def save_reconstructor_model(args, exp_dir, epoch, sme_model, reconstructor_model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'sme_model': sme_model.state_dict(),
            'reconstructor_model': reconstructor_model.state_dict(),
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
    """Load pre-trained SME model with smart compatibility handling"""
    print(f"Loading pre-trained SME model from: {sme_model_path}")
    
    if not sme_model_path.exists():
        raise FileNotFoundError(f"SME model not found at {sme_model_path}")
    
    checkpoint = torch.load(sme_model_path, map_location=device, weights_only=False)
    sme_args = checkpoint['args']
    
    # Create SME model with same architecture as was trained
    sme_model = SensitivityModel(
        chans=sme_args.sens_chans,
        num_pools=sme_args.sens_pools,
        num_adjacent=sme_args.num_adjacent,
        use_prompts=sme_args.use_prompts
    )
    
    # Try to load state dict, handle missing normalization layers gracefully
    try:
        sme_model.load_state_dict(checkpoint['model'], strict=True)
        print("✅ SME model loaded successfully with all parameters")
    except RuntimeError as e:
        if "Missing key(s)" in str(e) and any(norm_layer in str(e) for norm_layer in ["norm.weight", "norm.bias", "norm.running"]):
            print("📝 SME model has missing normalization layers - loading with strict=False")
            print("   (This is normal when using models trained before normalization was added)")
            sme_model.load_state_dict(checkpoint['model'], strict=False)
            print("✅ SME model loaded successfully (normalization layers initialized randomly)")
        else:
            print(f"❌ Unexpected loading error: {e}")
            raise e
    
    sme_model.to(device)
    
    # Freeze SME model parameters
    for param in sme_model.parameters():
        param.requires_grad = False
    
    print(f"📊 SME model from epoch {checkpoint['epoch']}, validation loss: {checkpoint['best_val_loss']:.4g}")
    
    return sme_model


def train_reconstructor(args):
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
        use_prompts=args.use_prompts,
        use_adaptive_input=args.use_adaptive_input,
        use_history_features=args.use_history_features,
        learnable_dc=args.learnable_dc,
        use_checkpointing=args.use_checkpointing
    )
    reconstructor_model.to(device=device)

    # Count parameters
    sme_params = sum(p.numel() for p in sme_model.parameters())
    recon_params = sum(p.numel() for p in reconstructor_model.parameters())
    trainable_params = sum(p.numel() for p in reconstructor_model.parameters() if p.requires_grad)
    
    print(f"SME Model Parameters (frozen): {sme_params:,}")
    print(f"Reconstructor Parameters: {recon_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Total Parameters: {sme_params + recon_params:,}")

    # Loss function and optimizer
    # Using a combined loss is often more robust than SSIM alone.
    loss_type = ReconstructionLoss(loss_type=args.loss_type, ssim_weight=0.85).to(device=device)
    # Use AdamW with specified learning rate and increased weight decay
    optimizer = torch.optim.AdamW(reconstructor_model.parameters(), lr=args.lr, weight_decay=3e-4) # Increased weight decay
    
    # Learning rate scheduler with warmup
    # First 2 epochs: linear warmup, then cosine annealing
    def lr_lambda(epoch):
        warmup_epochs = 2
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (args.num_epochs - warmup_epochs)))
    
    # Original scheduler:
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Alternative scheduler: ReduceLROnPlateau
    scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    use_plateau_scheduler = True  # Flag to indicate the change

    best_val_loss = float('inf')
    start_epoch = 0
    
    # Early stopping parameters
    patience = 3  # Number of epochs to wait for improvement
    patience_counter = 0
    min_delta = 0.001  # Minimum change to qualify as improvement

    # Data loaders
    train_loader = create_data_loaders(data_path=args.data_path_train, args=args, shuffle=True)
    val_loader = create_data_loaders(data_path=args.data_path_val, args=args)
    
    val_loss_log = np.empty((0, 2))
    
    print(f"Starting reconstructor training with {len(train_loader)} training batches and {len(val_loader)} validation batches")
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Reconstructor Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        # Training
        train_loss, train_time = train_reconstructor_epoch(
            args, epoch, sme_model, reconstructor_model, train_loader, optimizer, loss_type
        )
        
        # Validation
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate_reconstructor(
            args, sme_model, reconstructor_model, val_loader
        )
        
        # Update learning rate
        if 'use_plateau_scheduler' in locals() and use_plateau_scheduler:
            scheduler.step(val_loss) # Step based on validation loss
        else:
            scheduler.step() # Step based on epoch (original)
        
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "reconstructor_val_loss_log")
        np.save(file_path, val_loss_log)
        print(f"Reconstructor loss file saved! {file_path}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loss = torch.tensor(train_loss).to(device, non_blocking=True)
        val_loss = torch.tensor(val_loss).to(device, non_blocking=True)
        num_subjects = torch.tensor(num_subjects).to(device, non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < (best_val_loss - min_delta)
        
        if is_new_best:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"✨ New best validation loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter}/{patience} epochs")
            
            # Early stopping check
            if patience_counter >= patience:
                print(f"⏹️ Early stopping triggered! No improvement for {patience} epochs")
                print(f"Best validation loss: {best_val_loss:.6f}")
                break

        save_reconstructor_model(
            args, args.exp_dir, epoch + 1, sme_model, reconstructor_model, 
            optimizer, best_val_loss, is_new_best
        )
        
        print(
            f'Recon Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@Reconstructor NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
    
    print("="*60)
    print("🎉 RECONSTRUCTOR TRAINING COMPLETED!")
    print(f"Best validation loss: {best_val_loss:.4g}")
    print("Full PromptMR-plus model saved to:", args.exp_dir / 'best_reconstructor_model.pt')
    print("="*60)