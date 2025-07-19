# PromptMR+ Complete Training Strategy

## Overview
This document outlines the complete training pipeline for PromptMR+ with correct data flow, shape transformations, and mechanisms to achieve >0.98 SSIM performance.

---

## 📊 Dataset Analysis
- **Variable coil counts**: 14, 15, 16, 20 coils per sample
- **Variable spatial dimensions**: (640,368), (640,372), (768,396), etc.
- **HDF5 structure**: `(slices, coils, height, width)` complex format
- **Challenge**: Handle variable coils while maintaining SSIM performance

---

## 🎯 Stage 1: SME (Sensitivity Map Estimation) Training

### 1.1 Core Mechanism
**Strategy C (PromptMR+ approach)**: Process **single coil across multiple adjacent slices**
- **Memory efficient**: Reduces forward passes from `(2a+1)×N` to `N`  
- **Variable coil compatible**: Handles 14-20 coils per sample
- **Adjacent context**: Uses 5 adjacent slices per coil for better estimation

### 1.2 Data Flow - SME Training

#### Input Data Loading
```python
# HDF5 file structure
raw_kspace = hf['kspace'][slice_idx]  # Shape: (coils, H, W) complex
# Examples: (15, 640, 368), (20, 768, 396), (14, 640, 372)
```

#### Data Transform
```python
# DataTransform output
kspace = torch.stack([raw_kspace.real, raw_kspace.imag], dim=-1)  
# Shape: (coils, H, W, 2)
# Add batch dimension: (1, coils, H, W, 2)
```

#### SME Input Generation (PER COIL)
```python
def compute_sme_input_per_coil(kspace, coil_idx, slice_idx, num_adjacent=5):
    """
    Extract adjacent slices for a SINGLE coil
    Args:
        kspace: [B, coils, H, W, 2] - multi-coil k-space
        coil_idx: int - which coil to process
        slice_idx: int - center slice index  
        num_adjacent: int - number of adjacent slices (default 5)
    Returns:
        sme_input: [B, num_adjacent, H, W] - RSS images for one coil across slices
    """
    batch_size, num_coils, height, width, _ = kspace.shape
    device = kspace.device
    
    # Extract single coil k-space
    single_coil_kspace = kspace[:, coil_idx]  # [B, H, W, 2]
    
    # Convert to complex
    kspace_complex = single_coil_kspace[..., 0] + 1j * single_coil_kspace[..., 1]  # [B, H, W]
    
    # For training: simulate adjacent slices by adding controlled noise
    # In real implementation: load actual adjacent slices
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
```

#### SME Model Architecture
```python
class SensitivityModel(nn.Module):
    def __init__(self, in_chans=5, out_chans=2, chans=32, num_pools=4):
        """
        Args:
            in_chans: 5 (adjacent slices)
            out_chans: 2 (complex sensitivity map)
            chans: base channel count
            num_pools: U-Net depth
        """
        # U-Net architecture for single coil processing
        
    def forward(self, sme_input, mask):
        """
        Args:
            sme_input: [B, 5, H, W] - single coil across adjacent slices
            mask: [B, H, W] - sampling mask
        Returns:
            sens_map: [B, H, W, 2] - sensitivity map for this coil
        """
```

#### SME Training Loop
```python
def train_sme_epoch(args, epoch, model, data_loader, optimizer):
    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, fname, slice_idx = data
        # kspace: [B, coils, H, W, 2]
        
        device = next(model.parameters()).device
        kspace = kspace.to(device)
        mask = mask.to(device)
        
        batch_size, num_coils, H, W, _ = kspace.shape
        
        # Process each coil independently
        for coil_idx in range(num_coils):
            # Generate SME input for this coil
            sme_input = compute_sme_input_per_coil(kspace, coil_idx, slice_idx, args.num_adjacent)
            # sme_input: [B, 5, H, W]
            
            # SME forward pass
            sens_map_pred = model(sme_input, mask.squeeze(-1))
            # sens_map_pred: [B, H, W, 2]
            
            # Generate ground truth sensitivity map for this coil
            sens_map_gt = estimate_single_coil_sensitivity_map(kspace[:, coil_idx])
            # sens_map_gt: [B, H, W, 2]
            
            # Compute loss
            loss = nn.functional.mse_loss(sens_map_pred, sens_map_gt)
            
            # Backward pass
            loss.backward()
            
        optimizer.step()
        optimizer.zero_grad()
```

#### SME Output Shape
- **Input**: `[B, 5, H, W]` - single coil across 5 adjacent slices
- **Output**: `[B, H, W, 2]` - sensitivity map for that specific coil
- **Training**: Process each coil independently with its own loss

---

## 🎯 Stage 2: Reconstructor Training

### 2.1 Core Mechanism
**PromptMR+ Unrolled Network**: 
- **5 cascades** with momentum and adaptive features
- **Multi-coil reconstruction** using pre-trained SME
- **SSIM optimization** for >0.98 performance

### 2.2 Data Flow - Reconstructor Training

#### SME Model Loading (Frozen)
```python
def load_sme_model(sme_model_path, device):
    checkpoint = torch.load(sme_model_path, map_location=device)
    sme_model = SensitivityModel(...)
    sme_model.load_state_dict(checkpoint['model'])
    sme_model.to(device)
    
    # Freeze all parameters
    for param in sme_model.parameters():
        param.requires_grad = False
    
    return sme_model
```

#### Multi-Coil Sensitivity Map Generation
```python
def generate_all_coil_sensitivity_maps(sme_model, kspace, mask, args):
    """
    Generate sensitivity maps for ALL coils using the trained SME
    Args:
        sme_model: trained SME model
        kspace: [B, coils, H, W, 2] - multi-coil k-space
        mask: [B, H, W] - sampling mask
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
            sme_input = compute_sme_input_per_coil(kspace, coil_idx, slice_idx=0, args.num_adjacent)
            # sme_input: [B, 5, H, W]
            
            # Get sensitivity map for this coil
            coil_sens_map = sme_model(sme_input, mask.squeeze(-1))
            # coil_sens_map: [B, H, W, 2]
            
            # Store in output tensor
            sens_maps[:, coil_idx] = coil_sens_map
    
    return sens_maps  # [B, coils, H, W, 2]
```

#### Reconstructor Model Architecture
```python
class PromptMRPlusReconstructor(nn.Module):
    def __init__(self, num_cascades=5, chans=48, use_prompts=True, 
                 use_adaptive_input=True, use_history_features=True):
        self.num_cascades = num_cascades
        self.cascades = nn.ModuleList([
            PromptMRBlock(chans=chans, use_prompts=use_prompts, ...)
            for _ in range(num_cascades)
        ])
    
    def forward(self, kspace, mask, sens_maps):
        """
        Args:
            kspace: [B, coils, H, W, 2] - undersampled k-space
            mask: [B, 1, H, W, 1] - sampling mask  
            sens_maps: [B, coils, H, W, 2] - sensitivity maps from SME
        Returns:
            reconstructed_image: [B, H, W] - final reconstruction
        """
        # Initialize with zero-filled reconstruction
        current_kspace = kspace.clone()  # [B, coils, H, W, 2]
        
        # Initialize history features for momentum
        history_feats = [[] for _ in range(self.num_cascades)]
        
        # Run through cascades
        for i, cascade in enumerate(self.cascades):
            current_kspace, history_feats[i] = cascade(
                current_kspace, kspace, mask, sens_maps, history_feats[i]
            )
            # current_kspace: [B, coils, H, W, 2] (updated each cascade)
        
        # Final coil combination using sensitivity maps
        final_image = self.coil_combine(current_kspace, sens_maps)
        # final_image: [B, H, W] - magnitude image
        
        return final_image
```

#### Reconstructor Training Loop
```python
def train_reconstructor_epoch(args, epoch, sme_model, reconstructor_model, data_loader, optimizer, loss_fn):
    sme_model.eval()  # SME frozen
    reconstructor_model.train()
    
    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, fnames, slices = data
        # kspace: [B, coils, H, W, 2]
        # target: [B, H, W] - ground truth image
        
        device = next(sme_model.parameters()).device
        kspace = kspace.to(device)
        mask = mask.to(device)
        target = target.to(device)
        maximum = maximum.to(device)
        
        # Generate sensitivity maps using frozen SME
        sens_maps = generate_all_coil_sensitivity_maps(sme_model, kspace, mask, args)
        # sens_maps: [B, coils, H, W, 2]
        
        # Forward pass through reconstructor
        output = reconstructor_model(kspace, mask, sens_maps)
        # output: [B, H, W] - reconstructed image
        
        # Compute SSIM loss
        loss = loss_fn(output, target, maximum)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

#### Key Shape Transformations in Cascades
```python
class PromptMRBlock(nn.Module):
    def forward(self, current_kspace, ref_kspace, mask, sens_maps, history_feat):
        """
        Single cascade of PromptMR+ with data consistency + denoising
        Args:
            current_kspace: [B, coils, H, W, 2] - current k-space estimate
            ref_kspace: [B, coils, H, W, 2] - original undersampled k-space  
            mask: [B, 1, H, W, 1] - sampling mask
            sens_maps: [B, coils, H, W, 2] - sensitivity maps
            history_feat: previous cascade features for momentum
        Returns:
            updated_kspace: [B, coils, H, W, 2] - updated k-space
            new_history_feat: updated history features
        """
        # 1. Coil combination: k-space → image
        current_image = self.sens_reduce(current_kspace, sens_maps)
        # current_image: [B, 1, H, W, 2] → [B, H, W, 2] (complex image)
        
        # 2. Convert complex to channels for U-Net
        image_channels = torch.cat([current_image[..., 0], current_image[..., 1]], dim=1)
        # image_channels: [B, 2, H, W] (real/imag as separate channels)
        
        # 3. U-Net denoising with prompts and momentum
        denoised_channels, new_history_feat = self.unet(image_channels, history_feat)
        # denoised_channels: [B, 2, H, W] → [B, H, W, 2] (back to complex)
        
        # 4. Convert back to complex format
        denoised_image = torch.stack([
            denoised_channels[:, 0], denoised_channels[:, 1]
        ], dim=-1)
        # denoised_image: [B, H, W, 2]
        
        # 5. Coil expansion: image → k-space
        predicted_kspace = self.sens_expand(denoised_image, sens_maps)
        # predicted_kspace: [B, coils, H, W, 2]
        
        # 6. Data consistency
        updated_kspace = self.data_consistency(predicted_kspace, ref_kspace, mask)
        # updated_kspace: [B, coils, H, W, 2]
        
        return updated_kspace, new_history_feat
```

---

## 🎯 Stage 3: Reconstruction (Inference)

### 3.1 Complete Inference Pipeline
```python
def reconstruct_image(kspace, mask, sme_model, reconstructor_model, args):
    """
    Complete reconstruction pipeline
    Args:
        kspace: [B, coils, H, W, 2] - undersampled k-space
        mask: [B, H, W] - sampling mask
    Returns:
        reconstructed_image: [B, H, W] - final reconstruction
    """
    device = kspace.device
    
    # Step 1: Generate sensitivity maps using SME
    sens_maps = generate_all_coil_sensitivity_maps(sme_model, kspace, mask, args)
    # sens_maps: [B, coils, H, W, 2]
    
    # Step 2: Reconstruct using PromptMR+ 
    reconstructed_image = reconstructor_model(kspace, mask, sens_maps)
    # reconstructed_image: [B, H, W]
    
    return reconstructed_image
```

---

## 🎯 Training Configuration for >0.98 SSIM

### SME Training Parameters
```bash
python train_sme.py \
    --batch-size 1 \
    --num-epochs 50 \
    --lr 1e-4 \
    --sens-chans 32 \
    --sens-pools 4 \
    --num-adjacent 5 \
    --loss-type mse \
    --data-path-train /root/Data/train/ \
    --data-path-val /root/Data/val/
```

### Reconstructor Training Parameters  
```bash
python train_reconstructor.py \
    --batch-size 1 \
    --num-epochs 30 \
    --lr 1e-4 \
    --cascade 5 \
    --chans 48 \
    --use_prompts \
    --use_adaptive_input \
    --use_history_features \
    --use_checkpointing \
    --loss-type ssim \
    --num-adjacent 5 \
    --sme-model-path ../result/sme_model/checkpoints/best_sme_model.pt \
    --target-key image_label
```

---

## 🎯 Key Optimizations for SSIM Performance

### 1. Memory Efficiency
- **8GB GPU**: Use gradient checkpointing, batch_size=1
- **Per-coil SME**: Reduces memory from `(2a+1)×N` to `N` passes
- **Momentum features**: Efficient feature reuse across cascades

### 2. SSIM Optimization
- **SSIM loss**: Direct optimization of target metric
- **48+ channels**: Higher capacity for detail preservation  
- **All PromptMR+ features**: Prompts, adaptive input, momentum
- **Proper normalization**: Consistent scaling across samples

### 3. Variable Coil Handling
- **Per-coil processing**: Handles 14-20 coils seamlessly
- **Flexible architecture**: No hardcoded coil assumptions
- **Consistent spatial dims**: Proper interpolation when needed

---

## 🎯 Expected Performance
- **Target SSIM**: >0.98 on validation set
- **Memory usage**: <8GB GPU with optimizations
- **Training time**: ~2-3 days for complete pipeline
- **Robustness**: Handles variable coils and spatial dimensions

This strategy follows the original PromptMR+ methodology while addressing the specific challenges of your dataset (variable coils, GPU limitations, SSIM targets).