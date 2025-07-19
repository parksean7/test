"""
Fixed PromptMR+ Reconstructor Model
- Implements momentum layers for cross-cascade feature fusion
- Proper multi-coil sensitivity handling
- Memory-efficient implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import numpy as np

# Import your existing utilities
from ..common.utils import complex_abs, fft2c_new as fft2c, ifft2c_new as ifft2c


class DataConsistencyLayer(nn.Module):
    """Data consistency layer for k-space"""
    def __init__(self, noise_lvl=None):
        super().__init__()
        self.noise_lvl = noise_lvl

    def forward(self, pred_kspace, sampled_kspace, mask, sensitivity_maps=None):
        """
        Args:
            pred_kspace: Predicted k-space
            sampled_kspace: Original sampled k-space  
            mask: Sampling mask
            sensitivity_maps: Not used in k-space DC
        """
        # Expand mask dimensions if needed
        if mask.dim() == 4 and pred_kspace.dim() == 5:
            mask = mask.unsqueeze(-1)
        
        # Apply data consistency
        return mask * sampled_kspace + (1 - mask) * pred_kspace


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of convolution layers each followed by
    instance normalization, LeakyReLU activation.
    """
    def __init__(self, in_chans, out_chans, drop_prob=0.0):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image):
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """Transpose convolution block for upsampling"""
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.layers(x)


class ChannelAttentionBlock(nn.Module):
    """Channel Attention Block (CAB) for momentum fusion"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MomentumLayer(nn.Module):
    """
    FIXED: Momentum layer for multi-stage feature fusion
    Implements: Momentum(F_0^m, ..., F_t^m) = CAB(Conv_1×1(Concat(F_0^m, ..., F_t^m)))
    """
    def __init__(self, feature_dim, n_history=11):
        super().__init__()
        self.n_history = n_history
        self.feature_dim = feature_dim
        
        # 1x1 conv to reduce concatenated features
        self.conv1x1 = nn.Conv2d(feature_dim * (n_history + 1), feature_dim, 1)
        
        # Channel Attention Block for adaptive fusion
        self.cab = ChannelAttentionBlock(feature_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.conv1x1.weight)
        nn.init.zeros_(self.conv1x1.bias)
    
    def forward(self, current_feat, history_feats):
        """
        Args:
            current_feat: [B, C, H, W] current cascade features
            history_feats: List of [B, C, H, W] from previous cascades (max n_history)
        Returns:
            output: [B, C, H, W] fused features
            new_history: Updated history list
        """
        if history_feats is None:
            history_feats = []
        
        # Limit history to n_history most recent features
        history_feats = history_feats[-self.n_history:]
        
        # If no history, just return current (first cascade)
        if len(history_feats) == 0:
            return current_feat, [current_feat]
        
        # Concatenate current with history
        all_feats = [current_feat] + history_feats
        
        # Pad with current feature if we have fewer than n_history+1 features
        while len(all_feats) < self.n_history + 1:
            all_feats.append(current_feat)
        
        # Concatenate along channel dimension
        concat_feats = torch.cat(all_feats, dim=1)  # [B, C*(n_history+1), H, W]
        
        # Fuse with 1x1 conv
        fused = self.conv1x1(concat_feats)  # [B, C, H, W]
        
        # Apply channel attention
        output = self.cab(fused)  # [B, C, H, W]
        
        # Update history (keep only last n_history features)
        new_history = ([current_feat] + history_feats)[:self.n_history]
        
        return output, new_history


class PromptUnet(nn.Module):
    """
    FIXED: U-Net model with momentum layers for cross-cascade feature fusion
    """
    def __init__(self, in_chans, out_chans, num_pool_layers=4, chans=48, drop_prob=0.0, n_history=11):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.n_history = n_history

        # Down-sampling layers
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        # Up-sampling layers  
        self.up_conv = nn.ModuleList()
        self.up_sample_layers = nn.ModuleList()
        
        # FIXED: Add momentum layers for each decoder level
        self.momentum_layers = nn.ModuleList()
        
        ch_list = []  # Track channel dimensions for momentum layers
        temp_ch = ch
        for _ in range(num_pool_layers - 1):
            self.up_conv.append(TransposeConvBlock(temp_ch * 2, temp_ch))
            self.up_sample_layers.append(ConvBlock(temp_ch * 2, temp_ch, drop_prob))
            
            # Add momentum layer for this level
            self.momentum_layers.append(MomentumLayer(temp_ch, n_history))
            ch_list.append(temp_ch)
            
            temp_ch //= 2
        
        self.up_conv.append(TransposeConvBlock(temp_ch * 2, temp_ch))
        self.up_sample_layers.append(
            nn.Sequential(
                ConvBlock(temp_ch * 2, temp_ch, drop_prob),
                nn.Conv2d(temp_ch, out_chans, kernel_size=1),
            )
        )
        
        # Add momentum layer for final level
        self.momentum_layers.append(MomentumLayer(temp_ch, n_history))

    def forward(self, image, history_feats=None):
        """
        FIXED: Forward pass with momentum feature fusion
        
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
            history_feats: List of feature histories from previous cascades
        Returns:
            output: Output tensor of shape `(N, out_chans, H, W)`.
            new_history_feats: Updated history features for next cascade
        """
        # Initialize history if None
        if history_feats is None:
            history_feats = [[] for _ in range(len(self.momentum_layers))]
        
        # Ensure we have the right number of history lists
        while len(history_feats) < len(self.momentum_layers):
            history_feats.append([])
        
        stack = []
        output = image

        # Down-sampling (encoder)
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        # Bottleneck
        output = self.conv(output)

        # Up-sampling (decoder) with momentum fusion
        new_history_feats = []
        
        for i, (up_conv, up_layer, momentum_layer) in enumerate(zip(
            self.up_conv, self.up_sample_layers, self.momentum_layers
        )):
            # Upsample and concatenate with skip connection
            output = up_conv(output)
            if i < len(stack):
                output = torch.cat([output, stack.pop()], dim=1)
            output = up_layer(output)
            
            # FIXED: Apply momentum fusion at each decoder level
            if i < len(self.momentum_layers) - 1:  # Not the final layer
                output, new_history = momentum_layer(output, history_feats[i])
                new_history_feats.append(new_history)
            else:
                # Final layer - no momentum needed
                new_history_feats.append([])
        
        return output, new_history_feats


class SensReduce(nn.Module):
    """Sensitivity-based coil combination"""
    def __init__(self, coil_num):
        super().__init__()
        self.coil_num = coil_num

    def forward(self, kspace, sens_maps):
        """
        Args:
            kspace: Multi-coil k-space [B, C, H, W, 2]
            sens_maps: Sensitivity maps [B, C, H, W, 2]
        Returns:
            Combined image [B, 1, H, W, 2]
        """
        # Convert to image space
        x = ifft2c(kspace)
        
        # Complex conjugate of sensitivity maps
        sens_maps_conj = sens_maps.clone()
        sens_maps_conj[..., 1] = -sens_maps_conj[..., 1]
        
        # Complex multiplication with conjugate
        # x * conj(s)
        real = x[..., 0] * sens_maps_conj[..., 0] - x[..., 1] * sens_maps_conj[..., 1]
        imag = x[..., 0] * sens_maps_conj[..., 1] + x[..., 1] * sens_maps_conj[..., 0]
        
        # Sum over coils
        combined = torch.stack([real, imag], dim=-1).sum(dim=1, keepdim=True)
        
        return combined


class SensExpand(nn.Module):
    """Expand image to multi-coil using sensitivity maps"""
    def __init__(self, coil_num):
        super().__init__()
        self.coil_num = coil_num

    def forward(self, image, sens_maps):
        """
        Args:
            image: Combined image [B, 1, H, W, 2]
            sens_maps: Sensitivity maps [B, C, H, W, 2]
        Returns:
            Multi-coil k-space [B, C, H, W, 2]
        """
        # Complex multiplication
        # image * sens
        real = image[..., 0] * sens_maps[..., 0] - image[..., 1] * sens_maps[..., 1]
        imag = image[..., 0] * sens_maps[..., 1] + image[..., 1] * sens_maps[..., 0]
        
        # Stack and convert to k-space
        coil_images = torch.stack([real, imag], dim=-1)
        return fft2c(coil_images)


class PromptMRBlock(nn.Module):
    """
    FIXED: Single cascade block of PromptMR+ with momentum support
    """
    def __init__(self, num_adj_slices=5, chans=48, coil_num=15, n_history=11):
        super().__init__()
        self.num_adj_slices = num_adj_slices
        self.coil_num = coil_num
        
        # Components
        self.sens_reduce = SensReduce(coil_num)
        self.sens_expand = SensExpand(coil_num)
        
        # FIXED: U-Net with momentum layers
        self.unet = PromptUnet(
            in_chans=num_adj_slices * 2,  # Complex input
            out_chans=num_adj_slices * 2,  # Complex output
            num_pool_layers=4,
            chans=chans,
            drop_prob=0.0,
            n_history=n_history  # FIXED: Add momentum support
        )
        
        self.dc_layer = DataConsistencyLayer()

    def forward(self, current_kspace, sampled_kspace, mask, sens_maps, history_feats=None):
        """
        FIXED: Forward pass with momentum feature tracking
        
        Args:
            current_kspace: Current k-space estimate [B, C*adj, H, W, 2]
            sampled_kspace: Under-sampled k-space [B, C*adj, H, W, 2]
            mask: Sampling mask [B, 1, H, W, 1]
            sens_maps: Sensitivity maps [B, C, H, W, 2]
            history_feats: History features from previous cascades
        Returns:
            updated_kspace: Updated k-space [B, C*adj, H, W, 2]
            new_history_feats: Updated history features for next cascade
        """
        b, total_coils, h, w, _ = current_kspace.shape
        coils_per_slice = self.coil_num
        
        # Process each adjacent slice
        images = []
        for adj_idx in range(self.num_adj_slices):
            # Extract slice k-space
            start_idx = adj_idx * coils_per_slice
            end_idx = (adj_idx + 1) * coils_per_slice
            slice_kspace = current_kspace[:, start_idx:end_idx]
            
            # Reduce to image space using sensitivity maps
            slice_image = self.sens_reduce(slice_kspace, sens_maps)
            images.append(slice_image)
        
        # Stack adjacent images [B, num_adj, H, W, 2]
        images_stacked = torch.cat(images, dim=1)
        
        # Convert complex to channel dimension for U-Net
        # [B, num_adj, H, W, 2] -> [B, num_adj*2, H, W]
        b, c, h, w, _ = images_stacked.shape
        images_chan = images_stacked.permute(0, 1, 4, 2, 3).reshape(b, c*2, h, w)
        
        # FIXED: Apply U-Net denoising with momentum
        denoised, new_history_feats = self.unet(images_chan, history_feats)
        
        # Convert back to complex
        # [B, num_adj*2, H, W] -> [B, num_adj, H, W, 2]
        denoised = denoised.reshape(b, c, 2, h, w).permute(0, 1, 3, 4, 2)
        
        # Expand back to k-space and apply data consistency
        updated_kspace = []
        for adj_idx in range(self.num_adj_slices):
            # Get denoised image for this slice
            slice_image = denoised[:, adj_idx:adj_idx+1]
            
            # Expand to k-space
            slice_kspace = self.sens_expand(slice_image, sens_maps)
            
            # Apply data consistency
            start_idx = adj_idx * coils_per_slice
            end_idx = (adj_idx + 1) * coils_per_slice
            slice_sampled = sampled_kspace[:, start_idx:end_idx]
            
            dc_kspace = self.dc_layer(slice_kspace, slice_sampled, mask)
            updated_kspace.append(dc_kspace)
        
        # Stack back to full k-space
        updated_kspace = torch.cat(updated_kspace, dim=1)
        
        return updated_kspace, new_history_feats


class PromptMRPlusReconstructor(nn.Module):
    """
    FIXED: Full PromptMR+ reconstruction model with momentum support
    """
    def __init__(self, num_cascades=5, chans=48, num_adj_slices=5, 
                 use_prompts=False, use_adaptive_input=False, 
                 use_history_features=True, learnable_dc=False,
                 use_checkpointing=False, coil_num=15, n_history=11):
        super().__init__()
        
        self.num_cascades = num_cascades
        self.num_adj_slices = num_adj_slices
        self.center_slice = num_adj_slices // 2
        self.use_checkpointing = use_checkpointing
        self.use_history_features = use_history_features
        self.coil_num = coil_num
        
        # FIXED: Create cascade blocks with momentum support
        self.cascades = nn.ModuleList([
            PromptMRBlock(
                num_adj_slices=num_adj_slices,
                chans=chans,
                coil_num=coil_num,
                n_history=n_history if use_history_features else 0
            )
            for _ in range(num_cascades)
        ])
        
        # Initialize zero-filled reconstruction
        self.register_buffer('zero_filled_init', torch.tensor(0.0))

    def forward(self, kspace, mask, sens_maps):
        """
        FIXED: Forward pass with momentum feature tracking
        
        Args:
            kspace: Under-sampled k-space [B, C*adj, H, W, 2]
            mask: Sampling mask [B, 1, H, W, 1]
            sens_maps: Sensitivity maps [B, C, H, W, 2]
        Returns:
            reconstructed_image: [B, H, W] - central slice reconstruction
        """
        # Initialize with zero-filled reconstruction
        current_kspace = kspace.clone()
        
        # FIXED: Initialize history features for momentum
        history_feats = None
        if self.use_history_features:
            history_feats = [[] for _ in range(len(self.cascades))]
        
        # Run through cascades
        for i, cascade in enumerate(self.cascades):
            if self.use_checkpointing and self.training:
                # Use gradient checkpointing for memory efficiency
                current_kspace, new_history = torch.utils.checkpoint.checkpoint(
                    cascade, current_kspace, kspace, mask, sens_maps,
                    history_feats[i] if history_feats else None
                )
            else:
                current_kspace, new_history = cascade(
                    current_kspace, kspace, mask, sens_maps,
                    history_feats[i] if history_feats else None
                )
            
            # Update history features
            if self.use_history_features and history_feats:
                history_feats[i] = new_history
        
        # Extract central slice and convert to image
        b, total_coils, h, w, _ = current_kspace.shape
        coils_per_slice = self.coil_num
        
        # Get central slice k-space
        center_start = self.center_slice * coils_per_slice
        center_end = (self.center_slice + 1) * coils_per_slice
        center_kspace = current_kspace[:, center_start:center_end]
        
        # Convert to image space
        center_images = ifft2c(center_kspace)
        
        # Coil combination using sensitivity maps
        sens_maps_conj = sens_maps.clone()
        sens_maps_conj[..., 1] = -sens_maps_conj[..., 1]
        
        # Complex multiplication and sum
        real = center_images[..., 0] * sens_maps_conj[..., 0] - center_images[..., 1] * sens_maps_conj[..., 1]
        imag = center_images[..., 0] * sens_maps_conj[..., 1] + center_images[..., 1] * sens_maps_conj[..., 0]
        
        combined_complex = torch.stack([real, imag], dim=-1)
        combined_image = combined_complex.sum(dim=1)  # Sum over coils
        
        # Take magnitude
        magnitude = torch.sqrt(combined_image[..., 0] ** 2 + combined_image[..., 1] ** 2 + 1e-8)
        
        return magnitude


class ReconstructionLoss(nn.Module):
    """Loss function for reconstruction training - Modified to use existing SSIM implementation"""
    def __init__(self, loss_type='ssim'):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'ssim':
            # Use the existing SSIM implementation instead of pytorch_msssim
            from utils.common.utils import ssim_loss as custom_ssim_loss
            self.loss_fn = custom_ssim_loss
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'combined':
            # Use existing SSIM implementation for combined loss
            from utils.common.utils import ssim_loss as custom_ssim_loss
            self.ssim_fn = custom_ssim_loss
            self.l1 = nn.L1Loss()
    
    def forward(self, output, target, max_value):
        """
        Args:
            output: Predicted image [B, H, W]
            target: Ground truth image [B, H, W]
            max_value: Maximum values for normalization [B]
        """
        # Normalize
        output_norm = output / max_value.view(-1, 1, 1)
        target_norm = target / max_value.view(-1, 1, 1)
        
        if self.loss_type == 'ssim':
            # Use the existing ssim_loss function which already returns (1 - SSIM)
            return self.loss_fn(target_norm, output_norm)
        elif self.loss_type == 'l1':
            return self.loss_fn(output_norm, target_norm)
        elif self.loss_type == 'combined':
            ssim_loss = self.ssim_fn(target_norm, output_norm)
            l1_loss = self.l1(output_norm, target_norm)
            return 0.8 * ssim_loss + 0.2 * l1_loss
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


# For backward compatibility
def build_promptmr_model(args):
    """Build PromptMR+ model from args"""
    model = PromptMRPlusReconstructor(
        num_cascades=args.cascade,
        chans=args.chans,
        num_adj_slices=getattr(args, 'num_adjacent', 5),
        use_prompts=args.use_prompts if hasattr(args, 'use_prompts') else False,
        use_adaptive_input=args.use_adaptive_input if hasattr(args, 'use_adaptive_input') else False,
        use_history_features=args.use_history_features if hasattr(args, 'use_history_features') else True,
        learnable_dc=args.learnable_dc if hasattr(args, 'learnable_dc') else False,
        use_checkpointing=args.use_checkpointing if hasattr(args, 'use_checkpointing') else False,
        coil_num=getattr(args, 'num_coils', 15),
        n_history=getattr(args, 'n_history', 11)
    )
    return model