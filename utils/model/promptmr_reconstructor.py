"""
Updated PromptMR+ Reconstructor Model
Maintains compatibility with your existing training code
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


class PromptUnet(nn.Module):
    """
    U-Net model without attention blocks for memory efficiency
    """
    def __init__(self, in_chans, out_chans, num_pool_layers=4, chans=48, drop_prob=0.0):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers

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
        for _ in range(num_pool_layers - 1):
            self.up_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_sample_layers.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2
        self.up_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_sample_layers.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, out_chans, kernel_size=1),
            )
        )

    def forward(self, image):
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # Down-sampling
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # Up-sampling
        for up_conv_layer, up_sample_layer in zip(self.up_conv, self.up_sample_layers):
            output = up_conv_layer(output)
            output = torch.cat([output, stack.pop()], dim=1)
            output = up_sample_layer(output)

        return output


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
    """Single cascade block of PromptMR+"""
    def __init__(self, num_adj_slices=5, chans=48, coil_num=15):
        super().__init__()
        self.num_adj_slices = num_adj_slices
        self.coil_num = coil_num
        
        # Components
        self.sens_reduce = SensReduce(coil_num)
        self.sens_expand = SensExpand(coil_num)
        
        # U-Net for denoising (operating on adjacent slices)
        # Input: num_adj complex images (2 channels each)
        # Output: num_adj complex images
        self.unet = PromptUnet(
            in_chans=num_adj_slices * 2,  # Complex input
            out_chans=num_adj_slices * 2,  # Complex output
            num_pool_layers=4,
            chans=chans,
            drop_prob=0.0
        )
        
        self.dc_layer = DataConsistencyLayer()

    def forward(self, current_kspace, sampled_kspace, mask, sens_maps):
        """
        Args:
            current_kspace: Current k-space estimate [B, C*adj, H, W, 2]
            sampled_kspace: Under-sampled k-space [B, C*adj, H, W, 2]
            mask: Sampling mask [B, 1, H, W, 1]
            sens_maps: Sensitivity maps [B, C, H, W, 2]
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
        
        # Apply U-Net denoising
        denoised = self.unet(images_chan)
        
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
        return torch.cat(updated_kspace, dim=1)


class PromptMRPlusReconstructor(nn.Module):
    """
    Full PromptMR+ reconstruction model
    """
    def __init__(self, num_cascades=12, chans=48, num_adj_slices=5, 
                 use_prompts=False, use_adaptive_input=False, 
                 use_history_features=False, learnable_dc=False,
                 use_checkpointing=False):
        super().__init__()
        
        self.num_cascades = num_cascades
        self.num_adj_slices = num_adj_slices
        self.center_slice = num_adj_slices // 2
        
        # Determine number of coils from your data
        # You'll need to adjust this based on your actual data
        self.coil_num = 15  # Assuming 15 coils total
        
        # Create cascade blocks
        self.cascades = nn.ModuleList([
            PromptMRBlock(num_adj_slices, chans, self.coil_num)
            for _ in range(num_cascades)
        ])
        
        self.use_checkpointing = use_checkpointing

    def forward(self, kspace, mask, sens_maps):
        """
        Args:
            kspace: Under-sampled k-space [B, C*adj, H, W, 2]
            mask: Sampling mask [B, 1, H, W, 1]
            sens_maps: Sensitivity maps [B, C, H, W, 2]
        Returns:
            Reconstructed image [B, H, W]
        """
        # Initialize with input k-space
        current_kspace = kspace
        
        # Apply cascades
        for cascade in self.cascades:
            if self.use_checkpointing and self.training:
                current_kspace = torch.utils.checkpoint.checkpoint(
                    cascade, current_kspace, kspace, mask, sens_maps
                )
            else:
                current_kspace = cascade(current_kspace, kspace, mask, sens_maps)
        
        # Extract center slice and convert to image
        # Get k-space for center slice
        start_idx = self.center_slice * self.coil_num
        end_idx = (self.center_slice + 1) * self.coil_num
        center_kspace = current_kspace[:, start_idx:end_idx]
        
        # Convert to image space
        center_image = ifft2c(center_kspace)
        
        # Combine coils using RSS
        magnitude = torch.sqrt(torch.sum(center_image[..., 0]**2 + center_image[..., 1]**2, dim=1))
        
        return magnitude


class ReconstructionLoss(nn.Module):
    """Loss function for reconstruction matching your training code"""
    def __init__(self, loss_type='ssim'):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'ssim':
            from pytorch_msssim import SSIM
            self.loss_fn = SSIM(data_range=1.0, size_average=True, channel=1)
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'combined':
            from pytorch_msssim import SSIM
            self.ssim = SSIM(data_range=1.0, size_average=True, channel=1)
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
        
        # Add channel dimension for SSIM
        if output_norm.dim() == 3:
            output_norm = output_norm.unsqueeze(1)
            target_norm = target_norm.unsqueeze(1)
        
        if self.loss_type == 'ssim':
            # Return 1 - SSIM as loss
            return 1 - self.loss_fn(output_norm, target_norm)
        elif self.loss_type == 'l1':
            return self.loss_fn(output_norm, target_norm)
        elif self.loss_type == 'combined':
            ssim_loss = 1 - self.ssim(output_norm, target_norm)
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
        use_history_features=args.use_history_features if hasattr(args, 'use_history_features') else False,
        learnable_dc=args.learnable_dc if hasattr(args, 'learnable_dc') else False,
        use_checkpointing=args.use_checkpointing if hasattr(args, 'use_checkpointing') else False
    )
    return model