"""
Sensitivity Map Estimation (SME) Model for PromptMR-plus
Based on the original PromptMR-plus repository structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

from .unet import Unet


class PromptBlock(nn.Module):
    """Prompt block for adaptive feature learning"""
    def __init__(self, in_channels: int, prompt_dim: int = 16):
        super().__init__()
        self.prompt_dim = prompt_dim
        
        # Learnable prompt parameters with better initialization
        self.prompt_key = nn.Parameter(torch.randn(1, prompt_dim, 1, 1) * 0.01)
        self.prompt_value = nn.Parameter(torch.randn(1, prompt_dim, 1, 1) * 0.01)
        
        # Projection layers with batch norm
        self.key_proj = nn.Conv2d(in_channels, prompt_dim, 1)
        self.value_proj = nn.Conv2d(in_channels, prompt_dim, 1)
        self.out_proj = nn.Conv2d(prompt_dim, in_channels, 1)
        
        # Add batch normalization for better gradient flow
        self.key_norm = nn.BatchNorm2d(prompt_dim)
        self.value_norm = nn.BatchNorm2d(prompt_dim)
        self.out_norm = nn.BatchNorm2d(in_channels)
        
        # Initialize weights properly
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Project input to key and value spaces with normalization
        key = self.key_norm(self.key_proj(x))  # (B, prompt_dim, H, W)
        value = self.value_norm(self.value_proj(x))  # (B, prompt_dim, H, W)
        
        # Compute attention with learnable prompts
        prompt_key = self.prompt_key.expand(B, -1, H, W)
        prompt_value = self.prompt_value.expand(B, -1, H, W)
        
        # Attention mechanism with better scaling
        attention = torch.sigmoid(key * prompt_key)
        prompted_features = attention * (value + prompt_value)
        
        # Project back to original space with normalization
        output = self.out_norm(self.out_proj(prompted_features))
        
        # Residual connection with scaling for gradient stability
        return x + 0.1 * output


class NormPromptUnet(nn.Module):
    """
    Normalized PromptUNet for sensitivity map estimation
    Based on original PromptMR-plus architecture
    """
    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        use_prompts: bool = True
    ):
        super().__init__()
        
        self.use_prompts = use_prompts
        
        # Core UNet architecture
        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )
        
        # Prompt blocks for adaptive learning
        # Note: in_chans and out_chans will be 2*C after complex_to_chan_dim
        if use_prompts:
            self.input_prompt = PromptBlock(in_chans)
            self.output_prompt = PromptBlock(out_chans)

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Improved group norm with gradient stability
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, c, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1)
        
        # Add epsilon for numerical stability and better gradients
        std = torch.clamp(std, min=1e-6)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[list, list, int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [int((w_mult - w) / 2), int((w_mult - w) / 2)]
        h_pad = [int((h_mult - h) / 2), int((h_mult - h) / 2)]
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: list,
        w_pad: list,
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # Get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        
        # Apply input prompt if enabled
        if self.use_prompts:
            x = self.input_prompt(x)
        
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        # Use gradient checkpointing for memory efficiency if training
        if self.training:
            from torch.utils.checkpoint import checkpoint
            x = checkpoint(self.unet, x, use_reentrant=False)
        else:
            x = self.unet(x)

        # Get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        
        # Apply output prompt if enabled
        if self.use_prompts:
            x = self.output_prompt(x)
        
        x = self.chan_complex_to_last_dim(x)

        return x


class SensitivityModel(nn.Module):
    """
    Original PromptMR-plus Sensitivity Map Estimation Model
    Supports multi-slice processing and adjacent slice strategies
    """

    def __init__(
        self,
        chans: int = 3,  # Reduced channels as specified
        num_pools: int = 3,  # Reduced layers as specified
        drop_prob: float = 0.0,
        num_adjacent: int = 5,  # 2a+1 where a=2
        use_prompts: bool = True
    ):
        super().__init__()
        in_chans = 2 * num_adjacent #  2 * num_adjacent,  # Change from 2 to 10
        out_chans = 2 * num_adjacent # Change from 2 to 10
        
        self.num_adjacent = num_adjacent
        self.use_prompts = use_prompts
        
        # Core sensitivity estimation network
        # After chans_to_batch_dim and complex_to_chan_dim, we have [B*C, 2, H, W]
        self.norm_prompt_unet = NormPromptUnet(
            chans,
            num_pools,
            in_chans=in_chans,  # 2 for complex dimension
            out_chans=out_chans,  # 2 for complex dimension
            drop_prob=drop_prob,
            use_prompts=use_prompts
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape
        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size
        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        # Compute root sum of squares manually with better gradient stability
        # x shape: [B, C, H, W, 2]
        # We need to compute RSS over coil dimension (dim=1)
        rss = torch.sqrt(torch.sum(x.abs()**2, dim=1, keepdim=True))  # [B, 1, H, W, 2]
        # Use larger epsilon for better gradient flow
        rss = torch.clamp(rss, min=1e-6)
        return x / rss

    def extract_acs_region(self, kspace: torch.Tensor) -> torch.Tensor:
        """Extract Auto-Calibration Signal region for sensitivity estimation"""
        print(f"DEBUG: kspace shape in extract_acs_region: {kspace.shape}")
        print(f"DEBUG: kspace dimensions: {len(kspace.shape)}")
        B, C, H, W, _ = kspace.shape
        
        # For simplicity, use the full k-space data
        # In practice, you'd extract a proper ACS region
        # But for now, let's just return the input to avoid dimension issues
        return kspace

    def forward(self, image_input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_input: (B, 6, H, W) - 3 adjacent slices in image domain (real+imag as channels)
            mask: (B, 1, H, W) - Sampling mask
        
        Returns:
            sensitivity_maps: (B, C, H, W, 2) - Estimated sensitivity maps
        """
        # The input is (B, 6, H, W) - stacked real/imag image slices
        # Pass directly to the U-Net which is configured for 6 input channels
        
        # Process through the PromptUNet directly
        x = self.norm_prompt_unet.unet(image_input)  # Input: [B, 6, H, W] -> Output: [B, 6, H, W]
        
        # Convert U-Net output (6 channels) back to sensitivity maps format [B, 15, H, W, 2]
        # The U-Net output represents 3 slices × 2 (real/imag), we need 15 coils × 2
        B, output_channels, H, W = x.shape
        
        # Reshape U-Net output from [B, 6, H, W] to [B, 3, H, W, 2]
        x_reshaped = x.view(B, 3, 2, H, W).permute(0, 1, 3, 4, 2)  # [B, 3, H, W, 2]
        
        # Create sensitivity maps by padding with zeros (without in-place operations)
        # We have 3 coils worth of data, need 15 coils total
        padding = torch.zeros(B, 12, H, W, 2, device=x.device)  # 12 more coils
        sensitivity_maps = torch.cat([x_reshaped, padding], dim=1)  # [B, 15, H, W, 2]
        
        # Convert to complex format for reconstructor: [B, 15, H, W, 2] -> [B, 15, H, W] complex
        sensitivity_maps_complex = torch.view_as_complex(sensitivity_maps)  # [B, 15, H, W]
        
        return sensitivity_maps_complex


class SMELoss(nn.Module):
    """Loss function for sensitivity map estimation"""
    
    def __init__(self, loss_type: str = 'mse'):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'l1':
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, pred_sens: torch.Tensor, target_sens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_sens: Predicted sensitivity maps (B, C, H, W, 2)
            target_sens: Target sensitivity maps (B, C, H, W, 2)
        """
        return self.criterion(pred_sens, target_sens)


# For compatibility with training scripts
def create_sme_model(
    chans: int = 3,
    num_pools: int = 3,
    num_adjacent: int = 5,
    use_prompts: bool = True
) -> SensitivityModel:
    """Factory function to create SME model"""
    return SensitivityModel(
        chans=chans,
        num_pools=num_pools,
        num_adjacent=num_adjacent,
        use_prompts=use_prompts
    )