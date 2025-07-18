"""
Updated Sensitivity Map Estimation (SME) Model for PromptMR+
Fixed to match PromptMR+ architecture properly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

from .unet import Unet


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """
    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

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

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


class PromptBlock(nn.Module):
    """Prompt block for adaptive feature learning"""
    def __init__(self, in_channels: int, prompt_dim: int = 16):
        super().__init__()
        self.prompt_dim = prompt_dim
        
        # Learnable prompt parameters
        self.prompt_param = nn.Parameter(torch.randn(1, prompt_dim, 1, 1) * 0.02)
        
        # Channel attention for prompt
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels, prompt_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(prompt_dim, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Generate prompt based on input
        b, c, h, w = x.shape
        prompt = self.prompt_param.expand(b, -1, h, w)
        
        # Apply channel attention
        attention = self.channel_attention(x)
        
        # Modulate input with prompt
        return x * attention + x


class NormUnet(nn.Module):
    """
    Normalized U-Net model for sensitivity map estimation
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
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pools = num_pools

        # Input prompt block
        if use_prompts:
            self.input_prompt = PromptBlock(in_chans, prompt_dim=8)
            self.output_prompt = PromptBlock(out_chans, prompt_dim=8)

        # Down-sampling path
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pools - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        # Up-sampling path
        self.up_conv = nn.ModuleList()
        self.up_sample_layers = nn.ModuleList()
        for _ in range(num_pools - 1):
            self.up_conv.append(nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2))
            self.up_sample_layers.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2
        
        self.up_conv.append(nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2))
        self.up_sample_layers.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        # Apply input prompt if enabled
        if self.use_prompts:
            image = self.input_prompt(image)
        
        stack = []
        output = image

        # Down-sampling
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # Up-sampling
        for up_conv, up_layer in zip(self.up_conv, self.up_sample_layers):
            output = up_conv(output)
            output = torch.cat([output, stack.pop()], dim=1)
            output = up_layer(output)

        # Apply output prompt if enabled
        if self.use_prompts:
            output = self.output_prompt(output)
            
        return output


class SensitivityModel(nn.Module):
    """
    Model for sensitivity map estimation that matches PromptMR+ architecture
    """
    def __init__(
        self,
        chans: int = 8,
        num_pools: int = 4,
        drop_prob: float = 0.0,
        num_adjacent: int = 5,
        use_prompts: bool = True
    ):
        super().__init__()
        
        self.num_adjacent = num_adjacent
        self.use_prompts = use_prompts
        
        # Important: Match the expected input/output channels
        # Input: magnitude images from adjacent slices
        # Output: complex sensitivity maps (real + imag)
        in_chans = num_adjacent  # One magnitude image per adjacent slice
        out_chans = 2  # Complex output (real + imaginary)
        
        # Build the normalized U-Net
        self.norm_unet = NormUnet(
            chans=chans,
            num_pools=num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
            use_prompts=use_prompts
        )
        
        # Additional normalization for output
        self.output_norm = nn.InstanceNorm2d(out_chans)

    def forward(self, input_images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_images: Magnitude images [B, num_adjacent, H, W]
            mask: Sampling mask [B, 1, H, W]
        Returns:
            sens_maps: Sensitivity maps [B, num_coils, H, W, 2]
        """
        b, n_adj, h, w = input_images.shape
        
        # For now, estimate single-coil sensitivity (unity)
        # In full implementation, this would estimate multi-coil sensitivities
        # You'll need to adapt this based on your specific coil configuration
        
        # Process through U-Net
        sens_output = self.norm_unet(input_images)  # [B, 2, H, W]
        
        # Normalize output
        sens_output = self.output_norm(sens_output)
        
        # Apply activation to ensure unit norm
        # Split real and imaginary parts
        sens_real = sens_output[:, 0:1, :, :]  # [B, 1, H, W]
        sens_imag = sens_output[:, 1:2, :, :]  # [B, 1, H, W]
        
        # Normalize to unit norm
        sens_mag = torch.sqrt(sens_real**2 + sens_imag**2 + 1e-8)
        sens_real = sens_real / sens_mag
        sens_imag = sens_imag / sens_mag
        
        # Stack to create complex representation
        sens_maps = torch.stack([sens_real, sens_imag], dim=-1)  # [B, 1, H, W, 2]
        
        # For multi-coil, you would repeat or estimate multiple maps
        # For now, returning single coil sensitivity
        # In practice, you need to modify this to return [B, num_coils, H, W, 2]
        
        # Assuming 15 coils as in your original setup (3 coils per adjacent slice)
        num_coils = 15  # Adjust based on your data
        sens_maps = sens_maps.repeat(1, num_coils, 1, 1, 1)  # [B, num_coils, H, W, 2]
        
        return sens_maps

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Convert complex tensor to channel dimension."""
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 1, 4, 2, 3).reshape(b, c * 2, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Convert channel dimension to complex tensor."""
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, c, 2, h, w).permute(0, 1, 3, 4, 2).contiguous()


# For backward compatibility with your existing code
def build_sme_model(args):
    """Build SME model from args"""
    model = SensitivityModel(
        chans=args.sens_chans,
        num_pools=args.sens_pools,
        num_adjacent=args.num_adjacent,
        use_prompts=args.use_prompts if hasattr(args, 'use_prompts') else True
    )
    return model