"""
Fixed SME Model Implementation
- Fixes shape compatibility issues
- Properly outputs multi-coil sensitivity maps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


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


class PromptBlock(nn.Module):
    """Simple prompt block for learning adaptive features"""
    def __init__(self, in_chans, prompt_dim=8):
        super().__init__()
        self.prompt_dim = prompt_dim
        self.prompt_conv = nn.Conv2d(in_chans, prompt_dim, kernel_size=1)
        self.output_conv = nn.Conv2d(in_chans + prompt_dim, in_chans, kernel_size=1)
        
    def forward(self, x):
        prompt = self.prompt_conv(x)
        combined = torch.cat([x, prompt], dim=1)
        return self.output_conv(combined)


class SMEUnet(nn.Module):
    """
    U-Net for sensitivity map estimation
    """
    def __init__(self, in_chans, out_chans, num_pool_layers=4, chans=32, drop_prob=0.0, use_prompts=True):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_prompts = use_prompts

        # Prompt blocks
        if self.use_prompts:
            self.input_prompt = PromptBlock(in_chans, prompt_dim=8)
            self.output_prompt = PromptBlock(out_chans, prompt_dim=8)

        # Down-sampling path
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        # Up-sampling path
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
                nn.Conv2d(ch, out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        FIXED: Forward pass with proper size handling for skip connections
        
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

        # Down-sampling (encoder)
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        # Bottleneck
        output = self.conv(output)

        # Up-sampling (decoder) with proper size matching
        for up_conv, up_layer in zip(self.up_conv, self.up_sample_layers):
            # Upsample
            output = up_conv(output)
            
            # Get skip connection feature
            skip_feat = stack.pop()
            
            # FIXED: Ensure spatial dimensions match before concatenation
            if output.shape[2:] != skip_feat.shape[2:]:
                # Resize output to match skip connection size
                output = F.interpolate(
                    output, 
                    size=skip_feat.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Concatenate along channel dimension
            output = torch.cat([output, skip_feat], dim=1)
            
            # Apply conv layers
            output = up_layer(output)

        # Apply output prompt if enabled
        if self.use_prompts:
            output = self.output_prompt(output)
            
        return output


class SensitivityModel(nn.Module):
    """
    FIXED: Model for sensitivity map estimation that properly outputs multi-coil maps
    """
    def __init__(
        self,
        chans: int = 8,
        num_pools: int = 4,
        drop_prob: float = 0.0,
        num_adjacent: int = 5,
        use_prompts: bool = True,
        num_coils: int = 15  # FIXED: Add num_coils parameter
    ):
        super().__init__()
        
        self.num_adjacent = num_adjacent
        self.num_pools = num_pools
        self.use_prompts = use_prompts
        self.num_coils = num_coils  # FIXED: Store num_coils
        
        # FIXED: Input is magnitude images from adjacent slices
        # Output should be complex sensitivity maps for all coils
        in_chans = num_adjacent  # Adjacent RSS magnitude images
        out_chans = num_coils * 2  # Real and imaginary parts for each coil
        
        self.unet = SMEUnet(
            in_chans=in_chans,
            out_chans=out_chans,
            num_pool_layers=num_pools,
            chans=chans,
            drop_prob=drop_prob,
            use_prompts=use_prompts
        )
        
        # FIXED: Initialize sensitivity maps with proper normalization
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:  # ← Add this check
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None:  # ← Add this check
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:    # ← Add this check
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        FIXED: Forward pass for sensitivity map estimation
        
        Args:
            x: Adjacent RSS magnitude images [B, num_adjacent, H, W]
            mask: Sampling mask (not used in current implementation)
        Returns:
            Sensitivity maps [B, num_coils, H, W, 2]
        """
        B, num_adj, H, W = x.shape
        
        # Ensure we have the right number of adjacent slices
        if num_adj != self.num_adjacent:
            if num_adj < self.num_adjacent:
                # Pad with repeated slices
                pad_slices = self.num_adjacent - num_adj
                x = torch.cat([x, x[:, -1:].repeat(1, pad_slices, 1, 1)], dim=1)
            else:
                # Truncate to required number
                x = x[:, :self.num_adjacent]
        
        # Forward through U-Net
        # Input: [B, num_adjacent, H, W]
        # Output: [B, num_coils*2, H, W]
        sens_maps_flat = self.unet(x)
        
        # FIXED: Reshape to proper sensitivity map format
        # [B, num_coils*2, H, W] -> [B, num_coils, H, W, 2]
        sens_maps = sens_maps_flat.view(B, self.num_coils, 2, H, W)
        sens_maps = sens_maps.permute(0, 1, 3, 4, 2).contiguous()
        
        # FIXED: Apply proper normalization for sensitivity maps
        # Compute RSS and normalize
        sens_magnitude = torch.sqrt(
            sens_maps[..., 0] ** 2 + sens_maps[..., 1] ** 2 + 1e-8
        )  # [B, num_coils, H, W]
        
        # RSS across coils
        rss = torch.sqrt(torch.sum(sens_magnitude ** 2, dim=1, keepdim=True) + 1e-8)
        
        # Normalize sensitivity maps
        sens_maps_real = sens_maps[..., 0] / rss
        sens_maps_imag = sens_maps[..., 1] / rss
        
        # Stack back to complex format
        sens_maps_normalized = torch.stack([sens_maps_real, sens_maps_imag], dim=-1)
        
        return sens_maps_normalized

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Convert channel dimension to complex tensor."""
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, c, 2, h, w).permute(0, 1, 3, 4, 2).contiguous()

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Convert complex tensor to channel dimension."""
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 1, 4, 2, 3).reshape(b, c * 2, h, w)


# For backward compatibility with your existing code
def build_sme_model(args):
    """Build SME model from args"""
    # FIXED: Get num_coils from args or use default
    num_coils = getattr(args, 'num_coils', 15)
    
    model = SensitivityModel(
        chans=args.sens_chans,
        num_pools=args.sens_pools,
        num_adjacent=args.num_adjacent,
        use_prompts=getattr(args, 'use_prompts', True),
        num_coils=num_coils  # FIXED: Pass num_coils
    )
    return model