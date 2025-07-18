"""
PromptMR-plus Reconstructor Model
Based on the original PromptMR-plus repository structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Any

from .unet import Unet
from ..common.utils import center_crop


class PromptBlock(nn.Module):
    """Prompt block for adaptive feature learning"""
    def __init__(self, in_channels: int, prompt_dim: int = 16):
        super().__init__()
        self.prompt_dim = prompt_dim
        
        # Learnable prompt parameters with better initialization
        self.prompt_key = nn.Parameter(torch.randn(1, prompt_dim, 1, 1) * 0.01)
        self.prompt_value = nn.Parameter(torch.randn(1, prompt_dim, 1, 1) * 0.01)
        
        # Projection layers
        self.key_proj = nn.Conv2d(in_channels, prompt_dim, 1)
        
        # Single LayerNorm applied before projections (as in original paper)
        self.input_norm = nn.LayerNorm([in_channels])
        self.output_norm = nn.LayerNorm([in_channels])
        
        # Initialize weights with much smaller variance for stability
        nn.init.xavier_uniform_(self.key_proj.weight, gain=0.01)
        nn.init.xavier_uniform_(self.value_proj.weight, gain=0.01)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.01)
        
        # Initialize biases to zero
        nn.init.zeros_(self.key_proj.bias)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Apply LayerNorm to the input (reshape for LayerNorm)
        x_norm = x.permute(0, 2, 3, 1)  # (B, H, W, in_channels)
        x_norm = self.input_norm(x_norm).permute(0, 3, 1, 2)  # Back to (B, in_channels, H, W)
        
        # Project input to key and value spaces
        key = self.key_proj(x_norm)  # (B, prompt_dim, H, W)
        value = self.value_proj(x_norm)  # (B, prompt_dim, H, W)

        
        # Compute attention with learnable prompts
        prompt_key = self.prompt_key.expand(B, -1, H, W)
        prompt_value = self.prompt_value.expand(B, -1, H, W)
        
        # Attention mechanism with better scaling
        attention = torch.sigmoid(key * prompt_key)
        prompted_features = attention * (value + prompt_value)
        
        # Project back to original space
        output = self.out_proj(prompted_features)

        # Apply light dropout for regularization
        output = F.dropout(output, p=0.05, training=self.training)

        output = output.permute(0, 2, 3, 1)
        output = self.output_norm(output).permute(0, 3, 1, 2)

        # Apply layer normalization to output
        output = output.permute(0, 2, 3, 1)  # (B, H, W, in_channels)
        output = self.output_norm(output).permute(0, 3, 1, 2)  # Back to (B, in_channels, H, W)
        
        # Residual connection with balanced scaling for gradient stability
        return x + 0.05 * output


class AdaptiveInputBuffer(nn.Module):
    """Adaptive input buffer for handling variable inputs"""
    def __init__(self, in_channels: int, buffer_size: int = 3):
        super().__init__()
        self.buffer_size = buffer_size
        self.adaptation = nn.Conv2d(in_channels * buffer_size, in_channels, 1)
        
    def forward(self, current: torch.Tensor, history: List[torch.Tensor]) -> torch.Tensor:
        # Maintain a buffer of recent inputs
        if len(history) == 0:
            return current
        
        # Use only the most recent inputs
        recent_history = history[-self.buffer_size+1:] if len(history) >= self.buffer_size-1 else history
        
        # Pad with current if not enough history
        while len(recent_history) < self.buffer_size - 1:
            recent_history = [current] + recent_history
        
        # Concatenate current with history
        buffered_inputs = torch.cat([current] + recent_history, dim=1)
        
        # Adapt the concatenated features
        adapted = self.adaptation(buffered_inputs)
        
        return adapted


class HistoryFeatureTracker(nn.Module):
    """Track and integrate history features across cascades"""
    def __init__(self, feature_dim: int, history_length: int = 5):
        super().__init__()
        self.history_length = history_length
        self.feature_integration = nn.Conv2d(feature_dim * 2, feature_dim, 1)
        
    def forward(self, current_features: torch.Tensor, feature_history: List[torch.Tensor]) -> torch.Tensor:
        if len(feature_history) == 0:
            return current_features
        
        # Aggregate recent history
        recent_features = feature_history[-self.history_length:]
        if len(recent_features) > 1:
            aggregated_history = torch.stack(recent_features).mean(0)
        else:
            aggregated_history = recent_features[0]
        
        # Integrate current with history
        integrated = torch.cat([current_features, aggregated_history], dim=1)
        output = self.feature_integration(integrated)
        
        return output


class PromptUNet(nn.Module):
    """
    PromptUNet for denoising in PromptMR-plus reconstruction
    """
    def __init__(
        self,
        in_chans: int = 4,  # Current image + auxiliary inputs
        out_chans: int = 2,  # Real and imaginary parts
        chans: int = 8,  # Reduced channels as specified
        num_pool_layers: int = 4,  # Keep 4 layers as requested
        drop_prob: float = 0.2,  # Increased dropout for regularization
        use_prompts: bool = True,
        use_adaptive_input: bool = True,
        use_history_features: bool = True
    ):
        super().__init__()
        
        self.use_prompts = use_prompts
        self.use_adaptive_input = use_adaptive_input
        self.use_history_features = use_history_features
        
        # Core UNet
        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pool_layers,
            drop_prob=drop_prob,
        )
        
        # PromptMR-plus components (attention blocks disabled for stability)
        # Note: PromptBlocks are disabled to prevent gradient explosion
        # if use_prompts:
        #     self.input_prompt = PromptBlock(in_chans)
        #     self.output_prompt = PromptBlock(out_chans)
        
        if use_adaptive_input:
            self.adaptive_buffer = AdaptiveInputBuffer(in_chans)
        
        if use_history_features:
            self.history_tracker = HistoryFeatureTracker(out_chans)

    def forward(
        self, 
        x: torch.Tensor, 
        input_history: Optional[List[torch.Tensor]] = None,
        feature_history: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        
        # Adaptive input buffering
        if self.use_adaptive_input and input_history is not None:
            x = self.adaptive_buffer(x, input_history)
        
        # Input prompting (DISABLED for stability)
        # if self.use_prompts:
        #     x = self.input_prompt(x)
        
        # Core UNet processing
        output = self.unet(x)
        
        # Bound the output to prevent explosion and stabilize residual updates
        output = torch.tanh(output)
        
        # Output prompting (DISABLED for stability)
        # if self.use_prompts:
        #     output = self.output_prompt(output)
        
        # History feature integration
        if self.use_history_features and feature_history is not None:
            output = self.history_tracker(output, feature_history)
        
        return output


class DataConsistencyBlock(nn.Module):
    """Data consistency block for k-space constraint"""
    def __init__(self, learnable_dc: bool = True):
        super().__init__()
        self.learnable_dc = learnable_dc
        if learnable_dc:
            # Initialize with a smaller, more stable weight
            self.dc_weight = nn.Parameter(torch.ones(1) * 0.1)
    
    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """Apply sensitivity maps and forward FFT"""
        # x: [B, 1, H, W, 2], sens_maps: [B, C, H, W, 2]
        # Broadcast x to have coil dimension
        x_expanded = x.expand(-1, sens_maps.shape[1], -1, -1, -1)  # [B, C, H, W, 2]
        image_with_sens = x_expanded * sens_maps  # [B, C, H, W, 2]
        image_complex = torch.view_as_complex(image_with_sens)  # [B, C, H, W]
        kspace_complex = torch.fft.fft2(image_complex, norm='ortho')  # [B, C, H, W]
        return torch.view_as_real(kspace_complex)  # [B, C, H, W, 2]
    
    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """Apply inverse FFT and coil combination"""
        x = torch.fft.ifft2(torch.view_as_complex(x), norm='ortho')
        x_real = torch.view_as_real(x)
        sens_conj = torch.stack([sens_maps[..., 0], -sens_maps[..., 1]], dim=-1)
        return (x_real * sens_conj).sum(
            dim=1, keepdim=True
        )
    
    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply data consistency constraint
        """
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        mask_bool = mask.bool()
        
        # Soft data consistency
        if self.learnable_dc:
            soft_dc = torch.where(mask_bool, current_kspace - ref_kspace, zero) * self.dc_weight
        else:
            # If not learnable, use a fixed small weight for stability
            soft_dc = torch.where(mask_bool, current_kspace - ref_kspace, zero) * 0.1
        
        # Return the k-space after a soft data consistency update.
        # The highly unstable 'model_term' has been removed.
        return current_kspace - soft_dc


class PromptMRPlusReconstructor(nn.Module):
    """
    Original PromptMR-plus Reconstructor Model
    Implements the full cascade reconstruction with prompts
    """
    
    def __init__(
        self,
        num_cascades: int = 5,  # Reduced cascades as specified
        chans: int = 8,  # Reduced channels as specified
        sens_chans: int = 3,  # Not used here, for compatibility
        use_prompts: bool = True,
        use_adaptive_input: bool = True,
        use_history_features: bool = True,
        learnable_dc: bool = True,
        use_checkpointing: bool = False  # Can enable for memory saving
    ):
        super().__init__()
        
        self.num_cascades = num_cascades
        self.use_checkpointing = use_checkpointing
        
        # PromptUNet for each cascade (prompts disabled for stability)
        self.prompt_unets = nn.ModuleList([
            PromptUNet(
                in_chans=4,  # Current image (2) + auxiliary (2)
                out_chans=2,  # Real and imaginary output
                chans=chans,
                num_pool_layers=4,
                use_prompts=False,  # Disabled for stability
                use_adaptive_input=use_adaptive_input,
                use_history_features=use_history_features
            ) for _ in range(num_cascades)
        ])
        
        # Initialize UNet weights properly for better convergence
        for unet in self.prompt_unets:
            for m in unet.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        # Data consistency layers (using the new layer)
        self.dc_blocks = nn.ModuleList([  
            DataConsistencyBlock(learnable_dc=learnable_dc) 
            for _ in range(num_cascades)
        ])
        
        # Optional learnable combination weights (much smaller to match target scale)
        # Initialize near-zero to prevent large updates early in training
        # Use a parameter that will be passed through tanh for bounded updates
        self.cascade_weights = nn.Parameter(torch.randn(num_cascades))

    def prepare_input(self, image: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        """Prepare input for PromptUNet"""
        # Convert complex to real/imag channels
        image_real = image[..., 0]  # (B, 1, H, W)
        image_imag = image[..., 1]  # (B, 1, H, W)
        aux_real = auxiliary[..., 0]  # (B, 1, H, W)
        aux_imag = auxiliary[..., 1]  # (B, 1, H, W)
        
        # Concatenate to form 4-channel input
        return torch.cat([image_real, image_imag, aux_real, aux_imag], dim=1)

    def forward(
        self, 
        masked_kspace: torch.Tensor, 
        mask: torch.Tensor,
        sens_maps: torch.Tensor
    ) -> torch.Tensor:
        
        # Debug flag to avoid too much output
        debug_print = getattr(self, '_debug_counter', 0) < 3
        if debug_print:
            self._debug_counter = getattr(self, '_debug_counter', 0) + 1
        """
        Args:
            masked_kspace: (B, C, H, W, 2) - Under-sampled k-space data
            mask: (B, 1, H, W, 1) - Sampling mask (byte tensor)
            sens_maps: (B, C, H, W, 2) - Sensitivity maps from SME model
        
        Returns:
            reconstructed: (B, H, W) - Final reconstruction
        """
        
        # Initialize with zero-filled reconstruction
        current_kspace = masked_kspace.clone()
        kspace_complex = torch.view_as_complex(current_kspace)
        image_complex = torch.fft.ifft2(kspace_complex, norm='ortho')
        image_real = torch.view_as_real(image_complex)
        sens_conj = torch.stack([sens_maps[..., 0], -sens_maps[..., 1]], dim=-1)
        current_image = (image_real * sens_conj).sum(dim=1, keepdim=True)

        # Initialize auxiliary information (could be previous iteration or other info)
        auxiliary = current_image.clone()
        
        # History tracking
        input_history = []
        feature_history = []
        cascade_outputs = []
        
        for i in range(self.num_cascades):
            # Data consistency
            current_kspace = self.dc_blocks[i](
                current_kspace, masked_kspace, mask, sens_maps
            )
            
            # Update current image
            kspace_complex = torch.view_as_complex(current_kspace)
            image_complex = torch.fft.ifft2(kspace_complex, norm='ortho')
            image_real = torch.view_as_real(image_complex)
            sens_conj = torch.stack([sens_maps[..., 0], -sens_maps[..., 1]], dim=-1)
            current_image = (image_real * sens_conj).sum(dim=1, keepdim=True)
            
            # Prepare input for PromptUNet
            unet_input = self.prepare_input(current_image, auxiliary)
            
            # Store input history
            input_history.append(unet_input.detach())
            if len(input_history) > 5:  # Keep only recent history
                input_history = input_history[-5:]
            
            # Denoising with PromptUNet
            if self.use_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                denoised = checkpoint(
                    self.prompt_unets[i], 
                    unet_input, 
                    input_history[:-1] if len(input_history) > 1 else None,
                    feature_history,
                    use_reentrant=False
                )
            else:
                denoised = self.prompt_unets[i](
                    unet_input,
                    input_history[:-1] if len(input_history) > 1 else None,
                    feature_history
                )
            
            # Convert denoised output back to complex format
            denoised_complex = torch.stack([
                denoised[:, 0:1],  # Real part
                denoised[:, 1:2]   # Imaginary part
            ], dim=-1)
            
            # Update image with residual learning
            # The extra * 0.001 was making the update step too small.
            # Apply tanh to the weight and a tiny fixed scale for a stable, bounded update.
            scaled_weight = torch.tanh(self.cascade_weights[i]) * 1e-4
            current_image = current_image + scaled_weight * denoised_complex

            if i == 0 and debug_print:
                print(f"Cascade {i}: scaled_weight={scaled_weight.item():.6f}, denoised_range=[{denoised_complex.min():.6f}, {denoised_complex.max():.6f}]")
            
            # Update k-space
            image_with_sens = current_image * sens_maps
            image_complex = torch.view_as_complex(image_with_sens)
            current_kspace_complex = torch.fft.fft2(image_complex, norm='ortho')
            current_kspace = torch.view_as_real(current_kspace_complex)
            
            # Update auxiliary information (could be refined)
            auxiliary = current_image.clone()
            
            # Store feature history
            feature_history.append(denoised.detach())
            if len(feature_history) > 5:  # Keep only recent history
                feature_history = feature_history[-5:]
            
            # Store cascade output for potential ensemble
            cascade_outputs.append(current_image.clone())
        
        # Final reconstruction (take magnitude and crop)
        final_image = torch.sqrt(current_image[..., 0]**2 + current_image[..., 1]**2).squeeze(1)
        final_image = center_crop(final_image, 384, 384)
        
        # Debug: print final output range (limited prints)
        if debug_print:
            print(f"Final output range: [{final_image.min():.6f}, {final_image.max():.6f}]")
        
        return final_image


class ReconstructionLoss(nn.Module):
    """Loss function for reconstruction training"""
    
    def __init__(self, loss_type: str = 'ssim', ssim_weight: float = 0.8):
        super().__init__()
        self.loss_type = loss_type
        self.ssim_weight = ssim_weight
        
        if self.loss_type == 'ssim' or self.loss_type == 'combined':
            from utils.common.loss_function import SSIMLoss
            self.ssim_loss = SSIMLoss()
        
        if self.loss_type == 'l1' or self.loss_type == 'combined':
            self.l1_loss = nn.L1Loss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted reconstruction (B, H, W)
            target: Target image (B, H, W)
            max_val: Maximum value for normalization (B,)
        """
        if self.loss_type == 'ssim':
            return self.ssim_loss(pred, target, max_val)
        elif self.loss_type == 'l1':
            return self.l1_loss(pred, target)
        elif self.loss_type == 'combined':
            ssim_loss = self.ssim_loss(pred, target, max_val)
            l1_loss = self.l1_loss(pred, target)
            return self.ssim_weight * ssim_loss + (1 - self.ssim_weight) * l1_loss
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


# Factory function for compatibility
def create_promptmr_plus_reconstructor(
    num_cascades: int = 5,  # Default to 5 for consistency
    chans: int = 8,
    use_prompts: bool = True,
    learnable_dc: bool = True
) -> PromptMRPlusReconstructor:
    """Factory function to create PromptMR-plus reconstructor"""
    return PromptMRPlusReconstructor(
        num_cascades=num_cascades,
        chans=chans,
        use_prompts=use_prompts,
        learnable_dc=learnable_dc
    )