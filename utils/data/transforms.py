import torch
import numpy as np

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor.
    """
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    return data

class DataTransform:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key
        
    def __call__(self, mask, input, target, attrs, fname, slice):
        # Based on original PromptMR+ transforms.py
        # Keep it simple and follow the original pattern
        
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs.get(self.max_key, 1.0)
        else:
            target = -1
            maximum = -1
        
        # Apply mask to k-space data
        kspace_raw = to_tensor(input * mask)  # [slices, coils, H, W] complex
        
        # Select middle slice for reconstructor (single slice processing)
        num_slices = kspace_raw.shape[0]
        middle_slice_idx = num_slices // 2
        kspace_single_slice = kspace_raw[middle_slice_idx]  # [coils, H, W] complex
        
        # For reconstructor: convert to [coils, H, W, 2] format
        kspace_reconstructor = torch.stack([kspace_single_slice.real, kspace_single_slice.imag], dim=-1)  # [coils, H, W, 2]
        kspace_reconstructor = kspace_reconstructor.unsqueeze(0)  # Add batch dimension: [1, coils, H, W, 2]
        
        # For SME: Take 3 adjacent slices, sum over coils, IFFT, stack real/imag
        if num_slices >= 3:
            start_idx = (num_slices - 3) // 2
            kspace_3slices = kspace_raw[start_idx:start_idx+3]  # [3, coils, H, W] complex
        else:
            # Pad with replicated slices if needed
            kspace_3slices = kspace_raw
            while kspace_3slices.shape[0] < 3:
                kspace_3slices = torch.cat([kspace_3slices, kspace_raw[-1:]], dim=0)
            kspace_3slices = kspace_3slices[:3]  # Take first 3
        
        # Sum over coils for SME input
        kspace_sos = kspace_3slices.sum(dim=1)  # [3, H, W] complex
        
        # Convert to image domain and stack real/imag
        sme_input_slices = []
        for slice_idx in range(kspace_sos.shape[0]):
            # IFFT to get image
            image = torch.fft.ifft2(kspace_sos, norm='ortho')
            # Stack real/imag
            image_real_imag = torch.stack([image.real, image.imag], dim=0)  # [2, H, W]
            sme_input_slices.append(image_real_imag)
        
        # Concatenate to get [6, H, W] for SME
        sme_input = torch.cat(sme_input_slices, dim=0)  # [6, H, W]
        sme_input = sme_input.unsqueeze(0)  # Add batch dimension: [1, 6, H, W]
        
        # Return the reconstructor format as main output
        kspace = kspace_reconstructor
        
        # Create mask tensor - broadcast 1D mask to spatial dimensions  
        mask_tensor = torch.from_numpy(mask.astype(np.float32))
        if mask_tensor.ndim == 1:
            # Expand 1D mask to 2D spatial mask
            mask_tensor = mask_tensor.unsqueeze(0).expand(kspace.shape[-2], -1)
        
        # Add batch dimension
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, height, width)
        
        # Return both formats: reconstructor format as main, SME format as additional
        return mask_tensor, kspace, target, maximum, fname, slice_idx