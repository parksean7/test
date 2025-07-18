"""
Updated common utilities with proper complex operations
Maintains compatibility with your existing code
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import Optional
import random


def save_reconstructions(reconstructions, out_dir, targets=None):
    """
    Save reconstruction results maintaining your original format
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)
            if targets is not None and fname in targets:
                f.create_dataset('target', data=targets[fname])


def ssim_loss(gt, pred, maxval=None):
    """
    Compute SSIM loss (returns 1 - SSIM for use as loss)
    This matches your existing training code interface
    """
    # Handle different input shapes
    if gt.dim() == 3:  # [B, H, W]
        gt = gt.unsqueeze(1)  # [B, 1, H, W]
        pred = pred.unsqueeze(1)
    
    # Normalize if maxval provided
    if maxval is not None:
        if maxval.dim() > 0:
            maxval = maxval.view(-1, 1, 1, 1)
        gt = gt / maxval
        pred = pred / maxval
        data_range = 1.0
    else:
        data_range = gt.max()
    
    # Compute SSIM using torch
    return 1 - compute_ssim_torch(pred, gt, data_range=data_range)


def compute_ssim_torch(img1, img2, data_range=1.0, size_average=True):
    """
    Compute SSIM using PyTorch operations
    """
    # Constants
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # Create Gaussian kernel
    kernel_size = 11
    sigma = 1.5
    kernel = create_gaussian_kernel(kernel_size, sigma)
    kernel = kernel.to(img1.device)
    
    # Ensure 4D tensors [B, C, H, W]
    if img1.dim() == 3:
        img1 = img1.unsqueeze(1)
        img2 = img2.unsqueeze(1)
    
    # Apply Gaussian filter
    mu1 = F.conv2d(img1, kernel, padding=kernel_size//2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, kernel, padding=kernel_size//2, groups=img2.shape[1])
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=kernel_size//2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=kernel_size//2, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=kernel_size//2, groups=img1.shape[1]) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def create_gaussian_kernel(size, sigma):
    """Create a Gaussian kernel for SSIM computation"""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = g.unsqueeze(0) * g.unsqueeze(1)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    return kernel


def complex_abs(data: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute the absolute value of a complex tensor
    """
    if data.shape[dim] == 2:  # Real representation [*, 2]
        return torch.sqrt(data[..., 0] ** 2 + data[..., 1] ** 2)
    else:  # Complex tensor
        return torch.abs(data)


def fft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2 dimensional Fast Fourier Transform
    """
    if data.shape[-1] == 2:  # Real representation
        data = torch.view_as_complex(data)
        
    data = torch.fft.ifftshift(data, dim=(-2, -1))
    data = torch.fft.fft2(data, dim=(-2, -1), norm=norm)
    data = torch.fft.fftshift(data, dim=(-2, -1))
    
    return torch.view_as_real(data)


def ifft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2 dimensional Inverse Fast Fourier Transform
    """
    if data.shape[-1] == 2:  # Real representation
        data = torch.view_as_complex(data)
        
    data = torch.fft.ifftshift(data, dim=(-2, -1))
    data = torch.fft.ifft2(data, dim=(-2, -1), norm=norm)
    data = torch.fft.fftshift(data, dim=(-2, -1))
    
    return torch.view_as_real(data)


def rss(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS)
    """
    return torch.sqrt((data ** 2).sum(dim))


def rss_complex(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS) for complex data
    """
    if data.shape[-1] == 2:  # Real representation
        return torch.sqrt(torch.sum(data[..., 0] ** 2 + data[..., 1] ** 2, dim=dim))
    else:
        return torch.sqrt(torch.sum(torch.abs(data) ** 2, dim=dim))


def seed_fix(seed):
    """Fix random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize(data, mean, stddev, eps=0.):
    """
    Normalize data using mean and standard deviation
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.):
    """
    Instance normalization
    """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std


def normalize_max(data, eps=0.):
    """
    Normalize by maximum value
    """
    max_val = data.max()
    return data / (max_val + eps), max_val


# Additional utilities for multi-coil MRI
def complex_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Complex multiplication of two tensors
    Args:
        a, b: Complex tensors with last dimension size 2 (real, imag)
    """
    assert a.shape[-1] == 2 and b.shape[-1] == 2
    
    real = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
    imag = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
    
    return torch.stack([real, imag], dim=-1)


def complex_conj(data: torch.Tensor) -> torch.Tensor:
    """
    Complex conjugate
    """
    assert data.shape[-1] == 2
    conj = data.clone()
    conj[..., 1] = -conj[..., 1]
    return conj


def sens_expand(image: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    """
    Expand coil-combined image to multi-coil k-space using sensitivity maps
    Args:
        image: Coil-combined image [B, 1, H, W, 2]
        sens_maps: Sensitivity maps [B, C, H, W, 2]
    Returns:
        Multi-coil k-space [B, C, H, W, 2]
    """
    # Complex multiplication: image * sens_maps
    coil_imgs = complex_mul(image, sens_maps)
    
    # Convert to k-space
    return fft2c_new(coil_imgs)


def sens_reduce(kspace: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    """
    Reduce multi-coil k-space to single image using sensitivity maps
    Args:
        kspace: Multi-coil k-space [B, C, H, W, 2]
        sens_maps: Sensitivity maps [B, C, H, W, 2]
    Returns:
        Combined image [B, 1, H, W, 2]
    """
    # Convert to image space
    coil_imgs = ifft2c_new(kspace)
    
    # Complex multiplication with conjugate: coil_imgs * conj(sens_maps)
    sens_maps_conj = complex_conj(sens_maps)
    combined = complex_mul(coil_imgs, sens_maps_conj)
    
    # Sum over coils
    return combined.sum(dim=1, keepdim=True)


def mask_center(x: torch.Tensor, mask_from: Optional[int], mask_to: Optional[int]) -> torch.Tensor:
    """
    Mask center k-space locations
    """
    mask = torch.zeros_like(x)
    mask[:, :, mask_from:mask_to] = x[:, :, mask_from:mask_to]
    return mask


def center_crop(data: torch.Tensor, shape: tuple) -> torch.Tensor:
    """
    Apply a center crop to 2D images.
    Args:
        data: Tensor with shape (..., H, W)
        shape: Desired output shape (H, W)
    Returns:
        Center-cropped tensor with shape (..., H_crop, W_crop)
    """
    if data.shape[-2:] == shape:
        return data

    h_from = (data.shape[-2] - shape[0]) // 2
    w_from = (data.shape[-1] - shape[1]) // 2
    h_to = h_from + shape[0]
    w_to = w_from + shape[1]

    return data[..., h_from:h_to, w_from:w_to]

def complex_center_crop(data: torch.Tensor, shape: tuple) -> torch.Tensor:
    """
    Apply a center crop to complex 2D images
    """
    if data.shape[-3:-1] == shape:
        return data
        
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    
    return data[..., w_from:w_to, h_from:h_to, :]