"""
Updated data loading module for PromptMR+
Maintains original structure while fixing critical issues
"""

import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import pathlib
import random


class SliceData(Dataset):
    def __init__(self, root, num_adjacent=5, transform=None, use_dataset_cache=True):
        self.transform = transform
        self.num_adjacent = num_adjacent
        self.examples = []

        files = list(pathlib.Path(root).iterdir())
        for fname in sorted(files):
            if fname.suffix == '.h5':
                with h5py.File(fname, 'r') as hf:
                    if 'kspace' in hf:
                        num_slices = hf['kspace'].shape[0]
                        self.examples += [(fname, slice_idx) for slice_idx in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice_idx = self.examples[i]
        
        with h5py.File(fname, 'r') as hf:
            # Load k-space data
            kspace = hf['kspace'][()]
            
            # Get adjacent slices
            kspace_adj = self._get_adjacent_slices(kspace, slice_idx)
            
            # Create mask (you should load or generate proper mask here)
            mask = self._create_mask(kspace_adj.shape)
            
            # Apply mask to k-space
            masked_kspace = kspace_adj * mask[..., None]
            
            # Get target
            if 'reconstruction_rss' in hf:
                target = hf['reconstruction_rss'][slice_idx]
            else:
                # Zero-filled reconstruction as target for test data
                target = self._get_zero_filled_reconstruction(kspace[slice_idx])
            
            # Get maximum value for normalization
            if 'max' in hf.attrs:
                maximum = hf.attrs['max']
            else:
                maximum = np.max(np.abs(target))
            
            # Create SME input (magnitude images of adjacent slices)
            sme_input = self._create_sme_input(kspace_adj)
            
        # Convert to tensors
        masked_kspace = torch.from_numpy(masked_kspace).float()
        mask = torch.from_numpy(mask).float()
        target = torch.from_numpy(target).float()
        maximum = torch.tensor(maximum).float()
        sme_input = torch.from_numpy(sme_input).float()
        
        # Return with proper format matching your train_part.py expectations
        # (mask, kspace, target, maximum, fname, slice, sme_input)
        return mask, masked_kspace, target, maximum, fname.name, slice_idx, sme_input

    def _get_adjacent_slices(self, kspace, slice_idx):
        """Get adjacent slices with proper handling of boundaries"""
        num_slices, num_coils, h, w = kspace.shape
        half_adj = self.num_adjacent // 2
        
        # Collect adjacent slices
        slices = []
        for offset in range(-half_adj, half_adj + 1):
            idx = slice_idx + offset
            # Handle boundaries by clamping
            idx = max(0, min(idx, num_slices - 1))
            slices.append(kspace[idx])
        
        # Stack and reshape: [num_adj, coils, H, W] -> [coils*num_adj, H, W]
        kspace_adj = np.concatenate(slices, axis=0)
        
        # Convert complex to real representation if needed
        if np.iscomplexobj(kspace_adj):
            kspace_adj = np.stack([kspace_adj.real, kspace_adj.imag], axis=-1)
        elif kspace_adj.shape[-1] != 2:
            # If not complex and doesn't have last dim 2, add it
            kspace_adj = np.stack([kspace_adj, np.zeros_like(kspace_adj)], axis=-1)
            
        return kspace_adj

    def _create_mask(self, shape):
        """Create undersampling mask"""
        # Simple center + random mask for now
        # You should use your actual mask generation
        mask = np.zeros((1, shape[-2], 1), dtype=np.float32)
        
        # Center lines (ACS)
        center = shape[-2] // 2
        num_center = 24  # Adjust based on your acceleration
        mask[:, center - num_center//2:center + num_center//2, :] = 1
        
        # Random lines
        acceleration = 4  # Adjust based on your needs
        num_lines = shape[-2] // acceleration
        indices = np.random.choice(shape[-2], num_lines - num_center, replace=False)
        mask[:, indices, :] = 1
        
        return mask

    def _get_zero_filled_reconstruction(self, kspace_slice):
        """Get zero-filled reconstruction from k-space"""
        # Convert to image space
        if np.iscomplexobj(kspace_slice):
            image = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(kspace_slice, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
            # Root sum of squares combination
            return np.sqrt(np.sum(np.abs(image)**2, axis=0))
        else:
            # If not complex, assume it's already in image space
            return np.sqrt(np.sum(kspace_slice**2, axis=0))

    def _create_sme_input(self, kspace_adj):
        """Create input for sensitivity map estimation"""
        # Convert to image space and take magnitude
        # Shape: [coils*num_adj, H, W, 2] -> [num_adj, H, W]
        coils_per_adj = kspace_adj.shape[0] // self.num_adjacent
        
        sme_input = []
        for i in range(self.num_adjacent):
            start_idx = i * coils_per_adj
            end_idx = (i + 1) * coils_per_adj
            
            # Get k-space for this adjacent slice
            k_slice = kspace_adj[start_idx:end_idx]
            
            # Convert to complex if in real representation
            if k_slice.shape[-1] == 2:
                k_complex = k_slice[..., 0] + 1j * k_slice[..., 1]
            else:
                k_complex = k_slice
            
            # IFFT to image space
            img = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(k_complex, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
            
            # RSS combination
            rss = np.sqrt(np.sum(np.abs(img)**2, axis=0))
            sme_input.append(rss)
        
        # Stack to [num_adj, H, W]
        sme_input = np.stack(sme_input, axis=0)
        
        # Add channel dimension to match expected format [1, num_adj, H, W]
        return np.expand_dims(sme_input, axis=0)


def create_data_loaders(data_path, args, shuffle=False, isforward=False):
    """
    Create data loaders matching your original interface
    """
    if isforward:
        # For forward/test mode
        dataset = SliceData(
            root=data_path,
            num_adjacent=getattr(args, 'num_adjacent', 5),
            transform=None
        )
        
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return data_loader
    
    else:
        # For training - return both train and val loaders
        train_data = SliceData(
            root=args.data_path_train,
            num_adjacent=getattr(args, 'num_adjacent', 5),
            transform=None
        )
        
        val_data = SliceData(
            root=args.data_path_val,
            num_adjacent=getattr(args, 'num_adjacent', 5),
            transform=None
        )
        
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            dataset=val_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # For display loader (used in your training code)
        display_loader = DataLoader(
            dataset=val_data,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader, display_loader