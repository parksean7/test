import h5py
import random
from transforms import DataTransform  # Use the simple, correct one
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False, num_adjacent=5):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.num_adjacent = num_adjacent
        self.examples = []

        root_path = Path(root)
        print(f"Loading data from: {root_path}")

        h5_files = sorted(root_path.glob("*.h5"))
        print(f"Found {len(h5_files)} .h5 files")

        for fname in h5_files:
            try:
                num_slices = self._get_metadata(fname)
                if num_slices > 0:
                    self.examples += [(fname, slice_idx) for slice_idx in range(num_slices)]
            except Exception as e:
                print(f"Error reading file {fname}: {e}")

        print(f"Total examples: {len(self.examples)}")
        if len(self.examples) == 0:
            raise ValueError(f"No valid .h5 files found in {root_path}")

    def _get_metadata(self, fname):
        try:
            with h5py.File(fname, "r") as hf:
                if self.input_key in hf:
                    return hf[self.input_key].shape[0]
                elif not self.forward and self.target_key in hf:
                    return hf[self.target_key].shape[0]
                else:
                    print(f"Missing keys in {fname.name}: {list(hf.keys())}")
                    return 0
        except Exception as e:
            print(f"Error reading metadata from {fname}: {e}")
            return 0

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice_idx = self.examples[i]

        try:
            with h5py.File(fname, "r") as hf:
                input_data = hf[self.input_key][slice_idx]
                mask = np.array(hf["mask"]) if "mask" in hf else np.ones_like(input_data)
                if self.forward:
                    target = -1
                    attrs = -1
                else:
                    target = hf[self.target_key][slice_idx]
                    attrs = dict(hf.attrs)
        except Exception as e:
            print(f"Error loading data from {fname}: {e}")
            target = np.zeros(input_data.shape[-2:], dtype=np.float32)
            attrs = {}

        return self.transform(mask, input_data, target, attrs, fname.name, slice_idx)



def create_data_loaders(data_path, args, shuffle=False, isforward=False):
    """
    Create data loaders matching your original interface
    """
    if isforward:
        # For forward/test mode
        dataset = SliceData(
            root=data_path,
            transform=DataTransform(isforward, -1),
            input_key=args.input_key,
            target_key=-1,
            forward=True,
            num_adjacent=getattr(args, 'num_adjacent', 5)
        )
        
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        return data_loader
    
    else:
        # For training - return both train and val loaders
        train_data = SliceData(
            root=args.data_path_train,
            transform=DataTransform(False, args.max_key),
            input_key=args.input_key,
            target_key=args.target_key,
            forward=False,
            num_adjacent=getattr(args, 'num_adjacent', 5)
        )
        
        val_data = SliceData(
            root=args.data_path_val,
            transform=DataTransform(False, args.max_key),
            input_key=args.input_key,
            target_key=args.target_key,
            forward=False,
            num_adjacent=getattr(args, 'num_adjacent', 5)
        )
        
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            dataset=val_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # For display loader (used in your training code)
        display_loader = DataLoader(
            dataset=val_data,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )
        
        return train_loader, val_loader, display_loader