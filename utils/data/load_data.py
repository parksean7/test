import h5py
import random
from utils.data.transforms import DataTransform
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
        self.image_examples = []
        self.kspace_examples = []
        
        root_path = Path(root)
        print(f"Loading data from: {root_path}")
        
        # Load image files (for targets) if not in forward mode
        if not forward:
            image_dir = root_path / "image"
            if image_dir.exists():
                image_files = list(image_dir.glob("*.h5"))
                print(f"Found {len(image_files)} image files")
                for fname in sorted(image_files):
                    try:
                        num_slices = self._get_metadata(fname)
                        if num_slices > 0:
                            self.image_examples += [
                                (fname, slice_ind) for slice_ind in range(num_slices)
                            ]
                    except Exception as e:
                        print(f"Error reading image file {fname}: {e}")
            else:
                print(f"Warning: Image directory {image_dir} does not exist")

        # Load kspace files
        kspace_dir = root_path / "kspace"
        if kspace_dir.exists():
            kspace_files = list(kspace_dir.glob("*.h5"))
            print(f"Found {len(kspace_files)} kspace files")
            for fname in sorted(kspace_files):
                try:
                    num_slices = self._get_metadata(fname)
                    if num_slices > 0:
                        self.kspace_examples += [
                            (fname, slice_ind) for slice_ind in range(num_slices)
                        ]
                except Exception as e:
                    print(f"Error reading kspace file {fname}: {e}")
        else:
            print(f"Error: Kspace directory {kspace_dir} does not exist")
            
        print(f"Total image examples: {len(self.image_examples)}")
        print(f"Total kspace examples: {len(self.kspace_examples)}")
        
        if len(self.kspace_examples) == 0:
            raise ValueError(f"No valid kspace files found in {kspace_dir}")

    def _get_metadata(self, fname):
        try:
            with h5py.File(fname, "r") as hf:
                if self.input_key in hf.keys():
                    num_slices = hf[self.input_key].shape[0]
                elif self.target_key in hf.keys():
                    num_slices = hf[self.target_key].shape[0]
                else:
                    # Debug: print available keys
                    print(f"Available keys in {fname.name}: {list(hf.keys())}")
                    num_slices = 0
                return num_slices
        except Exception as e:
            print(f"Error reading metadata from {fname}: {e}")
            return 0

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        kspace_fname, dataslice = self.kspace_examples[i]
        
        # For training mode, find matching image file
        if not self.forward:
            # Find matching image file by name
            kspace_basename = kspace_fname.stem  # e.g., "brain_acc4_1"
            image_fname = None
            
            for img_fname, img_slice in self.image_examples:
                if img_fname.stem == kspace_basename and img_slice == dataslice:
                    image_fname = img_fname
                    break
            
            if image_fname is None:
                # If exact match not found, try to find by basename only
                for img_fname, img_slice in self.image_examples:
                    if img_fname.stem == kspace_basename:
                        image_fname = img_fname
                        break
                        
            if image_fname is None:
                raise ValueError(f"No matching image file found for kspace file {kspace_fname.name}")

        # Load kspace data
        try:
            with h5py.File(kspace_fname, "r") as hf:
                input_data = hf[self.input_key][dataslice]
                mask = np.array(hf["mask"])
        except Exception as e:
            print(f"Error loading kspace data from {kspace_fname}: {e}")
            raise

        # Load target data
        if self.forward:
            target = -1
            attrs = -1
        else:
            try:
                with h5py.File(image_fname, "r") as hf:
                    target = hf[self.target_key][dataslice]
                    attrs = dict(hf.attrs)
            except Exception as e:
                print(f"Error loading image data from {image_fname}: {e}")
                # Create dummy target to prevent crash
                target = np.zeros(input_data.shape[-2:], dtype=np.float32)
                attrs = {}
            
        return self.transform(mask, input_data, target, attrs, kspace_fname.name, dataslice)


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