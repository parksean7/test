import h5py
import random
from utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []
        self.num_adjacent_slices = 5 # set default num_adjacent to 5 in initialization
        
        if not forward:
            image_files = list(Path(root / "image").iterdir())
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname)

                self.image_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]

        kspace_files = list(Path(root / "kspace").iterdir())
        for fname in sorted(kspace_files):
            num_slices = self._get_metadata(fname)

            self.kspace_examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]


    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
            else:
                # Fallback: use any available key to get number of slices
                available_keys = list(hf.keys())
                if available_keys:
                    num_slices = hf[available_keys[0]].shape[0]
                else:
                    num_slices = 0
        return num_slices

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):

        kspace_fname, dataslice = self.kspace_examples[i]
        mask, kspace, target, maximum, fname_out, slice_out = self._load_kspace_slice(kspace_fname, dataslice)

        if not self.forward:
            image_fname, _ = self.image_examples[i]
            if image_fname.name != kspace_fname.name:
                raise ValueError(f"Image file {image_fname.name} does not match kspace file {kspace_fname.name}")
            # Note: target and attrs are already processed in _load_kspace_slice
            # target, attrs = self._load_image_slice(image_fname, dataslice, attrs)

        return mask, kspace, target, maximum, fname_out, slice_out


    def _load_kspace_slice(self, kspace_fname, dataslice):

        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice : dataslice + self.num_adjacent_slices]
            # Mask is 1D along readout direction, same for all slices
            mask = hf["mask"][:]  # Load full 1D mask

        if self.forward:   # redundant condition
            target = -1
            attrs = -1
        else:
            with h5py.File(kspace_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
            
        return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice)



def create_data_loaders(data_path, args, shuffle=False, isforward=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(isforward, max_key_),
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward,
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    return data_loader
