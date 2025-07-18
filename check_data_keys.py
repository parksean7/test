#!/usr/bin/env python3
import h5py
import os
from pathlib import Path

def check_hdf5_keys(data_path):
    """Check available keys in HDF5 files"""
    print(f"Checking data in: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"Path does not exist: {data_path}")
        return
    
    # Check image files
    image_path = Path(data_path) / "image"
    if image_path.exists():
        image_files = list(image_path.glob("*.h5"))
        if image_files:
            sample_file = image_files[0]
            print(f"\nChecking image file: {sample_file}")
            try:
                with h5py.File(sample_file, "r") as hf:
                    print("Available keys in image file:")
                    for key in hf.keys():
                        print(f"  - {key}: shape={hf[key].shape}, dtype={hf[key].dtype}")
            except Exception as e:
                print(f"Error reading image file: {e}")
    
    # Check kspace files
    kspace_path = Path(data_path) / "kspace"
    if kspace_path.exists():
        kspace_files = list(kspace_path.glob("*.h5"))
        if kspace_files:
            sample_file = kspace_files[0]
            print(f"\nChecking kspace file: {sample_file}")
            try:
                with h5py.File(sample_file, "r") as hf:
                    print("Available keys in kspace file:")
                    for key in hf.keys():
                        print(f"  - {key}: shape={hf[key].shape}, dtype={hf[key].dtype}")
            except Exception as e:
                print(f"Error reading kspace file: {e}")

if __name__ == "__main__":
    # Check server data paths
    server_train_path = "/root/Data/train/"
    server_val_path = "/root/Data/val/"
    
    print("=== SERVER DATA STRUCTURE ===")
    check_hdf5_keys(server_train_path)
    print("\n" + "="*50)
    check_hdf5_keys(server_val_path)