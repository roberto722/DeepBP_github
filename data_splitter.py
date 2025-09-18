#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split a dataset of .hdf5 files into train/val/test folders.

- Input: a folder containing .hdf5 files
- Output: three folders: train/, val/, test/
- The split ratios can be customized
"""

import os
import shutil
import random
from pathlib import Path

def split_dataset(
    input_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    # Ensure ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all .hdf5 files
    files = sorted([f for f in input_dir.glob("*.hdf5")])
    if len(files) == 0:
        raise RuntimeError(f"No .hdf5 files found in {input_dir}")

    # Shuffle files with fixed seed
    random.seed(seed)
    random.shuffle(files)

    # Compute split indices
    n_total = len(files)
    n_train = int(train_ratio * n_total)
    n_val   = int(val_ratio * n_total)
    # Remaining files go to test
    n_test  = n_total - n_train - n_val

    splits = {
        "train": files[:n_train],
        "val":   files[n_train:n_train+n_val],
        "test":  files[n_train+n_val:]
    }

    # Create split folders and copy files
    for split, split_files in splits.items():
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        print(f"Copying {len(split_files)} files to {split_dir}")
        for f in split_files:
            shutil.move(str(f), split_dir / f.name)

    print("Dataset split completed.")
    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")

if __name__ == "__main__":
    # Example usage
    INPUT_DIR  = "./data/Forearm2000_hdf5"
    OUTPUT_DIR = ",/data/Forearm2000_hdf5/splitted_dataset"

    split_dataset(INPUT_DIR, OUTPUT_DIR, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
