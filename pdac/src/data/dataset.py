"""
PyTorch dataset for loading CT scans and extracting K slices per scan
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib

from pdac.src.data.preprocessing import (
    clip_and_normalize_hu,
    to_slices_first,
    pick_k_slices_center,
)

# Dataset that loads CT scans and returns K representative slices per patient
class ScanKSliceDataset(Dataset):
    def __init__(self, df, cfg, transform):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transform = transform

        # read preprocessing settings from config
        self.hu_min = float(cfg["preprocess"]["hu_min"])
        self.hu_max = float(cfg["preprocess"]["hu_max"])
        self.k = int(cfg["preprocess"]["k_slices"])
        self.pool = int(cfg["preprocess"]["slice_pool"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(row["patient_id"])
        fp = str(row["filepath"])
        source = str(row.get("source", "UNKNOWN"))

        # load the NIfTI scan and get the 3D volume
        img = nib.load(fp)
        vol = img.get_fdata().astype(np.float32)
        # rearrange so slices are the first dimension
        vol = to_slices_first(vol)
        # window the HU values and normalize to [0,1]
        vol = clip_and_normalize_hu(vol, self.hu_min, self.hu_max)

        # pick K evenly-spaced slices from the center region
        slices = pick_k_slices_center(vol, self.k, self.pool)

        # apply transforms to each slice
        x_list = []
        for s in slices:
            s_u8 = (s * 255.0).clip(0, 255).astype(np.uint8)
            xt = self.transform(s_u8)
            x_list.append(xt)

        # stack into [K,3,H,W] tensor
        xk = torch.stack(x_list, dim=0)
        return xk, pid, source

