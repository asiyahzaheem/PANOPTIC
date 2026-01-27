from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib

from src.data.preprocessing import (
    clip_and_normalize_hu,
    to_slices_first,
    pick_k_slices_center,
)

class ScanKSliceDataset(Dataset):
    """
    df columns required:
      - patient_id
      - filepath
      - source (optional)
    returns:
      xk: [K,3,H,W]
      patient_id: str
      source: str
    """
    def __init__(self, df, cfg, transform):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transform = transform

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

        img = nib.load(fp)
        vol = img.get_fdata().astype(np.float32)     # [H,W,Z] or [X,Y,Z]
        vol = to_slices_first(vol)                   # [Z,H,W]
        vol = clip_and_normalize_hu(vol, self.hu_min, self.hu_max)

        slices = pick_k_slices_center(vol, self.k, self.pool)  # [K,H,W]

        x_list = []
        for s in slices:
            s_u8 = (s * 255.0).clip(0, 255).astype(np.uint8)
            xt = self.transform(s_u8)  # must output [3,224,224]
            x_list.append(xt)

        xk = torch.stack(x_list, dim=0)  # [K,3,224,224]
        return xk, pid, source
