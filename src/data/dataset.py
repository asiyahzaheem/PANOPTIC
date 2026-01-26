# src/data/dataset.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import (
    load_volume_slices_first,
    window_hu,
    normalize_scan_z,
    pick_central_slice_pool,
    to_uint8_01,
    center_crop_3d
)

class ScanDatasetKSlice(Dataset):
    """
    Outputs:
      xk: FloatTensor [K, 3, H, W]  (always 3-channel for ResNet)
      y : LongTensor []
    Robustness:
      - If a scan fails to load/decompress, it is logged and another sample is used.
      - Nothing is deleted. labels.csv is NOT edited.
    """
    def __init__(self, df, cfg: dict, transform=None, max_tries: int = 5):
        self.cfg = cfg
        self.transform = transform
        self.max_tries = max_tries
        self.df = df.reset_index(drop=True)


        self.filepaths = [Path(p) for p in df["filepath"].astype(str).tolist()]
        self.labels = df["label"].astype(int).tolist()

        Path("artifacts").mkdir(parents=True, exist_ok=True)
        self.bad_log_path = Path("artifacts/runtime_bad_scans.txt")

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _ensure_3ch(t: torch.Tensor) -> torch.Tensor:
        """
        Accepts [H,W] or [1,H,W] or [3,H,W] and returns [3,H,W].
        """
        if t.ndim == 2:
            t = t.unsqueeze(0)  # [1,H,W]
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        return t

    def _build_xk(self, path: Path) -> torch.Tensor:
        pp = self.cfg["preprocess"]
        k = int(pp["k_slices"])
        slice_pool = int(pp["slice_pool"])

        # 1) load volume (S,H,W)
        vol = load_volume_slices_first(path)

        # 2) HU window + per-scan normalization
        vol = window_hu(vol, float(pp["hu_min"]), float(pp["hu_max"]))
        vol = normalize_scan_z(vol)

        # Anatomy focused
        crop_frac = float(self.cfg["preprocess"].get("crop_frac", 0.6))
        vol = center_crop_3d(vol, frac=crop_frac)

        # 3) central pool -> pick K
        pool = pick_central_slice_pool(vol, slice_pool)  # (P,H,W)
        if pool.shape[0] >= k:
            start = (pool.shape[0] - k) // 2
            slices = pool[start : start + k]
        else:
            slices = pool

        xs = []
        for sl in slices:
            sl8 = to_uint8_01(sl.astype(np.float32))  # uint8 HxW in [0,255]

            if self.transform is not None:
                out = self.transform(sl8)  # could be [H,W] or [1,H,W] or [3,H,W]
                if not torch.is_tensor(out):
                    out = torch.as_tensor(out)
                out = out.float()
                out = self._ensure_3ch(out)
            else:
                # fallback: [1,H,W] in [0,1] then repeat to 3ch
                out = torch.from_numpy(sl8).float().unsqueeze(0) / 255.0
                out = out.repeat(3, 1, 1)

            xs.append(out)

        xk = torch.stack(xs, dim=0)  # [K,3,H,W]
        return xk

    def __getitem__(self, idx):
        tries = 0
        last_err = None

        while tries < self.max_tries:
            path = self.filepaths[idx]
            y = int(self.labels[idx])

            try:
                xk = self._build_xk(path)
                #return xk, torch.tensor(y, dtype=torch.long)
                source = str(self.df.iloc[idx]["source"])
                return xk, y, source


            except Exception as e:
                last_err = e
                with self.bad_log_path.open("a") as f:
                    f.write(f"{path},{repr(e)}\n")

                idx = np.random.randint(0, len(self.filepaths))
                tries += 1

        raise RuntimeError(f"Too many corrupted scans encountered. Last error: {repr(last_err)}")
