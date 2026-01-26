from __future__ import annotations
from pathlib import Path
import numpy as np
import nibabel as nib

def load_volume_slices_first(path: str | Path) -> np.ndarray:
    img = nib.load(str(path))
    vol = img.get_fdata().astype(np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D, got {vol.shape} for {path}")
    # Convert (H,W,S) -> (S,H,W)
    vol = np.transpose(vol, (2, 0, 1))
    return vol

def window_hu(vol: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    return np.clip(vol, hu_min, hu_max)

def normalize_scan_z(vol: np.ndarray) -> np.ndarray:
    m = float(np.mean(vol))
    s = float(np.std(vol)) + 1e-6
    return (vol - m) / s

def pick_central_slice_pool(vol: np.ndarray, pool_size: int) -> np.ndarray:
    S = vol.shape[0]
    if S <= pool_size:
        return vol
    start = (S - pool_size) // 2
    return vol[start:start + pool_size]

def to_uint8_01(slice_2d: np.ndarray) -> np.ndarray:
    mn, mx = float(slice_2d.min()), float(slice_2d.max())
    x = (slice_2d - mn) / (mx - mn + 1e-6)
    x = (x * 255.0).astype(np.uint8)
    return x

def center_crop_3d(vol: np.ndarray, frac: float = 0.6) -> np.ndarray:
    """
    vol: (S, H, W)
    frac: keep this fraction of H and W (e.g., 0.6 keeps central 60%)
    """
    assert vol.ndim == 3
    S, H, W = vol.shape
    new_h = max(1, int(H * frac))
    new_w = max(1, int(W * frac))

    y0 = (H - new_h) // 2
    x0 = (W - new_w) // 2
    return vol[:, y0:y0 + new_h, x0:x0 + new_w]
