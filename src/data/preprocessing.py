from __future__ import annotations
import numpy as np

def clip_and_normalize_hu(vol: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    vol = np.clip(vol, hu_min, hu_max)
    vol = (vol - hu_min) / (hu_max - hu_min + 1e-6)
    return vol.astype(np.float32)

def to_slices_first(vol: np.ndarray) -> np.ndarray:
    # Expect 3D, assume last axis is slices, move to [Z,H,W]
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {vol.shape}")
    return np.moveaxis(vol, -1, 0)

def pick_k_slices_center(vol_z_hw: np.ndarray, k: int, pool: int) -> np.ndarray:
    z = vol_z_hw.shape[0]
    if z <= 0:
        raise ValueError("Empty volume")

    pool = min(pool, z)
    start = max(0, (z - pool) // 2)
    end = start + pool
    cand = np.arange(start, end)

    if len(cand) < k:
        idx = np.resize(cand, k)
    else:
        # evenly spread inside pool
        idx = np.linspace(0, len(cand) - 1, k).round().astype(int)
        idx = cand[idx]

    return vol_z_hw[idx]  # [K,H,W]
