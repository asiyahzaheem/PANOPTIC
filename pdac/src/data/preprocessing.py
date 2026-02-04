"""
Utility functions for CT scan preprocessing (HU windowing, slice selection, etc.)
"""

from __future__ import annotations
import numpy as np

# Window CT values to a specific HU range and normalize to [0,1]
def clip_and_normalize_hu(vol: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    vol = np.clip(vol, hu_min, hu_max)
    vol = (vol - hu_min) / (hu_max - hu_min + 1e-6)
    return vol.astype(np.float32)

# Rearrange 3D volume so slices are the first axis
def to_slices_first(vol: np.ndarray) -> np.ndarray:
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {vol.shape}")
    return np.moveaxis(vol, -1, 0)

# Select K evenly-spaced slices from the center region of the scan
def pick_k_slices_center(vol_z_hw: np.ndarray, k: int, pool: int) -> np.ndarray:
    z = vol_z_hw.shape[0]
    if z <= 0:
        raise ValueError("Empty volume")

    # define the center pool of slices to sample from
    pool = min(pool, z)
    start = max(0, (z - pool) // 2)
    end = start + pool
    cand = np.arange(start, end)

    if len(cand) < k:
        idx = np.resize(cand, k)
    else:
        # pick K evenly-spaced slices from the pool
        idx = np.linspace(0, len(cand) - 1, k).round().astype(int)
        idx = cand[idx]

    return vol_z_hw[idx]

