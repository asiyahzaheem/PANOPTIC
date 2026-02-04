"""
Utilities for converting DICOM series to NIfTI format
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import csv
import time

import SimpleITK as sitk

# Tracks the result of a DICOM to NIfTI conversion
@dataclass
class ConvertResult:
    patient_id: str
    ok: bool
    out_path: str
    chosen_series_uid: str
    n_slices: int
    error: str

# Find all DICOM series IDs in a folder using SimpleITK's GDCM indexer
def _find_series_in_folder(folder: Path) -> List[str]:
    try:
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder))
        return list(series_ids) if series_ids else []
    except Exception:
        return []

# Pick the series with the most slices (works well for CT volumes)
def _choose_best_series(folder: Path, series_ids: List[str]) -> Tuple[Optional[str], int]:
    best_uid = None
    best_n = -1
    for uid in series_ids:
        try:
            files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), uid)
            n = len(files)
            if n > best_n:
                best_n = n
                best_uid = uid
        except Exception:
            continue
    return best_uid, best_n

# Convert the best DICOM series found in patient_root to a NIfTI file
def convert_patient_dicom_to_nifti(
    patient_id: str,
    patient_root: Path,
    out_dir: Path,
    *,
    overwrite: bool = False,
) -> ConvertResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{patient_id}.nii.gz"

    # skip if output already exists
    if out_path.exists() and not overwrite:
        return ConvertResult(patient_id, True, str(out_path), "SKIP_EXISTS", -1, "")

    t0 = time.time()
    try:
        # find all DICOM series in the folder
        series_ids = _find_series_in_folder(patient_root)
        if not series_ids:
            return ConvertResult(patient_id, False, str(out_path), "", 0, "No DICOM series IDs found")

        # pick the series with the most slices
        best_uid, best_n = _choose_best_series(patient_root, series_ids)
        if not best_uid or best_n <= 0:
            return ConvertResult(patient_id, False, str(out_path), "", 0, "Could not choose a valid series")

        # load the DICOM series
        files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(patient_root), best_uid)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)

        img = reader.Execute()

        # standardize orientation
        img = sitk.DICOMOrient(img, "LPS")

        # write NIfTI file
        sitk.WriteImage(img, str(out_path), True)

        dt = time.time() - t0
        return ConvertResult(patient_id, True, str(out_path), best_uid, len(files), f"OK ({dt:.1f}s)")

    except Exception as e:
        return ConvertResult(patient_id, False, str(out_path), "", 0, repr(e))

# Write conversion results to a CSV log file
def write_log_csv(log_path: Path, rows: List[ConvertResult]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "ok", "out_path", "chosen_series_uid", "n_slices", "error"])
        for r in rows:
            w.writerow([r.patient_id, int(r.ok), r.out_path, r.chosen_series_uid, r.n_slices, r.error])

