from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import csv
import time

import SimpleITK as sitk


@dataclass
class ConvertResult:
    patient_id: str
    ok: bool
    out_path: str
    chosen_series_uid: str
    n_slices: int
    error: str


def _find_series_in_folder(folder: Path) -> List[str]:
    """
    Returns a list of DICOM SeriesInstanceUIDs found under `folder`.
    Uses SimpleITK's GDCM indexer (fast; no need to manually parse DICOM tags).
    """
    try:
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder))
        return list(series_ids) if series_ids else []
    except Exception:
        return []


def _choose_best_series(folder: Path, series_ids: List[str]) -> Tuple[Optional[str], int]:
    """
    Heuristic: choose series with the most files (most slices).
    (Works well for CT volumes when multiple series exist.)
    Returns (series_uid, n_files).
    """
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


def convert_patient_dicom_to_nifti(
    patient_id: str,
    patient_root: Path,
    out_dir: Path,
    *,
    overwrite: bool = False,
) -> ConvertResult:
    """
    Converts the *best* DICOM series under patient_root to NIfTI:
      out_dir/{patient_id}.nii.gz

    If multiple nested subfolders exist, we run the DICOM series search
    starting at patient_root (SimpleITK scans recursively for series IDs).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{patient_id}.nii.gz"

    if out_path.exists() and not overwrite:
        return ConvertResult(patient_id, True, str(out_path), "SKIP_EXISTS", -1, "")

    t0 = time.time()
    try:
        series_ids = _find_series_in_folder(patient_root)
        if not series_ids:
            return ConvertResult(patient_id, False, str(out_path), "", 0, "No DICOM series IDs found")

        best_uid, best_n = _choose_best_series(patient_root, series_ids)
        if not best_uid or best_n <= 0:
            return ConvertResult(patient_id, False, str(out_path), "", 0, "Could not choose a valid series")

        files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(patient_root), best_uid)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)

        img = reader.Execute()

        # Optional safety: enforce canonical orientation-ish (doesn't fully standardize, but helps)
        img = sitk.DICOMOrient(img, "LPS")

        sitk.WriteImage(img, str(out_path), True)

        dt = time.time() - t0
        return ConvertResult(patient_id, True, str(out_path), best_uid, len(files), f"OK ({dt:.1f}s)")

    except Exception as e:
        return ConvertResult(patient_id, False, str(out_path), "", 0, repr(e))


def write_log_csv(log_path: Path, rows: List[ConvertResult]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "ok", "out_path", "chosen_series_uid", "n_slices", "error"])
        for r in rows:
            w.writerow([r.patient_id, int(r.ok), r.out_path, r.chosen_series_uid, r.n_slices, r.error])
