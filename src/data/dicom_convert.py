from __future__ import annotations
from pathlib import Path
import os
import SimpleITK as sitk

def _find_dicom_dirs(root: Path) -> list[Path]:
    dicom_dirs = []
    for dirpath, _, filenames in os.walk(root):
        if any(f.lower().endswith(".dcm") for f in filenames):
            dicom_dirs.append(Path(dirpath))
    return dicom_dirs

def convert_tcia_dicom_to_nifti(tcia_raw_dir: Path, tcia_nifti_dir: Path) -> None:
    tcia_nifti_dir.mkdir(parents=True, exist_ok=True)

    patients = sorted([p for p in tcia_raw_dir.iterdir() if p.is_dir()])
    for patient_dir in patients:
        patient_id = patient_dir.name
        out_path = tcia_nifti_dir / f"{patient_id}.nii.gz"
        if out_path.exists():
            print(f"[SKIP] exists: {out_path}")
            continue

        dicom_series_dirs = _find_dicom_dirs(patient_dir)
        if not dicom_series_dirs:
            print(f"[WARN] no DICOM found under {patient_dir}")
            continue

        # Choose the directory with the most .dcm files (usually the real CT series)
        def dcm_count(d: Path) -> int:
            try:
                return sum(1 for f in d.iterdir() if f.is_file() and f.name.lower().endswith(".dcm"))
            except Exception:
                return 0

        series_dir = max(dicom_series_dirs, key=dcm_count)

        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(series_dir))
        if not series_ids:
            print(f"[WARN] no series IDs in {series_dir}")
            continue

        series_id = series_ids[0]
        file_names = reader.GetGDCMSeriesFileNames(str(series_dir), series_id)
        reader.SetFileNames(file_names)

        print(f"[INFO] {patient_id}: reading {len(file_names)} slices from {series_dir}")
        image_3d = reader.Execute()
        sitk.WriteImage(image_3d, str(out_path))
        print(f"[OK] saved {out_path}")
