from __future__ import annotations
from pathlib import Path
import sys
import SimpleITK as sitk
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import load_config, ensure_dir


def find_dicom_series_dirs(patient_dir: Path) -> list[Path]:
    """
    CPTAC can be nested. A 'series dir' here is any folder containing .dcm files.
    """
    dirs = []
    for p in patient_dir.rglob("*"):
        if p.is_dir():
            try:
                if any(x.is_file() and x.suffix.lower() == ".dcm" for x in p.iterdir()):
                    dirs.append(p)
            except Exception:
                continue
    return dirs


def try_convert_series_dir(series_dir: Path, out_path: Path) -> bool:
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(str(series_dir))
    if not ids:
        return False
    files = reader.GetGDCMSeriesFileNames(str(series_dir), ids[0])
    if not files:
        return False
    reader.SetFileNames(files)
    img = reader.Execute()
    sitk.WriteImage(img, str(out_path))
    return True


def main():
    cfg = load_config("configs/config.yaml")
    dicom_root = Path(cfg["data"]["cptac_dicom_dir"])
    nifti_root = ensure_dir(Path(cfg["data"]["cptac_nifti_dir"]))

    patients = sorted([p for p in dicom_root.iterdir() if p.is_dir()])
    print(f"[CPTAC] patients found: {len(patients)}")

    failed = 0
    for pdir in tqdm(patients, desc="DICOM→NIfTI"):
        pid = pdir.name
        out = nifti_root / f"{pid}.nii.gz"

        if out.exists() and out.stat().st_size > 0:
            continue

        series_dirs = find_dicom_series_dirs(pdir)
        ok = False
        for sdir in series_dirs:
            try:
                ok = try_convert_series_dir(sdir, out)
                if ok:
                    break
            except Exception:
                continue

        if not ok:
            failed += 1
            print(f"[WARN] No valid series for {pid} (skipping)")

    print(f"[OK] Conversion done. failed={failed}. output={nifti_root}")


if __name__ == "__main__":
    main()
