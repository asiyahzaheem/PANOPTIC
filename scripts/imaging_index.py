from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import load_config, ensure_dir


def main():
    cfg = load_config("configs/config.yaml")

    nifti_dir = Path(cfg["data"]["cptac_nifti_dir"])
    artifacts = ensure_dir(Path(cfg["data"]["artifacts_dir"]))
    out_csv = artifacts / cfg["data"]["imaging_index_csv"]

    rows = []
    for f in sorted(nifti_dir.glob("*.nii.gz")):
        if f.name.startswith("._"):
            continue
        pid = f.name.replace(".nii.gz", "")
        rows.append({"patient_id": pid, "filepath": str(f), "source": "CPTAC"})

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}  (n={len(df)})")


if __name__ == "__main__":
    main()
