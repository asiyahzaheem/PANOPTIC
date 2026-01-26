from __future__ import annotations
from pathlib import Path
import pandas as pd

def build_labels_csv(tcia_nifti_dir: Path, nih_data_dir: Path, out_csv: Path) -> None:
    rows = []

    for p in sorted(tcia_nifti_dir.glob("*.nii*")):
        rows.append({"filepath": str(p), "label": 1, "source": "TCIA"})

    for p in sorted(nih_data_dir.glob("*.nii*")):
        rows.append({"filepath": str(p), "label": 0, "source": "NIH"})

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("[OK] wrote", out_csv)
    print(df["source"].value_counts())
    print(df["label"].value_counts())
