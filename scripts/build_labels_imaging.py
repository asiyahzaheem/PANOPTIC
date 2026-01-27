from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nifti_dir", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--label", type=int, default=1, help="For CPTAC PDAC cohort, label=1 by default")
    ap.add_argument("--source", type=str, default="CPTAC")
    args = ap.parse_args()

    nifti_dir = Path(args.nifti_dir)
    paths = sorted(nifti_dir.glob("*.nii.gz"))

    rows = []
    for p in paths:
        pid = p.stem.replace(".nii", "")  # handles .nii.gz
        rows.append({"patient_id": pid, "filepath": str(p), "label": args.label, "source": args.source})

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote {args.out_csv} with {len(df)} rows")


if __name__ == "__main__":
    main()
