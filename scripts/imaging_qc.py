from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import nibabel as nib
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pdac.src.utils.io import load_config, ensure_dir


def main():
    cfg = load_config("configs/config.yaml")
    artifacts = ensure_dir(Path(cfg["data"]["artifacts_dir"]))

    index_csv = artifacts / cfg["data"]["imaging_index_csv"]
    out_log = artifacts / cfg["data"]["imaging_qc_log_csv"]

    df = pd.read_csv(index_csv)
    bad = []

    print(f"[QC] checking {len(df)} scans (log-only)...")
    for _, r in tqdm(df.iterrows(), total=len(df)):
        fp = str(r["filepath"])
        try:
            img = nib.load(fp)
            _ = img.header.get_data_shape()
            _ = img.dataobj[0, 0, 0]  # catches gzip EOF
        except Exception as e:
            bad.append({
                "patient_id": r.get("patient_id", ""),
                "filepath": fp,
                "source": r.get("source", ""),
                "error": repr(e),
            })

    bad_df = pd.DataFrame(bad)
    bad_df.to_csv(out_log, index=False)

    print(f"[QC] bad scans: {len(bad_df)}")
    print(f"[QC] log saved: {out_log}")


if __name__ == "__main__":
    main()
