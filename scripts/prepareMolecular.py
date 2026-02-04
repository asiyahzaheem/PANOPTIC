"""
Converts the CCT molecular data file into a patient-by-gene CSV for downstream processing
"""

from pathlib import Path
import pandas as pd
from src.utils.io import load_config, resolve_path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    cfg = load_config("configs/config.yaml")

    artifacts = Path(cfg["data"]["artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)

    cct_path = Path(cfg["data"]["rnaseq_cct_dir"])
    out_csv = artifacts / cfg["data"]["molecular_index_csv"]

    # load the CCT table (genes as rows, patients as columns)
    df = pd.read_csv(cct_path, sep="\t", index_col=0)

    # flip it so patients are rows and genes are columns
    df = df.T.reset_index().rename(columns={"index": "patient_id"})

    df.to_csv(out_csv, index=False)
    print(f"Molecular index saved → {out_csv} ({df.shape[0]} patients)!")

if __name__ == "__main__":
    main()
