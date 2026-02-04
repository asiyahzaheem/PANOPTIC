"""
Extracts feature column names from the molecular index and saves them to a text file
"""

from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pdac.src.utils.io import load_config, ensure_dir


def main():
    cfg = load_config("configs/config.yaml")
    artifacts = ensure_dir(Path(cfg["data"]["artifacts_dir"]))

    index_csv = artifacts / cfg["data"]["molecular_index_csv"]
    out_txt = artifacts / cfg["data"]["molecular_feature_cols_txt"]

    df = pd.read_csv(index_csv) # load mol data index
    if "patient_id" not in df.columns:
        raise ValueError("molecular_index.csv must contain patient_id")

    feature_cols = [c for c in df.columns if c != "patient_id"] # all cols features except patient_id
    if not feature_cols:
        raise ValueError("No feature columns found in molecular_index.csv")

    out_txt.write_text("\n".join(feature_cols) + "\n")
    print(f"Wrote feature columns ({len(feature_cols)}) -> {out_txt}!")

if __name__ == "__main__":
    main()
