# api/services/mol_parser.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def read_molecular_tsv(tsv_path: Path) -> np.ndarray:
    df = pd.read_csv(tsv_path, sep="\t")
    emb_cols = [c for c in df.columns if c.startswith("emb_") and c.split("_")[1].isdigit()]
    if not emb_cols:
        raise ValueError("Molecular TSV must contain emb_0..emb_255 columns (single-row embedding).")
    emb_cols = sorted(emb_cols, key=lambda c: int(c.split("_")[1]))

    if len(df) < 1:
        raise ValueError("Molecular TSV is empty.")
    return df.loc[0, emb_cols].to_numpy(dtype=np.float32)
