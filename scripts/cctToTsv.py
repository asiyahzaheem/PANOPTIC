"""
Converts a CPTAC .cct RNA-seq file into TSV files per patienrt
"""

from __future__ import annotations
from pathlib import Path
import sys
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pdac.src.utils.io import load_config, ensure_dir

def _safe_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional limit for debugging (e.g., 3). If not set, writes all samples.",
    )
    args = ap.parse_args()

    cfg = load_config("configs/config.yaml")

    cct_path = Path(cfg["data"]["rnaseq_cct_dir"])
    out_dir = ensure_dir(Path(cfg["data"]["rnaseq_per_patient_tsv_dir"]))

    if not cct_path.exists():
        raise FileNotFoundError(f"CCT file not found: {cct_path}")

    # Structure: first column = gene identifiers, remaining columns = samples
    df = pd.read_csv(cct_path, sep="\t")

    if df.shape[1] < 2:
        raise ValueError("CCT appears to have <2 columns. Check file format or delimiter.")

    feature_col = df.columns[0]  # first column = gene identifiers
    genes = df[feature_col].astype(str)

    sample_cols = list(df.columns[1:])
    if args.max_samples is not None:
        sample_cols = sample_cols[: args.max_samples]

    print(f"[CCT] file: {cct_path}")
    print(f"[CCT] genes={len(genes)} | samples={len(sample_cols)}")
    print(f"[CCT] example samples: {sample_cols[:5]}")

    wrote = 0
    for sid in sample_cols:
        out_df = pd.DataFrame({"gene": genes, "value": df[sid].values})
        out_path = out_dir / f"{_safe_filename(str(sid))}.tsv" # Fix filename
        out_df.to_csv(out_path, sep="\t", index=False)
        wrote += 1

    print(f"Wrote {wrote} per-patient TSV files -> {out_dir}!")


if __name__ == "__main__":
    main()
