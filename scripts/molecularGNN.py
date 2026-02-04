"""
Loads molecular embeddings and labels, aligns them, and packages them for GNN training."""

from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pdac.src.utils.io import load_config, ensure_dir

# A mapping of subtype names to numeric IDs
SUBTYPE_NAME_TO_ID = {
    "Squamous": 0,
    "Pancreatic Progenitor": 1,
    "ADEX": 2,
    "Immunogenic": 3,
}

def _norm_patient_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper() #Normalize patient_id for stable merges across CSVs


def _resolve_labels(df: pd.DataFrame) -> tuple[pd.Series, dict]:
    sid = df["subtype_id"]

    # Numeric labels set up
    if pd.api.types.is_numeric_dtype(sid):
        y = sid.astype(int)
        meta = {"label_source": "numeric_subtype_id", "name_map_used": False}
    else:
        # Labels are stirngs
        y = sid.astype(str).str.strip().map(SUBTYPE_NAME_TO_ID)
        meta = {"label_source": "string_subtype_name", "name_map_used": True}

    # Valdiaiton
    bad = y.isna() | (~y.isin([0, 1, 2, 3]))
    if bad.any():
        bad_rows = df.loc[bad, ["patient_id", "subtype_id"]].head(20)
        raise ValueError(
            "Found invalid subtype_id values in molecular_labels.csv. "
            "Expected numeric 0,1,2,3 (or one of the known subtype names). "
            "Examples:\n" + bad_rows.to_string(index=False)
        )

    return y.astype(int), meta


def main() -> None:
    cfg = load_config("configs/config.yaml")
    artifacts = ensure_dir(Path(cfg["data"]["artifacts_dir"]))

    emb_csv = artifacts / cfg["data"]["molecular_embeddings_csv"]
    lab_csv = artifacts / cfg["data"]["molecular_labels_csv"]

    out_pt = artifacts / cfg["data"]["molecular_gnn_pt"]
    out_index = artifacts / cfg["data"]["molecular_gnn_index_csv"]

    emb = pd.read_csv(emb_csv)  # Load mol embeddings
    lab = pd.read_csv(lab_csv) # Load mol labels

    if "patient_id" not in emb.columns:
        raise ValueError(f"{emb_csv.name} must contain column: patient_id")
    if "patient_id" not in lab.columns or "subtype_id" not in lab.columns:
        raise ValueError(f"{lab_csv.name} must contain columns: patient_id, subtype_id")

    # Normalize patient_id
    emb["patient_id"] = _norm_patient_id(emb["patient_id"])
    lab["patient_id"] = _norm_patient_id(lab["patient_id"])

    # Patients that have BOTH embedding + label
    df = emb.merge(lab, on="patient_id", how="inner")

    # Restrict to patients that exist in imaging embeddings
    imaging_key = cfg["data"].get("imaging_embeddings_csv", None)
    if imaging_key:
        img_csv = artifacts / imaging_key
        if img_csv.exists():
            img = pd.read_csv(img_csv)
            if "patient_id" in img.columns:
                img["patient_id"] = _norm_patient_id(img["patient_id"])
                keep = set(img["patient_id"].tolist())
                before = len(df)
                df = df[df["patient_id"].isin(keep)].reset_index(drop=True)
                print(f"Filtered to imaging patients: {before} -> {len(df)}")
            else:
                print("[Imaging embeddings has no patient_id column; skipping alignment")
        else:
            print("[Imaging embeddings CSV not found; skipping alignment")

    # Validate labels
    df["y"], label_meta = _resolve_labels(df)
    counts = df["y"].value_counts().sort_index().to_dict()
    print(f"Labels: {label_meta} | counts={counts}")

    # X matrix from mol embeddings
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if len(emb_cols) == 0:
        raise ValueError("No embedding columns found. Expected emb_0, emb_1, ...")

    def _emb_key(col: str) -> int:
        try:
            return int(col.split("_")[1])
        except Exception:
            return 10**9

    emb_cols = sorted(emb_cols, key=_emb_key)

    X = torch.tensor(df[emb_cols].values, dtype=torch.float32) 
    y = torch.tensor(df["y"].values, dtype=torch.long)          
    patient_ids = df["patient_id"].astype(str).tolist()

    df_out = df[["patient_id", "subtype_id", "y"]].copy()
    df_out.to_csv(out_index, index=False)

    # GNN-ready
    torch.save(
        {
            "X_mol": X,                  
            "y": y,                      
            "patient_id": patient_ids,   
            "emb_cols": emb_cols,        
            "label_meta": label_meta,    
        },
        out_pt,
    )

    print(f"Wrote {out_pt}!")
    print(f"Wrote {out_index}!")
    print(f"SHAPES: X_mol={tuple(X.shape)} y={tuple(y.shape)}")


if __name__ == "__main__":
    main()
