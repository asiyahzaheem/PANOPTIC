# scripts/molecular_gnn.py
from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import load_config, ensure_dir

# If your labels ever come as strings, we can map them here.
# But your current molecular_labels.csv uses numeric subtype_id (0..3),
# so we will NOT use this map unless subtype_id is non-numeric.
SUBTYPE_NAME_TO_ID = {
    "Squamous": 0,
    "Pancreatic Progenitor": 1,
    "ADEX": 2,
    "Immunogenic": 3,
}


def _norm_patient_id(s: pd.Series) -> pd.Series:
    """Normalize patient_id for stable merges across CSVs."""
    return s.astype(str).str.strip().str.upper()


def _resolve_labels(df: pd.DataFrame) -> tuple[pd.Series, dict]:
    """
    Returns:
      y: pd.Series of int labels in {0,1,2,3}
      meta: dict with info about mapping applied
    """
    sid = df["subtype_id"]

    # Case 1: numeric labels already (your current setup)
    if pd.api.types.is_numeric_dtype(sid):
        y = sid.astype(int)
        meta = {"label_source": "numeric_subtype_id", "name_map_used": False}
    else:
        # Case 2: labels are strings (fallback)
        y = sid.astype(str).str.strip().map(SUBTYPE_NAME_TO_ID)
        meta = {"label_source": "string_subtype_name", "name_map_used": True}

    # Validate: must be exactly 0..3
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

    emb = pd.read_csv(emb_csv)
    lab = pd.read_csv(lab_csv)

    # ---- basic validation
    if "patient_id" not in emb.columns:
        raise ValueError(f"{emb_csv.name} must contain column: patient_id")
    if "patient_id" not in lab.columns or "subtype_id" not in lab.columns:
        raise ValueError(f"{lab_csv.name} must contain columns: patient_id, subtype_id")

    # ---- normalize IDs (prevents silent merge loss)
    emb["patient_id"] = _norm_patient_id(emb["patient_id"])
    lab["patient_id"] = _norm_patient_id(lab["patient_id"])

    # ---- merge: only patients that have BOTH embedding + label
    df = emb.merge(lab, on="patient_id", how="inner")

    # ---- OPTIONAL: restrict to patients that exist in imaging embeddings
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
                print(f"[ALIGN] Filtered to imaging patients: {before} -> {len(df)}")
            else:
                print("[ALIGN] imaging embeddings has no patient_id column; skipping alignment")
        else:
            print("[ALIGN] imaging embeddings CSV not found; skipping alignment")

    # ---- resolve / validate labels
    df["y"], label_meta = _resolve_labels(df)
    # Helpful debug summary
    counts = df["y"].value_counts().sort_index().to_dict()
    print(f"[LABELS] {label_meta} | counts={counts}")

    # ---- build X matrix from emb_0..emb_D
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if len(emb_cols) == 0:
        raise ValueError("No embedding columns found. Expected emb_0, emb_1, ...")

    # sort emb_i by i
    def _emb_key(col: str) -> int:
        # supports emb_0, emb_1, ...
        try:
            return int(col.split("_")[1])
        except Exception:
            return 10**9

    emb_cols = sorted(emb_cols, key=_emb_key)

    X = torch.tensor(df[emb_cols].values, dtype=torch.float32)  # [N, D]
    y = torch.tensor(df["y"].values, dtype=torch.long)          # [N]
    patient_ids = df["patient_id"].astype(str).tolist()

    # ---- save a clean index CSV too (easy to inspect)
    df_out = df[["patient_id", "subtype_id", "y"]].copy()
    df_out.to_csv(out_index, index=False)

    # ---- save GNN-ready artifact
    torch.save(
        {
            "X_mol": X,                  # node features (molecular)
            "y": y,                      # labels (0..3)
            "patient_id": patient_ids,   # ordering
            "emb_cols": emb_cols,        # feature names
            "label_meta": label_meta,    # how labels were interpreted
        },
        out_pt,
    )

    print(f"[OK] wrote {out_pt}")
    print(f"[OK] wrote {out_index}")
    print(f"[SHAPES] X_mol={tuple(X.shape)} y={tuple(y.shape)}")


if __name__ == "__main__":
    main()
