from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import load_config, ensure_dir


SUBTYPE_MAP = {
    "Squamous": 0,
    "Pancreatic Progenitor": 1,
    "ADEX": 2,
    "Immunogenic": 3,
}


def main():
    cfg = load_config("configs/config.yaml")
    artifacts = ensure_dir(Path(cfg["data"]["artifacts_dir"]))

    emb_csv = artifacts / cfg["data"]["molecular_embeddings_csv"]
    lab_csv = artifacts / cfg["data"]["molecular_labels_csv"]

    out_pt = artifacts / cfg["data"]["molecular_gnn_pt"]
    out_index = artifacts / cfg["data"]["molecular_gnn_index_csv"]

    emb = pd.read_csv(emb_csv)
    lab = pd.read_csv(lab_csv)

    # basic validation
    if "patient_id" not in emb.columns:
        raise ValueError("molecular_embeddings.csv must contain column: patient_id")
    if "patient_id" not in lab.columns or "subtype" not in lab.columns:
        raise ValueError("molecular_labels.csv must contain columns: patient_id, subtype")

    # merge -> only patients that have BOTH embedding + label
    df = emb.merge(lab, on="patient_id", how="inner")

    # OPTIONAL: restrict to patients that exist in imaging embeddings (recommended for multimodal)
    imaging_key = cfg["data"].get("imaging_embeddings_csv", None)
    if imaging_key:
        img_csv = artifacts / imaging_key
        if img_csv.exists():
            img = pd.read_csv(img_csv)
            if "patient_id" in img.columns:
                keep = set(img["patient_id"].astype(str).tolist())
                before = len(df)
                df = df[df["patient_id"].astype(str).isin(keep)].reset_index(drop=True)
                print(f"[ALIGN] Filtered to imaging patients: {before} -> {len(df)}")
            else:
                print("[ALIGN] imaging_embeddings.csv has no patient_id column; skipping alignment")
        else:
            print("[ALIGN] imaging_embeddings.csv not found; skipping alignment")

    # map subtype -> int
    df["y"] = df["subtype"].map(SUBTYPE_MAP)
    bad = df["y"].isna()
    if bad.any():
        bad_rows = df[bad][["patient_id", "subtype"]].head(10)
        raise ValueError(
            "Found unknown subtype values in molecular_labels.csv. "
            "Examples:\n" + bad_rows.to_string(index=False)
        )

    # build X matrix from emb_0..emb_255
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    emb_cols = sorted(emb_cols, key=lambda x: int(x.split("_")[1]))
    if len(emb_cols) == 0:
        raise ValueError("No embedding columns found. Expected emb_0, emb_1, ...")

    X = torch.tensor(df[emb_cols].values, dtype=torch.float32)  # [N, D]
    y = torch.tensor(df["y"].values, dtype=torch.long)          # [N]
    patient_ids = df["patient_id"].astype(str).tolist()

    # save a clean index CSV too (easy to inspect)
    df_out = df[["patient_id", "subtype", "y"]].copy()
    df_out.to_csv(out_index, index=False)

    # save GNN-ready artifact
    torch.save(
        {
            "X_mol": X,                 # node features (molecular)
            "y": y,                     # labels (0..3)
            "patient_id": patient_ids,  # ordering
            "subtype_map": SUBTYPE_MAP,
        },
        out_pt,
    )

    print(f"[OK] wrote {out_pt}")
    print(f"[OK] wrote {out_index}")
    print(f"[SHAPES] X_mol={tuple(X.shape)} y={tuple(y.shape)}")


if __name__ == "__main__":
    main()
