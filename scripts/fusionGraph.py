# scripts/fusionGraph.py  (or scripts/build_fusion_graph.py)
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import load_config, ensure_dir
from src.gnn.buildGraph import (
    standardize_fit, standardize_apply,
    knn_edge_index, connect_to_train_edges
)

def norm_pid(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

def can_stratify(y: np.ndarray) -> bool:
    counts = np.bincount(y)
    return counts.min() >= 2

def is_z_col(c: str) -> bool:
    # imaging features are z0..z511
    if not c.startswith("z"):
        return False
    tail = c[1:]
    return tail.isdigit()

def is_emb_col(c: str) -> bool:
    # molecular features are emb_0..emb_255
    if not c.startswith("emb_"):
        return False
    tail = c.split("_", 1)[1]
    return tail.isdigit()

def main():
    cfg = load_config("configs/config.yaml")
    artifacts = ensure_dir(Path(cfg["data"]["artifacts_dir"]))

    img_csv = artifacts / cfg["data"]["imaging_embeddings_csv"]
    mol_csv = artifacts / cfg["data"]["molecular_embeddings_csv"]
    lab_csv = artifacts / cfg["data"]["molecular_labels_csv"]
    out_pt  = artifacts / "fusion_graph.pt"

    img = pd.read_csv(img_csv)
    mol = pd.read_csv(mol_csv)
    lab = pd.read_csv(lab_csv)

    # --- validate required columns
    for name, df in [("imaging_embeddings", img), ("molecular_embeddings", mol), ("molecular_labels", lab)]:
        if "patient_id" not in df.columns:
            raise ValueError(f"{name} must contain column: patient_id")
    if "subtype_id" not in lab.columns:
        raise ValueError("molecular_labels.csv must contain column: subtype_id")

    # --- normalize patient_id across all tables (prevents merge mismatch)
    img["patient_id"] = norm_pid(img["patient_id"])
    mol["patient_id"] = norm_pid(mol["patient_id"])
    lab["patient_id"] = norm_pid(lab["patient_id"])

    # --- align by patient_id: imaging ∩ molecular ∩ labels
    df = img.merge(mol, on="patient_id", how="inner")
    df = df.merge(lab[["patient_id", "subtype_id"]], on="patient_id", how="inner")

    if len(df) == 0:
        raise ValueError("No aligned patients after merging imaging, molecular, and labels.")

    # --- feature columns (based on your headers)
    img_cols = [c for c in df.columns if is_z_col(c)]
    mol_cols = [c for c in df.columns if is_emb_col(c)]

    # sort by numeric index so columns are stable
    img_cols = sorted(img_cols, key=lambda c: int(c[1:]))                  # z0..z511
    mol_cols = sorted(mol_cols, key=lambda c: int(c.split("_")[1]))        # emb_0..emb_255

    if len(img_cols) == 0 or len(mol_cols) == 0:
        raise ValueError(
            f"Could not find feature columns. Found img_cols={len(img_cols)} mol_cols={len(mol_cols)}. "
            f"Expected imaging z0..zN and molecular emb_0..emb_N."
        )

    # --- build X = [img || mol], y, ids
    x = np.concatenate(
        [df[img_cols].to_numpy(dtype=np.float32), df[mol_cols].to_numpy(dtype=np.float32)],
        axis=1
    ).astype(np.float32)

    y = df["subtype_id"].astype(int).to_numpy()
    patient_ids = df["patient_id"].astype(str).tolist()

    print(f"Aligned patients: {len(df)} | x={x.shape} | classes={sorted(set(y.tolist()))}")

    # --- patient-level split (stratify if possible)
    idx_all = np.arange(len(df))

    strat1 = y if can_stratify(y) else None
    idx_tr, idx_tmp, y_tr, y_tmp = train_test_split(
        idx_all, y, test_size=0.2, random_state=42, stratify=strat1
    )

    strat2 = y_tmp if (strat1 is not None and can_stratify(y_tmp)) else None
    idx_va, idx_te, y_va, y_te = train_test_split(
        idx_tmp, y_tmp, test_size=0.5, random_state=42, stratify=strat2
    )

    # --- standardize using TRAIN only
    x_tr_std, mu, sd = standardize_fit(x[idx_tr])
    x_va_std = standardize_apply(x[idx_va], mu, sd)
    x_te_std = standardize_apply(x[idx_te], mu, sd)

    # --- build graphs (no leakage)
    k = int(cfg.get("gnn", {}).get("knn_k", 10))
    metric = cfg.get("gnn", {}).get("knn_metric", "cosine")

    edge_tr = knn_edge_index(x_tr_std, k=k, metric=metric)

    # train+val graph: [train, val], val connected only to train
    x_trva = np.concatenate([x_tr_std, x_va_std], axis=0)
    edge_va_to_tr = connect_to_train_edges(x_tr_std, x_va_std, k=k, metric=metric)
    edge_trva = torch.cat([edge_tr, edge_va_to_tr], dim=1)

    # train+test graph: [train, test], test connected only to train
    x_trte = np.concatenate([x_tr_std, x_te_std], axis=0)
    edge_te_to_tr = connect_to_train_edges(x_tr_std, x_te_std, k=k, metric=metric)
    edge_trte = torch.cat([edge_tr, edge_te_to_tr], dim=1)

    pack = {
        "patient_id": patient_ids,            # aligned ordering
        "x": torch.from_numpy(x).float(),     # raw (unstandardized)
        "y": torch.from_numpy(y).long(),
        "splits": {
            "train": idx_tr.tolist(),
            "val": idx_va.tolist(),
            "test": idx_te.tolist(),
        },
        "standardize": {
            "mu": torch.from_numpy(mu).float(),
            "sd": torch.from_numpy(sd).float(),
        },
        "feature_cols": {
            "imaging": img_cols,
            "molecular": mol_cols,
        },
        "graphs": {
            "train": {
                "x": torch.from_numpy(x_tr_std).float(),
                "edge_index": edge_tr,
            },
            "train_val": {
                "x": torch.from_numpy(x_trva).float(),
                "edge_index": edge_trva,
                "n_train": len(idx_tr),
            },
            "train_test": {
                "x": torch.from_numpy(x_trte).float(),
                "edge_index": edge_trte,
                "n_train": len(idx_tr),
            },
        },
    }

    torch.save(pack, out_pt)
    print(f"[OK] wrote fusion graph (pure tensors): {out_pt}")
    print(f"[SHAPES] train_x={tuple(pack['graphs']['train']['x'].shape)} "
          f"trva_x={tuple(pack['graphs']['train_val']['x'].shape)} "
          f"trte_x={tuple(pack['graphs']['train_test']['x'].shape)}")

if __name__ == "__main__":
    main()
