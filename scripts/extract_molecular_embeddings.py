# scripts/extract_molecular_embeddings.py
from pathlib import Path
import sys
import pandas as pd
import torch

from src.utils.io import load_config
from molecular.embedding_model import MolecularEmbedder
from molecular.subtype_assignment import assign_subtypes


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
@torch.no_grad()
def main():
    cfg = load_config("configs/config.yaml")

    artifacts = Path(cfg["data"]["artifacts_dir"])
    index_csv = artifacts / cfg["data"]["molecular_index_csv"]
    labels_csv = artifacts / cfg["data"]["molecular_labels_csv"]
    out_csv = artifacts / cfg["data"]["molecular_embeddings_csv"]
    out_pt = artifacts / cfg["data"]["molecular_embeddings_pt"]

    df = pd.read_csv(index_csv)

    patient_ids = df["patient_id"].tolist()
    X = df.drop(columns=["patient_id"])

    # ---------- Subtype assignment (ground truth) ----------
    subtypes = assign_subtypes(df)

    labels_df = pd.DataFrame({
        "patient_id": patient_ids,
        "subtype": subtypes
    })
    labels_df.to_csv(labels_csv, index=False)

    # ---------- Embedding extraction ----------
    model = MolecularEmbedder(
        input_dim=X.shape[1],
        emb_dim=cfg["molecular"]["embedding_dim"],
        hidden_dim=cfg["molecular"]["hidden_dim"],
    )
    model.eval()

    emb = model(torch.tensor(X.values, dtype=torch.float32))

    # Save CSV
    emb_df = pd.DataFrame(
        emb.numpy(),
        columns=[f"emb_{i}" for i in range(emb.shape[1])]
    )
    emb_df.insert(0, "patient_id", patient_ids)
    emb_df.to_csv(out_csv, index=False)

    # Save PT
    torch.save({
        "patient_ids": patient_ids,
        "embeddings": emb,
        "subtypes": subtypes,
    }, out_pt)

    print(f"[OK] Molecular embeddings saved → {out_pt}")

if __name__ == "__main__":
    main()
