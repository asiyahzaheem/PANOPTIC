from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.utils.io import load_config

# IMPORTANT: these imports must match your repo structure
from pdac.molecular.embeddingModel import MolecularEmbedder

def _load_feature_cols(cfg: dict) -> list[str]:
    artifacts = Path(cfg["data"]["artifacts_dir"])
    cols_path = artifacts / cfg["data"]["molecular_feature_cols_txt"]
    if not cols_path.exists():
        raise ValueError(f"Missing {cols_path}. Run scripts/exportMolecularCols.py first.")
    cols = [c.strip() for c in cols_path.read_text().splitlines() if c.strip()]
    if not cols:
        raise ValueError("Feature columns file is empty.")
    return cols

def _load_embedder(cfg: dict, input_dim: int, device: torch.device) -> MolecularEmbedder:
    artifacts = Path(cfg["data"]["artifacts_dir"])
    ckpt_path = artifacts / cfg["data"]["molecular_embedder_ckpt"]
    if not ckpt_path.exists():
        raise ValueError(
            f"Missing {ckpt_path}. Re-run scripts/extract_molecular_embeddings.py after adding model saving."
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    emb_dim = int(cfg["molecular"]["embedding_dim"])
    hidden_dim = int(cfg["molecular"]["hidden_dim"])

    model = MolecularEmbedder(input_dim=input_dim, emb_dim=emb_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model

def _parse_embedding_tsv(df: pd.DataFrame) -> np.ndarray:
    emb_cols = [c for c in df.columns if c.startswith("emb_") and c.split("_")[1].isdigit()]
    if not emb_cols:
        raise ValueError("Embedding TSV format requires emb_0..emb_255 columns.")
    emb_cols = sorted(emb_cols, key=lambda c: int(c.split("_")[1]))
    return df.loc[0, emb_cols].to_numpy(dtype=np.float32)

def _parse_expression_gene_value_tsv(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    # expects columns: gene, value
    if "gene" not in df.columns or "value" not in df.columns:
        raise ValueError("Expression TSV must have columns: gene, value")

    # Map gene -> value
    g2v = dict(zip(df["gene"].astype(str), df["value"].astype(float)))

    # Build vector in the exact feature column order
    x = np.zeros((len(feature_cols),), dtype=np.float32)
    for i, g in enumerate(feature_cols):
        if g in g2v:
            x[i] = float(g2v[g])
    return x

@torch.no_grad()
def load_molecular_embedding_from_uploaded_tsv(tsv_path: Path) -> np.ndarray:
    """
    Returns emb vector (256,) from either:
      - TSV with emb_0..emb_255 (single row), OR
      - TSV with gene/value pairs (raw expression), which will be embedded using MolecularEmbedder.
    """
    cfg = load_config("configs/config.yaml")
    df = pd.read_csv(tsv_path, sep="\t")

    if len(df) < 1:
        raise ValueError("Uploaded molecular TSV is empty.")

    # Case 1: already an embedding TSV
    if any(c.startswith("emb_") for c in df.columns):
        emb = _parse_embedding_tsv(df)
        return emb

    # Case 2: raw gene/value TSV -> build X -> embed
    feature_cols = _load_feature_cols(cfg)
    x_expr = _parse_expression_gene_value_tsv(df, feature_cols)  # [D]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_embedder(cfg, input_dim=len(feature_cols), device=device)

    x = torch.tensor(x_expr, dtype=torch.float32, device=device).unsqueeze(0)  # [1, D]
    emb = model(x)[0].detach().cpu().numpy().astype(np.float32)               # [256]
    return emb
