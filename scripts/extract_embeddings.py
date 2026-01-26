import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils.io import load_config
from src.data.dataset import ScanDatasetKSlice
from src.models.cnn_backbone import ResNet18Embedder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    ap.add_argument("--ckpt", required=True, help="Path to saved ckpt (.pt) containing embedder weights")
    ap.add_argument("--out_dir", default="artifacts/embeddings", help="Output dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    labels_csv = Path(cfg["paths"]["labels_csv"])
    df = pd.read_csv(labels_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load embedder
    embedder = ResNet18Embedder().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    embedder.load_state_dict(ckpt["embedder"])
    embedder.eval()

    # Dataset (no augmentation for deterministic embeddings)
    ds = ScanDatasetKSlice(df=df, cfg=cfg, transform=None)
    dl = DataLoader(ds, batch_size=cfg["train"].get("batch_size", 2), shuffle=False, num_workers=0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_embs = []
    with torch.no_grad():
        for xk, _y in dl:
            xk = xk.to(device)               # [B,K,3,H,W]
            B, K, C, H, W = xk.shape
            xflat = xk.view(B*K, C, H, W)     # [B*K,3,H,W]

            zflat = embedder(xflat)           # [B*K,512]
            zscan = zflat.view(B, K, -1).mean(1)  # [B,512]

            all_embs.append(zscan.cpu().numpy())

    embs = np.concatenate(all_embs, axis=0)   # [N,512]
    np.save(out_dir / "imaging_embeddings.npy", embs)

    index_df = df.copy()
    index_df["emb_index"] = np.arange(len(index_df))
    index_df.to_csv(out_dir / "imaging_embedding_index.csv", index=False)

    print("[OK] embeddings saved:", (out_dir / "imaging_embeddings.npy"))
    print("[OK] index saved:", (out_dir / "imaging_embedding_index.csv"))
    print("[OK] shape:", embs.shape)

if __name__ == "__main__":
    main()
