from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import torch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import load_config, ensure_dir
from src.utils.seed import set_seed

from torch_geometric.data import Data


def load_imaging_embeddings(imaging_pt: Path):
    """
    Expect imaging_embeddings.pt to contain either:
      - dict with keys: X_img (Tensor [N,D]), patient_id (list[str])
      OR
      - dict with embeddings keyed by patient_id
    Adjust here if your imaging pt uses different keys.
    """
    obj = torch.load(imaging_pt, map_location="cpu")
    if isinstance(obj, dict) and "X_img" in obj and "patient_id" in obj:
        return obj["patient_id"], obj["X_img"].float()

    # Fallback: if it's like {"patient_id": {"emb": [...]}, ...}
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        patient_ids = []
        feats = []
        for pid, v in obj.items():
            if isinstance(v, dict) and "embedding" in v:
                feats.append(torch.tensor(v["embedding"], dtype=torch.float32))
            elif torch.is_tensor(v):
                feats.append(v.float())
            else:
                continue
            patient_ids.append(pid)
        X = torch.stack(feats, dim=0)
        return patient_ids, X

    raise ValueError("Unknown imaging_embeddings.pt structure. Open it once and match keys here.")


def knn_edge_index(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Build undirected kNN graph edges from features x [N,D] using cosine similarity.
    Returns edge_index [2, E]
    """
    x = torch.nn.functional.normalize(x, p=2, dim=1)
    sim = x @ x.t()  # [N,N]
    sim.fill_diagonal_(-1)

    N = x.size(0)
    knn = torch.topk(sim, k=min(k, N-1), dim=1).indices  # [N,k]

    src = torch.arange(N).unsqueeze(1).repeat(1, knn.size(1)).reshape(-1)
    dst = knn.reshape(-1)

    # Make undirected by adding reverse edges
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return edge_index


def stratified_split(y: np.ndarray, val_frac: float, test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))

    train_mask = np.zeros(len(y), dtype=bool)
    val_mask = np.zeros(len(y), dtype=bool)
    test_mask = np.zeros(len(y), dtype=bool)

    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        rng.shuffle(cls_idx)

        n = len(cls_idx)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))

        test_i = cls_idx[:n_test]
        val_i = cls_idx[n_test:n_test + n_val]
        train_i = cls_idx[n_test + n_val:]

        test_mask[test_i] = True
        val_mask[val_i] = True
        train_mask[train_i] = True

    return train_mask, val_mask, test_mask


def main():
    cfg = load_config("configs/config.yaml")
    set_seed(cfg["train_gnn"]["seed"])

    artifacts = ensure_dir(Path(cfg["data"]["artifacts_dir"]))
    imaging_pt = artifacts / Path(cfg["data"]["imaging_embeddings_pt"]).name
    molecular_pt = artifacts / Path(cfg["data"]["molecular_gnn_pt"]).name
    out_graph = artifacts / Path(cfg["data"]["fusion_graph_pt"]).name

    mol = torch.load(molecular_pt, map_location="cpu")
    mol_ids = mol["patient_id"]
    X_mol = mol["X_mol"].float()
    y = mol["y"].long()

    img_ids, X_img = load_imaging_embeddings(imaging_pt)

    mol_map = {pid: i for i, pid in enumerate(mol_ids)}
    img_map = {pid: i for i, pid in enumerate(img_ids)}

    common = sorted(set(mol_ids).intersection(set(img_ids)))
    if len(common) < 8:
        raise RuntimeError(f"Too few aligned patients: {len(common)}. Need more overlap to train a GNN.")

    mol_idx = [mol_map[p] for p in common]
    img_idx = [img_map[p] for p in common]

    X_m = X_mol[mol_idx]
    X_i = X_img[img_idx]
    y_c = y[mol_idx]

    x = torch.cat([X_i, X_m], dim=1)  # late fusion node feature
    edge_index = knn_edge_index(x, k=cfg["gnn"]["k"])

    y_np = y_c.numpy()
    tr, va, te = stratified_split(
        y_np,
        val_frac=cfg["train_gnn"]["val_frac"],
        test_frac=cfg["train_gnn"]["test_frac"],
        seed=cfg["train_gnn"]["seed"]
    )

    data = Data(
        x=x,
        y=y_c,
        edge_index=edge_index,
    )
    data.patient_id = common
    data.train_mask = torch.tensor(tr)
    data.val_mask = torch.tensor(va)
    data.test_mask = torch.tensor(te)

    torch.save(
        {
            "data": data,
            "subtype_map": mol.get("subtype_map", None),
        },
        out_graph
    )
    print(f"[OK] wrote fusion graph: {out_graph}")
    print(f"Aligned patients: {len(common)} | x={tuple(x.shape)} | edges={edge_index.shape[1]}")


if __name__ == "__main__":
    main()
