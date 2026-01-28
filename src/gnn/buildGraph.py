# src/gnn/graph_build.py
from __future__ import annotations
import numpy as np
import torch

def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-8
    return (x - mu) / sd, mu, sd

def standardize_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd

def knn_edge_index(x: np.ndarray, k: int = 10, metric: str = "cosine") -> torch.LongTensor:
    """
    x: (N,D) numpy
    returns edge_index: (2,E) torch.long, symmetric (i<->j)
    """
    N = x.shape[0]
    if N <= 1:
        return torch.zeros((2, 0), dtype=torch.long)

    if metric == "cosine":
        xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        sim = xn @ xn.T  # (N,N)
        np.fill_diagonal(sim, -np.inf)
        nbrs = np.argsort(-sim, axis=1)[:, : min(k, N - 1)]
    else:
        # euclidean
        d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        nbrs = np.argsort(d2, axis=1)[:, : min(k, N - 1)]

    rows = np.repeat(np.arange(N), nbrs.shape[1])
    cols = nbrs.reshape(-1)

    # make symmetric
    edges = np.vstack([np.concatenate([rows, cols]), np.concatenate([cols, rows])])
    return torch.from_numpy(edges).long()

def connect_to_train_edges(x_train: np.ndarray, x_new: np.ndarray, k: int = 10, metric: str = "cosine") -> torch.LongTensor:
    """
    Build edges from each new node -> k nearest TRAIN nodes (and symmetric back).
    Output edge_index in the combined graph indexing:
      train nodes: [0..Ntr-1], new nodes: [Ntr..Ntr+Nnew-1]
    """
    Ntr = x_train.shape[0]
    Nnew = x_new.shape[0]
    if Ntr == 0 or Nnew == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    if metric == "cosine":
        tr = x_train / (np.linalg.norm(x_train, axis=1, keepdims=True) + 1e-8)
        nw = x_new / (np.linalg.norm(x_new, axis=1, keepdims=True) + 1e-8)
        sim = nw @ tr.T  # (Nnew, Ntr)
        nbrs = np.argsort(-sim, axis=1)[:, : min(k, Ntr)]
    else:
        d2 = ((x_new[:, None, :] - x_train[None, :, :]) ** 2).sum(-1)
        nbrs = np.argsort(d2, axis=1)[:, : min(k, Ntr)]

    new_idx = np.arange(Nnew) + Ntr
    rows = np.repeat(new_idx, nbrs.shape[1])
    cols = nbrs.reshape(-1)

    edges = np.vstack([np.concatenate([rows, cols]), np.concatenate([cols, rows])])
    return torch.from_numpy(edges).long()
