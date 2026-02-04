"""
Graph construction utilities for GNN (KNN edges, standardization, etc.)
"""

from __future__ import annotations
import numpy as np
import torch

# Fit z-score normalization parameters (mean and std) on training data
def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-8
    return (x - mu) / sd, mu, sd

# Apply z-score normalization using pre-computed mean and std
def standardize_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd

# Build a KNN graph where each node connects to its k nearest neighbors (symmetric edges)
def knn_edge_index(x: np.ndarray, k: int = 10, metric: str = "cosine") -> torch.LongTensor:
    N = x.shape[0]
    if N <= 1:
        return torch.zeros((2, 0), dtype=torch.long)

    # compute nearest neighbors using cosine similarity or euclidean distance
    if metric == "cosine":
        xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        sim = xn @ xn.T  # (N,N)
        np.fill_diagonal(sim, -np.inf)
        nbrs = np.argsort(-sim, axis=1)[:, : min(k, N - 1)]
    else:
        # euclidean distance
        d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        nbrs = np.argsort(d2, axis=1)[:, : min(k, N - 1)]

    rows = np.repeat(np.arange(N), nbrs.shape[1])
    cols = nbrs.reshape(-1)

    # create symmetric edges (if i->j exists, add j->i)
    edges = np.vstack([np.concatenate([rows, cols]), np.concatenate([cols, rows])])
    return torch.from_numpy(edges).long()

# Connect new nodes to their k nearest training nodes (for inductive inference, prevents leakage)
def connect_to_train_edges(x_train: np.ndarray, x_new: np.ndarray, k: int = 10, metric: str = "cosine") -> torch.LongTensor:
    Ntr = x_train.shape[0]
    Nnew = x_new.shape[0]
    if Ntr == 0 or Nnew == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    # find k nearest train nodes for each new node
    if metric == "cosine":
        tr = x_train / (np.linalg.norm(x_train, axis=1, keepdims=True) + 1e-8)
        nw = x_new / (np.linalg.norm(x_new, axis=1, keepdims=True) + 1e-8)
        sim = nw @ tr.T  # (Nnew, Ntr)
        nbrs = np.argsort(-sim, axis=1)[:, : min(k, Ntr)]
    else:
        d2 = ((x_new[:, None, :] - x_train[None, :, :]) ** 2).sum(-1)
        nbrs = np.argsort(d2, axis=1)[:, : min(k, Ntr)]

    # new nodes start at index Ntr in the combined graph
    new_idx = np.arange(Nnew) + Ntr
    rows = np.repeat(new_idx, nbrs.shape[1])
    cols = nbrs.reshape(-1)

    edges = np.vstack([np.concatenate([rows, cols]), np.concatenate([cols, rows])])
    return torch.from_numpy(edges).long()

