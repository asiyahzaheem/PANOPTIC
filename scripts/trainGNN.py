"""
Trains the GNN model on the fusion graph (imaging + molecular features)
"""

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
from src.utils.io import load_config
from src.models.gnnModel import GraphSAGEClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# simple accuracy metric
def accuracy(logits, y):
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()

def main():
    cfg = load_config("configs/config.yaml")
    base = Path(cfg["data"]["base_dir"])
    artifacts = Path(cfg["data"]["artifacts_dir"])
    graph_pt = artifacts / "fusion_graph.pt"

    # load the pre-built fusion graph
    pack = torch.load(graph_pt, map_location="cpu", weights_only=False)

    # read training hyperparameters from config
    num_classes = int(cfg.get("gnn", {}).get("num_classes", 4))
    hidden = int(cfg.get("gnn", {}).get("hidden_dim", 128))
    lr = float(cfg.get("gnn", {}).get("lr", 1e-3))
    wd = float(cfg.get("gnn", {}).get("weight_decay", 1e-4))
    epochs = int(cfg.get("gnn", {}).get("epochs", 200))
    dropout = float(cfg.get("gnn", {}).get("dropout", 0.3))
    patience = int(cfg.get("gnn", {}).get("patience", 30))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load training graph (only train nodes and their edges)
    gtr = pack["graphs"]["train"]
    x_tr = gtr["x"].to(device)
    ei_tr = gtr["edge_index"].to(device)

    # extract train labels
    idx_train = np.array(pack["splits"]["train"], dtype=int)
    y_full = pack["y"].to(device)
    y_tr = y_full[idx_train].clone()

    # load train+val graph (val nodes only connect to train, no leakage)
    gva = pack["graphs"]["train_val"]
    x_trva = gva["x"].to(device)
    ei_trva = gva["edge_index"].to(device)
    n_train = int(gva["n_train"])

    # extract val labels
    idx_val = np.array(pack["splits"]["val"], dtype=int)
    # in the train+val graph, val nodes come after train nodes
    y_va = y_full[idx_val].clone()

    # load train+test graph for final evaluation
    gte = pack["graphs"]["train_test"]
    x_trte = gte["x"].to(device)
    ei_trte = gte["edge_index"].to(device)
    n_train_te = int(gte["n_train"])
    idx_test = np.array(pack["splits"]["test"], dtype=int)
    y_te = y_full[idx_test].clone()

    # initialize model and optimizer
    in_dim = x_tr.shape[1]
    model = GraphSAGEClassifier(in_dim=in_dim, hidden=hidden, num_classes=num_classes, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # track best model for early stopping
    best_val = -1.0
    best_state = None
    bad_epochs = 0

    # main training loop
    for ep in range(1, epochs + 1):
        model.train()
        logits_tr = model(x_tr, ei_tr)
        loss = F.cross_entropy(logits_tr, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # evaluate on val and test sets
        model.eval()
        with torch.no_grad():
            # train accuracy
            tr_acc = accuracy(logits_tr, y_tr)

            # val accuracy (only evaluate the val nodes in the train+val graph)
            logits_trva = model(x_trva, ei_trva)
            logits_val = logits_trva[n_train:]
            va_acc = accuracy(logits_val, y_va)

            # test accuracy
            logits_trte = model(x_trte, ei_trte)
            logits_test = logits_trte[n_train_te:]
            te_acc = accuracy(logits_test, y_te)

        # update best model if val improves
        if va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if ep == 1 or ep % 10 == 0:
            print(f"EPOCH {ep:03d} | loss {loss.item():.4f} | train {tr_acc:.3f} | val {va_acc:.3f} | test {te_acc:.3f}")

        # stop if no improvement for patience epochs
        if bad_epochs >= patience:
            break

    # save the best model
    out_dir = Path(cfg["data"].get("models_dir", "models"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gnn_best.pt"
    torch.save({"state_dict": best_state, "cfg": cfg}, out_path)
    print(f"Best val={best_val:.3f} saved -> {out_path}")

if __name__ == "__main__":
    main()
