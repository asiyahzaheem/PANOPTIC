from __future__ import annotations
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import load_config, ensure_dir
from src.utils.seed import set_seed
from src.models.gnnModel import GraphSAGEClassifier


def acc(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def main():
    cfg = load_config("configs/config.yaml")
    set_seed(cfg["train_gnn"]["seed"])

    artifacts = ensure_dir(Path(cfg["data"]["artifacts_dir"]))
    graph_pt = artifacts / Path(cfg["data"]["fusion_graph_pt"]).name
    models_dir = ensure_dir(Path(cfg["data"]["gnn_models_dir"]))

    pack = torch.load(graph_pt, map_location="cpu", weights_only=False)
    data = pack["data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    in_dim = data.x.size(1)
    out_dim = int(data.y.max().item() + 1)

    model = GraphSAGEClassifier(
        in_dim=in_dim,
        hidden_dim=cfg["gnn"]["hidden_dim"],
        out_dim=out_dim,
        num_layers=cfg["gnn"]["num_layers"],
        dropout=cfg["gnn"]["dropout"],
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["train_gnn"]["lr"], weight_decay=cfg["train_gnn"]["weight_decay"])

    best_val = -1.0
    best_path = models_dir / "gnn_best.pt"

    for epoch in range(1, cfg["train_gnn"]["epochs"] + 1):
        model.train()
        opt.zero_grad()
        logits = model(data.x, data.edge_index)

        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            tr_acc = acc(logits[data.train_mask], data.y[data.train_mask])
            va_acc = acc(logits[data.val_mask], data.y[data.val_mask])
            te_acc = acc(logits[data.test_mask], data.y[data.test_mask])

        if va_acc > best_val:
            best_val = va_acc
            torch.save({"model": model.state_dict(), "cfg": cfg}, best_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | loss {loss.item():.4f} | train {tr_acc:.3f} | val {va_acc:.3f} | test {te_acc:.3f}")

    print(f"[OK] best val={best_val:.3f} saved -> {best_path}")


if __name__ == "__main__":
    main()
