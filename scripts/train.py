from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from pandas.errors import EmptyDataError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import load_config, resolve_path
from src.utils.seed import set_seed
from src.data.dataset import ScanDatasetKSlice
from src.models.cnn_backbone import ResNet18Embedder, LinearHead


# ----------------------------
# Transforms
# ----------------------------
def repeat_to_3ch(x):
    return x.repeat(3, 1, 1)

def make_transform(resize: int, crop: int):
    return T.Compose([
        T.ToPILImage(),
        T.Resize(resize),
        T.RandomResizedCrop(crop, scale=(0.8, 1.0)),
        T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3)], p=0.8),
        T.ToTensor(),
        repeat_to_3ch,
    ])


# ----------------------------
# Train / Val Epoch
# ----------------------------
def run_epoch(embedder, head, loader, optimizer, criterion, device, train=True):
    embedder.train() if train else embedder.eval()
    head.train() if train else head.eval()

    total_loss, correct, total = 0.0, 0, 0

    for xk, y, _ in tqdm(loader, desc=("train" if train else "val"), leave=False):
        B, K, C, H, W = xk.shape
        xk = xk.to(device)
        y = y.to(device).float()

        xflat = xk.view(B * K, C, H, W)
        zflat = embedder(xflat)               # [B*K, D]
        logits_flat = head(zflat)             # [B*K, 1]

        logits_scan = logits_flat.view(B, K, 1).mean(dim=1).squeeze(1)  # [B]
        loss = criterion(logits_scan, y)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * B
        preds = (torch.sigmoid(logits_scan) >= 0.5).float()
        correct += (preds == y).sum().item()
        total += y.numel()

    return total_loss / max(total, 1), correct / max(total, 1)


# ----------------------------
# Eval by Source (NIH / TCIA)
# ----------------------------
def eval_by_source(embedder, head, loader, device):
    embedder.eval()
    head.eval()

    stats = {}
    counts = {}

    with torch.no_grad():
        for xk, y, source in loader:
            B, K, C, H, W = xk.shape
            xk = xk.to(device)
            y = y.to(device).float()

            xflat = xk.view(B * K, C, H, W)
            zflat = embedder(xflat)
            logits_flat = head(zflat)

            logits_scan = logits_flat.view(B, K, 1).mean(dim=1).squeeze(1)
            preds = (torch.sigmoid(logits_scan) >= 0.5).float()

            for i in range(B):
                src = source[i]
                stats[src] = stats.get(src, 0) + float(preds[i] == y[i])
                counts[src] = counts.get(src, 0) + 1

    return {k: stats[k] / counts[k] for k in stats}


# ----------------------------
# Main
# ----------------------------
def main():
    cfg = load_config("configs/config.yaml")
    set_seed(cfg["train"]["seed"])

    base = Path(cfg["data"]["base_dir"])
    labels_csv = resolve_path(base, cfg["data"]["labels_csv"])
    models_dir = resolve_path(base, cfg["data"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(labels_csv)

    # ---- QC filtering (safe)
    qc_log = Path("artifacts/data_qc_log.csv")
    bad_set = set()

    if qc_log.exists() and qc_log.stat().st_size > 0:
        try:
            bad_df = pd.read_csv(qc_log)
            if "filepath" in bad_df.columns:
                bad_set = set(bad_df["filepath"].astype(str))
        except EmptyDataError:
            pass

    if bad_set:
        before = len(df)
        df = df[~df["filepath"].astype(str).isin(bad_set)].reset_index(drop=True)
        print(f"[TRAIN] Skipping QC scans: {before} -> {len(df)}")
    else:
        print("[TRAIN] No QC log found/usable; using all scans")

    # ---- Stratified split (label + source)
    df["strat_key"] = df["source"].astype(str) + "_" + df["label"].astype(str)
    train_df, val_df = train_test_split(
        df,
        test_size=cfg["train"]["val_frac"],
        random_state=cfg["train"]["seed"],
        stratify=df["strat_key"],
    )

    print("TRAIN source counts:\n", train_df["source"].value_counts())
    print("VAL source counts:\n", val_df["source"].value_counts())

    train_ds = ScanDatasetKSlice(train_df, cfg, transform=None)
    val_ds   = ScanDatasetKSlice(val_df, cfg, transform=None)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedder = ResNet18Embedder().to(device)
    head = LinearHead(embedder.out_dim).to(device)

    optimizer = torch.optim.Adam(
        list(embedder.parameters()) + list(head.parameters()),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    criterion = nn.BCEWithLogitsLoss()

    writer = SummaryWriter("runs/pdac_cnn")
    best_val = 0.0
    best_path = models_dir / "cnn_pdac_mvp_best.pt"

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        tr_loss, tr_acc = run_epoch(embedder, head, train_loader, optimizer, criterion, device, True)
        va_loss, va_acc = run_epoch(embedder, head, val_loader, optimizer, criterion, device, False)

        writer.add_scalar("loss/train", tr_loss, epoch)
        writer.add_scalar("acc/train", tr_acc, epoch)
        writer.add_scalar("loss/val", va_loss, epoch)
        writer.add_scalar("acc/val", va_acc, epoch)

        domain_acc = eval_by_source(embedder, head, val_loader, device)

        print(
            f"Epoch {epoch} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
            f"TCIA acc {domain_acc.get('TCIA', 'NA')} | "
            f"NIH acc {domain_acc.get('NIH', 'NA')}"
        )

        if va_acc > best_val:
            best_val = va_acc
            torch.save(
                {"embedder": embedder.state_dict(), "head": head.state_dict(), "cfg": cfg},
                best_path,
            )
            print(f"[OK] saved best -> {best_path}")

    writer.close()


if __name__ == "__main__":
    main()
