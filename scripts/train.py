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
def repeat_to_3ch(x: torch.Tensor) -> torch.Tensor:
    # x: [1,H,W] -> [3,H,W]
    return x.repeat(3, 1, 1)

def make_transform(resize: int, crop: int, train: bool) -> T.Compose:
    if train:
        return T.Compose([
            T.ToPILImage(),
            T.Resize(resize),
            T.RandomResizedCrop(crop, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            repeat_to_3ch,
        ])
    else:
        return T.Compose([
            T.ToPILImage(),
            T.Resize(resize),
            T.CenterCrop(crop),
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

    for batch in tqdm(loader, desc=("train" if train else "val"), leave=False):
        # dataset is expected to return (xk, y, source)
        xk, y, _source = batch

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
# Eval by Source (generic)
# ----------------------------
def eval_by_source(embedder, head, loader, device):
    embedder.eval()
    head.eval()

    correct = {}
    counts = {}

    with torch.no_grad():
        for xk, y, source in loader:
            B, K, C, H, W = xk.shape
            xk = xk.to(device)
            y = y.to(device).float()

            xflat = xk.view(B * K, C, H, W)
            zflat = embedder(xflat)
            logits_flat = head(zflat)

            logits_scan = logits_flat.view(B, K, 1).mean(dim=1).squeeze(1)  # [B]
            preds = (torch.sigmoid(logits_scan) >= 0.5).float()

            for i in range(B):
                src = str(source[i])
                correct[src] = correct.get(src, 0.0) + float(preds[i].item() == y[i].item())
                counts[src] = counts.get(src, 0) + 1

    return {src: correct[src] / counts[src] for src in correct}


# ----------------------------
# Main
# ----------------------------
def main():
    cfg = load_config("configs/config.yaml")
    set_seed(cfg["train"]["seed"])

    base = Path(cfg["data"]["base_dir"])

    labels_csv = resolve_path(base, cfg["data"]["labels_csv"])
    models_dir = resolve_path(base, cfg["data"]["models_dir"])
    runs_dir   = resolve_path(base, cfg["data"]["runs_dir"])
    qc_log     = resolve_path(base, cfg["data"]["qc_log_csv"])

    models_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(labels_csv)

    # ---- Basic sanity filters (prevents dumb crashes)
    df["filepath"] = df["filepath"].astype(str)
    df = df[df["filepath"].apply(lambda p: Path(p).exists())].reset_index(drop=True)

    # ---- QC filtering (skip-only, safe)
    bad_set = set()
    if qc_log.exists() and qc_log.stat().st_size > 0:
        try:
            bad_df = pd.read_csv(qc_log)
            if "filepath" in bad_df.columns:
                bad_set = set(bad_df["filepath"].astype(str).tolist())
        except EmptyDataError:
            bad_set = set()

    if bad_set:
        before = len(df)
        df = df[~df["filepath"].isin(bad_set)].reset_index(drop=True)
        print(f"[TRAIN] Skipping QC scans: {before} -> {len(df)}")
    else:
        print("[TRAIN] No QC log found/usable; using all scans")

    # ---- Stratified split (label + source) with safe fallback
    if "source" not in df.columns:
        df["source"] = "UNKNOWN"

    df["strat_key"] = df["source"].astype(str) + "_" + df["label"].astype(str)
    key_counts = df["strat_key"].value_counts()

    stratify_col = df["strat_key"]
    if (key_counts < 2).any():
        # too small to stratify safely
        stratify_col = None
        print("[WARN] Stratify disabled (some groups < 2 samples).")

    train_df, val_df = train_test_split(
        df,
        test_size=cfg["train"]["val_frac"],
        random_state=cfg["train"]["seed"],
        shuffle=True,
        stratify=stratify_col,
    )

    print("TRAIN source counts:\n", train_df["source"].value_counts())
    print("VAL source counts:\n", val_df["source"].value_counts())
    print("TRAIN label counts:\n", train_df["label"].value_counts())
    print("VAL label counts:\n", val_df["label"].value_counts())

    # ---- Transforms actually used
    train_tf = make_transform(cfg["preprocess"]["patch_resize"], cfg["preprocess"]["patch_crop"], train=True)
    val_tf   = make_transform(cfg["preprocess"]["patch_resize"], cfg["preprocess"]["patch_crop"], train=False)

    train_ds = ScanDatasetKSlice(train_df, cfg, transform=train_tf)
    val_ds   = ScanDatasetKSlice(val_df,   cfg, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedder = ResNet18Embedder().to(device)
    head = LinearHead(embedder.out_dim).to(device)

    optimizer = torch.optim.Adam(
        list(embedder.parameters()) + list(head.parameters()),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    criterion = nn.BCEWithLogitsLoss()

    writer = SummaryWriter(str(runs_dir))
    print("[TB] TensorBoard logging to:", runs_dir)

    best_val = 0.0
    best_path = models_dir / "cnn_pdac_mvp_best.pt"

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        tr_loss, tr_acc = run_epoch(embedder, head, train_loader, optimizer, criterion, device, True)
        va_loss, va_acc = run_epoch(embedder, head, val_loader, optimizer, criterion, device, False)

        writer.add_scalar("loss/train", tr_loss, epoch)
        writer.add_scalar("acc/train", tr_acc, epoch)
        writer.add_scalar("loss/val", va_loss, epoch)
        writer.add_scalar("acc/val", va_acc, epoch)
        writer.flush()

        by_src = eval_by_source(embedder, head, val_loader, device)
        by_src_str = " | ".join([f"{k} acc {v:.4f}" for k, v in sorted(by_src.items())])

        print(
            f"Epoch {epoch} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f}"
            + (f" | {by_src_str}" if by_src_str else "")
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
