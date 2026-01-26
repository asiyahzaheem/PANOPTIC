from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
from tqdm import tqdm

from src.utils.io import load_config, resolve_path
from src.data.dataset import ScanDatasetKSlice
from src.models.cnn_backbone import CNNBackbone


def repeat_to_3ch(x: torch.Tensor) -> torch.Tensor:
    return x.repeat(3, 1, 1)


def make_eval_transform(resize: int, crop: int):
 
    return T.Compose([
        T.ToPILImage(),
        T.Resize(resize),
        T.CenterCrop(crop),
        T.ToTensor(),
        repeat_to_3ch,
    ])


@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    for xk, y in tqdm(loader, desc="eval", leave=False):
        B, K, C, H, W = xk.shape
        xk = xk.to(device)
        y = y.to(device)

        logits_flat, _ = model(xk.view(B*K, C, H, W))
        logits_scan = logits_flat.view(B, K, 1).mean(dim=1)  
        probs = torch.sigmoid(logits_scan).view(-1).cpu().numpy()
        preds = (probs >= 0.5).astype(np.int32)

        y_true.extend(y.view(-1).cpu().numpy().astype(np.int32).tolist())
        y_pred.extend(preds.tolist())
        y_prob.extend(probs.tolist())

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def main():
    cfg = load_config("configs/config.yaml")
    base = Path(cfg["data"]["base_dir"])

    labels_csv = resolve_path(base, cfg["data"]["labels_csv"])
    models_dir = resolve_path(base, cfg["data"]["models_dir"])
    best_path = models_dir / "cnn_pdac_mvp_best.pt"

    df = pd.read_csv(labels_csv)

    train_parts, val_parts = [], []
    for src in df["source"].unique():
        src_df = df[df["source"] == src]
        tr, va = train_test_split(
            src_df,
            test_size=cfg["train"]["val_split"],
            random_state=cfg["train"]["seed"],
        )
        train_parts.append(tr)
        val_parts.append(va)

    val_df = pd.concat(val_parts).reset_index(drop=True)

    tfm = make_eval_transform(cfg["preprocess"]["patch_resize"], cfg["preprocess"]["patch_crop"])
    val_ds = ScanDatasetKSlice(
        val_df,
        hu_min=cfg["preprocess"]["hu_min"],
        hu_max=cfg["preprocess"]["hu_max"],
        k_slices=cfg["preprocess"]["k_slices"],
        slice_pool=cfg["preprocess"]["slice_pool"],
        transform=tfm
    )
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNBackbone(pretrained=False).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))

    y_true, y_pred, y_prob = eval_loader(model, val_loader, device)

    
    acc = (y_true == y_pred).mean()
    print("\n=== Overall Validation ===")
    print("Accuracy:", float(acc))
    print("Confusion matrix [[TN, FP],[FN, TP]]:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))


    print("\n=== Per-source Validation Accuracy ===")
    for src in ["TCIA", "NIH"]:
        src_mask = (val_df["source"].values == src)
        if src_mask.sum() == 0:
            continue
        src_acc = (y_true[src_mask] == y_pred[src_mask]).mean()
        print(f"{src}: {float(src_acc):.4f}  (n={int(src_mask.sum())})")


    print("\n=== Sample predictions (first 10) ===")
    for i in range(min(10, len(val_df))):
        print(f"{val_df.iloc[i]['source']} | y={y_true[i]} | p={y_prob[i]:.4f} | pred={y_pred[i]} | {val_df.iloc[i]['filepath']}")


if __name__ == "__main__":
    main()
