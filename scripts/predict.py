from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt


from src.utils.io import load_config, resolve_path
from src.data.preprocessing import (
    load_volume_slices_first,
    window_hu,
    normalize_scan_z,
    pick_central_slice_pool,
    to_uint8_01,
)
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
def predict_scan(model, nifti_path: Path, cfg, device):
    vol = load_volume_slices_first(nifti_path)
    vol = window_hu(vol, cfg["preprocess"]["hu_min"], cfg["preprocess"]["hu_max"])
    vol = normalize_scan_z(vol)
    pool = pick_central_slice_pool(vol, cfg["preprocess"]["slice_pool"])

    k = cfg["preprocess"]["k_slices"]
    if pool.shape[0] > k:
        start = (pool.shape[0] - k) // 2
        slices = pool[start:start + k]
    else:
        slices = pool


    tfm = make_eval_transform(cfg["preprocess"]["patch_resize"],
                              cfg["preprocess"]["patch_crop"])
    xs = []
    for sl in slices:
        sl8 = to_uint8_01(sl.astype(np.float32))
        xs.append(tfm(sl8))
    x = torch.stack(xs, dim=0).to(device) 


    logits, _ = model(x)         
    probs = torch.sigmoid(logits).view(-1)
    scan_prob = float(probs.mean())

    return scan_prob, probs.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nifti", required=True, help="Path to .nii.gz CT scan")
    parser.add_argument("--plot", action="store_true", help="Plot slice probabilities")
    args = parser.parse_args()

    cfg = load_config("configs/config.yaml")
    base = Path(cfg["data"]["base_dir"])

    models_dir = resolve_path(base, cfg["data"]["models_dir"])
    model_path = models_dir / "cnn_pdac_mvp_best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNBackbone(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scan_prob, slice_probs = predict_scan(
        model,
        Path(args.nifti),
        cfg,
        device
    )

    print("\n=== PDAC Prediction ===")
    print(f"Scan: {args.nifti}")
    print(f"PDAC probability: {scan_prob:.4f}")
    print(f"Predicted class: {'PDAC' if scan_prob >= 0.5 else 'Non-PDAC'}")
    print(f"Slice probs (first 5): {slice_probs[:5]}")

    if args.plot:
        plt.figure()
        plt.plot(slice_probs, marker="o")
        plt.title("Slice-level PDAC probabilities (K slices)")
        plt.xlabel("Slice index (selected central slices)")
        plt.ylabel("PDAC probability")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.show()





if __name__ == "__main__":
    main()
