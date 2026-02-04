# api/services/ct_embedder.py
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parents[2]  # .../pdac/api/services -> .../pdac
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import ScanKSliceDataset
from pdac.src.models.cnnBackbone import ResNet18Embedder

def repeat_to_3ch(x: torch.Tensor) -> torch.Tensor:
    return x.repeat(3, 1, 1)

def make_transform(resize: int, crop: int):
    return T.Compose([
        T.ToPILImage(),
        T.Resize(resize),
        T.CenterCrop(crop),
        T.ToTensor(),
        repeat_to_3ch,
    ])

@torch.no_grad()
def extract_ct_embedding_single(ct_path: Path, cfg: dict) -> np.ndarray:
    """
    Returns z-vector (512,) for a single CT NIfTI file using the SAME pipeline
    (ScanKSliceDataset + ResNet18Embedder) used in imaging_embeddings generation.
    """
    ct_path = Path(ct_path)
    if not ct_path.exists():
        raise ValueError(f"CT file not found: {ct_path}")

    # MVP: only accept nifti
    suf = "".join(ct_path.suffixes).lower()
    if suf not in [".nii", ".nii.gz"]:
        raise ValueError("CT upload must be .nii or .nii.gz for this API MVP.")

    # Build a 1-row index dataframe matching what ScanKSliceDataset expects:
    # imaging_index.csv had columns: patient_id, filepath, source
    df = pd.DataFrame([{
        "patient_id": "UPLOAD_CASE",
        "filepath": str(ct_path),
        "source": "UPLOAD",
    }])

    tfm = make_transform(cfg["preprocess"]["patch_resize"], cfg["preprocess"]["patch_crop"])
    ds = ScanKSliceDataset(df, cfg, transform=tfm)

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["embeddings"]["num_workers"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = ResNet18Embedder().to(device).eval()

    for xk, pid, source in loader:
        # xk: [B,K,3,224,224]
        B, K, C, H, W = xk.shape
        xk = xk.to(device)
        xflat = xk.view(B * K, C, H, W)
        zflat = embedder(xflat)               # [B*K,512]
        zscan = zflat.view(B, K, -1).mean(1)  # [B,512]
        z = zscan[0].detach().cpu().numpy().astype(np.float32)
        return z

    raise RuntimeError("Failed to extract CT embedding (no batches produced).")
