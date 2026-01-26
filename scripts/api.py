from __future__ import annotations
import os
import tempfile
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as T
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from src.utils.io import load_config, resolve_path
from src.data.preprocessing import (
    load_volume_slices_first,
    window_hu,
    normalize_scan_z,
    pick_central_slice_pool,
    to_uint8_01,
)
from src.models.cnn_backbone import CNNBackbone

app = FastAPI(title="PDAC CNN MVP API")

cfg = None
device = None
model = None
tfm = None


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
    vol = load_volume_slices_first(nifti_path)  # (S,H,W)
    vol = window_hu(vol, cfg["preprocess"]["hu_min"], cfg["preprocess"]["hu_max"])
    vol = normalize_scan_z(vol)

    pool = pick_central_slice_pool(vol, cfg["preprocess"]["slice_pool"])

    k = cfg["preprocess"]["k_slices"]
    if pool.shape[0] > k:
        start = (pool.shape[0] - k) // 2
        slices = pool[start:start + k]
    else:
        slices = pool

    xs = []
    for sl in slices:
        sl8 = to_uint8_01(sl.astype(np.float32))
        xs.append(tfm(sl8))
    x = torch.stack(xs, dim=0).to(device)  

    logits, _ = model(x)  
    probs = torch.sigmoid(logits).view(-1).cpu().numpy()
    scan_prob = float(probs.mean())
    pred = int(scan_prob >= 0.5)

    return scan_prob, pred, probs.tolist()


@app.on_event("startup")
def _startup():
    global cfg, device, model, tfm

    cfg = load_config("configs/config.yaml")
    base = Path(cfg["data"]["base_dir"])
    models_dir = resolve_path(base, cfg["data"]["models_dir"])
    model_path = models_dir / "cnn_pdac_mvp_best.pt"

    if not model_path.exists():
        raise RuntimeError(f"Model not found: {model_path}. Train first with scripts/train.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNBackbone(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    tfm = make_eval_transform(cfg["preprocess"]["patch_resize"], cfg["preprocess"]["patch_crop"])

    print(f"[API] Loaded model from: {model_path}")
    print(f"[API] Device: {device}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # basic validation
    name = file.filename or ""
    if not (name.endswith(".nii") or name.endswith(".nii.gz")):
        raise HTTPException(status_code=400, detail="Upload a .nii or .nii.gz file")

    # TEMP file
    suffix = ".nii.gz" if name.endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        content = await file.read()
        tmp.write(content)

    try:
        scan_prob, pred, slice_probs = predict_scan(model, tmp_path, cfg, device)
        return JSONResponse({
            "filename": name,
            "pdac_probability": round(scan_prob, 6),
            "prediction": "PDAC" if pred == 1 else "Non-PDAC",
            "slice_probs": slice_probs,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {repr(e)}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
