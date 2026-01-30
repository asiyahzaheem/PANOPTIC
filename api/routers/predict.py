from __future__ import annotations
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from api.core.config import get_cfg, artifacts_dir, fusion_graph_path, model_ckpt_path
from api.services.uploadService import save_upload
from api.services.molParser import read_molecular_tsv
from api.services.ctEmbedder import extract_ct_embedding_single
from api.services.predictorService import PredictorService
from api.schemas.predict import PredictResponse

router = APIRouter()

_cfg = get_cfg()
_upload_dir = artifacts_dir(_cfg) / "uploads"

_predictor = PredictorService(
    fusion_graph_pt=fusion_graph_path(_cfg),
    model_pt=model_ckpt_path(_cfg),
    cfg=_cfg,
)

@router.post("/predict", response_model=PredictResponse)
async def predict(
    ct_file: UploadFile = File(..., description="CT scan .nii or .nii.gz"),
    molecular_file: UploadFile = File(..., description="Molecular TSV with emb_0..emb_255"),
    explain: str = Form("simple", description="none|simple|detailed"),
):
    try:
        ct_path = save_upload(_upload_dir, ct_file)
        mol_path = save_upload(_upload_dir, molecular_file)

        emb_vec = read_molecular_tsv(mol_path)
        z_vec = extract_ct_embedding_single(ct_path, _cfg)

        result = _predictor.predict(z_vec=z_vec, emb_vec=emb_vec, explain=explain)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
