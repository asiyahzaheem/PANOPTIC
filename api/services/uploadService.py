# api/services/upload_service.py
from __future__ import annotations
from pathlib import Path
import uuid
from fastapi import UploadFile

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_upload(upload_dir: Path, f: UploadFile) -> Path:
    ensure_dir(upload_dir)
    suffix = "".join(Path(f.filename).suffixes)  # handles .nii.gz
    out = upload_dir / f"{uuid.uuid4().hex}{suffix}"
    with out.open("wb") as w:
        w.write(f.file.read())
    return out
