# api/core/config.py
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]  # .../pdac/api -> .../pdac
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import load_config

def get_cfg() -> dict:
    return load_config("configs/config.yaml")

def artifacts_dir(cfg: dict) -> Path:
    return Path(cfg["data"]["artifacts_dir"])

def fusion_graph_path(cfg: dict) -> Path:
    return artifacts_dir(cfg) / cfg["data"]["fusion_graph_pt"]

def model_ckpt_path(cfg: dict) -> Path:
    # you save trainGNN output to cfg["data"]["models_dir"]/gnn_best.pt
    models_dir = Path(cfg["data"].get("models_dir", "models"))
    # models_dir in your config is "models" (relative), so make it relative to pdac/
    # (same as your training script does)
    return models_dir / "gnn_best.pt"
