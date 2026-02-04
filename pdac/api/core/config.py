# api/core/config.py
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]  # .../pdac/api -> .../pdac
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pdac.src.utils.io import load_config

def get_cfg() -> dict:
    _fallback_root = Path(__file__).resolve().parents[3]
    cfg_path = _fallback_root / "configs" / "config.yaml"
    cfg = load_config(cfg_path)
    # Use explicit project_root from config if set (avoids wrong path when multiple copies exist)
    root = cfg.get("data", {}).get("project_root")
    cfg["_project_root"] = Path(root) if root else _fallback_root
    return cfg

def _project_root(cfg: dict) -> Path:
    return cfg.get("_project_root", Path(__file__).resolve().parents[3])

def artifacts_dir(cfg: dict) -> Path:
    p = Path(cfg["data"]["artifacts_dir"])
    return p if p.is_absolute() else _project_root(cfg) / p

def fusion_graph_path(cfg: dict) -> Path:
    return artifacts_dir(cfg) / cfg["data"]["fusion_graph_pt"]

def model_ckpt_path(cfg: dict) -> Path:
    models_dir = Path(cfg["data"].get("models_dir", "models"))
    p = _project_root(cfg) / models_dir / "gnn_best.pt"
    return p

