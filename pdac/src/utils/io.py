from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any, Dict

def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve_path(base_dir: str | Path, relative: str) -> Path:
    return Path(base_dir) / relative


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

