from __future__ import annotations
from pathlib import Path
import yaml

def load_config(path: str | Path):
    path = Path(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def resolve_path(base: Path, maybe_path: str | Path) -> Path:
    p = Path(maybe_path)
    if p.is_absolute():
        return p
    return (base / p).resolve()
