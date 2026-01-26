from pathlib import Path
import yaml

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def resolve_path(base_dir: Path, p: str) -> Path:
    """
    Resolves a path string to a Path.

    Rules:
    - If p is absolute -> return it.
    - If p starts with "artifacts/" or "runs/" (repo-local) -> resolve relative to repo root.
    - Else -> resolve relative to base_dir (Drive).
    """
    pth = Path(p)

    if pth.is_absolute():
        return pth

    # repo root = directory containing configs/ (assumes scripts run from repo root)
    repo_root = Path.cwd()

    if str(pth).startswith("artifacts") or str(pth).startswith("runs"):
        return (repo_root / pth).resolve()

    return (base_dir / pth).resolve()
