from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path

_cwd=Path.cwd()
_project_root=_cwd if (_cwd/'Start-Workbench.cmd').is_file() else Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("HI3D_WORKBENCH_ROOT", _project_root)).resolve()
DATA = ROOT / "data"
MODELS = ROOT / "models"
SOURCE_REVISION = "23dcc684bc6e1fab86ecafb99ec5270edad122f9"
VERSION = "0.2.1"
MODEL_REVISION = "705fc0901c5d896cdcbf1dbf1feaa173e62bdd3a"

def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)

def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

def digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def config():
    default = ROOT / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    cfg = {"worker_python": str(default), "worker_command": [], "minimum_free_vram_mb": 1500, "port": 8378}
    cfg.update(read_json(ROOT / "config.local.json", {}))
    return cfg

def child_path(root: Path, name: str):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("路径超出工作目录")
    return path
