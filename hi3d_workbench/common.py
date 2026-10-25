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
DEFAULT_MODELS = ROOT / "models"
SOURCE_REVISION = "23dcc684bc6e1fab86ecafb99ec5270edad122f9"
VERSION = "0.4.7"
MODEL_REVISION = "705fc0901c5d896cdcbf1dbf1feaa173e62bdd3a"

def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            # Commit bytes before publishing the replacement. Without this a
            # hard reset can leave a valid-looking filename full of zeroes.
            f.flush()
            os.fsync(f.fileno())
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
    cfg = {"worker_python": str(default), "worker_command": [], "minimum_free_vram_mb": 1500, "port": 8378, "models_dir": ""}
    cfg.update(read_json(ROOT / "config.local.json", {}))
    return cfg

def resolve_model_dir(value=None, *, root: Path | None = None):
    """Resolve one shared weight root independently of the shell's directory.

    This only validates a location; it never creates, moves or removes weights.
    Sources, Python runtimes and user data remain under ROOT.
    """
    base = Path(root if root is not None else ROOT).resolve()
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ValueError("模型目录必须为路径字符串，留空可恢复默认")
    value = value.strip()
    if any(c in value for c in ('\x00', '\r', '\n')):
        raise ValueError("模型目录包含无效字符")
    candidate = Path(value) if value else base / 'models'
    if os.name == 'nt':
        if (candidate.drive or candidate.root) and not candidate.is_absolute():
            raise ValueError("请填写完整绝对路径，或相对于工作台目录的路径")
        for part in candidate.parts[1:] if candidate.anchor else candidate.parts:
            if part in {'.', '..'}:
                continue
            if (any(c in part for c in '<>:"|?*') or part.endswith((' ', '.'))
                    or Path(part).is_reserved()):
                raise ValueError("模型目录包含 Windows 不支持的名称")
    try:
        resolved = (candidate if candidate.is_absolute() else base / candidate).resolve()
        for ancestor in (resolved, *resolved.parents):
            if ancestor.exists():
                if not ancestor.is_dir():
                    raise ValueError("模型目录或其上级路径是文件，请选择文件夹")
                break
    except OSError as exc:
        raise ValueError(f"模型目录无法访问：{exc}") from exc
    return resolved

# A worker starts in a new process and resolves the same persisted setting as
# the web server. Never derive a relative models_dir from a worker's cwd.
MODELS = resolve_model_dir(config().get('models_dir'))

def child_path(root: Path, name: str):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("路径超出工作目录")
    return path
