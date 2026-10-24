"""Separately managed, pinned model components for the optional asset pipeline.

These components never modify the Hi3D core model manifest. Downloading does
not accept third-party gated licenses on behalf of the user.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from .common import ROOT, atomic_json

MODEL_ROOT = ROOT / "models" / "extensions"
_LOCK = threading.Lock()
_STATE = {"state": "idle", "message": "扩展模型尚未下载"}
_STATE_PATH = None
SPECS = {
    "background": {
        "repo": "danielgatis/rembg (U2Netp ONNX release)",
        "revision": "v0.0.0-u2netp-309c8469258dda74",
        "license": "rembg MIT; U-2-Net Apache-2.0",
        "files": ["u2netp.onnx", "LICENSE.rembg.txt", "LICENSE.U2Net.txt"],
        "artifacts": {
            "u2netp.onnx": {"url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx", "bytes": 4574861, "sha256": "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8"},
            "LICENSE.rembg.txt": {"url": "https://raw.githubusercontent.com/danielgatis/rembg/c5b48b0417a0254df7d461efa14573597f369a83/LICENSE.txt", "bytes": 1069, "sha256": "90a3215072968fd304669c5389f04f1274a587abdd0507d99dead0f5511f8999"},
            "LICENSE.U2Net.txt": {"url": "https://raw.githubusercontent.com/xuebinqin/U-2-Net/ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29/LICENSE", "bytes": 11357, "sha256": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"},
        },
    },
    "coarse": {
        "repo": "tencent/Hunyuan3D-2mini",
        "revision": "f90a0f7df7d5e6f71109cf333f6a95a0ae3194a6",
        "license": "Tencent Hunyuan 3D 2.0 Community License",
        "files": ["hunyuan3d-dit-v2-mini-turbo/config.yaml", "hunyuan3d-dit-v2-mini-turbo/model.fp16.safetensors", "LICENSE", "NOTICE", "README.md"],
    },
    "texture": {
        "repo": "microsoft/TRELLIS.2-4B",
        "revision": "af44b45f2e35a493886929c6d786e563ec68364d",
        "license": "MIT; external image encoder has its own license",
        "files": ["texturing_pipeline.json", "README.md"] + [
            f"ckpts/{stem}.{ext}" for stem in (
                "shape_enc_next_dc_f16c32_fp16", "tex_dec_next_dc_f16c32_fp16",
                "slat_flow_imgshape2tex_dit_1_3B_512_bf16", "slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
            ) for ext in ("json", "safetensors")],
    },
    "texture_encoder": {
        "repo": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "revision": "3c276edd87d6f6e569ff0c4400e086807d0f3881",
        "upstream_repo": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "upstream_revision": "ea8dc2863c51be0a264bab82070e3e8836b02d51",
        "upstream_identical": True,
        "expected_weight_sha256": "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179",
        "license": "DINOv3 License; unchanged license in publicly published byte-identical mirror",
        "files": ["config.json", "model.safetensors", "preprocessor_config.json", "LICENSE.md", "README.md"],
    },
}


def runtime_python() -> Path:
    override = os.environ.get("HI3D_PIPELINE_PYTHON")
    if override:
        return Path(override).resolve()
    for base in (ROOT / "runtime-pipeline", ROOT / ".pipelineenv"):
        for name in ("Scripts/python.exe", "python.exe", "bin/python"):
            candidate = base / name
            if candidate.is_file():
                return candidate
    return ROOT / ".pipelineenv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _hash(path: Path, git_blob: bool = False) -> str:
    h = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        h.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _signature(path: Path):
    try:
        st = path.stat()
        result = [st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino]
        if os.name == "nt":
            from .models import _windows_change_time
            result.append(_windows_change_time(path))
        return result
    except OSError:
        return None


def _verified_signature(path: Path, expected, git_blob=False, size=None):
    """Bind a successful hash only to the file that was actually read.

    A replacement between hashing and saving metadata must never inherit the
    previous file's successful verification, even if its byte count is equal.
    """
    before = _signature(path)
    if before is None or not expected or (size is not None and before[0] != size):
        return None
    try:
        valid = _hash(path, git_blob=git_blob) == expected
    except OSError:
        return None
    after = _signature(path)
    return before if valid and before == after else None


def _manifest(root):
    value = _read(root / "manifest.json")
    components = value.get("components", {})
    errors = []
    if (root / "manifest.json").is_file() and not value:
        errors.append("扩展模型清单损坏或为空，请重新校验/下载")
    if not isinstance(components, dict):
        components = {}
        errors.append("扩展模型 components 记录损坏；请重新校验/下载")
    clean = {}
    for key, entry in components.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("files", {}), dict):
            errors.append(f"扩展模型 {key} 记录损坏")
            continue
        if any(not isinstance(item, dict) for item in entry.get("files", {}).values()):
            errors.append(f"扩展模型 {key} 文件记录损坏")
            continue
        clean[key] = entry
    return {**value, "components": clean}, errors


def _ranged_download(spec, name, destination, size, expected):
    """Bounded resumable ranges avoid stalled Xet transfers on some networks."""
    import requests
    from huggingface_hub import hf_hub_url, get_token
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == size and _hash(destination) == expected:
        return destination
    part_size = 4 * 1024 * 1024
    parts = destination.parent / ("." + destination.name + ".parts")
    parts.mkdir(exist_ok=True)
    url = hf_hub_url(spec["repo"], name, revision=spec["revision"])
    token = get_token()
    def fetch(index):
        begin = index * part_size
        end = min(size - 1, begin + part_size - 1)
        part = parts / str(index)
        if part.is_file() and part.stat().st_size == end-begin+1:
            return
        headers = {"Range": f"bytes={begin}-{end}"}
        if token:
            headers["Authorization"] = "Bearer " + token
        temp = part.with_suffix(".tmp")
        for attempt in range(5):
            try:
                with requests.get(url + f"?download=true&part={index}", headers=headers, stream=True, timeout=(20, 90)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {begin}-{end}/{size}":
                        raise ValueError("服务器未返回请求的下载区间")
                    with temp.open("wb") as stream:
                        received = 0
                        for block in response.iter_content(256*1024):
                            received += len(block)
                            if received > end-begin+1:
                                raise ValueError("下载区间响应超出请求长度")
                            stream.write(block)
                    if temp.stat().st_size != end-begin+1:
                        raise ValueError("下载区间不完整")
                    os.replace(temp, part)
                return
            except Exception:
                temp.unlink(missing_ok=True)
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
    total = (size+part_size-1)//part_size
    # Bounded concurrency can be lowered on rate-limited networks. Every range
    # is checked and the assembled immutable file still requires its full SHA.
    workers = max(1, min(40, int(os.environ.get("HI3D_DOWNLOAD_WORKERS", "40"))))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, i) for i in range(total)]
        for done, future in enumerate(as_completed(futures), 1):
            future.result()
            _STATE.update(file_bytes=min(size, done*part_size), file_total=size)
            if done % 50 == 0:
                if _STATE_PATH is not None:
                    atomic_json(_STATE_PATH, _STATE)
                print(f"PIPELINE_DOWNLOAD {name} {done}/{total}", flush=True)
    temp = destination.with_name(destination.name + ".assembled")
    with temp.open("wb") as out:
        for i in range(total):
            with (parts / str(i)).open("rb") as stream:
                for block in iter(lambda: stream.read(part_size), b""):
                    out.write(block)
    if _hash(temp) != expected:
        temp.unlink(missing_ok=True)
        # Never reuse untrusted ranges forever after a failed final hash.
        # Keep the evidence for diagnosis and start fresh on the next attempt.
        quarantine = parts.with_name(parts.name + f".invalid-{time.time_ns()}")
        if parts.resolve().parent != destination.parent.resolve():
            raise ValueError("下载分片目录超出目标目录")
        parts.rename(quarantine)
        raise ValueError(f"下载内容校验失败: {name}")
    os.replace(temp, destination)
    # Parts remain resumable until successful assembly, then release disk space.
    for part in parts.iterdir():
        part.unlink()
    parts.rmdir()
    return destination


def status(model_root: str | Path | None = None) -> dict:
    root = Path(model_root or MODEL_ROOT)
    manifest, errors = _manifest(root)
    components = {}
    for key, spec in SPECS.items():
        entry = manifest.get("components", {}).get(key, {})
        valid = entry.get("repo") == spec["repo"] and entry.get("revision") == spec["revision"]
        missing = []
        for name in spec["files"]:
            item = entry.get("files", {}).get(name, {})
            signature = _signature(root / key / name)
            if signature is None or item.get("verified") is not True or item.get("signature") != signature:
                missing.append(name)
        provenance = {k: spec[k] for k in ("upstream_repo", "upstream_revision", "upstream_identical", "expected_weight_sha256") if k in spec}
        components[key] = {"ready": bool(valid and not missing), "repo": spec["repo"], "revision": spec["revision"], "license": spec["license"], "missing": missing, "path": str(root / key), **provenance}
    executable = runtime_python()
    runtime_root = executable.parent.parent if executable.parent.name.lower() in {"scripts", "bin"} else executable.parent
    runtime = _read(runtime_root / "pipeline-runtime.json")
    runtime_ready = executable.is_file() and runtime.get("ready") is True
    download_state = {**_STATE, **_read(root / "download-state.json")}
    if root.is_dir():
        from filelock import FileLock, Timeout
        probe = FileLock(str(root / ".download.lock"))
        try:
            probe.acquire(timeout=0)
        except Timeout:
            download_state["state"] = "downloading"
        else:
            probe.release()
            if download_state.get("state") == "downloading":
                download_state.update(state="interrupted", message="扩展下载已中断，可恢复已校验分片")
    ready = runtime_ready and all(x["ready"] for x in components.values())
    if ready and download_state.get("state") == "idle" and download_state.get("message") == "扩展模型尚未下载":
        download_state["message"] = "扩展模型已就绪"
    return {"ready": ready,
            "background_ready": components["background"]["ready"],
            "coarse_ready": runtime_ready and components["coarse"]["ready"],
            "texture_ready": runtime_ready and components["texture"]["ready"] and components["texture_encoder"]["ready"],
            "components": components, "runtime": {**runtime, "ready": runtime_ready, "python": str(executable)},
            "download": download_state, "model_root": str(root), "errors": errors}


def download(selection: str = "all", model_root: str | Path | None = None, progress_callback=None) -> dict:
    """Synchronous, resumable Hub downloads followed by content-hash validation."""
    if selection not in {"all", "coarse", "texture", "background", "encoder"}:
        raise ValueError("selection must be all, coarse, texture, background or encoder")
    global _STATE_PATH
    from huggingface_hub import HfApi, hf_hub_download
    from filelock import FileLock
    root = Path(model_root or MODEL_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    keys = {"all": list(SPECS), "coarse": ["background", "coarse"],
            "texture": ["background", "texture", "texture_encoder"],
            "background": ["background"], "encoder": ["texture_encoder"]}[selection]
    with _LOCK, FileLock(str(root / ".download.lock"), timeout=0):
        _STATE_PATH = root / "download-state.json"
        _STATE.update(state="downloading", message="正在下载并校验固定版本扩展模型")
        atomic_json(_STATE_PATH, _STATE)
        try:
            manifest, _ = _manifest(root)
            manifest.setdefault("components", {})
            for key in keys:
                spec = SPECS[key]
                if "artifacts" not in spec:
                    info = HfApi().model_info(spec["repo"], revision=spec["revision"], files_metadata=True)
                    remote = {x.rfilename: x for x in info.siblings}
                entry = {"repo": spec["repo"], "revision": spec["revision"], "license": spec["license"], "files": {}}
                for field in ("upstream_repo", "upstream_revision", "upstream_identical", "expected_weight_sha256"):
                    if field in spec:
                        entry[field] = spec[field]
                if "upstream_repo" in spec:
                    original = HfApi().model_info(spec["upstream_repo"], revision=spec["upstream_revision"], files_metadata=True)
                    original_files = {x.rfilename: x for x in original.siblings}
                    equivalence = {}
                    for name in spec["files"]:
                        a, b = remote[name], original_files[name]
                        if a.blob_id != b.blob_id or a.size != b.size or bool(a.lfs) != bool(b.lfs) or (a.lfs and a.lfs.sha256 != b.lfs.sha256):
                            raise ValueError(f"公开副本与固定官方版本不一致: {name}")
                        equivalence[name] = {"git_blob": a.blob_id, "bytes": a.size, "sha256": a.lfs.sha256 if a.lfs else None}
                    if remote["model.safetensors"].lfs.sha256 != spec["expected_weight_sha256"]:
                        raise ValueError("编码器权重与预期官方SHA256不一致")
                    entry["upstream_equivalence"] = equivalence
                for name in spec["files"]:
                    file_size = spec["artifacts"][name]["bytes"] if "artifacts" in spec else remote[name].size
                    _STATE.update(component=key, file=name, file_bytes=0, file_total=file_size, message=f"下载/校验 {key}/{name}")
                    atomic_json(_STATE_PATH, _STATE)
                    if progress_callback:
                        progress_callback(dict(_STATE))
                    if "artifacts" in spec:
                        from types import SimpleNamespace
                        import requests
                        artifact = spec["artifacts"][name]
                        meta = SimpleNamespace(size=artifact["bytes"], lfs=SimpleNamespace(sha256=artifact["sha256"]))
                        path = root / key / name
                        if not path.is_file() or _hash(path) != artifact["sha256"]:
                            response = requests.get(artifact["url"], timeout=(20, 120))
                            response.raise_for_status()
                            path.parent.mkdir(parents=True, exist_ok=True)
                            temp = path.with_name(path.name + ".tmp")
                            temp.write_bytes(response.content)
                            if _hash(temp) != artifact["sha256"]:
                                temp.unlink(missing_ok=True)
                                raise ValueError(f"背景组件校验失败: {name}")
                            os.replace(temp, path)
                    else:
                        meta = remote[name]
                        path = _ranged_download(spec, name, root / key / name, meta.size, meta.lfs.sha256) if meta.lfs and meta.size > 32*1024*1024 else Path(hf_hub_download(spec["repo"], name, revision=spec["revision"], local_dir=root / key))
                    expected = meta.lfs.sha256 if meta.lfs else meta.blob_id
                    signature = _verified_signature(path, expected, git_blob=not bool(meta.lfs), size=meta.size)
                    if signature is None and "artifacts" not in spec and (not meta.lfs or meta.size <= 32*1024*1024):
                        # Hub's local-dir cache may reuse a damaged config or
                        # small weight whose ETag metadata is still current.
                        path = Path(hf_hub_download(spec["repo"], name, revision=spec["revision"],
                                                   local_dir=root / key, force_download=True))
                        signature = _verified_signature(path, expected, git_blob=not bool(meta.lfs), size=meta.size)
                    if signature is None:
                        raise RuntimeError(f"扩展模型内容校验失败: {key}/{name}")
                    _STATE.update(file_bytes=meta.size)
                    entry["files"][name] = {"hash": expected, "algorithm": "sha256" if meta.lfs else "git-sha1", "bytes": meta.size, "verified": True, "signature": signature}
                    manifest["components"][key] = entry
                    atomic_json(root / "manifest.json", manifest)
            _STATE.update(state="complete", message="扩展模型下载及内容校验完成")
            atomic_json(_STATE_PATH, _STATE)
        except Exception as exc:
            _STATE.update(state="failed", message=str(exc))
            atomic_json(_STATE_PATH, _STATE)
            raise
    return status(root)


def verify(model_root: str | Path | None = None) -> dict:
    """Hash local files without racing a downloader in another process."""
    from filelock import FileLock, Timeout
    root = Path(model_root or MODEL_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(root / ".download.lock"))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        return {**status(root), "verification_errors": ["扩展模型下载或校验仍在运行，请稍后重新校验"]}
    try:
        manifest, errors = _manifest(root)
        for key, spec in SPECS.items():
            entry = manifest.get("components", {}).get(key, {})
            for field in ("repo", "revision"):
                if entry.get(field) != spec[field]:
                    errors.append(f"{key}/manifest:{field}")
            for name in spec["files"]:
                item = entry.get("files", {}).get(name, {})
                path = root / key / name
                signature = _verified_signature(path, item.get("hash"), item.get("algorithm") == "git-sha1", item.get("bytes"))
                item["verified"] = signature is not None
                if signature is not None:
                    item["signature"] = signature
                else:
                    errors.append(f"{key}/{name}")
        if manifest:
            atomic_json(root / "manifest.json", manifest)
    finally:
        lock.release()
    return {**status(root), "verification_errors": errors}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["status", "download", "verify"], default="status", nargs="?")
    parser.add_argument("--selection", choices=["all", "coarse", "texture", "background", "encoder"], default="all")
    args = parser.parse_args()
    result = download(args.selection, progress_callback=lambda e: print(json.dumps(e, ensure_ascii=False), flush=True)) if args.action == "download" else globals()[args.action]()
    print(json.dumps(result, ensure_ascii=False, indent=2))
