"""Pinned optional acceleration assets; importing this module never imports Torch."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import urllib.request
import uuid

VOXEL_ACCELERATION_WHEEL = {
    "package": "hi3d-voxel-cuda", "version": "0.1.3",
    "filename": "hi3d_voxel_cuda-0.1.3-cp312-cp312-win_amd64.whl",
    "url": "https://huggingface.co/t8star/Hi3d-Comfy/resolve/d22a4b9aaa08ec255ad3ae982d211c99f4a4b7c9/runtime-wheels/v0.4.9/hi3d_voxel_cuda-0.1.3-cp312-cp312-win_amd64.whl",
    "bytes": 533673,
    "sha256": "ee2c0465aaf83bea3090197edfcc35a610e43c16809b0f97f370204b1a60ceb6",
    "python": [3, 12], "torch": "2.8.0+cu128", "cuda": "12.8", "platform": "win32",
    "architectures": "8.6;8.9;12.0+PTX", "license": "MIT",
    "build_id": "e477c69457c650dc680c02c2a1fb2ca58d3429ad-t8-two-pass-fpstrict-v2",
    "source": "https://github.com/quantaji/o-voxel-gpu/tree/e477c69457c650dc680c02c2a1fb2ca58d3429ad",
}


FLASH_ATTENTION_WHEEL = {
    "package": "flash-attn",
    "version": "2.8.3",
    "filename": "flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp312-cp312-win_amd64.whl",
    "url": "https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu128torch2.8.0cxx11abiFALSE-cp312-cp312-win_amd64.whl",
    "bytes": 127588011,
    "sha256": "709d6663d74f96bb0304b5ba2ffee84a4dafbcb56d73d35bc90b7ca8cd44261b",
    "python": [3, 12],
    "torch": "2.8.0+cu128",
    "cuda": "12.8",
    "platform": "win32",
    "architectures": ["sm80", "sm90", "sm100", "sm120"],
    "license": "BSD-3-Clause",
    "release": "https://github.com/kingbri1/flash-attention/releases/tag/v2.8.3",
    "build_source": "https://github.com/kingbri1/flash-attention/blob/v2.8.3/setup.py",
}

UV_ACCELERATION_WHEELS = (
    {
        "package": "llvmlite", "version": "0.44.0",
        "filename": "llvmlite-0.44.0-cp312-cp312-win_amd64.whl",
        "url": "https://files.pythonhosted.org/packages/e2/3b/a9a17366af80127bd09decbe2a54d8974b6d8b274b39bf47fbaedeec6307/llvmlite-0.44.0-cp312-cp312-win_amd64.whl",
        "bytes": 30332380,
        "sha256": "eae7e2d4ca8f88f89d315b48c6b741dcb925d6a1042da694aa16ab3dd4cbd3a1",
        "python": [3, 12], "numpy": "1.26.4", "platform": "win32",
        "license": "BSD", "release": "https://pypi.org/project/llvmlite/0.44.0/",
    },
    {
        "package": "numba", "version": "0.61.2",
        "filename": "numba-0.61.2-cp312-cp312-win_amd64.whl",
        "url": "https://files.pythonhosted.org/packages/68/1d/ddb3e704c5a8fb90142bf9dc195c27db02a08a99f037395503bfbc1d14b3/numba-0.61.2-cp312-cp312-win_amd64.whl",
        "bytes": 2831929,
        "sha256": "97cf4f12c728cf77c9c1d7c23707e4d8fb4632b46275f8f3397de33e5877af18",
        "python": [3, 12], "numpy": "1.26.4", "platform": "win32",
        "license": "BSD", "release": "https://pypi.org/project/numba/0.61.2/",
        "compatibility": "https://numba.readthedocs.io/en/0.61.2/user/installing.html#version-support-information",
    },
)


def validate_flash_environment(environment: dict) -> None:
    """Reject incompatible ABIs before any download or pip invocation."""
    expected = FLASH_ATTENTION_WHEEL
    mismatches = []
    for field in ("platform", "python", "torch", "cuda"):
        value = environment.get(field)
        if field == "python" and isinstance(value, tuple):
            value = list(value)
        if value != expected[field]:
            mismatches.append(f"{field}={value!r} (requires {expected[field]!r})")
    if environment.get("implementation") != "CPython":
        mismatches.append("requires CPython")
    if (environment.get("pointer_bits") != 64
            or str(environment.get("machine", "")).lower() not in {"amd64", "x86_64"}):
        mismatches.append("requires Windows x86-64")
    if mismatches:
        raise RuntimeError("Native FlashAttention ABI mismatch: " + "; ".join(mismatches))


def validate_uv_environment(environment: dict) -> None:
    """Match the existing NumPy ABI without importing or changing Torch."""
    mismatches = []
    for field, expected in (("platform", "win32"), ("python", [3, 12]), ("numpy", "1.26.4")):
        value = environment.get(field)
        if field == "python" and isinstance(value, tuple):
            value = list(value)
        if value != expected:
            mismatches.append(f"{field}={value!r} (requires {expected!r})")
    if environment.get("implementation") != "CPython":
        mismatches.append("requires CPython")
    if (environment.get("pointer_bits") != 64
            or str(environment.get("machine", "")).lower() not in {"amd64", "x86_64"}):
        mismatches.append("requires Windows x86-64")
    if mismatches:
        raise RuntimeError("Numba UV acceleration ABI mismatch: " + "; ".join(mismatches))


def validate_voxel_environment(environment: dict) -> None:
    try:
        validate_flash_environment(environment)
    except RuntimeError as error:
        raise RuntimeError(str(error).replace('Native FlashAttention', 'GPU voxelization')) from error


def _verified(path: Path, spec: dict) -> bool:
    if not path.is_file() or path.stat().st_size != spec["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == spec["sha256"]


def download_verified_wheel(wheelhouse: Path, spec: dict | None = None) -> Path:
    """Stream one fixed asset into the runtime, checking size and SHA before publish.

    Existing valid wheels are reused. A bad cache is replaced only after the new
    transfer passes verification. No dependencies, credentials, or GPU calls.
    """
    spec = FLASH_ATTENTION_WHEEL if spec is None else spec
    name = spec["filename"]
    if not name or "/" in name or "\\" in name or Path(name).name != name:
        raise ValueError("Unsafe acceleration wheel filename")
    wheelhouse = Path(wheelhouse)
    wheelhouse.mkdir(parents=True, exist_ok=True)
    destination = wheelhouse / name
    if _verified(destination, spec):
        return destination
    temporary = destination.with_name(f".{name}.{uuid.uuid4().hex}.download.tmp")
    digest = hashlib.sha256()
    received = 0
    request = urllib.request.Request(spec["url"], headers={"User-Agent": "Hi3D-Workbench-runtime-setup"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
            advertised = response.headers.get("Content-Length")
            if advertised is not None and int(advertised) != spec["bytes"]:
                raise RuntimeError("Pinned acceleration wheel size mismatch (HTTP Content-Length)")
            while block := response.read(4 * 1024 * 1024):
                received += len(block)
                if received > spec["bytes"]:
                    raise RuntimeError("Pinned acceleration wheel exceeds expected size")
                digest.update(block)
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        if received != spec["bytes"] or digest.hexdigest() != spec["sha256"]:
            raise RuntimeError("Pinned acceleration wheel size/SHA256 mismatch")
        os.replace(temporary, destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)
