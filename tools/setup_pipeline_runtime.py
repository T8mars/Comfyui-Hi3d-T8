"""Install a private Windows runtime for coarse generation and mesh texturing."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "hunyuan": ("Tencent-Hunyuan/Hunyuan3D-2", "f8db63096c8282cb27354314d896feba5ba6ff8a"),
    "trellis2": ("microsoft/TRELLIS.2", "75fbf0183001ed9876c8dbb35de6b68552ee08bd"),
}
SOURCE_ARCHIVE_HASHES = {
    "Tencent-Hunyuan/Hunyuan3D-2": "1be140879646034a98941e494072ce0ab26918fdc2fb59210443d56d41bc5622",
    "microsoft/TRELLIS.2": "d7abd456f75585c5b4ce112a4abdb9bd3cc3e72b6074470a65e0bf3a89b3e06f",
}
WHEEL_REVISION = "b0e84c0f29857a444e8bcb5b9be2ea218e96efd0"
WHEELS = ["cumesh-1.0", "flex_gemm-0.0.1", "nvdiffrast-0.4.0", "nvdiffrec_render-0.0.0", "o_voxel-0.0.1"]
WHEEL_GIT_HASHES = {
    "cumesh-1.0": "68e2001152617bb0bbc3b9c3bd1e177ad6e0f149",
    "flex_gemm-0.0.1": "2affe0d0fc25df0c9e082690c878adb827bc287f",
    "nvdiffrast-0.4.0": "800fe38d603bf9424b81b3d260a67522bb74a83d",
    "nvdiffrec_render-0.0.0": "0477532bb6d664cc749697fc7daf7d35338478b3",
    "o_voxel-0.0.1": "2a535ee57cf1d3401e4826ee8750c31298f3ee9a",
}
CUDA_WHEELS = {
    "torch": {"version": "2.8.0+cu128", "url": "https://download-r2.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp312-cp312-win_amd64.whl", "sha256": "0ad925202387f4e7314302a1b4f8860fa824357f9b1466d7992bf276370ebcff"},
    "torchvision": {"version": "0.23.0+cu128", "url": "https://download-r2.pytorch.org/whl/cu128/torchvision-0.23.0%2Bcu128-cp312-cp312-win_amd64.whl", "sha256": "20fa9c7362a006776630b00b8a01919fedcf504a202b81358d32c5aef39956fe"},
}


def run(args):
    print("PIPELINE_SETUP", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True, cwd=ROOT)


def source(repo, revision, target):
    manifest = target / "source-manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        files = data.get("files")
        if (data.get("repo") == repo and data.get("revision") == revision
                and data.get("archive_sha256") == SOURCE_ARCHIVE_HASHES[repo]
                and isinstance(files, dict) and files
                and all((target / n).resolve().is_relative_to(target.resolve()) and (target / n).is_file()
                        and hashlib.sha256((target / n).read_bytes()).hexdigest() == h for n, h in files.items())):
            return
        raise RuntimeError(f"Source directory changed: {target}. Preserve changes and use a new runtime directory.")
    print(f"PIPELINE_SETUP source {repo}@{revision}", flush=True)
    payload = urllib.request.urlopen(f"https://codeload.github.com/{repo}/zip/{revision}", timeout=180).read()
    if hashlib.sha256(payload).hexdigest() != SOURCE_ARCHIVE_HASHES[repo]:
        raise RuntimeError(f"Upstream source archive hash mismatch: {repo}")
    hashes = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for entry in archive.infolist():
            relative = Path(*Path(entry.filename).parts[1:])
            if not relative.parts or entry.is_dir():
                continue
            # The demos/media/training data are not part of this inference runtime.
            if relative.parts[0] in {"assets", ".github", "data_toolkit", "docker"}:
                continue
            destination = (target / relative).resolve()
            if not destination.is_relative_to(target.resolve()):
                raise ValueError("Unsafe upstream archive path")
            destination.parent.mkdir(parents=True, exist_ok=True)
            body = archive.read(entry)
            destination.write_bytes(body)
            hashes[relative.as_posix()] = hashlib.sha256(body).hexdigest()
    manifest.write_text(json.dumps({"repo": repo, "revision": revision, "archive_sha256": hashlib.sha256(payload).hexdigest(), "files": hashes}, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, default=ROOT / ".pipelineenv")
    parser.add_argument("--sources-only", action="store_true")
    parser.add_argument("--wheels-only", action="store_true", help="Verify/install only the small native extension wheels; does not mark runtime ready")
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    if runtime in {(ROOT / ".venv").resolve(), (ROOT / ".appenv").resolve()}:
        raise ValueError("Existing Hi3D and application runtimes must not be modified")
    runtime.mkdir(parents=True, exist_ok=True)
    python = runtime / "Scripts/python.exe"
    if not python.is_file():
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError("Run setup with Python 3.12; Windows extension wheels are pinned to cp312")
        run([sys.executable, "-m", "venv", runtime])
    for name, (repo, revision) in SOURCES.items():
        source(repo, revision, runtime / "sources" / name)
    if args.sources_only:
        return
    if not args.wheels_only:
        found = subprocess.run([str(python), "-c", "import importlib.metadata as m,json; print(json.dumps({d.metadata['Name'].lower():d.version for d in m.distributions()}))"], capture_output=True, text=True, check=True)
        versions = json.loads(found.stdout)
        cuda_requirements = [f"{name} @ {info['url']}#sha256={info['sha256']}" for name, info in CUDA_WHEELS.items() if versions.get(name) != info["version"]]
        if cuda_requirements:
            run([python, "-m", "pip", "install", *cuda_requirements, "-c", ROOT / "requirements-pipeline-windows.txt"])
        run([python, "-m", "pip", "install", "-r", ROOT / "requirements-pipeline-windows.txt"])
    for stem in WHEELS:
        filename = f"{stem}-cp312-cp312-win_amd64.whl"
        url = f"https://raw.githubusercontent.com/visualbruno/ComfyUI-Trellis2/{WHEEL_REVISION}/wheels/Windows/Torch280/{filename}"
        wheel = runtime / "wheelhouse" / filename
        wheel.parent.mkdir(exist_ok=True)
        payload = wheel.read_bytes() if wheel.is_file() else urllib.request.urlopen(url, timeout=120).read()
        git_hash = hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()
        if git_hash != WHEEL_GIT_HASHES[stem]:
            raise RuntimeError(f"Pinned Windows wheel hash mismatch: {filename}")
        wheel.write_bytes(payload)
        run([python, "-m", "pip", "install", "--no-deps", wheel])
    if args.wheels_only:
        print("PIPELINE_SETUP native wheels installed; remaining dependencies still required", flush=True)
        return
    # Import checks are deliberately CPU-only; full GPU inference is a separate validation.
    run([python, "-m", "pip", "check"])
    probe = ("import sys; from pathlib import Path; "
             "import torch,trimesh,transformers,scipy,cumesh,flex_gemm,nvdiffrast.torch,o_voxel; "
             "assert torch.__version__ == '2.8.0+cu128'; "
             "sys.path[:0]=[str(Path(sys.prefix)/'sources'/'hunyuan'),str(Path(sys.prefix)/'sources'/'trellis2')]; "
             "from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline; "
             "from trellis2.pipelines import Trellis2TexturingPipeline")
    run([python, "-c", probe])
    (runtime / "pipeline-runtime.json").write_text(json.dumps({"ready": True, "python": "3.12", "torch": "2.8.0+cu128", "sources": SOURCES, "cuda_wheels": CUDA_WHEELS, "windows_wheels_revision": WHEEL_REVISION, "gpu_validation": "not_run"}, indent=2), encoding="utf-8")
    print("PIPELINE_SETUP ready", flush=True)


if __name__ == "__main__":
    main()
