"""Install a private Windows runtime for coarse generation and mesh texturing."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from hi3d_workbench.engines.acceleration_wheels import (  # noqa: E402
    FLASH_ATTENTION_WHEEL, UV_ACCELERATION_WHEELS, download_verified_wheel,
    validate_flash_environment, validate_uv_environment,
)
SOURCES = {
    "hunyuan": ("Tencent-Hunyuan/Hunyuan3D-2", "f8db63096c8282cb27354314d896feba5ba6ff8a"),
    "trellis2": ("microsoft/TRELLIS.2", "75fbf0183001ed9876c8dbb35de6b68552ee08bd"),
    "hunyuan21": ("Tencent-Hunyuan/Hunyuan3D-2.1", "82920d643c0dc2f7bfd7255f45f62d386edfe60c"),
    "ultratex_bridge": ("visualbruno/ComfyUI-UltraTex", "8cc29849dd076ddec0fe5a1ec6207945dedbbbf1"),
    "comfy_core": ("Comfy-Org/ComfyUI", "7cebca52227c63b4180ceed448cbd7ad22903b6d"),
}
SOURCE_ARCHIVE_HASHES = {
    "Tencent-Hunyuan/Hunyuan3D-2": "1be140879646034a98941e494072ce0ab26918fdc2fb59210443d56d41bc5622",
    "microsoft/TRELLIS.2": "d7abd456f75585c5b4ce112a4abdb9bd3cc3e72b6074470a65e0bf3a89b3e06f",
    "Tencent-Hunyuan/Hunyuan3D-2.1": "1c560ba93e2471305276f36170a7e3c3857f39b14313790533dbc06ab01ed2a3",
    "visualbruno/ComfyUI-UltraTex": "2316ed9da49ab435d98fac305d68f155095dc8084315211386e34fa2f42d7235",
    "Comfy-Org/ComfyUI": "d6928c3c1d71577d99e5ce9df1eb2cd9335aa0aa321deac08c9d73fe9ff9b6cc",
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


def runtime_python(runtime):
    """Support both the development venv and the shipped embedded runtime."""
    for candidate in (runtime / "Scripts" / "python.exe", runtime / "python.exe"):
        if candidate.is_file():
            return candidate
    return None


def _probe_json(python, code):
    result = subprocess.run([str(python), "-c", code], capture_output=True,
                            text=True, encoding="utf-8", errors="replace", check=True, cwd=ROOT)
    # Some installed packages print a banner when importing Torch/extensions.
    return json.loads(result.stdout.strip().splitlines()[-1])


def probe_attention_environment(python):
    return _probe_json(python,
        "import json,platform,struct,sys,torch; "
        "print(json.dumps({'platform':sys.platform,'python':list(sys.version_info[:2]),"
        "'implementation':platform.python_implementation(),'machine':platform.machine(),"
        "'pointer_bits':struct.calcsize('P')*8,'torch':torch.__version__,'cuda':torch.version.cuda}))")


def probe_flash_import(python):
    """Check extension ABI and required functions without allocating CUDA tensors."""
    return _probe_json(python,
        "import json,torch,flash_attn; "
        "from flash_attn import flash_attn_varlen_qkvpacked_func,flash_attn_varlen_kvpacked_func,flash_attn_varlen_func; "
        "assert flash_attn.__version__ == '2.8.3', flash_attn.__version__; "
        "print(json.dumps({'version':flash_attn.__version__,'import_validation':'passed','gpu_validation':'not_run'}))")


def probe_uv_environment(python):
    # Query NumPy's installed version before downloading; no Torch/CUDA import.
    return _probe_json(python,
        "import importlib.metadata as m,json,platform,struct,sys; "
        "print(json.dumps({'platform':sys.platform,'python':list(sys.version_info[:2]),"
        "'implementation':platform.python_implementation(),'machine':platform.machine(),"
        "'pointer_bits':struct.calcsize('P')*8,'numpy':m.version('numpy')}))")


def probe_numba_import(python):
    """Check the two packages and NumPy; do not JIT-compile or run the packer."""
    return _probe_json(python,
        "import json,numpy,numba,llvmlite; "
        "assert numpy.__version__ == '1.26.4', numpy.__version__; "
        "assert numba.__version__ == '0.61.2', numba.__version__; "
        "assert llvmlite.__version__ == '0.44.0', llvmlite.__version__; "
        "print(json.dumps({'numpy':numpy.__version__,'numba':numba.__version__,"
        "'llvmlite':llvmlite.__version__,'import_validation':'passed'}))")


def _write_acceleration_receipt(runtime, receipt, filename, field):
    path = runtime / filename
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    manifest = runtime / "pipeline-runtime.json"
    if manifest.is_file():
        try:
            value = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Runtime manifest must be an object")
            value[field] = receipt
            temporary = manifest.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, manifest)
        except (OSError, ValueError) as exc:
            print(f"PIPELINE_SETUP {filename} saved; existing runtime manifest unchanged: {exc}", flush=True)


def _write_attention_receipt(runtime, receipt):
    _write_acceleration_receipt(runtime, receipt, "attention-acceleration.json", "attention_acceleration")


def install_optional_attention(python, runtime, *, skip=False):
    """An unavailable accelerator must never invalidate a working SDPA runtime."""
    receipt = {"schema": 1, "status": "skipped" if skip else "unavailable", "backend": "sdpa",
               "wheel": dict(FLASH_ATTENTION_WHEEL), "wheel_verified": False,
               "import_validation": "not_run", "gpu_validation": "not_run",
               "updated_utc": datetime.now(timezone.utc).isoformat()}
    if skip:
        receipt["reason"] = "Native attention installation explicitly skipped; using existing SDPA compatibility path"
    else:
        try:
            receipt["environment"] = probe_attention_environment(python)
            validate_flash_environment(receipt["environment"])
            print("PIPELINE_SETUP optional FlashAttention: checking pinned Windows wheel (Torch remains unchanged)", flush=True)
            wheel = download_verified_wheel(runtime / "wheelhouse")
            receipt["wheel_verified"] = True
            run([python, "-m", "pip", "install", "--no-deps", "--no-index", "--force-reinstall",
                 "--disable-pip-version-check", "--no-cache-dir", wheel])
            receipt["import_validation"] = "failed"
            result = probe_flash_import(python)
            if result.get("import_validation") != "passed":
                raise RuntimeError("Native FlashAttention import validation did not pass")
            receipt.update(status="installed", backend="flash_attn", import_validation="passed",
                           reason="Pinned extension imported; GPU kernels and per-model quality still require separate validation")
            print("PIPELINE_SETUP FlashAttention imported; GPU validation not run", flush=True)
        except Exception as exc:
            receipt["reason"] = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
                receipt["reason"] += ": " + str(exc.stderr).strip()[-2000:]
            print(f"PIPELINE_SETUP optional FlashAttention unavailable; SDPA fallback retained: {receipt['reason']}", flush=True)
    _write_attention_receipt(runtime, receipt)
    return receipt


def install_optional_uv(python, runtime, *, skip=False):
    """Install the upstream optional CPU packer without dependency resolution."""
    receipt = {"schema": 1, "status": "skipped" if skip else "unavailable", "backend": "torch",
               "wheels": [dict(spec) for spec in UV_ACCELERATION_WHEELS], "wheels_verified": False,
               "import_validation": "not_run", "jit_validation": "not_run",
               "speed_validation": "not_run", "gpu_validation": "not_run",
               "updated_utc": datetime.now(timezone.utc).isoformat()}
    if skip:
        receipt["reason"] = "Optional Numba installation explicitly skipped; upstream Torch UV packer remains available"
    else:
        try:
            receipt["environment"] = probe_uv_environment(python)
            validate_uv_environment(receipt["environment"])
            print("PIPELINE_SETUP optional Numba UV packer: checking pinned wheels (NumPy/Torch remain unchanged)", flush=True)
            # Verify both assets before the first installation. Dependency order
            # is explicit so pip never resolves or upgrades NumPy/Torch/LLVM.
            cached = [download_verified_wheel(runtime / "wheelhouse", spec) for spec in UV_ACCELERATION_WHEELS]
            receipt["wheels_verified"] = True
            for wheel in cached:
                run([python, "-m", "pip", "install", "--no-deps", "--no-index", "--force-reinstall",
                     "--disable-pip-version-check", "--no-cache-dir", wheel])
            receipt["import_validation"] = "failed"
            result = probe_numba_import(python)
            if result.get("import_validation") != "passed":
                raise RuntimeError("Optional Numba import validation did not pass")
            receipt.update(status="installed", backend="numba", import_validation="passed", versions=result,
                           reason="Pinned packages imported; first-use JIT, packer correctness and speed require separate validation")
            print("PIPELINE_SETUP Numba imported; packer JIT and speed validation not run", flush=True)
        except Exception as exc:
            receipt["reason"] = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
                receipt["reason"] += ": " + str(exc.stderr).strip()[-2000:]
            print(f"PIPELINE_SETUP optional Numba UV packer unavailable; Torch fallback retained: {receipt['reason']}", flush=True)
    _write_acceleration_receipt(runtime, receipt, "uv-acceleration.json", "uv_acceleration")
    return receipt


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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, default=ROOT / ".pipelineenv")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sources-only", action="store_true")
    mode.add_argument("--wheels-only", action="store_true", help="Verify/install only the small native extension wheels; does not mark runtime ready")
    mode.add_argument("--attention-only", action="store_true", help="Add optional pinned FlashAttention to an existing compatible runtime; never changes Torch or runtime readiness")
    mode.add_argument("--uv-only", action="store_true", help="Add optional pinned Numba/llvmlite to an existing CP312 Windows runtime; never changes NumPy/Torch or runtime readiness")
    parser.add_argument("--skip-native-attention", action="store_true", help="Keep SDPA; skip the optional FlashAttention wheel")
    parser.add_argument("--skip-uv-acceleration", action="store_true", help="Keep the upstream Torch UV packer; skip optional Numba/llvmlite")
    args = parser.parse_args(argv)
    if args.attention_only and args.skip_native_attention:
        parser.error("--attention-only cannot be combined with --skip-native-attention")
    if args.uv_only and args.skip_uv_acceleration:
        parser.error("--uv-only cannot be combined with --skip-uv-acceleration")
    runtime = args.runtime.resolve()
    if runtime in {(ROOT / ".venv").resolve(), (ROOT / ".appenv").resolve()}:
        raise ValueError("Existing Hi3D and application runtimes must not be modified")
    python = runtime_python(runtime)
    if args.attention_only or args.uv_only:
        if python is None:
            mode_name = "--attention-only" if args.attention_only else "--uv-only"
            raise RuntimeError(f"{mode_name} requires an existing runtime with python.exe or Scripts/python.exe")
        if args.attention_only:
            install_optional_attention(python, runtime)
        else:
            install_optional_uv(python, runtime)
        return
    runtime.mkdir(parents=True, exist_ok=True)
    if python is None:
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError("Run setup with Python 3.12; Windows extension wheels are pinned to cp312")
        run([sys.executable, "-m", "venv", runtime])
        python = runtime / "Scripts" / "python.exe"
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
    attention_receipt = install_optional_attention(python, runtime, skip=args.skip_native_attention)
    uv_receipt = install_optional_uv(python, runtime, skip=args.skip_uv_acceleration)
    # Import checks are deliberately CPU-only; full GPU inference is a separate validation.
    run([python, "-m", "pip", "check"])
    probe = ("import sys; from pathlib import Path; "
             "import torch,trimesh,transformers,scipy,cumesh,flex_gemm,nvdiffrast.torch,o_voxel; "
             "assert torch.__version__ == '2.8.0+cu128'; "
             "sys.path[:0]=[str(Path(sys.prefix)/'sources'/'hunyuan'),str(Path(sys.prefix)/'sources'/'trellis2')]; "
             "from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline; "
             "from trellis2.pipelines import Trellis2TexturingPipeline; "
             "sys.path.insert(0,str(Path(sys.prefix)/'sources'/'hunyuan21'/'hy3dshape')); "
             "from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline; "
             "sys.path.insert(0,str(Path(sys.prefix)/'sources'/'hunyuan21'/'hy3dpaint')); "
             "from hi3d_workbench.engines.hunyuan_texture import pipeline_classes; "
             "pipeline_classes(Path(sys.prefix)/'sources'/'hunyuan21'); import spandrel; "
             "sys.path[:0]=[str(Path(sys.prefix)/'sources'/'comfy_core'),str(Path(sys.prefix)/'sources'/'ultratex_bridge')]; "
             "import comfy.cli_args; comfy.cli_args.args.cpu=True; import comfy.sd; "
             "from ultratex_core import prep,sampling,bake")
    run([python, "-c", probe])
    (runtime / "pipeline-runtime.json").write_text(json.dumps({"ready": True, "features": ["hunyuan21", "trellis_geometry", "hunyuan_paint21", "ultratex"], "python": "3.12", "torch": "2.8.0+cu128", "sources": SOURCES, "cuda_wheels": CUDA_WHEELS, "windows_wheels_revision": WHEEL_REVISION, "attention_acceleration": attention_receipt, "uv_acceleration": uv_receipt, "gpu_validation": "not_run"}, indent=2), encoding="utf-8")
    print("PIPELINE_SETUP ready", flush=True)


if __name__ == "__main__":
    main()
