"""Optional accelerator installation checks use local bytes and fake subprocesses."""
import hashlib
import io
import json
import subprocess
from pathlib import Path

import pytest

from hi3d_workbench.engines import acceleration_wheels as wheels
from tools import setup_pipeline_runtime as setup


@pytest.fixture
def environment():
    return {"platform": "win32", "python": [3, 12], "implementation": "CPython",
            "machine": "AMD64", "pointer_bits": 64, "torch": "2.8.0+cu128", "cuda": "12.8"}


@pytest.mark.parametrize("field,value", [
    ("platform", "linux"), ("python", [3, 10]), ("torch", "2.11.0+cu128"),
    ("torch", "2.8.0+cpu"), ("cuda", "13.0"), ("implementation", "PyPy"),
    ("machine", "ARM64"), ("pointer_bits", 32),
])
def test_incompatible_native_abi_is_rejected(environment, field, value):
    environment[field] = value
    with pytest.raises(RuntimeError, match="ABI mismatch"):
        wheels.validate_flash_environment(environment)


def test_matching_abi_accepts_cpython_x64(environment):
    wheels.validate_flash_environment(environment)


@pytest.fixture
def artifact():
    payload = b"fixture wheel contents; never an executable package"
    spec = {"filename": "fixture.whl", "url": "https://fixture.invalid/wheel",
            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    return payload, spec


class Response(io.BytesIO):
    def __init__(self, payload, length=None):
        super().__init__(payload)
        self.headers = {} if length is None else {"Content-Length": str(length)}

    def read(self, count=-1):
        assert count > 0, "A wheel must be streamed, not buffered in one read"
        return super().read(count)


def test_download_is_verified_streamed_and_valid_cache_reused(tmp_path, artifact, monkeypatch):
    payload, spec = artifact
    calls = []

    def transfer(request, timeout):
        calls.append((request.full_url, timeout))
        return Response(payload, len(payload))

    monkeypatch.setattr(wheels.urllib.request, "urlopen", transfer)
    target = wheels.download_verified_wheel(tmp_path / "wheelhouse", spec)
    assert target.read_bytes() == payload
    assert wheels.download_verified_wheel(target.parent, spec) == target
    assert calls == [(spec["url"], 120)]
    assert not list(target.parent.glob("*.tmp"))


@pytest.mark.parametrize("bad_payload,header", [
    (b"wrong bytes", None), (b"", None), (b"x" * 200, None),
    (b"correct contents do not matter for wrong advertised size", 999),
])
def test_bad_transfer_is_not_published_or_allowed_to_replace_cache(tmp_path, artifact, monkeypatch,
                                                                  bad_payload, header):
    _, spec = artifact
    existing = tmp_path / spec["filename"]
    existing.write_bytes(b"untrusted old cache")
    monkeypatch.setattr(wheels.urllib.request, "urlopen", lambda *a, **k: Response(bad_payload, header))
    with pytest.raises(RuntimeError, match="size|SHA256"):
        wheels.download_verified_wheel(tmp_path, spec)
    assert existing.read_bytes() == b"untrusted old cache"
    assert not list(tmp_path.glob("*.tmp"))


def test_corrupt_cache_is_repaired_only_after_verified_transfer(tmp_path, artifact, monkeypatch):
    payload, spec = artifact
    (tmp_path / spec["filename"]).write_bytes(b"x" * len(payload))
    monkeypatch.setattr(wheels.urllib.request, "urlopen", lambda *a, **k: Response(payload))
    assert wheels.download_verified_wheel(tmp_path, spec).read_bytes() == payload


def test_same_size_wrong_hash_never_publishes(tmp_path, artifact, monkeypatch):
    payload, spec = artifact
    monkeypatch.setattr(wheels.urllib.request, "urlopen", lambda *a, **k: Response(b"x" * len(payload)))
    with pytest.raises(RuntimeError, match="SHA256"):
        wheels.download_verified_wheel(tmp_path, spec)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name", ["../outside.whl", "..\\outside.whl", "/outside.whl"])
def test_download_cannot_escape_runtime_wheelhouse(tmp_path, artifact, name):
    _, spec = artifact
    with pytest.raises(ValueError, match="filename"):
        wheels.download_verified_wheel(tmp_path, {**spec, "filename": name})


def test_optional_setup_checks_abi_before_network_or_pip_and_keeps_old_readiness(tmp_path, environment, monkeypatch):
    manifest = {"ready": True, "gpu_validation": "historic_success", "custom_field": {"keep": 1}}
    (tmp_path / "pipeline-runtime.json").write_text(json.dumps(manifest))
    environment["torch"] = "2.11.0+cu128"
    monkeypatch.setattr(setup, "probe_attention_environment", lambda *a: environment)
    monkeypatch.setattr(setup, "download_verified_wheel", lambda *a: pytest.fail("must not download incompatible ABI"))
    monkeypatch.setattr(setup, "run", lambda *a: pytest.fail("must not invoke pip for incompatible ABI"))
    receipt = setup.install_optional_attention(tmp_path / "python.exe", tmp_path)
    assert receipt["backend"] == "sdpa" and receipt["status"] == "unavailable"
    assert "ABI mismatch" in receipt["reason"] and not receipt["wheel_verified"]
    current = json.loads((tmp_path / "pipeline-runtime.json").read_text())
    for field, value in manifest.items():
        assert current[field] == value
    assert current["attention_acceleration"] == receipt


def test_optional_success_installs_local_wheel_without_dependencies_or_torch_changes(tmp_path, environment, monkeypatch):
    commands = []
    cached = tmp_path / "wheelhouse" / wheels.FLASH_ATTENTION_WHEEL["filename"]
    monkeypatch.setattr(setup, "probe_attention_environment", lambda *a: environment)
    monkeypatch.setattr(setup, "download_verified_wheel", lambda directory: cached)
    monkeypatch.setattr(setup, "run", lambda args: commands.append(args))
    monkeypatch.setattr(setup, "probe_flash_import", lambda *a: {"import_validation": "passed", "version": "2.8.3"})
    receipt = setup.install_optional_attention(tmp_path / "python.exe", tmp_path)
    assert receipt["backend"] == "flash_attn" and receipt["status"] == "installed"
    assert receipt["wheel_verified"] and receipt["import_validation"] == "passed"
    assert receipt["gpu_validation"] == "not_run"
    assert len(commands) == 1 and commands[0][-1] == cached
    assert "--no-deps" in commands[0] and "--no-index" in commands[0]
    assert "--no-cache-dir" in commands[0] and "--upgrade" not in commands[0]
    assert not (tmp_path / "pipeline-runtime.json").exists(), "optional install cannot mark an unprepared runtime ready"
    assert json.loads((tmp_path / "attention-acceleration.json").read_text()) == receipt


def test_failed_extension_import_is_explicit_sdpa_fallback(tmp_path, environment, monkeypatch, capsys):
    monkeypatch.setattr(setup, "probe_attention_environment", lambda *a: environment)
    monkeypatch.setattr(setup, "download_verified_wheel", lambda *a: tmp_path / "fixture.whl")
    monkeypatch.setattr(setup, "run", lambda *a: None)

    def incompatible(*args):
        raise subprocess.CalledProcessError(1, ["python", "-c"], stderr="DLL load failed")

    monkeypatch.setattr(setup, "probe_flash_import", incompatible)
    receipt = setup.install_optional_attention(tmp_path / "python.exe", tmp_path)
    assert receipt["backend"] == "sdpa" and receipt["import_validation"] == "failed"
    assert receipt["wheel_verified"] and "DLL load failed" in receipt["reason"]
    assert "SDPA fallback retained" in capsys.readouterr().out


def test_download_failure_keeps_sdpa_optional(tmp_path, environment, monkeypatch):
    monkeypatch.setattr(setup, "probe_attention_environment", lambda *a: environment)

    def offline(*args):
        raise OSError("offline fixture")

    monkeypatch.setattr(setup, "download_verified_wheel", offline)
    monkeypatch.setattr(setup, "run", lambda *a: pytest.fail("unverified wheel must never reach pip"))
    receipt = setup.install_optional_attention(tmp_path / "python.exe", tmp_path)
    assert receipt["status"] == "unavailable" and "offline fixture" in receipt["reason"]


def test_explicit_skip_never_probes_downloads_or_installs(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "probe_attention_environment", lambda *a: pytest.fail("skip cannot probe"))
    monkeypatch.setattr(setup, "run", lambda *a: pytest.fail("skip cannot install"))
    receipt = setup.install_optional_attention(tmp_path / "python.exe", tmp_path, skip=True)
    assert receipt["status"] == "skipped" and not receipt["wheel_verified"]


@pytest.mark.parametrize("layout", ["python.exe", "Scripts/python.exe"])
def test_attention_only_supports_portable_and_venv_without_base_setup(tmp_path, monkeypatch, layout):
    python = tmp_path / layout
    python.parent.mkdir(parents=True, exist_ok=True)
    python.touch()
    installs = []
    monkeypatch.setattr(setup, "install_optional_attention", lambda *args: installs.append(args))
    monkeypatch.setattr(setup, "source", lambda *a: pytest.fail("attention-only cannot refresh sources"))
    monkeypatch.setattr(setup, "run", lambda *a: pytest.fail("attention-only cannot install base dependencies"))
    setup.main(["--runtime", str(tmp_path), "--attention-only"])
    assert installs == [(python, tmp_path)]


def test_attention_only_never_creates_runtime(tmp_path):
    target = tmp_path / "missing-runtime"
    with pytest.raises(RuntimeError, match="existing runtime"):
        setup.main(["--runtime", str(target), "--attention-only"])
    assert not target.exists()


@pytest.fixture
def uv_environment(environment):
    return {**environment, "numpy": "1.26.4"}


@pytest.mark.parametrize("field,value", [
    ("platform", "linux"), ("python", [3, 10]), ("numpy", "2.2.0"),
    ("numpy", "1.26.3"), ("implementation", "PyPy"), ("pointer_bits", 32), ("machine", "ARM64"),
])
def test_uv_abi_rejects_unvalidated_runtime_before_modifying_numpy(uv_environment, field, value):
    uv_environment[field] = value
    with pytest.raises(RuntimeError, match="UV acceleration ABI mismatch"):
        wheels.validate_uv_environment(uv_environment)


def test_uv_abi_is_independent_of_torch_and_cuda(uv_environment):
    uv_environment.update(torch="unrelated-version", cuda=None)
    wheels.validate_uv_environment(uv_environment)


def test_uv_setup_verifies_both_wheels_then_installs_llvmlite_before_numba(tmp_path, uv_environment, monkeypatch):
    operations = []
    manifest = {"ready": True, "gpu_validation": "old_receipt", "attention_acceleration": {"keep": "native"}}
    (tmp_path / "pipeline-runtime.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(setup, "probe_uv_environment", lambda *a: uv_environment)

    def download(directory, spec):
        operations.append(("verify", spec["package"]))
        return directory / spec["filename"]

    monkeypatch.setattr(setup, "download_verified_wheel", download)
    monkeypatch.setattr(setup, "run", lambda args: operations.append(("install", args)))
    monkeypatch.setattr(setup, "probe_numba_import", lambda *a: {"import_validation": "passed", "numba": "0.61.2", "llvmlite": "0.44.0"})
    result = setup.install_optional_uv(tmp_path / "python.exe", tmp_path)
    assert operations[:2] == [("verify", "llvmlite"), ("verify", "numba")]
    assert [call[1][-1].name.split("-")[0] for call in operations[2:]] == ["llvmlite", "numba"]
    for _, command in operations[2:]:
        assert "--no-deps" in command and "--no-index" in command
        assert "--force-reinstall" in command and "--no-cache-dir" in command
        assert "--upgrade" not in command
    assert result["status"] == "installed" and result["backend"] == "numba"
    assert result["wheels_verified"] and result["import_validation"] == "passed"
    assert all(result[field] == "not_run" for field in ("jit_validation", "speed_validation", "gpu_validation"))
    current = json.loads((tmp_path / "pipeline-runtime.json").read_text())
    assert current == {**manifest, "uv_acceleration": result}
    assert json.loads((tmp_path / "uv-acceleration.json").read_text()) == result


def test_uv_wrong_numpy_never_downloads_or_changes_packages(tmp_path, uv_environment, monkeypatch):
    uv_environment["numpy"] = "2.2.0"
    monkeypatch.setattr(setup, "probe_uv_environment", lambda *a: uv_environment)
    monkeypatch.setattr(setup, "download_verified_wheel", lambda *a: pytest.fail("bad NumPy ABI cannot download"))
    monkeypatch.setattr(setup, "run", lambda *a: pytest.fail("NumPy must never be upgraded/downgraded"))
    result = setup.install_optional_uv(tmp_path / "python.exe", tmp_path)
    assert result["backend"] == "torch" and "numpy=" in result["reason"]
    assert not result["wheels_verified"] and not (tmp_path / "pipeline-runtime.json").exists()


def test_uv_second_wheel_verification_failure_cannot_install_first(tmp_path, uv_environment, monkeypatch):
    monkeypatch.setattr(setup, "probe_uv_environment", lambda *a: uv_environment)

    def download(directory, spec):
        if spec["package"] == "numba":
            raise RuntimeError("fixture hash mismatch")
        return directory / spec["filename"]

    monkeypatch.setattr(setup, "download_verified_wheel", download)
    monkeypatch.setattr(setup, "run", lambda *a: pytest.fail("verify all assets before pip"))
    result = setup.install_optional_uv(tmp_path / "python.exe", tmp_path)
    assert result["backend"] == "torch" and not result["wheels_verified"]
    assert "hash mismatch" in result["reason"]


def test_uv_failed_import_retains_torch_fallback_with_diagnostic(tmp_path, uv_environment, monkeypatch, capsys):
    monkeypatch.setattr(setup, "probe_uv_environment", lambda *a: uv_environment)
    monkeypatch.setattr(setup, "download_verified_wheel", lambda directory, spec: directory / spec["filename"])
    monkeypatch.setattr(setup, "run", lambda *a: None)

    def failed(*args):
        raise subprocess.CalledProcessError(1, ["python", "-c"], stderr="Numba fixture DLL error")

    monkeypatch.setattr(setup, "probe_numba_import", failed)
    result = setup.install_optional_uv(tmp_path / "python.exe", tmp_path)
    assert result["backend"] == "torch" and result["status"] == "unavailable"
    assert result["wheels_verified"] and result["import_validation"] == "failed"
    assert "fixture DLL error" in result["reason"]
    assert "Torch fallback retained" in capsys.readouterr().out


def test_uv_skip_never_touches_native_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "probe_uv_environment", lambda *a: pytest.fail("explicit skip cannot probe"))
    monkeypatch.setattr(setup, "run", lambda *a: pytest.fail("explicit skip cannot install"))
    result = setup.install_optional_uv(tmp_path / "python.exe", tmp_path, skip=True)
    assert result["status"] == "skipped" and result["backend"] == "torch"


@pytest.mark.parametrize("layout", ["python.exe", "Scripts/python.exe"])
def test_uv_only_never_runs_base_or_attention_install(tmp_path, monkeypatch, layout):
    python = tmp_path / layout
    python.parent.mkdir(parents=True, exist_ok=True)
    python.touch()
    installs = []
    monkeypatch.setattr(setup, "install_optional_uv", lambda *args: installs.append(args))
    monkeypatch.setattr(setup, "source", lambda *a: pytest.fail("UV-only cannot change upstream sources"))
    monkeypatch.setattr(setup, "run", lambda *a: pytest.fail("UV-only cannot run base setup"))
    monkeypatch.setattr(setup, "install_optional_attention", lambda *a: pytest.fail("UV-only cannot install attention"))
    setup.main(["--runtime", str(tmp_path), "--uv-only"])
    assert installs == [(python, tmp_path)]


def test_uv_only_missing_runtime_is_never_created(tmp_path):
    target = tmp_path / "missing-runtime"
    with pytest.raises(RuntimeError, match="existing runtime"):
        setup.main(["--runtime", str(target), "--uv-only"])
    assert not target.exists()


def test_uv_only_cannot_be_combined_with_skip(tmp_path):
    with pytest.raises(SystemExit):
        setup.main(["--runtime", str(tmp_path), "--uv-only", "--skip-uv-acceleration"])
