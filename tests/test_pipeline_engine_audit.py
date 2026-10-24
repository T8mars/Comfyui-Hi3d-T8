"""CPU-only regressions from the post-v0.3 engine audit.

Every artifact is temporary. Transfers are local fixtures, never Hub requests.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from test_asset_pipeline import tiny_extensions
from hi3d_workbench import common, pipeline_models, pipeline_worker
from hi3d_workbench.engines import run_stage


def test_extension_file_changed_during_verification_is_never_marked_ready(tiny_extensions, monkeypatch):
    target = tiny_extensions / 'coarse' / 'weights.bin'
    original_hash = pipeline_models._hash

    def changed_after_hash(path, git_blob=False):
        result = original_hash(path, git_blob)
        if path == target:
            target.write_bytes(b'x' * target.stat().st_size)
        return result

    monkeypatch.setattr(pipeline_models, '_hash', changed_after_hash)
    result = pipeline_models.verify()
    assert 'coarse/weights.bin' in result['verification_errors']
    assert not result['coarse_ready']
    assert not common.read_json(tiny_extensions / 'manifest.json')['components']['coarse']['files']['weights.bin']['verified']


@pytest.mark.parametrize('lfs', [False, True])
def test_extension_download_repairs_corrupt_small_hub_cache(tiny_extensions, monkeypatch, lfs):
    import huggingface_hub
    payload = b'correct pinned extension bytes'
    sha = hashlib.sha256(payload).hexdigest()
    blob = hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()
    target = tiny_extensions / 'texture_encoder' / 'weights.bin'
    target.write_bytes(b'corrupt local Hub cache')
    calls = []

    class Api:
        def model_info(self, repo, **kwargs):
            return SimpleNamespace(siblings=[SimpleNamespace(rfilename='weights.bin', size=len(payload),
                lfs=SimpleNamespace(sha256=sha) if lfs else None, blob_id=blob)])

    def transfer(repo, name, local_dir, force_download=False, **kwargs):
        path = Path(local_dir) / name
        calls.append(force_download)
        if force_download:
            path.write_bytes(payload)
        return str(path)

    monkeypatch.setattr(huggingface_hub, 'HfApi', Api)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', transfer)
    monkeypatch.setattr(pipeline_models, '_STATE', {'state': 'idle'})
    result = pipeline_models.download('encoder')
    assert calls == [False, True]
    assert target.read_bytes() == payload
    assert result['components']['texture_encoder']['ready']


def test_extension_download_never_binds_hash_to_a_changed_file(tiny_extensions, monkeypatch):
    import huggingface_hub
    payload = b'correct pinned extension bytes'
    sha = hashlib.sha256(payload).hexdigest()
    target = tiny_extensions / 'texture_encoder' / 'weights.bin'
    before = (tiny_extensions / 'manifest.json').read_bytes()
    original_hash = pipeline_models._hash
    calls = []

    class Api:
        def model_info(self, repo, **kwargs):
            return SimpleNamespace(siblings=[SimpleNamespace(rfilename='weights.bin', size=len(payload),
                lfs=SimpleNamespace(sha256=sha))])

    def transfer(repo, name, local_dir, force_download=False, **kwargs):
        calls.append(force_download)
        target.write_bytes(payload)
        return str(target)

    def changed_after_hash(path, git_blob=False):
        result = original_hash(path, git_blob)
        if path == target:
            target.write_bytes(b'x' * target.stat().st_size)
        return result

    monkeypatch.setattr(huggingface_hub, 'HfApi', Api)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', transfer)
    monkeypatch.setattr(pipeline_models, '_hash', changed_after_hash)
    monkeypatch.setattr(pipeline_models, '_STATE', {'state': 'idle'})
    with pytest.raises(RuntimeError, match='内容校验失败'):
        pipeline_models.download('encoder')
    assert calls == [False, True]
    assert (tiny_extensions / 'manifest.json').read_bytes() == before
    state = pipeline_models.status()
    assert state['download']['state'] == 'failed'
    assert not state['components']['texture_encoder']['ready']


@pytest.mark.parametrize('field', ['repo', 'revision'])
def test_extension_verify_reports_wrong_pinned_provenance(tiny_extensions, field):
    path = tiny_extensions / 'manifest.json'
    manifest = common.read_json(path)
    manifest['components']['coarse'][field] = 'different-value'
    common.atomic_json(path, manifest)
    result = pipeline_models.verify()
    assert f'coarse/manifest:{field}' in result['verification_errors']
    assert not result['coarse_ready']


def test_ready_extensions_do_not_claim_they_have_never_been_downloaded(tiny_extensions, monkeypatch):
    monkeypatch.setattr(pipeline_models, '_STATE', {'state': 'idle', 'message': '扩展模型尚未下载'})
    state = pipeline_models.status()
    assert state['ready'] and state['download']['message'] == '扩展模型已就绪'
    for state_name in ('failed', 'interrupted', 'complete'):
        common.atomic_json(tiny_extensions / 'download-state.json', {'state': state_name, 'message': 'keep diagnostic'})
        state = pipeline_models.status()
        assert state['download'] == {'state': state_name, 'message': 'keep diagnostic'}


def test_extension_verify_respects_a_real_other_process_download_lock(tiny_extensions, tmp_path):
    marker = tmp_path / 'lock-ready'
    script = ('from filelock import FileLock\nfrom pathlib import Path\nimport sys\n'
              'with FileLock(sys.argv[1]):\n'
              ' Path(sys.argv[2]).write_text("ready")\n'
              ' sys.stdin.readline()\n')
    child = subprocess.Popen([sys.executable, '-c', script, str(tiny_extensions / '.download.lock'), str(marker)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    import time
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            assert child.poll() is None
            time.sleep(.03)
        assert marker.is_file()
        original = (tiny_extensions / 'manifest.json').read_bytes()
        result = pipeline_models.verify()
        assert result['download']['state'] == 'downloading'
        assert result['verification_errors'] and '仍在运行' in result['verification_errors'][0]
        assert (tiny_extensions / 'manifest.json').read_bytes() == original
    finally:
        child.communicate(b'\n', timeout=10)


def test_bad_download_ranges_are_quarantined_before_retry(tmp_path, monkeypatch):
    import requests
    import huggingface_hub
    payload = b'correct range bytes'
    sha = hashlib.sha256(payload).hexdigest()
    destination = tmp_path / 'model.bin'
    parts = tmp_path / '.model.bin.parts'
    parts.mkdir()
    (parts / '0').write_bytes(b'x' * len(payload))
    monkeypatch.setattr(pipeline_models, '_STATE_PATH', None)
    monkeypatch.setattr(pipeline_models, '_STATE', {'state': 'idle'})
    monkeypatch.setattr(huggingface_hub, 'get_token', lambda: None)
    with pytest.raises(ValueError, match='内容校验失败'):
        pipeline_models._ranged_download({'repo': 'fixture/model', 'revision': 'fixed'}, 'model.bin', destination, len(payload), sha)
    assert not parts.exists() and not destination.exists()
    assert len(list(tmp_path.glob('.model.bin.parts.invalid-*'))) == 1
    calls = []

    class Response:
        status_code = 206
        headers = {'Content-Range': f'bytes 0-{len(payload)-1}/{len(payload)}'}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, size): yield payload

    def transfer(url, **kwargs):
        calls.append(kwargs['headers']['Range'])
        return Response()

    monkeypatch.setattr(requests, 'get', transfer)
    pipeline_models._ranged_download({'repo': 'fixture/model', 'revision': 'fixed'}, 'model.bin', destination, len(payload), sha)
    assert destination.read_bytes() == payload and calls == [f'bytes=0-{len(payload)-1}']
    assert not parts.exists()


def test_range_download_rejects_a_server_ignoring_range(tmp_path, monkeypatch):
    import requests
    import huggingface_hub
    payload = b'correct range bytes'
    monkeypatch.setattr(huggingface_hub, 'get_token', lambda: None)
    monkeypatch.setattr(pipeline_models.time, 'sleep', lambda seconds: None)

    class Response:
        status_code = 200
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, size): pytest.fail('an unbounded non-range body must not be read')

    monkeypatch.setattr(requests, 'get', lambda *args, **kwargs: Response())
    target = tmp_path / 'model.bin'
    with pytest.raises(ValueError, match='下载区间'):
        pipeline_models._ranged_download({'repo': 'fixture/model', 'revision': 'fixed'}, 'model.bin', target,
            len(payload), hashlib.sha256(payload).hexdigest())
    assert not target.exists()


def test_foreground_uses_valid_alpha_and_keep_without_loading_onnx(tmp_path, monkeypatch):
    # Failing imports would expose an accidental dependency or model load.
    monkeypatch.setitem(sys.modules, 'onnxruntime', None)
    image = Image.new('RGBA', (10, 10), (20, 60, 120, 128))
    source = tmp_path / 'alpha.png'; image.save(source)
    output, used = run_stage.foreground({'image_path': str(source)}, tmp_path, 'coarse')
    assert used == 'input_alpha' and np.array_equal(np.asarray(output), np.asarray(image))
    opaque = tmp_path / 'opaque.png'; image.convert('RGB').save(opaque)
    output, used = run_stage.foreground({'image_path': str(opaque), 'background': 'keep'}, tmp_path, 'texture')
    assert used == 'keep_opaque' and output.getchannel('A').getextrema() == (255, 255)


def test_empty_transparent_foreground_fails_before_model_inference(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'onnxruntime', None)
    source = tmp_path / 'empty.png'; Image.new('RGBA', (10, 10), (0, 0, 0, 0)).save(source)
    with pytest.raises(ValueError, match='有效前景'):
        run_stage.foreground({'image_path': str(source)}, tmp_path, 'coarse')


@pytest.mark.parametrize('kind', ['changed', 'traversal'])
def test_pinned_engine_sources_reject_mutation_and_escape(tmp_path, monkeypatch, kind):
    runtime = tmp_path / 'runtime'
    source = runtime / 'sources' / 'hunyuan'; source.mkdir(parents=True)
    candidate = source / 'engine.py'; candidate.write_bytes(b'original source')
    relative = 'engine.py'
    expected_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if kind == 'changed': candidate.write_bytes(b'changed source')
    else:
        outside = source.parent / 'outside.py'; outside.write_bytes(b'original source')
        relative = '../outside.py'
    common.atomic_json(source / 'source-manifest.json', {'revision': 'f8db63096c8282cb27354314d896feba5ba6ff8a',
        'files': {relative: expected_hash}})
    monkeypatch.setattr(sys, 'prefix', str(runtime))
    before = list(sys.path)
    with pytest.raises(RuntimeError, match='源码校验失败'):
        run_stage.source_dir('hunyuan')
    assert sys.path == before


def test_child_progress_is_logged_and_stage_completion_is_not_pipeline_completion(tmp_path, capsys):
    request = tmp_path / 'request.json'; request.write_text('{}')
    script = tmp_path / 'progress.py'
    script.write_text('print(\'HI3D_EVENT {"stage":"complete","value":5}\', flush=True)\n'
                      'print(\'HI3D_EVENT [1,2]\', flush=True)\n'
                      'print(\'HI3D_EVENT {invalid}\', flush=True)\n')
    pipeline_worker.run_child([sys.executable, str(script)], request, tmp_path, 'texture')
    output = capsys.readouterr().out
    event = json.loads(output.splitlines()[0].removeprefix('HI3D_EVENT '))
    assert event['stage'] == 'texture_complete' and event['value'] == 5
    assert '[1,2]' in output and '{invalid}' in output
    assert '"stage":"complete"' in (tmp_path / 'stage.log').read_text()
    identity = common.read_json(tmp_path / 'process.json')
    assert identity['request'] == str(request) and identity['stage'] == 'texture'


def test_child_failure_uses_saved_stage_diagnostic(tmp_path):
    request = tmp_path / 'request.json'; request.write_text('{}')
    common.atomic_json(tmp_path / 'failure.json', {'error': 'fixture CUDA OOM'})
    with pytest.raises(RuntimeError, match='texture 阶段失败：fixture CUDA OOM'):
        pipeline_worker.run_child([sys.executable, '-c', 'raise SystemExit(7)'], request, tmp_path, 'texture')


@pytest.mark.parametrize('case,stage,message', [
    ('not-object', 'coarse', 'JSON对象'),
    ('not-glb', 'coarse', 'GLB path'),
    ('bad-background', 'coarse', 'background must'),
    ('missing-image', 'coarse', '参考图片不存在'),
    ('missing-mesh', 'texture', '贴图网格不存在'),
    ('bad-resolution', 'coarse', '基础几何分辨率'),
    ('bad-texture-size', 'texture', '贴图尺寸'),
])
def test_stage_preflight_records_errors_without_loading_a_model(tmp_path, monkeypatch, case, stage, message):
    image = tmp_path / 'input.png'; Image.new('RGBA', (10, 10), (20, 60, 120, 128)).save(image)
    mesh = tmp_path / 'input.glb'; mesh.write_bytes(b'placeholder never parsed')
    output = tmp_path / 'result.glb'; output.write_bytes(b'previous deliverable preserved')
    before = output.read_bytes()
    payload = {'image_path': str(image), 'mesh_path': str(mesh), 'output_path': str(output),
        'resolution': 256 if stage == 'coarse' else 512}
    if case == 'not-object': payload = []
    elif case == 'not-glb': payload['output_path'] = str(tmp_path / 'bad.obj')
    elif case == 'bad-background': payload['background'] = 'unsupported'
    elif case == 'missing-image': payload['image_path'] = str(tmp_path / 'absent.png')
    elif case == 'missing-mesh': payload['mesh_path'] = str(tmp_path / 'absent.glb')
    elif case == 'bad-resolution': payload['resolution'] = 999
    elif case == 'bad-texture-size': payload['texture_size'] = 999
    request = tmp_path / 'request.json'; common.atomic_json(request, payload)
    monkeypatch.setattr(sys, 'argv', ['run_stage', '--request', str(request), '--stage', stage])
    monkeypatch.setitem(sys.modules, 'torch', None)
    assert run_stage.main() == 1
    failure = common.read_json(tmp_path / 'failure.json')
    assert failure['status'] == 'failed' and message in failure['error']
    assert output.read_bytes() == before and not (tmp_path / 'result.partial.glb').exists()


def test_stage_no_cuda_fails_explicitly_and_forces_offline_before_model_import(tmp_path, monkeypatch):
    image = tmp_path / 'input.png'; Image.new('RGBA', (10, 10), (20, 60, 120, 128)).save(image)
    request = tmp_path / 'request.json'
    common.atomic_json(request, {'image_path': str(image), 'output_path': str(tmp_path / 'result.glb')})
    monkeypatch.setattr(sys, 'argv', ['run_stage', '--request', str(request), '--stage', 'coarse'])
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: False, is_initialized=lambda: False)))
    monkeypatch.setenv('HF_HUB_OFFLINE', 'audit-before')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', 'audit-before')
    assert run_stage.main() == 1
    assert os.environ['HF_HUB_OFFLINE'] == '1' and os.environ['TRANSFORMERS_OFFLINE'] == '1'
    failure = common.read_json(tmp_path / 'failure.json')
    assert failure['status'] == 'failed' and 'NVIDIA CUDA GPU' in failure['error']
    assert not (tmp_path / 'result.glb').exists()
