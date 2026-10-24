"""Second v0.3.1 audit: CPU fixtures and isolated short-lived processes only."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from test_asset_pipeline import tiny_extensions
from test_job_lifecycle import lifecycle, wait_until
from hi3d_workbench import common, jobs, pipeline_models, pipeline_worker
from hi3d_workbench.engines import run_stage


@pytest.fixture
def completed_stage_fixture(tmp_path, monkeypatch):
    """Replace only CUDA/model calls; execute real stage publication logic."""
    image = tmp_path / 'input.png'
    Image.new('RGBA', (10, 10), (30, 60, 120, 128)).save(image)
    output = tmp_path / 'result.glb'
    output.write_bytes(b'previous user deliverable')
    request = tmp_path / 'request.json'
    common.atomic_json(request, {'image_path': str(image), 'output_path': str(output), 'resolution': 256})
    cuda = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True,
        reset_peak_memory_stats=lambda: None, synchronize=lambda: None,
        max_memory_allocated=lambda: 4096, max_memory_reserved=lambda: 8192,
        get_device_name=lambda: 'CPU test fixture, no GPU', get_device_capability=lambda: (12, 0))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda, __version__='fixture',
        version=SimpleNamespace(cuda='fixture')))
    monkeypatch.setattr(sys, 'argv', ['run_stage', '--request', str(request), '--stage', 'coarse'])

    def stage(payload, foreground, target, callback):
        target.write_bytes(b'new generated GLB stand-in' * 10)
        return {'geometry_generated': True, 'texture_generated': False}

    monkeypatch.setattr(run_stage, 'coarse', stage)
    return output, cuda


@pytest.mark.parametrize('boundary', ['synchronize', 'metrics', 'report'])
def test_stage_failure_before_publication_preserves_previous_output(completed_stage_fixture, monkeypatch, boundary):
    output, cuda = completed_stage_fixture
    before = output.read_bytes()

    def failed():
        raise RuntimeError('fixture post-inference failure')

    if boundary == 'synchronize':
        cuda.synchronize = failed
    elif boundary == 'metrics':
        cuda.max_memory_allocated = failed
    else:
        write = run_stage.atomic_json

        def failed_success_report(path, value):
            if path == output.with_suffix('.stage.json') and value.get('status') == 'succeeded':
                failed()
            write(path, value)

        monkeypatch.setattr(run_stage, 'atomic_json', failed_success_report)
    assert run_stage.main() == 1
    assert output.read_bytes() == before
    assert not output.with_name('result.partial.glb').exists()
    failure = common.read_json(output.parent / 'failure.json')
    assert failure['status'] == 'failed' and failure['error'] == 'fixture post-inference failure'


def test_range_download_stops_before_writing_an_oversized_response(tmp_path, monkeypatch):
    import huggingface_hub
    import requests
    payload = b'expected bounded range'
    destination = tmp_path / 'model.bin'
    calls = []
    monkeypatch.setattr(huggingface_hub, 'get_token', lambda: None)
    monkeypatch.setattr(pipeline_models.time, 'sleep', lambda seconds: None)

    class Response:
        status_code = 206
        headers = {'Content-Range': f'bytes 0-{len(payload)-1}/{len(payload)}'}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, size):
            calls.append('stream')
            yield payload + b'excess bytes must not be persisted'
            pytest.fail('must stop consuming a response after the requested range is exceeded')

    monkeypatch.setattr(requests, 'get', lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match='下载区间'):
        pipeline_models._ranged_download({'repo': 'fixture/model', 'revision': 'fixed'},
            'model.bin', destination, len(payload), hashlib.sha256(payload).hexdigest())
    assert len(calls) == 5
    assert not destination.exists()
    assert not list(tmp_path.rglob('*.tmp'))


@pytest.mark.parametrize('report', [[], None, {'status': 'failed', 'error': 'saved stage failure'}])
def test_exit_zero_with_invalid_or_failed_report_is_not_job_success(lifecycle, report):
    _, manager, script, image, mesh = lifecycle
    script.write_text('from pathlib import Path\nimport json,sys\n'
        'folder=Path(sys.argv[-1]).parent\n'
        '(folder/"refined.glb").write_bytes(b"fixture result")\n'
        f'(folder/"report.json").write_text(json.dumps({report!r}))\n')
    job = manager.create(image, mesh, {'resolution': 512})
    wait_until(lambda: manager.active is None)
    state = manager.get(job['id'])
    assert state['state'] == 'failed'
    assert '报告' in state['message']
    assert (manager.root / job['id'] / 'input.png').read_bytes() == image.read_bytes()


@pytest.mark.parametrize('saved', ['[]', 'null', '{bad-json', '{"error":null}'])
def test_child_failure_with_broken_diagnostic_preserves_real_exit_code(tmp_path, saved):
    request = tmp_path / 'request.json'
    request.write_text('{}')
    (tmp_path / 'failure.json').write_text(saved)
    with pytest.raises(RuntimeError, match='texture 阶段失败：进程退出 7'):
        pipeline_worker.run_child([sys.executable, '-c', 'raise SystemExit(7)'], request, tmp_path, 'texture')


@pytest.mark.parametrize('saved', ['[]', 'null', '{bad-json', '{"error":null}'])
def test_job_failure_with_broken_diagnostic_preserves_real_exit_code(lifecycle, saved):
    _, manager, script, image, mesh = lifecycle
    script.write_text('from pathlib import Path\nimport sys\n'
        'folder=Path(sys.argv[-1]).parent\n'
        f'(folder/"failure.json").write_text({saved!r})\n'
        'raise SystemExit(7)\n')
    job = manager.create(image, mesh, {'resolution': 512})
    wait_until(lambda: manager.active is None)
    state = manager.get(job['id'])
    assert state['state'] == 'failed' and '进程退出 7' in state['message']
    assert state['failure']['error'] == state['message']
    assert (manager.root / job['id'] / 'input.png').read_bytes() == image.read_bytes()


def test_stage_publish_failure_keeps_old_deliverable_and_records_failure(completed_stage_fixture, monkeypatch):
    output, _ = completed_stage_fixture
    before = output.read_bytes()
    replace = run_stage.os.replace

    def refuse_publication(source, target):
        if Path(target) == output:
            raise PermissionError('fixture output open in another application')
        replace(source, target)

    monkeypatch.setattr(run_stage.os, 'replace', refuse_publication)
    assert run_stage.main() == 1
    assert output.read_bytes() == before
    assert not output.with_name('result.partial.glb').exists()
    assert common.read_json(output.with_suffix('.stage.json'))['status'] == 'failed'
    assert 'output open' in common.read_json(output.parent / 'failure.json')['error']


def test_stage_success_only_publishes_after_synchronization_and_clears_stale_failure(completed_stage_fixture):
    output, cuda = completed_stage_fixture
    previous = output.read_bytes()
    checked = []

    def synchronized():
        assert output.read_bytes() == previous
        assert output.with_name('result.partial.glb').is_file()
        checked.append(True)

    cuda.synchronize = synchronized
    common.atomic_json(output.parent / 'failure.json', {'error': 'previous attempt'})
    assert run_stage.main() == 0
    assert checked == [True] and output.read_bytes() != previous
    assert not output.with_name('result.partial.glb').exists()
    assert not (output.parent / 'failure.json').exists()
    assert common.read_json(output.with_suffix('.stage.json'))['status'] == 'succeeded'


def test_short_engine_output_does_not_replace_previous_deliverable(completed_stage_fixture, monkeypatch):
    output, _ = completed_stage_fixture
    previous = output.read_bytes()

    def empty(payload, foreground, target, callback):
        target.write_bytes(b'too short')
        return {'geometry_generated': True}

    monkeypatch.setattr(run_stage, 'coarse', empty)
    assert run_stage.main() == 1
    assert output.read_bytes() == previous
    assert not output.with_name('result.partial.glb').exists()
    assert '有效GLB' in common.read_json(output.parent / 'failure.json')['error']


@pytest.mark.parametrize('failure', ['auth', 'missing-file'])
def test_metadata_download_failure_keeps_manifest_and_releases_crossprocess_lock(tiny_extensions, monkeypatch, failure):
    import huggingface_hub
    from filelock import FileLock
    original = (tiny_extensions / 'manifest.json').read_bytes()

    class Api:
        def model_info(self, repo, **kwargs):
            if failure == 'auth':
                raise RuntimeError('fixture 403 access denied')
            return SimpleNamespace(siblings=[])

    monkeypatch.setattr(huggingface_hub, 'HfApi', Api)
    monkeypatch.setattr(pipeline_models, '_STATE', {'state': 'idle'})
    with pytest.raises((RuntimeError, KeyError)):
        pipeline_models.download('encoder')
    assert (tiny_extensions / 'manifest.json').read_bytes() == original
    assert common.read_json(tiny_extensions / 'download-state.json')['state'] == 'failed'
    with FileLock(str(tiny_extensions / '.download.lock'), timeout=0):
        pass


def test_range_download_retries_truncated_or_timed_out_response_then_recovers(tmp_path, monkeypatch):
    import requests
    import huggingface_hub
    payload = b'fixed pinned bytes'
    destination = tmp_path / 'model.bin'
    calls = []
    sleeps = []
    monkeypatch.setattr(huggingface_hub, 'get_token', lambda: None)
    monkeypatch.setattr(pipeline_models.time, 'sleep', sleeps.append)

    class Response:
        status_code = 206
        headers = {'Content-Range': f'bytes 0-{len(payload)-1}/{len(payload)}'}
        def __init__(self, attempt): self.attempt = attempt
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, size):
            yield payload[:3]
            if self.attempt == 1:
                raise requests.Timeout('fixture interrupted transfer')
            if self.attempt == 2:
                return
            yield payload[3:]

    def transfer(url, **kwargs):
        calls.append(kwargs['headers']['Range'])
        return Response(len(calls))

    monkeypatch.setattr(requests, 'get', transfer)
    pipeline_models._ranged_download({'repo': 'fixture/model', 'revision': 'fixed'},
        'model.bin', destination, len(payload), hashlib.sha256(payload).hexdigest())
    assert destination.read_bytes() == payload
    assert sleeps == [1, 2] and len(calls) == 3
    assert not list(tmp_path.rglob('*.tmp'))
    assert not (tmp_path / '.model.bin.parts').exists()


def test_range_resume_fetches_only_missing_ranges_and_verifies_assembled_bytes(tmp_path, monkeypatch):
    import requests
    import huggingface_hub
    first = b'A' * (4 * 1024 * 1024)
    tail = b'fixed last range'
    payload = first + tail
    parts = tmp_path / '.model.bin.parts'
    parts.mkdir()
    (parts / '0').write_bytes(first)
    destination = tmp_path / 'model.bin'
    calls = []
    monkeypatch.setattr(huggingface_hub, 'get_token', lambda: None)
    monkeypatch.setattr(pipeline_models, '_STATE_PATH', None)

    class Response:
        status_code = 206
        headers = {'Content-Range': f'bytes {len(first)}-{len(payload)-1}/{len(payload)}'}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, size): yield tail

    def transfer(url, **kwargs):
        calls.append(kwargs['headers']['Range'])
        return Response()

    monkeypatch.setattr(requests, 'get', transfer)
    pipeline_models._ranged_download({'repo': 'fixture/model', 'revision': 'fixed'},
        'model.bin', destination, len(payload), hashlib.sha256(payload).hexdigest())
    assert destination.read_bytes() == payload
    assert calls == [f'bytes={len(first)}-{len(payload)-1}']
    assert not parts.exists()

