"""Third audit: isolated CPU fixtures, no model downloads or GPU inference."""
import hashlib
import json
import sys
from pathlib import Path

import pytest
import trimesh

from test_asset_pipeline import _pipeline_request, _textured_fixture
from test_job_lifecycle import lifecycle
from test_pipeline_second_audit import completed_stage_fixture
from hi3d_workbench import common, jobs, pipeline_models, pipeline_worker
from hi3d_workbench.engines import run_stage


@pytest.mark.parametrize('boundary', ['input.png', 'coarse.glb', 'request.json', 'state.json'])
def test_failed_job_setup_discards_only_its_unpublished_folder(lifecycle, monkeypatch, boundary):
    _, manager, _, image, mesh = lifecycle
    previous = manager.root / 'previous-user-job'
    previous.mkdir()
    (previous / 'result.glb').write_bytes(b'existing user output')
    source_bytes = (image.read_bytes(), mesh.read_bytes())
    copy = jobs.shutil.copy2
    write = jobs.atomic_json
    starts = []
    monkeypatch.setattr(jobs.threading.Thread, 'start', lambda self: starts.append(True))

    def copy_file(source, target):
        copy(source, target)
        if Path(target).name == boundary:
            raise OSError('fixture setup disk full')

    def write_file(target, value):
        write(target, value)
        if target.name == boundary:
            raise OSError('fixture setup disk full')

    monkeypatch.setattr(jobs.shutil, 'copy2', copy_file)
    monkeypatch.setattr(jobs, 'atomic_json', write_file)
    with pytest.raises(OSError, match='setup disk full'):
        manager.create(image, mesh, {'resolution': 512})
    assert list(manager.root.iterdir()) == [previous]
    assert (previous / 'result.glb').read_bytes() == b'existing user output'
    assert (image.read_bytes(), mesh.read_bytes()) == source_bytes
    assert manager.active is None and manager.process is None
    assert manager._memory_states == {} and starts == []


@pytest.mark.parametrize('stage', ['coarse', 'texture', 'refine'])
@pytest.mark.parametrize('report', [None, {}, {'status': 'failed', 'error': 'fixture stage failure'}])
def test_supervisor_rejects_exit_zero_without_explicit_stage_success(tmp_path, monkeypatch, stage, report):
    request = _pipeline_request(tmp_path, mode='image_to_asset' if stage == 'coarse' else stage,
        generate_texture=stage == 'texture', refine_geometry=False)
    if stage != 'coarse':
        trimesh.creation.box().export(tmp_path / 'coarse.glb')
    original_input = (tmp_path / 'input.png').read_bytes()

    def exited_zero(command, request_path, directory, child_stage):
        assert child_stage == stage
        if stage == 'refine':
            output = directory / 'refined.glb'
            report_path = directory / 'report.json'
        else:
            payload = common.read_json(request_path)
            output = Path(payload['output_path'])
            report_path = output.with_suffix('.stage.json')
        if stage == 'texture':
            _textured_fixture(output)
        else:
            trimesh.creation.box().export(output)
        if report is not None:
            common.atomic_json(report_path, report)

    monkeypatch.setattr(pipeline_worker, 'run_child', exited_zero)
    monkeypatch.setattr(pipeline_worker, 'config', lambda: {'worker_command': [sys.executable]})
    with pytest.raises((ValueError, RuntimeError), match='报告'):
        pipeline_worker.run(request)
    assert not (tmp_path / 'report.json').exists()
    assert (tmp_path / 'input.png').read_bytes() == original_input
    if stage == 'coarse':
        assert not (tmp_path / 'coarse.glb').exists()
    elif stage == 'refine':
        assert not (tmp_path / 'refined.glb').exists()
    else:
        assert not (tmp_path / 'textured.glb').exists()


@pytest.mark.parametrize('diagnostic', ['result.stage.json', 'failure.json'])
def test_partial_result_removed_even_when_failure_diagnostic_cannot_be_saved(completed_stage_fixture, monkeypatch, diagnostic):
    output, cuda = completed_stage_fixture
    previous = output.read_bytes()

    def failed_sync():
        raise RuntimeError('fixture CUDA synchronization failure')

    cuda.synchronize = failed_sync
    write = run_stage.atomic_json

    def failed_diagnostic(path, value):
        if path.name == diagnostic and value.get('status') == 'failed':
            raise OSError('fixture diagnostic disk full')
        write(path, value)

    monkeypatch.setattr(run_stage, 'atomic_json', failed_diagnostic)
    with pytest.raises(OSError, match='diagnostic disk full'):
        run_stage.main()
    assert output.read_bytes() == previous
    assert not output.with_name('result.partial.glb').exists()


@pytest.mark.parametrize('capability', [(7, 5), (8, 0)])
def test_windows_texture_architecture_floor_is_checked_before_foreground_or_model(completed_stage_fixture, monkeypatch, capability):
    if sys.platform != 'win32':
        pytest.skip('Windows precompiled extension architecture guard')
    output, cuda = completed_stage_fixture
    previous = output.read_bytes()
    request = output.parent / 'request.json'
    payload = common.read_json(request)
    payload.update(mesh_path=str(output), resolution=512)
    common.atomic_json(request, payload)
    monkeypatch.setattr(sys, 'argv', ['run_stage', '--request', str(request), '--stage', 'texture'])
    cuda.get_device_capability = lambda: capability
    monkeypatch.setattr(run_stage, 'foreground', lambda *args: pytest.fail('must reject before foreground/model'))
    assert run_stage.main() == 1
    failure = common.read_json(output.parent / 'failure.json')
    assert '8.6' in failure['error']
    assert output.read_bytes() == previous


def test_corrupt_reference_image_fails_before_calling_model(completed_stage_fixture, monkeypatch):
    output, _ = completed_stage_fixture
    previous = output.read_bytes()
    (output.parent / 'input.png').write_bytes(b'not a decoded image')
    monkeypatch.setattr(run_stage, 'coarse', lambda *args: pytest.fail('must not load model'))
    assert run_stage.main() == 1
    assert output.read_bytes() == previous
    assert common.read_json(output.parent / 'failure.json')['status'] == 'failed'
    assert not output.with_name('result.partial.glb').exists()


def test_failure_emit_cannot_prevent_partial_cleanup(completed_stage_fixture, monkeypatch):
    output, cuda = completed_stage_fixture
    previous = output.read_bytes()
    cuda.synchronize = lambda: (_ for _ in ()).throw(RuntimeError('fixture original failure'))
    emit = run_stage.emit

    def closed_pipe(stage, **values):
        if stage.endswith('_failed'):
            raise BrokenPipeError('fixture supervisor closed progress pipe')
        emit(stage, **values)

    monkeypatch.setattr(run_stage, 'emit', closed_pipe)
    with pytest.raises(BrokenPipeError, match='progress pipe'):
        run_stage.main()
    assert common.read_json(output.parent / 'failure.json')['error'] == 'fixture original failure'
    assert output.read_bytes() == previous
    assert not output.with_name('result.partial.glb').exists()


def test_valid_pinned_download_destination_is_used_without_network(tmp_path, monkeypatch):
    import requests
    import huggingface_hub
    path = tmp_path / 'model.bin'
    payload = b'fixture immutable cached bytes'
    path.write_bytes(payload)
    monkeypatch.setattr(requests, 'get', lambda *a, **k: pytest.fail('no network for verified bytes'))
    monkeypatch.setattr(huggingface_hub, 'get_token', lambda: pytest.fail('no credential lookup for verified bytes'))
    result = pipeline_models._ranged_download({'repo': 'fixture/model', 'revision': 'fixed'},
        path.name, path, len(payload), hashlib.sha256(payload).hexdigest())
    assert result == path and path.read_bytes() == payload


def test_range_response_with_wrong_total_is_rejected_before_body_read(tmp_path, monkeypatch):
    import requests
    import huggingface_hub
    payload = b'pinned bytes'
    calls = []
    monkeypatch.setattr(huggingface_hub, 'get_token', lambda: None)
    monkeypatch.setattr(pipeline_models.time, 'sleep', lambda *args: None)

    class Response:
        status_code = 206
        headers = {'Content-Range': f'bytes 0-{len(payload)-1}/{len(payload)+1}'}
        def __enter__(self): return self
        def __exit__(self, *args): calls.append('closed')
        def raise_for_status(self): pass
        def iter_content(self, size): pytest.fail('invalid range body must not be read')

    monkeypatch.setattr(requests, 'get', lambda *a, **k: Response())
    destination = tmp_path / 'model.bin'
    with pytest.raises(ValueError, match='下载区间'):
        pipeline_models._ranged_download({'repo': 'fixture/model', 'revision': 'fixed'},
            destination.name, destination, len(payload), hashlib.sha256(payload).hexdigest())
    assert calls == ['closed'] * 5
    assert not destination.exists() and not list(tmp_path.rglob('*.tmp'))


@pytest.mark.parametrize('algorithm', ['sha256', 'git-sha1'])
def test_verification_checks_real_blob_size_hash_and_signature(tmp_path, algorithm):
    payload = b'fixture model config or weights'
    path = tmp_path / 'artifact'
    path.write_bytes(payload)
    git_blob = algorithm == 'git-sha1'
    expected = hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest() if git_blob else hashlib.sha256(payload).hexdigest()
    signature = pipeline_models._verified_signature(path, expected, git_blob=git_blob, size=len(payload))
    assert signature == pipeline_models._signature(path)
    assert pipeline_models._verified_signature(path, expected, git_blob=git_blob, size=len(payload)+1) is None
    path.write_bytes(b'x' * len(payload))
    assert pipeline_models._verified_signature(path, expected, git_blob=git_blob, size=len(payload)) is None


def test_cancel_before_worker_launch_never_spawns_a_process(lifecycle, monkeypatch):
    _, manager, _, image, mesh = lifecycle
    monkeypatch.setattr(jobs.threading.Thread, 'start', lambda self: None)
    monkeypatch.setattr(jobs.subprocess, 'Popen', lambda *a, **k: pytest.fail('cancelled before Popen'))
    job = manager.create(image, mesh, {'resolution': 512})
    assert manager.active == job['id']
    manager.cancel(job['id'])
    manager._run(job['id'])
    assert manager.active is None and manager.process is None
    assert manager.get(job['id'])['state'] == 'cancelled'
    assert manager.cancelled == set()
    assert (manager.root / job['id'] / 'input.png').read_bytes() == image.read_bytes()
