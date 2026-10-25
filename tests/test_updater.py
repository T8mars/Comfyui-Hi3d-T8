import hashlib
import json
import stat
import zipfile
from pathlib import Path

import httpx
import pytest
from filelock import FileLock, Timeout
from hi3d_workbench import updater as update
from hi3d_workbench.common import atomic_json


def package(root, files=None, release='0.4.9'):
    files = files or {'hi3d_workbench/example.py': b'new program', 'docs/new.txt': b'new docs'}
    target = root / 'data/updates' / f'Hi3D-Workbench-update-{release}.zip'
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'w') as bundle:
        for name, content in files.items():
            bundle.writestr(name, content)
    value = {'schema': 1, 'repository': update.REPOSITORY, 'version': release, 'runtime_api': 1,
             'minimum_version': '0.4.0', 'archive': target.name, 'bytes': target.stat().st_size,
             'sha256': update.sha256(target),
             'files': {name: {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()} for name, content in files.items()}}
    atomic_json(target.parent / 'pending.json', value)
    return target, value


def test_install_preserves_weights_runtime_config_and_history(tmp_path):
    sentinels = {'models/weights.bin': b'weights', 'runtime/python.exe': b'python',
                 'runtime-pipeline/Lib/example.py': b'private runtime', 'config.local.json': b'{"port":8378}',
                 'data/jobs/old/report.json': b'old result', 'comfy-runtime/user/workflows/demo.json': b'canvas'}
    for name, content in sentinels.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    old = tmp_path / 'hi3d_workbench/example.py'
    old.parent.mkdir()
    old.write_bytes(b'old program')
    package(tmp_path)
    assert update.apply_pending(tmp_path)
    assert old.read_bytes() == b'new program'
    assert (tmp_path / 'docs/new.txt').read_bytes() == b'new docs'
    assert all((tmp_path / name).read_bytes() == content for name, content in sentinels.items())
    journal = update.read(tmp_path / 'data/updates/transaction.json')
    assert journal['phase'] == 'complete'
    assert (tmp_path / 'data/updates/backups' / journal['id'] / 'hi3d_workbench/example.py').read_bytes() == b'old program'


@pytest.mark.parametrize('name', ['../outside.py', '/outside.py', 'docs/../../outside.py', 'docs//duplicate.py',
    'docs/./duplicate.py', 'docs\\outside.py', 'docs/CON.txt', 'docs/trailing.', 'docs/file:stream',
    'models/weights.safetensors', 'runtime/python.exe', 'config.local.json', 'data/jobs/result.json',
    '.secrets/token.txt', 'hi3d_workbench/payload.pyd', 'upstream/Other/a.py'])
def test_rejects_unsafe_or_protected_paths(tmp_path, name):
    _, manifest = package(tmp_path, {name: b'bad'})
    with pytest.raises(ValueError):
        update.apply_pending(tmp_path)
    assert not (tmp_path / 'data/updates/transaction.json').exists()


def test_corrupt_archive_never_modifies_program(tmp_path):
    archive, _ = package(tmp_path)
    archive.write_bytes(b'broken download')
    with pytest.raises(ValueError, match='校验失败'):
        update.apply_pending(tmp_path)
    assert not (tmp_path / 'hi3d_workbench').exists()


def test_unlisted_archive_file_is_rejected(tmp_path):
    archive, manifest = package(tmp_path)
    with zipfile.ZipFile(archive, 'a') as bundle:
        bundle.writestr('docs/unlisted.txt', b'extra')
    manifest.update(bytes=archive.stat().st_size, sha256=update.sha256(archive))
    with pytest.raises(ValueError, match='清单不符'):
        update.validate_archive(archive, manifest, tmp_path)


def test_duplicate_archive_entry_is_rejected(tmp_path):
    archive, manifest = package(tmp_path)
    with pytest.warns(UserWarning), zipfile.ZipFile(archive, 'a') as bundle:
        bundle.writestr('docs/new.txt', b'new docs')
    manifest.update(bytes=archive.stat().st_size, sha256=update.sha256(archive))
    with pytest.raises(ValueError, match='清单不符'):
        update.validate_archive(archive, manifest, tmp_path)


def test_symlink_archive_entry_is_rejected(tmp_path):
    archive, manifest = package(tmp_path, {'docs/link.txt': b'target'})
    info = zipfile.ZipInfo('docs/link.txt')
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr(info, b'target')
    manifest.update(bytes=archive.stat().st_size, sha256=update.sha256(archive))
    with pytest.raises(ValueError, match='符号链接'):
        update.validate_archive(archive, manifest, tmp_path)


def test_changed_runtime_requires_new_complete_bundle(tmp_path):
    _, manifest = package(tmp_path)
    manifest['runtime_api'] = 2
    with pytest.raises(ValueError, match='完整版'):
        update.validate_manifest(manifest)


def test_source_checkout_does_not_auto_install(tmp_path):
    (tmp_path / '.git').mkdir()
    package(tmp_path)
    assert not update.preferences(tmp_path)['auto_update']
    with pytest.raises(ValueError, match='Git'):
        update.apply_pending(tmp_path)


def test_running_server_prevents_install(tmp_path):
    package(tmp_path)
    with FileLock(tmp_path / 'data/server.lock'):
        with pytest.raises(Timeout):
            update.apply_pending(tmp_path)
    assert not (tmp_path / 'hi3d_workbench').exists()


def test_write_failure_rolls_back_every_changed_file(tmp_path, monkeypatch):
    original = tmp_path / 'hi3d_workbench/example.py'
    original.parent.mkdir()
    original.write_bytes(b'old program')
    package(tmp_path)
    replace = update.replace_bytes
    def fail_new_doc(path, content):
        if path == tmp_path / 'docs/new.txt':
            raise OSError('simulated locked file')
        replace(path, content)
    monkeypatch.setattr(update, 'replace_bytes', fail_new_doc)
    with pytest.raises(OSError, match='locked'):
        update.apply_pending(tmp_path)
    assert original.read_bytes() == b'old program'
    assert update.read(tmp_path / 'data/updates/transaction.json')['phase'] == 'rolled_back'
    assert (tmp_path / 'data/updates/pending.json').exists()


def test_interrupted_update_restores_before_reloading_version(tmp_path):
    package(tmp_path)
    original = tmp_path / 'hi3d_workbench/example.py'
    original.parent.mkdir()
    original.write_bytes(b'half installed')
    identity = 'a' * 32
    backup = tmp_path / 'data/updates/backups' / identity / 'hi3d_workbench/example.py'
    backup.parent.mkdir(parents=True)
    backup.write_bytes(b'old program')
    atomic_json(tmp_path / 'data/updates/transaction.json', {'id': identity, 'phase': 'installing',
        'before': {'hi3d_workbench/example.py': {'existed': True, 'sha256': update.sha256(backup)}}})
    assert update.apply_pending(tmp_path) is True
    assert original.read_bytes() == b'old program'
    assert (tmp_path / 'data/updates/pending.json').exists()
    assert update.apply_pending(tmp_path) is True
    assert original.read_bytes() == b'new program'


def test_download_and_stage_from_official_release(tmp_path, monkeypatch):
    archive, manifest = package(tmp_path)
    payload = archive.read_bytes()
    archive.unlink()
    (archive.parent / 'pending.json').unlink()
    prefix = update.DOWNLOAD + 'v0.4.9/'
    release = {'tag_name': 'v0.4.9', 'draft': False, 'prerelease': False, 'assets': [
        {'name': 'update-manifest.json', 'browser_download_url': prefix + 'update-manifest.json'},
        {'name': archive.name, 'size': len(payload), 'digest': 'sha256:' + manifest['sha256'], 'browser_download_url': prefix + archive.name}]}
    def respond(request):
        if str(request.url) == update.API:
            return httpx.Response(200, json=release)
        if str(request.url).endswith('update-manifest.json'):
            return httpx.Response(200, json=manifest)
        return httpx.Response(200, content=payload)
    monkeypatch.setattr(update, 'client', lambda: httpx.Client(transport=httpx.MockTransport(respond)))
    assert update.check(tmp_path)['phase'] == 'staged'
    assert archive.read_bytes() == payload
    assert not (tmp_path / 'hi3d_workbench').exists()


def test_network_failure_preserves_program_and_reports_error(tmp_path, monkeypatch):
    def fail(_):
        raise httpx.ConnectError('offline')
    monkeypatch.setattr(update, 'get_json', fail)
    assert update.check(tmp_path)['phase'] == 'error'
    assert not (tmp_path / 'hi3d_workbench').exists()


def test_download_refuses_other_repository(tmp_path):
    _, manifest = package(tmp_path)
    with pytest.raises(ValueError, match='本项目'):
        update.download('https://github.com/other/repo/file.zip', manifest, tmp_path / 'new.zip')


def test_corrupt_pending_can_be_checked_again(tmp_path, monkeypatch):
    archive, _ = package(tmp_path)
    archive.write_bytes(b'corrupt cache')
    monkeypatch.setattr(update, 'get_json', lambda _: {'tag_name': 'v' + update.VERSION, 'assets': []})
    assert update.check(tmp_path)['phase'] == 'current'
    assert not (archive.parent / 'pending.json').exists()
    assert list(archive.parent.glob('invalid-pending-*.json'))


def test_python_bytecode_is_invalidated_on_install(tmp_path):
    target = tmp_path / 'hi3d_workbench/__pycache__/example.cpython-310.pyc'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'stale bytecode')
    package(tmp_path)
    update.apply_pending(tmp_path)
    assert not target.exists()


def test_individual_file_hash_is_checked_even_with_correct_archive_hash(tmp_path):
    archive, manifest = package(tmp_path)
    manifest['files']['docs/new.txt']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='SHA256'):
        update.validate_archive(archive, manifest, tmp_path)


def test_case_collision_and_parent_file_are_rejected(tmp_path):
    _, manifest = package(tmp_path, {'docs/name.txt': b'a', 'docs/NAME.txt': b'b'})
    with pytest.raises(ValueError, match='冲突'):
        update.validate_manifest(manifest)
    _, manifest = package(tmp_path, {'docs/item': b'a', 'docs/item/child.txt': b'b'})
    with pytest.raises(ValueError, match='冲突'):
        update.validate_manifest(manifest)


def test_orphan_gpu_worker_prevents_install_even_without_server_lock(tmp_path, monkeypatch):
    import psutil
    package(tmp_path)
    class Worker:
        info = {'cmdline': ['python', '-m', 'hi3d_workbench.engines.run_stage', '--request', str(tmp_path / 'data/jobs/a/request.json')]}
    monkeypatch.setattr(psutil, 'process_iter', lambda _: [Worker()])
    with pytest.raises(ValueError, match='工作进程'):
        update.apply_pending(tmp_path)
    assert not (tmp_path / 'hi3d_workbench').exists()


def test_disabling_auto_update_keeps_pending_for_explicit_install(tmp_path):
    package(tmp_path)
    update.set_preferences(False, tmp_path)
    assert not update.apply_on_start(tmp_path)
    assert not (tmp_path / 'hi3d_workbench').exists()
    assert update.apply_pending(tmp_path)


def test_corrupt_transaction_blocks_startup_even_when_auto_update_disabled(tmp_path):
    package(tmp_path)
    update.set_preferences(False, tmp_path)
    (tmp_path / 'data/updates/transaction.json').write_bytes(b'corrupt transaction')
    assert update.recovery_required(tmp_path)
    with pytest.raises(ValueError, match='事务记录损坏'):
        update.apply_on_start(tmp_path)
    assert not (tmp_path / 'hi3d_workbench').exists()
