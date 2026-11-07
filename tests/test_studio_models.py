import hashlib
import json
import subprocess
import shutil
import sys

import pytest

from filelock import FileLock

from test_workbench import testroot
from hi3d_workbench.studio import models


def isolate(monkeypatch, tmp_path):
    root = tmp_path / 'weights'
    root.mkdir()
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    monkeypatch.setattr(models, 'model_root', lambda: root)
    monkeypatch.setattr(models, 'runtime_root', lambda: runtime)
    monkeypatch.setattr(models, '_STATE', {'state': 'idle', 'message': ''})
    return root, runtime


def test_malformed_nested_manifests_are_not_ready_or_server_errors(monkeypatch, tmp_path):
    root, runtime = isolate(monkeypatch, tmp_path)
    for bad in ([], None, 'invalid'):
        (root / 'manifest.json').write_text(json.dumps({'components': bad}), encoding='utf-8')
        (runtime / 'studio-runtime.json').write_text(json.dumps({'features': bad}), encoding='utf-8')
        state = models.status()
        assert not any(value['ready'] for value in state['components'].values())
        assert not any(value['ready'] for value in state['features'].values())
    (root / 'manifest.json').write_text(json.dumps({'components': {'p3sam': {'files': []}}}), encoding='utf-8')
    assert not models.status()['components']['p3sam']['ready']


def test_actual_cross_process_lock_overrides_idle_memory_state(monkeypatch, tmp_path):
    root, _ = isolate(monkeypatch, tmp_path)
    code = ('import sys; from filelock import FileLock; '
            'guard=FileLock(sys.argv[1]); guard.acquire(); '
            'print("LOCKED",flush=True); sys.stdin.readline(); guard.release()')
    process = subprocess.Popen([sys.executable, '-u', '-c', code, str(root / '.download.lock')],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == 'LOCKED'
        assert process.poll() is None
        assert models.download_busy()
        assert models.status()['download']['state'] == 'busy'
    finally:
        process.communicate('\n', timeout=10)
    assert not models.download_busy()


def test_abandoned_progress_does_not_claim_live_download(monkeypatch, tmp_path):
    root, _ = isolate(monkeypatch, tmp_path)
    (root / '.download-state.json').write_text(json.dumps({
        'model_root': str(root.resolve()), 'state': 'downloading', 'file_bytes': 123}), encoding='utf-8')
    assert models.status()['download']['state'] == 'interrupted'
    with FileLock(root / '.download.lock'):
        assert models.status()['download']['state'] == 'downloading'
    assert models.status()['download']['state'] == 'interrupted'


def test_verify_rebuilds_receipt_from_pinned_hashes_after_manifest_loss(monkeypatch, tmp_path):
    root, _ = isolate(monkeypatch, tmp_path)
    data = b'fixed official content'
    component = root / 'p3sam'
    component.mkdir()
    (component / 'weights.bin').write_bytes(data)
    monkeypatch.setattr(models, 'CATALOG', {'p3sam': {'repo': 'author/model', 'revision': 'fixed',
        'license': 'MIT', 'gated': False, 'files': [{'name': 'weights.bin', 'size': len(data),
        'hash': hashlib.sha256(data).hexdigest(), 'algorithm': 'sha256'}]}})
    monkeypatch.setattr(models, 'FEATURES', {'segment': {'name': 'test', 'components': ['p3sam'], 'runtime': 'parts'}})
    state = models.verify('p3sam')
    assert state['verification_errors'] == []
    assert state['components']['p3sam']['ready']
    with monkeypatch.context() as cached:
        def unexpected_hash(*args):
            raise AssertionError('有效文件签名不应触发重复全量读取')
        cached.setattr(models, '_verify', unexpected_hash)
        assert models.verified_weight('p3sam', 'weights.bin') == component / 'weights.bin'
    (component / 'weights.bin').write_bytes(b'changed' + data[7:])
    assert not models.status()['components']['p3sam']['ready']
    with pytest.raises(RuntimeError, match='损坏'):
        models.verified_weight('p3sam', 'weights.bin')
    assert models.verify('p3sam')['verification_errors'] == ['p3sam/weights.bin']


def test_startup_rehashes_installed_copies_without_downloading_missing_models(monkeypatch,tmp_path):
    root,_=isolate(monkeypatch,tmp_path)
    data=b'fixed locally installed model'
    spec={'repo':'author/model','revision':'fixed','license':'MIT','gated':False,'files':[
        {'name':'weights.bin','size':len(data),'hash':hashlib.sha256(data).hexdigest(),'algorithm':'sha256'}]}
    monkeypatch.setattr(models,'CATALOG',{'p3sam':spec,'not_installed':spec})
    monkeypatch.setattr(models,'FEATURES',{'segment':{'name':'fixture','components':['p3sam'],'runtime':'parts'}})
    component=root/'p3sam';component.mkdir();(component/'weights.bin').write_bytes(data)
    def no_download(*args,**kwargs):raise AssertionError('Startup must never download weights')
    monkeypatch.setattr(models,'_transfer',no_download)
    states=[]
    assert models.pending_verification()==['p3sam']
    state=models.verify(installed_only=True,callback=lambda value:states.append(value))
    assert not state['verification_errors'] and state['components']['p3sam']['ready']
    assert not state['components']['not_installed']['ready']
    assert states[0]['state']=='verifying' and states[0]['total_files']==1
    assert states[-1]['state']=='complete' and states[-1]['verified_files']==1
    copied=tmp_path/'copied-weights';shutil.copytree(root,copied)
    monkeypatch.setattr(models,'model_root',lambda:copied)
    assert models.pending_verification()==['p3sam']
    from hi3d_workbench.server import initial_studio_verification
    initial_studio_verification()
    assert models.status()['components']['p3sam']['ready']
    assert models.pending_verification()==[]
    with monkeypatch.context() as warm:
        warm.setattr(models,'_verify',no_download)
        assert not models.verify(installed_only=True)['verification_errors']
    (copied/'p3sam/weights.bin').write_bytes(b'X'+data[1:])
    failed=models.verify(installed_only=True)
    assert failed['verification_errors']==['p3sam/weights.bin']
    assert failed['download']['state']=='failed' and not failed['components']['p3sam']['ready']
