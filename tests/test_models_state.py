"""Model state regressions; use tiny local artifacts and no network/GPU."""
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

# Import only inside the fixture, after test collection. Other test modules
# establish their own root before importing common; eagerly importing here
# would bypass their isolation through Python's shared module cache.
models=None


@pytest.fixture
def isolated_models(tmp_path,monkeypatch):
    global models
    monkeypatch.setenv('HI3D_WORKBENCH_ROOT',str(tmp_path))
    from hi3d_workbench import models as model_module
    models=model_module
    monkeypatch.setattr(models,'MODELS',tmp_path/'models')
    monkeypatch.setattr(models,'DATA',tmp_path/'data')
    monkeypatch.setattr(models,'STATE',{'state':'idle','message':'fixture'})
    monkeypatch.setattr(models,'LOCK',threading.Lock())
    monkeypatch.setattr(models,'FILE_LOCK',None)
    specs={key:{'repo':'fixture/'+key,'revision':'fixture-revision','files':['weights.bin'],'license':'fixture'}
           for key in ('hi3d','dino','clip')}
    monkeypatch.setattr(models,'SPECS',specs)
    monkeypatch.setattr(models,'config',lambda:{'download_workers':1,'download_trust_env':False})
    payload=b'original fixture model'
    manifest={'components':{}}
    for key,spec in specs.items():
        target=models.MODELS/key/'weights.bin';target.parent.mkdir(parents=True)
        target.write_bytes(payload)
        manifest['components'][key]={'revision':spec['revision'],'repo':spec['repo'],
                                    'files':{'weights.bin':{'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)}}}
    models.atomic_json(models.MODELS/'manifest.json',manifest)
    return payload


def test_verification_tracks_corruption_and_recovers_without_poll_hashes(isolated_models,monkeypatch):
    assert not models.status()['ready']  # Legacy manifest alone is not proof.
    assert models.verify_all()['ok']
    with monkeypatch.context() as local:
        local.setattr(models,'digest',lambda path:pytest.fail('status must not hash model content'))
        assert models.status()['ready']
    path=models.MODELS/'hi3d'/'weights.bin'
    old_mtime=path.stat().st_mtime_ns
    # Equal byte count and restored mtime must still invalidate old success.
    path.write_bytes(b'x'*len(isolated_models))
    os.utime(path,ns=(path.stat().st_atime_ns,old_mtime))
    state=models.status()
    assert not state['ready'] and not state['components']['hi3d']['verified']
    assert state['components']['dino']['verified']
    assert not models.verify_all()['ok']
    assert models.status()['components']['hi3d']['verification']=='failed'
    path.write_bytes(isolated_models)
    # An obsolete failure does not accuse a replacement file of corruption.
    state=models.status()
    assert state['components']['hi3d']['verification']=='required'
    assert state['components']['hi3d']['errors']==[]
    assert models.verify_all()['ok'] and models.status()['ready']


def test_manifest_change_and_file_replacement_invalidate_success(isolated_models):
    assert models.verify_all()['ok']
    manifest_path=models.MODELS/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest['components']['hi3d']['revision']='another-revision'
    models.atomic_json(manifest_path,manifest)
    assert not models.status()['components']['hi3d']['verified']
    assert models.status()['components']['clip']['verified']
    assert not models.verify_all()['ok']
    assert 'hi3d/manifest:revision' in models.verify_all()['errors']
    manifest['components']['hi3d']['revision']=models.SPECS['hi3d']['revision']
    models.atomic_json(manifest_path,manifest)
    assert models.verify_all()['ok']
    path=models.MODELS/'hi3d'/'weights.bin'
    previous=path.stat()
    replacement=path.with_suffix('.replacement');replacement.write_bytes(isolated_models)
    os.utime(replacement,ns=(previous.st_atime_ns,previous.st_mtime_ns))
    os.replace(replacement,path)
    assert not models.status()['ready']
    assert models.verify_all()['ok'] and models.status()['ready']


def test_file_changed_during_verification_is_never_reported_verified(isolated_models,monkeypatch):
    original_digest=models.digest
    changed=models.MODELS/'hi3d'/'weights.bin'
    def racing_digest(path):
        value=original_digest(path)
        if path==changed:path.write_bytes(b'x'*len(isolated_models))
        return value
    monkeypatch.setattr(models,'digest',racing_digest)
    result=models.verify_all()
    assert not result['ok'] and 'hi3d/weights.bin' in result['errors']
    assert not models.status()['ready']


def test_interrupted_download_recovers_but_live_cli_lock_wins(isolated_models,tmp_path):
    models.atomic_json(models.DATA/'model-download.json',{'state':'downloading','message':'old progress'})
    assert models.status()['download']['state']=='interrupted'
    # A real second process represents the standalone model CLI, not a fake
    # persisted flag. Releasing it must immediately make retry possible.
    ready=tmp_path/'cli-ready'
    script=('from filelock import FileLock\nfrom pathlib import Path\nimport sys\n'
            'with FileLock(sys.argv[1]):\n'
            ' Path(sys.argv[2]).write_text("ready")\n'
            ' sys.stdin.readline()\n')
    child=subprocess.Popen([sys.executable,'-c',script,str(models.MODELS/'.download.lock'),str(ready)],
                           stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    try:
        deadline=time.monotonic()+10
        while not ready.exists() and time.monotonic()<deadline:
            assert child.poll() is None,'CLI lock subprocess exited early'
            time.sleep(.03)
        assert ready.exists()
        assert models.status()['download']['state']=='downloading'
        assert models.download_all() is False
    finally:
        child.communicate(b'\n',timeout=10)
    assert models.status()['download']['state']=='interrupted'


def test_hash_mismatch_discards_bad_range_cache_then_downloads_fresh(isolated_models,monkeypatch):
    import requests
    payload=b'correct downloaded bytes'
    sha=hashlib.sha256(payload).hexdigest()
    parts=models.MODELS/'.downloads'/(sha+'-4m');parts.mkdir(parents=True)
    (parts/'0').write_bytes(b'x'*len(payload))
    destination=models.MODELS/'hi3d'/'fresh.bin'
    with pytest.raises(ValueError,match='已清理'):
        models.ranged_download('fixture/model','rev','fresh.bin',destination,len(payload),sha)
    assert not (parts/'0').exists() and not destination.exists()
    calls=[]
    class Response:
        status_code=206
        headers={'Content-Range':f'bytes 0-{len(payload)-1}/{len(payload)}'}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def raise_for_status(self):pass
        def iter_content(self,size):yield payload
    class Session:
        def get(self,*args,**kwargs):calls.append(kwargs['headers']['Range']);return Response()
    monkeypatch.setattr(requests,'Session',Session)
    models.ranged_download('fixture/model','rev','fresh.bin',destination,len(payload),sha)
    assert destination.read_bytes()==payload and len(calls)==1


def test_download_completion_clears_stale_failure_and_refreshes_binding(isolated_models,monkeypatch):
    import huggingface_hub
    # Real downloader control path with local fake transfers: first transfer
    # reuses a corrupt small file; force_download must recover it once.
    path=models.MODELS/'hi3d'/'weights.bin';path.write_bytes(b'x'*len(isolated_models))
    assert not models.verify_all()['ok']
    sha=hashlib.sha256(isolated_models).hexdigest()
    calls=[]
    class Api:
        def model_info(self,repo,**kwargs):
            return SimpleNamespace(sha='fixture-revision',siblings=[SimpleNamespace(rfilename='weights.bin',size=len(isolated_models),lfs={'sha256':sha})])
    def transfer(repo,name,local_dir,force_download=False,**kwargs):
        target=Path(local_dir)/name
        calls.append((repo,force_download))
        if force_download:target.write_bytes(isolated_models)
        return str(target)
    monkeypatch.setattr(huggingface_hub,'HfApi',Api)
    monkeypatch.setattr(huggingface_hub,'hf_hub_download',transfer)
    assert models.download_all() is True
    deadline=time.monotonic()+10
    while models.LOCK.locked() and time.monotonic()<deadline:time.sleep(.02)
    assert not models.LOCK.locked()
    state=models.status()
    assert state['download']['state']=='complete',state
    assert state['ready'] and all(c['verified'] for c in state['components'].values())
    assert ('fixture/hi3d',True) in calls


def test_non_lfs_config_corruption_is_repaired_against_upstream_git_blob(isolated_models,monkeypatch):
    import huggingface_hub
    # Git-managed JSON/config files have a blob_id, not an LFS SHA256. A Hub
    # cache hit must not bless damaged bytes by replacing our local manifest.
    correct=b'{"fixture":"official configuration"}'
    blob=hashlib.sha1(f'blob {len(correct)}\0'.encode()+correct).hexdigest()
    for key in models.SPECS:
        (models.MODELS/key/'weights.bin').write_bytes(b'corrupt cached JSON')
    calls=[]
    class Api:
        def model_info(self,repo,**kwargs):
            return SimpleNamespace(sha='fixture-revision',siblings=[SimpleNamespace(rfilename='weights.bin',size=len(correct),lfs=None,blob_id=blob)])
    def transfer(repo,name,local_dir,force_download=False,**kwargs):
        target=Path(local_dir)/name;calls.append((repo,force_download))
        if force_download:target.write_bytes(correct)
        return str(target)
    monkeypatch.setattr(huggingface_hub,'HfApi',Api)
    monkeypatch.setattr(huggingface_hub,'hf_hub_download',transfer)
    assert models.download_all()
    deadline=time.monotonic()+10
    while models.LOCK.locked() and time.monotonic()<deadline:time.sleep(.02)
    assert not models.LOCK.locked()
    assert models.status()['ready'],models.status()
    assert all((models.MODELS/key/'weights.bin').read_bytes()==correct for key in models.SPECS)
    assert len([force for repo,force in calls if force])==3


def test_manifest_repository_must_match_fixed_component(isolated_models):
    manifest=json.loads((models.MODELS/'manifest.json').read_text())
    manifest['components']['hi3d']['repo']='another/model'
    models.atomic_json(models.MODELS/'manifest.json',manifest)
    result=models.verify_all()
    assert not result['ok'] and 'hi3d/manifest:repo' in result['errors']
    assert not models.status()['ready']


@pytest.mark.parametrize('kind,payload',[('manifest',b'{broken'),('manifest',b'{"components":[]}'),
    ('verification',b'{broken'),('verification',b'{"schema":1,"components":{"hi3d":[]}}'),
    ('download',b'{broken'),('download',b'{"state":["downloading"]}')])
def test_corrupt_model_state_is_readable_and_never_silently_trusted(isolated_models,kind,payload):
    assert models.verify_all()['ok']
    path={'manifest':models.MODELS/'manifest.json','verification':models.DATA/'model-verification.json',
          'download':models.DATA/'model-download.json'}[kind]
    path.write_bytes(payload)
    state=models.status()
    assert state['state_errors'] and path.name in ' '.join(state['state_errors'])
    assert '损坏' in state['download']['message']
    if kind!='download':assert not state['ready']
    else:assert state['download']['state']=='interrupted'  # Existing real hashes remain usable, with an explicit warning.
    assert path.read_bytes()==payload  # A read-only poll never overwrites evidence.
    if kind=='verification':
        assert models.verify_all()['ok']
        assert any(item.read_bytes()==payload for item in (models.DATA/'model-state-backups').iterdir())
    elif kind=='manifest':
        assert not models.verify_all()['ok'] and path.read_bytes()==payload


def test_download_repairs_corrupt_metadata_with_exact_backups(isolated_models,monkeypatch):
    import huggingface_hub
    broken={models.MODELS/'manifest.json':b'{broken manifest',models.DATA/'model-verification.json':b'{broken verification',
            models.DATA/'model-download.json':b'{broken download'}
    for path,payload in broken.items():path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
    sha=hashlib.sha256(isolated_models).hexdigest()
    class Api:
        def model_info(self,repo,**kwargs):
            return SimpleNamespace(sha='fixture-revision',siblings=[SimpleNamespace(rfilename='weights.bin',size=len(isolated_models),lfs={'sha256':sha})])
    monkeypatch.setattr(huggingface_hub,'HfApi',Api)
    monkeypatch.setattr(huggingface_hub,'hf_hub_download',lambda repo,name,local_dir,**kwargs:str(Path(local_dir)/name))
    assert models.download_all()
    deadline=time.monotonic()+10
    while models.LOCK.locked() and time.monotonic()<deadline:time.sleep(.02)
    assert not models.LOCK.locked()
    assert models.status()['ready'],models.status()
    backups=[path.read_bytes() for path in (models.DATA/'model-state-backups').iterdir()]
    assert all(payload in backups for payload in broken.values())
