"""Custom weight roots: local tiny files, no downloads or model inference."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
from filelock import FileLock

# Establish the same import-time sandbox as the rest of the API test suite.
from test_workbench import client, testroot
from hi3d_workbench import common, models, pipeline_models, server

assert common.ROOT == testroot


@pytest.fixture
def paths(client, tmp_path, monkeypatch):
    root=tmp_path/'工作台 根目录';root.mkdir()
    default=root/'models'
    monkeypatch.setattr(common,'ROOT',root)
    monkeypatch.setattr(common,'DEFAULT_MODELS',default)
    monkeypatch.setattr(common,'MODELS',default)
    monkeypatch.setattr(server,'ROOT',root)
    monkeypatch.setattr(server,'DATA',root/'data')
    monkeypatch.setattr(server,'MODELS',default)
    monkeypatch.setattr(server,'GPU_OPERATION',threading.Lock())
    monkeypatch.setattr(server,'VERIFY_LOCK',threading.Lock())
    monkeypatch.setattr(server,'PIPELINE_MODELS_LOCK',threading.Lock())
    monkeypatch.setattr(server,'MODEL_PATH_LOCK',threading.RLock())
    monkeypatch.setattr(server,'operation',None)
    monkeypatch.setattr(server,'verification_error','previous directory error')
    monkeypatch.setattr(server,'probe_recovery_workers',lambda:[])
    monkeypatch.setattr(server.manager,'active',None)
    monkeypatch.setattr(server.manager,'recovery_errors',[])
    monkeypatch.setattr(server.manager,'_recovery_workers',[])
    monkeypatch.setattr(models,'MODELS',default)
    monkeypatch.setattr(models,'DEFAULT_MODELS',default)
    monkeypatch.setattr(models,'DATA',root/'data')
    monkeypatch.setattr(models,'LOCK',threading.Lock())
    monkeypatch.setattr(models,'STATE',{'state':'idle','message':'fixture core idle'})
    monkeypatch.setattr(models,'FILE_LOCK',None)
    monkeypatch.setattr(pipeline_models,'ROOT',root)
    monkeypatch.setattr(pipeline_models,'MODEL_ROOT',default/'extensions')
    monkeypatch.setattr(pipeline_models,'_LOCK',threading.Lock())
    monkeypatch.setattr(pipeline_models,'_STATE',{'state':'idle','message':'fixture extensions idle'})
    monkeypatch.setattr(pipeline_models,'_STATE_PATH',None)
    common.atomic_json(root/'config.local.json',{'port':8382,'download_workers':3,'custom_setting':{'keep':True}})
    yield SimpleNamespace(client=client,root=root,default=default,custom=tmp_path/'模型 仓库',monkeypatch=monkeypatch)


def test_get_default_does_not_create_weight_directory(paths):
    result=paths.client.get('/api/model-path')
    assert result.status_code==200
    value=result.json()
    assert value=={'models_dir':'','resolved_path':str(paths.default),'default_path':str(paths.default),
                   'extensions_path':str(paths.default/'extensions'),'is_default':True,'exists':False,
                   'restart_required':False,'busy':False,'busy_reason':None}
    assert not paths.default.exists()


@pytest.mark.parametrize('relative',[False,True])
def test_save_immediately_switches_all_roots_and_preserves_settings(paths,relative):
    value='共享 权重' if relative else str(paths.custom)
    expected=paths.root/value if relative else paths.custom
    result=paths.client.post('/api/model-path',json={'models_dir':value})
    assert result.status_code==200,result.text
    data=result.json()
    assert data['resolved_path']==str(expected.resolve())
    assert data['models_dir']==value and data['restart_required'] is False
    assert not data['busy'] and not data['exists'] and not data['is_default']
    assert common.MODELS==server.MODELS==models.MODELS==expected.resolve()
    assert pipeline_models.MODEL_ROOT==expected.resolve()/'extensions'
    assert server.verification_error is None
    saved=common.read_json(paths.root/'config.local.json')
    assert saved=={'port':8382,'download_workers':3,'custom_setting':{'keep':True},'models_dir':value}
    assert not expected.exists()  # Saving is not a download or migration.
    health=paths.client.get('/api/health').json()
    assert health['model_path']==data
    assert health['models']['model_root']==str(expected.resolve())
    assert health['pipeline_models']['model_root']==str(expected.resolve()/'extensions')


def test_relative_paths_are_fixed_to_workbench_root_not_current_directory(paths,tmp_path):
    elsewhere=tmp_path/'unrelated cwd';elsewhere.mkdir()
    paths.monkeypatch.chdir(elsewhere)
    response=paths.client.post('/api/model-path',json={'models_dir':'../外部 模型'})
    assert response.status_code==200
    assert Path(response.json()['resolved_path'])==(paths.root/'../外部 模型').resolve()
    assert not (elsewhere/'../外部 模型').resolve().exists()


@pytest.mark.parametrize('empty',['',None,'   '])
def test_restore_default_never_moves_or_deletes_weights(paths,empty):
    for root,payload in ((paths.default,b'original'),(paths.custom,b'custom')):
        target=root/'hi3d'/'weights.bin';target.parent.mkdir(parents=True);target.write_bytes(payload)
    assert paths.client.post('/api/model-path',json={'models_dir':str(paths.custom)}).status_code==200
    result=paths.client.post('/api/model-path',json={'models_dir':empty})
    assert result.status_code==200 and result.json()['is_default']
    assert result.json()['models_dir']=='' and result.json()['exists']
    assert models.MODELS==paths.default and pipeline_models.MODEL_ROOT==paths.default/'extensions'
    assert (paths.default/'hi3d/weights.bin').read_bytes()==b'original'
    assert (paths.custom/'hi3d/weights.bin').read_bytes()==b'custom'


@pytest.mark.parametrize('value',[123,False,[],{},'bad\x00path','bad\npath'])
def test_invalid_path_is_rejected_without_changing_configuration(paths,value):
    before=(paths.root/'config.local.json').read_bytes()
    response=paths.client.post('/api/model-path',json={'models_dir':value})
    assert response.status_code in {400,422}
    assert (paths.root/'config.local.json').read_bytes()==before
    assert common.MODELS==models.MODELS==server.MODELS==paths.default


@pytest.mark.parametrize('as_child',[False,True])
def test_existing_file_or_file_ancestor_cannot_be_selected(paths,as_child):
    target=paths.root/'a file';target.write_bytes(b'preserve')
    response=paths.client.post('/api/model-path',json={'models_dir':str(target/'models' if as_child else target)})
    assert response.status_code==400
    assert target.read_bytes()==b'preserve' and models.MODELS==paths.default


def test_default_path_cannot_be_an_existing_file(paths):
    paths.default.write_bytes(b'not a directory')
    before=(paths.root/'config.local.json').read_bytes()
    assert paths.client.post('/api/model-path',json={'models_dir':''}).status_code==400
    assert paths.default.read_bytes()==b'not a directory'
    assert (paths.root/'config.local.json').read_bytes()==before


@pytest.mark.skipif(os.name!='nt',reason='Windows path rules')
@pytest.mark.parametrize('value',['C:relative','\\relative','folder/CON','bad?name','bad:name','folder.'])
def test_ambiguous_or_reserved_windows_paths_are_rejected(paths,value):
    assert paths.client.post('/api/model-path',json={'models_dir':value}).status_code==400


def test_path_mutation_keeps_session_and_origin_protection(paths):
    assert paths.client.post('/api/model-path',json={'models_dir':str(paths.custom)},headers={'x-hi3d-token':'bad'}).status_code==403
    assert paths.client.post('/api/model-path',json={'models_dir':''},headers={'origin':'https://external.example'}).status_code==403
    assert common.MODELS==paths.default


@pytest.mark.parametrize('busy',['task','gpu','core_verify','core_download','pipeline_operation','pipeline_download','operation','recovery','probe'])
def test_busy_operations_block_switch_and_report_reason(paths,busy):
    lock=None
    if busy=='task':paths.monkeypatch.setattr(server.manager,'active','running-test-job')
    elif busy=='operation':paths.monkeypatch.setattr(server,'operation','diagnosing')
    elif busy=='recovery':
        import psutil
        paths.monkeypatch.setattr(server.manager,'_recovery_workers',[(os.getpid(),psutil.Process().create_time())])
    elif busy=='probe':paths.monkeypatch.setattr(server,'probe_recovery_workers',lambda:[{'pid':123,'created':1}])
    else:
        lock={'gpu':server.GPU_OPERATION,'core_verify':server.VERIFY_LOCK,'core_download':models.LOCK,
              'pipeline_operation':server.PIPELINE_MODELS_LOCK,'pipeline_download':pipeline_models._LOCK}[busy]
        lock.acquire()
    before=(paths.root/'config.local.json').read_bytes()
    try:
        info=paths.client.get('/api/model-path').json()
        assert info['busy'] and info['busy_reason']
        assert paths.client.post('/api/model-path',json={'models_dir':str(paths.custom)}).status_code==409
        assert (paths.root/'config.local.json').read_bytes()==before
        assert common.MODELS==models.MODELS==server.MODELS==paths.default
    finally:
        if lock:lock.release()
    # No failed save may leave an extra operation guard held.
    if busy not in {'task','operation','recovery','probe'}:
        assert paths.client.post('/api/model-path',json={'models_dir':str(paths.custom)}).status_code==200


@pytest.mark.parametrize('which',['old_core','old_extensions','target_core','target_extensions'])
def test_actual_download_locks_block_switch_on_old_and_new_roots(paths,which):
    location={'old_core':paths.default,'old_extensions':paths.default/'extensions',
              'target_core':paths.custom,'target_extensions':paths.custom/'extensions'}[which]
    location.mkdir(parents=True)
    with FileLock(str(location/'.download.lock'),thread_local=False):
        response=paths.client.post('/api/model-path',json={'models_dir':str(paths.custom)})
        assert response.status_code==409,response.text
        assert models.MODELS==paths.default
    assert paths.client.post('/api/model-path',json={'models_dir':str(paths.custom)}).status_code==200


def _tiny_core(root):
    specs={key:{'repo':'fixture/'+key,'revision':'fixture-revision','files':['weights.bin'],'license':'fixture'}
           for key in ('hi3d','dino','clip')}
    manifest={'components':{}}
    for key,spec in specs.items():
        path=root/key/'weights.bin';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture')
        manifest['components'][key]={'repo':spec['repo'],'revision':spec['revision'],
                                     'files':{'weights.bin':{'sha256':hashlib.sha256(b'fixture').hexdigest(),'bytes':7}}}
    common.atomic_json(root/'manifest.json',manifest)
    return specs


def test_verification_and_download_receipts_are_bound_to_selected_root(paths):
    paths.monkeypatch.setattr(models,'SPECS',_tiny_core(paths.default))
    assert models.verify_all()['ok'] and models.status()['ready']
    # A complete receipt for another directory must not look like this root's download.
    common.atomic_json(models.DATA/'model-download.json',{'state':'complete','message':'OLD DIRECTORY',
                                                          'model_root':str(paths.default.resolve())})
    _tiny_core(paths.custom)
    assert paths.client.post('/api/model-path',json={'models_dir':str(paths.custom)}).status_code==200
    info=models.status()
    assert all(component['present'] for component in info['components'].values())
    assert not info['ready'] and info['download']['message']!='OLD DIRECTORY'
    result=paths.client.post('/api/models/verify')
    assert result.status_code==200 and result.json()['ok']
    assert result.json()['model_root']==str(paths.custom)
    assert models.status()['ready']
    assert common.read_json(models.DATA/'model-verification.json')['model_root']==str(paths.custom)


def test_download_starts_in_new_directory_without_network(paths):
    assert paths.client.post('/api/model-path',json={'models_dir':str(paths.custom)}).status_code==200
    captured=[]
    class DeferredThread:
        def __init__(self,**kwargs):captured.append(kwargs['target'])
        def start(self):pass
    paths.monkeypatch.setattr(models.threading,'Thread',DeferredThread)
    try:
        assert models.download_all()
        assert (paths.custom/'.download.lock').is_file()
        receipt=common.read_json(models.DATA/'model-download.json')
        assert receipt['state']=='downloading' and receipt['model_root']==str(paths.custom)
        assert len(captured)==1 and not paths.default.exists()
    finally:
        if models.FILE_LOCK is not None:models.FILE_LOCK.release()
        if models.LOCK.locked():models.LOCK.release()


def test_extension_verify_reads_and_updates_only_selected_directory(paths):
    specs={key:{**spec,'files':['weights.bin'],'artifacts':{}} for key,spec in pipeline_models.SPECS.items()}
    paths.monkeypatch.setattr(pipeline_models,'SPECS',specs)
    extension=paths.custom/'extensions'
    manifest={'components':{}}
    for key,spec in specs.items():
        target=extension/key/'weights.bin';target.parent.mkdir(parents=True);target.write_bytes(b'extension')
        manifest['components'][key]={'repo':spec['repo'],'revision':spec['revision'],
            'files':{'weights.bin':{'hash':hashlib.sha256(b'extension').hexdigest(),'algorithm':'sha256',
                                    'bytes':9,'verified':False}}}
    common.atomic_json(extension/'manifest.json',manifest)
    assert paths.client.post('/api/model-path',json={'models_dir':str(paths.custom)}).status_code==200
    response=paths.client.post('/api/pipeline/models/verify')
    assert response.status_code==200 and not response.json()['verification_errors']
    assert response.json()['model_root']==str(extension)
    checked=common.read_json(extension/'manifest.json')
    assert all(entry['files']['weights.bin']['verified'] for entry in checked['components'].values())
    assert not paths.default.exists()


def test_switch_cannot_reuse_verification_even_for_hardlinked_weights(paths):
    specs=_tiny_core(paths.default)
    paths.monkeypatch.setattr(models,'SPECS',specs)
    manifest=common.read_json(paths.default/'manifest.json')
    for key in specs:
        target=paths.custom/key/'weights.bin';target.parent.mkdir(parents=True)
        try:os.link(paths.default/key/'weights.bin',target)
        except OSError as exc:pytest.skip(f'filesystem does not support hardlinks: {exc}')
    common.atomic_json(paths.custom/'manifest.json',manifest)
    # Verify after making hardlinks so even ctime/change-time are identical.
    assert models.verify_all()['ok'] and models.status()['ready']
    assert paths.client.post('/api/model-path',json={'models_dir':str(paths.custom)}).status_code==200
    assert not models.status()['ready']


def test_pipeline_status_does_not_reuse_another_roots_transient_state(paths):
    pipeline_models._STATE.update(state='failed',message='OLD TRANSIENT FAILURE')
    paths.monkeypatch.setattr(pipeline_models,'_STATE_PATH',paths.default/'extensions/download-state.json')
    state=pipeline_models.status(paths.custom/'extensions')
    assert state['download']['state']=='idle'
    assert state['download']['message']!='OLD TRANSIENT FAILURE'
    assert state['model_root']==str(paths.custom/'extensions')


def test_failed_config_commit_keeps_all_current_roots(paths):
    before=(paths.root/'config.local.json').read_bytes()
    def fail(*args):raise OSError('simulated write failure')
    paths.monkeypatch.setattr(server,'atomic_json',fail)
    with pytest.raises(OSError,match='simulated'):
        server.save_model_path(server.ModelPathInput(models_dir=str(paths.custom)))
    assert common.MODELS==models.MODELS==server.MODELS==paths.default
    assert pipeline_models.MODEL_ROOT==paths.default/'extensions'
    assert (paths.root/'config.local.json').read_bytes()==before
    assert not server.GPU_OPERATION.locked() and not server.VERIFY_LOCK.locked()


@pytest.mark.parametrize('value',['relative 模型',None])
def test_new_worker_process_reads_the_same_persisted_paths(paths,value,tmp_path):
    response=paths.client.post('/api/model-path',json={'models_dir':value})
    assert response.status_code==200
    project=Path(__file__).resolve().parents[1]
    script=('import json;from hi3d_workbench.common import MODELS,ROOT;'
            'from hi3d_workbench.pipeline_models import MODEL_ROOT;'
            'print(json.dumps([str(ROOT),str(MODELS),str(MODEL_ROOT)]))')
    result=subprocess.run([sys.executable,'-c',script],cwd=tmp_path,
                          env={**os.environ,'HI3D_WORKBENCH_ROOT':str(paths.root),'PYTHONPATH':str(project)},
                          text=True,encoding='utf-8',capture_output=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)==[str(paths.root),response.json()['resolved_path'],response.json()['extensions_path']]
