import json
import threading
import time
from pathlib import Path

import pytest
from test_workbench import client
from hi3d_workbench import server, runtime_acceleration
from hi3d_workbench.engines.acceleration_wheels import VOXEL_ACCELERATION_WHEEL
from tools import setup_pipeline_runtime as setup


@pytest.fixture
def environment():
    return {'platform':'win32','python':[3,12],'implementation':'CPython','machine':'AMD64',
            'pointer_bits':64,'torch':'2.8.0+cu128','cuda':'12.8'}


def test_optional_voxel_verifies_before_install_and_preserves_runtime(tmp_path,environment,monkeypatch):
    old={'ready':True,'gpu_validation':'historic','custom':{'keep':1}}
    (tmp_path/'pipeline-runtime.json').write_text(json.dumps(old))
    commands=[]
    monkeypatch.setattr(setup,'probe_attention_environment',lambda *a:environment)
    monkeypatch.setattr(setup,'download_verified_wheel',lambda folder,spec:folder/spec['filename'])
    monkeypatch.setattr(setup,'run',lambda command:commands.append(command))
    monkeypatch.setattr(setup,'probe_voxel_import',lambda *a:{'import_validation':'passed','kernel_probe':'passed',
        'build_id':VOXEL_ACCELERATION_WHEEL['build_id']})
    result=setup.install_optional_voxel(tmp_path/'python.exe',tmp_path)
    assert result['status']=='installed' and result['wheel_verified']
    assert result['full_model_validation']=='not_run'
    assert len(commands)==1
    assert all(flag in commands[0] for flag in ['--no-deps','--no-index','--no-cache-dir'])
    assert 'o_voxel' not in ' '.join(map(str,commands[0]))
    assert json.loads((tmp_path/'pipeline-runtime.json').read_text())=={**old,'voxel_acceleration':result}


def test_optional_voxel_abi_mismatch_does_not_download_or_change_runtime(tmp_path,environment,monkeypatch):
    environment['torch']='2.11.0+cu128'
    monkeypatch.setattr(setup,'probe_attention_environment',lambda *a:environment)
    monkeypatch.setattr(setup,'download_verified_wheel',lambda *a:pytest.fail('incompatible ABI cannot download'))
    monkeypatch.setattr(setup,'run',lambda *a:pytest.fail('incompatible ABI cannot install'))
    result=setup.install_optional_voxel(tmp_path/'python.exe',tmp_path)
    assert result['status']=='unavailable' and result['backend']=='cpu:official'
    assert not result['wheel_verified'] and 'ABI mismatch' in result['reason']
    assert not (tmp_path/'pipeline-runtime.json').exists()


def test_voxel_only_never_installs_base_dependencies(tmp_path,monkeypatch):
    python=tmp_path/'python.exe';python.touch()
    calls=[]
    def install(*args):calls.append(args);return {'status':'installed'}
    monkeypatch.setattr(setup,'install_optional_voxel',install)
    monkeypatch.setattr(setup,'run',lambda *a:pytest.fail('voxel-only cannot install base runtime'))
    setup.main(['--runtime',str(tmp_path),'--voxel-only'])
    assert calls==[(python,tmp_path)]


@pytest.mark.parametrize('receipt',[{},[],{'status':'installed','wheel':[]},
    {'status':'installed','wheel_verified':True,'kernel_probe':'passed','wheel':{'sha256':'outdated'}}])
def test_status_never_claims_invalid_receipt_is_installed(tmp_path,monkeypatch,receipt):
    monkeypatch.setattr(runtime_acceleration,'runtime_root',lambda:tmp_path)
    (tmp_path/'voxel-acceleration.json').write_text(json.dumps(receipt))
    assert not runtime_acceleration.status()['installed']


def test_installer_holds_gpu_lock_until_background_completion(client,monkeypatch,tmp_path):
    entered=threading.Event();release=threading.Event();finished=threading.Event()
    monkeypatch.setattr(runtime_acceleration,'status',lambda:{'runtime_exists':True,'installed':True,'receipt':{}})
    monkeypatch.setattr(runtime_acceleration,'runtime_root',lambda:tmp_path)
    monkeypatch.setattr(server.manager,'close_resident',lambda:None)
    def probe(command):
        assert command[-1]=='--voxel-only'
        entered.set();release.wait(5);return 'verified',0
    monkeypatch.setattr(server,'run_probe',probe)
    try:
        response=client.post('/api/runtime/voxel-acceleration/install')
        assert response.status_code==200 and entered.wait(2)
        assert server.operation=='installing_acceleration' and server.GPU_OPERATION.locked()
        assert client.post('/api/runtime/unload').status_code==409
        assert client.post('/api/runtime/voxel-acceleration/install').status_code==409
    finally:
        release.set()
        deadline=time.monotonic()+5
        while server.GPU_OPERATION.locked() and time.monotonic()<deadline:time.sleep(.01)
    assert not server.GPU_OPERATION.locked() and server.operation is None
    assert server.ACCELERATION_INSTALL['state']=='succeeded'


def test_thread_start_failure_releases_gpu_guard(client,monkeypatch):
    monkeypatch.setattr(runtime_acceleration,'status',lambda:{'runtime_exists':True,'installed':False,'receipt':{}})
    monkeypatch.setattr(server.manager,'close_resident',lambda:None)
    class Thread:
        def __init__(self,*a,**kw):pass
        def start(self):raise RuntimeError('fixture thread failure')
    monkeypatch.setattr(server.threading,'Thread',Thread)
    with pytest.raises(RuntimeError,match='fixture thread failure'):
        client.post('/api/runtime/voxel-acceleration/install')
    assert server.operation is None and not server.GPU_OPERATION.locked()
