"""Exercise concurrent requests and legacy assets without model processes."""
import io
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest


@pytest.fixture
def isolated_server(tmp_path,monkeypatch):
    # No application imports during collection: test_workbench establishes its
    # own import root, while this fixture also works safely when run alone.
    monkeypatch.setenv('HI3D_WORKBENCH_ROOT',str(tmp_path))
    from fastapi.testclient import TestClient
    from hi3d_workbench import server,jobs
    data=tmp_path/'data';assets=data/'assets';assets.mkdir(parents=True,exist_ok=True)
    monkeypatch.setattr(server,'ROOT',tmp_path)
    monkeypatch.setattr(server,'DATA',data)
    monkeypatch.setattr(server,'MODELS',tmp_path/'models')
    monkeypatch.setattr(server,'ASSETS',assets)
    monkeypatch.setattr(jobs,'DATA',data)
    manager=jobs.JobManager()
    monkeypatch.setattr(server,'manager',manager)
    monkeypatch.setattr(server,'GPU_OPERATION',threading.Lock())
    monkeypatch.setattr(server,'VERIFY_LOCK',threading.Lock())
    monkeypatch.setattr(server,'operation',None)
    monkeypatch.setattr(server,'verification_error',None)
    monkeypatch.setattr(server,'config',lambda:{'worker_python':sys.executable,'worker_command':[],'port':8378})
    monkeypatch.setattr(server.models,'DATA',data)
    monkeypatch.setattr(server.models,'MODELS',tmp_path/'models')
    monkeypatch.setattr(server.models,'LOCK',threading.Lock())
    monkeypatch.setattr(server.models,'status',lambda:{'ready':True,'components':{'fixture':{'present':True}},'download':{'state':'idle'}})
    monkeypatch.setattr(server.models,'verify_all',lambda:{'ok':True,'errors':[]})
    actual_popen=server.subprocess.Popen
    def controlled_probe_only(command,*args,**kwargs):
        from pathlib import Path
        # Explicit temporary probe scripts may exercise process cleanup;
        # the production -m worker must remain unreachable in service tests.
        if len(command)>1 and command[0]==sys.executable and Path(command[1]).resolve().is_relative_to(data) and Path(command[1]).is_file():
            return actual_popen(command,*args,**kwargs)
        pytest.fail('A regression must never launch uncontrolled GPU diagnostics')
    monkeypatch.setattr(server.subprocess,'Popen',controlled_probe_only)
    monkeypatch.setattr(manager,'create',lambda *args,**kwargs:pytest.fail('A regression must never launch inference'))
    assert server.DATA.is_relative_to(tmp_path) and manager.root.is_relative_to(tmp_path)
    with TestClient(server.app,base_url='http://127.0.0.1') as client:
        client.headers['x-hi3d-token']=client.get('/api/session').json()['token']
        yield server,client


def test_only_loopback_hosts_are_trusted(isolated_server):
    server,client=isolated_server
    assert client.get('/api/session',headers={'host':'127.0.0.1:8378'}).status_code==200
    assert client.get('/api/session',headers={'host':'localhost:8378'}).status_code==200
    assert client.get('/api/session',headers={'host':'testserver'}).status_code==400
    assert client.get('/api/session',headers={'host':'attacker.example'}).status_code==400


def _input_pair(client):
    from PIL import Image
    import trimesh
    image=io.BytesIO();Image.new('RGBA',(16,16),(80,100,120,128)).save(image,format='PNG')
    a=client.post('/api/assets',files={'file':('reference.png',image.getvalue(),'image/png')})
    b=client.post('/api/assets',files={'file':('coarse.glb',trimesh.creation.box().export(file_type='glb'),'application/octet-stream')})
    assert a.status_code==b.status_code==200
    return {'image':a.json()['id'],'mesh':b.json()['id'],'attention':'sdpa'}


def test_preflight_blocks_duplicate_submit_and_gpu_diagnostic(isolated_server,monkeypatch):
    server,client=isolated_server
    inputs=_input_pair(client)
    entered=threading.Event();release=threading.Event();created=[]
    def verify():
        entered.set()
        assert release.wait(10),'Test never released the simulated weight verification'
        return {'ok':True,'errors':[]}
    def create(image,mesh,options):
        created.append(options)
        return {'id':'simulated-job','state':'queued','options':options}
    monkeypatch.setattr(server.models,'verify_all',verify)
    monkeypatch.setattr(server.manager,'create',create)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending=pool.submit(client.post,'/api/jobs',json=inputs)
        try:
            assert entered.wait(5)
            health=client.get('/api/health').json()
            assert health['operation']=='preparing' and health['model_checking']
            duplicate=client.post('/api/jobs',json=inputs)
            diagnostic=client.post('/api/diagnostics')
            assert duplicate.status_code==diagnostic.status_code==409
            assert created==[]
        finally:
            release.set()
        assert pending.result(timeout=5).status_code==200
    assert len(created)==1
    assert server.operation is None and not server.GPU_OPERATION.locked() and not server.VERIFY_LOCK.locked()


def test_active_inference_rejects_diagnostic(isolated_server):
    server,client=isolated_server
    server.manager.active='simulated-running-job'
    try:
        result=client.post('/api/diagnostics')
        assert result.status_code==409 and '已有精修任务' in result.json()['detail']
        assert client.get('/api/health').json()['active']=='simulated-running-job'
        assert not server.GPU_OPERATION.locked()
    finally:
        server.manager.active=None


def _legacy_job(server,name='legacy-job'):
    from PIL import Image
    import trimesh
    folder=server.manager.root/name;folder.mkdir()
    Image.new('RGBA',(24,16),(30,90,150,128)).save(folder/'input.png')
    trimesh.creation.box(extents=[1,2,3]).export(folder/'coarse.glb')
    options={'resolution':512,'steps':25,'seed':3,'offload':True,'attention':'sdpa'}
    server.atomic_json(folder/'state.json',{'id':name,'state':'failed','stage':'failed','created':1,'options':options})
    return folder,options


def test_legacy_reuse_creates_assets_and_preserves_exact_originals(isolated_server):
    server,client=isolated_server
    folder,options=_legacy_job(server)
    before={p.name:p.read_bytes() for p in folder.iterdir()}
    response=client.post('/api/jobs/legacy-job/reuse')
    assert response.status_code==200,response.text
    result=response.json()
    assert result['options']==options
    assert result['image']['id']!=result['mesh']['id']
    assert client.get(f"/api/assets/{result['image']['id']}/image.png").content==before['input.png']
    assert client.get(result['mesh']['url']).content==before['coarse.glb']
    assert result['mesh']['geometry']['faces']==12
    repeated=client.post('/api/jobs/legacy-job/reuse').json()
    assert repeated['image']['id']!=result['image']['id']
    assert repeated['mesh']['id']!=result['mesh']['id']
    assert {p.name:p.read_bytes() for p in folder.iterdir()}==before


def test_job_available_files_reports_only_real_allowed_artifacts(isolated_server):
    server,client=isolated_server
    folder,_=_legacy_job(server)
    (folder/'worker.log').write_text('simulated error',encoding='utf-8')
    (folder/'private.txt').write_text('not an exported artifact',encoding='utf-8')
    result=client.get('/api/jobs/legacy-job').json()
    assert set(result['available_files'])=={'coarse.glb','input.png','worker.log'}
    assert 'refined.glb' not in result['available_files']
    assert client.get('/api/jobs/legacy-job/files/refined.glb').status_code==404
    (folder/'refined.glb').write_bytes((folder/'coarse.glb').read_bytes())
    result=client.get('/api/jobs/legacy-job').json()
    assert 'refined.glb' in result['available_files'] and 'refined.stl' not in result['available_files']
    assert client.get('/api/jobs/legacy-job/files/private.txt').status_code==404
