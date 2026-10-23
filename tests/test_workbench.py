import io
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pytest
import trimesh
from PIL import Image

# Isolate test artifacts from the user's workbench jobs and models.
import tempfile
testroot=Path(tempfile.mkdtemp(prefix='hi3d-tests-'))
os.environ['HI3D_WORKBENCH_ROOT']=str(testroot)
from hi3d_workbench import common
if common.ROOT != testroot:
    raise RuntimeError('测试隔离路径未生效：请在新的 Python 进程运行 pytest；已停止以保护现有模型与任务')
from hi3d_workbench.geometry import load_mesh,inspect_mesh,export_mesh
from hi3d_workbench.server import app,manager,models
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    with TestClient(app,base_url='http://127.0.0.1') as c:
        c.headers['x-hi3d-token']=c.get('/api/session').json()['token']
        yield c

def upload_pair(client):
    buf=io.BytesIO();Image.new('RGBA',(64,64),(80,200,150,128)).save(buf,format='PNG')
    a=client.post('/api/assets',files={'file':('中文 空格.png',buf.getvalue(),'image/png')})
    m=trimesh.creation.box().export(file_type='glb')
    b=client.post('/api/assets',files={'file':('model.glb',m,'application/octet-stream')})
    assert a.status_code==b.status_code==200
    return a.json(),b.json()

def test_inputs_and_alpha(client):
    a,b=upload_pair(client)
    im=Image.open(io.BytesIO(client.get('/api/assets/'+a['id']+'/image.png').content))
    assert im.getextrema()[3]==(128,128)
    assert b['geometry']['watertight'] and b['geometry']['components']==1
    assert '壁厚' in b['geometry']['unchecked']

def test_reject_invalid_and_external_resource(client,tmp_path):
    assert client.post('/api/assets',files={'file':('bad.glb',b'invalid')}).status_code==400
    assert client.post('/api/assets',files={'file':('external.gltf',b'{}')}).status_code==400
    buf=io.BytesIO();np.savez(buf,vertices=[[float('nan'),0,0]],faces=[[0,0,0]])
    assert client.post('/api/assets',files={'file':('bad.npz',buf.getvalue())}).status_code==400
    # A .glb container can still refer to external buffers. Keep uploads self-contained.
    import struct
    exported=trimesh.creation.box().export(file_type='gltf')
    tree=json.loads(exported['model.gltf'])
    for buffer in tree['buffers']:
        name=buffer['uri'];external=tmp_path/name
        external.write_bytes(exported[name]);buffer['uri']=str(external)
    header=json.dumps(tree).encode();header+=b' '*((-len(header))%4)
    glb=struct.pack('<4sII',b'glTF',2,20+len(header))+struct.pack('<I4s',len(header),b'JSON')+header
    assert client.post('/api/assets',files={'file':('external.glb',glb)}).status_code==400

def test_origin_and_token(client):
    assert client.post('/api/models/verify',headers={'x-hi3d-token':'wrong'}).status_code==403
    assert client.post('/api/models/verify',headers={'origin':'http://attacker.example'}).status_code==403
    assert client.get('/api/jobs/none/files/request.json').status_code==404

def test_models_required_and_params(client):
    a,b=upload_pair(client)
    assert client.post('/api/jobs',json={'image':a['id'],'mesh':b['id'],'resolution':256}).status_code==400
    assert client.post('/api/jobs',json={'image':a['id'],'mesh':b['id']}).status_code==409
    assert client.post('/api/jobs',json={'image':a['id'],'mesh':b['id'],'steps':0}).status_code==422
    assert client.post('/api/jobs',json={'image':a['id'],'mesh':b['id'],'seed':2**53}).status_code==422

def test_export_height_and_does_not_modify_source(client):
    folder=manager.root/'export-test';folder.mkdir(exist_ok=True)
    mesh=trimesh.creation.box(extents=[2,3,4]);mesh.export(folder/'coarse.glb')
    before=(folder/'coarse.glb').read_bytes()
    r=client.post('/api/jobs/export-test/export',json={'source':'coarse','format':'stl','height_mm':80})
    assert r.status_code==200
    out=trimesh.load(io.BytesIO(client.get(r.json()['url']).content),file_type='stl')
    assert out.extents[2]==pytest.approx(80)
    r=client.post('/api/jobs/export-test/export',json={'source':'coarse','format':'glb','height_mm':80})
    out=trimesh.load(io.BytesIO(client.get(r.json()['url']).content),file_type='glb',force='mesh')
    assert out.extents[2]==pytest.approx(.08)
    assert r.json()['units']=='m' and r.json()['height_mm']==80
    assert (folder/'coarse.glb').read_bytes()==before

def test_scene_instance_transforms_and_components():
    scene=trimesh.Scene();scene.add_geometry(trimesh.creation.box(),node_name='a')
    transform=np.eye(4);transform[0,0]=-1;transform[0,3]=4
    scene.add_geometry(trimesh.creation.box(),node_name='b',transform=transform)
    p=testroot/'scene.glb';scene.export(p)
    m=load_mesh(p)
    assert m.extents[0]==pytest.approx(5)
    assert inspect_mesh(m)['components']==2

def test_complete_result_required_and_cancel(monkeypatch):
    from hi3d_workbench.jobs import JobManager
    atomic=common.atomic_json
    atomic(testroot/'config.local.json',{'worker_python':sys.executable,'worker_command':[sys.executable]})
    # Subprocess with a fake worker may validate orchestration only, never model quality.
    script=testroot/'fake_worker.py';script.write_text('import time\nprint("start",flush=True)\ntime.sleep(20)\n')
    atomic(testroot/'config.local.json',{'worker_command':[sys.executable,str(script)]})
    jm=JobManager();im=testroot/'in.png';im.write_bytes(b'x');m=testroot/'in.glb';m.write_bytes(b'x')
    job=jm.create(im,m,{'resolution':512});jid=job['id']
    for _ in range(100):
        if jm.process:break
        time.sleep(.02)
    with pytest.raises(RuntimeError):jm.create(im,m,{})
    jm.cancel(jid)
    for _ in range(100):
        if jm.get(jid)['state']=='cancelled':break
        time.sleep(.03)
    assert jm.get(jid)['state']=='cancelled'
    assert (jm.root/jid/'input.png').is_file()
    assert not (jm.root/jid/'refined.glb').exists()

def test_bridge_padding_and_batch_rejection():
    # Bridge accepts numpy arrays too; use no torch dependency in this test.
    from types import SimpleNamespace
    import importlib.util
    spec=importlib.util.spec_from_file_location('hi3d_test_nodes',Path(__file__).parents[1]/'comfy_nodes.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    m=SimpleNamespace(vertices=np.zeros((1,6,3)),faces=np.zeros((1,4,3),dtype=np.int64),vertex_counts=[3],face_counts=[1])
    v,f=mod.arrays(m);assert v.shape==(3,3) and f.shape==(1,3)
    m.vertices=np.zeros((2,6,3));m.faces=np.zeros((2,4,3),dtype=np.int64)
    with pytest.raises(ValueError):mod.arrays(m)

def test_manifest_corruption_detected(tmp_path,monkeypatch):
    monkeypatch.setattr(common,'MODELS',tmp_path/'models')
    monkeypatch.setattr(models,'MODELS',tmp_path/'models')
    monkeypatch.setattr(models,'DATA',tmp_path/'data')
    manifest={'components':{}}
    for key,spec in models.SPECS.items():
        entry={'files':{},'repo':spec['repo'],'revision':spec['revision']}
        for name in spec['files']:
            p=common.MODELS/key/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
            entry['files'][name]={'sha256':common.digest(p)}
        manifest['components'][key]=entry
    common.atomic_json(common.MODELS/'manifest.json',manifest)
    assert models.verify_all()['ok']
    (common.MODELS/'hi3d/vae/model.safetensors').write_bytes(b'corrupt')
    assert not models.verify_all()['ok']

def test_second_server_rejected_without_changing_jobs(client):
    before=manager.list()
    with pytest.raises(RuntimeError,match='已有服务'):
        with TestClient(app,base_url='http://127.0.0.1'):pass
    assert manager.list()==before
    assert client.get('/api/health').status_code==200

def test_recovery_marks_incomplete_tasks():
    from hi3d_workbench.jobs import JobManager
    folder=manager.root/'interrupted-fixture'
    common.atomic_json(folder/'state.json',{'id':folder.name,'state':'running','created':1})
    another=JobManager();another.recover()
    assert another.get(folder.name)['state']=='interrupted'
