"""Asset-pipeline boundaries; temporary fixtures are not model-quality evidence.

Reuse test_workbench's import-time isolation rather than rebinding an already
imported ROOT. No test downloads weights or invokes an actual model engine.
"""
import io
import json
import os
import struct
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import numpy as np
import psutil
import pytest
import trimesh
from PIL import Image

from test_workbench import client, upload_pair, testroot
from hi3d_workbench import common, server, jobs, pipeline_models, pipeline_worker
from hi3d_workbench.asset_pipeline import inspect_textured_glb, export_textured_glb

assert common.ROOT == testroot
assert server.DATA.is_relative_to(testroot)


@pytest.fixture
def backend(client,monkeypatch):
    readiness={'hi3d':True,'coarse':True,'texture':True,'background':True}
    captures=[];checks=[]
    monkeypatch.setattr(server.models,'LOCK',threading.Lock())
    monkeypatch.setattr(server.models,'status',lambda:{'ready':readiness['hi3d'],'components':{},'download':{'state':'idle'}})
    monkeypatch.setattr(server.models,'verify_all',lambda:checks.append('hi3d') or {'ok':True,'errors':[]})
    monkeypatch.setattr(server.pipeline_models,'status',lambda:{'coarse_ready':readiness['coarse'],'texture_ready':readiness['texture'],'background_ready':readiness['background']})
    def capture(image,mesh,options):
        captures.append((image,mesh,options))
        return {'id':'orchestration-fixture','state':'queued','options':options}
    monkeypatch.setattr(server.manager,'create',capture)
    yield client,readiness,captures,checks


@pytest.mark.parametrize('mode,flags,needs_mesh,requires_hi3d,requires_coarse,requires_texture',[
    ('refine',{},True,True,False,False),
    ('refine',{'generate_texture':True},True,True,False,True),
    ('texture',{},True,False,False,True),
    ('image_to_asset',{},False,False,True,True),
    ('image_to_asset',{'refine_geometry':True},False,True,True,True),
    ('image_to_asset',{'refine_geometry':False},False,False,True,True),
    ('image_to_asset',{'refine_geometry':False,'generate_texture':False},False,False,True,False),
])
def test_modes_require_only_their_declared_model_components(backend,mode,flags,needs_mesh,requires_hi3d,requires_coarse,requires_texture):
    client,ready,captures,checks=backend
    image,mesh=upload_pair(client)
    payload={'image':image['id'],'mode':mode,'attention':'sdpa',**flags}
    if needs_mesh:payload['mesh']=mesh['id']
    requirements={'hi3d':requires_hi3d,'coarse':requires_coarse,'texture':requires_texture}
    for key,needed in requirements.items():
        ready.update(hi3d=True,coarse=True,texture=True);ready[key]=False
        captures.clear();checks.clear()
        response=client.post('/api/jobs',json=payload)
        assert response.status_code==(409 if needed else 200),response.text
        if needed:assert captures==[]
        else:
            assert len(captures)==1 and bool(checks)==requires_hi3d
            assert (captures[0][1] is not None)==needs_mesh


@pytest.mark.parametrize('mesh_value',[None,''])
def test_image_only_accepts_no_mesh_and_records_no_mesh_asset(backend,mesh_value):
    client,ready,captures,checks=backend
    image,_=upload_pair(client)
    response=client.post('/api/jobs',json={'image':image['id'],'mesh':mesh_value,'mode':'image_to_asset','refine_geometry':False})
    assert response.status_code==200,response.text
    assert captures[0][1] is None and captures[0][2]['mesh_asset'] is None


def test_job_manager_persists_image_only_inputs_without_touching_a_mesh(tmp_path,monkeypatch):
    monkeypatch.setattr(jobs,'DATA',tmp_path/'data')
    manager=jobs.JobManager()
    calls=[]
    class DeferredThread:
        def __init__(self,**kwargs):calls.append(kwargs)
        def start(self):pass  # Exercise persistence only, never a model process.
    monkeypatch.setattr(jobs.threading,'Thread',DeferredThread)
    image=tmp_path/'input.png';Image.new('RGBA',(8,8),(50,90,120,255)).save(image)
    state=manager.create(image,None,{'mode':'image_to_asset','resolution':512})
    folder=manager.root/state['id']
    assert (folder/'input.png').read_bytes()==image.read_bytes()
    assert not (folder/'coarse.glb').exists()
    assert common.read_json(folder/'request.json')['mode']=='image_to_asset'
    assert calls and calls[0]['target']==manager._run
    assert manager.get(state['id'])['state']=='queued'


@pytest.mark.parametrize('mode',['refine','texture'])
def test_existing_model_modes_reject_missing_mesh_before_model_process(backend,mode):
    client,ready,captures,checks=backend
    image,_=upload_pair(client)
    response=client.post('/api/jobs',json={'image':image['id'],'mode':mode})
    assert response.status_code==400 and captures==[]


@pytest.mark.parametrize('mode,flags',[('image_to_asset',{}),('texture',{}),('refine',{'generate_texture':True})])
def test_opaque_auto_background_requires_local_segmentation_model_but_explicit_keep_does_not(backend,mode,flags):
    client,ready,captures,checks=backend
    _,mesh=upload_pair(client)
    data=io.BytesIO();Image.new('RGB',(12,12),(50,100,150)).save(data,format='PNG')
    image=client.post('/api/assets',files={'file':('opaque.png',data.getvalue(),'image/png')}).json()
    ready['background']=False
    payload={'image':image['id'],'mesh':mesh['id'],'mode':mode,**flags}
    response=client.post('/api/jobs',json=payload)
    assert response.status_code==409 and captures==[]
    response=client.post('/api/jobs',json={**payload,'background':'keep'})
    assert response.status_code==200,response.text


@pytest.mark.parametrize('field,value',[
    ('mode','../../engine'),('background','shell'),('texture_resolution',1536),('texture_size',8192),
    ('resolution',256),('attention','custom-kernel'),('image','../../escape'),('mesh','../../escape'),
])
def test_untrusted_options_cannot_select_paths_commands_or_unsupported_modes(backend,field,value):
    client,ready,captures,checks=backend
    image,mesh=upload_pair(client)
    payload={'image':image['id'],'mesh':mesh['id'],'mode':'refine',field:value}
    response=client.post('/api/jobs',json=payload)
    assert response.status_code==400 and captures==[]


def test_http_extra_fields_never_become_worker_commands_or_output_paths(backend):
    client,ready,captures,checks=backend
    image,mesh=upload_pair(client)
    injected={'worker_command':['attacker'],'model_root':'C:/outside','output_path':'C:/outside/result.glb','image_path':'C:/private.png'}
    response=client.post('/api/jobs',json={'image':image['id'],'mesh':mesh['id'],**injected})
    assert response.status_code==200,response.text
    assert not (set(captures[0][2])&set(injected))


def _saved_job(mode='image_to_asset',mesh=False):
    folder=server.manager.root/('asset-fixture-'+uuid.uuid4().hex);folder.mkdir()
    Image.new('RGBA',(12,12),(50,90,120,128)).save(folder/'input.png')
    if mesh:trimesh.creation.box().export(folder/'coarse.glb')
    common.atomic_json(folder/'state.json',{'id':folder.name,'created':1,'state':'failed','options':{'mode':mode,'seed':7,'resolution':512}})
    return folder


def test_failed_image_only_job_reuse_preserves_image_without_requiring_partial_coarse(client):
    folder=_saved_job()
    before={path.name:path.read_bytes() for path in folder.iterdir()}
    response=client.post(f'/api/jobs/{folder.name}/reuse')
    assert response.status_code==200,response.text
    assert response.json()['mesh'] is None
    assert client.get('/api/assets/'+response.json()['image']['id']+'/image.png').content==before['input.png']
    assert {path.name:path.read_bytes() for path in folder.iterdir()}==before


def test_failed_texture_job_reuse_requires_and_preserves_original_geometry(client):
    folder=_saved_job('texture',mesh=True)
    before=(folder/'coarse.glb').read_bytes()
    response=client.post(f'/api/jobs/{folder.name}/reuse')
    assert response.status_code==200,response.text
    assert client.get(response.json()['mesh']['url']).content==before
    assert (folder/'coarse.glb').read_bytes()==before
    (folder/'coarse.glb').unlink()
    assert client.post(f'/api/jobs/{folder.name}/reuse').status_code==404


def _textured_fixture(path):
    mesh=trimesh.creation.box(extents=[1,2,4])
    uv=(np.asarray(mesh.vertices)[:,:2]+[.5,1])/[1,2]
    color=Image.fromarray(np.array([[[210,60,90,255],[25,170,90,255]],[[40,80,200,255],[230,200,50,255]]],dtype=np.uint8),'RGBA')
    rough=Image.fromarray(np.array([[[0,60,220],[0,120,40]],[[0,200,90],[0,240,180]]],dtype=np.uint8),'RGB')
    material=trimesh.visual.material.PBRMaterial(baseColorTexture=color,metallicRoughnessTexture=rough,metallicFactor=1,roughnessFactor=1)
    mesh.visual=trimesh.visual.TextureVisuals(uv=uv,material=material)
    scene=trimesh.Scene();scene.add_geometry(mesh,node_name='asset')
    scene.export(path)
    return uv,color,rough


def test_real_uv_and_embedded_pbr_images_survive_scaled_export(tmp_path):
    source=tmp_path/'source.glb';target=tmp_path/'height-80mm.glb'
    uv,color,rough=_textured_fixture(source)
    original=source.read_bytes()
    inspected=inspect_textured_glb(source)
    assert inspected['texture_generated'] and inspected['uv'] and inspected['embedded']
    info=export_textured_glb(source,target,80)
    assert info['units']=='m' and info['height_mm']==80
    scene=trimesh.load(target,force='scene',process=False,resolver={},allow_remote=False)
    assert scene.extents[2]==pytest.approx(.08)
    actual=next(iter(scene.geometry.values()))
    assert np.allclose(actual.visual.uv,uv)
    assert np.array_equal(np.asarray(actual.visual.material.baseColorTexture),np.asarray(color))
    assert np.array_equal(np.asarray(actual.visual.material.metallicRoughnessTexture),np.asarray(rough))
    assert source.read_bytes()==original


def test_unscaled_textured_export_is_exact_copy(tmp_path):
    source=tmp_path/'original.glb';target=tmp_path/'export.glb';_textured_fixture(source)
    export_textured_glb(source,target)
    assert target.read_bytes()==source.read_bytes()


def test_uv_seam_topology_counts_closed_surface_without_hiding_a_real_opening(tmp_path,client):
    from hi3d_workbench.asset_pipeline import inspect_asset_geometry
    from hi3d_workbench.geometry import inspect_mesh,load_mesh
    box=trimesh.creation.box()
    vertices=[];faces=[];uv=[]
    for normal in np.unique(np.asarray(box.face_normals),axis=0):
        pair=box.faces[np.all(box.face_normals==normal,axis=1)]
        corners=np.unique(pair);offset=len(vertices)
        assert len(corners)==4 and len(pair)==2
        vertices.extend(box.vertices[corners])
        faces.extend(np.searchsorted(corners,pair)+offset)
        uv.extend([[0,0],[1,0],[1,1],[0,1]])
    # Six independent UV islands: 4 vertices per side, but only 8 distinct
    # positions. Index topology alone incorrectly sees six open components.
    seam_box=trimesh.Trimesh(vertices=np.asarray(vertices),faces=np.asarray(faces),process=False)
    material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new('RGBA',(2,2),(180,100,60,255)),
        metallicRoughnessTexture=Image.new('RGB',(2,2),(0,140,60)))
    seam_box.visual=trimesh.visual.TextureVisuals(uv=np.asarray(uv,dtype=np.float32),material=material)
    closed=tmp_path/'uv-seam-box.glb';seam_box.export(closed)
    original=closed.read_bytes();original_sha=common.digest(closed)
    raw=inspect_mesh(load_mesh(closed))
    assert raw['vertices']==24 and raw['boundary_edges']>0 and raw['components']>1
    topology=inspect_asset_geometry(closed)
    assert topology['watertight'] and topology['boundary_edges']==0 and topology['components']==1
    assert topology['vertices']==24 and topology['topology_vertices']==8
    basis=topology['topology_basis'].replace('-',' ').lower()
    assert 'exact' in basis and 'position' in basis
    assert topology['warnings'] and closed.read_bytes()==original and common.digest(closed)==original_sha
    uploaded=client.post('/api/assets',files={'file':('uv-seam-box.glb',original,'model/gltf-binary')})
    assert uploaded.status_code==200,uploaded.text
    metadata=uploaded.json();http_geometry=metadata['geometry']
    assert http_geometry['vertices']==24 and http_geometry['topology_vertices']==8
    assert http_geometry['watertight'] and http_geometry['boundary_edges']==0 and http_geometry['components']==1
    assert common.digest(closed)==original_sha
    # A history reuse copies the canonical mesh without reverting to the raw
    # split-index diagnosis or changing the task's original geometry bytes.
    folder=_saved_job('texture',mesh=True)
    canonical=client.get(metadata['url']).content;(folder/'coarse.glb').write_bytes(canonical)
    history_sha=common.digest(folder/'coarse.glb')
    reused=client.post(f'/api/jobs/{folder.name}/reuse')
    assert reused.status_code==200,reused.text
    reused_geometry=reused.json()['mesh']['geometry']
    assert reused_geometry['vertices']==24 and reused_geometry['topology_vertices']==8
    assert reused_geometry['watertight'] and reused_geometry['boundary_edges']==0 and reused_geometry['components']==1
    assert common.digest(folder/'coarse.glb')==history_sha and common.digest(closed)==original_sha
    # Remove the two triangles on the +Z side. Exact-position welding must
    # preserve this real opening, rather than interpreting every asset as closed.
    seam_box.update_faces(np.asarray(seam_box.face_normals)[:,2]<.5)
    assert len(seam_box.faces)==10
    opened=tmp_path/'uv-seam-open-box.glb';seam_box.export(opened)
    open_original=opened.read_bytes()
    open_topology=inspect_asset_geometry(opened)
    assert not open_topology['watertight'] and open_topology['boundary_edges']>0
    assert open_topology['components']==1
    assert opened.read_bytes()==open_original


def test_additional_material_maps_cannot_reference_external_images(tmp_path):
    source=tmp_path/'source.glb';_textured_fixture(source)
    raw=source.read_bytes();json_length=struct.unpack('<I',raw[12:16])[0]
    tree=json.loads(raw[20:20+json_length].decode().rstrip('\x00 \r\n\t'))
    image=len(tree['images']);tree['images'].append({'uri':'https://example.invalid/normal-map.png'})
    texture=len(tree['textures']);tree['textures'].append({'source':image})
    tree['materials'][0]['normalTexture']={'index':texture}
    encoded=json.dumps(tree).encode();encoded+=b' '*((-len(encoded))%4)
    tail=raw[20+json_length:]
    source.write_bytes(struct.pack('<4sII',b'glTF',2,20+len(encoded)+len(tail))+struct.pack('<I4s',len(encoded),b'JSON')+encoded+tail)
    with pytest.raises(ValueError,match='嵌入|外部'):inspect_textured_glb(source)


@pytest.mark.parametrize('format',['stl','ply'])
def test_api_does_not_silently_drop_materials_for_textured_export(client,format):
    folder=_saved_job('texture',mesh=True);_textured_fixture(folder/'textured.glb')
    response=client.post(f'/api/jobs/{folder.name}/export',json={'source':'textured','format':format})
    assert response.status_code==400 and 'GLB' in response.json()['detail']


def test_textured_api_export_checks_real_materials_and_preserves_dimensions(client):
    folder=_saved_job('texture',mesh=True);_textured_fixture(folder/'textured.glb')
    original=(folder/'textured.glb').read_bytes()
    result=client.post(f'/api/jobs/{folder.name}/export',json={'source':'textured','format':'glb','height_mm':80})
    assert result.status_code==200,result.text
    info=result.json();scene=trimesh.load(io.BytesIO(client.get(info['url']).content),file_type='glb',force='scene',process=False)
    assert scene.extents[2]==pytest.approx(.08) and info['texture_generated'] and info['uv']
    assert (folder/'textured.glb').read_bytes()==original
    assert client.get(info['dimensions']).json()['height_mm']==80


def test_geometry_only_glb_is_never_accepted_as_finished_texturing(client,tmp_path):
    path=tmp_path/'geometry-only.glb';trimesh.creation.box().export(path)
    with pytest.raises(ValueError,match='材质|贴图'):inspect_textured_glb(path)
    folder=_saved_job('texture',mesh=True);(folder/'textured.glb').write_bytes(path.read_bytes())
    response=client.post(f'/api/jobs/{folder.name}/export',json={'source':'textured','format':'glb'})
    assert response.status_code==400


def _fake_stages(monkeypatch,material_output=True):
    calls=[]
    def run_child(command,request,directory,stage):
        value=common.read_json(request);calls.append((stage,value))
        if stage=='refine':
            trimesh.creation.box(extents=[1,2,4]).export(directory/'refined.glb')
            common.atomic_json(directory/'report.json',{'status':'succeeded','peak_allocated_mb':10})
        else:
            output=Path(value['output_path'])
            if stage=='texture' and material_output:_textured_fixture(output)
            else:trimesh.creation.box(extents=[1,2,4]).export(output)
            common.atomic_json(output.with_suffix('.stage.json'),{'fixture':'orchestration only','peak_vram_mb':5})
    monkeypatch.setattr(pipeline_worker,'run_child',run_child)
    monkeypatch.setattr(pipeline_worker,'config',lambda:{'worker_python':sys.executable,'worker_command':[]})
    return calls


def _pipeline_request(folder,**values):
    Image.new('RGBA',(12,12),(30,60,120,255)).save(folder/'input.png')
    request=folder/'request.json'
    common.atomic_json(request,{'mode':'image_to_asset','seed':1,'resolution':512,'steps':25,'offload':True,'generate_texture':True,**values})
    return request


def test_supervisor_sequences_coarse_refine_then_textures_final_mesh(monkeypatch,tmp_path):
    calls=_fake_stages(monkeypatch)
    request=_pipeline_request(tmp_path,refine_geometry=True)
    pipeline_worker.run(request)
    assert [stage for stage,_ in calls]==['coarse','refine','texture']
    assert calls[-1][1]['mesh_path']==str(tmp_path/'refined.glb')
    report=common.read_json(tmp_path/'report.json')
    assert report['final_file']=='textured.glb' and report['texture_generated']
    assert inspect_textured_glb(tmp_path/'textured.glb')['uv']


def test_supervisor_does_not_claim_success_when_engine_returns_untextured_mesh(monkeypatch,tmp_path):
    _fake_stages(monkeypatch,material_output=False)
    request=_pipeline_request(tmp_path,refine_geometry=True)
    with pytest.raises(ValueError,match='材质|贴图'):pipeline_worker.run(request)
    assert not (tmp_path/'report.json').exists()
    assert (tmp_path/'coarse.glb').is_file() and (tmp_path/'refined.glb').is_file()


def test_supervisor_can_skip_hi3d_and_texture_for_geometry_only_image_mode(monkeypatch,tmp_path):
    calls=_fake_stages(monkeypatch)
    request=_pipeline_request(tmp_path,refine_geometry=False,generate_texture=False)
    pipeline_worker.run(request)
    report=common.read_json(tmp_path/'report.json')
    assert [stage for stage,_ in calls]==['coarse']
    assert report['final_file']=='coarse.glb' and not report['texture_generated']


def test_image_pipeline_omitting_refine_flag_textures_coarse_geometry_directly(monkeypatch,tmp_path):
    calls=_fake_stages(monkeypatch)
    request=_pipeline_request(tmp_path)
    assert 'refine_geometry' not in common.read_json(request)
    pipeline_worker.run(request)
    assert [stage for stage,_ in calls]==['coarse','texture']
    assert calls[-1][1]['mesh_path']==str(tmp_path/'coarse.glb')
    assert not (tmp_path/'refined.glb').exists()
    report=common.read_json(tmp_path/'report.json')
    assert report['final_file']=='textured.glb' and report['texture_generated']


def test_original_refine_mode_still_runs_hi3d_when_optional_refine_flag_is_omitted(monkeypatch,tmp_path):
    calls=_fake_stages(monkeypatch)
    trimesh.creation.box().export(tmp_path/'coarse.glb')
    request=_pipeline_request(tmp_path,mode='refine',generate_texture=False)
    assert 'refine_geometry' not in common.read_json(request)
    pipeline_worker.run(request)
    assert [stage for stage,_ in calls]==['refine']
    report=common.read_json(tmp_path/'report.json')
    assert report['final_file']=='refined.glb' and not report['texture_generated']


@pytest.fixture
def tiny_extensions(tmp_path,monkeypatch):
    root=tmp_path/'extensions';root.mkdir()
    specs={key:{'repo':'fixture/'+key,'revision':'fixture-revision','license':'fixture','files':['weights.bin']} for key in pipeline_models.SPECS}
    monkeypatch.setattr(pipeline_models,'SPECS',specs)
    monkeypatch.setattr(pipeline_models,'MODEL_ROOT',root)
    executable=tmp_path/'runtime'/'python.exe';executable.parent.mkdir();executable.write_bytes(b'not executable')
    common.atomic_json(executable.parent/'pipeline-runtime.json',{'ready':True})
    monkeypatch.setattr(pipeline_models,'runtime_python',lambda:executable)
    manifest={'components':{}}
    for key,spec in specs.items():
        path=root/key/'weights.bin';path.parent.mkdir();path.write_bytes(b'fixture weight bytes')
        manifest['components'][key]={'repo':spec['repo'],'revision':spec['revision'],'files':{'weights.bin':{'hash':pipeline_models._hash(path),'algorithm':'sha256','verified':True,'signature':pipeline_models._signature(path)}}}
    common.atomic_json(root/'manifest.json',manifest)
    return root


@pytest.mark.parametrize('payload',[{'components':[]},{'components':{'coarse':[]}},{'components':{'coarse':{'files':[]}}}])
def test_bad_extension_manifest_schema_is_unready_instead_of_crashing(tiny_extensions,payload):
    common.atomic_json(tiny_extensions/'manifest.json',payload)
    state=pipeline_models.status()
    assert not state['coarse_ready'] and not state['texture_ready']


def test_same_size_corruption_with_restored_mtime_invalidates_extension_ready(tiny_extensions):
    assert pipeline_models.status()['coarse_ready']
    path=tiny_extensions/'coarse'/'weights.bin';before=path.stat()
    path.write_bytes(b'x'*before.st_size);os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
    assert not pipeline_models.status()['coarse_ready']


@pytest.mark.parametrize('identity',['matching','wrong-time','invalid-json-object','missing'])
def test_stage_recovery_reaps_only_verified_identity_and_blocks_unknown_survivor(tmp_path,monkeypatch,identity):
    monkeypatch.setattr(jobs,'DATA',tmp_path/'data')
    manager=jobs.JobManager();folder=manager.root/'stage-recovery';stage=folder/'stages'/'texture';stage.mkdir(parents=True)
    request=stage/'request.json';request.write_text('{}')
    common.atomic_json(folder/'state.json',{'id':folder.name,'created':1,'state':'running','options':{'mode':'texture'}})
    command=[sys.executable,'-c','import time;time.sleep(30)','-m','hi3d_workbench.engines.run_stage','--request',str(request)]
    process=subprocess.Popen(command)
    created=psutil.Process(process.pid).create_time()
    try:
        if identity=='invalid-json-object':common.atomic_json(stage/'process.json',[])
        elif identity!='missing':common.atomic_json(stage/'process.json',{'pid':process.pid,'created':created+(100 if identity=='wrong-time' else 0),'request':str(request)})
        manager.recover()
        assert manager.get(folder.name)['state']=='interrupted'
        if identity=='matching':
            assert process.wait(timeout=5) is not None and not manager.recovery_blocked
        else:
            assert process.poll() is None and manager.recovery_blocked
            with pytest.raises(RuntimeError,match='残留工作进程'):manager.create(folder/'input.png',None,{})
    finally:
        jobs._kill_worker_tree(process.pid,created,request)
        process.wait(timeout=5)
    assert not manager.recovery_blocked


@pytest.mark.parametrize('identity',['matching','wrong-time','missing','unrelated','access-denied'])
def test_supervisor_exit_cleans_or_guards_surviving_stage_before_releasing_job(tmp_path,monkeypatch,identity):
    """A dead supervisor must not release CUDA while a separate stage survives."""
    monkeypatch.setattr(jobs,'DATA',tmp_path/'data')
    manager=jobs.JobManager();folder=manager.root/'dead-supervisor';stage=folder/'stages'/'texture';stage.mkdir(parents=True)
    request=stage/'request.json';request.write_text('{}')
    parent_request=folder/'request.json';parent_request.write_text('{}')
    common.atomic_json(folder/'state.json',{'id':folder.name,'created':1,'state':'queued','options':{'mode':'texture','generate_texture':True}})
    supervisor=tmp_path/'supervisor-exits.py'
    supervisor.write_text('import time\nprint("supervisor fixture exits",flush=True)\ntime.sleep(.05)\nraise SystemExit(3)\n')
    monkeypatch.setattr(jobs,'config',lambda:{'worker_command':[sys.executable,str(supervisor)]})
    stage_process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)','-m','hi3d_workbench.engines.run_stage','--request',str(request)])
    created=psutil.Process(stage_process.pid).create_time()
    actual_kill=jobs._kill_worker_tree
    outsider=None;outside_created=None;outside_request=None
    try:
        record={'pid':stage_process.pid,'created':created,'request':str(request)}
        if identity=='wrong-time':record['created']+=100
        elif identity=='unrelated':
            outside_request=tmp_path/'different-request.json';outside_request.write_text('{}')
            outsider=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)','-m','hi3d_workbench.engines.run_stage','--request',str(outside_request)])
            outside_created=psutil.Process(outsider.pid).create_time()
            record.update(pid=outsider.pid,created=outside_created,request=str(outside_request))
        if identity!='missing':common.atomic_json(stage/'process.json',record)
        if identity=='access-denied':
            def deny_only_this_stage(pid,stamp,path):
                if pid==stage_process.pid:raise psutil.AccessDenied(pid)
                return actual_kill(pid,stamp,path)
            monkeypatch.setattr(jobs,'_kill_worker_tree',deny_only_this_stage)
        manager.active=folder.name
        manager._run(folder.name)  # Runs only the temporary supervisor script.
        assert manager.active is None and manager.process is None
        assert manager.get(folder.name)['state']=='failed'
        assert '3' in manager.get(folder.name)['message']
        if identity=='matching':
            assert stage_process.wait(timeout=5) is not None
            assert not manager.recovery_blocked
        else:
            assert stage_process.poll() is None and manager.recovery_blocked
            assert any(worker['pid']==stage_process.pid for worker in manager.recovery_workers)
            with pytest.raises(RuntimeError,match='残留工作进程'):manager.create(folder/'input.png',None,{})
        if outsider is not None:assert outsider.poll() is None  # A forged stage identity never kills another request.
    finally:
        actual_kill(stage_process.pid,created,request);stage_process.wait(timeout=5)
        if outsider is not None:
            actual_kill(outsider.pid,outside_created,outside_request);outsider.wait(timeout=5)
    assert not manager.recovery_blocked
