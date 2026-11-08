"""Idol's distinct output contract and preservation boundaries (no GPU mocks as evidence)."""
import json
from pathlib import Path

import pytest

from test_asset_pipeline import backend
from test_workbench import client,upload_pair
from hi3d_workbench import server,pipeline_worker
from hi3d_workbench.common import atomic_json
from hi3d_workbench.engines.idol import settings,camera_metadata


def test_selective_idol_download_requires_its_own_background_model():
    from hi3d_workbench.pipeline_models import selection_components
    assert set(selection_components('engine:coarse:idol'))=={'coarse_idol','idol_background'}


@pytest.mark.parametrize('key,value',[
    ('idol_gaussians',32769),('idol_gaussians',True),('idol_gaussians',1048608),
    ('idol_cfg',float('nan')),('idol_cfg',float('inf')),('idol_decode_seed',-1),
    ('idol_extract_mesh','false'),('idol_min_opacity',1.1),('idol_crop','unknown')])
def test_invalid_splat_settings_fail_before_inference(key,value):
    with pytest.raises(ValueError):settings({key:value})


def test_predicted_camera_retains_native_convention_and_gltf_direction():
    import math
    camera=camera_metadata([2,0,0,.5,.5])
    assert camera['yaw']==pytest.approx(math.pi/2)
    assert camera['gltf_direction']==[1.,0.,-0.]
    assert camera['radius']==2
    assert camera['fov']==pytest.approx(math.degrees(2*math.atan(.25)))
    with pytest.raises(ValueError):camera_metadata([float('nan'),0,0,1,1])


def test_private_runtime_receipt_cannot_hide_changed_files(tmp_path):
    from hi3d_workbench.pipeline_models import idol_runtime_ready
    from hi3d_workbench.common import digest
    overlay=tmp_path/'engines/idol';source=tmp_path/'sources/idol_mesh'
    overlay.mkdir(parents=True);source.mkdir(parents=True)
    dependency=overlay/'av.py';dependency.write_bytes(b'first')
    module=source/'nodes_gaussian_splat.py';module.write_bytes(b'upstream')
    atomic_json(tmp_path/'idol-runtime.json',{'ready':True,'source_revision':'7cebca52227c63b4180ceed448cbd7ad22903b6d',
        'files':{'av.py':digest(dependency)}})
    atomic_json(source/'source-manifest.json',{'revision':'0decb39ee5b5e17c200334589fc317dbbe98310e',
        'files':{'nodes_gaussian_splat.py':digest(module)}})
    assert idol_runtime_ready(tmp_path)
    dependency.write_bytes(b'other')
    assert not idol_runtime_ready(tmp_path)


@pytest.mark.parametrize('corrupt',[False,True])
def test_portable_audit_verifies_isolated_idol_bytes(tmp_path,corrupt):
    import hashlib
    import zipfile
    from tools.audit_release import verify_idol_archive
    content=b'private dependency';module=b'fixed upstream geometry'
    receipt={'ready':True,'source_revision':'7cebca52227c63b4180ceed448cbd7ad22903b6d',
        'files':{'av/__init__.py':hashlib.sha256(content).hexdigest()}}
    source={'revision':'0decb39ee5b5e17c200334589fc317dbbe98310e',
        'files':{'nodes_gaussian_splat.py':hashlib.sha256(module).hexdigest()}}
    archive=tmp_path/'runtime.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('runtime-pipeline/idol-runtime.json',json.dumps(receipt))
        z.writestr('runtime-pipeline/sources/idol_mesh/source-manifest.json',json.dumps(source))
        z.writestr('runtime-pipeline/sources/idol_mesh/nodes_gaussian_splat.py',module)
        z.writestr('runtime-pipeline/engines/idol/av/__init__.py',b'changed' if corrupt else content)
    with zipfile.ZipFile(archive) as z:
        if corrupt:
            with pytest.raises(AssertionError,match='hash mismatch'):verify_idol_archive(z,'')
        else:assert verify_idol_archive(z,'')=='idol'


def test_splat_only_job_does_not_require_mesh_painter_or_u2net(backend,monkeypatch):
    client,ready,captures,checks=backend
    monkeypatch.setattr(server.pipeline_models,'status',lambda:{'background_ready':False,
        'engines':{'coarse':{'idol':{'ready':True}},'texture':{'trellis2':{'ready':False}}}})
    image,_=upload_pair(client)
    response=client.post('/api/jobs',json={'image':image['id'],'mode':'image_to_asset',
        'geometry_engine':'idol','idol_extract_mesh':False,'generate_texture':False,'idol_gaussians':32768})
    assert response.status_code==200,response.text
    assert captures[0][2]['idol_gaussians']==32768
    assert captures[0][2]['geometry_steps']==50


def test_idol_with_pbr_checks_the_texture_foreground_dependency(backend,monkeypatch):
    import io
    from PIL import Image
    client,ready,captures,checks=backend
    monkeypatch.setattr(server.pipeline_models,'status',lambda:{'background_ready':False,
        'engines':{'coarse':{'idol':{'ready':True}},'texture':{'trellis2':{'ready':True}}}})
    pixels=io.BytesIO();Image.new('RGB',(64,64),'white').save(pixels,format='PNG')
    upload=client.post('/api/assets',files={'file':('opaque.png',pixels.getvalue(),'image/png')}).json()
    body={'image':upload['id'],'mode':'image_to_asset','geometry_engine':'idol','generate_texture':True}
    response=client.post('/api/jobs',json=body)
    assert response.status_code==409 and '自动前景' in response.text
    assert captures==[]
    assert client.post('/api/jobs',json={**body,'background':'keep'}).status_code==200


@pytest.mark.parametrize('extra',[{'generate_texture':True},{'generate_texture':False,'refine_geometry':True}])
def test_splat_cannot_be_passed_directly_to_mesh_processors(backend,extra):
    client,ready,captures,checks=backend
    image,_=upload_pair(client)
    response=client.post('/api/jobs',json={'image':image['id'],'mode':'image_to_asset',
        'geometry_engine':'idol','idol_extract_mesh':False,**extra})
    assert response.status_code==400
    assert captures==[]


def test_original_splat_is_published_when_mesh_stage_fails(tmp_path,monkeypatch):
    folder=tmp_path/'job';folder.mkdir();(folder/'input.png').write_bytes(b'input-preserved')
    request=folder/'request.json'
    atomic_json(request,{'mode':'image_to_asset','geometry_engine':'idol','seed':0,'offload':True,
        'generate_texture':False,'resolution':512})
    def failed_child(command,request_path,directory,stage):
        (directory/'splat.ply').write_bytes(b'complete original splat')
        (directory/'splat.json').write_text('{"gaussians":32768}',encoding='utf-8')
        (directory/'splat-preview.partial.mp4').write_bytes(b'incomplete')
        raise RuntimeError('mesh extraction produced no surface')
    monkeypatch.setattr(pipeline_worker,'run_child',failed_child)
    with pytest.raises(RuntimeError,match='no surface'):pipeline_worker._run(request)
    assert (folder/'input.png').read_bytes()==b'input-preserved'
    assert (folder/'splat.ply').read_bytes()==b'complete original splat'
    assert not (folder/'splat-preview.mp4').exists()
    assert not (folder/'report.json').exists()


def test_completed_splat_is_visible_before_mesh_child_exits(tmp_path):
    import sys
    stage=tmp_path/'job/stages/coarse';stage.mkdir(parents=True)
    request=stage/'request.json';atomic_json(request,{'engine':'idol'})
    script='''import json,sys,time
from pathlib import Path
p=Path(sys.argv[-1]).parent
(p/'splat.ply').write_bytes(b'complete native artifact')
print('HI3D_EVENT '+json.dumps({'stage':'coarse_splat_saved'}),flush=True)
end=time.monotonic()+3
while not (p.parent.parent/'splat.ply').is_file() and time.monotonic()<end:time.sleep(.01)
assert (p.parent.parent/'splat.ply').read_bytes()==b'complete native artifact'
(p/'observed-before-exit').write_text('yes')
'''
    pipeline_worker.run_child([sys.executable,'-u','-c',script],request,stage,'coarse')
    assert (stage/'observed-before-exit').is_file()


@pytest.mark.parametrize('corrupt',[None,'splat.ply','splat.spz','splat.ksplat','count','final_hash'])
def test_splat_success_requires_native_files_and_matching_receipts(tmp_path,corrupt):
    from hi3d_workbench.jobs import _validate_splat_result
    from hi3d_workbench.common import digest
    count=32768
    names='x y z scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3 opacity f_dc_0 f_dc_1 f_dc_2'.split()
    header='ply\nformat binary_little_endian 1.0\nelement vertex 32768\n'+''.join(f'property float {n}\n' for n in names)+'end_header\n'
    (tmp_path/'splat.ply').write_bytes(header.encode()+b'\0'*(count*len(names)*4))
    (tmp_path/'splat.spz').write_bytes(b'spz-contract-fixture')
    (tmp_path/'splat.ksplat').write_bytes(b'ksplat-contract-fixture')
    files={p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in tmp_path.iterdir()}
    atomic_json(tmp_path/'splat.json',{'gaussians':count,'files':files})
    report={'representation':'3d_gaussian_splat','final_file':'splat.ply','final_sha256':digest(tmp_path/'splat.ply')}
    if corrupt=='final_hash':report['final_sha256']='0'*64
    elif corrupt in files:(tmp_path/corrupt).write_bytes(b'broken')
    options={'idol_gaussians':count+32 if corrupt=='count' else count}
    if corrupt:
        with pytest.raises(ValueError):_validate_splat_result(tmp_path,options,report)
    else:_validate_splat_result(tmp_path,options,report)


@pytest.mark.parametrize('corrupt',[False,True])
def test_job_manager_accepts_verified_native_output_without_glb(tmp_path,monkeypatch,corrupt):
    # This real child tests task orchestration/serialization, never model quality.
    import sys,time
    from hi3d_workbench import jobs
    script=tmp_path/'native_contract_worker.py'
    script.write_text('''import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[sys.argv.index('--request')+1]).parent
names='x y z scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3 opacity f_dc_0 f_dc_1 f_dc_2'.split()
header='ply\\nformat binary_little_endian 1.0\\nelement vertex 32768\\n'+''.join('property float '+n+'\\n' for n in names)+'end_header\\n'
(p/'splat.ply').write_bytes(header.encode()+b'\\0'*(32768*len(names)*4))
(p/'splat.spz').write_bytes(b'contract-spz')
(p/'splat.ksplat').write_bytes(b'contract-ksplat')
files={n:{'bytes':(p/n).stat().st_size,'sha256':hashlib.sha256((p/n).read_bytes()).hexdigest()} for n in ('splat.ply','splat.spz','splat.ksplat')}
(p/'splat.json').write_text(json.dumps({'gaussians':32768,'files':files}))
(p/'report.json').write_text(json.dumps({'status':'succeeded','representation':'3d_gaussian_splat','final_file':'splat.ply','final_sha256':files['splat.ply']['sha256']}))
'''+("(p/'splat.ksplat').write_bytes(b'broken')\n" if corrupt else ''),encoding='utf-8')
    monkeypatch.setattr(jobs,'ROOT',tmp_path)
    monkeypatch.setattr(jobs,'DATA',tmp_path/'data')
    monkeypatch.setattr(jobs,'config',lambda:{'worker_command':[sys.executable,str(script)]})
    source=tmp_path/'input.png';source.write_bytes(b'input-preserved')
    manager=jobs.JobManager()
    job=manager.create(source,None,{'mode':'image_to_asset','geometry_engine':'idol',
        'idol_extract_mesh':False,'idol_gaussians':32768,'generate_texture':False,'model_policy':'release'})
    end=time.monotonic()+10
    while manager.active and time.monotonic()<end:time.sleep(.02)
    assert manager.active is None
    result=manager.get(job['id'])
    assert result['state']==('failed' if corrupt else 'succeeded')
    assert not (manager.root/job['id']/'coarse.glb').exists()
