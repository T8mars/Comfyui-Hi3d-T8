import json

import numpy as np
import pytest
import trimesh

from test_workbench import client,testroot
from test_studio_parts import source_model
from hi3d_workbench.studio import parts_models, material_transfer, retopology
from hi3d_workbench.studio.scene import instances


def coverage(count):
    return {'source_vertex_samples':1000,'vertices_over_3_percent_extent':count,
            'normalized_distance_p95':.01,'normalized_distance_max':.06}


def test_completion_guard_accepts_the_boundary_but_rejects_major_original_surface_loss():
    parts_models.validate_completion_coverage(coverage(50))
    with pytest.raises(RuntimeError,match='原部件外形'):
        parts_models.validate_completion_coverage(coverage(51))
    for value in ({},{'source_vertex_samples':0},{'source_vertex_samples':1000,'vertices_over_3_percent_extent':1001}):
        with pytest.raises(RuntimeError,match='记录无效'):
            parts_models.validate_completion_coverage(value)


def test_all_parts_are_checked_before_baking_and_original_partition_survives_failure(tmp_path,monkeypatch):
    pytest.importorskip('torch')
    source=tmp_path/'source.glb';document=source_model(source)
    labels=[np.arange(len(row.faces),dtype=np.int32)%2 for row in instances(document)]
    folder=tmp_path/'job';folder.mkdir();target=folder/'output.glb'
    class Pipeline:
        def __call__(self,**kwargs):
            scene=trimesh.Scene()
            for index in range(2):scene.add_geometry(trimesh.creation.icosphere(subdivisions=1),geom_name=str(index))
            return scene,None
    monkeypatch.setattr(parts_models,'_xpart',lambda emit,**kwargs:Pipeline())
    from hi3d_workbench.studio import xpart_sampling
    monkeypatch.setattr(xpart_sampling,'generate',lambda pipe,**kwargs:(pipe()[0],{'fixture':True}))
    calls=[]
    def check(*args,**kwargs):
        calls.append(kwargs['stage']);return coverage(0 if len(calls)==1 else 400)
    monkeypatch.setattr(retopology,'geometry_coverage',check)
    def unexpected_transfer(*args,**kwargs):raise AssertionError('Material baking ran before all parts passed')
    monkeypatch.setattr(material_transfer,'run',unexpected_transfer)
    with pytest.raises(RuntimeError,match='原部件外形'):
        parts_models.complete(source,target,document,labels,{0:'first',1:'second'},
            {'part_steps':50,'part_resolution':256,'seed':23},lambda *a,**kw:None)
    assert calls==['asset_part_validation','asset_part_validation']
    assert not target.exists()
    assert (folder/'segmentation/output.glb').is_file() and (folder/'segmentation/parts.zip').is_file()
    for original,published in [('output.glb','original-segmentation.glb'),
                                ('parts.json','original-parts.json'),('parts.zip','original-parts.zip')]:
        assert (folder/'segmentation'/original).read_bytes()==(folder/published).read_bytes()
    from hi3d_workbench.studio.api import source_parts_path,FILES,SOURCE_FILES
    assert source_parts_path(folder/'original-segmentation.glb')==folder/'original-parts.json'
    assert 'original-segmentation.glb' in FILES & SOURCE_FILES
    # A later completed output's metadata must never be applied to the original.
    from hi3d_workbench.common import atomic_json
    atomic_json(folder/'parts.json',{'schema':1,'mesh_sha256':'different completed mesh'})
    assert source_parts_path(folder/'original-segmentation.glb')==folder/'original-parts.json'
    saved=json.loads((folder/'part-coverage.json').read_text(encoding='utf-8'))
    assert [item['status'] for item in saved]==['passed','rejected']
    assert (folder/'stages/studio-blender/completed_0000.obj').is_file()
    assert (folder/'stages/studio-blender/completed_0001.obj').is_file()


def test_failed_completion_original_partition_is_downloadable_and_editable_via_http(client,tmp_path):
    import time
    import uuid
    from hi3d_workbench import server
    from hi3d_workbench.common import atomic_json,digest
    from hi3d_workbench.studio.parts import write_parts
    jid=uuid.uuid4().hex;folder=server.manager.root/jid;folder.mkdir()
    document=source_model(tmp_path/'colored.glb')
    labels=[np.arange(len(row.faces),dtype=np.int32)%2 for row in instances(document)]
    output=folder/'original-segmentation.glb'
    write_parts(document,labels,{0:'body',1:'head'},output,engine='original partition',
                source_sha256=digest(tmp_path/'colored.glb'))
    (folder/'parts.json').replace(folder/'original-parts.json')
    (folder/'parts.zip').replace(folder/'original-parts.zip')
    atomic_json(folder/'parts.json',{'schema':1,'mesh_sha256':'completed output identity'})
    atomic_json(folder/'state.json',{'id':jid,'created':time.time(),'state':'failed','options':{'mode':'studio'}})
    job=client.get('/api/jobs/'+jid).json()
    assert {'original-segmentation.glb','original-parts.json','original-parts.zip'} <= set(job['available_files'])
    for name in ('original-segmentation.glb','original-parts.zip'):
        response=client.get('/api/studio/jobs/'+jid+'/files/'+name)
        assert response.status_code==200 and response.content==(folder/name).read_bytes()
    source=f'job:{jid}:original-segmentation.glb'
    info=client.get('/api/studio/source',params={'source':source})
    assert info.status_code==200 and info.json()['parts']['mesh_sha256']==digest(output)
    assert any(row['value']==source for row in client.get('/api/studio/sources').json())
