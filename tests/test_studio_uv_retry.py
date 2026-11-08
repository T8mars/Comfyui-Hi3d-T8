import json
import shutil

import numpy as np
import pytest
import trimesh

from test_workbench import testroot
from hi3d_workbench import common
from hi3d_workbench.studio import material_transfer
from hi3d_workbench.studio.uv_atlas import UVLayoutError,validate_cached
from hi3d_workbench.studio.gltf import Document


def setup_case(tmp_path,monkeypatch,*,cpu_error=None,gpu_error=None,available=True):
    monkeypatch.setattr(common,'DATA',tmp_path/'data')
    geometry=tmp_path/'mesh.obj'
    positions=np.array([[0,0,0],[1,0,0],[0,1,0]],dtype=np.float32)
    faces=np.array([0,1,2],dtype=np.uint32)
    trimesh.Trimesh(positions,faces.reshape(-1,3),process=False).export(geometry)
    document=Document({'asset':{'version':'2.0'},'scene':0,'scenes':[{'nodes':[0]}],
        'nodes':[{'mesh':0}],'meshes':[{'primitives':[]}]})
    document.tree['meshes'][0]['primitives']=[{'attributes':{
        'POSITION':document.add_accessor(positions),
        'TEXCOORD_0':document.add_accessor(np.array([[0,0],[1,0],[0,1]],dtype=np.float32))},
        'indices':document.add_accessor(faces)}]
    source=tmp_path/'source.glb';document.save(source);calls=[]
    def select(path,backend):
        if backend=='gpu' and not available:raise RuntimeError('No private CUDA runtime')
        return 'cpu' if backend=='auto' else backend
    def create(path,target,*,backend,**kwargs):
        calls.append(backend)
        error=cpu_error if backend=='cpu' else gpu_error
        if error:raise error
        shutil.copyfile(source,target)
        return {'backend':backend,'zero_uv_faces':0,'duplicate_uv_faces':0}
    def transfer(source,atlas,output,**kwargs):
        validate_cached(atlas);shutil.copyfile(atlas,output)
        return {'material_transfer':'fixture projection','notice':'fixture'}
    monkeypatch.setattr(material_transfer,'select_backend',select)
    monkeypatch.setattr(material_transfer,'create',create)
    monkeypatch.setattr(material_transfer,'transfer',transfer)
    return source,geometry,calls


def test_auto_retries_layout_only_with_separate_validated_gpu_cache(tmp_path,monkeypatch):
    source,geometry,calls=setup_case(tmp_path,monkeypatch,cpu_error=UVLayoutError('Too many degenerate UVs'))
    for index in range(2):
        folder=tmp_path/str(index);output=folder/'result.glb';events=[]
        report=material_transfer.run(source,geometry,output,folder,uv_backend='auto',
            emit=lambda *args,**kwargs:events.append(args))
        assert report['uv_backend']=='gpu' and report['uv']['cache_hit']==bool(index)
        assert report['uv_retry']=={'requested':'auto','initial_backend':'cpu','used_backend':'gpu','reason':'Too many degenerate UVs'}
        assert json.loads((folder/'report.json').read_text())==report
        assert any('CPU UV' in row[1] for row in events)
        validate_cached(output)
    assert calls==['cpu','gpu','cpu']
    manifests=list((common.DATA/'studio-cache/uv').glob('*/manifest.json'))
    assert len(manifests)==1
    assert json.loads(manifests[0].read_text())['identity']['backend']=='gpu'


@pytest.mark.parametrize('requested,error,available,expected',[
    ('cpu',UVLayoutError('CPU layout rejected'),True,['cpu']),
    ('auto',RuntimeError('Original triangle changed'),True,['cpu']),
    ('auto',UVLayoutError('CPU layout rejected'),False,['cpu']),
])
def test_explicit_cpu_geometry_failures_and_missing_gpu_do_not_silently_change_backend(tmp_path,monkeypatch,requested,error,available,expected):
    source,geometry,calls=setup_case(tmp_path,monkeypatch,cpu_error=error,available=available)
    output=tmp_path/'stage/result.glb'
    with pytest.raises(RuntimeError):material_transfer.run(source,geometry,output,output.parent,uv_backend=requested)
    assert calls==expected and not output.exists()


def test_failed_gpu_retry_propagates_without_loop_or_partial_publication(tmp_path,monkeypatch):
    source,geometry,calls=setup_case(tmp_path,monkeypatch,
        cpu_error=UVLayoutError('CPU layout rejected'),gpu_error=UVLayoutError('GPU layout rejected'))
    output=tmp_path/'stage/result.glb'
    with pytest.raises(UVLayoutError,match='GPU layout rejected'):
        material_transfer.run(source,geometry,output,output.parent,uv_backend='auto')
    assert calls==['cpu','gpu'] and not output.exists()
    assert not list((common.DATA/'studio-cache/uv').glob('*/manifest.json'))
