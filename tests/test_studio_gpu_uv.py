"""GPU handoff contract and strict geometry/UV acceptance, without a GPU mock model."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import trimesh

from test_workbench import testroot
from hi3d_workbench.studio import uv_atlas,processes
from hi3d_workbench.studio.gltf import Document
from hi3d_workbench import jobs


def geometry(path):
    vertices=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0.]],dtype=np.float32)
    faces=np.array([[0,1,2],[0,2,3]],dtype=np.uint32)
    trimesh.Trimesh(vertices,faces,process=False).export(path)
    return vertices,faces


def test_auto_keeps_small_cpu_work_and_requires_available_private_gpu_runtime(tmp_path,monkeypatch):
    source=tmp_path/'mesh.obj';geometry(source)
    def missing():raise RuntimeError('missing private runtime')
    monkeypatch.setattr(processes,'pipeline_command',missing)
    assert uv_atlas.select_backend(source,'auto')=='cpu'
    with pytest.raises(RuntimeError,match='private runtime'):uv_atlas.select_backend(source,'gpu')
    monkeypatch.setattr(processes,'pipeline_command',lambda:['private-python'])
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False)))
    with pytest.raises(RuntimeError,match='CUDA'):uv_atlas.select_backend(source,'gpu')
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True)))
    assert uv_atlas.select_backend(source,'gpu')=='gpu'
    with pytest.raises(ValueError,match='方式'):uv_atlas.select_backend(source,'unknown')


@pytest.mark.parametrize('damage',[None,'identity','missing_face','winding','zero_uv'])
def test_gpu_stage_requires_current_input_every_oriented_face_and_valid_uv(tmp_path,monkeypatch,damage):
    source=tmp_path/'mesh.obj';vertices,faces=geometry(source)
    target=tmp_path/'atlas.glb';owner=tmp_path/'job'
    monkeypatch.setattr(processes,'pipeline_command',lambda:['private-python','-m','hi3d_workbench.studio.gpu_uv'])
    calls=[]
    def stage(command,payload,directory):
        assert directory==owner/'stages/studio-uv'
        with np.load(payload['arrays'],allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['vertices'],vertices)
            np.testing.assert_array_equal(saved['faces'],faces)
        mapping=faces.reshape(-1);indices=np.arange(6).reshape(-1,3)
        uv=np.array([[0,0],[.4,0],[0,.4],[.6,.6],[1,.6],[.6,1]],dtype=np.float32)
        if damage=='missing_face':indices=indices[:1]
        if damage=='winding':indices=indices[:,::-1]
        if damage=='zero_uv':uv[:3]=0
        metadata={'input_sha256':'wrong' if damage=='identity' else payload['input_sha256']}
        np.savez_compressed(payload['output'],mapping=mapping,faces=indices,uv=uv,metadata=np.asarray(json.dumps(metadata)))
        calls.append(payload)
    monkeypatch.setattr(processes,'run_stage',stage)
    if damage:
        with pytest.raises(RuntimeError):uv_atlas.create(source,target,backend='gpu',owner=owner,size=1024)
        assert not target.exists()
    else:
        report=uv_atlas.create(source,target,backend='gpu',owner=owner,size=1024)
        assert report['oriented_geometry_preserved'] and report['zero_uv_faces']==report['duplicate_uv_faces']==0
        doc=Document.load(target);primitive=doc.tree['meshes'][0]['primitives'][0]
        output=doc.accessor(primitive['attributes']['POSITION'])[doc.accessor(primitive['indices']).reshape(-1,3)]
        np.testing.assert_array_equal(output,vertices[faces])
        uv_atlas.validate_cached(target)
    assert len(calls)==1


def test_gpu_stage_is_included_in_job_ownership_and_restart_recovery(tmp_path):
    request=tmp_path/'request.json';stage=tmp_path/'stages/studio-uv/request.json'
    assert jobs._worker_request(['private-python','-m','hi3d_workbench.studio.gpu_uv','--request',str(stage)])==str(stage)
    assert str(stage) in jobs._job_requests(request)


def test_structurally_bad_cached_layout_is_rejected_even_if_glb_loads(tmp_path):
    document=Document({'asset':{'version':'2.0'}});target=tmp_path/'empty.glb';document.save(target)
    with pytest.raises(RuntimeError,match='缺少网格'):uv_atlas.validate_cached(target)
