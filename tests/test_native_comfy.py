import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import trimesh
from hi3d_native import models
from hi3d_native import runtime


@pytest.fixture
def nodes(monkeypatch,tmp_path):
    spec=importlib.util.spec_from_file_location('standalone_comfy_nodes',Path(__file__).parents[1]/'comfy_nodes.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules,'folder_paths',SimpleNamespace(
        get_input_directory=lambda:str(tmp_path),get_annotated_filepath=lambda value:str(tmp_path/value)))
    monkeypatch.setattr(module,'mesh_value',lambda v,f:SimpleNamespace(vertices=v,faces=f))
    return module


@pytest.mark.parametrize('extension',['glb','ply','stl'])
def test_uploaded_mesh_formats_load_locally_without_backend(nodes,tmp_path,extension):
    mesh=trimesh.creation.box();path=tmp_path/('uploaded.'+extension);mesh.export(path)
    value=nodes.Hi3DLoadMesh().load(path.name)[0]
    assert value.vertices.shape[1]==3 and value.faces.shape[1]==3
    assert np.allclose(np.ptp(value.vertices,axis=0),[1,1,1])
    assert path.name in nodes.Hi3DLoadMesh.INPUT_TYPES()['required']['mesh_file'][0]


def test_uploaded_glb_scene_preserves_instance_transform(nodes,tmp_path):
    scene=trimesh.Scene();transform=np.eye(4);transform[0,3]=4
    scene.add_geometry(trimesh.creation.box(),transform=transform)
    scene.export(tmp_path/'scene.glb')
    value=nodes.Hi3DLoadMesh().load('scene.glb')[0]
    assert np.allclose(value.vertices.mean(0),[4,0,0])


def test_uploaded_mesh_cannot_resolve_outside_input(nodes,tmp_path):
    path=tmp_path.parent/'external-coarse.glb';trimesh.creation.box().export(path)
    with pytest.raises(ValueError,match='上传'):nodes.Hi3DLoadMesh().load('../external-coarse.glb')


def test_custom_model_root_and_yaml_empty_target_are_respected(tmp_path,monkeypatch):
    custom=tmp_path/'custom';default=tmp_path/'default'
    monkeypatch.setitem(sys.modules,'folder_paths',SimpleNamespace(models_dir=str(default),
        folder_names_and_paths={'hi3d':([],set())},get_folder_paths=lambda name:[str(custom)]))
    assert models.model_root('')==custom.resolve()
    assert models.model_root(str(tmp_path/'explicit'))==(tmp_path/'explicit').resolve()


def test_missing_models_never_download_without_explicit_option(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'huggingface_hub',SimpleNamespace(snapshot_download=lambda **kw:pytest.fail('no implicit download')))
    with pytest.raises(FileNotFoundError,match='download_missing'):models.prepare(tmp_path,False)


def test_optional_download_is_pinned_and_only_fetches_core(tmp_path,monkeypatch):
    requests=[]
    def download(repo,**kw):
        requests.append((repo,kw))
        for path in models.required_paths(tmp_path):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture')
    monkeypatch.setitem(sys.modules,'huggingface_hub',SimpleNamespace(snapshot_download=download))
    assert models.prepare(tmp_path,True)==tmp_path
    repo,kwargs=requests[0]
    assert repo==models.REPOSITORY and len(kwargs['revision'])==40
    assert not any('extensions/' in name for name in kwargs['allow_patterns'])


def test_native_vendor_matches_all_canonical_pinned_hashes():
    manifest=json.loads((runtime.VENDOR/'source-manifest.json').read_text(encoding='utf-8'))
    assert manifest['revision']==runtime.SOURCE_REVISION
    assert 'sparc3d/ops/flexicubes/LICENSE.txt' in manifest['files']
    assert len([name for name in manifest['files'] if name.endswith('.py')])==61
    for name,digest in manifest['files'].items():
        body=(runtime.VENDOR/name).read_bytes().replace(b'\r\n',b'\n')
        assert hashlib.sha256(body).hexdigest()==digest,name


def test_native_source_corruption_is_rejected_before_any_model_import(tmp_path,monkeypatch):
    package=tmp_path/'sparc3d';package.mkdir();(package/'__init__.py').write_bytes(b'tampered')
    (tmp_path/'source-manifest.json').write_text(json.dumps({'revision':runtime.SOURCE_REVISION,
        'files':{'sparc3d/__init__.py':hashlib.sha256(b'original').hexdigest()}}))
    monkeypatch.setattr(runtime,'VENDOR',tmp_path);monkeypatch.setattr(runtime,'_SOURCE_CHECKED',False)
    with pytest.raises(RuntimeError,match='校验失败'):runtime.source()


def test_native_canvas_has_upload_mesh_model_handle_and_no_service_url():
    root=Path(__file__).parents[1];workflow=json.loads((root/'examples/Hi3D-Refine.json').read_text(encoding='utf-8'))
    assert workflow['version']==.4 and isinstance(workflow['nodes'],list)
    assert {node['type'] for node in workflow['nodes']}=={'Hi3DModelLoader','LoadImage','Hi3DLoadMesh','Hi3DRefine','SaveGLB'}
    assert len(workflow['links'])==5
    assert 'http://' not in json.dumps(workflow) and 'HI3D_BACKEND' not in json.dumps(workflow)
    assert 'hi3d_workbench' not in (root/'comfy_nodes.py').read_text(encoding='utf-8')


def test_node_package_relative_imports_work_without_workbench(monkeypatch,tmp_path):
    root=Path(__file__).parents[1]
    spec=importlib.util.spec_from_file_location('native_node_package_fixture',root/'__init__.py',submodule_search_locations=[str(root)])
    package=importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,spec.name,package)
    spec.loader.exec_module(package)
    model=package.NODE_CLASS_MAPPINGS['Hi3DModelLoader']().load(model_root=str(tmp_path))[0]
    assert model.root==tmp_path and model.pipeline is None
    assert type(model).__module__=='native_node_package_fixture.hi3d_native.runtime'
    assert package.WEB_DIRECTORY=='./web'
