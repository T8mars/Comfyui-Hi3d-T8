"""A valid ZIP CRC must not hide stale application or ComfyUI source."""
import importlib.util
from pathlib import Path
import zipfile
import pytest

@pytest.fixture
def audit_module():
    path=Path(__file__).parents[1]/'tools/audit_release.py'
    spec=importlib.util.spec_from_file_location('release_audit_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def test_source_audit_rejects_changed_job_manager(tmp_path,audit_module):
    root=tmp_path/'source'
    names=['hi3d_workbench/jobs.py','hi3d_workbench/server.py','hi3d_workbench/common.py','hi3d_workbench/web/app.js']
    for name in names:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'current-source')
    for corrupt in [False,True]:
        archive=tmp_path/f'{corrupt}.zip'
        with zipfile.ZipFile(archive,'w') as z:
            for name in names:z.writestr(name,b'old-buggy-job-manager' if corrupt and name.endswith('jobs.py') else b'current-source')
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            if corrupt:
                with pytest.raises(AssertionError,match='jobs.py'):audit_module.verify_owned(z,source_root=root)
            else:assert audit_module.verify_owned(z,source_root=root)==4

def test_node_audit_rejects_stale_comfy_node(tmp_path,audit_module):
    root=tmp_path/'source';names=['__init__.py','comfy_nodes.py','requirements.txt','NOTICE.md','LICENSE','README.md']
    for name in names:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'current-source')
    archive=tmp_path/'nodes.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for name in names:z.writestr('ComfyUI-Hi3D/'+('README.md' if name.startswith('docs/') else name),b'stale-node' if name=='comfy_nodes.py' else b'current-source')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        with pytest.raises(AssertionError,match='comfy_nodes.py'):audit_module.verify_nodes(z,source_root=root)


def test_release_file_inventory_omits_machine_specific_numba_cache(tmp_path):
    path=Path(__file__).parents[1]/'tools/build_release.py'
    spec=importlib.util.spec_from_file_location('release_builder_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for name in ('keep.py','cached.pyc','packer.nbi','packer.nbc','__pycache__/other.py'):
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b'x')
    assert [entry.name for entry in module.files(tmp_path)]==['keep.py']


def test_shared_custom_model_folder_only_bundles_catalogued_files(tmp_path):
    import json
    from tools.build_release import copy_model_components
    source=tmp_path/'共享 模型';target=tmp_path/'portable/models'
    for name,content in [('hi3d/vae/model.safetensors',b'weights'),('hi3d/LICENSE',b'license'),
                         ('private.txt',b'private'),('other-engine/huge.safetensors',b'other'),
                         ('hi3d/unrelated.txt',b'unlisted')]:
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    component={'files':{'vae/model.safetensors':{'bytes':7},'LICENSE':{'bytes':7}},'license':'original'}
    (source/'manifest.json').write_text(json.dumps({'components':{'hi3d':component,'unused':{'files':{}}}}))
    copy_model_components(source,target,['hi3d'])
    assert {p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()} == {
        'hi3d/vae/model.safetensors','hi3d/LICENSE','manifest.json'}
    assert (target/'hi3d/vae/model.safetensors').read_bytes()==b'weights'
    assert json.loads((target/'manifest.json').read_text())['components']=={'hi3d':component}
    assert (source/'private.txt').read_bytes()==b'private'


@pytest.mark.parametrize('name',['../../private.txt','C:/private.txt','../other/model.safetensors'])
def test_model_bundle_rejects_catalogue_path_escape(tmp_path,name):
    import json
    from tools.build_release import copy_model_components
    source=tmp_path/'source';source.mkdir()
    (source/'manifest.json').write_text(json.dumps({'components':{'hi3d':{'files':{name:{}}}}}))
    with pytest.raises(ValueError,match='artifact path'):
        copy_model_components(source,tmp_path/'target',['hi3d'])
