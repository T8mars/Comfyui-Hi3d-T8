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
