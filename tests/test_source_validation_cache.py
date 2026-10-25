import hashlib
import json
import os
import sys

import pytest
from test_workbench import client  # Establish isolated paths before importing runtime modules.

from hi3d_workbench.engines import run_stage
from hi3d_workbench.engines.voxel_acceleration import supports_architecture


@pytest.fixture
def source(tmp_path,monkeypatch):
    directory=tmp_path/'sources/trellis2';directory.mkdir(parents=True)
    file=directory/'component.py';file.write_bytes(b'original source')
    manifest={'revision':'75fbf0183001ed9876c8dbb35de6b68552ee08bd',
              'files':{'component.py':hashlib.sha256(file.read_bytes()).hexdigest()}}
    (directory/'source-manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    monkeypatch.setattr(sys,'prefix',str(tmp_path))
    monkeypatch.setattr(sys,'path',list(sys.path))
    monkeypatch.setattr(run_stage,'_SOURCE_VERIFIED',{})
    return directory,file,manifest


def test_verified_source_reuses_hash_only_while_file_identity_is_unchanged(source,monkeypatch):
    directory,file,_=source
    assert run_stage.source_dir('trellis2')==directory
    original=type(file).read_bytes
    reads=[]
    def read(path):
        if path==file:reads.append(path)
        return original(path)
    monkeypatch.setattr(type(file),'read_bytes',read)
    assert run_stage.source_dir('trellis2')==directory and not reads
    before=file.stat()
    file.write_bytes(b'changed  source')
    os.utime(file,ns=(before.st_atime_ns,before.st_mtime_ns))
    with pytest.raises(RuntimeError,match='校验失败'):run_stage.source_dir('trellis2')
    assert reads


def test_changed_source_manifest_is_rechecked(source):
    directory,file,manifest=source
    run_stage.source_dir('trellis2')
    manifest['files']['component.py']='0'*64
    (directory/'source-manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    with pytest.raises(RuntimeError,match='校验失败'):run_stage.source_dir('trellis2')


@pytest.mark.parametrize('capability,expected',[( (8,6),True),((8,9),True),((9,0),False),((12,0),True),((12,1),True),((7,5),False)])
def test_native_architecture_guard_uses_real_cubin_and_ptx_coverage(capability,expected):
    assert supports_architecture('8.6;8.9;12.0+PTX',capability) is expected
