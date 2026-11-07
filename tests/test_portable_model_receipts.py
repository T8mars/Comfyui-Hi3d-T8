import hashlib
import json
import shutil
from pathlib import Path

import pytest

from test_workbench import testroot
from hi3d_workbench import common, models
from tools import verify_portable_models as portable


def test_copied_core_receipt_requires_real_rehash_and_rejects_changed_bytes(tmp_path,monkeypatch):
    source=tmp_path/'source';target=tmp_path/'copied'
    spec={'repo':'test/model','revision':'pinned','files':['weights.bin'],'license':'fixture'}
    payload=b'actual model bytes'
    def configure(root):
        for module in (common,models):
            for name,value in [('ROOT',root),('MODELS',root/'models'),('DATA',root/'data')]:
                monkeypatch.setattr(module,name,value,raising=False)
        monkeypatch.setattr(common,'__file__',str(root/'hi3d_workbench/common.py'))
        monkeypatch.setattr(portable.sys,'executable',str(root/'runtime/python.exe'))
    configure(source);monkeypatch.setattr(models,'SPECS',{'tiny':spec})
    weight=source/'models/tiny/weights.bin';weight.parent.mkdir(parents=True)
    weight.write_bytes(payload)
    common.atomic_json(source/'models/manifest.json',{'components':{'tiny':{
        'repo':spec['repo'],'revision':spec['revision'],'files':{
            'weights.bin':{'sha256':hashlib.sha256(payload).hexdigest()}}}}})
    assert models.verify_all()['ok']
    shutil.copytree(source,target);configure(target)
    with pytest.raises(RuntimeError,match='verification failed'):portable.verify(target,'core',True)
    assert portable.verify(target,'core')['ready']
    assert portable.verify(target,'core',True)['ready']
    copied=target/'models/tiny/weights.bin';copied.write_bytes(b'X'+payload[1:])
    with pytest.raises(RuntimeError,match='verification failed'):portable.verify(target,'core')
    assert not models.status()['ready']


def test_portable_verifier_rejects_source_interpreter(tmp_path,monkeypatch):
    monkeypatch.setattr(common,'ROOT',tmp_path)
    monkeypatch.setattr(common,'MODELS',tmp_path/'models')
    with pytest.raises(RuntimeError,match='copied application'):portable.verify(tmp_path,'core')


def test_builder_uses_private_runtimes_and_removes_developer_overrides(tmp_path,monkeypatch):
    from tools import build_release as builder
    monkeypatch.setenv('HI3D_STUDIO_RUNTIME','E:/private/runtime')
    monkeypatch.setenv('PYTHONPATH','E:/private/source')
    calls=[]
    monkeypatch.setattr(builder.subprocess,'run',lambda command,**kwargs:calls.append((command,kwargs)))
    builder.verify_copied_models(tmp_path,True,True)
    assert len(calls)==5
    assert [Path(command[0]).parent.name for command,_ in calls]==[
        'runtime','runtime','runtime-pipeline','runtime-assets','runtime']
    assert sum('--check-only' in command for command,_ in calls)==2
    for command,options in calls:
        assert Path(command[2])==tmp_path/'tools/verify_portable_models.py'
        assert options['cwd']==tmp_path and options['check'] is True
        assert options['env']['HI3D_WORKBENCH_ROOT']==str(tmp_path)
        assert 'HI3D_STUDIO_RUNTIME' not in options['env'] and 'PYTHONPATH' not in options['env']
