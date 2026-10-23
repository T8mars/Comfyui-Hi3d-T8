"""Exercise actual archive selection with disposable files, never real secrets."""
import importlib.util
import json
from pathlib import Path
import zipfile
import pytest

ROOT=Path(__file__).resolve().parents[1]

def tool(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_portable_excludes_local_secrets_and_deleted_api_example(tmp_path):
    root=tmp_path/'portable'
    files={'runtime/python.exe':b'fixture','runtime/Lib/site-packages/transformers/data/__init__.py':b'public module','models/manifest.json':b'{}','README.md':b'public',
        '.env':b'fixture-not-a-real-secret','.secrets/token.txt':b'fixture','personal-photo.png':b'fixture',
        'docs/.env.local':b'fixture','data/input.png':b'fixture','examples/hi3d_refine_api.json':b'{}',
        'examples/Hi3D-Refine.json':b'{"nodes":[]}'}
    for name,data in files.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    path=tool('zip_portable').build(root)
    with zipfile.ZipFile(path) as z:
        assert set(z.namelist())=={'portable/'+p for p in ['runtime/python.exe','runtime/Lib/site-packages/transformers/data/__init__.py','models/manifest.json','README.md','examples/Hi3D-Refine.json']}
        tool('audit_release').verify_paths([n.removeprefix('portable/') for n in z.namelist()])

@pytest.mark.parametrize('name',['.env','.secrets/token.txt','docs/.env.local','../escape.py','C:/escape.py','data/private.png','examples/hi3d_refine_api.json'])
def test_release_audit_rejects_private_or_obsolete_paths(name):
    with pytest.raises(AssertionError):tool('audit_release').verify_paths([name])
