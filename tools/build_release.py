"""Build source/node zips and an optional relocatable Windows runtime directory."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hi3d_workbench.common import VERSION
DIST=ROOT/'dist'
ALLOWED=['docs','validation','samples/sculpted-vase.glb','samples/sculpted-vase.png','samples/README.md','licenses','LICENSE','hi3d_workbench','scripts','examples','tools','tests','__init__.py','comfy_nodes.py','requirements.txt','requirements-worker-windows.txt','pyproject.toml','Start-Workbench.cmd','README.md','README.zh-CN.md','NOTICE.md','VALIDATION.md','upstream-source-manifest.json','config.example.json']

def files(root):
    return [p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.pyc')]

def build_source():
    DIST.mkdir(exist_ok=True)
    out=DIST/f'Hi3D-Workbench-source-{VERSION}.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ALLOWED:
            p=ROOT/name
            for f in files(p) if p.is_dir() else [p]:
                if f.is_file():z.write(f,f.relative_to(ROOT))
        source=ROOT/'upstream/Hi3Dv1.0'
        for name in ['sparc3d','LICENSE','NOTICE','THIRD_PARTY_LICENSES.md','AUTHORS','docs','docker/constraints.txt','pyproject.toml','README.md','MANIFEST.in']:
            p=source/name
            for f in files(p) if p.is_dir() else [p]:
                if f.is_file():z.write(f,f.relative_to(ROOT))
    with zipfile.ZipFile(DIST/f'ComfyUI-Hi3D-nodes-{VERSION}.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['__init__.py','comfy_nodes.py','requirements.txt','pyproject.toml','README.md','README.zh-CN.md','docs/COMFYUI.md','docs/images/comfyui-canvas.png','NOTICE.md','LICENSE']:
            z.write(ROOT/name,Path('ComfyUI-Hi3D')/name)
        for f in files(ROOT/'examples'):z.write(f,Path('ComfyUI-Hi3D')/f.relative_to(ROOT))
    return out

def portable(base,include_models,name):
    if Path(name).name!=name:raise ValueError('发行目录名称不能包含路径')
    if include_models:
        from hi3d_workbench.models import verify_all
        if not verify_all()['ok']:raise SystemExit('模型尚未完成完整校验，不能生成完整离线包')
    out=DIST/name
    if out.exists():raise SystemExit('发行目录已存在，不覆盖现有发行版')
    out.mkdir()
    with zipfile.ZipFile(build_source()) as z:z.extractall(out)
    runtime=out/'runtime';runtime.mkdir()
    for f in base.iterdir():
        if f.is_file() and (f.name in ['python.exe','python3.dll','python310.dll','LICENSE.txt'] or f.suffix.lower()=='.dll'):
            shutil.copy2(f,runtime/f.name)
    for name in ['DLLs','include','libs']:
        shutil.copytree(base/name,runtime/name)
    # Standard library only, excluding all base-environment third-party packages.
    with zipfile.ZipFile(runtime/'python310.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in files(base/'Lib'):
            rel=f.relative_to(base/'Lib')
            if rel.parts[0] not in {'site-packages','test','tests','idlelib','tkinter','ensurepip'}:z.write(f,rel)
    site=runtime/'Lib/site-packages';site.parent.mkdir(exist_ok=True)
    shutil.copytree(ROOT/'.venv/Lib/site-packages',site,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (runtime/'python310._pth').write_text('python310.zip\nDLLs\nLib/site-packages\n..\nimport site\n')
    packages=subprocess.check_output([str(runtime/'python.exe'),'-m','pip','freeze'],cwd=out,text=True,encoding='utf-8')
    (out/'runtime-packages.txt').write_text(packages,encoding='utf-8')
    (out/'config.local.json').write_text(json.dumps({'worker_python':'runtime/python.exe','worker_command':[],'minimum_free_vram_mb':1500,'port':8378}),encoding='utf-8')
    if include_models:
        shutil.copytree(ROOT/'models',out/'models',ignore=shutil.ignore_patterns('.cache','.downloads','*.tmp','*.incomplete'))
    # No input images, jobs or personal paths included.
    return out

def manifest():
    values={}
    for f in DIST.glob('*.zip'):
        h=hashlib.sha256()
        with f.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        values[f.name]={'bytes':f.stat().st_size,'sha256':h.hexdigest()}
    (DIST/'SHA256.json').write_text(json.dumps(values,indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--portable-base',type=Path);p.add_argument('--include-models',action='store_true');p.add_argument('--name',default=f'Hi3D-Workbench-Windows-{VERSION}');args=p.parse_args()
    print(portable(args.portable_base,args.include_models,args.name) if args.portable_base else build_source())
    manifest()
