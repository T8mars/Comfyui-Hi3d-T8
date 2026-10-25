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
ALLOWED=['docs','validation','samples/sculpted-vase.glb','samples/sculpted-vase.png','samples/README.md','licenses','LICENSE','hi3d_workbench','scripts','examples','tools','tests','__init__.py','comfy_nodes.py','requirements.txt','requirements-worker-windows.txt','requirements-pipeline-windows.txt','roadmap.md','pyproject.toml','Start-Workbench.cmd','Update-Workbench.cmd','README.md','README.zh-CN.md','NOTICE.md','VALIDATION.md','upstream-source-manifest.json','config.example.json']

def files(root):
    return [p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.pyc')]

def build_source(dist=None):
    dist=Path(dist or DIST)
    dist.mkdir(parents=True,exist_ok=True)
    out=dist/f'Hi3D-Workbench-source-{VERSION}.zip'
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
    with zipfile.ZipFile(dist/f'ComfyUI-Hi3D-nodes-{VERSION}.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['__init__.py','comfy_nodes.py','requirements.txt','pyproject.toml','README.md','README.zh-CN.md','docs/COMFYUI.md','docs/images/comfyui-canvas.png','NOTICE.md','LICENSE']:
            z.write(ROOT/name,Path('ComfyUI-Hi3D')/name)
        for f in files(ROOT/'examples'):z.write(f,Path('ComfyUI-Hi3D')/f.relative_to(ROOT))
    return out

def build_update(dist=None, source_archive=None):
    """A verified code overlay; runtimes, weights and user data are excluded."""
    from hi3d_workbench.updater import owned_name, validate_archive, RUNTIME_API
    dist=Path(dist or DIST)
    source_archive=source_archive or build_source(dist)
    out=dist/f'Hi3D-Workbench-update-{VERSION}.zip'
    entries={}
    with zipfile.ZipFile(source_archive) as source, zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as target:
        for name in source.namelist():
            owned_name(name)
            content=source.read(name)
            entries[name]={'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
            target.writestr(name,content)
    value={'schema':1,'repository':'T8mars/Comfyui-Hi3d-T8','version':VERSION,
           'runtime_api':RUNTIME_API,'minimum_version':'0.4.0','archive':out.name,
           'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
           'files':entries,'preserved':['models','runtime','runtime-pipeline','data','config.local.json']}
    (dist/'update-manifest.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    validate_archive(out,value,ROOT)
    return out

def pipeline_runtime(out):
    """Make the private cp312 runtime relocatable without copying venv launchers."""
    source=ROOT/'.pipelineenv'
    state=json.loads((source/'pipeline-runtime.json').read_text(encoding='utf-8'))
    if not state.get('ready'):raise RuntimeError('扩展运行环境尚未就绪')
    data=dict(line.split(' = ',1) for line in (source/'pyvenv.cfg').read_text().splitlines() if ' = ' in line)
    base=Path(data['home'])
    target=out/'runtime-pipeline';target.mkdir()
    for path in base.iterdir():
        if path.is_file() and (path.name in {'python.exe','python3.dll','python312.dll','LICENSE.txt'} or path.suffix.lower()=='.dll'):
            shutil.copy2(path,target/path.name)
    for name in ('DLLs','include','libs'):
        if (base/name).is_dir():shutil.copytree(base/name,target/name)
    with zipfile.ZipFile(target/'python312.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in files(base/'Lib'):
            rel=path.relative_to(base/'Lib')
            if rel.parts[0] not in {'site-packages','test','tests','idlelib','tkinter','ensurepip'}:z.write(path,rel)
    shutil.copytree(source/'Lib/site-packages',target/'Lib/site-packages',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(source/'sources',target/'sources',ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
    shutil.copy2(source/'pipeline-runtime.json',target/'pipeline-runtime.json')
    (target/'python312._pth').write_text('python312.zip\nDLLs\nLib/site-packages\n..\nimport site\n')
    packages=subprocess.check_output([str(target/'python.exe'),'-m','pip','freeze'],cwd=out,text=True,encoding='utf-8')
    (out/'runtime-pipeline-packages.txt').write_text(packages,encoding='utf-8')
    return target

def portable(base,include_models,name,include_pipeline=False,pipeline_engines=(),dist=None):
    if Path(name).name!=name:raise ValueError('发行目录名称不能包含路径')
    selected=[]
    if include_pipeline:
        from hi3d_workbench.pipeline_models import selection_components
        selected=list(dict.fromkeys(selection_components('all') +
            [key for choice in pipeline_engines for key in selection_components(choice)]))
    elif pipeline_engines:raise ValueError('Engine selections require --include-pipeline')
    if include_models:
        from hi3d_workbench.models import verify_all
        if not verify_all()['ok']:raise SystemExit('模型尚未完成完整校验，不能生成完整离线包')
        if include_pipeline:
            from hi3d_workbench.pipeline_models import verify
            state=verify()
            if not state['ready'] or any(not state['components'][key]['ready'] for key in selected):
                raise SystemExit('所选扩展模型或环境尚未完整校验，不能生成完整资产工作台包')
    parent=Path(dist or DIST).resolve()
    parent.mkdir(parents=True,exist_ok=True)
    out=parent/name
    if out.exists():raise SystemExit('发行目录已存在，不覆盖现有发行版')
    out.mkdir()
    with zipfile.ZipFile(build_source(parent)) as z:z.extractall(out)
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
    if include_pipeline:pipeline_runtime(out)
    if include_models:
        omitted=['.cache','.downloads','*.tmp','*.incomplete','*.parts*','*.assembled','*.bak','extensions','download-state.json','.download.lock']
        shutil.copytree(ROOT/'models',out/'models',ignore=shutil.ignore_patterns(*omitted))
        if include_pipeline:
            extension=out/'models/extensions';extension.mkdir()
            installed=json.loads((ROOT/'models/extensions/manifest.json').read_text(encoding='utf-8'))
            for key in selected:
                shutil.copytree(ROOT/'models/extensions'/key,extension/key,ignore=shutil.ignore_patterns(*omitted))
            from hi3d_workbench.common import atomic_json
            atomic_json(extension/'manifest.json',{'components':{key:installed['components'][key] for key in selected}})
    from hi3d_workbench.common import atomic_json
    atomic_json(out/'bundle-manifest.json',{'version':VERSION,'pipeline_runtime':include_pipeline,
        'models_included':include_models,'extension_components':selected if include_models else [],
        'extra_engine_selections':list(pipeline_engines),
        'optional_engines':'Other supported engines can be downloaded separately in the workbench.'})
    # No input images, jobs or personal paths included.
    return out

def manifest(dist=None):
    dist=Path(dist or DIST)
    values={}
    for f in dist.glob('*.zip'):
        h=hashlib.sha256()
        with f.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        values[f.name]={'bytes':f.stat().st_size,'sha256':h.hexdigest()}
    (dist/'SHA256.json').write_text(json.dumps(values,indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--portable-base',type=Path);p.add_argument('--include-models',action='store_true');p.add_argument('--include-pipeline',action='store_true');p.add_argument('--pipeline-engines',nargs='*',default=[]);p.add_argument('--dist-root',type=Path,default=DIST);p.add_argument('--name',default=f'Hi3D-Workbench-Windows-{VERSION}');args=p.parse_args()
    if args.portable_base:
        print(portable(args.portable_base,args.include_models,args.name,args.include_pipeline,args.pipeline_engines,args.dist_root))
    else:
        source=build_source(args.dist_root)
        print(source)
        print(build_update(args.dist_root,source))
    manifest(args.dist_root)
