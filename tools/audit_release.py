"""Check final release CRCs, fixed model/upstream hashes and data exclusion."""
import argparse
import datetime
import hashlib
import json
import sys
import zipfile
from pathlib import Path,PurePosixPath

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hi3d_workbench.common import VERSION

def stream_hash(stream):
    value=hashlib.sha256()
    for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()

def _compare(z,name,source):
    with source.open('rb') as stream:expected=stream_hash(stream)
    try:
        with z.open(name) as stream:actual=stream_hash(stream)
    except KeyError:raise AssertionError(f'发行包缺少文件：{name}') from None
    assert actual==expected,f'发行包文件不是当前版本：{name}'

def verify_paths(names):
    assert len(names)==len(set(names)),'发行包包含重名条目'
    for name in names:
        parts=PurePosixPath(name).parts
        assert parts and not name.startswith(('/', '\\')) and '..' not in parts and ':' not in parts[0],name
        logical=parts[1:] if parts[0]=='ComfyUI-Hi3D' else parts
        assert logical and logical[0] not in {'data','research'},name
        assert not any(p in {'.git','.secrets','.cache','.downloads','__pycache__'} or p=='.env' or p.startswith('.env.') for p in parts),f'发行包包含本机私有或缓存文件：{name}'
        assert not any('.parts' in p or p.endswith('.assembled') for p in parts),f'发行包包含未完成或损坏的下载分片：{name}'
        assert parts[-1]!='hi3d_refine_api.json',name

def verify_owned(z,prefix='',source_root=None):
    root=source_root or ROOT
    for name in ['hi3d_workbench/jobs.py','hi3d_workbench/server.py','hi3d_workbench/common.py','hi3d_workbench/web/app.js']:
        assert (root/name).is_file(),f'校验源码不完整：{name}'
    paths=[]
    for directory in ['hi3d_workbench','scripts','examples','tools','tests','docs','licenses']:
        paths.extend(p for p in (root/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    for name in ['__init__.py','comfy_nodes.py','requirements.txt','requirements-worker-windows.txt','requirements-pipeline-windows.txt','roadmap.md','pyproject.toml','Start-Workbench.cmd','README.md','README.zh-CN.md','NOTICE.md','LICENSE','config.example.json','upstream-source-manifest.json','VALIDATION.md']:
        if (root/name).is_file():paths.append(root/name)
    for path in paths:_compare(z,prefix+path.relative_to(root).as_posix(),path)
    return len(paths)

def verify_nodes(z,source_root=None):
    root=source_root or ROOT;count=0
    for name in ['__init__.py','comfy_nodes.py','requirements.txt','NOTICE.md','LICENSE']:
        _compare(z,'ComfyUI-Hi3D/'+name,root/name);count+=1
    _compare(z,'ComfyUI-Hi3D/README.md',root/'README.md');count+=1
    for name in ['pyproject.toml','README.zh-CN.md','docs/COMFYUI.md','docs/images/comfyui-canvas.png']:
        if (root/name).is_file():_compare(z,'ComfyUI-Hi3D/'+name,root/name);count+=1
    for path in (root/'examples').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            _compare(z,'ComfyUI-Hi3D/'+path.relative_to(root).as_posix(),path);count+=1
    return count


def verify_bundle(bundle,components,runtime):
    """A declaration must describe the exact installed model dependency closure."""
    from hi3d_workbench.pipeline_models import selection_components
    from hi3d_workbench.engine_catalog import engine, covers_all_engines
    assert bundle.get('version')==VERSION,'发行清单版本错误'
    assert bundle.get('models_included') is True,'完整包未声明模型'
    assert bundle.get('pipeline_runtime') is True and runtime.get('ready') is True,'发行清单扩展运行时不匹配'
    expected=set(selection_components('all'))
    for selection in bundle.get('extra_engine_selections',[]):
        expected.update(selection_components(selection))
        if selection.startswith('engine:'):
            _,stage,name=selection.split(':')
            feature=engine(stage,name)['runtime_feature']
            assert feature=='baseline' or feature in runtime.get('features',[]),'发行清单缺少运行时特性'
    declared=bundle.get('extension_components',[])
    assert len(declared)==len(set(declared)),'发行清单包含重复组件'
    assert set(declared)==set(components)==expected,'发行清单组件与引擎依赖或实际文件不匹配'
    if 'full_engine_coverage' in bundle:
        assert bundle['full_engine_coverage'] is covers_all_engines(declared),'发行清单完整引擎覆盖声明错误'
    return {'declared_components':sorted(expected),'engine_dependency_closure':'passed'}

def audit(release,archive_dir=None):
    dist=Path(archive_dir).resolve() if archive_dir else release.parent
    catalog=json.loads((dist/'SHA256.json').read_text(encoding='utf-8'))
    source_manifest=json.loads((ROOT/'upstream-source-manifest.json').read_text(encoding='utf-8'))
    results={}
    for name in [release.name+'.zip',f'Hi3D-Workbench-source-{VERSION}.zip',f'ComfyUI-Hi3D-nodes-{VERSION}.zip']:
        path=dist/name
        print('Auditing',name,flush=True)
        with path.open('rb') as stream:actual=stream_hash(stream)
        assert actual==catalog[name]['sha256'] and path.stat().st_size==catalog[name]['bytes']
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None,'CRC failure'
            names=z.namelist();prefix=release.name+'/' if name==release.name+'.zip' else ''
            if prefix:assert all(n.startswith(prefix) for n in names)
            stripped=[n.removeprefix(prefix) for n in names]
            verify_paths(stripped)
            assert not any(n.startswith(('data/','.git/')) or '/__pycache__/' in n for n in stripped)
            assert not any('/assets/example' in n or n.startswith('research/') for n in stripped)
            result={'crc':'passed','files':len(names),'bytes':path.stat().st_size,'sha256':actual,'personal_data_excluded':True}
            if 'nodes' not in name:
                count=0
                for rel,expected in source_manifest['files'].items():
                    with z.open(prefix+'upstream/Hi3Dv1.0/'+rel) as stream:assert stream_hash(stream)==expected,rel
                    count+=1
                result['upstream_python_sha256_verified']=count
                result['integration_files_match_source']=verify_owned(z,prefix)
                assert 'By T8star' in z.read(prefix+'hi3d_workbench/web/index.html').decode('utf-8')
            else:result['node_files_match_source']=verify_nodes(z)
            if prefix:
                manifest=json.loads(z.read(prefix+'models/manifest.json'));count=0
                for key,component in manifest['components'].items():
                    for rel,entry in component['files'].items():
                        with z.open(prefix+f'models/{key}/{rel}') as stream:assert stream_hash(stream)==entry['sha256'],(key,rel)
                        count+=1
                result['model_artifacts_sha256_verified']=count
                extension_name=prefix+'models/extensions/manifest.json'
                if extension_name in names:
                    extension=json.loads(z.read(extension_name));verified=0
                    bundle=json.loads(z.read(prefix+'bundle-manifest.json'))
                    runtime=json.loads(z.read(prefix+'runtime-pipeline/pipeline-runtime.json'))
                    result['bundle_manifest']=verify_bundle(bundle,extension.get('components',{}),runtime)
                    from hi3d_workbench.pipeline_models import SPECS
                    assert not set(extension.get('components',{}))-set(SPECS),'发行包含未知扩展组件'
                    for key,spec in SPECS.items():
                        if spec.get('optional') and key not in extension.get('components',{}):continue
                        component=extension.get('components',{}).get(key,{})
                        assert component.get('repo')==spec['repo'] and component.get('revision')==spec['revision'],f'扩展组件版本不匹配：{key}'
                        assert all(rel in component.get('files',{}) for rel in spec['files']),f'扩展组件文件不完整：{key}'
                    for key,component in extension['components'].items():
                        for rel,entry in component['files'].items():
                            artifact_name=prefix+f'models/extensions/{key}/{rel}'
                            assert z.getinfo(artifact_name).file_size==entry['bytes'],artifact_name
                            h=hashlib.sha256() if entry['algorithm']=='sha256' else hashlib.sha1()
                            if entry['algorithm']=='git-sha1':h.update(f"blob {entry['bytes']}\0".encode('ascii'))
                            with z.open(artifact_name) as stream:
                                for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
                            assert h.hexdigest()==entry['hash'],artifact_name
                            verified+=1
                    assert prefix+'runtime-pipeline/python.exe' in names,'完整资产包缺少隔离扩展运行时'
                    assert json.loads(z.read(prefix+'runtime-pipeline/pipeline-runtime.json')).get('ready') is True,'扩展运行时未完成安装校验'
                    result['extension_artifacts_verified']=verified
                config=json.loads(z.read(prefix+'config.local.json'))
                assert config['worker_python']=='runtime/python.exe' and config['worker_command']==[]
                assert prefix+'runtime/python.exe' in names and prefix+'Start-Workbench.cmd' in names
            else:assert not any(n.startswith(('models/','runtime/')) for n in names)
        results[name]=result
    report={'version':VERSION,'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archives':results,'sha256_catalog':'SHA256.json'}
    target=dist/f'RELEASE-VERIFIED-{VERSION}.json'
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('release',type=Path);parser.add_argument('--archive-dir',type=Path);args=parser.parse_args()
    audit(args.release.resolve(),args.archive_dir)
