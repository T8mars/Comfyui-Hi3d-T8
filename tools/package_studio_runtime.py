"""Package a verified private asset runtime with only relative runtime paths."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil

REQUIRED_FEATURES=('meshoptimizer','retopology','parts','skintokens','unirig','animation','kimodo')
IGNORE=shutil.ignore_patterns('__pycache__','*.pyc','*.nbi','*.nbc','.git','direct_url.json','__editable__*')
REQUIRED_NOTICES={'parts-code','skintokens-code','unirig-code','kimodo-code','meshoptimizer',
    'sonata','autoremesher-core','fpsample','motion-correction-eigen'}
REQUIRED_NOTICES.update('model-'+name for name in json.loads(
    (Path(__file__).resolve().parents[1]/'hi3d_workbench/studio/model-catalog.json').read_text(encoding='utf-8')))


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def validate_notices(folder):
    folder=Path(folder).resolve()
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    rows=manifest.get('components',[])
    if manifest.get('schema')!=1 or not isinstance(rows,list):
        raise RuntimeError('资产许可清单格式无效')
    names=[row.get('component') for row in rows if isinstance(row,dict)]
    if len(names)!=len(rows) or len(set(names))!=len(names) or not REQUIRED_NOTICES.issubset(names):
        raise RuntimeError('资产许可清单缺少组件或包含重复组件')
    for row in rows:
        if not row.get('upstream') or not row.get('revision') or not row.get('files'):
            raise RuntimeError('资产许可缺少来源、版本或文件')
        for entry in row['files']:
            name=entry.get('path','')
            path=PurePosixPath(name)
            if (not name or '\\' in name or path.is_absolute() or PureWindowsPath(name).drive
                    or any(part in {'.','..'} for part in path.parts)):
                raise RuntimeError('资产许可文件路径无效')
            target=(folder/name).resolve()
            if not target.is_relative_to(folder) or not target.is_file():
                raise RuntimeError('资产许可文件不存在或越界：'+name)
            if target.stat().st_size!=entry.get('bytes') or sha(target)!=entry.get('sha256'):
                raise RuntimeError('资产许可文件校验失败：'+name)
    return manifest


def validate_sources(folder):
    folder=Path(folder).resolve()
    manifest=json.loads((folder/'source-manifest.json').read_text(encoding='utf-8'))
    components=manifest.get('components',{})
    if manifest.get('schema')!=1 or set(components)!={'parts','skintokens','unirig','kimodo'}:
        raise RuntimeError('资产源码清单缺少固定版本组件')
    for name,spec in components.items():
        if len(spec.get('revision',''))!=40 or not spec.get('files'):
            raise RuntimeError('资产源码版本或文件清单无效：'+name)
        directory=folder/name
        for relative,entry in spec['files'].items():
            path=PurePosixPath(relative)
            if path.is_absolute() or PureWindowsPath(relative).drive or '\\' in relative or '..' in path.parts:
                raise RuntimeError('资产源码清单路径无效')
            target=(directory/relative).resolve()
            if not target.is_relative_to(directory) or not target.is_file():
                raise RuntimeError('资产源码文件缺失或越界：'+name+'/'+relative)
            data=target.read_bytes()
            actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            if len(data)!=entry.get('bytes') or actual!=entry.get('sha1'):
                raise RuntimeError('资产源码校验失败：'+name+'/'+relative)
        for target in directory.rglob('*'):
            if (target.is_file() and not any(part in {'__pycache__','.git'} for part in target.parts)
                    and target.suffix not in {'.pyc','.nbi','.nbc'}
                    and target.relative_to(directory).as_posix() not in spec['files']):
                raise RuntimeError('资产源码包含未记录文件：'+name+'/'+target.relative_to(directory).as_posix())
    return manifest


def native_directory(source):
    base=Path(source)/'autoremesher'
    core=next((path for path in (base/'core/autoremesher-core.exe',base/'autoremesher-core.exe') if path.is_file()),None)
    if core is not None:
        receipt=json.loads((core.parent/'build-receipt.json').read_text(encoding='utf-8'))
        if receipt.get('source_revision')!='4a49f82f720f647691c09a7d8a16fa63be208aad' or receipt.get('sha256')!=sha(core):
            raise RuntimeError('AutoRemesher 控制台程序与构建回执不匹配')
        for name in ('msvcp140.dll','vcruntime140.dll','vcruntime140_1.dll'):
            path=core.parent/name
            if not path.is_file() or receipt.get('runtime_dlls',{}).get(name)!=sha(path):
                raise RuntimeError('AutoRemesher 缺少已校验的运行库：'+name)
        allowed={'autoremesher-core.exe','build-receipt.json','msvcp140.dll','vcruntime140.dll','vcruntime140_1.dll'}
        if any(path.is_file() and path.relative_to(core.parent).as_posix() not in allowed for path in core.parent.rglob('*')):
            raise RuntimeError('AutoRemesher 控制台目录包含未记录文件')
        return core.parent
    raise RuntimeError('资产发行需要已校验的 AutoRemesher 控制台程序，不能打包旧 Qt GUI')


def validate_evidence(source,state):
    source=Path(source).resolve()
    for name,path in [('source_manifest_sha256','sources/source-manifest.json'),
                      ('notice_manifest_sha256','licenses/asset-studio/manifest.json')]:
        if state.get(name)!=sha(source/path):raise RuntimeError('资产环境来源／许可回执已变化')
    if state['features']['meshoptimizer'].get('sha256')!=sha(source/'meshoptimizer.dll'):
        raise RuntimeError('减面运行库与已验收版本不一致')
    for name in REQUIRED_FEATURES:
        entries=state['features'][name].get('validation_reports',[])
        if not entries:raise RuntimeError('资产环境缺少真实运行报告：'+name)
        for entry in entries:
            relative=entry.get('path','');path=PurePosixPath(relative)
            target=(source/relative).resolve()
            if (not relative.startswith('validation/runtime/') or path.is_absolute() or '\\' in relative
                    or '..' in path.parts or PureWindowsPath(relative).drive or not target.is_relative_to(source/'validation')):
                raise RuntimeError('资产运行证据路径无效')
            if not target.is_file() or sha(target)!=entry.get('sha256'):
                raise RuntimeError('资产运行证据丢失或已改变：'+name)
            report=json.loads(target.read_text(encoding='utf-8'))
            if (report.get('status')!='succeeded' or report.get('mode')!='studio'
                    or any(report.get(key)!=entry.get(key) for key in ('action','engine','input_sha256','output_sha256'))):
                raise RuntimeError('资产运行证据内容不匹配：'+name)


def validate(source,pipeline):
    source,pipeline=Path(source).resolve(),Path(pipeline).resolve()
    state=json.loads((source/'studio-runtime.json').read_text(encoding='utf-8'))
    missing=[name for name in REQUIRED_FEATURES if state.get('features',{}).get(name,{}).get('ready') is not True]
    if missing:
        raise RuntimeError('资产环境未全部验收，拒绝打包：'+', '.join(missing))
    required=['meshoptimizer.dll','Lib/site-packages',
        'engines/kimodo-v510','engines/unirig','sources/parts','sources/skintokens','sources/unirig','sources/kimodo',
        'fpsample-build.json','motion-correction-build.json','licenses']
    for name in required:
        if not (source/name).exists():raise RuntimeError('资产运行环境缺少 '+name)
    native_directory(source)
    validate_notices(source/'licenses/asset-studio')
    validate_sources(source/'sources')
    validate_evidence(source,state)
    for name in ['python.exe','python312.dll','python3.dll','python312.zip','DLLs']:
        if not (pipeline/name).exists():raise RuntimeError('Python 3.12 运行环境缺少 '+name)
    blender=Path(state.get('blender','blender/blender.exe'))
    blender=blender if blender.is_absolute() else source/blender
    if not blender.is_file():raise RuntimeError('Blender 运行环境不存在')
    if state.get('blender_sha256')!=sha(blender):raise RuntimeError('Blender 程序与已验收版本不一致')
    return state,blender


def package(source,pipeline,target):
    source,pipeline,target=(Path(path).resolve() for path in (source,pipeline,target))
    state,blender=validate(source,pipeline)
    if target.exists():raise ValueError('资产发行目录已存在，不覆盖现有运行环境')
    if target.is_relative_to(source) or target.is_relative_to(pipeline):
        raise ValueError('发行目录不能位于源运行环境内')
    relative_pipeline=Path(os.path.relpath(pipeline/'Lib/site-packages',target)).as_posix()
    if Path(relative_pipeline).is_absolute():raise RuntimeError('运行环境未生成相对路径')
    target.mkdir(parents=True)
    for path in pipeline.iterdir():
        if path.is_file() and (path.name in {'python.exe','python312.zip','LICENSE.txt'} or path.suffix.lower()=='.dll'):
            shutil.copy2(path,target/path.name)
    shutil.copytree(pipeline/'DLLs',target/'DLLs',ignore=IGNORE)
    for name in ['Lib','sources','licenses','validation']:
        shutil.copytree(source/name,target/name,ignore=IGNORE)
    shutil.copytree(native_directory(source),target/'autoremesher',ignore=IGNORE)
    for name in ['kimodo-v510','unirig']:
        shutil.copytree(source/'engines'/name,target/'engines'/name,ignore=IGNORE)
    shutil.copy2(source/'meshoptimizer.dll',target/'meshoptimizer.dll')
    for name in ['fpsample-build.json','motion-correction-build.json']:
        receipt=json.loads((source/name).read_text(encoding='utf-8'))
        receipt.pop('build_project',None)
        (target/name).write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    shutil.copytree(blender.parent,target/'blender',ignore=IGNORE)
    state['blender']='blender/'+blender.name
    (target/'studio-runtime.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    (target/'python312._pth').write_text('python312.zip\nDLLs\nLib/site-packages\n'+relative_pipeline+'\n..\nimport site\n',encoding='utf-8')
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=Path('runtime-assets'))
    parser.add_argument('--pipeline',type=Path,required=True)
    parser.add_argument('--target',type=Path,required=True)
    args=parser.parse_args();print(package(args.source,args.pipeline,args.target))
