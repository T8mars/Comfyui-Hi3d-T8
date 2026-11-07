"""Record operational readiness from real worker outputs, separately from quality.

The input plan maps every required runtime feature to existing report.json files.
It never creates successful test results or marks visual/bundle acceptance passed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hi3d_workbench.common import atomic_copy,atomic_json,digest
from tools.package_studio_runtime import REQUIRED_FEATURES,validate_notices,validate_sources,native_directory

ACTIONS={'meshoptimizer':{'simplify'},'retopology':{'retopology'},'parts':{'segment','complete_parts'},
         'skintokens':{'rig'},'unirig':{'rig'},'animation':{'retarget'},'kimodo':{'motion'}}


def inspect_report(path,feature,root=ROOT):
    from hi3d_workbench.studio.gltf import Document
    from hi3d_workbench.studio.scene import instances
    path=Path(path).resolve();root=Path(root).resolve()
    if not path.is_relative_to(root/'data') or path.name!='report.json':
        raise ValueError('运行环境证据必须来自工作目录 data 下的 report.json')
    report=json.loads(path.read_text(encoding='utf-8'))
    if report.get('status')!='succeeded' or report.get('mode')!='studio' or report.get('action') not in ACTIONS[feature]:
        raise ValueError('运行环境证据操作不匹配：'+feature)
    engine=str(report.get('engine','')).lower()
    if feature in {'skintokens','unirig','meshoptimizer','kimodo'} and feature not in engine:
        raise ValueError('运行环境证据引擎不匹配：'+feature)
    for filename,key,count_key in [('coarse.glb','input_sha256','input_faces'),('output.glb','output_sha256','output_faces')]:
        model=path.parent/filename
        if not model.is_file() or digest(model)!=report.get(key):
            raise RuntimeError('运行环境证据文件缺失或哈希改变：'+str(model))
        document=Document.load(model);count=sum(len(row.faces) for row in instances(document))
        if not count or count!=report.get(count_key):raise RuntimeError('运行环境证据面数不匹配')
        if filename=='output.glb':
            if feature in {'skintokens','unirig','animation','kimodo'} and not document.tree.get('skins'):
                raise RuntimeError('运行环境证据缺少实际蒙皮')
            if feature in {'animation','kimodo'} and not document.tree.get('animations'):
                raise RuntimeError('运行环境证据缺少实际动画')
    return report


def record(plan,root=ROOT):
    from hi3d_workbench.studio.models import status
    root=Path(root).resolve();runtime=root/'runtime-assets'
    if set(plan)!=set(REQUIRED_FEATURES):raise ValueError('必须提供全部资产运行环境的证据计划')
    validate_sources(runtime/'sources');validate_notices(runtime/'licenses/asset-studio');native_directory(runtime)
    missing=[name for name,row in status()['components'].items() if not row['ready']]
    if missing:raise RuntimeError('资产模型校验回执无效：'+', '.join(missing))
    records={}
    # Validate every selected source/output before writing any readiness flag.
    for feature,paths in plan.items():
        if not isinstance(paths,list) or not paths:raise ValueError('证据计划为空：'+feature)
        reports=[(root/relative,inspect_report(root/relative,feature,root)) for relative in paths]
        if {report['action'] for _,report in reports}!=ACTIONS[feature]:
            raise ValueError('证据计划未覆盖全部操作：'+feature)
        records[feature]=reports
    state=json.loads((runtime/'studio-runtime.json').read_text(encoding='utf-8'))
    evidence=runtime/'validation/runtime';evidence.mkdir(parents=True,exist_ok=True)
    for feature,reports in records.items():
        entries=[]
        for index,(path,report) in enumerate(reports):
            destination=evidence/(feature+'-'+str(index)+'-'+report['action']+'.json')
            atomic_copy(path,destination)
            entries.append({'path':destination.relative_to(runtime).as_posix(),'sha256':digest(destination),
                'input_sha256':report['input_sha256'],'output_sha256':report['output_sha256'],
                'action':report['action'],'engine':report['engine']})
        previous=state.setdefault('features',{}).get(feature,{})
        state['features'][feature]={**previous,'ready':True,'runtime_validation':entries[0]['path'],
            'report_sha256':entries[0]['sha256'],'validation_reports':entries,
            'quality_scope':'operational runtime verified on recorded inputs; visual, browser and full-bundle acceptance remain separate'}
    state['source_manifest_sha256']=digest(runtime/'sources/source-manifest.json')
    state['notice_manifest_sha256']=digest(runtime/'licenses/asset-studio/manifest.json')
    blender=Path(state.get('blender','blender/blender.exe'))
    state['blender_sha256']=digest(blender if blender.is_absolute() else runtime/blender)
    atomic_json(runtime/'studio-runtime.json',state)
    return {'runtime_features':list(records),'reports':sum(len(rows) for rows in records.values()),
            'quality_accepted':False,'full_bundle_accepted':False}


if __name__=='__main__':
    from hi3d_workbench.common import configure_stdio
    configure_stdio();parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(record(json.loads(args.plan.read_text(encoding='utf-8')))),flush=True)
