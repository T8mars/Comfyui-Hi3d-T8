"""Local single-character motion import, retargeting and animated exports."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import atomic_json,read_json
from .animation import Motion,clip_info,sample_glb,source_rig
from .gltf import Document
from .humanoid import identify
from .skin import Rig,extract,validate


def rig_metadata(rig):
    roles=identify(rig)
    return {'bones':[{'index':index,'name':name,'parent':int(rig.parents[index]),
        'position':rig.worlds[index,:3,3].tolist()} for index,name in enumerate(rig.names)],
        'mapping':{role:rig.names[index] for role,index in roles['roles'].items()},
        'missing_roles':roles['missing_roles']}


def fbx_import(path,folder,*,inspect_only=False,clip=0,armature=None):
    from .processes import blender_command,run_stage
    stage=Path(folder)/'studio-blender'/('inspect-motion' if inspect_only else 'import-motion')
    output=stage/'motion.npz';report=stage/'report.json'
    from filelock import FileLock
    Path(folder).mkdir(parents=True,exist_ok=True)
    with FileLock(str(Path(folder)/'.motion-import.lock'),timeout=30):
        try:
            run_stage(blender_command(),{'operation':'import_motion','source':str(Path(path).resolve()),
                'output':str(output),'report':str(report),'inspect_only':inspect_only,'clip':clip,'armature':armature,'fps':30},stage)
        except RuntimeError as exc:
            lines=(stage/'stage.log').read_text(encoding='utf-8',errors='replace').splitlines()
            error=next((line for line in reversed(lines) if line.startswith(('ValueError:','RuntimeError:'))),str(exc))
            raise ValueError('FBX 动作读取失败：'+error[-500:]) from exc
        value=read_json(report)
        arrays=None
        if not inspect_only:
            with np.load(output,allow_pickle=False) as data:
                arrays={key:data[key].copy() for key in ('times','worlds','name')}
    rig=Rig(value['rig']['names'],np.asarray(value['rig']['parents'],dtype=np.int64),np.asarray(value['rig']['rest_worlds']))
    rig.validate()
    if inspect_only:
        return {**rig_metadata(rig),'clips':value['clips'],'armature':value['armature'],
                'armatures':value['armatures'],'source_fps':value['source_fps']}
    result=Motion(rig,arrays['times'],arrays['worlds'],str(arrays['name'])).validate()
    return result


def inspect_motion(path,folder,*,up_axis='Y',armature=None):
    path=Path(path)
    if path.suffix.lower()=='.fbx':
        return fbx_import(path,folder,inspect_only=True,armature=armature)
    if path.suffix.lower()=='.bvh':
        from .bvh import load
        motion=load(path,up_axis=up_axis)
        return {**rig_metadata(motion.rig),'clips':[{'index':0,'name':motion.name,'duration':float(motion.times[-1])}],
                'source_fps':float(1/np.median(np.diff(motion.times))),'armatures':[],'armature':None}
    document=Document.load(path,allow_compressed_geometry=True)
    clips=clip_info(document)
    if not clips or not any(clip['duration']>0 for clip in clips):
        raise ValueError('GLB 没有可用动画片段')
    return {**rig_metadata(source_rig(document)),'clips':clips,'armatures':[],'armature':None}


def load_motion(path,folder,request):
    path=Path(path)
    if not path.is_file():
        raise ValueError('上传的动作文件不存在')
    if path.suffix.lower()=='.fbx':
        return fbx_import(path,folder,clip=request.get('motion_clip',0),armature=request.get('motion_armature'))
    if path.suffix.lower()=='.bvh':
        from .bvh import load
        return load(path,up_axis=request.get('motion_up','Y'))
    if path.suffix.lower()=='.glb':
        document=Document.load(path,allow_compressed_geometry=True)
        motion=sample_glb(document,request.get('motion_clip',0))
        # Compressed meshes may still supply a usable skeleton, but cannot be
        # sampled without decoding. Keep the joint-only detector in that case.
        compressed=any('KHR_draco_mesh_compression' in primitive.get('extensions',{})
            for mesh in document.tree.get('meshes',[]) for primitive in mesh.get('primitives',[]))
        readable=not compressed and all(primitive.get('mode',4)==4 and 'POSITION' in primitive.get('attributes',{})
            for mesh in document.tree.get('meshes',[]) for primitive in mesh.get('primitives',[]))
        if readable and document.tree.get('skins'):
            from .foot_support import soles,surface_contacts
            from .retarget import body_scale
            roles=identify(motion.rig,request.get('source_mapping'))['roles']
            length=body_scale(motion.rig,roles)
            supports=soles(document,motion.rig,roles,length)
            motion.foot_contacts=surface_contacts(supports,motion,length)
            if motion.foot_contacts is not None:
                floor=min(sole.height(motion.rig.worlds) for sole in supports.values())
                clearance=[max(0.,min(sole.height(pose) for sole in supports.values())-floor) for pose in motion.worlds]
                motion.metadata={'foot_contact_source':'imported skinned sole estimate','sole_clearance':clearance}
        return motion
    raise ValueError('动作文件需为 BVH、FBX 或 GLB')


def run(source,target,request,emit):
    if request['action']=='motion':
        from .motion_kimodo import generate
        motion=generate(request,target.parent,emit)
    else:
        emit('asset_motion_import','读取动作骨架、片段与坐标信息',device='cpu')
        motion=load_motion(request['motion_path'],target.parent,request)
    document=Document.load(source)
    from .retarget import apply
    output,details=apply(document,motion,request,emit)
    if motion.metadata and 'engine' in motion.metadata:
        details['retarget_engine']=details['engine']
        details['engine']=motion.metadata['engine']
        details['generation']=motion.metadata
    validate(output,allow_static=True)
    output.save(target)
    # Preserve the source motion for inspection without a model reload.
    motion_data=dict(times=motion.times,worlds=motion.worlds,rest_worlds=motion.rig.worlds,
        parents=motion.rig.parents,names=np.asarray(motion.rig.names),name=np.asarray(motion.name))
    if motion.foot_contacts is not None:motion_data['foot_contacts']=motion.foot_contacts
    temporary=target.parent/'motion.tmp.npz';np.savez_compressed(temporary,**motion_data)
    temporary.replace(target.parent/'motion.npz')
    from .skeleton import inspect
    atomic_json(target.parent/'rig.json',inspect(target))
    from .rigging import export_fbx
    details.update(export_fbx(target,target.parent,emit))
    atomic_json(target.parent/'motion.json',details)
    return details
