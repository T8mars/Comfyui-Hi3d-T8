"""Validate user controls and transfer poses into the motion model's rig."""
from __future__ import annotations

import numpy as np

from .animation import Motion,rotations,sample_glb
from .humanoid import identify
from .retarget import apply,between,body_scale,relative
from .skin import Rig,bind
from .gltf import Document


def validate_controls(values,duration,joints):
    if not isinstance(values,list) or len(values)>80:
        raise ValueError('最多支持 80 个动作约束')
    result=[];used=set()
    for item in values:
        if not isinstance(item,dict) or item.get('type') not in {'root','pose'}:
            raise ValueError('动作约束类型无效')
        time=item.get('time')
        if isinstance(time,bool) or not isinstance(time,(int,float)) or not np.isfinite(time) or not 0<=time<=duration:
            raise ValueError('约束时间需在动作时长范围内')
        frame=round(time*30)
        key=(item['type'],frame)
        if key in used:
            raise ValueError('同一帧存在重复的路径点或关键姿态')
        used.add(key)
        value={'type':item['type'],'time':float(time),'frame':frame}
        if item['type']=='root':
            point=np.asarray([item.get('x'),item.get('z')],dtype=float)
            if not np.isfinite(point).all() or np.any(np.abs(point)>1e6):
                raise ValueError('路径点坐标无效')
            value.update(x=float(point[0]),z=float(point[1]))
        else:
            points=np.asarray(item.get('joints'),dtype=float)
            if points.shape!=(joints,3) or not np.isfinite(points).all():
                raise ValueError('关键姿态的关节数量或坐标与当前角色不一致，请重新添加')
            value['joints']=points
        result.append(value)
    return result


def fitted_pose(rig,points):
    """Fit swings to edited joint positions while retaining original lengths."""
    order=rig.validate();points=np.asarray(points,dtype=np.float64)
    rest=relative(rig,rig.worlds);worlds=rig.worlds.copy();rest_rot=rotations(rig.worlds)
    for index in order:
        parent=int(rig.parents[index])
        worlds[index,:3,3]=points[index] if parent==-1 else (worlds[parent] @ np.r_[rest[index,:3,3],1])[:3]
        children=np.flatnonzero(rig.parents==index)
        if len(children):
            vectors=rig.worlds[children,:3,3]-rig.worlds[index,:3,3]
            scores=vectors[:,1] if parent==-1 else np.linalg.norm(vectors,axis=1)
            child=int(children[int(np.argmax(scores))])
            rotation=between(rig.worlds[child,:3,3]-rig.worlds[index,:3,3],points[child]-points[index]) @ rest_rot[index]
        elif parent!=-1:
            rotation=rotations(worlds[parent]) @ rest_rot[parent].T @ rest_rot[index]
        else:
            rotation=rest_rot[index]
        worlds[index,:3,:3]=rotation*np.linalg.norm(rig.worlds[index,:3,:3],axis=0)[None]
    return worlds


def skeleton_document(rig):
    doc=Document({'asset':{'version':'2.0'},'scene':0,'scenes':[{'nodes':[0]}],
        'nodes':[{'mesh':0}],'meshes':[{'primitives':[]}]})
    positions=doc.add_accessor(np.array([[0.,0,0],[1,0,0],[0,1,0]],dtype=np.float32))
    indices=doc.add_accessor(np.array([0,1,2],dtype=np.uint32))
    doc.tree['meshes'][0]['primitives']=[{'attributes':{'POSITION':positions},'indices':indices}]
    def weights(points):
        result=np.zeros((len(points),len(rig.names)),dtype=np.float32);result[:,0]=1
        return result
    return bind(doc,rig,weights)[0]


def transfer_controls(controls,source,target,options):
    """Return model-space paths and full poses, without importing Torch."""
    source_roles=identify(source,options.get('joint_mapping'))
    target_roles=identify(target)
    if 'hips' not in source_roles['roles'] or 'hips' not in target_roles['roles']:
        raise ValueError('无法识别角色骨盆，请先修正骨骼映射')
    alignment=target_roles['basis'] @ source_roles['basis'].T
    ratio=1/options['motion_scale'] if options.get('motion_scale') else body_scale(target,target_roles['roles'])/body_scale(source,source_roles['roles'])
    source_root=source.worlds[source_roles['roles']['hips'],:3,3].copy();source_root[1]=0
    target_root=target.worlds[target_roles['roles']['hips'],:3,3].copy();target_root[1]=0
    paths={item['frame']:(alignment @ (np.array([item['x'],0.,item['z']])-source_root)*ratio+target_root)[[0,2]]
           for item in controls if item['type']=='root'}
    doc=skeleton_document(target) if any(item['type']=='pose' for item in controls) else None
    poses=[]
    for item in controls:
        if item['type']!='pose':continue
        motion=Motion(source,np.array([0.,1/30]),np.array([source.worlds,fitted_pose(source,item['joints'])]),'key pose').validate()
        output,_=apply(doc,motion,{'source_mapping':options.get('joint_mapping'),
            'motion_scale':ratio,'foot_lock':False,'rest_pose':'auto'})
        worlds=sample_glb(output).worlds[-1]
        if item['frame'] in paths:
            root=target_roles['roles']['hips'];delta=paths[item['frame']]-worlds[root,[0,2],3]
            worlds[:,0,3]+=delta[0];worlds[:,2,3]+=delta[1]
        poses.append({'frame':item['frame'],'worlds':worlds})
    return paths,poses
