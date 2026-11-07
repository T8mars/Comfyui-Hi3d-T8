"""Read saved animated GLBs and measure every frame's actual skinned surface.

This is an acceptance aid, not an animation-quality pass flag. It uses stored
joint tracks and direct glTF linear blend skinning, without invoking retargeting,
neural weights or the production posed-vertex helper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from hi3d_workbench.common import atomic_json,digest
from hi3d_workbench.studio.gltf import Document
from hi3d_workbench.studio.scene import instances,morphed_positions
from hi3d_workbench.studio.skeleton import node_worlds
from hi3d_workbench.studio.animation import sample_glb


def deform(position,indices,weights,palette):
    if (len(position) and np.all(indices[:,0]==indices[0,0])
            and np.all(weights[:,0]==1) and not np.any(weights[:,1:])):
        matrix=palette[indices[0,0]]
        return position@matrix[:3,:3].T+matrix[:3,3]
    result=np.zeros_like(position,dtype=np.float64)
    for column in range(indices.shape[1]):
        active=weights[:,column]>0
        if not active.any():continue
        matrices=palette[indices[active,column]]
        transformed=np.einsum('nij,nj->ni',matrices[:,:3,:3],position[active])+matrices[:,:3,3]
        result[active]+=transformed*weights[active,column,None]
    return result


def edge_lengths(points,faces):
    return np.linalg.norm(points[faces[:,[1,2,0]]]-points[faces],axis=2)


def analyze(source,result,*,floor=None,progress=lambda *args:None):
    source,result=Path(source),Path(result)
    before,after=Document.load(source),Document.load(result)
    old,new=instances(before),instances(after)
    if len(old)!=len(new):raise ValueError('动画改变了原网格图元数量')
    for key in ('materials','textures','samplers'):
        if before.tree.get(key,[])!=after.tree.get(key,[]):raise ValueError('动画改变了原 '+key)
    images=lambda doc:[bytes(doc.view(row['bufferView'])) for row in doc.tree.get('images',[])]
    if images(before)!=images(after):raise ValueError('动画改变了原内嵌贴图')
    motion=sample_glb(after,fps=30)
    node_to_slot={node:index for index,node in enumerate(motion.rig.nodes)}
    rest_worlds,_=node_worlds(before.tree)
    rows=[];minimum=np.full(3,np.inf);maximum=np.full(3,-np.inf)
    for original,row in zip(old,new):
        if original.node!=row.node or not np.array_equal(original.faces,row.faces):
            raise ValueError('动画改变了原网格节点或有向三角面')
        attrs=original.definition['attributes'];target_attrs=row.definition['attributes']
        if set(attrs)!=set(target_attrs):raise ValueError('动画丢失了原顶点属性')
        for name,index in attrs.items():
            if not np.array_equal(before.accessor(index),after.accessor(target_attrs[name])):
                raise ValueError('动画改变了原顶点属性 '+name)
        targets=original.definition.get('targets',[]);new_targets=row.definition.get('targets',[])
        if len(targets)!=len(new_targets):raise ValueError('动画改变了原形变目标')
        for a,b in zip(targets,new_targets):
            if set(a)!=set(b) or any(not np.array_equal(before.accessor(a[key]),after.accessor(b[key])) for key in a):
                raise ValueError('动画改变了原形变属性')
        if not np.array_equal(morphed_positions(before,original),morphed_positions(after,row)):
            raise ValueError('动画改变了默认形变权重')
        skin_index=before.tree['nodes'][original.node].get('skin')
        if skin_index is None:raise ValueError('此检查要求网格已蒙皮')
        skin=before.tree['skins'][skin_index]
        if skin!=after.tree['skins'][after.tree['nodes'][row.node]['skin']]:
            raise ValueError('动画改变了原蒙皮关节或逆绑定矩阵')
        joints=skin['joints']
        if any(node not in node_to_slot for node in joints):raise ValueError('保存的动画缺少网格关节')
        inverse=(before.accessor(skin['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1)
                 if 'inverseBindMatrices' in skin else np.repeat(np.eye(4)[None],len(joints),axis=0))
        if 'inverseBindMatrices' in skin and not np.array_equal(before.accessor(skin['inverseBindMatrices']),after.accessor(skin['inverseBindMatrices'])):
            raise ValueError('动画改变了逆绑定矩阵数值')
        indices=[];weights=[]
        for slot in range(8):
            if f'JOINTS_{slot}' not in attrs:continue
            indices.append(before.accessor(attrs[f'JOINTS_{slot}']))
            weights.append(before.accessor(attrs[f'WEIGHTS_{slot}'],normalized=True))
        if not indices:raise ValueError('保存的模型缺少实际蒙皮属性')
        indices=np.concatenate(indices,axis=1).astype(np.int64)
        weights=np.concatenate(weights,axis=1).astype(np.float64)
        if (np.any(indices<0) or np.any(indices>=len(joints)) or not np.isfinite(weights).all()
                or np.any(weights<0) or not np.allclose(weights.sum(1),1,atol=1e-5)):
            raise ValueError('保存的模型蒙皮权重无效')
        positions=morphed_positions(before,original).astype(np.float64)
        baseline=deform(positions,indices,weights,rest_worlds[joints]@inverse)
        minimum=np.minimum(minimum,baseline.min(0));maximum=np.maximum(maximum,baseline.max(0))
        rows.append({'positions':positions,'indices':indices,'weights':weights,'inverse':inverse,
            'slots':[node_to_slot[node] for node in joints],'faces':row.faces,
            'baseline_edges':edge_lengths(baseline,row.faces),
            'part':str(row.definition.get('extras',{}).get('studio_part','unassigned'))})
    span=float((maximum-minimum).max())
    if not np.isfinite(span) or span<=1e-10:raise ValueError('保存的模型尺度无效')
    floor=float(minimum[1]) if floor is None else float(floor)
    if not np.isfinite(floor):raise ValueError('地面高度无效')
    records=[];part_minimum={row['part']:[] for row in rows}
    for frame,(time,worlds) in enumerate(zip(motion.times,motion.worlds)):
        severe=0;ratio_max=0.;low=np.inf;by_part={};examples=[]
        for index,row in enumerate(rows):
            points=deform(row['positions'],row['indices'],row['weights'],worlds[row['slots']]@row['inverse'])
            if not np.isfinite(points).all():raise ValueError('保存的动画出现非有限坐标')
            current=edge_lengths(points,row['faces']);baseline=row['baseline_edges']
            ratio=current/np.maximum(baseline,span*1e-6)
            bad=(ratio.max(1)>4)&((current-baseline).max(1)>span*.02)
            count=int(bad.sum());severe+=count;ratio_max=max(ratio_max,float(ratio.max(initial=0)))
            low=min(low,float(points[:,1].min()))
            by_part[row['part']]=min(by_part.get(row['part'],np.inf),float(points[:,1].min()))
            if count and len(examples)<8:examples.append({'row':index,'faces':np.flatnonzero(bad)[:8].tolist()})
        for key in part_minimum:part_minimum[key].append(by_part[key])
        records.append({'frame':frame,'time':float(time),'severely_stretched_faces':severe,
            'maximum_edge_ratio':ratio_max,'minimum_y':low,'ground_penetration':max(0.,floor-low),'examples':examples})
        if frame%15==0 or frame+1==len(motion.times):progress(frame+1,len(motion.times))
    return {'status':'analyzed','quality_accepted':False,'source_sha256':digest(source),'output_sha256':digest(result),
        'scope':'every saved clip frame at 30 fps; direct glTF LBS; no browser or collision acceptance',
        'frames':len(records),'faces':sum(len(row['faces']) for row in rows),'span':span,'floor':floor,
        'geometry_attributes_materials_preserved':True,
        'max_severely_stretched_faces':max(row['severely_stretched_faces'] for row in records),
        'max_ground_penetration':max(row['ground_penetration'] for row in records),
        'stretch_threshold':'edge ratio >4 and edge growth >2% of original model span',
        'minimum_y_by_part':part_minimum,'records':records}


if __name__=='__main__':
    from hi3d_workbench.common import configure_stdio
    configure_stdio();parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--floor',type=float)
    args=parser.parse_args()
    report=analyze(args.source,args.result,floor=args.floor,
        progress=lambda current,total:print(f'[mesh validation] {current}/{total} frames',flush=True))
    atomic_json(args.output,report)
    print(json.dumps({key:report[key] for key in ('frames','faces','max_severely_stretched_faces','max_ground_penetration')}),flush=True)
