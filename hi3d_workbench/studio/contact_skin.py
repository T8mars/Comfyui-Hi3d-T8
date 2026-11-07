"""Separate suspected arm/leg contacts using existing neural skin weights.

No surface is deleted or regenerated. Anatomical shoulder/hip blends remain;
distal limb/body and incompatible limb weights can be separated. The source rig
is retained, and an exact face map supports independent preservation QA.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import numpy as np

from ..common import atomic_json, digest
from .gltf import Document
from .humanoid import identify
from .scene import instances, make_meshes_unique
from .skin import extract, posed_vertices, validate, validate_existing_rig


GROUP_NAMES=('body','left_arm','right_arm','left_leg','right_leg')


def verify_saved_partition(source, target, directory):
    """Re-read the saved GLBs and hash-bound face map; no trusted pass flags."""
    source, target, directory = map(Path, (source, target, directory))
    report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
    if (digest(source) != report.get('input_sha256')
            or digest(target) != report.get('output_sha256')):
        raise ValueError('接触分离校验的模型哈希不匹配')
    if type(report.get('changed')) is not bool:
        raise ValueError('接触分离校验记录缺少明确的修改状态')
    if not report['changed']:
        if digest(source) != digest(target):
            raise ValueError('未修正的接触分离结果不是原文件副本')
        validate(Document.load(target))
        return {'byte_identical': True}
    mapping = directory / 'face-map.npz'
    if digest(mapping) != report.get('face_map_sha256'):
        raise ValueError('接触分离校验的面映射哈希不匹配')
    with np.load(mapping, allow_pickle=False) as saved:
        rows = saved['source_rows']
        if rows.ndim != 1 or rows.dtype.kind not in 'iu':
            raise ValueError('接触分离校验的原图元索引无效')
        plans = [(int(row), saved[f'faces_{i}']) for i, row in enumerate(rows)]
    return validate_partition(Document.load(source), Document.load(target), plans)


def limb_groups(rig):
    roles=identify(rig)['roles'];groups=np.zeros(len(rig.names),dtype=np.int32)
    roots=[roles.get(side+'_shoulder',roles.get(side+'_upper_arm')) for side in ('left','right')]
    roots += [roles.get(side+'_thigh') for side in ('left','right')]
    if any(root is None for root in roots) or len(set(roots))!=4:
        return None
    for group,root in enumerate(roots,1):
        nodes=[root]
        for node in nodes:nodes.extend(np.flatnonzero(rig.parents==node).tolist())
        if np.any(groups[nodes]):return None
        groups[nodes]=group
    return groups


def attachment_strength(rig,points):
    """Protect attachment capsules, then smoothly separate distal contacts."""
    roles=identify(rig)['roles'];positions=rig.worlds[:,:3,3]
    span=max(float(np.ptp(positions,axis=0).max()),1e-8)
    strengths=np.zeros((len(points),5),dtype=np.float64)
    for group,(side,root,child) in enumerate([
            ('left','upper_arm','forearm'),('right','upper_arm','forearm'),
            ('left','thigh','shin'),('right','thigh','shin')],1):
        joint=roles.get(side+'_'+root)
        if joint is None:continue
        parent=int(rig.parents[joint]);a=positions[joint]
        b=positions[parent] if parent>=0 else a
        endpoint=roles.get(side+'_'+child)
        length=np.linalg.norm(positions[endpoint]-a) if endpoint is not None else span*.25
        radius=max(float(length)*.35,span*.01)
        edge=b-a;denominator=float(edge@edge)
        u=np.clip((points-a)@edge/max(denominator,1e-16),0,1)
        distance=np.linalg.norm(points-(a+u[:,None]*edge),axis=1)
        value=np.clip((distance-radius)/radius,0,1)
        strengths[:,group]=value*value*(3-2*value)
    return strengths


def pair_strength(strengths,left,right):
    if (left in (1,2)) != (right in (1,2)):
        return np.ones(len(strengths))  # An arm/leg blend is never a joint blend.
    return np.minimum(strengths[:,left],strengths[:,right])


def contact_mask(mass,strengths,minimum):
    result=np.zeros(len(mass),dtype=bool)
    for group in range(1,5):
        result|=(mass[:,0]>minimum)&(mass[:,group]>minimum)&(strengths[:,group]>1e-6)
        for other in range(group+1,5):
            result|=(mass[:,group]>minimum)&(mass[:,other]>minimum)&(pair_strength(strengths,group,other)>1e-6)
    return result


def separated_weights(weights,groups,owner,strengths,minimum):
    mass=np.stack([(weights*(groups==k)).sum(1) for k in range(5)],1)
    result=weights.copy()
    if owner==0:
        for limb in range(1,5):
            active=(mass[:,0]>minimum)&(mass[:,limb]>minimum)
            result*=np.where(groups==limb,1-(strengths[:,limb]*active)[:,None],1)
    else:
        active=(mass[:,0]>minimum)&(mass[:,owner]>minimum)
        result*=np.where(groups==0,1-(strengths[:,owner]*active)[:,None],1)
        for other in range(1,5):
            if other==owner:continue
            active=(mass[:,owner]>minimum)&(mass[:,other]>minimum)
            amount=pair_strength(strengths,owner,other)*active
            result*=np.where(groups==other,1-amount[:,None],1)
    if np.any(result.sum(1)<=1e-12):raise RuntimeError('接触修正产生无权重顶点')
    result/=result.sum(1,keepdims=True)
    return result


def validate_partition(before,after,plans):
    """Check every source face, serialized attributes, visible pose and rig."""
    old,new=instances(before),instances(after)
    if len(plans)!=len(new):raise RuntimeError('接触分离的面映射不完整')
    seen=[np.zeros(len(row.faces),dtype=np.uint32) for row in old]
    left,right=posed_vertices(before),posed_vertices(after)
    span=max(float(np.ptp(np.concatenate(left),axis=0).max()),1e-10)
    maximum=0.;uv=0
    for row,xyz,(index,selected) in zip(new,right,plans):
        if type(index) is not int or not 0<=index<len(old):raise RuntimeError('接触分离引用未知原图元')
        source=old[index];selected=np.asarray(selected)
        if (selected.ndim!=1 or selected.dtype.kind not in 'iu' or len(selected)!=len(row.faces)
                or np.any(selected<0) or np.any(selected>=len(source.faces)) or len(np.unique(selected))!=len(selected)):
            raise RuntimeError('接触分离引用重复或越界的原三角面')
        seen[index][selected]+=1
        if row.node!=source.node or row.definition.get('material')!=source.definition.get('material'):
            raise RuntimeError('接触分离改变了网格节点或材质归属')
        def metadata(primitive):
            value = {key: val for key, val in primitive.items()
                     if key not in {'attributes', 'indices', 'targets', 'extras'}}
            extras = {key: val for key, val in primitive.get('extras', {}).items()
                      if key != 'studio_contact_group'}
            return value, extras
        if metadata(source.definition) != metadata(row.definition):
            raise RuntimeError('接触分离改变了原图元元数据或部件归属')
        attrs=source.definition['attributes'];actual=row.definition['attributes']
        if set(attrs)!=set(actual):raise RuntimeError('接触分离丢失了顶点属性')
        source_targets=source.definition.get('targets',[]);targets=row.definition.get('targets',[])
        if len(source_targets)!=len(targets):raise RuntimeError('接触分离丢失了形变目标')
        comparisons=[(attrs,actual),*zip(source_targets,targets)]
        for expected,observed in comparisons:
            if set(expected)!=set(observed):raise RuntimeError('接触分离改变了形变属性')
            for name,accessor in expected.items():
                if name.startswith('WEIGHTS_'):continue
                a=before.accessor(accessor);b=after.accessor(observed[name])
                for start in range(0,len(selected),65536):
                    choice=selected[start:start+65536]
                    if not np.array_equal(a[source.faces[choice]],b[row.faces[start:start+65536]]):
                        raise RuntimeError('接触分离改变了原三角面的 '+name)
                uv+=int(name.startswith('TEXCOORD_'))
        for start in range(0,len(selected),65536):
            expected=left[index][source.faces[selected[start:start+65536]]]
            actual_xyz=xyz[row.faces[start:start+65536]]
            maximum=max(maximum,float(np.abs(expected-actual_xyz).max(initial=0)))
    if any(np.any(values!=1) for values in seen):raise RuntimeError('接触分离遗漏或重复了原三角面')
    if maximum>span*1e-5:raise RuntimeError('接触分离改变了模型初始可见形状')
    for key in ('materials','textures','samplers'):
        if before.tree.get(key,[])!=after.tree.get(key,[]):raise RuntimeError('接触分离改变了 '+key)
    old_skins,new_skins=before.tree.get('skins',[]),after.tree.get('skins',[])
    if len(old_skins)!=len(new_skins):raise RuntimeError('接触分离改变了蒙皮数量')
    for a,b in zip(old_skins,new_skins):
        if ({k:v for k,v in a.items() if k!='inverseBindMatrices'}!=
                {k:v for k,v in b.items() if k!='inverseBindMatrices'}):
            raise RuntimeError('接触分离改变了蒙皮关节或骨架引用')
        if ('inverseBindMatrices' in a)!=('inverseBindMatrices' in b):raise RuntimeError('接触分离改变了绑定矩阵')
        if 'inverseBindMatrices' in a and not np.array_equal(before.accessor(a['inverseBindMatrices']),after.accessor(b['inverseBindMatrices'])):
            raise RuntimeError('接触分离改变了绑定矩阵数据')
    # Saving a GLB compacts/remaps accessor indices. Compare decoded animation
    # samples and rig semantics, rather than the remapped JSON references.
    validate_existing_rig(before,after)
    a,b=extract(before),extract(after)
    if a.names!=b.names or a.nodes!=b.nodes or not np.array_equal(a.parents,b.parents) or not np.allclose(a.worlds,b.worlds,atol=1e-9):
        raise RuntimeError('接触分离改变了原骨架')
    if len(before.tree.get('images',[]))!=len(after.tree.get('images',[])):raise RuntimeError('接触分离丢失了图片')
    for a,b in zip(before.tree.get('images',[]),after.tree.get('images',[])):
        if bytes(before.view(a['bufferView']))!=bytes(after.view(b['bufferView'])):
            raise RuntimeError('接触分离改变了原贴图')
    validate(after)
    return {'all_original_faces_once':True,'initial_geometry_preserved':True,
        'max_world_triangle_error':maximum,'uv_attributes_preserved':uv,
        'pbr_and_images_preserved':True,'skeleton_and_animation_preserved':True}


def release(source,target,directory,emit=lambda *a,**k:None,*,minimum_weight=.05):
    source,target,directory=map(Path,(source,target,directory))
    if source.resolve()==target.resolve():raise ValueError('接触修正需要保留独立的原绑定文件')
    if not .01<=minimum_weight<=.25:raise ValueError('接触权重阈值无效')
    directory.mkdir(parents=True,exist_ok=True)
    original=Document.load(source);validate(original);rig=extract(original);groups=limb_groups(rig)
    report={'engine':'neural limb-contact partition','minimum_weight':minimum_weight,
        'changed':False,'corrected_vertex_copies':0,'input_sha256':digest(source)}
    def unchanged(reason):
        shutil.copy2(source,target);report.update(reason=reason,output_sha256=digest(target))
        atomic_json(directory/'report.json',report)
        return Document.load(target),report
    if groups is None:return unchanged('骨架缺少明确的人形四肢，未进行接触分离')
    output=Document(original.tree,original.binary);make_meshes_unique(output)
    new_meshes={};plans=[];changed=0;rows=instances(output)
    visible=posed_vertices(original)
    for index,row in enumerate(rows):
        emit('asset_rig_contacts','检查四肢与衣物的接触权重',device='cpu',
            progress_current=index,progress_total=len(rows),progress_unit='primitives')
        attrs=row.definition['attributes']
        if any(name.startswith('WEIGHTS_') and name!='WEIGHTS_0' for name in attrs):
            raise ValueError('接触分离需要本工作台生成的四权重蒙皮')
        ji=output.accessor(attrs['JOINTS_0']);w=output.accessor(attrs['WEIGHTS_0'],normalized=True)
        local=output.tree['skins'][output.tree['nodes'][row.node]['skin']]['joints']
        local_groups=np.array([groups[rig.nodes.index(node)] for node in local]);g=local_groups[ji]
        mass=np.stack([(w*(g==k)).sum(1) for k in range(5)],1)
        strengths=attachment_strength(rig,visible[index])
        suspicious=contact_mask(mass,strengths,minimum_weight)
        if not suspicious.any():
            new_meshes.setdefault(row.mesh,[]).append(row.definition)
            plans.append((index,np.arange(len(row.faces),dtype=np.int32)));continue
        labels=mass[row.faces].mean(1).argmax(1)
        for group in np.unique(labels):
            chosen=np.flatnonzero(labels==group).astype(np.int32)
            used=np.unique(row.faces[chosen])
            primitive=copy.deepcopy(row.definition);output.subset(primitive,chosen)
            attributes=primitive['attributes'];j=output.accessor(attributes['JOINTS_0'])
            weights=output.accessor(attributes['WEIGHTS_0'],normalized=True).copy();part=local_groups[j]
            updated=separated_weights(weights,part,int(group),strengths[used],minimum_weight)
            fix=np.any(np.abs(updated-weights)>1e-7,axis=1);weights=updated
            attributes['WEIGHTS_0']=output.add_accessor(weights.astype(np.float32),target=34962)
            primitive.setdefault('extras',{})['studio_contact_group']=GROUP_NAMES[group]
            new_meshes.setdefault(row.mesh,[]).append(primitive);plans.append((index,chosen));changed+=int(fix.sum())
    if not changed:return unchanged('没有检测到需要分离的四肢／衣物混合权重')
    for mesh,primitives in new_meshes.items():output.tree['meshes'][mesh]['primitives']=primitives
    emit('asset_rig_contacts','验证全部原面、UV、贴图、骨架和动画保持',device='cpu')
    output.save(target);saved=Document.load(target)
    preservation=validate_partition(original,saved,plans)
    mapping=directory/'face-map.npz'
    np.savez_compressed(mapping,source_rows=np.array([i for i,_ in plans],dtype=np.int32),
        **{f'faces_{n}':faces for n,(_,faces) in enumerate(plans)})
    report.update(changed=True,corrected_vertex_copies=changed,input_primitives=len(rows),
        output_primitives=len(plans),faces=sum(len(r.faces) for r in rows),preservation=preservation,
        output_sha256=digest(target),face_map_sha256=digest(mapping),
        notice='已按 AI 骨骼归属分离四肢／衣物的疑似粘连接触，并保护肩髋附近混合权重；原三角面与材质保留，分离处可能有开放边界。')
    atomic_json(directory/'report.json',report)
    return saved,report
