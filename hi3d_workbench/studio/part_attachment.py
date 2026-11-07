"""Explicit rigid part-to-bone correction; never an implicit neural fallback."""
from __future__ import annotations

import shutil
import copy
import numpy as np

from ..common import atomic_json
from .gltf import Document
from .parts import read_parts
from .scene import instances
from . import skin


def add_ground_joints(document, original_rig, choices, names):
    """Add rigid attachment pivots without changing existing joints or shape."""
    from .skeleton import node_worlds
    worlds,_=node_worlds(document.tree)
    rows=instances(document);visible=skin.posed_vertices(document,allow_static=True)
    existing=document.tree.get('extras',{}).get('hi3d_rigid_attachments',[])
    if not isinstance(existing,list) or any(not isinstance(record,dict) for record in existing):
        raise ValueError('附件地面修正记录无效')
    added={};records=[]
    for part,(bone,avoid_ground) in choices.items():
        if not avoid_ground:continue
        if len(original_rig.nodes)+len(added)>=512:raise ValueError('增加附件关节后超过 512 个骨骼')
        candidates=[xyz for row,xyz in zip(rows,visible) if row.definition.get('extras',{}).get('studio_part')==part]
        if not candidates:raise ValueError('所选部件没有可绑定的顶点')
        points=np.concatenate(candidates);parent=original_rig.nodes[bone]
        pivot=points[np.linalg.norm(points-worlds[parent,:3,3],axis=1).argmin()]
        desired=np.eye(4);desired[:3,3]=pivot
        node=len(document.tree['nodes']);name=f'Hi3D_attachment_{part}_{node}'
        definition={'name':name};skin.set_local(definition,np.linalg.inv(worlds[parent])@desired)
        document.tree['nodes'].append(definition)
        document.tree['nodes'][parent].setdefault('children',[]).append(node)
        added[part]=node
        records.append({'part':part,'name':names[part],'node':node,'parent':parent,'avoid_ground':True})
    kept=[record for record in existing if record.get('part') not in choices]+records
    if existing or records:document.tree.setdefault('extras',{})['hi3d_rigid_attachments']=kept
    if not added:return {}
    # Canonical joint order keeps all original joint indices as a prefix even
    # when an imported character has several skins with different local lists.
    joints=list(original_rig.nodes)+list(added.values());lookup={node:i for i,node in enumerate(joints)}
    old=copy.deepcopy(document.tree['skins']);all_worlds,_=node_worlds(document.tree)
    for row in rows:
        index=document.tree['nodes'][row.node].get('skin')
        if index is None:continue
        mapped=np.array([lookup[node] for node in old[index]['joints']],dtype=np.uint16)
        attrs=row.definition['attributes']
        for key,accessor in list(attrs.items()):
            if key.startswith('JOINTS_'):
                attrs[key]=document.add_accessor(mapped[document.accessor(accessor)],target=34962)
        definition=document.tree['skins'][index]
        matrices=np.linalg.inv(all_worlds[joints])@row.world
        definition['joints']=joints
        definition['inverseBindMatrices']=document.add_accessor(matrices.transpose(0,2,1).reshape(len(joints),16))
    return added


def attach(document, names, attachments, emit=lambda *a, **k: None):
    rig = skin.extract(document)
    if not isinstance(attachments, list) or not 1 <= len(attachments) <= 512:
        raise ValueError('请指定 1–512 项部件骨骼绑定')
    chosen = {}
    for item in attachments:
        if not isinstance(item, dict):
            raise ValueError('部件骨骼绑定格式无效')
        part, bone = item.get('part'), item.get('bone')
        if type(part) is not int or part not in names or part in chosen:
            raise ValueError('部件不存在或重复指定')
        if type(bone) is not int or not 0 <= bone < len(rig.names):
            raise ValueError('绑定目标骨骼不存在')
        avoid_ground=item.get('avoid_ground',False)
        if type(avoid_ground) is not bool:raise ValueError('附件地面修正选项无效')
        chosen[part] = (bone,avoid_ground)
    # Imported assets may be posed away from their inverse-bind pose. Freeze
    # that visible pose before changing weights, keeping morphs and animation.
    output = skin.freeze_current_pose(document)
    helpers=add_ground_joints(output,rig,chosen,names)
    expanded=skin.extract(output)
    rows = instances(output)
    changed = {part: 0 for part in chosen}
    for index, row in enumerate(rows):
        part = row.definition.get('extras', {}).get('studio_part')
        if part not in chosen:
            continue
        emit('asset_rig_attachment', '将选定部件固定到关节，保留原面与材质', device='cpu',
             progress_current=index + 1, progress_total=len(rows), progress_unit='primitives')
        node = output.tree['nodes'][row.node]
        if 'skin' not in node:
            matrices = np.linalg.inv(expanded.worlds) @ row.world
            node['skin'] = len(output.tree['skins'])
            output.tree['skins'].append({'name': 'Hi3D rigid part', 'joints': list(expanded.nodes),
                'inverseBindMatrices': output.add_accessor(matrices.transpose(0,2,1).reshape(len(expanded.nodes),16))})
        joints = output.tree['skins'][node['skin']]['joints']
        target = helpers.get(part,rig.nodes[chosen[part][0]])
        if target not in joints:
            raise ValueError('目标骨骼不在该部件使用的蒙皮中，请先统一角色骨架')
        attributes = row.definition['attributes']
        for key in list(attributes):
            if key.startswith(('JOINTS_', 'WEIGHTS_')):
                del attributes[key]
        values = np.zeros((len(row.positions), 4), dtype=np.float32)
        values[:, 0] = 1.
        indices = np.zeros((len(row.positions), 4), dtype=np.uint16)
        indices[:, 0] = joints.index(target)
        attributes['JOINTS_0'] = output.add_accessor(indices, target=34962)
        attributes['WEIGHTS_0'] = output.add_accessor(values, target=34962)
        changed[part] += len(row.faces)
    if any(not faces for faces in changed.values()):
        raise ValueError('所选部件没有可绑定的三角面')
    skin.validate(output, allow_static=True)
    preservation = skin.validate_preservation(document, output, allow_static=True)
    preservation.update(skin.validate_existing_rig(document, output,allow_extra_joints=bool(helpers)))
    return output, {'engine': 'explicit rigid part attachment', 'preservation': preservation,
        'attachments': [{'part': part, 'name': names[part], 'bone': bone,
                         'bone_name': rig.names[bone], 'faces': changed[part],**({'avoid_ground':True,'helper_node':helpers[part]} if avoid else {})}
                        for part, (bone,avoid) in chosen.items()],
        'notice': '选定部件已刚性固定到指定骨骼，不再参与多关节弯曲；原始绑定 GLB 保留，可回退。'}


def run(source, target, request, emit):
    document, _, names = read_parts(source, request['parts_path'])
    output, details = attach(document, names, request.get('part_attachments'), emit)
    shutil.copy2(source, target.parent / 'rig-original.glb')
    output.save(target)
    saved = Document.load(target)
    details['preservation'] = skin.validate_preservation(document, saved, allow_static=True)
    details['preservation'].update(skin.validate_existing_rig(document, saved,allow_extra_joints=any(
        item.get('avoid_ground') for item in request.get('part_attachments',[]))))
    from .skeleton import inspect
    from .rigging import export_fbx
    atomic_json(target.parent / 'rig.json', {**inspect(target), 'details': details})
    details.update(export_fbx(target, target.parent, emit))
    return details
