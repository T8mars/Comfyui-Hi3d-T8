"""Material-preserving glTF skeleton binding and rest-pose editing."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .gltf import Document
from .scene import instances, make_meshes_unique
from .skeleton import node_worlds


@dataclass
class Rig:
    names: list[str]
    parents: np.ndarray
    worlds: np.ndarray
    nodes: list[int] | None = None

    def validate(self):
        count = len(self.names)
        self.parents = np.asarray(self.parents)
        self.worlds = np.asarray(self.worlds, dtype=np.float64)
        if not 1 <= count <= 512 or self.parents.shape != (count,) or self.parents.dtype.kind not in 'iu':
            raise ValueError('角色骨骼数量或父级数据无效')
        if (len(set(self.names)) != count or any(not isinstance(name, str) or not name or len(name) > 200 for name in self.names)
                or self.worlds.shape != (count, 4, 4) or not np.isfinite(self.worlds).all()
                or not np.allclose(self.worlds[:, 3], [0, 0, 0, 1])
                or np.any(np.abs(np.linalg.det(self.worlds[:, :3, :3])) < 1e-12)):
            raise ValueError('骨骼名称或静止姿态矩阵无效')
        if self.nodes is not None and (len(self.nodes) != count or len(set(self.nodes)) != count
                or any(type(node) is not int or node < 0 for node in self.nodes)):
            raise ValueError('骨骼节点列表数量或索引无效')
        state = np.zeros(count, dtype=np.uint8)
        order = []
        def visit(index):
            if state[index] == 1:
                raise ValueError('骨架父级存在循环')
            if state[index] == 2:
                return
            state[index] = 1
            parent = int(self.parents[index])
            if parent != -1:
                if not 0 <= parent < count or parent == index:
                    raise ValueError('骨架父级超出范围')
                visit(parent)
            state[index] = 2
            order.append(index)
        for index in range(count):
            visit(index)
        return order


def extract(document):
    tree = document.tree
    worlds, node_parents = node_worlds(tree)
    joints = []
    for skin in tree.get('skins', []):
        values = skin.get('joints', [])
        if not isinstance(values, list) or len(values) != len(set(values)):
            raise ValueError('骨架关节列表无效')
        for value in values:
            if type(value) is not int or not 0 <= value < len(worlds):
                raise ValueError('骨架引用不存在的节点')
            if value not in joints:
                joints.append(value)
    if not joints:
        raise ValueError('模型没有骨架，请先自动绑定')
    parents, names = [], []
    for index, node in enumerate(joints):
        parent = int(node_parents[node])
        while parent != -1 and parent not in joints:
            parent = int(node_parents[parent])
        parents.append(joints.index(parent) if parent != -1 else -1)
        name = tree['nodes'][node].get('name', f'bone_{index}')
        name = str(name)[:180] or f'bone_{index}'
        if name in names:
            name += f'__{index}'
        names.append(name)
    rig = Rig(names, np.asarray(parents, dtype=np.int64), worlds[joints].copy(), joints)
    rig.validate()
    return rig


def oriented_rig(points, parents, names=None):
    """Use a stable bone frame: local Y follows the child and Z faces forward."""
    points = np.asarray(points, dtype=np.float64)
    parents = np.asarray(parents, dtype=np.int64)
    count = len(points)
    if points.shape != (count, 3) or not np.isfinite(points).all():
        raise ValueError('预测关节坐标无效')
    worlds = np.repeat(np.eye(4)[None], count, axis=0)
    worlds[:, :3, 3] = points
    rig = Rig(names or [f'bone_{index}' for index in range(count)], parents, worlds)
    order = rig.validate()
    for index in order:
        children = np.flatnonzero(parents == index)
        if len(children):
            vectors = points[children] - points[index]
            # For branched roots prefer the ascending spine; other joints use
            # the longest child, avoiding tiny terminal/decoration bones.
            scores = vectors[:, 1] if parents[index] == -1 else np.linalg.norm(vectors, axis=1)
            direction = vectors[int(np.argmax(scores))]
        elif parents[index] != -1:
            direction = points[index] - points[parents[index]]
        else:
            direction = np.array([0., 1., 0.])
        length = np.linalg.norm(direction)
        if length < 1e-10:
            rotation = worlds[parents[index], :3, :3] if parents[index] != -1 else np.eye(3)
        else:
            y = direction / length
            forward = np.array([0., 0., 1.]) if abs(y[2]) < .95 else np.array([1., 0., 0.])
            x = np.cross(y, forward); x /= np.linalg.norm(x)
            z = np.cross(x, y)
            rotation = np.column_stack((x, y, z))
        worlds[index, :3, :3] = rotation
    return rig


def set_local(node, matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('骨骼局部姿态矩阵无效')
    scale = np.linalg.norm(matrix[:3, :3], axis=0)
    if np.any(scale < 1e-12):
        raise ValueError('骨骼局部姿态包含零缩放')
    rotation = matrix[:3, :3] / scale
    if (np.any(scale < 1e-12) or not np.isfinite(matrix).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0):
        raise ValueError('骨骼局部姿态包含不支持的错切或反射')
    for key in ('matrix', 'translation', 'rotation', 'scale'):
        node.pop(key, None)
    node.update(translation=matrix[:3, 3].tolist(), rotation=Rotation.from_matrix(rotation).as_quat().tolist(),
                scale=scale.tolist())


def pack_weights(values, joint_count):
    values = np.asarray(values, dtype=np.float32)
    if (values.ndim != 2 or values.shape[1] != joint_count or not np.isfinite(values).all()
            or np.any(values < -1e-6)):
        raise ValueError('模型返回的蒙皮权重无效')
    values = np.maximum(values, 0)
    sums = values.sum(axis=1)
    if np.any(sums <= 1e-12):
        raise ValueError('模型存在没有有效蒙皮权重的顶点')
    keep = min(4, joint_count)
    indices = np.argpartition(values, joint_count - keep, axis=1)[:, -keep:]
    weights = np.take_along_axis(values, indices, axis=1)
    order = np.argsort(-weights, axis=1, kind='stable')
    indices = np.take_along_axis(indices, order, axis=1)
    weights = np.take_along_axis(weights, order, axis=1)
    retained = weights.sum(axis=1)
    discarded = np.maximum(0, 1 - retained / sums)
    weights /= retained[:, None]
    joints = np.zeros((len(values), 4), dtype=np.uint16)
    packed = np.zeros((len(values), 4), dtype=np.float32)
    joints[:, :keep], packed[:, :keep] = indices, weights
    return joints, packed, discarded


def bind(document, rig, provider, *, use_existing=False):
    """Bind bounded blocks of world-space vertices to a model weight provider."""
    rig.validate()
    output = freeze_current_pose(document)
    make_meshes_unique(output)
    old_animations = len(output.tree.get('animations', []))
    old_worlds, _ = node_worlds(output.tree)
    if use_existing:
        if rig.nodes is None or any(not 0 <= node < len(old_worlds) for node in rig.nodes):
            raise ValueError('已有骨架节点缺失')
        if not np.allclose(old_worlds[rig.nodes], rig.worlds, atol=1e-7):
            raise ValueError('已有骨架的静止姿态发生变化')
        joints = list(rig.nodes)
    else:
        joints = list(range(len(output.tree['nodes']), len(output.tree['nodes']) + len(rig.names)))
        for index, name in enumerate(rig.names):
            parent = int(rig.parents[index])
            local = np.linalg.inv(rig.worlds[parent]) @ rig.worlds[index] if parent != -1 else rig.worlds[index]
            node = {'name': name}
            set_local(node, local)
            children = np.flatnonzero(rig.parents == index)
            if len(children):
                node['children'] = [joints[value] for value in children]
            output.tree['nodes'].append(node)
        roots = [joints[index] for index in np.flatnonzero(rig.parents == -1)]
        if not output.tree.get('scenes'):
            _, parents = node_worlds(output.tree)
            output.tree['scenes'] = [{'nodes': [index for index in range(len(old_worlds)) if parents[index] == -1]}]
            output.tree['scene'] = 0
        for scene in output.tree['scenes']:
            scene.setdefault('nodes', []).extend(roots)
        output.tree.pop('animations', None)
        output.tree.get('extras',{}).pop('hi3d_rigid_attachments',None)
    rows = instances(output)
    output.tree['skins'] = []
    node_skin = {}
    discarded_sum, discarded_max, vertex_count = 0., 0., 0
    for row in rows:
        if row.node not in node_skin:
            matrices = np.linalg.inv(rig.worlds) @ row.world
            inverse_binds = output.add_accessor(matrices.transpose(0, 2, 1).reshape(len(joints), 16))
            node_skin[row.node] = len(output.tree['skins'])
            output.tree['skins'].append({'name': 'Hi3D rig', 'joints': joints, 'inverseBindMatrices': inverse_binds})
        output.tree['nodes'][row.node]['skin'] = node_skin[row.node]
        from .scene import morphed_positions
        positions = morphed_positions(output, row)
        xyz = positions @ row.world[:3, :3].T + row.world[:3, 3]
        joint_values = np.empty((len(xyz), 4), dtype=np.uint16)
        weight_values = np.empty((len(xyz), 4), dtype=np.float32)
        for start in range(0, len(xyz), 2048):
            stop = min(len(xyz), start + 2048)
            values = provider(xyz[start:stop])
            indices, weights, discarded = pack_weights(values, len(joints))
            if len(weights) != stop - start:
                raise ValueError('蒙皮权重与顶点数量不匹配')
            joint_values[start:stop], weight_values[start:stop] = indices, weights
            discarded_sum += float(discarded.sum())
            discarded_max = max(discarded_max, float(discarded.max(initial=0)))
            vertex_count += len(weights)
        attrs = row.definition['attributes']
        for name in list(attrs):
            if name.startswith(('JOINTS_', 'WEIGHTS_')):
                del attrs[name]
        attrs['JOINTS_0'] = output.add_accessor(joint_values, target=34962)
        attrs['WEIGHTS_0'] = output.add_accessor(weight_values, target=34962)
    validate(output)
    return output, {'joints': len(joints), 'influences_per_vertex': 4,
        'mean_discarded_weight': discarded_sum / max(1, vertex_count), 'max_discarded_weight': discarded_max,
        'existing_skeleton': use_existing, 'removed_animation_clips': 0 if use_existing else old_animations}


def freeze_current_pose(document):
    """Rebind the visible initial skin pose without changing the source file.

    Imported rigs need not be at their inverse-bind pose. Baking their current
    linear blend into vertex attributes avoids snapping back when replacing
    weights. Joint nodes, material payloads, morph deltas and animation targets
    remain available; inverse binds are then reset for the current joint pose.
    """
    output = Document(document.tree, document.binary)
    if not output.tree.get('skins'):
        return output
    make_meshes_unique(output)
    worlds, _ = node_worlds(output.tree)
    old_skins = copy.deepcopy(output.tree['skins'])
    output.tree['skins'] = []
    assigned = {}
    for row in instances(output):
        node = output.tree['nodes'][row.node]
        old_index = document.tree['nodes'][row.node].get('skin')
        if old_index is None:
            continue
        if type(old_index) is not int or not 0 <= old_index < len(old_skins):
            raise ValueError('原模型引用无效蒙皮')
        old = old_skins[old_index]
        joints = old['joints']
        ibm = (document.accessor(old['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)
               if 'inverseBindMatrices' in old else np.repeat(np.eye(4)[None], len(joints), axis=0))
        matrices = np.linalg.inv(row.world) @ worlds[joints] @ ibm
        if not np.allclose(matrices, np.eye(4), atol=1e-7):
            attrs = row.definition['attributes']
            groups = [(output.accessor(attrs[f'JOINTS_{slot}']), output.accessor(attrs[f'WEIGHTS_{slot}'], normalized=True))
                      for slot in range(8) if f'JOINTS_{slot}' in attrs and f'WEIGHTS_{slot}' in attrs]
            if not groups:
                raise ValueError('原模型缺少可用蒙皮权重')
            for attributes in [attrs, *row.definition.get('targets', [])]:
                morph = attributes is not attrs
                for name in ('POSITION', 'NORMAL', 'TANGENT'):
                    if name not in attributes:
                        continue
                    source_index = attributes[name]
                    values = output.accessor(source_index)
                    transformed = values.copy()
                    for start in range(0, len(values), 2048):
                        stop = min(len(values), start + 2048)
                        blend = np.zeros((stop - start, 4, 4), dtype=np.float64)
                        total = np.zeros(stop - start)
                        for indices, weights in groups:
                            if np.any(indices >= len(joints)) or np.any(weights < 0):
                                raise ValueError('原模型蒙皮索引或权重无效')
                            for influence in range(4):
                                weight = weights[start:stop, influence]
                                blend += matrices[indices[start:stop, influence]] * weight[:, None, None]
                                total += weight
                        if not np.allclose(total, 1, atol=2e-4):
                            raise ValueError('原模型蒙皮权重总和不是 1')
                        vector = np.einsum('nij,nj->ni', blend[:, :3, :3], values[start:stop, :3])
                        if name == 'POSITION' and not morph:
                            vector += blend[:, :3, 3]
                        elif name in {'NORMAL', 'TANGENT'} and not morph:
                            vector /= np.maximum(np.linalg.norm(vector, axis=1, keepdims=True), 1e-12)
                        transformed[start:stop, :3] = vector
                    attributes[name] = output.add_accessor(transformed, target=34962)
        if row.node not in assigned:
            updated = copy.deepcopy(old)
            inverse = np.linalg.inv(worlds[joints]) @ row.world
            updated['inverseBindMatrices'] = output.add_accessor(inverse.transpose(0, 2, 1).reshape(len(joints), 16))
            assigned[row.node] = len(output.tree['skins'])
            output.tree['skins'].append(updated)
        node['skin'] = assigned[row.node]
    return output


def validate(document, *, allow_static=False):
    worlds, _ = node_worlds(document.tree)
    skins = document.tree.get('skins', [])
    if not skins:
        raise ValueError('结果没有骨架与蒙皮')
    bound = 0
    for row in instances(document):
        index = document.tree['nodes'][row.node].get('skin')
        if index is None and allow_static and not any(key.startswith(('JOINTS_', 'WEIGHTS_')) for key in row.definition['attributes']):
            continue  # Rigid accessories may be parented to animated joints.
        if type(index) is not int or not 0 <= index < len(skins):
            raise ValueError('角色网格未绑定骨架')
        bound += 1
        skin = skins[index]
        joints = skin.get('joints', [])
        if not joints or len(joints) > 512 or len(set(joints)) != len(joints):
            raise ValueError('蒙皮骨骼列表无效')
        if any(type(node) is not int or not 0 <= node < len(worlds) for node in joints):
            raise ValueError('蒙皮引用无效骨骼节点')
        if 'inverseBindMatrices' in skin:
            matrices = document.accessor(skin['inverseBindMatrices'])
            if matrices.shape != (len(joints), 16):
                raise ValueError('逆绑定矩阵数量无效')
        total = np.zeros(len(row.positions), dtype=np.float64)
        groups = 0
        for slot in range(8):
            attrs = row.definition['attributes']
            ji, wi = attrs.get(f'JOINTS_{slot}'), attrs.get(f'WEIGHTS_{slot}')
            if ji is None and wi is None:
                continue
            if ji is None or wi is None:
                raise ValueError('关节索引与蒙皮权重属性不完整')
            indices = document.accessor(ji)
            weights = document.accessor(wi, normalized=True)
            if (indices.shape != (len(total), 4) or indices.dtype.kind != 'u'
                    or weights.shape != indices.shape or np.any(indices >= len(joints))
                    or np.any(weights < 0) or np.any(weights > 1)):
                raise ValueError('关节索引或蒙皮权重超出范围')
            total += weights.sum(axis=1)
            groups += 1
        if not groups or not np.allclose(total, 1, atol=2e-4):
            raise ValueError('蒙皮权重总和不是 1')
    if not bound:
        raise ValueError('角色没有实际绑定骨架的网格')
    return True


def edit(document, changes):
    """Move rest joints while preserving the source mesh's current rest shape."""
    rig = extract(document)
    if not isinstance(changes, list) or len(changes) > 512:
        raise ValueError('关节修正列表无效')
    desired = {}
    for change in changes:
        if not isinstance(change, dict) or type(change.get('index')) is not int or not 0 <= change['index'] < len(rig.names):
            raise ValueError('关节修正编号无效')
        position = np.asarray(change.get('position'), dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('关节修正坐标无效')
        node = rig.nodes[change['index']]
        target = rig.worlds[change['index']].copy()
        target[:3, 3] = position
        desired[node] = target
    output = Document(document.tree, document.binary)
    before, parents = node_worlds(document.tree)
    new_worlds = before.copy()
    visited = set()
    joint_nodes = set(rig.nodes)
    def update(index):
        if index in visited:
            return
        parent = int(parents[index])
        if parent != -1:
            update(parent)
        parent_world = new_worlds[parent] if parent != -1 else np.eye(4)
        if index in desired:
            set_local(output.tree['nodes'][index], np.linalg.inv(parent_world) @ desired[index])
            new_worlds[index] = desired[index]
        elif index in joint_nodes:
            # The editor moves individual joint markers. Keep unedited joint
            # heads fixed rather than silently moving their entire subtree.
            set_local(output.tree['nodes'][index], np.linalg.inv(parent_world) @ before[index])
            new_worlds[index] = before[index]
        else:
            from .skeleton import local_matrix
            new_worlds[index] = parent_world @ local_matrix(output.tree['nodes'][index])
        visited.add(index)
    for index in range(len(before)):
        update(index)
    for skin in output.tree.get('skins', []):
        joints = skin['joints']
        ibm = (document.accessor(skin['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)
               if 'inverseBindMatrices' in skin else np.repeat(np.eye(4)[None], len(joints), axis=0))
        updated = np.linalg.inv(new_worlds[joints]) @ before[joints] @ ibm
        skin['inverseBindMatrices'] = output.add_accessor(updated.transpose(0, 2, 1).reshape(len(joints), 16))
    clips = len(output.tree.pop('animations', []))
    validate(output)
    return output, {'edited_joints': len(desired), 'removed_animation_clips': clips,
        'notice': '已修正静止关节并保持静止网格；旧动画保留在原文件，可重新重定向到新骨架。'}


def posed_vertices(document, *, allow_static=False):
    """Evaluate the initial pose for independent geometry/coordinate checks."""
    if not allow_static:
        validate(document)
    worlds, _ = node_worlds(document.tree)
    output = []
    from .scene import morphed_positions
    for row in instances(document):
        positions = morphed_positions(document, row)
        index = document.tree['nodes'][row.node].get('skin')
        if index is None and allow_static:
            output.append(positions @ row.world[:3, :3].T + row.world[:3, 3])
            continue
        skin = document.tree['skins'][index]
        joints = skin['joints']
        ibm = (document.accessor(skin['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)
               if 'inverseBindMatrices' in skin else np.repeat(np.eye(4)[None], len(joints), axis=0))
        matrices = worlds[joints] @ ibm
        xyz = np.column_stack((positions, np.ones(len(positions))))
        posed = np.zeros((len(xyz), 4), dtype=np.float64)
        attrs = row.definition['attributes']
        for slot in range(8):
            if f'JOINTS_{slot}' not in attrs:
                continue
            indices = document.accessor(attrs[f'JOINTS_{slot}'])
            weights = document.accessor(attrs[f'WEIGHTS_{slot}'], normalized=True)
            for influence in range(4):
                posed += np.einsum('nij,nj->ni', matrices[indices[:, influence]], xyz) * weights[:, influence, None]
        output.append(posed[:, :3])
    return output


def validate_existing_rig(source, result, *, allow_extra_joints=False):
    before,after=extract(source),extract(result)
    count=len(before.names)
    if (len(after.names)<count or (not allow_extra_joints and len(after.names)!=count)
            or before.names!=after.names[:count] or before.nodes!=after.nodes[:count]
            or not np.array_equal(before.parents,after.parents[:count])
            or not np.allclose(before.worlds,after.worlds[:count],atol=1e-7)):
        raise ValueError('已有骨架的节点、层级或静止姿态未保留')
    expected=source.tree.get('animations',[]);actual=result.tree.get('animations',[])
    if len(expected)!=len(actual):
        raise ValueError('已有动画片段未保留')
    channels=0
    for left,right in zip(expected,actual):
        if ({key:value for key,value in left.items() if key!='samplers'}!=
                {key:value for key,value in right.items() if key!='samplers'}
                or len(left.get('samplers',[]))!=len(right.get('samplers',[]))):
            raise ValueError('已有动画通道或片段信息未保留')
        for a,b in zip(left.get('samplers',[]),right.get('samplers',[])):
            if ({key:value for key,value in a.items() if key not in {'input','output'}}!=
                    {key:value for key,value in b.items() if key not in {'input','output'}}
                    or any(not np.array_equal(source.accessor(a[key]),result.accessor(b[key])) for key in ('input','output'))):
                raise ValueError('已有动画时间、插值或关键帧数据未保留')
        channels+=len(left.get('channels',[]))
    return {'existing_skeleton_preserved':True,'animation_clips_preserved':len(expected),'animation_channels_preserved':channels}


def validate_preservation(source, result, *, allow_static=False):
    """Compare visible initial geometry and original PBR payloads after binding."""
    before, after = instances(source), instances(result)
    if len(before) != len(after):
        raise RuntimeError('绑定改变了原模型网格分组数量')
    expected = posed_vertices(source, allow_static=True)
    actual = posed_vertices(result, allow_static=allow_static)
    error, uv_count = 0., 0
    for left, right, old, new in zip(before, after, expected, actual):
        if old.shape != new.shape or not np.array_equal(left.faces, right.faces):
            raise RuntimeError('绑定改变了原模型顶点或三角面归属')
        deviation = float(np.max(np.abs(old - new), initial=0))
        span = float(np.ptp(old, axis=0).max(initial=0))
        if deviation > max(1e-6, span * 2e-6):
            raise RuntimeError(f'绑定改变了模型的初始外形或比例，最大偏移 {deviation:g}')
        error = max(error, deviation)
        for name, accessor in left.definition['attributes'].items():
            if name.startswith('TEXCOORD_'):
                candidate = right.definition['attributes'].get(name)
                if candidate is None or not np.array_equal(source.accessor(accessor), result.accessor(candidate)):
                    raise RuntimeError('绑定改变了原始 UV')
                uv_count += 1
    if any(source.tree.get(key, []) != result.tree.get(key, []) for key in ('materials', 'textures', 'samplers')):
        raise RuntimeError('绑定改变了原 PBR 材质设置')
    old_images, new_images = source.tree.get('images', []), result.tree.get('images', [])
    if len(old_images) != len(new_images) or any(
            bytes(source.view(left['bufferView'])) != bytes(result.view(right['bufferView']))
            for left, right in zip(old_images, new_images)):
        raise RuntimeError('绑定改变了原内嵌贴图数据')
    return {'initial_geometry_preserved': True, 'max_world_position_error': error,
            'uv_attributes_preserved': uv_count, 'pbr_settings_preserved': True,
            'embedded_images_preserved': len(old_images)}
