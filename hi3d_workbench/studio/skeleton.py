"""Scene transforms and inspectable glTF skeleton metadata."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .gltf import Document


def local_matrix(node):
    if 'matrix' in node:
        matrix = np.asarray(node['matrix'], dtype=np.float64)
        if matrix.size != 16:
            raise ValueError('节点变换矩阵无效')
        matrix = matrix.reshape(4, 4, order='F')
    else:
        matrix = np.eye(4)
        rotation = np.asarray(node.get('rotation', [0, 0, 0, 1]), dtype=float)
        if rotation.shape != (4,) or not np.isfinite(rotation).all() or np.linalg.norm(rotation) < 1e-12:
            raise ValueError('节点旋转无效')
        matrix[:3, :3] = Rotation.from_quat(rotation).as_matrix() @ np.diag(node.get('scale', [1, 1, 1]))
        matrix[:3, 3] = node.get('translation', [0, 0, 0])
    if not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError('节点包含无效仿射变换')
    return matrix


def node_worlds(tree):
    nodes = tree.get('nodes', [])
    parents = np.full(len(nodes), -1, dtype=np.int64)
    for index, node in enumerate(nodes):
        for child in node.get('children', []):
            if not isinstance(child, int) or not 0 <= child < len(nodes) or parents[child] != -1:
                raise ValueError('节点层级索引无效或包含多个父节点')
            parents[child] = index
    matrices = np.empty((len(nodes), 4, 4), dtype=np.float64)
    state = np.zeros(len(nodes), dtype=np.uint8)
    def visit(index):
        if state[index] == 1:
            raise ValueError('节点层级存在循环')
        if state[index] == 2:
            return matrices[index]
        state[index] = 1
        local = local_matrix(nodes[index])
        matrices[index] = visit(int(parents[index])) @ local if parents[index] != -1 else local
        state[index] = 2
        return matrices[index]
    for index in range(len(nodes)):
        visit(index)
    return matrices, parents


def inspect(source):
    from .scene import instances
    document = Document.load(source)
    tree = document.tree
    meshes = instances(document)
    bones = []
    worlds, parents = node_worlds(tree)
    if tree.get('skins'):
        from .skin import extract
        rig = extract(document)
        bones = [{'index': index, 'node': rig.nodes[index], 'name': name,
                  'parent': int(rig.parents[index]), 'position': rig.worlds[index, :3, 3].tolist()}
                 for index, name in enumerate(rig.names)]
    animations = []
    for animation in tree.get('animations', []):
        duration = 0.0
        for sampler in animation.get('samplers', []):
            times = document.accessor(sampler['input'])
            if len(times):
                duration = max(duration, float(times[-1]))
        animations.append({'name': animation.get('name', f'Clip {len(animations) + 1}'), 'duration': duration})
    from .humanoid import identify
    mapping=identify(rig) if bones else {'roles':{}}
    return {'mapping':{role:rig.names[index] for role,index in mapping['roles'].items()} if bones else {},
            'faces': sum(len(row.faces) for row in meshes),
            'vertices': sum(len(row.positions) for row in meshes), 'primitives': len(meshes),
            'materials': len(tree.get('materials', [])), 'bones': bones, 'animations': animations}
