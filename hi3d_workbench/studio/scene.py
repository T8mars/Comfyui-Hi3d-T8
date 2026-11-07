"""Read mesh instances without discarding transforms or material attributes."""
from __future__ import annotations

from dataclasses import dataclass
import copy

import numpy as np

from .gltf import Document
from .skeleton import node_worlds


@dataclass
class PrimitiveInstance:
    node: int
    mesh: int
    primitive: int
    definition: dict
    positions: np.ndarray
    faces: np.ndarray
    world: np.ndarray

    def world_positions(self):
        return self.positions @ self.world[:3, :3].T + self.world[:3, 3]


def instances(document: Document):
    primitives = {(m, p): (item, positions, faces)
                  for m, p, item, positions, faces in document.primitives()}
    worlds, _ = node_worlds(document.tree)
    meshes = document.tree.get('meshes', [])
    result = []
    for node_index, node in enumerate(document.tree.get('nodes', [])):
        if 'mesh' not in node:
            continue
        if 'EXT_mesh_gpu_instancing' in node.get('extensions', {}):
            raise ValueError('请先展开 GPU 实例，再进行资产加工')
        mesh_index = node['mesh']
        if type(mesh_index) is not int or not 0 <= mesh_index < len(meshes):
            raise ValueError('场景引用不存在的网格')
        for primitive_index in range(len(meshes[mesh_index].get('primitives', []))):
            item, positions, faces = primitives[mesh_index, primitive_index]
            if positions.ndim != 2 or positions.shape[1] != 3:
                raise ValueError('模型顶点必须为三维坐标')
            result.append(PrimitiveInstance(node_index, mesh_index, primitive_index,
                                           item, positions, faces, worlds[node_index]))
    if not result:
        raise ValueError('场景没有可加工的网格实例')
    return result


def morphed_positions(document: Document, row: PrimitiveInstance):
    """Evaluate default morphs in mesh space, before transforms and skinning.

    Node weights override mesh defaults. Missing weights mean zero, and glTF
    permits negative weights and weights greater than one.
    """
    targets = row.definition.get('targets', [])
    mesh = document.tree['meshes'][row.mesh]
    weights = document.tree['nodes'][row.node].get('weights', mesh.get('weights'))
    weights = np.zeros(len(targets)) if weights is None else np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(targets),) or not np.isfinite(weights).all():
        raise ValueError('形变目标权重数量或数值无效')
    if not np.any(weights):
        return row.positions
    result = row.positions.astype(np.float64, copy=True)
    for weight, target in zip(weights, targets):
        if weight and 'POSITION' in target:
            delta = document.accessor(target['POSITION'])
            if delta.shape != result.shape or not np.isfinite(delta).all():
                raise ValueError('形变目标顶点数量或坐标无效')
            result += weight * delta
    return result


def combined(document: Document, *, apply_morphs=False):
    """Face order is exactly the concatenation of the returned instances."""
    rows = instances(document)
    vertices, faces = [], []
    offset = 0
    for row in rows:
        positions = morphed_positions(document, row) if apply_morphs else row.positions
        vertices.append(positions @ row.world[:3, :3].T + row.world[:3, 3])
        triangles = row.faces[:, [0, 2, 1]] if np.linalg.det(row.world[:3, :3]) < 0 else row.faces
        faces.append(triangles + offset)
        offset += len(row.positions)
    return np.concatenate(vertices), np.concatenate(faces), rows


def make_meshes_unique(document: Document):
    """Give each node its own mesh definition before assigning a scene budget.

    Accessors stay shared until an edit creates replacement data. Node targets,
    skeletons and morph animations remain attached to their original nodes.
    """
    instances(document)  # Validate every referenced mesh before changing any.
    original = document.tree.get('meshes', [])
    meshes = []
    for node in document.tree.get('nodes', []):
        if 'mesh' in node:
            meshes.append(copy.deepcopy(original[node['mesh']]))
            node['mesh'] = len(meshes) - 1
    document.tree['meshes'] = meshes
