"""Owned rigging operations, bounded weight transfer and real FBX export."""
from __future__ import annotations

import shutil

import numpy as np
from scipy.spatial import cKDTree

from ..common import atomic_json, read_json
from .gltf import Document
from . import skin


class SampledWeights:
    """The upstream eight-neighbour inverse-distance transfer, in bounded blocks."""

    def __init__(self, points, weights, world_to_model):
        self.points = np.asarray(points, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float32)
        self.matrix = np.asarray(world_to_model, dtype=np.float64)
        if (self.points.ndim != 2 or self.points.shape[1] != 3 or not len(self.points)
                or self.weights.ndim != 2 or len(self.weights) != len(self.points)
                or self.matrix.shape != (4, 4) or not np.isfinite(self.points).all()
                or not np.isfinite(self.weights).all() or np.any(self.weights < 0)
                or not np.isfinite(self.matrix).all()):
            raise ValueError('采样蒙皮数据或坐标变换无效')
        self.tree = cKDTree(self.points)

    def __call__(self, points):
        points = np.asarray(points, dtype=np.float64)
        query = points @ self.matrix[:3, :3].T + self.matrix[:3, 3]
        k = min(8, len(self.points))
        distances, indices = self.tree.query(query, k=k)
        if k == 1:
            return self.weights[indices]
        interpolation = 1 / (distances + 1e-8)
        interpolation /= interpolation.sum(axis=1, keepdims=True)
        return np.einsum('nk,nkj->nj', interpolation, self.weights[indices])


def export_fbx(source, folder, emit):
    from .processes import blender_command, run_stage
    emit('asset_export', '准备导出并重新导入 FBX，验证骨架与网格', device='cpu')
    stage = folder / 'studio-blender' / 'fbx'
    report = stage / 'report.json'
    run_stage(blender_command(), {'operation': 'export_fbx', 'source': str(source),
        'output': str(folder / 'output.fbx'), 'report': str(report)}, stage)
    value = read_json(report)
    if value.get('fbx_roundtrip') is not True:
        raise RuntimeError('FBX 导出未通过重新导入验证')
    return value


def run(source, target, request, emit):
    document = Document.load(source)
    if request['action'] == 'edit_rig':
        emit('asset_rig_edit', '修正静止关节并保持原模型形状', device='cpu')
        output, details = skin.edit(document, request.get('bone_edits', []))
        details['engine'] = 'gltf-rest-pose-edit'
    elif request.get('rig_engine', 'skintokens') == 'skintokens':
        from .rig_skintokens import generate
        output, details = generate(document, request, emit, folder=target.parent)
    else:
        from .rig_unirig import generate
        output, details = generate(document, request, emit, folder=target.parent)
    skin.validate(output)
    output.save(target)
    details['preservation'] = skin.validate_preservation(document, Document.load(target))
    if request['action']=='rig' and request.get('use_skeleton'):
        if details.get('existing_skeleton') is not True:
            raise RuntimeError('已有骨架蒙皮流程未保留原骨架，已停止输出')
        details['preservation'].update(skin.validate_existing_rig(document,Document.load(target)))
    if request['action']=='rig' and request.get('release_contacts',False):
        from .contact_skin import release
        original_binding=target.parent/'rig-original.glb'
        shutil.copy2(target,original_binding)
        output,contact=release(original_binding,target,target.parent/'studio-contact',emit)
        details['contact_release']=contact
        if contact['changed']:
            details.setdefault('warnings',[]).append(contact['notice'])
            details['preservation']['contact_split_triangle_preservation']=True
            details['preservation']['max_world_position_error']+=contact['preservation']['max_world_triangle_error']
        if request.get('use_skeleton'):
            details['preservation'].update(skin.validate_existing_rig(document,output))
    from .skeleton import inspect
    atomic_json(target.parent / 'rig.json', {**inspect(target), 'details': details})
    details.update(export_fbx(target, target.parent, emit))
    return details
