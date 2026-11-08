"""P3-SAM segmentation and reversible, material-preserving part editing.

Segmentation labels always refer to original triangles. Reconstructed X-Part
geometry follows a separate path and never replaces the original segmentation.
"""
from __future__ import annotations

import copy
import json
import os
import zipfile
from pathlib import Path

import numpy as np

from ..common import atomic_copy, atomic_json, digest, read_json
from .gltf import Document
from .scene import combined, instances


def _integer(value, label):
    if type(value) is not int or not 0 <= value < 1_000_000:
        raise ValueError(f'{label}无效')
    return value


def _name(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 80 or any(ord(c) < 32 for c in value):
        raise ValueError('部件名称需为 1–80 个可见字符')
    return value.strip()


def read_parts(source, metadata_path):
    value = read_json(Path(metadata_path))
    if (not isinstance(value, dict) or value.get('schema') != 1
            or value.get('mesh_sha256') != digest(Path(source)) or not isinstance(value.get('parts'), list)):
        raise ValueError('部件记录与当前模型不一致，请重新拆件')
    names = {}
    for part in value['parts']:
        if not isinstance(part, dict):
            raise ValueError('部件记录无效')
        index = _integer(part.get('id'), '部件编号')
        if index in names:
            raise ValueError('部件编号重复')
        names[index] = _name(part.get('name'))
    if not names:
        raise ValueError('没有可编辑的部件')
    document = Document.load(source)
    labels = []
    for row in instances(document):
        part = row.definition.get('extras', {}).get('studio_part')
        if type(part) is not int or part not in names:
            raise ValueError('模型部件归属不完整')
        labels.append(np.full(len(row.faces), part, dtype=np.int32))
    return document, labels, names


def _completion_fields(previous, metadata_path, directory):
    """Keep reconstruction provenance, separately from current part closure."""
    if previous.get('geometry_completed') is not True:return {}
    origin=previous.get('completion_origin')
    if not isinstance(origin,dict):
        origin={key:previous.get(key) for key in ('engine','mesh_sha256','source_sha256')}
    result={'geometry_completed':True,'completion_inherited':True,
            'completion_origin':copy.deepcopy(origin),'original_segmentation_preserved':False}
    if previous.get('original_segmentation')!='original-segmentation.glb':return result
    # The worker's coarse.glb is a copy in the new job. Its sidecar still lives
    # beside the previous result, which owns these immutable original artifacts.
    source=Path(metadata_path).parent;directory=Path(directory)
    original=source/'original-segmentation.glb';sidecar=source/'original-parts.json'
    try:
        metadata=read_json(sidecar,{})
        valid=(original.is_file() and metadata.get('schema')==1
               and metadata.get('mesh_sha256')==digest(original))
    except (OSError,ValueError,AttributeError):valid=False
    if not valid:
        result['original_segmentation_error']='原分割附件缺失或与记录不匹配，未复制错误的引用'
        return result
    for name in ('original-segmentation.glb','original-parts.json','original-parts.zip'):
        path=source/name
        if path.is_file() and path.resolve()!=(directory/name).resolve():
            atomic_copy(path,directory/name,immutable=True)
    result.update(original_segmentation='original-segmentation.glb',original_segmentation_preserved=True)
    return result


def carry_metadata(source, target, metadata_path):
    """Keep editable part identities after binding/motion, with a fresh hash.

    Contact separation may split a primitive; it must not change its part ID or
    total face count. Do not copy a stale sidecar whose hash names another GLB.
    """
    before, labels, names = read_parts(source, metadata_path)
    after = Document.load(target)
    counts = {part: 0 for part in names}
    expected = {part: sum(int(np.count_nonzero(values == part)) for values in labels) for part in names}
    for row in instances(after):
        part = row.definition.get('extras', {}).get('studio_part')
        if type(part) is not int or part not in names:
            raise ValueError('加工后模型丢失了原部件归属')
        counts[part] += len(row.faces)
    if counts != expected:
        raise ValueError('加工改变了原部件的三角面数量')
    previous=read_json(Path(metadata_path));value=copy.deepcopy(previous)
    completion=_completion_fields(previous,metadata_path,Path(target).parent)
    if completion:
        value.pop('original_segmentation',None);value.update(completion)
    value.update(source_sha256=digest(source), mesh_sha256=digest(target),
                 faces=sum(counts.values()), metadata_preserved_after_processing=True)
    atomic_json(Path(target).parent / 'parts.json', value)
    report={'part_metadata_preserved':True,'parts':len(names)}
    if completion:report['original_segmentation_preserved']=completion['original_segmentation_preserved']
    return report


def apply_edits(document, labels, names, edits):
    """Edits apply to source face identities even after an earlier merge."""
    rows = instances(document)
    if len(labels) != len(rows):
        raise ValueError('部件面标签数量无效')
    labels = [np.asarray(value, dtype=np.int32).copy() for value in labels]
    names = dict(names)
    aliases = {key: key for key in names}
    by_primitive = {}
    for index, row in enumerate(rows):
        if labels[index].shape != (len(row.faces),):
            raise ValueError('部件面标签长度无效')
        by_primitive.setdefault((row.mesh, row.primitive), []).append(index)

    def resolve(value):
        value = _integer(value, '部件编号')
        if value not in aliases:
            raise ValueError('修正引用不存在的部件')
        return aliases[value]

    if not isinstance(edits, list) or len(edits) > 500:
        raise ValueError('部件修正列表无效')
    for edit in edits:
        if not isinstance(edit, dict):
            raise ValueError('部件修正无效')
        operation = edit.get('operation')
        if operation == 'create':
            part = _integer(edit.get('part'), '新部件编号')
            if part in aliases or len(names) >= 512:
                raise ValueError('新部件编号重复或部件数量超过 512')
            aliases[part] = part
            names[part] = _name(edit.get('name'))
        elif operation == 'rename':
            names[resolve(edit.get('part'))] = _name(edit.get('name'))
        elif operation == 'merge':
            ids = edit.get('parts')
            if not isinstance(ids, list) or len(ids) < 2:
                raise ValueError('请至少选择两个部件合并')
            selected = {resolve(value) for value in ids}
            target = min(selected)
            for values in labels:
                values[np.isin(values, list(selected))] = target
            for original, current in aliases.items():
                if current in selected:
                    aliases[original] = target
            for part in selected - {target}:
                names.pop(part, None)
            names[target] = _name(edit.get('name', names[target]))
        elif operation == 'faces':
            key = (_integer(edit.get('mesh'), '网格编号'), _integer(edit.get('primitive'), '图元编号'))
            affected = by_primitive.get(key, [])
            # Export gives each instance its own mesh, so UI face picks remain
            # unambiguous. Reject arbitrary instanced input instead of editing
            # several visually unrelated copies at once.
            if len(affected) != 1:
                raise ValueError('面选择引用无效或网格实例不唯一')
            values = edit.get('faces')
            if not isinstance(values, list) or not values or len(values) > 100_000:
                raise ValueError('请选择要修正的三角面')
            index = affected[0]
            if any(type(face) is not int or not 0 <= face < len(labels[index]) for face in values):
                raise ValueError('面选择超出原网格范围')
            labels[index][values] = resolve(edit.get('target_part'))
        else:
            raise ValueError('未知部件修正操作')
    used = set(np.concatenate(labels).tolist())
    if not used <= names.keys():
        raise ValueError('三角面引用未知部件')
    return labels, {key: value for key, value in names.items() if key in used}


def export_scene(document, labels, names, target, *, only=None):
    """Split mesh resources, keeping original nodes as transform/animation parents."""
    rows = instances(document)
    if len(rows) != len(labels):
        raise ValueError('三角面标签不完整')
    output = Document(document.tree, document.binary)
    nodes = output.tree['nodes']
    old_meshes = copy.deepcopy(document.tree['meshes'])
    output.tree['meshes'] = []
    by_node = {}
    for row, values in zip(rows, labels):
        values = np.asarray(values)
        if (values.shape != (len(row.faces),) or values.dtype.kind not in 'iu'
                or not set(values.tolist()) <= names.keys()):
            raise ValueError('三角面归属无效')
        by_node.setdefault(row.node, []).append((row, values))
    total = 0
    children = {}
    for node_index, primitive_rows in by_node.items():
        parent = nodes[node_index]
        skin = parent.pop('skin', None)
        weights = parent.pop('weights', None)
        parent.pop('mesh', None)
        for row, values in primitive_rows:
            for part in sorted(np.unique(values).tolist()):
                if only is not None and part != only:
                    continue
                chosen = np.flatnonzero(values == part)
                primitive = copy.deepcopy(row.definition)
                output.subset(primitive, chosen)
                primitive.setdefault('extras', {})['studio_part'] = part
                mesh = copy.deepcopy(old_meshes[row.mesh])
                mesh['primitives'] = [primitive]
                mesh['name'] = names[part]
                child = {'name': names[part], 'mesh': len(output.tree['meshes']),
                         'extras': {'studio_part': part}}
                if skin is not None:
                    child['skin'] = skin
                if weights is not None:
                    child['weights'] = weights
                output.tree['meshes'].append(mesh)
                parent.setdefault('children', []).append(len(nodes))
                children.setdefault(node_index, []).append(len(nodes))
                nodes.append(child)
                total += len(chosen)
    # Morph animation channels target a mesh node rather than an armature.
    # Fan them out when that mesh becomes several independently named pieces.
    for animation in output.tree.get('animations', []):
        channels = []
        for channel in animation.get('channels', []):
            destination = channel.get('target', {})
            if destination.get('path') == 'weights' and destination.get('node') in children:
                for child in children[destination['node']]:
                    duplicated = copy.deepcopy(channel)
                    duplicated['target']['node'] = child
                    channels.append(duplicated)
            else:
                channels.append(channel)
        animation['channels'] = channels
    if not total:
        raise ValueError('部件没有可导出的三角面')
    output.save(target)
    actual = sum(len(row.faces) for row in instances(Document.load(target)))
    if actual != total:
        raise ValueError('部件导出三角面数量不一致')
    return total


def write_parts(document, labels, names, target, *, engine, source_sha256):
    target = Path(target)
    expected = sum(len(row.faces) for row in instances(document))
    if export_scene(document, labels, names, target) != expected:
        raise ValueError('拆件丢失了原始三角面')
    result = Document.load(target)
    grouped = {part: [] for part in names}
    for row in instances(result):
        part = row.definition['extras']['studio_part']
        grouped[part].append((row.world_positions(), row.faces))
    parts = []
    import trimesh
    for part, rows in grouped.items():
        if not rows:
            continue
        vertices, faces, offset = [], [], 0
        for xyz, triangles in rows:
            vertices.append(xyz); faces.append(triangles.astype(np.int64, copy=False) + offset); offset += len(xyz)
        xyz, triangles = np.concatenate(vertices), np.concatenate(faces)
        # Only this diagnostic copy merges exactly coincident UV seam vertices.
        diagnostic = trimesh.Trimesh(xyz, triangles, process=True)
        boundary = int(np.count_nonzero(np.bincount(diagnostic.edges_unique_inverse) == 1))
        parts.append({'id': part, 'name': names[part], 'faces': len(triangles),
                      'bounds': [xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist()],
                      'boundary_edges': boundary, 'watertight': bool(diagnostic.is_watertight)})
    metadata = {'schema': 1, 'engine': engine, 'source_sha256': source_sha256,
                'mesh_sha256': digest(target), 'parts': parts, 'faces': expected,
                'geometry_completed': False,
                'notice': '保留原三角面、UV 与材质；切口可能开放，尚未生成装配榫或打印连接结构。'}
    atomic_json(target.parent / 'parts.json', metadata)
    directory = target.parent / 'parts'
    directory.mkdir(exist_ok=True)
    files = []
    for part in parts:
        path = directory / f'part_{part["id"]:04d}.glb'
        export_scene(result, [np.full(len(row.faces), row.definition['extras']['studio_part'], dtype=np.int32)
                              for row in instances(result)], names, path, only=part['id'])
        files.append(path)
    temporary = target.parent / 'parts.zip.tmp'
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(target, 'assembly.glb')
            archive.write(target.parent / 'parts.json', 'parts.json')
            for path in files:
                archive.write(path, 'parts/' + path.name)
        os.replace(temporary, target.parent / 'parts.zip')
    finally:
        temporary.unlink(missing_ok=True)
    return metadata


def run(source, target, request, emit):
    source, target = Path(source), Path(target)
    action = request['action']
    if action == 'segment':
        from .parts_models import segment
        document = Document.load(source)
        vertices, faces, rows = combined(document)
        labels = segment(vertices, faces, request, emit, target.parent)
        if (labels.shape != (len(faces),) or labels.dtype.kind not in 'iu' or np.any(labels < 0)):
            raise ValueError('P3-SAM 未给全部原始三角面分配有效部件')
        ids = sorted(np.unique(labels).tolist())
        remap = {value: index for index, value in enumerate(ids)}
        labels = np.asarray([remap[int(value)] for value in labels], dtype=np.int32)
        names = {index: f'部件 {index + 1}' for index in range(len(ids))}
        split = np.cumsum([len(row.faces) for row in rows])[:-1]
        labels = np.split(labels, split)
        engine = 'P3-SAM'
    else:
        document, labels, names = read_parts(source, request['parts_path'])
        labels, names = apply_edits(document, labels, names, request.get('part_edits', []))
        engine = 'original face partition'
    if action == 'complete_parts':
        from .parts_models import complete
        return complete(source, target, document, labels, names, request, emit)
    emit('asset_parts_export', '正在保存带原材质的组合模型与独立部件', device='cpu')
    result = write_parts(document, labels, names, target, engine=engine, source_sha256=digest(source))
    if action=='edit_parts':
        previous=read_json(Path(request['parts_path']))
        completion=_completion_fields(previous,request['parts_path'],target.parent)
        if completion:
            result.update(completion,notice='基于补全成品编辑现有部件；本次未重新生成几何，当前切口与闭合状态以部件报告为准。')
            atomic_json(target.parent/'parts.json',result)
            from .parts_models import zipfile_for_update
            with zipfile_for_update(target.parent/'parts.zip') as archive:
                if completion['original_segmentation_preserved']:
                    archive.write(target.parent/'original-segmentation.glb','original-segmentation.glb')
                    archive.write(target.parent/'original-parts.json','original-parts.json')
    return {'engine': engine, 'parts': len(result['parts']), 'faces': result['faces'],
            'geometry_completed':result['geometry_completed'],'materials_preserved':True,'notice':result['notice'],
            **({'original_segmentation_preserved':result['original_segmentation_preserved']} if 'original_segmentation_preserved' in result else {})}
