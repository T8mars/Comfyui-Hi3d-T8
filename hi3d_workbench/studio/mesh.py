"""Material-preserving simplification using the pinned meshoptimizer backend."""
from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path

import numpy as np

from .gltf import Document
from .scene import make_meshes_unique


def _library(path):
    if not Path(path).is_file():
        raise RuntimeError('网格优化运行环境未安装，请在模型与环境中安装资产加工组件')
    library = ctypes.CDLL(str(path))
    pointer = ctypes.c_void_p
    library.meshopt_simplifyWithAttributes.argtypes = [pointer, pointer, ctypes.c_size_t,
        pointer, ctypes.c_size_t, ctypes.c_size_t, pointer, ctypes.c_size_t,
        pointer, ctypes.c_size_t, pointer, ctypes.c_size_t, ctypes.c_float,
        ctypes.c_uint, ctypes.POINTER(ctypes.c_float)]
    library.meshopt_simplifyWithAttributes.restype = ctypes.c_size_t
    return library


def distribute_budget(counts, target):
    """Allocate one scene budget while retaining each nonempty primitive."""
    if target < len(counts):
        raise ValueError('目标面数小于模型的材质分组数量，请提高面数或合并材质')
    total = sum(counts)
    if target >= total:
        return list(counts)
    result = np.ones(len(counts), dtype=np.int64)
    left = target - len(counts)
    room = np.asarray(counts, dtype=np.int64) - 1
    if left and room.sum():
        exact = room / room.sum() * left
        additions = np.floor(exact).astype(np.int64)
        result += additions
        left -= int(additions.sum())
        order = np.argsort(-(exact - additions), kind='stable')
        for index in order[:left]:
            result[index] += 1
    return result.tolist()


def simplify(source, target, *, target_faces=None, ratio=None, max_error=0.01,
             priority='quality', library_path=None, progress=None):
    from ..common import ROOT
    if priority not in {'quality', 'count'}:
        raise ValueError('无效减面策略')
    if not isinstance(max_error, (int, float)) or not math.isfinite(max_error) or not 0 <= max_error <= 1:
        raise ValueError('最大形状误差应为 0 到 1')
    doc = Document.load(source)
    make_meshes_unique(doc)
    primitives = list(doc.primitives())
    if not primitives:
        raise ValueError('模型没有三角面')
    counts = [len(item[4]) for item in primitives]
    before = sum(counts)
    if ratio is not None:
        if not math.isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError('保留比例应大于 0 且不超过 1')
        if target_faces is not None:
            raise ValueError('目标面数和保留比例请任选一种')
        target_faces = max(len(primitives), math.ceil(before * ratio))
    if isinstance(target_faces, bool) or not isinstance(target_faces, int) or target_faces < 1:
        raise ValueError('请输入有效的目标三角面数')
    budgets = distribute_budget(counts, target_faces)
    library = _library(library_path or ROOT / 'runtime-assets' /
                       ('meshoptimizer.dll' if os.name == 'nt' else 'libmeshoptimizer.so'))
    results = []
    for number, ((mesh_index, primitive_index, primitive, positions, faces), budget) in enumerate(zip(primitives, budgets)):
        indices = np.ascontiguousarray(faces.reshape(-1), dtype=np.uint32)
        vertices = np.ascontiguousarray(positions, dtype=np.float32)
        attr_list, weight_list = [], []
        for name, index in primitive['attributes'].items():
            if name == 'NORMAL' or name.startswith(('TEXCOORD_', 'COLOR_', 'WEIGHTS_')):
                values = doc.accessor(index, normalized=True).astype(np.float32).reshape(len(vertices), -1)
                attr_list.append(values)
                weight_list.extend([10.0 if name.startswith('TEXCOORD_') else 1.0] * values.shape[1])
        attributes = np.ascontiguousarray(np.concatenate(attr_list, axis=1) if attr_list
                                         else np.empty((len(vertices), 0)), dtype=np.float32)
        weights = np.ascontiguousarray(weight_list, dtype=np.float32)
        output = np.empty_like(indices)
        error = ctypes.c_float(0)
        actual = len(indices)
        if budget < len(faces):
            # Default options preserve seams/borders. The count-priority mode
            # lifts only the error bound, with the resulting error reported.
            actual = library.meshopt_simplifyWithAttributes(output.ctypes.data, indices.ctypes.data,
                len(indices), vertices.ctypes.data, len(vertices), vertices.strides[0],
                attributes.ctypes.data if attributes.shape[1] else None,
                attributes.strides[0] if attributes.shape[1] else 0,
                weights.ctypes.data if len(weights) else None, len(weights), None,
                budget * 3, max_error if priority == 'quality' else np.finfo(np.float32).max,
                0, ctypes.byref(error))
            if actual == 0 or actual > len(indices) or actual % 3:
                raise RuntimeError('减面后端未返回有效三角网格')
            selected = output[:actual]
            used, remap = np.unique(selected, return_inverse=True)
            doc.replace_primitive(primitive, used, remap.reshape(-1, 3))
        results.append({'mesh': mesh_index, 'primitive': primitive_index, 'before': len(faces),
                        'target': budget, 'after': actual // 3, 'normalized_error': float(error.value)})
        if progress:
            progress(number + 1, len(primitives))
    doc.save(target)
    after = sum(row['after'] for row in results)
    return {'engine': 'meshoptimizer', 'input_faces': before, 'target_faces': target_faces,
            'output_faces': after, 'target_reached': after <= target_faces,
            'priority': priority, 'max_error': max_error, 'primitives': results,
            'materials_preserved': True, 'vertex_attributes_preserved': True,
            'message': '已达到目标面数' if after <= target_faces else '为保护形状与接缝，实际面数高于目标；可提高误差或选择面数优先'}
