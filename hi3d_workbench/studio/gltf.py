"""Embedded glTF editing that keeps materials, skins and animation data intact."""
from __future__ import annotations

import copy
import json
import os
import struct
from pathlib import Path

import numpy as np

DTYPES = {5120: 'i1', 5121: 'u1', 5122: '<i2', 5123: '<u2', 5125: '<u4', 5126: '<f4'}
WIDTHS = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


class Document:
    def __init__(self, tree, binary=b''):
        self.tree = copy.deepcopy(tree)
        self.binary = bytearray(binary)

    @classmethod
    def load(cls, path, *, allow_compressed_geometry=False):
        raw = Path(path).read_bytes()
        if len(raw) < 20 or struct.unpack_from('<4sII', raw) != (b'glTF', 2, len(raw)):
            raise ValueError('需要完整的 GLB 2 文件')
        tree = None
        binary = b''
        offset = 12
        while offset < len(raw):
            if offset + 8 > len(raw):
                raise ValueError('GLB 分块头不完整')
            length, kind = struct.unpack_from('<I4s', raw, offset)
            offset += 8
            if length % 4 or offset + length > len(raw):
                raise ValueError('GLB 分块长度无效')
            payload = raw[offset:offset + length]
            if kind == b'JSON':
                if tree is not None:
                    raise ValueError('GLB 包含重复场景描述')
                tree = json.loads(payload.decode('utf-8').rstrip('\x00 \r\n\t'))
            elif kind == b'BIN\x00':
                if binary:
                    raise ValueError('GLB 包含多个二进制块')
                binary = payload
            offset += length
        if not isinstance(tree, dict) or tree.get('asset', {}).get('version') != '2.0':
            raise ValueError('GLB 场景描述无效')
        if any(item.get('uri') for item in tree.get('buffers', [])):
            raise ValueError('请上传内嵌缓冲区的 GLB')
        if any('uri' in item for item in tree.get('images', [])):
            raise ValueError('请将贴图内嵌到 GLB 后上传')
        if len(tree.get('buffers', [])) > 1:
            raise ValueError('GLB 只能包含一个内嵌缓冲区')
        required = set(tree.get('extensionsRequired', []))
        if not allow_compressed_geometry and required & {'KHR_draco_mesh_compression', 'EXT_meshopt_compression'}:
            raise ValueError('请先解压 Draco / meshopt 压缩网格，再上传加工')
        doc = cls(tree, binary)
        for index in range(len(tree.get('bufferViews', []))):
            doc.view(index)
        return doc

    def view(self, index):
        item = self.tree['bufferViews'][index]
        start, length = item.get('byteOffset', 0), item['byteLength']
        if (item.get('buffer', 0) != 0 or not isinstance(start, int) or not isinstance(length, int)
                or start < 0 or length < 0 or start + length > len(self.binary)):
            raise ValueError('GLB 缓冲区引用无效')
        return memoryview(self.binary)[start:start + length]

    def accessor(self, index, *, normalized=False):
        item = self.tree['accessors'][index]
        if 'sparse' in item:
            raise ValueError('请先将稀疏 glTF 顶点属性展开后再加工')
        try:
            dtype = np.dtype(DTYPES[item['componentType']])
            width = WIDTHS[item['type']]
            count = item['count']
        except (KeyError, TypeError) as exc:
            raise ValueError('GLB 顶点属性格式不受支持') from exc
        if not isinstance(count, int) or count < 0:
            raise ValueError('GLB 属性数量无效')
        if 'bufferView' not in item:
            result = np.zeros((count, width), dtype=dtype)
        else:
            view = self.tree['bufferViews'][item['bufferView']]
            data = self.view(item['bufferView'])
            offset = item.get('byteOffset', 0)
            stride = view.get('byteStride', dtype.itemsize * width)
            if (offset < 0 or stride < dtype.itemsize * width or stride % dtype.itemsize
                    or offset + max(0, count - 1) * stride + (dtype.itemsize * width if count else 0) > len(data)):
                raise ValueError('GLB 属性超出缓冲区')
            result = np.ndarray((count, width), dtype=dtype, buffer=data,
                                offset=offset, strides=(stride, dtype.itemsize)).copy()
        if normalized and item.get('normalized') and np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            result = result.astype(np.float32) / info.max
            if info.min < 0:
                result = np.maximum(result, -1)
        if item['type'] == 'SCALAR':
            result = result.reshape(-1)
        if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
            raise ValueError('GLB 包含非有限数值')
        return result

    def add_view(self, payload, target=None):
        self.binary.extend(b'\x00' * (-len(self.binary) % 4))
        start = len(self.binary)
        self.binary.extend(payload)
        item = {'buffer': 0, 'byteOffset': start, 'byteLength': len(payload)}
        if target is not None:
            item['target'] = target
        views = self.tree.setdefault('bufferViews', [])
        views.append(item)
        return len(views) - 1

    def add_accessor(self, values, *, template=None, target=None):
        values = np.ascontiguousarray(values)
        if values.dtype.kind == 'f':
            values = values.astype('<f4')
        component = next((key for key, dtype in DTYPES.items() if values.dtype == np.dtype(dtype)), None)
        if component is None:
            raise ValueError(f'无法写入 glTF 属性类型 {values.dtype}')
        width = values.shape[1] if values.ndim == 2 else 1
        kind = next((key for key, size in WIDTHS.items() if size == width), None)
        if values.ndim not in {1, 2} or kind is None:
            raise ValueError('无法写入 glTF 属性维度')
        item = copy.deepcopy(template or {})
        for key in ('sparse', 'min', 'max', 'byteOffset'):
            item.pop(key, None)
        item.update(bufferView=self.add_view(values.tobytes(), target), componentType=component,
                    count=len(values), type=kind)
        if len(values) and values.dtype.kind == 'f':
            item['min'] = np.min(values.reshape(len(values), -1), axis=0).tolist()
            item['max'] = np.max(values.reshape(len(values), -1), axis=0).tolist()
        accessors = self.tree.setdefault('accessors', [])
        accessors.append(item)
        return len(accessors) - 1

    def primitives(self):
        for mesh_index, mesh in enumerate(self.tree.get('meshes', [])):
            for primitive_index, primitive in enumerate(mesh.get('primitives', [])):
                if primitive.get('mode', 4) != 4 or 'POSITION' not in primitive.get('attributes', {}):
                    raise ValueError('资产加工仅支持三角网格')
                positions = self.accessor(primitive['attributes']['POSITION'])
                indices = (self.accessor(primitive['indices']) if 'indices' in primitive
                           else np.arange(len(positions), dtype=np.uint32))
                if len(indices) % 3 or len(indices) == 0 or indices.min() < 0 or indices.max() >= len(positions):
                    raise ValueError('GLB 三角面索引无效')
                yield mesh_index, primitive_index, primitive, positions, indices.reshape(-1, 3)

    def subset(self, primitive, face_indices):
        """Replace one primitive, retaining every vertex attribute and morph target."""
        positions = self.accessor(primitive['attributes']['POSITION'])
        indices = (self.accessor(primitive['indices']) if 'indices' in primitive
                   else np.arange(len(positions), dtype=np.uint32)).reshape(-1, 3)
        faces = indices[np.asarray(face_indices, dtype=np.int64)]
        if not len(faces):
            raise ValueError('部件没有三角面')
        used, remap = np.unique(faces.reshape(-1), return_inverse=True)
        self.replace_primitive(primitive, used, remap.reshape(-1, 3))

    def replace_primitive(self, primitive, used, faces):
        for attrs in [primitive['attributes'], *primitive.get('targets', [])]:
            for name, index in list(attrs.items()):
                values = self.accessor(index)
                attrs[name] = self.add_accessor(values[used], template=self.tree['accessors'][index], target=34962)
        dtype = np.uint16 if len(used) <= 65535 else np.uint32
        primitive['indices'] = self.add_accessor(np.asarray(faces, dtype=dtype).reshape(-1), target=34963)

    def compact(self):
        """Drop orphaned accessors/views left by edits, preserving embedded images."""
        accessors = set()
        refs = []
        for mesh in self.tree.get('meshes', []):
            for primitive in mesh.get('primitives', []):
                for mapping in [primitive.get('attributes', {}), *primitive.get('targets', [])]:
                    refs.extend((mapping, key) for key in mapping)
                if 'indices' in primitive:
                    refs.append((primitive, 'indices'))
        for skin in self.tree.get('skins', []):
            if 'inverseBindMatrices' in skin:
                refs.append((skin, 'inverseBindMatrices'))
        for animation in self.tree.get('animations', []):
            for sampler in animation.get('samplers', []):
                refs.extend((sampler, key) for key in ('input', 'output'))
        for mapping, key in refs:
            accessors.add(mapping[key])
        # Unknown extensions can contain accessor/view references. Preserve all
        # in that case instead of silently invalidating extension-owned data.
        if self.tree.get('extensionsUsed'):
            return
        accessor_map = {old: new for new, old in enumerate(sorted(accessors))}
        kept = [self.tree['accessors'][old] for old in sorted(accessors)]
        views = {item['bufferView'] for item in kept if 'bufferView' in item}
        views.update(im['bufferView'] for im in self.tree.get('images', []) if 'bufferView' in im)
        view_map = {old: new for new, old in enumerate(sorted(views))}
        payloads = [(old, bytes(self.view(old))) for old in sorted(views)]
        old_views = self.tree.get('bufferViews', [])
        self.binary = bytearray()
        self.tree['bufferViews'] = []
        for old, payload in payloads:
            new = self.add_view(payload, old_views[old].get('target'))
            if 'byteStride' in old_views[old]:
                self.tree['bufferViews'][new]['byteStride'] = old_views[old]['byteStride']
        for mapping, key in refs:
            mapping[key] = accessor_map[mapping[key]]
        for item in kept:
            if 'bufferView' in item:
                item['bufferView'] = view_map[item['bufferView']]
        for im in self.tree.get('images', []):
            if 'bufferView' in im:
                im['bufferView'] = view_map[im['bufferView']]
        self.tree['accessors'] = kept

    def save(self, path, *, compact=True):
        if compact:
            self.compact()
        self.tree['buffers'] = [{'byteLength': len(self.binary)}]
        payload = json.dumps(self.tree, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
        payload += b' ' * (-len(payload) % 4)
        binary = bytes(self.binary) + b'\x00' * (-len(self.binary) % 4)
        size = 12 + 8 + len(payload) + 8 + len(binary)
        raw = struct.pack('<4sII', b'glTF', 2, size)
        raw += struct.pack('<I4s', len(payload), b'JSON') + payload
        raw += struct.pack('<I4s', len(binary), b'BIN\x00') + binary
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        try:
            with temporary.open('wb') as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            Document.load(temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
