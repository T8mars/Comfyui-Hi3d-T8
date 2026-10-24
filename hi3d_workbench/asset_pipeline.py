"""Material-aware GLB validation/export, independent of the geometry-only bridge."""
from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import numpy as np
import trimesh


def inspect_asset_geometry(path: Path) -> dict:
    """Measure topology without mistaking coincident UV seam vertices for holes.

    Only exactly equal positions are combined in a temporary diagnostic mesh.
    There is no tolerance welding, repair, or modification of the deliverable.
    """
    from .geometry import load_mesh, inspect_mesh
    mesh = load_mesh(path)
    positions, inverse = np.unique(np.asarray(mesh.vertices), axis=0, return_inverse=True)
    topology = trimesh.Trimesh(positions, inverse[np.asarray(mesh.faces)], process=False)
    info = inspect_mesh(topology)
    info.update(vertices=len(mesh.vertices), topology_vertices=len(positions),
                topology_basis='exact duplicate positions combined for diagnosis only; no tolerance or repair',
                warnings=['拓扑统计临时合并完全相同坐标的 UV 接缝顶点；未修改交付网格，也不代表可打印。'])
    return info


def inspect_textured_glb(path: Path) -> dict:
    """Require embedded UV-based base-color and metallic/roughness maps.

    A GLB filename, vertex colors or a material factor alone is not evidence of
    completed AI texturing. This checks the exported deliverable, not an engine
    claim in its report.
    """
    with path.open('rb') as stream:
        magic, version, size = struct.unpack('<4sII', stream.read(12))
        if magic != b'glTF' or version != 2 or size != path.stat().st_size:
            raise ValueError('贴图结果不是完整的 GLB 2 文件')
        length, kind = struct.unpack('<I4s', stream.read(8))
        if kind != b'JSON' or length > size - 20:
            raise ValueError('贴图 GLB 缺少有效场景描述')
        tree = json.loads(stream.read(length).decode('utf-8').rstrip('\x00 \t\n\r'))
    for buffer in tree.get('buffers', []):
        if buffer.get('uri') and not buffer['uri'].startswith('data:'):
            raise ValueError('贴图 GLB 仍依赖外部缓冲区')
    images, textures = tree.get('images', []), tree.get('textures', [])
    for im in images:
        if im.get('uri') and not str(im['uri']).startswith('data:image/'):
            raise ValueError('贴图 GLB 仍依赖外部图片')
        if 'bufferView' not in im and not str(im.get('uri','')).startswith('data:image/'):
            raise ValueError('贴图 GLB 包含未嵌入的图片')
    materials = tree.get('materials', [])
    def mapped(info):
        index = info.get('index') if isinstance(info, dict) else None
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(textures):
            return False
        source = textures[index].get('source')
        if not isinstance(source, int) or not 0 <= source < len(images):
            return False
        im = images[source]
        return 'bufferView' in im or str(im.get('uri', '')).startswith('data:image/')
    count = 0
    for mesh in tree.get('meshes', []):
        for primitive in mesh.get('primitives', []):
            count += 1
            material = primitive.get('material', -1)
            if not isinstance(material, int) or not 0 <= material < len(materials):
                raise ValueError('贴图结果有网格缺少材质')
            pbr = materials[material].get('pbrMetallicRoughness', {})
            if not mapped(pbr.get('baseColorTexture')) or not mapped(pbr.get('metallicRoughnessTexture')):
                raise ValueError('贴图结果缺少嵌入式颜色或金属度/粗糙度贴图')
            for info in (pbr['baseColorTexture'], pbr['metallicRoughnessTexture']):
                if f'TEXCOORD_{info.get("texCoord", 0)}' not in primitive.get('attributes', {}):
                    raise ValueError('贴图结果缺少材质使用的 UV 坐标')
    if not count:
        raise ValueError('贴图结果没有可显示的网格')
    # Decode image bytes and geometry as well as validating JSON references.
    scene = trimesh.load(path, force='scene', process=False, resolver={}, allow_remote=False)
    if not scene.geometry or not np.isfinite(scene.bounds).all():
        raise ValueError('贴图结果几何无效')
    for mesh in scene.geometry.values():
        uv = getattr(mesh.visual, 'uv', None)
        material = getattr(mesh.visual, 'material', None)
        if uv is None or not np.isfinite(uv).all() or len(uv) != len(mesh.vertices):
            raise ValueError('贴图结果 UV 无效')
        for key in ('baseColorTexture', 'metallicRoughnessTexture'):
            image = getattr(material, key, None)
            if image is None:
                raise ValueError('贴图图片无法解码')
            image.load()
    return {'texture_generated': True, 'uv': True, 'embedded': True,
            'images': len(images), 'materials': len(materials), 'primitives': count,
            'channels': ['base_color', 'metallic', 'roughness']}


def export_textured_glb(source: Path, target: Path, height_mm: float | None = None) -> dict:
    info = inspect_textured_glb(source)
    scene = trimesh.load(source, force='scene', process=False, resolver={}, allow_remote=False)
    if height_mm is not None:
        height = float(scene.extents[2])
        if not np.isfinite(height_mm) or not 0 < height_mm <= 100000 or height <= 1e-12:
            raise ValueError('目标高度或 Z 轴尺寸无效')
        scene.apply_scale(height_mm / 1000 / height)
        scene.export(target)
        inspect_textured_glb(target)
    else:
        shutil.copy2(source, target)
    return {'format': '.glb', 'height_axis': 'Z', 'height_mm': height_mm,
            'units': 'm' if height_mm else 'normalized/unknown', 'bounds': scene.bounds.tolist(), **info}
