"""Join static embedded GLBs while retaining actual PBR texture payloads."""
from __future__ import annotations

import copy

from .gltf import Document


def join_static(sources, target):
    result = Document({'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': []}]})
    sections = ('accessors', 'images', 'samplers', 'textures', 'materials', 'meshes', 'nodes')
    for path, part, name in sources:
        source = Document.load(path)
        tree = source.tree
        if tree.get('skins') or tree.get('animations') or tree.get('cameras'):
            raise ValueError('此合并接口仅用于静态部件')
        offsets = {key: len(result.tree.setdefault(key, [])) for key in sections}
        views = {}
        for index, definition in enumerate(tree.get('bufferViews', [])):
            value = result.add_view(bytes(source.view(index)), definition.get('target'))
            if 'byteStride' in definition:
                result.tree['bufferViews'][value]['byteStride'] = definition['byteStride']
            views[index] = value
        def adjust_textures(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key.lower().endswith('texture') and isinstance(item, dict) and 'index' in item:
                        item['index'] += offsets['textures']
                    adjust_textures(item)
            elif isinstance(value, list):
                for item in value:
                    adjust_textures(item)
        for section in sections:
            for original in tree.get(section, []):
                item = copy.deepcopy(original)
                if section == 'accessors':
                    if 'sparse' in item:
                        raise ValueError('请先展开稀疏部件属性')
                    if 'bufferView' in item:
                        item['bufferView'] = views[item['bufferView']]
                elif section == 'images':
                    item['bufferView'] = views[item['bufferView']]
                elif section == 'textures':
                    if 'source' in item:
                        item['source'] += offsets['images']
                    if 'sampler' in item:
                        item['sampler'] += offsets['samplers']
                    for extension in item.get('extensions', {}).values():
                        if 'source' in extension:
                            extension['source'] += offsets['images']
                elif section == 'materials':
                    adjust_textures(item)
                elif section == 'meshes':
                    item['name'] = name
                    for primitive in item.get('primitives', []):
                        for attributes in [primitive.get('attributes', {}), *primitive.get('targets', [])]:
                            for key in attributes:
                                attributes[key] += offsets['accessors']
                        if 'indices' in primitive:
                            primitive['indices'] += offsets['accessors']
                        if 'material' in primitive:
                            primitive['material'] += offsets['materials']
                        primitive.setdefault('extras', {})['studio_part'] = part
                elif section == 'nodes':
                    if 'skin' in item or 'camera' in item:
                        raise ValueError('静态部件包含非静态节点引用')
                    if 'mesh' in item:
                        item['mesh'] += offsets['meshes']
                        item.setdefault('extras', {})['studio_part'] = part
                    if 'children' in item:
                        item['children'] = [child + offsets['nodes'] for child in item['children']]
                result.tree[section].append(item)
        scene = tree.get('scenes', [])[tree.get('scene', 0)]
        result.tree['scenes'][0]['nodes'].extend(node + offsets['nodes'] for node in scene['nodes'])
        for key in ('extensionsUsed', 'extensionsRequired'):
            if tree.get(key):
                result.tree[key] = sorted(set(result.tree.get(key, [])) | set(tree[key]))
    result.save(target)
