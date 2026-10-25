"""Progress adapter for CuMesh's GPU clustering plus CPU xatlas UV flow.

Based on JeffreyXiang/CuMesh ``cumesh/cumesh.py::CuMesh.uv_unwrap``
(MIT license). The installed SDK and native kernels are reused unchanged.
Parameterization uses its original defaults; packing explicitly estimates a
single adaptive atlas at the requested density with a four-texel gutter.
No SDK source or process-global callback is modified.
"""
from __future__ import annotations

import threading


def atlas_progress(atlas, notify):
    """Native callbacks can originate on xatlas worker threads."""
    lock = threading.RLock()
    def factory():
        def callback(category, value):
            lowered = ''.join(c for c in str(category).lower() if c.isalpha())
            if 'addmesh' in lowered or 'addingmesh' in lowered:
                phase = 'uv_addmesh'
            elif 'pack' in lowered:
                phase = 'uv_pack'
            elif 'buildoutput' in lowered or 'buildingoutput' in lowered:
                phase = 'uv_assemble'
            else:
                phase = 'uv_charts'
            with lock:
                notify(phase, f'xatlas {category}：{int(value)}%', device='cpu',
                       current=int(value), total=100, unit='percent',
                       native_category=str(category), packing_backend='xatlas')
            return True
        return callback
    atlas._pbar = None
    atlas._create_progress_callback = factory


def unwrap(handler, *, notify, texture_size=2048, padding=4):
    import torch
    import cumesh
    notify('uv_segment', 'CuMesh：清理已明确退化的面并使用 GPU 聚类', device='cuda')
    handler.remove_degenerate_faces()
    handler.compute_charts()
    new_vertices, _ = handler.read()
    num_charts, _, chart_vmap, chart_faces, vertex_offsets, face_offsets = handler.read_atlas_charts()
    num_charts = int(num_charts)
    if not num_charts:
        raise RuntimeError('CuMesh GPU 聚类没有生成有效图块')
    notify('uv_segment', f'CuMesh GPU 聚类完成：{num_charts} 个图块', device='cuda',
           current=num_charts, total=num_charts, unit='charts')
    chart_vertices = new_vertices[chart_vmap].cpu()
    chart_faces = chart_faces.cpu()
    vertex_offsets, face_offsets, chart_vmap = vertex_offsets.cpu(), face_offsets.cpu(), chart_vmap.cpu()
    atlas = cumesh.Atlas()
    atlas_progress(atlas, notify)
    source_maps = []
    notify('uv_extract', f'向 xatlas 添加 {num_charts} 个 GPU 分区', device='cpu',
           current=0, total=num_charts, unit='charts')
    for index in range(num_charts):
        v0, v1 = vertex_offsets[index], vertex_offsets[index+1]
        f0, f1 = face_offsets[index], face_offsets[index+1]
        local_faces = chart_faces[f0:f1]-v0
        source_maps.append(chart_vmap[v0:v1])
        atlas.add_mesh(chart_vertices[v0:v1], local_faces)
        notify('uv_extract', f'准备 UV 分区：{index+1}/{num_charts}', device='cpu',
               current=index+1, total=num_charts, unit='charts')
    notify('uv_charts', 'CPU xatlas：按 GPU 图块进行参数化', device='cpu')
    atlas.compute_charts(verbose=True)
    notify('uv_pack', 'CPU xatlas：统一单页排布 UV 图块', device='cpu', packing_backend='xatlas')
    # Zero density means an estimated density with maxResolution=0 in xatlas:
    # the same page grows rather than silently introducing another atlas page.
    atlas.pack_charts(resolution=int(texture_size), padding=int(padding), texels_per_unit=0., verbose=True)
    mappings, faces, uvs = [], [], []
    cursor = 0
    notify('uv_restore', '恢复 xatlas 原生索引映射', device='cpu')
    for index in range(num_charts):
        mapping, local_faces, uv = atlas.get_mesh(index)
        mappings.append(source_maps[index][mapping])
        faces.append(local_faces+cursor)
        uvs.append(uv)
        cursor += mapping.shape[0]
        notify('uv_restore', f'恢复 UV 图块：{index+1}/{num_charts}', device='cpu',
               current=index+1, total=num_charts, unit='charts')
    mapping = torch.cat(mappings, dim=0)
    return new_vertices.cpu()[mapping], torch.cat(faces, dim=0), torch.cat(uvs, dim=0), mapping
