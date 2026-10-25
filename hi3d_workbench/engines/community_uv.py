"""Original-index adapter for ComfyUI's community GPU UV unwrapper.

Derived from Comfy-Org/ComfyUI ``comfy_extras/nodes_mesh_postprocess.py``
``_uv_unwrap``, revision 7cebca52227c63b4180ceed448cbd7ad22903b6d,
GPL-3.0. The algorithms remain in the integrity-checked upstream mesh3d
modules. This adapter omits automatic vertex welding, which would merge hard
edge normals, and reports actual phase counters instead of a weighted percent.
No upstream source, global progress hook, or module function is modified.
"""
from __future__ import annotations

import math
import os
from pathlib import Path


SOURCE_REVISION = '7cebca52227c63b4180ceed448cbd7ad22903b6d'


class LocalUVRepairError(RuntimeError):
    """A structural UV failure must not silently retry the whole mesh on CPU."""


def _load_pack_api(notify, importer=None):
    """Use the optional Numba path only if its import actually works."""
    import importlib
    import sys
    importer = importer or importlib.import_module
    # Set this before pack imports model_management. The bundled Torch uses
    # cu128, while Comfy's experimental dynamic allocator requires cu130.
    cli = importer('comfy.cli_args')
    cli.args.disable_dynamic_vram = True
    try:
        importer('numba')
    except Exception as error:
        # Upstream handles ImportError only. Broken LLVM/DLL installations can
        # raise OSError or other ordinary exceptions; route those to the exact
        # upstream Torch fallback without editing pinned source files.
        notify('uv_dependency', '可选 Numba 排布不可用，使用 Torch：'+f'{type(error).__name__}: {error}',
               device='cpu', packing_backend='torch', packing_fallback_reason=f'{type(error).__name__}: {error}')
        missing = object()
        previous = sys.modules.get('numba', missing)
        sys.modules['numba'] = None
        try:
            return importer('comfy_extras.mesh3d.uv_unwrap.pack')
        finally:
            if previous is missing:
                sys.modules.pop('numba', None)
            else:
                sys.modules['numba'] = previous
    return importer('comfy_extras.mesh3d.uv_unwrap.pack')


def _packing_runtime(pack_api):
    if bool(getattr(pack_api, '_HAVE_NUMBA_PACK', False)):
        return 'numba', 'cpu', 'cpu'
    import comfy.model_management as mm
    device = mm.get_torch_device()
    return 'torch', getattr(device, 'type', str(device).split(':')[0]), str(device)


def _segment_callback(notify):
    iteration = 0
    def progress(_upstream_done, _upstream_total):
        nonlocal iteration
        iteration += 1
        # Upstream's F-N+n_merge is not a completed-merge count: chart IDs are
        # not compacted every iteration, so that number can even go backwards.
        notify('uv_segment', f'GPU 图块聚类：第 {iteration} 次合并迭代',
               device='cuda', current=iteration, unit='iterations')
    return progress


def _zero_uv_faces(faces, uv):
    import numpy as np
    triangle = np.asarray(uv, dtype=np.float64)[faces]
    a, b = triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0]
    area = np.abs(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])
    return np.flatnonzero((area == 0.) | ~np.isfinite(area))


def _duplicate_uv_faces(faces, uv, *, return_pairs=False):
    """Keep the first UV triangle; return every exact mirrored/reused copy."""
    import numpy as np
    triangle = np.asarray(uv, dtype=np.float64)[faces]
    order = np.lexsort((triangle[:, :, 1], triangle[:, :, 0]), axis=1)
    canonical = np.ascontiguousarray(np.take_along_axis(triangle, order[:, :, None], axis=1)).reshape(-1, 6)
    canonical[canonical == 0.] = 0.
    keys = canonical.view(np.dtype((np.void, 48))).reshape(-1)
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    duplicate = np.ones(len(faces), dtype=bool)
    duplicate[first] = False
    extra = np.flatnonzero(duplicate)
    return (extra, first[inverse[extra]]) if return_pairs else extra


def _same_source_triangle(source_vertices, mapping, faces, duplicate, first):
    """Prove whether UV reuse represents the same exact 3D surface."""
    import numpy as np
    keys = []
    for selected in (duplicate, first):
        triangle = np.asarray(source_vertices, dtype=np.float64)[mapping[faces[selected]]].copy()
        triangle[triangle == 0.] = 0.
        order = np.lexsort((triangle[:, :, 2], triangle[:, :, 1], triangle[:, :, 0]), axis=1)
        values = np.ascontiguousarray(np.take_along_axis(triangle, order[:, :, None], axis=1))
        keys.append(values.reshape(-1, 9).view(np.dtype((np.void, 72))).reshape(-1))
    return keys[0] == keys[1]


def _coincident_source_faces(source_vertices, source_faces):
    """Count exact source geometry occurrences without deleting any surface."""
    import numpy as np
    _, groups = np.unique(np.asarray(source_vertices, dtype=np.float64), axis=0, return_inverse=True)
    triangles = groups[source_faces]
    canonical = np.sort(triangles, axis=1)
    _, inverse, counts = np.unique(canonical, axis=0, return_inverse=True, return_counts=True)
    start = triangles.argmin(axis=1)
    oriented = np.take_along_axis(triangles, (start[:, None]+np.arange(3))%3, axis=1)
    positive = np.bincount(inverse, weights=(oriented[:, 1] < oriented[:, 2]), minlength=len(counts))
    return {'source_coincident_face_occurrences': int((counts-1).sum()),
            'source_double_sided_coincident_pairs': int(np.minimum(positive, counts-positive).sum())}


def _placement_pixels(uvs, offsets, px, py, sw, theta, scale, chart_h):
    """Upstream placement in float64 pixel space, without clipping slivers.

    The upstream final clip(0,1) can collapse an otherwise valid boundary face.
    Keeping pixel coordinates allows the single atlas boundary to grow instead.
    """
    import numpy as np
    chart = np.repeat(np.arange(len(offsets)-1), np.diff(offsets))
    values = np.asarray(uvs, dtype=np.float64)
    cosine, sine = np.cos(theta)[chart], np.sin(theta)[chart]
    u = values[:, 0]*cosine-values[:, 1]*sine
    v = values[:, 0]*sine+values[:, 1]*cosine
    umin, vmin = np.full(len(offsets)-1, np.inf), np.full(len(offsets)-1, np.inf)
    np.minimum.at(umin, chart, u)
    np.minimum.at(vmin, chart, v)
    u, v = (u-umin[chart])*scale[chart], (v-vmin[chart])*scale[chart]
    rotated = np.asarray(sw)[chart].astype(bool)
    return np.column_stack((np.where(rotated, np.asarray(chart_h)[chart]-v, u)+np.asarray(px)[chart],
                            np.where(rotated, u, v)+np.asarray(py)[chart]))


def _normalized_pixels(pixels, side):
    import numpy as np
    result = np.asarray(pixels, dtype=np.float64)/float(side)
    result[:, 1] = 1.-result[:, 1]
    return result.astype(np.float32)


def _save_diagnostics(arrays, metadata, *, prefix='community'):
    """Explicit opt-in diagnostics; never used as a trusted UV cache."""
    directory = os.environ.get('HI3D_UV_DIAGNOSTICS_DIR')
    if not directory:
        return
    import json
    import numpy as np
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination/(prefix+'-raw.npz.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays, metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    os.replace(temporary, destination/(prefix+'-raw.npz'))
    (destination/(prefix+'-raw.json')).write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf8')


def _repair_local_faces(source_vertices, mapping, faces, pixels, side, resolution, padding, notify):
    """Split only collapsed UV faces into guarded islands in an added strip.

    Original vertex coordinates, corner normals and oriented triangles remain
    untouched. Source IDs are copied for the three new UV seam vertices. The
    existing atlas stays in float64 pixel coordinates until the final glTF
    float32 UV conversion, so retries cannot accumulate rounding drift.
    """
    import numpy as np
    mapping = np.asarray(mapping, dtype=np.int64).copy()
    faces = np.asarray(faces, dtype=np.int64).copy()
    pixels = np.asarray(pixels, dtype=np.float64).copy()
    if not np.isfinite(pixels).all():
        raise LocalUVRepairError('UV 排布含非有限坐标，不能用局部孤岛修补')
    minimum = np.minimum(pixels.min(axis=0), 0.)
    pixels -= minimum
    side = max(float(side)-float(minimum.min()), float(pixels.max()), 1.)
    budget = min(4096, max(16, len(faces)//100))
    repaired = set()
    height_adjusted = set()
    original_side = side
    summary = {'repaired_faces': 0, 'repair_budget': budget, 'repair_passes': 0,
               'minimum_height_adjusted_faces': 0, 'source_revision': SOURCE_REVISION,
               'repaired_zero_uv_faces': 0, 'repaired_duplicate_uv_faces': 0}
    repaired_zero, repaired_duplicate = set(), set()
    repaired_coincident, repaired_collisions = set(), set()
    for attempt in range(4):
        result = _normalized_pixels(pixels, side)
        zero = _zero_uv_faces(faces, result)
        duplicate, duplicate_first = _duplicate_uv_faces(faces, result, return_pairs=True)
        bad = np.union1d(zero, duplicate)
        if not len(bad):
            used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
            summary.update(original_atlas_side=original_side, final_atlas_side=side,
                           atlas_linear_scale=original_side/side)
            return mapping[used], inverse.reshape(-1, 3), result[used], summary
        prospective = repaired | set(map(int, bad))
        if len(prospective) > budget or attempt == 3:
            raise LocalUVRepairError(f'UV 局部退化超过安全修复范围：{len(prospective)}/{len(faces)} 面，'
                                     f'上限 {budget}，已拒绝全网 CPU 重算')
        if attempt == 0:
            summary.update(_coincident_source_faces(source_vertices, mapping[faces]))
        original_indices = mapping[faces[bad]]
        triangle = np.asarray(source_vertices, dtype=np.float64)[original_indices]
        edges = np.roll(triangle, -1, axis=1)-triangle
        edge_lengths = np.einsum('nki,nki->nk', edges, edges)
        start = edge_lengths.argmax(axis=1)
        corners = (start[:, None]+np.arange(3))%3
        ordered = np.take_along_axis(triangle, corners[:, :, None], axis=1)
        a, b = ordered[:, 1]-ordered[:, 0], ordered[:, 2]-ordered[:, 0]
        length2 = np.einsum('ni,ni->n', a, a)
        height = np.hypot.reduce(np.cross(a, b), axis=1)/length2
        x = np.einsum('ni,ni->n', a, b)/length2
        if not np.isfinite(height).all() or not np.isfinite(x).all() or np.any(height <= 0.):
            raise LocalUVRepairError('UV 局部修补无法证明原三角面具有非零面积')
        same_geometry = _same_source_triangle(source_vertices, mapping, faces, duplicate, duplicate_first)
        repaired_coincident.update(map(int, duplicate[same_geometry]))
        repaired_collisions.update(map(int, duplicate[~same_geometry]))
        # Dedicated islands have at least two target texels of area before
        # the tiny strip expansion. Very thin triangles may need UV-only
        # height inflation; report its count rather than deleting their faces.
        factor = max(1., side/float(resolution))
        edge_pixels, minimum_height = 8.*factor, .5*factor
        pixel_height = np.maximum(height*edge_pixels, minimum_height)
        height_adjusted.update(map(int, bad[height*edge_pixels < minimum_height]))
        summary['minimum_height_adjusted_faces'] = len(height_adjusted)
        pitch = (8.+2.*padding)*factor
        columns = max(1, int(side//pitch))
        rows = np.arange(len(bad))//columns
        columns_index = np.arange(len(bad))%columns
        strip_y = math.ceil(float(pixels[:, 1].max()))+padding*factor
        base = np.column_stack(((columns_index*pitch+padding*factor),
                                strip_y+rows*pitch+padding*factor))
        new_pixels = np.empty((len(bad), 3, 2), dtype=np.float64)
        new_pixels[:, 0] = base
        new_pixels[:, 1] = base+np.array([edge_pixels, 0.])
        new_pixels[:, 2] = base+np.column_stack((np.clip(x, 0., 1.)*edge_pixels, pixel_height))
        # Restore original corner order after the cyclic longest-edge choice.
        aligned = np.empty_like(new_pixels)
        np.put_along_axis(aligned, corners[:, :, None], new_pixels, axis=1)
        offset = len(mapping)
        mapping = np.concatenate((mapping, original_indices.reshape(-1)))
        pixels = np.vstack((pixels, aligned.reshape(-1, 2)))
        faces[bad] = np.arange(offset, offset+3*len(bad)).reshape(-1, 3)
        side = max(side, math.ceil(float(pixels.max()))+padding*factor)
        repaired = prospective
        repaired_zero.update(map(int, zero))
        repaired_duplicate.update(map(int, duplicate))
        summary.update(repaired_faces=len(repaired), repair_passes=attempt+1,
                       repaired_zero_uv_faces=len(repaired_zero),
                       repaired_duplicate_uv_faces=len(repaired_duplicate),
                       repaired_coincident_uv_faces=len(repaired_coincident),
                       repaired_unrelated_uv_collisions=len(repaired_collisions))
        notify('uv_repair', f'局部重建 {len(bad)} 个退化/复用 UV 面，累计 {len(repaired)}/{budget}，保留原几何',
               device='cpu', current=len(bad), total=len(bad), unit='faces',
               uv_repair=dict(summary), local_uv_repaired_faces=len(repaired), atlas_side=side)
    raise AssertionError('unreachable UV repair state')


def unwrap(vertices, faces, resolution, padding, notify):
    import numpy as np
    import torch
    from .run_stage import source_dir
    source_dir('comfy_core')
    notify('uv_dependency', '加载社区 UV 依赖与可选 LLVM/Numba 运行时', device='cpu')
    from comfy_extras.mesh3d.uv_unwrap import mesh as mesh_api
    from comfy_extras.mesh3d.uv_unwrap import segment as segment_api
    from comfy_extras.mesh3d.uv_unwrap import parameterize as param_api
    pack_api = _load_pack_api(notify)

    if not torch.cuda.is_available():
        raise RuntimeError('社区 GPU UV 展开需要可用 CUDA 设备')
    device = torch.device('cuda')
    # Source identity is kept throughout. In particular, distance=0 in the
    # upstream node means automatic welding, not disabled welding.
    notify('uv_adjacency', '构建原始索引的 UV 邻接图', device='cuda')
    v_in = torch.from_numpy(np.ascontiguousarray(vertices, dtype=np.float32)).to(device)
    f_in = torch.from_numpy(np.ascontiguousarray(faces, dtype=np.int64)).to(device)
    mesh = mesh_api.build_mesh(v_in, f_in)

    notify('uv_segment', 'GPU PEC 图块聚类', device='cuda')
    face_chart = segment_api.cluster_charts_pec(mesh, max_cost=1.0, progress_callback=_segment_callback(notify))
    n_charts = int(face_chart.max().item()) + 1 if face_chart.numel() else 0
    if not n_charts:
        raise RuntimeError('GPU UV 聚类没有生成有效图块')
    areas_cpu = mesh_api.chart_3d_areas(mesh.face_area, face_chart, n_charts).detach().cpu()
    notify('uv_extract', f'提取 {n_charts} 个 UV 图块', device='cpu')
    face_chart_np = face_chart.cpu().numpy()
    faces_np = mesh.faces.cpu().numpy()
    vertices_np = mesh.vertices.cpu().numpy()
    face_face_np = mesh.face_face.cpu().numpy()
    order = np.argsort(face_chart_np, kind='stable')
    chart_counts = np.bincount(face_chart_np, minlength=n_charts)
    chart_offsets = np.zeros(n_charts + 1, dtype=np.int64)
    np.cumsum(chart_counts, out=chart_offsets[1:])
    faces_sorted = faces_np[order]
    chart_sorted = face_chart_np[order]
    n_verts = max(vertices_np.shape[0], 1)
    chart_of_slot = np.repeat(chart_sorted, 3)
    keys, local_flat = np.unique(chart_of_slot * n_verts + faces_sorted.reshape(-1), return_inverse=True)
    used_verts = keys % n_verts
    vert_counts = np.bincount(keys // n_verts, minlength=n_charts)
    vert_offsets = np.zeros(n_charts + 1, dtype=np.int64)
    np.cumsum(vert_counts, out=vert_offsets[1:])
    local_faces = (local_flat - vert_offsets[chart_of_slot]).reshape(-1, 3)
    pos_in_chart = np.empty(order.size, dtype=np.int64)
    pos_in_chart[order] = np.arange(order.size) - chart_offsets[chart_sorted]
    ff_sorted = face_face_np[order]
    ff_safe = np.maximum(ff_sorted, 0)
    keep = (ff_sorted >= 0) & (face_chart_np[ff_safe] == chart_sorted[:, None])
    local_ff = np.where(keep, pos_in_chart[ff_safe], -1)
    chart_of_vert = (keys // n_verts).astype(np.int64)
    verts_concat = vertices_np[used_verts].astype(np.float64)
    global_faces = local_faces + vert_offsets[chart_sorted][:, None]
    face_pos = pos_in_chart[order]
    notify('uv_parameterize', '投影并检查 UV 图块', device='cpu', current=0, total=n_charts, unit='charts')
    uv0 = param_api.ortho_project_concat(verts_concat, chart_of_vert, n_charts)
    rms, mx, n_flip, n_zero = param_api.stretch_metrics_concat(
        verts_concat, uv0, global_faces, chart_sorted, n_charts)
    valid = (vert_counts >= 3) & (chart_counts > 0)
    auto = valid & (chart_counts <= 5)
    flip_ok = (n_flip == 0) | (n_flip == chart_counts)
    candidates = valid & ~auto & flip_ok & (n_zero == 0) & (rms <= 1.5) & (mx <= 2.0)
    ortho_ok = auto.copy()
    checked = int(auto.sum())
    for chart in np.flatnonzero(candidates):
        f0, f1 = chart_offsets[chart:chart+2]
        v0, v1 = vert_offsets[chart:chart+2]
        if not param_api._uv_boundary_self_intersects(uv0[v0:v1], local_faces[f0:f1], local_ff[f0:f1]):
            ortho_ok[chart] = True
        # Rejected candidates still need solving below; count only completed
        # charts, not the separate boundary-check operation.
        if ortho_ok[chart]:
            checked += 1
        notify('uv_parameterize', f'UV 图块投影：{checked}/{n_charts}', device='cpu',
               current=checked, total=n_charts, unit='charts')

    lscm_mask = valid & ~ortho_ok
    batchable = vert_counts <= param_api.LSCM_BATCH_MAX_VERTS
    lscm_ids = np.flatnonzero(lscm_mask & batchable)
    big_ids = np.flatnonzero(lscm_mask & ~batchable)
    notify('uv_parameterize', f'GPU 批量参数化 {len(lscm_ids)} 个图块', device='cuda',
           current=int(ortho_ok.sum()), total=n_charts, unit='charts')
    solved = param_api.lscm_charts_batch(
        verts_concat, uv0, global_faces, face_pos, chart_sorted, chart_of_vert,
        vert_offsets, lscm_ids, n_charts, device=device)
    completed = int(ortho_ok.sum()) + len(lscm_ids)
    notify('uv_parameterize', f'UV 图块参数化：{completed}/{n_charts}', device='cuda',
           current=completed, total=n_charts, unit='charts')
    for chart in big_ids:
        f0, f1 = chart_offsets[chart:chart+2]
        v0, v1 = vert_offsets[chart:chart+2]
        uvs = param_api.lscm_chart(torch.from_numpy(verts_concat[v0:v1]),
                                 torch.from_numpy(local_faces[f0:f1]),
                                 torch.from_numpy(local_ff[f0:f1]), pin_positions=uv0[v0:v1])
        solved[int(chart)] = uvs.detach().cpu().numpy().astype(np.float32)
        completed += 1
        notify('uv_parameterize', f'大型 UV 图块参数化：{completed}/{n_charts}', device='cpu',
               current=completed, total=n_charts, unit='charts')

    uv0_f32 = uv0.astype(np.float32)
    chart_uvs = []
    for chart in range(n_charts):
        v0, v1 = vert_offsets[chart:chart+2]
        value = None if ortho_ok[chart] else solved.get(int(chart))
        if value is not None and np.isfinite(value).all() and value.size:
            bbox = value.max(axis=0) - value.min(axis=0)
            if max(float(bbox.max()), 1e-12) / max(float(bbox.min()), 1e-12) > 100.:
                value = None
        else:
            value = None
        chart_uvs.append(value if value is not None else
                         (uv0_f32[v0:v1] if valid[chart] else np.zeros((v1-v0, 2), dtype=np.float32)))
    uv_all = np.concatenate(chart_uvs)
    ua, ub, uc = uv_all[global_faces[:, 0]], uv_all[global_faces[:, 1]], uv_all[global_faces[:, 2]]
    uv_triangle_area = .5 * np.abs((ub[:, 0]-ua[:, 0])*(uc[:, 1]-ua[:, 1]) -
                                  (uc[:, 0]-ua[:, 0])*(ub[:, 1]-ua[:, 1]))
    uv_area = np.bincount(chart_sorted, weights=uv_triangle_area.astype(np.float64), minlength=n_charts)
    areas_3d = areas_cpu.numpy().astype(np.float64)
    density = math.sqrt(float(resolution or 1024)**2 * .62 / (float(areas_3d.sum()) or 1.))

    packing_backend, packing_device, packing_device_name = _packing_runtime(pack_api)
    def pack_progress(done, total):
        notify('uv_pack', f'排布 UV 图块：{int(done)}/{int(total)}', device=packing_device,
               current=int(done), total=int(total), unit='chart_operations',
               packing_backend=packing_backend, packing_device=packing_device_name)

    notify('uv_pack', f'统一排布 UV 图块（{packing_backend}，{packing_device_name}）', device=packing_device,
           packing_backend=packing_backend, packing_device=packing_device_name)
    px, py, sw, th, sc, chh, width, height = pack_api.pack_bitmap_concat(
        uv_all, vert_offsets, local_faces, chart_offsets, areas_3d, uv_area,
        texels_per_unit=density, padding_texels=padding, progress_callback=pack_progress)
    out_faces = np.zeros_like(faces_np)
    out_faces[order] = global_faces
    pixels = _placement_pixels(uv_all, vert_offsets, px, py, sw, th, sc, chh)
    side = float(max(width, height, 1))
    raw_uv = _normalized_pixels(pixels, side)
    prepack_bad = _zero_uv_faces(global_faces, uv_all)
    raw_bad = _zero_uv_faces(out_faces, raw_uv)
    raw_duplicate = _duplicate_uv_faces(out_faces, raw_uv)
    diagnostics = {'source_revision': SOURCE_REVISION, 'faces': len(out_faces), 'charts': n_charts,
                   'prepack_zero_faces': len(prepack_bad), 'postpack_zero_faces': len(raw_bad),
                   'prepack_bad_face_sample': order[prepack_bad[:32]].tolist(),
                   'postpack_duplicate_faces': len(raw_duplicate),
                   'postpack_duplicate_face_sample': raw_duplicate[:32].tolist(),
                   'packing_backend': packing_backend, 'packing_device': packing_device_name,
                   'postpack_bad_face_sample': raw_bad[:32].tolist(), 'atlas_width': int(width),
                   'atlas_height': int(height), 'unclipped_uv_min': raw_uv.min(axis=0).tolist(),
                   'unclipped_uv_max': raw_uv.max(axis=0).tolist()}
    if os.environ.get('HI3D_UV_DIAGNOSTICS_DIR'):
        clipped = pack_api.apply_placements_concat(uv_all, vert_offsets, px, py, sw, th, sc, chh, width, height)
        clipped = np.asarray(clipped, dtype=np.float32).copy()
        clipped[:, 1] = 1.-clipped[:, 1]
        native_bad = _zero_uv_faces(out_faces, clipped)
        diagnostics['native_clipped_zero_faces'] = len(native_bad)
        diagnostics['native_clipped_bad_face_sample'] = native_bad[:32].tolist()
        _save_diagnostics({'mapping': used_verts, 'faces': out_faces, 'uv': raw_uv,
                           'native_clipped_uv': clipped, 'pixel_uv': pixels, 'source_vertices': vertices,
                           'source_faces': faces, 'uv_before_pack': uv_all, 'face_chart': face_chart_np,
                           'prepack_bad_face_indices': order[prepack_bad],
                           'postpack_bad_face_indices': raw_bad, 'native_clipped_bad_face_indices': native_bad,
                           'postpack_duplicate_face_indices': raw_duplicate}, diagnostics)
    notify('uv_validate', f'UV 退化诊断：排布前 {len(prepack_bad)} 面，F32 排布后 {len(raw_bad)} 面',
           device='cpu', uv_diagnostics=diagnostics)
    out_mapping, out_faces, out_uv, repair = _repair_local_faces(
        vertices, used_verts, out_faces, pixels, side, resolution, padding, notify)
    if os.environ.get('HI3D_UV_DIAGNOSTICS_DIR'):
        import json
        (Path(os.environ['HI3D_UV_DIAGNOSTICS_DIR'])/'community-repair.json').write_text(
            json.dumps({**repair, 'postrepair_zero_faces': len(_zero_uv_faces(out_faces, out_uv))},
                       ensure_ascii=False, indent=2), encoding='utf8')
    if repair['repaired_faces']:
        notify('uv_repair', f'局部 UV 修复完成：{repair["repaired_faces"]} 面，原有向三角面全部保留',
               device='cpu', uv_repair=repair, local_uv_repaired_faces=repair['repaired_faces'])
    return out_mapping, out_faces, out_uv
