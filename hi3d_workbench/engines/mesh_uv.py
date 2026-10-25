"""Prepare verified, reusable UVs without changing source geometry or normals."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
import zipfile
from pathlib import Path

UV_ALGORITHM_VERSION = 'verified-native-first-uv-v2'
UV_SOURCE_REVISION = '7cebca52227c63b4180ceed448cbd7ad22903b6d'
_CPU_CHUNK_FACES = 32768


def has_split_normals(mesh):
    """Coincident positions may still belong to distinct hard-edge corners."""
    import numpy as np
    vertices=np.asarray(mesh.vertices,dtype=np.float32)
    _,first,inverse=np.unique(vertices,axis=0,return_index=True,return_inverse=True)
    normals=np.asarray(mesh.vertex_normals)
    return bool(np.any(np.linalg.norm(normals-normals[first[inverse]],axis=1)>1e-4))


def match_source_vertices(mesh,positions):
    """Map native compacted vertices back to the supplied surface positions."""
    import numpy as np
    from scipy.spatial import cKDTree
    if has_split_normals(mesh):
        raise RuntimeError('重合顶点存在不同硬边法线，需要按原始索引展开 UV')
    original=np.asarray(mesh.vertices)
    distance,index=cKDTree(np.asarray(original,dtype=np.float32)).query(positions,workers=4)
    tolerance=max(float(np.ptp(original,axis=0).max())*2e-6,2e-7)
    if np.max(distance,initial=0)>tolerance:
        raise RuntimeError('UV 展开改变了输入顶点，已停止导出')
    return original[index],np.asarray(mesh.vertex_normals)[index]


def _position_keys(positions):
    """Exact keys in the float32 coordinate format received by CuMesh."""
    import numpy as np
    values=np.array(positions,dtype=np.float32,order='C',copy=True)
    if values.ndim!=2 or values.shape[1]!=3 or not np.isfinite(values).all():
        raise RuntimeError('UV 展开返回了无效顶点坐标')
    values[values==0]=0  # Treat signed zero as the same position.
    return values.view(np.dtype((np.void,12))).reshape(-1)


def _oriented_face_keys(faces):
    """Canonical cyclic order, preserving each triangle's winding."""
    import numpy as np
    faces=np.asarray(faces,dtype=np.int64)
    if faces.ndim!=2 or faces.shape[1]!=3:
        raise RuntimeError('UV 展开返回了无效三角面')
    start=faces.argmin(axis=1)
    ordered=np.ascontiguousarray(np.take_along_axis(faces,(start[:,None]+np.arange(3))%3,axis=1))
    return ordered.view(np.dtype((np.void,24))).reshape(-1)


def _require_face_subset(source_faces,result_faces):
    """Accept reordering/cyclic rotation and deletion, never new/reversed faces."""
    import numpy as np
    source,count=np.unique(_oriented_face_keys(source_faces),return_counts=True)
    result,result_count=np.unique(_oriented_face_keys(result_faces),return_counts=True)
    where=np.searchsorted(source,result)
    if (np.any(where>=len(source)) or not np.array_equal(source[where],result)
            or np.any(result_count>count[where])):
        raise RuntimeError('UV 展开改变了输入三角面，无法证明原始顶点对应关系')


def compacted_source_indices(mesh,positions,faces):
    """Recover exact source indices after native compaction/reordering.

    Only coincident vertices need incident-face signatures. The signature at a
    corner is its two neighbouring position IDs in winding order. Removed faces
    may make the native signature a subset, but exactly one source candidate
    must explain it; ambiguous overlapping topology falls back to original xref.
    """
    import numpy as np
    from collections import Counter
    source_faces=np.asarray(mesh.faces,dtype=np.int64)
    native_faces=np.asarray(faces,dtype=np.int64)
    keys,first,source_group,counts=np.unique(_position_keys(mesh.vertices),return_index=True,
        return_inverse=True,return_counts=True)
    native_keys=_position_keys(positions)
    native_group=np.searchsorted(keys,native_keys)
    if np.any(native_group>=len(keys)) or not np.array_equal(keys[native_group],native_keys):
        raise RuntimeError('UV 展开改变了输入顶点，无法使用精确拓扑映射')
    if (native_faces.ndim!=2 or native_faces.shape[1]!=3 or
            np.any(native_faces<0) or np.any(native_faces>=len(positions))):
        raise RuntimeError('UV 展开返回了无效三角面索引')
    mapping=first[native_group].copy()
    ambiguous=np.flatnonzero(counts[native_group]>1)
    if len(ambiguous):
        repeated=counts>1
        candidates={}
        for index in np.flatnonzero(repeated[source_group]):
            candidates.setdefault(int(source_group[index]),[]).append(int(index))

        def incidents(vertex_faces,groups):
            selected=vertex_faces[np.any(repeated[groups[vertex_faces]],axis=1)]
            result={}
            for face in selected:
                group=groups[face]
                for corner,index in enumerate(face):
                    if repeated[group[corner]]:
                        signature=(int(group[(corner+1)%3]),int(group[(corner+2)%3]))
                        result.setdefault(int(index),Counter())[signature]+=1
            return result

        original_incident=incidents(source_faces,source_group)
        native_incident=incidents(native_faces,native_group)
        for index in ambiguous:
            signature=native_incident.get(int(index),{})
            matches=[candidate for candidate in candidates[int(native_group[index])]
                if signature and all(original_incident.get(candidate,{}).get(face,0)>=count
                    for face,count in signature.items())]
            if len(matches)!=1:
                raise RuntimeError('重合顶点的相邻面无法唯一对应，需要按原始索引展开 UV')
            mapping[index]=matches[0]
    _require_face_subset(source_faces,mapping[native_faces])
    return mapping


def _original_index_atlas(mesh,cumesh,torch,trimesh):
    """Safety fallback for topology whose source identity cannot be proved."""
    import numpy as np
    vertices=np.asarray(mesh.vertices)
    normals=np.asarray(mesh.vertex_normals)
    atlas=cumesh.Atlas()
    atlas.add_mesh(torch.from_numpy(np.ascontiguousarray(vertices,dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(mesh.faces,dtype=np.int32)),
        normals=torch.from_numpy(np.ascontiguousarray(normals,dtype=np.float32)))
    atlas.compute_charts(normal_seam_weight=1001.)
    atlas.pack_charts(padding=2)
    mapping,faces,uv=atlas.get_mesh(0)
    mapping=mapping.numpy().astype(np.int64)
    uv_array=uv.numpy().copy();uv_array[:,1]=1-uv_array[:,1]
    if not np.isfinite(uv_array).all():raise RuntimeError('UV 展开产生无效坐标')
    return trimesh.Trimesh(vertices[mapping],faces.numpy().astype(np.int64),
        vertex_normals=normals[mapping],process=False,
        visual=trimesh.visual.TextureVisuals(uv=uv_array))


def _legacy_cumesh(mesh, notify, *, texture_size=2048, padding=4):
    import numpy as np
    import torch
    import trimesh
    import cumesh
    vertices=np.asarray(mesh.vertices)
    normals=np.asarray(mesh.vertex_normals)
    faces_array=np.asarray(mesh.faces)
    if len(vertices)>=faces_array.size and len(np.unique(faces_array))==faces_array.size:
        raise RuntimeError('CuMesh 不支持无共享索引的独立三角面；改用原索引分块展开')
    handler=cumesh.CuMesh()
    positions=torch.from_numpy(np.asarray(vertices,dtype=np.float32)).cuda()
    triangles=torch.from_numpy(np.asarray(mesh.faces,dtype=np.int32)).cuda()
    handler.init(positions,triangles)
    from .native_uv import unwrap
    unwrapped,faces,uv,mapping=unwrap(handler, notify=notify, texture_size=texture_size, padding=padding)
    # CuMesh removes degenerate faces and compacts/reorders vertices before
    # packing. Its returned vmap addresses that compacted mesh, not the input.
    # Recover identity using the compacted topology before applying that vmap.
    compact_positions,compact_faces=handler.read()
    compact_vertices=compact_positions.detach().cpu().numpy()
    compact_triangles=compact_faces.detach().cpu().numpy().astype(np.int64)
    native=unwrapped.detach().cpu().numpy()
    vmap=mapping.detach().cpu().numpy().astype(np.int64)
    face_array=faces.detach().cpu().numpy().astype(np.int64)
    uv_array=uv.detach().cpu().numpy().copy()
    try:
        source_indices=compacted_source_indices(mesh,compact_vertices,compact_triangles)
        if (face_array.ndim!=2 or face_array.shape[1]!=3 or
                np.any(face_array<0) or np.any(face_array>=len(vmap)) or
                np.any(vmap<0) or np.any(vmap>=len(compact_vertices)) or
                not np.array_equal(_position_keys(native),_position_keys(compact_vertices[vmap]))):
            raise RuntimeError('UV 顶点映射与压缩网格不一致')
        _require_face_subset(compact_triangles,vmap[face_array])
        original_indices=source_indices[vmap]
    except RuntimeError as error:
        # Identity is sometimes intrinsically ambiguous at a welded hard edge
        # or overlapping shell. Face/corner provenance is still provable: pair
        # every oriented position triangle with an original occurrence, then
        # split native UV vertices again wherever source corner indices differ.
        notify('uv_validate', '按三角面角点恢复原始硬边索引：'+str(error), device='cpu',
               fallback_reason=str(error))
        uv_array[:,1]=1-uv_array[:,1]
        original_indices, face_array, uv_array = _corner_mapped_arrays(mesh,native,face_array,uv_array,
                                                                      validate_layout=False)
        result = _result_from_arrays(mesh,original_indices,face_array,uv_array)
        del handler,positions,triangles,unwrapped,faces,uv,mapping,compact_positions,compact_faces
        torch.cuda.empty_cache()
        return result,'native_CuMesh_corner_mapped',original_indices
    # CuMesh raster coordinates are V-down; trimesh/glTF use V-up.
    uv_array[:,1]=1-uv_array[:,1]
    if not np.isfinite(uv_array).all():raise RuntimeError('UV 展开产生无效坐标')
    result=trimesh.Trimesh(vertices[original_indices],face_array,vertex_normals=normals[original_indices],process=False,
        visual=trimesh.visual.TextureVisuals(uv=uv_array))
    del handler,positions,triangles,unwrapped,faces,uv,mapping,compact_positions,compact_faces
    torch.cuda.empty_cache()
    return result,'native_CuMesh_topology_mapped',original_indices


def _corner_mapped_arrays(mesh, positions, faces, uv, *, validate_layout=True):
    """Recover original face corners even when a compacted vertex is ambiguous.

    Overlapping oriented face occurrences are paired one-to-one, never dropped.
    A shared UV vertex is split if it represents distinct original vertices;
    this preserves every original hard-edge corner normal without guessing
    which coincident source vertex a compacted vertex represents.
    """
    import numpy as np
    source_faces, _ = _valid_source_faces(mesh.vertices, mesh.faces)
    faces = np.asarray(faces, dtype=np.int64)
    positions = np.asarray(positions)
    uv = np.asarray(uv)
    if (faces.ndim != 2 or faces.shape[1] != 3 or np.any(faces < 0)
            or np.any(faces >= len(positions)) or uv.shape != (len(positions), 2)):
        raise RuntimeError('UV 角点恢复收到无效网格')
    keys, source_group = np.unique(_position_keys(mesh.vertices), return_inverse=True)
    native_keys = _position_keys(positions)
    native_group = np.searchsorted(keys, native_keys)
    if np.any(native_group >= len(keys)) or not np.array_equal(keys[native_group], native_keys):
        raise RuntimeError('UV 展开改变了输入坐标，无法恢复原始角点')
    source_groups = source_group[source_faces]
    native_groups = native_group[faces]
    # Distinct positions give a unique cyclic alignment. A triangle collapsed
    # by float32 conversion cannot prove that and must use a safe index path.
    for group in (source_groups, native_groups):
        if np.any((group[:, 0] == group[:, 1]) | (group[:, 1] == group[:, 2]) | (group[:, 2] == group[:, 0])):
            raise RuntimeError('UV 角点存在坐标退化，无法证明循环对应关系')
    source_start = source_groups.argmin(axis=1)
    native_start = native_groups.argmin(axis=1)
    source_corners = (source_start[:,None]+np.arange(3))%3
    native_corners = (native_start[:,None]+np.arange(3))%3
    source_order = np.argsort(_oriented_face_keys(source_groups),kind='stable')
    native_order = np.argsort(_oriented_face_keys(native_groups),kind='stable')
    if not np.array_equal(_oriented_face_keys(source_groups)[source_order],
                          _oriented_face_keys(native_groups)[native_order]):
        raise RuntimeError('UV 展开改变、丢失或重复了输入三角面，无法恢复角点')
    source_indices = np.take_along_axis(source_faces,source_corners,axis=1)[source_order]
    native_indices = np.take_along_axis(faces,native_corners,axis=1)[native_order]
    # Retain the native face ordering and cyclic rotation. np.unique sorts
    # native UV ID first, so an unambiguous mapping keeps existing vertex order.
    source_at_native = np.empty_like(source_indices)
    source_at_native[native_order] = source_indices
    original_corners = np.empty_like(source_at_native)
    np.put_along_axis(original_corners,native_corners,source_at_native,axis=1)
    pairs = np.stack((faces.reshape(-1),original_corners.reshape(-1)),axis=1)
    unique, inverse = np.unique(pairs,axis=0,return_inverse=True)
    mapping = unique[:,1]
    out_faces = inverse.reshape(-1,3)
    out_uv = uv[unique[:,0]]
    return _verify_arrays(mesh,source_faces,mapping,out_faces,out_uv,validate_layout=validate_layout)


def _valid_source_faces(vertices, faces):
    """Remove only provably zero-area triangles; no size/angle threshold."""
    import numpy as np
    faces = np.asarray(faces, dtype=np.int64)
    vertices = np.asarray(vertices)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError('UV 输入必须是有效、非空的有限坐标三角网格')
    triangle = vertices[faces].astype(np.float64)
    cross = np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0])
    nonzero = np.any(cross != 0., axis=1)
    repeated = ((faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0]))
    keep = nonzero & ~repeated
    if not keep.any():
        raise ValueError('UV 输入没有非退化三角面')
    return faces[keep], int((~keep).sum())


def _require_complete_faces(source_faces, result_faces):
    """Compare oriented triangle multisets, including occurrence counts."""
    import numpy as np
    original = np.sort(_oriented_face_keys(source_faces))
    output = np.sort(_oriented_face_keys(result_faces))
    if not np.array_equal(original, output):
        raise RuntimeError('UV 展开改变、丢失或重复了非退化三角面，已拒绝结果')


def _verify_arrays(mesh, source_faces, mapping, faces, uv, *, validate_layout=True):
    """The only accepted provenance is an explicit original source index."""
    import numpy as np
    mapping = np.asarray(mapping)
    faces = np.asarray(faces)
    uv = np.asarray(uv)
    if (mapping.ndim != 1 or mapping.dtype.kind not in 'iu' or not len(mapping)
            or len(mapping) > len(source_faces)*3 or np.any(mapping < 0) or np.any(mapping >= len(mesh.vertices))
            or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu'
            or np.any(faces < 0) or np.any(faces >= len(mapping))
            or uv.shape != (len(mapping), 2) or uv.dtype.kind != 'f' or not np.isfinite(uv).all()
            or np.any(uv < -1e-5) or np.any(uv > 1.00001)):
        raise RuntimeError('UV 坐标或原始索引映射无效，已拒绝结果')
    _require_complete_faces(source_faces, mapping[faces])
    if validate_layout:
        _verify_uv_layout(faces, uv)
    return mapping.astype(np.int64, copy=False), faces.astype(np.int64, copy=False), uv


def _verify_uv_layout(faces, uv):
    """Reject collapsed/reused atlas triangles without a quadratic face scan.

    Native chart packers enforce disjoint chart placement. This additional
    linear/sort guard catches lost pages, zero-area islands and exact mirrored
    reuse; it is not an exhaustive arbitrary-polygon intersection solver.
    """
    import numpy as np
    triangles = np.asarray(uv, dtype=np.float64)[faces]
    edge1, edge2 = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
    double_area = np.abs(edge1[:, 0]*edge2[:, 1]-edge1[:, 1]*edge2[:, 0])
    if np.any(double_area == 0.):
        raise RuntimeError('UV 展开产生零面积三角形，已拒绝不可贴图结果')
    if float(double_area.sum())*.5 > 1.001:
        raise RuntimeError('UV 图块面积超出单张图集，可能发生重叠或多页布局')
    order = np.lexsort((triangles[:, :, 1], triangles[:, :, 0]), axis=1)
    canonical = np.ascontiguousarray(np.take_along_axis(triangles, order[:, :, None], axis=1)).reshape(-1, 6)
    keys = canonical.view(np.dtype((np.void, 48))).reshape(-1)
    if len(np.unique(keys)) != len(keys):
        raise RuntimeError('UV 存在完全重叠或镜像复用的三角面，已拒绝结果')


def _result_from_arrays(mesh, mapping, faces, uv):
    import numpy as np
    import trimesh
    # Always copy source attributes. A cache/native implementation may supply
    # indices and UVs, never substitute rounded/moved coordinates or normals.
    return trimesh.Trimesh(np.asarray(mesh.vertices)[mapping], faces,
                           vertex_normals=np.asarray(mesh.vertex_normals)[mapping], process=False,
                           visual=trimesh.visual.TextureVisuals(uv=uv))


def _digest_arrays(arrays):
    import numpy as np
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode('ascii'))
        digest.update(json.dumps(array.shape).encode('ascii'))
        digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def _cache_identity(mesh, texture_size, backend, padding):
    import numpy as np
    attributes = _digest_arrays((np.asarray(mesh.vertices, dtype=np.float64),
                                np.asarray(mesh.faces, dtype=np.int64),
                                np.asarray(mesh.vertex_normals, dtype=np.float64)))
    algorithm_digest = hashlib.sha256()
    for source in (Path(__file__), Path(__file__).with_name('community_uv.py'),
                   Path(__file__).with_name('native_uv.py')):
        algorithm_digest.update(source.name.encode('utf8'))
        algorithm_digest.update(source.read_bytes())
    params = {'algorithm': UV_ALGORITHM_VERSION, 'algorithm_sha256': algorithm_digest.hexdigest(),
              'source_revision': UV_SOURCE_REVISION,
              'geometry_sha256': attributes, 'texture_size': int(texture_size),
              'padding': int(padding), 'backend': backend, 'welding': False,
              'cpu_chunk_faces': _CPU_CHUNK_FACES}
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode('utf8')).hexdigest(), params


def _read_uv_cache(path, identity, mesh, source_faces):
    import numpy as np
    import zipfile
    # Do not let a malformed compressed cache allocate arbitrary amounts of
    # memory before provenance checks. A UV mesh needs at most 3 vertices/face.
    maximum = max(len(source_faces)*160 + 65536, 131072)
    with zipfile.ZipFile(path) as archive:
        if sum(info.file_size for info in archive.infolist()) > maximum:
            raise RuntimeError('UV 缓存解压大小超出原网格界限')
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {'mapping', 'faces', 'uv', 'metadata'}:
            raise RuntimeError('UV 缓存字段不匹配')
        metadata = json.loads(str(data['metadata'].item()))
        if metadata.get('identity') != identity or metadata.get('algorithm') != UV_ALGORITHM_VERSION:
            raise RuntimeError('UV 缓存身份不匹配')
        mapping, faces, uv = data['mapping'], data['faces'], data['uv']
        if metadata.get('payload_sha256') != _digest_arrays((mapping, faces, uv)):
            raise RuntimeError('UV 缓存内容校验失败')
    return _verify_arrays(mesh, source_faces, mapping, faces, uv), metadata


def _write_uv_cache(path, identity, params, arrays, policy, removed):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    mapping, faces, uv = arrays
    metadata = {**params, 'identity': identity, 'policy': policy, 'removed_degenerate_faces': removed,
                'payload_sha256': _digest_arrays((mapping, faces, uv))}
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, mapping=mapping, faces=faces, uv=uv,
                                metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _topology_chunks(faces, max_faces=_CPU_CHUNK_FACES):
    """Local face-graph partitions; preserve all indices, positions and normals."""
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    count = len(faces)
    if count <= max_faces:
        return [np.arange(count, dtype=np.int64)]
    # Pair only manifold index edges. Nonmanifold edges and authored hard
    # edges are already chart boundaries and need no guessed geometric weld.
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    owners = np.tile(np.arange(count), 3)
    keys = np.ascontiguousarray(edges).view(np.dtype((np.void, 16))).reshape(-1)
    order = np.argsort(keys, kind='stable')
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1])+1]
    ends = np.r_[starts[1:], len(order)]
    pairs = starts[(ends-starts) == 2]
    left, right = owners[order[pairs]], owners[order[pairs+1]]
    graph = coo_matrix((np.ones(len(left)*2, dtype=np.int8),
                        (np.r_[left, right], np.r_[right, left])), shape=(count, count)).tocsr()
    # RCM puts connected, spatially local face neighborhoods together. Its
    # subranges add seams only; packing remains global across every partition.
    traversal = reverse_cuthill_mckee(graph, symmetric_mode=True)
    return [traversal[start:start+max_faces] for start in range(0, count, max_faces)]


def _chunked_atlas(mesh, source_faces, texture_size, padding, notify):
    import numpy as np
    import torch
    import cumesh
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    notify('uv_partition', f'按原始拓扑分区 {len(source_faces)} 个面', device='cpu')
    chunks = _topology_chunks(source_faces, max_faces=_CPU_CHUNK_FACES)
    atlas = cumesh.Atlas()
    source_maps = []
    for index, chunk in enumerate(chunks):
        selected = source_faces[chunk]
        mapping, local = np.unique(selected, return_inverse=True)
        atlas.add_mesh(torch.from_numpy(np.ascontiguousarray(vertices[mapping], dtype=np.float32)),
                       torch.from_numpy(np.ascontiguousarray(local.reshape(-1, 3), dtype=np.int32)),
                       normals=torch.from_numpy(np.ascontiguousarray(normals[mapping], dtype=np.float32)))
        source_maps.append(mapping)
        notify('uv_partition', f'准备原索引图块：{index+1}/{len(chunks)}', device='cpu',
               current=index+1, total=len(chunks), unit='chunks')

    # CuMesh exposes the native xatlas callback without requiring stdout/tqdm
    # parsing. Category progress is a native local phase percent, not a global
    # task percentage. No process-global progress hook is installed.
    callback_lock = threading.RLock()
    def progress_factory():
        def callback(category, percent):
            with callback_lock:
                notify('uv_charts' if 'pack' not in str(category).lower() else 'uv_pack',
                       f'xatlas {category}：{int(percent)}%', device='cpu',
                       current=int(percent), total=100, unit='percent', native_category=str(category))
            return True
        return callback
    atlas._pbar = None
    atlas._create_progress_callback = progress_factory
    notify('uv_charts', f'CPU 安全展开 {len(chunks)} 个原索引分区', device='cpu')
    atlas.compute_charts(normal_seam_weight=1001., verbose=True)
    notify('uv_pack', '统一排布所有分区（保留硬边法线）', device='cpu')
    # In xatlas, texels_per_unit==0 estimates density from resolution and sets
    # maxResolution=0: the SAME atlas grows until all charts fit. A nonzero
    # density together with fixed resolution may instead create multiple pages,
    # whose atlasIndex this lightweight wrapper does not expose. Keep this
    # explicit, and never normalize each partition independently.
    atlas.pack_charts(resolution=int(texture_size), texels_per_unit=0., padding=int(padding), verbose=True)
    mappings, face_arrays, uv_arrays = [], [], []
    cursor = 0
    for index, source_map in enumerate(source_maps):
        mapping, faces, uv = atlas.get_mesh(index)
        mapping = np.asarray(mapping.numpy(), dtype=np.int64)
        faces = np.asarray(faces.numpy(), dtype=np.int64)
        uv = np.array(uv.numpy(), copy=True)
        uv[:, 1] = 1. - uv[:, 1]
        if np.any(mapping < 0) or np.any(mapping >= len(source_map)):
            raise RuntimeError('xatlas 返回了无效原索引')
        mappings.append(source_map[mapping])
        face_arrays.append(faces+cursor)
        uv_arrays.append(uv)
        cursor += len(mapping)
        notify('uv_assemble', f'合并 UV 分区：{index+1}/{len(chunks)}', device='cpu',
               current=index+1, total=len(chunks), unit='chunks')
    return np.concatenate(mappings), np.concatenate(face_arrays), np.concatenate(uv_arrays)


def _legacy_arrays(mesh, source_faces, notify, *, texture_size=2048, padding=4):
    """Compatibility path when the installed runtime predates community PEC."""
    import numpy as np
    import trimesh
    # CuMesh must receive the same explicit valid-face set we later verify.
    # Keep the complete original vertex array (including unused vertices) so
    # every native xref remains an index into the caller's original attributes.
    native_input = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(source_faces),
                                   vertex_normals=np.asarray(mesh.vertex_normals), process=False)
    result, policy, mapping = _legacy_cumesh(native_input, notify,
                                             texture_size=texture_size, padding=padding)
    # Prove original identity and complete oriented geometry BEFORE touching
    # a layout. Only UV zero-area/reuse checks are deferred to after repair.
    mapping, faces, uv = _verify_arrays(mesh, source_faces, mapping, result.faces, result.visual.uv,
                                        validate_layout=False)
    from .community_uv import (_repair_local_faces, _save_diagnostics, _zero_uv_faces,
                               _duplicate_uv_faces)
    pixels = np.asarray(uv, dtype=np.float64)*float(texture_size)
    pixels[:, 1] = float(texture_size)-pixels[:, 1]
    metadata = {'backend': 'CuMesh', 'faces': len(faces), 'texture_size': texture_size,
                'padding': padding, 'postpack_zero_faces': len(_zero_uv_faces(faces, uv)),
                'postpack_duplicate_faces': len(_duplicate_uv_faces(faces, uv))}
    _save_diagnostics({'mapping': mapping, 'faces': faces, 'uv': uv, 'pixel_uv': pixels,
                       'source_vertices': np.asarray(mesh.vertices), 'source_faces': source_faces},
                      metadata, prefix='native')
    notify('uv_validate', '检查 CuMesh 图集并局部修复退化或复用面', device='cpu', uv_diagnostics=metadata)
    new_mapping, new_faces, new_uv, repair = _repair_local_faces(
        mesh.vertices, mapping, faces, pixels, texture_size, texture_size, padding, notify)
    arrays = _verify_arrays(mesh, source_faces, new_mapping, new_faces, new_uv)
    if repair['repaired_faces']:
        policy += '_local_repair'
        notify('uv_repair', f'CuMesh UV 局部修复完成：{repair["repaired_faces"]} 面',
               device='cpu', uv_repair=repair, local_uv_repaired_faces=repair['repaired_faces'])
    return _result_from_arrays(mesh, *arrays), arrays, policy


def ensure_uv(mesh, *, emit=None, cache_dir=None, texture_size=2048, backend='auto'):
    """Return ``(mesh, policy)``; preserve geometry, normals and existing UVs.

    ``auto`` prefers the verified native CuMesh GPU-clustered atlas, then
    community GPU PEC without welding, then an explicitly reported original-
    index partitioned CPU atlas. Each backend runs at most once. Explicit
    backend requests never select another backend. No path simplifies a mesh.
    Cache files contain UVs and source indices only and are verified on load.
    """
    import numpy as np
    if backend not in {'auto', 'community', 'comfy_gpu', 'cumesh', 'cpu'}:
        raise ValueError('未知 UV 后端：'+str(backend))
    if not isinstance(texture_size, int) or not 128 <= texture_size <= 16384:
        raise ValueError('UV 贴图尺寸必须在 128 至 16384 之间')
    if getattr(mesh.visual, 'uv', None) is not None:
        return mesh, 'preserved_existing'
    started = time.perf_counter()
    emit_lock = threading.RLock()
    last_event = {}
    def notify(phase, message, *, device='cpu', current=None, total=None, unit=None, **extra):
        if emit is None:
            return
        now = time.perf_counter()
        # Native callbacks may arrive from xatlas worker threads. Serialize
        # emissions and avoid hundreds of JSON/disk writes per second.
        with emit_lock:
            previous = last_event.get(phase)
            finished = total is not None and current == total
            if previous and current is not None and not finished and now-previous < .15:
                return
            last_event[phase] = now
            event = {'stage': 'unwrapping', 'phase': phase, 'device': device,
                     'message': message, 'elapsed_seconds': round(now-started, 3), **extra}
            if current is not None:
                event['progress_current'] = int(current)
            if total is not None:
                event['progress_total'] = int(total)
            if unit is not None:
                event['progress_unit'] = unit
            emit(**event)

    source_faces, removed = _valid_source_faces(mesh.vertices, mesh.faces)
    if removed:
        notify('uv_validate', f'仅排除 {removed} 个零面积退化面', removed_degenerate_faces=removed)
    normals = np.asarray(mesh.vertex_normals)
    if normals.shape != np.asarray(mesh.vertices).shape or not np.isfinite(normals).all():
        raise ValueError('UV 输入包含无效法线')
    padding = 4
    identity, params = _cache_identity(mesh, texture_size, backend, padding)
    cache_path = Path(cache_dir)/f'{identity}.npz' if cache_dir is not None else None
    if cache_path is not None and cache_path.is_file():
        try:
            arrays, metadata = _read_uv_cache(cache_path, identity, mesh, source_faces)
            notify('uv_cache', '复用校验通过的 UV 缓存', current=1, total=1, unit='cache',
                   cache_hit=True, cache_key=identity, uv_backend=metadata.get('policy'),
                   removed_degenerate_faces=removed)
            return _result_from_arrays(mesh, *arrays), 'cached_'+metadata.get('policy', 'verified_uv')
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as error:
            notify('uv_cache', 'UV 缓存无效，重新展开：'+str(error), cache_hit=False, cache_key=identity)

    arrays = None
    policy = None
    failures = []
    from .community_uv import LocalUVRepairError
    choices = ('cumesh', 'community', 'cpu') if backend == 'auto' else (
        'community' if backend == 'comfy_gpu' else backend,)
    labels = {'cumesh': 'CuMesh', 'community': 'community GPU', 'cpu': 'CPU 原索引分区'}
    for selected in choices:
        try:
            if selected == 'cumesh':
                _, arrays, policy = _legacy_arrays(mesh, source_faces, notify,
                                                    texture_size=texture_size, padding=padding)
            elif selected == 'community':
                from .community_uv import unwrap
                arrays = unwrap(np.asarray(mesh.vertices), source_faces, texture_size, padding, notify)
                policy = 'community_GPU_PEC_original_indices'
            else:
                arrays = _chunked_atlas(mesh, source_faces, texture_size, padding, notify)
                policy = 'native_xatlas_partitioned_original_indices'
            # Every backend crosses the same final provenance/layout boundary,
            # including results that have already passed their local checks.
            arrays = _verify_arrays(mesh, source_faces, *arrays)
            break
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            # A bounded repair failure is structural. Switching algorithms to
            # evade its safety budget would conceal the rejected input/output.
            if isinstance(error, LocalUVRepairError) or backend != 'auto' or selected == 'cpu':
                raise
            arrays, policy = None, None
            failures.append(labels[selected]+': '+str(error))
            next_backend = choices[choices.index(selected)+1]
            notify('uv_fallback', labels[selected]+' 路径拒绝：'+str(error)+'；改用 '+labels[next_backend],
                   fallback_reason=str(error), failed_backend=selected, next_backend=next_backend)
    result = _result_from_arrays(mesh, *arrays)
    notify('uv_validate', f'UV 校验完成：{len(source_faces)} 个面、原位置与法线完整保留',
           current=len(source_faces), total=len(source_faces), unit='faces', uv_backend=policy,
           removed_degenerate_faces=removed, fallback_reason='; '.join(failures) or None)
    if cache_path is not None:
        try:
            _write_uv_cache(cache_path, identity, params, arrays, policy, removed)
        except OSError as error:
            notify('uv_cache', 'UV 已生成，但缓存保存失败：'+str(error), cache_hit=False)
    return result, policy
