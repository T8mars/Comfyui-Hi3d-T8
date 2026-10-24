"""Prepare new UVs using the native runtime before loading diffusion models."""
from __future__ import annotations


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


def ensure_uv(mesh):
    import numpy as np
    if getattr(mesh.visual,'uv',None) is not None:return mesh,'preserved_existing'
    import torch
    import trimesh
    import cumesh
    vertices=np.asarray(mesh.vertices)
    normals=np.asarray(mesh.vertex_normals)
    # Fully unmerged STL-style triangles have no indexed adjacency. The pinned
    # CuMesh adjacency kernel launches an invalid zero-size grid for this case.
    # The CPU atlas keeps their original corner indices without launching CUDA.
    faces_array=np.asarray(mesh.faces)
    if len(vertices)>=faces_array.size and len(np.unique(faces_array))==faces_array.size:
        return _original_index_atlas(mesh,cumesh,torch,trimesh),'native_xatlas_disconnected_triangles'
    handler=cumesh.CuMesh()
    positions=torch.from_numpy(np.asarray(vertices,dtype=np.float32)).cuda()
    triangles=torch.from_numpy(np.asarray(mesh.faces,dtype=np.int32)).cuda()
    handler.init(positions,triangles)
    unwrapped,faces,uv,mapping=handler.uv_unwrap(return_vmaps=True)
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
    except RuntimeError:
        # A few hard edges must not force the entire million-face mesh into
        # slow unclustered xatlas. Use it only if exact identity is unprovable.
        del handler,positions,triangles,unwrapped,faces,uv,mapping,compact_positions,compact_faces
        torch.cuda.empty_cache()
        return _original_index_atlas(mesh,cumesh,torch,trimesh),'native_xatlas_original_index_fallback'
    # CuMesh raster coordinates are V-down; trimesh/glTF use V-up.
    uv_array[:,1]=1-uv_array[:,1]
    if not np.isfinite(uv_array).all():raise RuntimeError('UV 展开产生无效坐标')
    result=trimesh.Trimesh(vertices[original_indices],face_array,vertex_normals=normals[original_indices],process=False,
        visual=trimesh.visual.TextureVisuals(uv=uv_array))
    del handler,positions,triangles,unwrapped,faces,uv,mapping,compact_positions,compact_faces
    torch.cuda.empty_cache()
    return result,'native_CuMesh_topology_mapped'
