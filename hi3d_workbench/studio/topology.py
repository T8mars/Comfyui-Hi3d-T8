"""Separate vertex fans at non-manifold contacts without deleting any face."""
from __future__ import annotations

import numpy as np


def split_fans(mesh, emit=lambda *a, **kw: None):
    import trimesh
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if not len(faces) or not np.isfinite(vertices).all():
        raise ValueError('拓扑副本需要非空、有限的三角网格')
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError('拓扑副本的三角面索引无效')
    emit('asset_topology', '按相邻面扇区拆分非流形连接，保持原三角面与坐标', device='cpu')
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    _, first, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0,
        return_index=True, return_inverse=True, return_counts=True)
    last = np.full(len(counts), -1, dtype=np.int64)
    np.maximum.at(last, inverse, np.arange(len(edges)))
    corners = np.arange(faces.size).reshape(-1, 3)
    endpoint_corners = corners[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    reverse = edges[:, 0] > edges[:, 1]
    endpoint_corners[reverse] = endpoint_corners[reverse, ::-1]
    manifold = counts == 2
    a = endpoint_corners[first[manifold]].reshape(-1)
    b = endpoint_corners[last[manifold]].reshape(-1)
    # Each corner can connect through at most two incident manifold edges.
    # Components are vertex fans (paths/cycles), rather than whole objects.
    graph = coo_matrix((np.ones(len(a), dtype=np.uint8), (a, b)), shape=(faces.size, faces.size)).tocsr()
    number, labels = connected_components(graph, directed=False)
    _, representative = np.unique(labels, return_index=True)
    output_vertices = vertices[faces.reshape(-1)[representative]]
    output_faces = labels.reshape(-1, 3)
    output = trimesh.Trimesh(output_vertices, output_faces, process=False)
    # Check in bounded blocks; do not create full face-position copies.
    for start in range(0, len(faces), 65536):
        if not np.array_equal(output.vertices[output.faces[start:start + 65536]],
                              vertices[faces[start:start + 65536]]):
            raise RuntimeError('拓扑拆分改变了原三角面坐标，已停止')
    remaining = np.bincount(output.edges_unique_inverse)
    if np.any(remaining > 2):
        raise RuntimeError('拓扑拆分后仍存在非流形边，已停止')
    report = {'engine': 'manifold vertex-fan splitting', 'faces': len(faces),
        'input_vertices': len(vertices), 'output_vertices': number,
        'nonmanifold_edges_before': int((counts > 2).sum()), 'nonmanifold_edges_after': 0,
        'boundary_edges_before': int((counts == 1).sum()), 'boundary_edges_after': int((remaining == 1).sum()),
        'oriented_triangles_preserved': True, 'maximum_position_delta': 0.}
    emit('asset_topology', '拓扑扇区拆分完成，原三角面与坐标逐项验证通过', device='cpu')
    return output, report
