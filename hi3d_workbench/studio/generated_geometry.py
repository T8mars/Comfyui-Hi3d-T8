"""Prepare newly generated bare surfaces for glTF's float32 positions.

This never processes an uploaded mesh or its original face partition. A VAE
surface can have distinct float64 world vertices that coincide after glTF
serialization. Weld those exact float32 positions and remove only faces whose
corners then coincide. Keep every other oriented triangle, including slivers.
"""
from __future__ import annotations

import numpy as np
import trimesh


def prepare(mesh):
    vertices=np.asarray(mesh.vertices,dtype=np.float64)
    faces=np.asarray(mesh.faces)
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or not len(vertices)
            or faces.ndim!=2 or faces.shape[1]!=3 or not len(faces)
            or faces.dtype.kind not in 'iu' or not np.isfinite(vertices).all()
            or np.any(faces<0) or np.any(faces>=len(vertices))):
        raise ValueError('新生成部件的顶点或三角面无效')
    with np.errstate(over='ignore'):
        encoded=vertices.astype(np.float32)
    if not np.isfinite(encoded).all():raise ValueError('新生成部件超出 GLB 坐标精度范围')
    unique,inverse=np.unique(encoded,axis=0,return_inverse=True)
    mapped=inverse[faces]
    keep=(mapped[:,0]!=mapped[:,1])&(mapped[:,1]!=mapped[:,2])&(mapped[:,2]!=mapped[:,0])
    if not keep.any():raise ValueError('新生成部件在 GLB 坐标精度下没有有效三角面')
    triangles=unique[mapped[keep]].astype(np.float64)
    area=np.hypot.reduce(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
    if np.any(area==0):raise ValueError('新生成部件仍有不同顶点构成的零面积面，已保留原拆件并停止')
    output=trimesh.Trimesh(unique,mapped[keep],process=False)
    output.remove_unreferenced_vertices()
    closed=bool(mesh.is_watertight);prepared_closed=bool(output.is_watertight)
    winding=bool(mesh.is_winding_consistent);prepared_winding=bool(output.is_winding_consistent)
    if closed and not prepared_closed:raise ValueError('新生成部件在 GLB 坐标精度下无法保持闭合，已停止')
    if winding and not prepared_winding:raise ValueError('新生成部件在 GLB 坐标精度下无法保持面朝向，已停止')
    return output,{'scope':'new generated surface only; original input and face partition untouched',
        'input_faces':len(faces),'output_faces':len(output.faces),
        'merged_float32_vertices':len(encoded)-len(unique),'removed_collapsed_faces':int((~keep).sum()),
        'unused_vertices_removed':len(unique)-len(output.vertices),
        'max_position_roundoff':float(np.abs(vertices-encoded.astype(np.float64)).max()),
        'valid_oriented_triangles_preserved':True,'input_watertight':closed,'output_watertight':prepared_closed,
        'input_winding_consistent':winding,'output_winding_consistent':prepared_winding}
