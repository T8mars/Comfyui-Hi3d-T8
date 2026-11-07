from collections import Counter
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from test_workbench import testroot
from hi3d_workbench.studio.generated_geometry import prepare


def corners(vertices,faces):
    return Counter(tuple(map(tuple,vertices[face].astype(np.float64))) for face in faces)


def test_actual_xpart_vase_float32_edge_collapse_keeps_every_valid_oriented_triangle():
    value=json.loads((Path(__file__).parent/'fixtures/xpart-vase-float32-edge-collapse.json').read_text())
    mesh=trimesh.Trimesh(value['vertices'],value['faces'],process=False)
    encoded=mesh.vertices.astype(np.float32);triangle=encoded[mesh.faces].astype(np.float64)
    keep=np.hypot.reduce(np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]),axis=1)>0
    assert (~keep).sum()==2 and (mesh.area_faces>0).all()
    output,report=prepare(mesh)
    assert report['removed_collapsed_faces']==2 and report['merged_float32_vertices']==1
    assert corners(output.vertices,output.faces)==corners(encoded,mesh.faces[keep])


def test_collapsing_a_subdivided_cube_edge_retains_closed_volume_and_winding():
    cube=trimesh.creation.box();vertices=cube.vertices+[1,1,1]
    a,b=map(int,cube.faces[0,:2]);point=vertices[a]+(vertices[b]-vertices[a])*2**-28
    inserted=len(vertices);faces=[]
    for face in cube.faces:
        for k in range(3):
            u,v,w=map(int,face[np.arange(k,k+3)%3])
            if {u,v}=={a,b}:
                faces.extend([[u,inserted,w],[inserted,v,w]]);break
        else:faces.append(face.tolist())
    mesh=trimesh.Trimesh(np.vstack([vertices,point]),faces,process=False)
    assert mesh.is_watertight and len(mesh.faces)==14
    output,report=prepare(mesh)
    assert len(output.faces)==12 and report['removed_collapsed_faces']==2
    assert output.is_watertight and output.is_winding_consistent
    assert output.volume==pytest.approx(cube.volume)


def test_positive_area_slivers_are_retained_without_a_relative_area_cutoff():
    mesh=trimesh.Trimesh([[0,0,0],[1,0,0],[.25,1e-20,0]],[[0,1,2]],process=False)
    output,report=prepare(mesh)
    assert len(output.faces)==1 and output.area_faces[0]>0
    assert report['removed_collapsed_faces']==0


def test_collinear_and_entirely_collapsed_generated_parts_fail_explicitly():
    for points in ([[0,0,0],[1,0,0],[.5,0,0]],[[1,1,1],[1+1e-10,1,1],[1,1+1e-10,1]]):
        with pytest.raises(ValueError,match='零面积|有效三角面'):
            prepare(trimesh.Trimesh(points,[[0,1,2]],process=False))
