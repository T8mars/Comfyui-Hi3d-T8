"""Catch loss of a nonzero dimension when GLB float32 positions are encoded."""
import io

import numpy as np
import pytest
import trimesh
from test_server_operations import isolated_server


def offset_box():
    mesh=trimesh.creation.box()
    mesh.apply_translation([1e8,0,0])
    assert np.allclose(mesh.extents,[1,1,1])
    return mesh


def test_glb_precision_rejects_partial_axis_collapse():
    from hi3d_workbench.geometry import validate
    with pytest.raises(ValueError,match='精度|居中'):
        validate(offset_box())


@pytest.mark.parametrize('format',['npz','glb'])
def test_mesh_upload_does_not_silently_flatten_a_valid_box(isolated_server,format):
    server,client=isolated_server
    mesh=offset_box();buffer=io.BytesIO()
    if format=='npz':np.savez(buffer,vertices=mesh.vertices,faces=mesh.faces)
    else:
        scene=trimesh.Scene()
        transform=np.eye(4);transform[0,3]=1e8
        scene.add_geometry(trimesh.creation.box(),transform=transform)
        buffer.write(scene.export(file_type='glb'))
    before=set(server.ASSETS.iterdir())
    response=client.post('/api/assets',files={'file':('offset-box.'+format,buffer.getvalue(),'application/octet-stream')})
    assert response.status_code==400,response.text
    assert '居中' in response.json()['detail']
    assert set(server.ASSETS.iterdir())==before


def test_exact_planar_and_representable_offset_meshes_are_allowed():
    from hi3d_workbench.geometry import validate
    plane=trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[1,1,0],[0,1,0]],faces=[[0,1,2],[0,2,3]],process=False)
    assert np.array_equal(validate(plane).vertices,plane.vertices)
    mesh=trimesh.creation.box(extents=[16,1,1]);mesh.apply_translation([1e8,0,0])
    assert np.array_equal(validate(mesh).vertices,mesh.vertices)
