"""Scene assembly must retain each primitive's triangles beyond narrow indices."""
import numpy as np
import pytest
import trimesh

from test_workbench import testroot
from hi3d_workbench.studio.gltf import Document
from hi3d_workbench.studio.scene import combined, instances
from hi3d_workbench.studio.retopology import visible_mesh
from hi3d_workbench.studio.parts import write_parts


def indexed_instances(dtype=np.uint16, *, mirror=True):
    small=dtype==np.uint8
    mesh=trimesh.creation.icosphere(subdivisions=2 if small else 5)
    positions=np.asarray(mesh.vertices,dtype=np.float32)
    faces=np.asarray(mesh.faces,dtype=dtype)
    count=2 if small else 7
    document=Document({'asset':{'version':'2.0'},'scene':0,
        'scenes':[{'nodes':list(range(count))}],'nodes':[],
        'meshes':[{'primitives':[]}]})
    document.tree['meshes'][0]['primitives']=[{
        'attributes':{'POSITION':document.add_accessor(positions)},
        'indices':document.add_accessor(faces.reshape(-1),target=34963)}]
    expected=[]
    for index in range(count):
        reflection=mirror and index==count-1
        scale=np.array([-1.,1.,1.]) if reflection else np.ones(3)
        translation=np.array([index*3.,0.,0.])
        document.tree['nodes'].append({'mesh':0,'translation':translation.tolist(),'scale':scale.tolist()})
        world=positions.astype(np.float64)*scale+translation
        expected.append(world[mesh.faces[:,[0,2,1]] if reflection else mesh.faces])
    return document,np.concatenate(expected),count


@pytest.mark.parametrize('dtype',[np.uint8,np.uint16,np.uint32])
def test_combined_and_remesh_input_preserve_original_triangles_across_index_limits(dtype):
    source,expected,count=indexed_instances(dtype)
    assert all(row.faces.dtype==dtype for row in instances(source))
    vertices,faces,rows=combined(source)
    assert len(rows)==count
    np.testing.assert_array_equal(vertices[faces],expected)
    remesh_input=visible_mesh(source)
    np.testing.assert_array_equal(remesh_input.triangles,expected)
    assert remesh_input.is_watertight and remesh_input.is_winding_consistent


def test_large_merged_part_reports_actual_closed_surface_without_index_wrap(tmp_path):
    source,expected,_=indexed_instances(mirror=False)
    labels=[np.zeros(len(row.faces),dtype=np.int32) for row in instances(source)]
    target=tmp_path/'parts.glb'
    metadata=write_parts(source,labels,{0:'seven shells'},target,engine='source partition',source_sha256='fixture')
    assert metadata['faces']==len(expected) and len(metadata['parts'])==1
    assert metadata['parts'][0]['boundary_edges']==0
    assert metadata['parts'][0]['watertight'] is True
    # Read the serialized primitives individually: exported original geometry
    # must stay valid independently of the assembly/diagnostic implementation.
    exported=Document.load(target);actual=[]
    for row in instances(exported):actual.append(row.world_positions()[row.faces])
    np.testing.assert_array_equal(np.concatenate(actual),expected)
