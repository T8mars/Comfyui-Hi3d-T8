import numpy as np
import trimesh

from hi3d_workbench.studio.topology import split_fans


def test_manifold_box_keeps_every_oriented_triangle_and_closed_surface():
    mesh = trimesh.creation.box()
    original = mesh.vertices[mesh.faces].copy()
    result, report = split_fans(mesh)
    np.testing.assert_array_equal(result.vertices[result.faces], original)
    np.testing.assert_array_equal(mesh.vertices[mesh.faces], original)
    assert len(result.vertices) == 8 and result.is_watertight
    assert report['boundary_edges_after'] == report['nonmanifold_edges_after'] == 0


def test_three_faces_at_one_edge_are_separated_without_removing_any_face():
    vertices = [[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]]
    faces = [[0, 1, 2], [1, 0, 3], [0, 1, 4]]
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    result, report = split_fans(mesh)
    assert report['nonmanifold_edges_before'] == 1 and report['nonmanifold_edges_after'] == 0
    assert len(result.faces) == 3 and report['boundary_edges_after'] == 9
    np.testing.assert_array_equal(result.vertices[result.faces], mesh.vertices[mesh.faces])


def test_two_closed_tetrahedra_touching_at_a_point_get_separate_vertex_fans():
    vertices = [[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3],
             [0, 4, 5], [0, 6, 4], [0, 5, 6], [4, 6, 5]]
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    result, report = split_fans(mesh)
    assert report['nonmanifold_edges_before'] == 0
    assert len(result.vertices) == 8 and result.is_watertight
    assert len(result.split(only_watertight=False)) == 2
    np.testing.assert_array_equal(result.vertices[result.faces], mesh.vertices[mesh.faces])
