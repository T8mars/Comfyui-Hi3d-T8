"""Inspect a real worker result against its original PBR surface and quad OBJ."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import trimesh

from hi3d_workbench.common import atomic_json, digest, read_json
from hi3d_workbench.studio.gltf import Document
from hi3d_workbench.studio.material_projection import Surface
from hi3d_workbench.studio.retopology import visible_mesh, geometry_coverage, validate_coverage
from hi3d_workbench.studio.uv_atlas import oriented_keys


def components(mesh):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    edges = mesh.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    graph = coo_matrix((np.ones(len(edges), dtype=np.uint8), (edges[:, 0], edges[:, 1])),
                       shape=(len(mesh.vertices), len(mesh.vertices))).tocsr()
    _, labels = connected_components(graph, directed=False)
    return len(np.unique(labels[np.unique(mesh.faces)]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--samples', type=int, default=10000)
    args = parser.parse_args()
    if not 1 <= args.samples <= 100000:
        parser.error('samples must be between 1 and 100000')
    folder = args.directory.resolve()
    left = trimesh.creation.box()
    right = left.copy(); right.apply_translation([3, 0, 0])
    missing_check = geometry_coverage(trimesh.util.concatenate([left, right]), left)
    assert missing_check['source_vertex_samples'] == 16
    assert missing_check['vertices_over_3_percent_extent'] == 8
    try:
        validate_coverage(missing_check)
    except RuntimeError:
        pass
    else:
        raise AssertionError('A missing half of the original model must be rejected')
    report = read_json(folder / 'report.json')
    source = Document.load(folder / 'coarse.glb')
    output = Document.load(folder / 'output.glb')
    assert digest(folder / 'coarse.glb') == report['input_sha256']
    assert digest(folder / 'output.glb') == report['output_sha256']
    before, after = visible_mesh(source), visible_mesh(output)
    geometry=(folder/'studio-blender/tessellate/triangles.obj' if report['details'].get('geometry_tessellation')
              else folder/'output.obj')
    obj = trimesh.load(geometry, force='mesh', process=False)
    # A UV atlas may duplicate/reorder vertices, but not change oriented faces.
    assert len(after.faces) == report['output_faces'] == len(obj.faces)
    # The atlas records the original OBJ positions as float32 GLB accessors.
    triangle_xyz = np.concatenate([obj.vertices[obj.faces].reshape(-1, 3),
                                   after.vertices[after.faces].reshape(-1, 3)]).astype(np.float32)
    _, inverse = np.unique(triangle_xyz, axis=0, return_inverse=True)
    first = inverse[:len(obj.faces) * 3].reshape(-1, 3)
    second = inverse[len(obj.faces) * 3:].reshape(-1, 3)
    first, second = oriented_keys(first), oriented_keys(second)
    np.testing.assert_array_equal(first[np.lexsort(first.T[::-1])], second[np.lexsort(second.T[::-1])])
    assert output.tree.get('materials') and output.tree.get('images')
    rng = np.random.default_rng(23)
    triangles = after.vertices[after.faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    indices = rng.choice(len(triangles), args.samples, p=area / area.sum())
    uv = rng.random((args.samples, 2))
    uv[uv.sum(1) > 1] = 1 - uv[uv.sum(1) > 1]
    bary = np.c_[1 - uv.sum(1), uv]
    points = np.einsum('ni,nij->nj', bary, triangles[indices])
    donor = Surface(source)
    actual = Surface(output)
    expected_values, distances = donor.sample(points)
    actual_values, _ = actual.sample(points)
    errors = {}
    for channel in expected_values:
        values = np.abs(actual_values[channel] - expected_values[channel])
        if channel == 'normal':
            values = np.degrees(np.arccos(np.clip((actual_values[channel] * expected_values[channel]).sum(1), -1, 1)))
        else:
            values = values.max(1)
        errors[channel] = {'median': float(np.median(values)), 'p95': float(np.percentile(values, 95)), 'max': float(values.max())}
    unexpected_black = ((expected_values['base'][:, :3].max(1) > .1)
                        & (actual_values['base'][:, :3].max(1) < .001))
    coverage = geometry_coverage(before, after)
    try:
        validate_coverage(coverage)
        coverage_passed = True
    except RuntimeError:
        coverage_passed = False
    result = {'status': 'checked', 'geometry_coverage_passed': coverage_passed,
        'input_faces': len(before.faces), 'output_faces': len(after.faces),
        'input_components': components(before), 'output_components': components(after),
        'quad_faces': report['details']['quad_faces'], 'polygons': report['details']['polygons'],
        'oriented_geometry_preserved_in_atlas': True, 'input_matches_worker_checksum': True,
        'samples': args.samples, 'unexpected_black_samples': int(unexpected_black.sum()),
        'projection_distance_p95': float(np.percentile(distances, 95)),
        'channel_errors_at_nearest_original_surface': errors,
        'original_surface_coverage': coverage,
        'missing_component_detection_verified': True,
        'notice': 'Numerical comparison only; inspect the saved multi-view renders for visual quality.'}
    atomic_json(folder / 'surface-check.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
