from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from hi3d_workbench.studio.gltf import Document
from hi3d_workbench.studio.mesh import distribute_budget, simplify


def test_budget_preserves_small_material_groups():
    assert distribute_budget([1000, 2, 10], 20) == [18, 1, 1]
    assert sum(distribute_budget([500, 500, 500], 101)) == 101
    with pytest.raises(ValueError):
        distribute_budget([1, 1, 1], 2)


def textured_sphere(path):
    sphere = trimesh.creation.uv_sphere(count=[40, 40])
    xyz = sphere.vertices
    uv = np.column_stack((np.arctan2(xyz[:, 1], xyz[:, 0]) / (2 * np.pi) + .5,
                          np.arccos(np.clip(xyz[:, 2], -1, 1)) / np.pi))
    image = Image.fromarray(np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3))
    sphere.visual = trimesh.visual.TextureVisuals(uv=uv,
        material=trimesh.visual.material.PBRMaterial(baseColorTexture=image, roughnessFactor=.27, metallicFactor=.63))
    sphere.export(path)


def test_subset_preserves_actual_uv_pixels_and_material(tmp_path):
    source, target = tmp_path / 'source.glb', tmp_path / 'subset.glb'
    textured_sphere(source)
    doc = Document.load(source)
    tree = Document.load(source).tree
    old_images = [bytes(doc.view(im['bufferView'])) for im in tree['images']]
    primitive = next(doc.primitives())[2]
    old_positions = doc.accessor(primitive['attributes']['POSITION'])
    old_uv = doc.accessor(primitive['attributes']['TEXCOORD_0'])
    faces = doc.accessor(primitive['indices']).reshape(-1, 3)
    selection = np.arange(len(faces) // 2)
    used = np.unique(faces[selection])
    doc.subset(primitive, selection)
    doc.save(target)
    loaded = Document.load(target)
    prim = next(loaded.primitives())[2]
    np.testing.assert_array_equal(loaded.accessor(prim['attributes']['POSITION']), old_positions[used])
    np.testing.assert_array_equal(loaded.accessor(prim['attributes']['TEXCOORD_0']), old_uv[used])
    assert loaded.tree['materials'] == tree['materials']
    assert [bytes(loaded.view(im['bufferView'])) for im in loaded.tree['images']] == old_images
    assert target.stat().st_size < source.stat().st_size


def test_rejects_external_buffers_before_reading(tmp_path):
    source = tmp_path / 'bad.glb'
    doc = Document({'asset': {'version': '2.0'}, 'buffers': [{'uri': 'file:///secret'}]})
    # save intentionally replaces the buffers: build a raw external reference.
    import json, struct
    payload = json.dumps(doc.tree).encode(); payload += b' ' * (-len(payload) % 4)
    source.write_bytes(struct.pack('<4sII', b'glTF', 2, 20 + len(payload)) + struct.pack('<I4s', len(payload), b'JSON') + payload)
    with pytest.raises(ValueError, match='内嵌'):
        Document.load(source)


@pytest.mark.skipif(not Path('runtime-assets/meshoptimizer.dll').is_file(), reason='native backend is optional')
def test_real_simplifier_reduces_mesh_without_replacing_textures(tmp_path):
    source, target = tmp_path / 'source.glb', tmp_path / 'low.glb'
    textured_sphere(source)
    before = Document.load(source)
    images = [bytes(before.view(im['bufferView'])) for im in before.tree['images']]
    report = simplify(source, target, target_faces=800, max_error=.05,
                      library_path=Path('runtime-assets/meshoptimizer.dll').resolve())
    assert report['output_faces'] < report['input_faces']
    assert report['output_faces'] <= 800
    after = Document.load(target)
    assert [bytes(after.view(im['bufferView'])) for im in after.tree['images']] == images
    assert before.tree['materials'] == after.tree['materials']
    assert target.stat().st_size < source.stat().st_size


@pytest.mark.skipif(not Path('runtime-assets/meshoptimizer.dll').is_file(), reason='native backend is optional')
def test_scene_budget_counts_repeated_mesh_instances(tmp_path):
    from hi3d_workbench.studio.scene import instances
    from hi3d_workbench.studio.skeleton import inspect
    source, target = tmp_path / 'instances.glb', tmp_path / 'low.glb'
    trimesh.creation.icosphere(subdivisions=3).export(source)
    doc = Document.load(source)
    original = instances(doc)[0]
    clone = {'mesh': original.mesh, 'translation': [3, 2, 1]}
    doc.tree['scenes'][0]['nodes'].append(len(doc.tree['nodes']))
    doc.tree['nodes'].append(clone)
    doc.save(source)
    count = sum(len(row.faces) for row in instances(Document.load(source)))
    assert count == 2560
    report = simplify(source, target, target_faces=800, max_error=.1,
                      library_path=Path('runtime-assets/meshoptimizer.dll').resolve())
    result = Document.load(target)
    assert report['input_faces'] == count
    assert report['output_faces'] == sum(len(row.faces) for row in instances(result)) <= 800
    assert inspect(target)['faces'] == report['output_faces']
    assert clone['translation'] == result.tree['nodes'][-1]['translation']
