import copy
import zipfile

import numpy as np
import pytest
from PIL import Image
import trimesh

from test_workbench import testroot
from hi3d_workbench.common import digest
from hi3d_workbench.studio.gltf import Document
from hi3d_workbench.studio.join import join_static
from hi3d_workbench.studio.parts import apply_edits, export_scene, read_parts, write_parts
from hi3d_workbench.studio.parts_models import parameterized_mask_source
from hi3d_workbench.studio.scene import instances


def source_model(path, *, instance=False):
    mesh = trimesh.creation.icosphere(subdivisions=1)
    image = Image.fromarray(np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3))
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.random.default_rng(1).random((len(mesh.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(baseColorTexture=image,
                                                   roughnessFactor=.21, metallicFactor=.77))
    mesh.export(path)
    document = Document.load(path)
    if instance:
        node = next(index for index, value in enumerate(document.tree['nodes']) if 'mesh' in value)
        duplicate = copy.deepcopy(document.tree['nodes'][node])
        duplicate.pop('matrix', None)
        duplicate['translation'] = [3, 2, 1]
        duplicate['scale'] = [2, .5, 3]
        document.tree['scenes'][0]['nodes'].append(len(document.tree['nodes']))
        document.tree['nodes'].append(duplicate)
        document.save(path)
    return Document.load(path)


def textured_triangles(document):
    values = []
    for row in instances(document):
        xyz = row.world_positions()
        uv = document.accessor(row.definition['attributes']['TEXCOORD_0'])
        values.append(np.concatenate((xyz, uv), axis=1)[row.faces].reshape(len(row.faces), -1))
    values = np.concatenate(values)
    return values[np.lexsort(values.T[::-1])]


def test_split_keeps_instanced_geometry_uv_materials_and_embedded_pixels(tmp_path):
    source, target = tmp_path / 'source.glb', tmp_path / 'output.glb'
    before = source_model(source, instance=True)
    original_hash = digest(source)
    rows = instances(before)
    labels = [np.arange(len(row.faces), dtype=np.int32) % 2 for row in rows]
    metadata = write_parts(before, labels, {0: 'Left piece', 1: 'Right piece'}, target,
                           engine='validation fixture', source_sha256=original_hash)
    after = Document.load(target)
    np.testing.assert_array_equal(textured_triangles(before), textured_triangles(after))
    assert before.tree['materials'] == after.tree['materials']
    assert [bytes(before.view(image['bufferView'])) for image in before.tree['images']] == [
        bytes(after.view(image['bufferView'])) for image in after.tree['images']]
    assert metadata['faces'] == sum(len(row.faces) for row in rows)
    assert digest(source) == original_hash
    with zipfile.ZipFile(tmp_path / 'parts.zip') as archive:
        assert set(archive.namelist()) == {'assembly.glb', 'parts.json', 'parts/part_0000.glb', 'parts/part_0001.glb'}
        assert archive.read('assembly.glb') == target.read_bytes()
    _, loaded_labels, names = read_parts(target, tmp_path / 'parts.json')
    assert set(names) == {0, 1}
    assert sum(len(value) for value in loaded_labels) == metadata['faces']


def test_face_edits_merge_aliases_and_original_source_identity(tmp_path):
    source, target = tmp_path / 'source.glb', tmp_path / 'output.glb'
    doc = source_model(source)
    labels = [np.arange(len(row.faces), dtype=np.int32) % 3 for row in instances(doc)]
    write_parts(doc, labels, {0: 'A', 1: 'B', 2: 'C'}, target,
                engine='fixture', source_sha256=digest(source))
    doc, labels, names = read_parts(target, tmp_path / 'parts.json')
    row = instances(doc)[0]
    original_count = sum(len(value) for value in labels)
    edited, names = apply_edits(doc, labels, names, [
        {'operation': 'merge', 'parts': [0, 1], 'name': 'AB'},
        {'operation': 'rename', 'part': 1, 'name': 'Renamed AB'},
        {'operation': 'faces', 'mesh': row.mesh, 'primitive': row.primitive, 'faces': [0], 'target_part': 2}])
    assert names == {0: 'Renamed AB', 2: 'C'}
    assert edited[0][0] == 2 and labels[0][0] == 0
    assert sum(len(value) for value in edited) == original_count
    with pytest.raises(ValueError, match='超出'):
        apply_edits(doc, labels, {0: 'A', 1: 'B', 2: 'C'}, [
            {'operation': 'faces', 'mesh': row.mesh, 'primitive': 0, 'faces': [999999], 'target_part': 1}])
    with pytest.raises(ValueError, match='不一致'):
        read_parts(source, tmp_path / 'parts.json')


def test_new_part_can_split_selected_faces_and_be_exported(tmp_path):
    source, target = tmp_path / 'source.glb', tmp_path / 'split.glb'
    document = source_model(source)
    row = instances(document)[0]
    labels = [np.zeros(len(row.faces), dtype=np.int32)]
    edited, names = apply_edits(document, labels, {0: 'Body'}, [
        {'operation': 'create', 'part': 1, 'name': 'New section'},
        {'operation': 'faces', 'mesh': row.mesh, 'primitive': row.primitive,
         'faces': [0, 1, 2], 'target_part': 1}])
    result = write_parts(document, edited, names, target, engine='fixture', source_sha256=digest(source))
    assert {part['id']: part['faces'] for part in result['parts']} == {0: len(row.faces) - 3, 1: 3}
    with pytest.raises(ValueError, match='重复'):
        apply_edits(document, labels, {0: 'Body'}, [{'operation': 'create', 'part': 0, 'name': 'Duplicate'}])


def test_split_fans_out_morph_animation_channels(tmp_path):
    source, target = tmp_path / 'morph.glb', tmp_path / 'split.glb'
    doc = source_model(source)
    row = instances(doc)[0]
    delta = np.zeros_like(row.positions); delta[:, 1] = .1
    row.definition['targets'] = [{'POSITION': doc.add_accessor(delta, target=34962)}]
    doc.tree['meshes'][row.mesh]['weights'] = [0.0]
    doc.tree['nodes'][row.node]['weights'] = [0.0]
    times = doc.add_accessor(np.array([0, 1], dtype=np.float32))
    weights = doc.add_accessor(np.array([0, 1], dtype=np.float32))
    doc.tree['animations'] = [{'samplers': [{'input': times, 'output': weights}],
        'channels': [{'sampler': 0, 'target': {'node': row.node, 'path': 'weights'}}]}]
    export_scene(doc, [np.arange(len(row.faces), dtype=np.int32) % 2], {0: 'A', 1: 'B'}, target)
    loaded = Document.load(target)
    channels = loaded.tree['animations'][0]['channels']
    assert len(channels) == 2
    for channel in channels:
        node = loaded.tree['nodes'][channel['target']['node']]
        assert node['weights'] == [0.0]
        assert len(loaded.tree['meshes'][node['mesh']]['primitives'][0]['targets']) == 1


def test_static_join_preserves_each_material_and_image_payload(tmp_path):
    first, second, joined = [tmp_path / name for name in ('a.glb', 'b.glb', 'joined.glb')]
    a, b = source_model(first), source_model(second)
    b.tree['materials'][0]['pbrMetallicRoughness']['metallicFactor'] = .05
    b.save(second)
    join_static([(first, 2, 'A'), (second, 7, 'B')], joined)
    output = Document.load(joined)
    assert len(output.tree['materials']) == 2
    assert {row.definition['extras']['studio_part'] for row in instances(output)} == {2, 7}
    assert [mat['pbrMetallicRoughness']['metallicFactor'] for mat in output.tree['materials']] == [.77, .05]
    for row in instances(output):
        material = output.tree['materials'][row.definition['material']]
        texture = output.tree['textures'][material['pbrMetallicRoughness']['baseColorTexture']['index']]
        image = output.tree['images'][texture['source']]
        assert bytes(output.view(image['bufferView'])) == bytes(a.view(a.tree['images'][0]['bufferView']))


def test_upstream_budget_fix_uses_requested_values_and_refuses_changed_source():
    code = 'def mesh_sam(point_num, prompt_num):\n    point_num = 100000\n    prompt_num = 400\n    return point_num, prompt_num\n'
    namespace = {}
    exec(parameterized_mask_source(code), namespace)
    assert namespace['mesh_sam'](12000, 16) == (12000, 16)
    with pytest.raises(RuntimeError, match='版本不匹配'):
        parameterized_mask_source('def changed(): pass')
