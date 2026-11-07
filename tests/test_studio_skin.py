import copy
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from test_workbench import testroot
from test_studio_parts import source_model
from hi3d_workbench.studio.gltf import Document
from hi3d_workbench.studio.scene import instances
from hi3d_workbench.studio.skin import Rig, bind, edit, extract, oriented_rig, pack_weights, posed_vertices, validate, validate_preservation
from hi3d_workbench.studio.rigging import SampledWeights
from hi3d_workbench.studio.rig_skintokens import local_config


def rig_two():
    return oriented_rig(np.array([[0, 0, 0], [0, 1, 0]], dtype=float), [-1, 0], ['hips', 'hand'])


def provider(points):
    values = np.zeros((len(points), 2), dtype=np.float32)
    values[:, 1] = 1
    return values


def test_reskin_validation_checks_animation_values_after_accessor_compaction(tmp_path):
    from test_studio_retarget import animated_two
    from hi3d_workbench.studio.skin import validate_existing_rig
    source=animated_two(tmp_path)
    result,_=bind(source,extract(source),provider,use_existing=True)
    target=tmp_path/'existing.glb';result.save(target)
    report=validate_existing_rig(source,Document.load(target))
    assert report['animation_clips_preserved']==1 and report['animation_channels_preserved']==2
    result.tree['animations']=[]
    with pytest.raises(ValueError,match='动画片段'):
        validate_existing_rig(source,result)


def test_motion_validation_allows_rigid_accessories_but_requires_a_bound_body(tmp_path):
    result,_=bind(source_model(tmp_path/'source.glb'),rig_two(),provider)
    accessory=__import__('copy').deepcopy(result.tree['meshes'][0])
    for primitive in accessory['primitives']:
        primitive['attributes']={key:value for key,value in primitive['attributes'].items()
                                 if not key.startswith(('JOINTS_','WEIGHTS_'))}
    mesh_index=len(result.tree['meshes']);result.tree['meshes'].append(accessory)
    node_index=len(result.tree['nodes']);result.tree['nodes'].append({'mesh':mesh_index,'name':'rigid prop'})
    result.tree['nodes'][extract(result).nodes[-1]].setdefault('children',[]).append(node_index)
    assert validate(result,allow_static=True)
    with pytest.raises(ValueError,match='未绑定'):
        validate(result)
    for node in result.tree['nodes']:
        if 'skin' in node:
            node.pop('skin');node['mesh']=mesh_index
    with pytest.raises(ValueError,match='没有实际绑定'):
        validate(result,allow_static=True)


def test_binding_keeps_real_world_geometry_uv_and_pbr_on_transformed_instances(tmp_path):
    original = source_model(tmp_path / 'source.glb', instance=True)
    before = [row.world_positions() for row in instances(original)]
    result, report = bind(original, rig_two(), provider)
    result.save(tmp_path / 'rigged.glb')
    loaded = Document.load(tmp_path / 'rigged.glb')
    assert validate(loaded)
    assert report['joints'] == 2 and report['max_discarded_weight'] == 0
    for expected, actual in zip(before, posed_vertices(loaded)):
        np.testing.assert_allclose(actual, expected, atol=2e-6)
    for left, right in zip(instances(original), instances(loaded)):
        np.testing.assert_array_equal(original.accessor(left.definition['attributes']['TEXCOORD_0']),
                                      loaded.accessor(right.definition['attributes']['TEXCOORD_0']))
    assert original.tree['materials'] == loaded.tree['materials']
    assert [bytes(original.view(im['bufferView'])) for im in original.tree['images']] == [
        bytes(loaded.view(im['bufferView'])) for im in loaded.tree['images']]
    assert 'skins' not in original.tree


def test_joint_rotation_moves_vertices_about_the_actual_world_pivot(tmp_path):
    source = source_model(tmp_path / 'source.glb', instance=True)
    result, _ = bind(source, rig_two(), provider)
    skeleton = extract(result)
    joint = result.tree['nodes'][skeleton.nodes[1]]
    joint['rotation'] = Rotation.from_euler('z', 90, degrees=True).as_quat().tolist()
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    pivot = np.array([0., 1., 0.])
    for row, actual in zip(instances(source), posed_vertices(result)):
        expected = (row.world_positions() - pivot) @ rotation.T + pivot
        np.testing.assert_allclose(actual, expected, atol=2e-6)


def test_editing_joint_keeps_rest_shape_even_when_old_bind_matrix_is_not_identity(tmp_path):
    source = source_model(tmp_path / 'source.glb')
    rigged, _ = bind(source, rig_two(), provider)
    skeleton = extract(rigged)
    rigged.tree['nodes'][skeleton.nodes[1]]['rotation'] = Rotation.from_euler('z', 35, degrees=True).as_quat().tolist()
    before = posed_vertices(rigged)
    result, info = edit(rigged, [{'index': 1, 'position': [.2, 1.25, .4]}])
    for old, actual in zip(before, posed_vertices(result)):
        np.testing.assert_allclose(actual, old, atol=2e-6)
    np.testing.assert_allclose(extract(result).worlds[1, :3, 3], [.2, 1.25, .4])
    assert info['edited_joints'] == 1


def test_moving_one_rest_joint_does_not_silently_move_unedited_child_markers(tmp_path):
    source = source_model(tmp_path / 'source.glb')
    first, _ = bind(source, rig_two(), provider)
    before = extract(first).worlds.copy()
    result, _ = edit(first, [{'index': 0, 'position': [.2, 0, .1]}])
    after = extract(result).worlds
    np.testing.assert_allclose(after[0, :3, 3], [.2, 0, .1])
    np.testing.assert_allclose(after[1], before[1])
    assert validate_preservation(first, result)['initial_geometry_preserved']


def test_existing_skeleton_and_animation_targets_are_retained(tmp_path):
    source = source_model(tmp_path / 'source.glb')
    first, _ = bind(source, rig_two(), provider)
    rig = extract(first)
    times = first.add_accessor(np.array([0., 1.], dtype=np.float32))
    rotations = first.add_accessor(Rotation.from_euler('z', [0, 90], degrees=True).as_quat())
    first.tree['animations'] = [{'name': 'wave', 'samplers': [{'input': times, 'output': rotations}],
        'channels': [{'sampler': 0, 'target': {'node': rig.nodes[1], 'path': 'rotation'}}]}]
    result, info = bind(first, rig, provider, use_existing=True)
    assert extract(result).nodes == rig.nodes
    assert result.tree['animations'] == first.tree['animations']
    assert info['existing_skeleton'] and info['removed_animation_clips'] == 0


def test_reskinning_an_existing_non_bind_pose_does_not_snap_the_mesh_back(tmp_path):
    source = source_model(tmp_path / 'source.glb', instance=True)
    first, _ = bind(source, rig_two(), provider)
    rig = extract(first)
    first.tree['nodes'][rig.nodes[1]]['rotation'] = Rotation.from_euler('z', 35, degrees=True).as_quat().tolist()
    before = posed_vertices(first)
    second, _ = bind(first, extract(first), provider, use_existing=True)
    for expected, actual in zip(before, posed_vertices(second)):
        np.testing.assert_allclose(actual, expected, atol=2e-6)
    assert extract(first).nodes == extract(second).nodes
    assert validate_preservation(first, second)['initial_geometry_preserved']


def test_preservation_check_detects_changed_pbr_even_when_geometry_passes(tmp_path):
    source = source_model(tmp_path / 'source.glb')
    result, _ = bind(source, rig_two(), provider)
    assert validate_preservation(source, result)['embedded_images_preserved'] > 0
    result.tree['materials'][0]['pbrMetallicRoughness']['roughnessFactor'] = .01
    with pytest.raises(RuntimeError, match='PBR'):
        validate_preservation(source, result)


def test_four_influence_packing_reports_discarded_mass_and_refuses_zero_or_bad_weights():
    joints, weights, discarded = pack_weights(np.array([[.4, .3, .15, .1, .05]], dtype=np.float32), 5)
    assert set(joints[0]) == {0, 1, 2, 3}
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    np.testing.assert_allclose(discarded, [.05], atol=1e-6)
    for invalid in (np.zeros((2, 3)), np.full((2, 3), np.nan), np.full((2, 3), -1)):
        with pytest.raises(ValueError):
            pack_weights(invalid, 3)


def test_bad_rig_cycle_and_invalid_joint_indices_are_rejected(tmp_path):
    with pytest.raises(ValueError, match='循环'):
        Rig(['a', 'b'], np.array([1, 0]), np.repeat(np.eye(4)[None], 2, axis=0)).validate()
    source = source_model(tmp_path / 'source.glb')
    result, _ = bind(source, rig_two(), provider)
    primitive = instances(result)[0].definition
    primitive['attributes']['JOINTS_0'] = result.add_accessor(np.full((len(instances(result)[0].positions), 4), 99, dtype=np.uint16))
    with pytest.raises(ValueError, match='超出范围'):
        validate(result)


def test_weight_interpolation_uses_original_world_coordinates_and_eight_neighbours():
    points = np.column_stack((np.arange(10), np.zeros(10), np.zeros(10))).astype(float)
    values = np.column_stack((np.linspace(0, 1, 10), np.linspace(1, 0, 10))).astype(np.float32)
    matrix = np.eye(4); matrix[:3, :3] *= 2; matrix[:3, 3] = [3, 0, 0]
    transfer = SampledWeights(points, values, matrix)
    query = np.array([[.25, 0, 0], [1., 0, 0]])
    actual = transfer(query)
    for row, world in zip(actual, query):
        model = world * 2 + [3, 0, 0]
        distances = np.linalg.norm(points - model, axis=1)
        nearest = np.argsort(distances, kind='stable')[:8]
        weights = 1 / (distances[nearest] + 1e-8); weights /= weights.sum()
        np.testing.assert_allclose(row, weights @ values[nearest], atol=1e-7)
    single = SampledWeights(points[:1], values[:1], np.eye(4))
    np.testing.assert_array_equal(single(query), np.repeat(values[:1], 2, axis=0))


def test_local_skeleton_configs_are_independent_of_cwd_and_cannot_escape_source(tmp_path):
    config = tmp_path / 'configs' / 'skeleton'; config.mkdir(parents=True)
    (config / 'human.yaml').write_text('parts: {}', encoding='utf-8')
    value = {'nested': [{'skeleton_path': {'human': './configs/skeleton/human.yaml'}}]}
    actual = local_config(value, tmp_path)
    assert actual['nested'][0]['skeleton_path']['human'] == str((config / 'human.yaml').resolve())
    assert value['nested'][0]['skeleton_path']['human'].startswith('./')
    with pytest.raises(RuntimeError, match='超出'):
        local_config({'skeleton_path': {'bad': '../outside.yaml'}}, tmp_path)
