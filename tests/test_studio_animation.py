import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from test_workbench import testroot
from test_studio_parts import source_model
from test_studio_skin import rig_two, provider
from hi3d_workbench.studio.animation import interpolate, sample_glb, write_clip
from hi3d_workbench.studio.skin import bind, extract
from hi3d_workbench.studio.gltf import Document


def test_sampling_uses_world_hierarchy_and_real_quaternion_rotation(tmp_path):
    document, _ = bind(source_model(tmp_path / 'source.glb'), rig_two(), provider)
    rig = extract(document)
    times = document.add_accessor(np.array([0., 1.], dtype=np.float32))
    translation = document.add_accessor(np.array([[0., 0., 0.], [2., 0., 0.]], dtype=np.float32))
    rotation = document.add_accessor(Rotation.from_euler('z', [0, 90], degrees=True).as_quat())
    document.tree['animations'] = [{'name': 'move', 'samplers': [{'input': times, 'output': translation},
        {'input': times, 'output': rotation}], 'channels': [
        {'sampler': 0, 'target': {'node': rig.nodes[0], 'path': 'translation'}},
        {'sampler': 1, 'target': {'node': rig.nodes[1], 'path': 'rotation'}}]}]
    motion = sample_glb(document)
    assert motion.worlds.shape == (31, 2, 4, 4)
    np.testing.assert_allclose(motion.worlds[15, 0, :3, 3], [1, 0, 0])
    np.testing.assert_allclose(motion.worlds[15, 1, :3, 3], [1, 1, 0])
    np.testing.assert_allclose(motion.worlds[-1, 1, :3, :3], Rotation.from_euler('z', 90, degrees=True).as_matrix(), atol=1e-7)


def test_gltf_step_cubic_and_shortest_quaternion_interpolation():
    times = np.array([0., 1.])
    values = np.array([[0., 0., 0.], [2., 0., 0.]])
    np.testing.assert_array_equal(interpolate(times, values, [0, .9, 1], 'translation', 'STEP')[:, 0], [0, 0, 2])
    cubic = np.array([[0, 0, 0], [0, 0, 0], [2, 0, 0], [2, 0, 0], [2, 0, 0], [0, 0, 0]], dtype=float)
    np.testing.assert_allclose(interpolate(times, cubic, [.25, .5, .75], 'translation', 'CUBICSPLINE')[:, 0], [.5, 1, 1.5])
    quaternion = Rotation.from_euler('y', [0, 170], degrees=True).as_quat(); quaternion[1] *= -1
    expected = Rotation.from_euler('y', 85, degrees=True).as_matrix()
    actual = interpolate(times, quaternion, [.5], 'rotation')
    np.testing.assert_allclose(Rotation.from_quat(actual).as_matrix()[0], expected, atol=1e-7)
    for invalid in ([0, 0], [1, .5], [-1, 1]):
        with pytest.raises(ValueError, match='时间'):
            interpolate(invalid, values, [.5], 'translation')


def test_writing_animation_keeps_geometry_images_and_original_source(tmp_path):
    document, _ = bind(source_model(tmp_path / 'source.glb', instance=True), rig_two(), provider)
    rig = extract(document)
    rotations = np.repeat(np.array([0., 0., 0., 1.])[None, None], 4, axis=0)
    rotations = np.repeat(rotations, 2, axis=1)
    rotations[2] *= -1
    translation = np.repeat(np.array([[0, 0, 0], [0, 1, 0]], dtype=float)[None], 4, axis=0)
    translation[:, 0, 2] = np.linspace(0, 1, 4)
    output = write_clip(document, rig, [0, 1/3, 2/3, 1], rotations, translation, 'single actor')
    output.save(tmp_path / 'animated.glb')
    assert 'animations' not in document.tree
    assert output.tree['materials'] == document.tree['materials']
    assert [bytes(output.view(im['bufferView'])) for im in output.tree['images']] == [
        bytes(document.view(im['bufferView'])) for im in document.tree['images']]
    np.testing.assert_allclose(sample_glb(output).worlds[-1, 0, :3, 3], [0, 0, 1])
    with pytest.raises(ValueError, match='旋转'):
        write_clip(document, rig, [0, 1/3, 2/3, 1], np.zeros_like(rotations), translation, 'bad')


def test_rejects_duplicate_animation_channels_instead_of_silent_last_wins(tmp_path):
    document, _ = bind(source_model(tmp_path / 'source.glb'), rig_two(), provider)
    rig = extract(document)
    times = document.add_accessor(np.array([0., 1.], dtype=np.float32))
    values = document.add_accessor(np.zeros((2, 3), dtype=np.float32))
    channel = {'sampler': 0, 'target': {'node': rig.nodes[0], 'path': 'translation'}}
    document.tree['animations'] = [{'samplers': [{'input': times, 'output': values}], 'channels': [channel, channel]}]
    with pytest.raises(ValueError, match='重复'):
        sample_glb(document)


@pytest.mark.parametrize('duration,fps,frames', [
    (.1,30,4),(.3,30,10),(.3001,30,11),(.125,24,4),(.35,60,22),
])
def test_saved_float32_timestamps_do_not_add_a_spurious_sampling_interval(tmp_path,duration,fps,frames):
    document,_=bind(source_model(tmp_path/'source.glb'),rig_two(),provider)
    rig=extract(document)
    times=document.add_accessor(np.array([0.,duration],dtype=np.float32))
    values=document.add_accessor(np.array([[0.,0.,0.],[1.,0.,0.]],dtype=np.float32))
    document.tree['animations']=[{'samplers':[{'input':times,'output':values}],
        'channels':[{'sampler':0,'target':{'node':rig.nodes[0],'path':'translation'}}]}]
    path=tmp_path/'short-clip.glb';document.save(path)
    actual=sample_glb(Document.load(path),fps=fps)
    assert len(actual.times)==frames
    assert actual.times[-1]==float(np.float32(duration))
    np.testing.assert_allclose(actual.worlds[-1,0,:3,3],[1,0,0])
