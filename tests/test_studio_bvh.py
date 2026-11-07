import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from test_workbench import testroot
from hi3d_workbench.studio.bvh import load


BVH = '''HIERARCHY
ROOT Hips
{
 OFFSET 0 0 0
 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
 JOINT Hand
 {
  OFFSET 0 100 0
  CHANNELS 3 Zrotation Xrotation Yrotation
  End Site { OFFSET 0 10 0 }
 }
}
MOTION
Frames: 2
Frame Time: 0.03333333333333333
0 0 0 0 0 0 0 0 0
100 0 0 90 30 20 40 0 0
'''


def test_bvh_channel_order_units_and_parent_rotation_are_not_guessed(tmp_path):
    path = tmp_path / 'punch.bvh'; path.write_text(BVH, encoding='utf-8')
    motion = load(path, scale=.01)
    assert motion.rig.names == ['Hips', 'Hand', 'Hand_EndSite']
    root = Rotation.from_euler('ZXY', [90, 30, 20], degrees=True).as_matrix()
    np.testing.assert_allclose(motion.worlds[1, 0, :3, :3] / .01, root, atol=1e-7)
    np.testing.assert_allclose(motion.worlds[1, 1, :3, 3], [1, 0, 0] + root @ [0, 1, 0], atol=1e-7)
    expected_hand = root @ Rotation.from_euler('z', 40, degrees=True).as_matrix()
    np.testing.assert_allclose(motion.worlds[1, 1, :3, :3] / .01, expected_hand, atol=1e-7)
    z_up = load(path, scale=.01, up_axis='Z')
    np.testing.assert_allclose(z_up.rig.worlds[1, :3, 3], [0, 0, -1], atol=1e-7)


def test_bvh_refuses_incomplete_nan_duplicate_channels_and_extra_motion(tmp_path):
    path = tmp_path / 'bad.bvh'
    for bad in (BVH.replace('Frames: 2', 'Frames: 3'), BVH.replace('100 0 0 90', 'nan 0 0 90'),
                BVH.replace('Zrotation Xrotation Yrotation', 'Zrotation Zrotation Yrotation'), BVH + 'not-a-number'):
        path.write_text(bad, encoding='utf-8')
        with pytest.raises(ValueError):
            load(path)
