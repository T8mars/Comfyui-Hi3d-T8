import numpy as np
import pytest

from test_workbench import testroot
from test_studio_retarget import human_rig
from test_studio_skin import rig_two
from hi3d_workbench.studio.motion_constraints import validate_controls,fitted_pose,transfer_controls
from hi3d_workbench.studio.animation import Motion
from hi3d_workbench.studio import skin


def test_controls_reject_stale_poses_bad_times_and_duplicate_frames():
    valid=validate_controls([{'type':'root','time':1,'x':2,'z':-1}],2,2)
    assert valid[0]['frame']==30
    for controls in ([{'type':'root','time':float('nan'),'x':0,'z':0}],
                     [{'type':'root','time':3,'x':0,'z':0}],
                     [{'type':'root','time':1,'x':float('inf'),'z':0}],
                     [{'type':'pose','time':1,'joints':[[0,0,0]]}],
                     [{'type':'other','time':1}]):
        with pytest.raises(ValueError):validate_controls(controls,2,2)
    with pytest.raises(ValueError,match='重复'):
        validate_controls([{'type':'root','time':1,'x':0,'z':0},{'type':'root','time':1.001,'x':1,'z':0}],2,2)


def test_path_positions_follow_character_axes_and_manual_motion_scale():
    source,target=human_rig(90),human_rig()
    controls=validate_controls([{'type':'root','time':1,'x':1,'z':0}],2,len(source.names))
    paths,poses=transfer_controls(controls,source,target,{})
    np.testing.assert_allclose(paths[30],[0,1],atol=1e-7)
    assert not poses
    paths,_=transfer_controls(controls,source,target,{'motion_scale':2})
    np.testing.assert_allclose(paths[30],[0,.5],atol=1e-7)


def test_key_pose_transfer_fits_lengths_and_transfers_root_offset():
    source=rig_two()
    target=skin.oriented_rig([[0.,0,0],[0,2,0]],[-1,0],['hips','hand'])
    points=np.array([[.5,0,0],[1.5,0,0]])
    fitted=fitted_pose(source,points)
    np.testing.assert_allclose(fitted[:,:3,3],points,atol=1e-7)
    controls=validate_controls([{'type':'pose','time':1,'joints':points.tolist()}],2,2)
    _,poses=transfer_controls(controls,source,target,{})
    np.testing.assert_allclose(poses[0]['worlds'][:,:3,3],[[1,0,0],[3,0,0]],atol=2e-7)
    distorted=np.array([[.5,0,0],[10,0,0]])
    actual=fitted_pose(source,distorted)
    assert np.linalg.norm(actual[1,:3,3]-actual[0,:3,3])==pytest.approx(1)


def test_motion_contact_labels_are_bounded_and_match_every_source_frame():
    rig=rig_two();times=np.array([0.,1.]);worlds=np.repeat(rig.worlds[None],2,axis=0)
    assert Motion(rig,times,worlds,'stand',foot_contacts=np.ones((2,4))).validate()
    for contacts in (np.ones((2,6)),np.full((2,4),float('nan')),np.full((2,4),1.1)):
        with pytest.raises(ValueError,match='接触'):
            Motion(rig,times,worlds,'bad',foot_contacts=contacts).validate()
