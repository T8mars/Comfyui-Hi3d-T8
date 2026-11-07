import numpy as np

from test_workbench import testroot
from test_studio_retarget import human_rig
from hi3d_workbench.studio.humanoid import identify
from hi3d_workbench.studio.contact_skin import attachment_strength,contact_mask,separated_weights


def test_shoulder_blend_is_kept_but_a_distal_arm_to_torso_contact_can_be_separated():
    rig=human_rig();roles=identify(rig)['roles'];points=rig.worlds[:,:3,3]
    probe=np.array([points[roles['left_upper_arm']],points[roles['hips']]-[0,.3,0]])
    strength=attachment_strength(rig,probe)
    mass=np.zeros((2,5));mass[:,0]=.6;mass[:,1]=.4
    np.testing.assert_array_equal(contact_mask(mass,strength,.05),[False,True])
    weights=np.array([[.6,.4,0,0],[.6,.4,0,0]])
    groups=np.array([[0,1,0,0],[0,1,0,0]])
    changed=separated_weights(weights,groups,0,strength,.05)
    np.testing.assert_allclose(changed[0],weights[0])
    np.testing.assert_allclose(changed[1],[1,0,0,0])


def test_pelvis_blend_is_kept_while_a_lower_left_right_leg_bridge_is_separated():
    rig=human_rig();roles=identify(rig)['roles'];points=rig.worlds[:,:3,3]
    hips=points[roles['hips']]
    strength=attachment_strength(rig,np.array([hips,hips-[0,.6,0]]))
    mass=np.zeros((2,5));mass[:,3]=.6;mass[:,4]=.4
    np.testing.assert_array_equal(contact_mask(mass,strength,.05),[False,True])
    weights=np.array([[.6,.4,0,0],[.6,.4,0,0]])
    groups=np.array([[3,4,0,0],[3,4,0,0]])
    changed=separated_weights(weights,groups,3,strength,.05)
    np.testing.assert_allclose(changed[0],weights[0])
    np.testing.assert_allclose(changed[1],[1,0,0,0])
