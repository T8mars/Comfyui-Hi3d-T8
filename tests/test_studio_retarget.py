import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from test_workbench import testroot
from test_studio_parts import source_model
from test_studio_skin import rig_two, provider
from hi3d_workbench.studio import skin
from hi3d_workbench.studio.animation import sample_glb
from hi3d_workbench.studio.humanoid import identify
from hi3d_workbench.studio.retarget import apply, between, fk, solve_leg, contact_mask


def animated_two(tmp_path):
    doc,_=skin.bind(source_model(tmp_path/'source.glb'),rig_two(),provider)
    rig=skin.extract(doc)
    ti=doc.add_accessor(np.array([0.,1.],dtype=np.float32))
    translation=doc.add_accessor(np.array([[0.,0,0],[2.,0,0]],dtype=np.float32))
    rotation=doc.add_accessor(Rotation.from_euler('z',[0,90],degrees=True).as_quat())
    doc.tree['animations']=[{'name':'turn','samplers':[{'input':ti,'output':translation},{'input':ti,'output':rotation}],
        'channels':[{'sampler':0,'target':{'node':rig.nodes[0],'path':'translation'}},
                    {'sampler':1,'target':{'node':rig.nodes[1],'path':'rotation'}}]}]
    return doc


def test_same_rig_retarget_reconstructs_all_world_poses_and_retains_pbr(tmp_path):
    source=animated_two(tmp_path);motion=sample_glb(source)
    output,report=apply(source,motion,{'foot_lock':False})
    actual=sample_glb(output)
    np.testing.assert_allclose(actual.worlds,motion.worlds,atol=2e-7)
    assert report['mapped_joints']==2
    assert output.tree['materials']==source.tree['materials']
    assert [bytes(output.view(i['bufferView'])) for i in output.tree['images']]==[
        bytes(source.view(i['bufferView'])) for i in source.tree['images']]


def test_proportion_speed_and_in_place_are_actual_animation_changes(tmp_path):
    source=animated_two(tmp_path);motion=sample_glb(source)
    rig=skin.oriented_rig([[0.,0,0],[0,2,0]],[-1,0],['hips','hand'])
    target,_=skin.bind(source_model(tmp_path/'target.glb'),rig,provider)
    output,report=apply(target,motion,{'speed':2,'foot_lock':False})
    pose=sample_glb(output)
    assert report['scale']==pytest.approx(2)
    assert pose.times[-1]==pytest.approx(.5)
    np.testing.assert_allclose(pose.worlds[-1,0,:3,3],[4,0,0],atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(pose.worlds[: ,1,:3,3]-pose.worlds[:,0,:3,3],axis=1),2,atol=1e-7)
    inplace,_=apply(target,motion,{'root_motion':'in_place','foot_lock':False})
    np.testing.assert_allclose(sample_glb(inplace).worlds[:,0,:3,3],0,atol=1e-7)


def human_rig(angle=0):
    points=np.array([[0,1,0],[0,1.2,0],[0,1.5,0],[0,1.6,0],[0,1.8,0],
        [.2,1.5,0],[.5,1.3,0],[.8,1.1,0],[-.2,1.5,0],[-.5,1.3,0],[-.8,1.1,0],
        [.12,.9,0],[.12,.5,.05],[.12,.1,0],[.12,.05,.2],[-.12,.9,0],[-.12,.5,.05],[-.12,.1,0],[-.12,.05,.2]])
    rotation=Rotation.from_euler('y',angle,degrees=True).as_matrix()
    points=points@rotation.T
    parents=[-1,0,1,2,3,2,5,6,2,8,9,0,11,12,13,0,15,16,17]
    return skin.oriented_rig(points,parents,[f'bone_{i}' for i in range(19)])


def test_geometry_roles_work_with_generic_names_and_rotated_human_skeletons():
    for angle in (0,90,180):
        rig=human_rig(angle);result=identify(rig);roles=result['roles']
        assert {key:roles[key] for key in ('hips','head','left_upper_arm','left_forearm','right_hand','left_thigh','right_foot')}=={
            'hips':0,'head':4,'left_upper_arm':5,'left_forearm':6,'right_hand':10,'left_thigh':11,'right_foot':17}
        expected=Rotation.from_euler('y',angle,degrees=True).apply([0.,0,1])
        assert np.dot(result['basis'][:,2],expected)>.99
    with pytest.raises(ValueError,match='同一个'):
        identify(human_rig(),{'left_hand':'bone_7','right_hand':'bone_7'})


def cartoon_finger_rig():
    import json
    from pathlib import Path
    data=json.loads((Path(__file__).parent/'fixtures/skintokens-cartoon-finger-chain.json').read_text(encoding='utf-8'))
    return skin.oriented_rig(data['positions'],data['parents'],data['names']),data['expected_roles']


@pytest.mark.parametrize('angle,scale',[(0,1),(90,.01),(180,10)])
def test_actual_cartoon_fingers_are_not_mapped_as_elbows_or_wrists(angle,scale):
    rig,expected=cartoon_finger_rig()
    points=Rotation.from_euler('y',angle,degrees=True).apply(rig.worlds[:,:3,3])*scale
    transformed=skin.oriented_rig(points,rig.parents,rig.names)
    roles=identify(transformed)['roles']
    assert {role:roles[role] for role in expected}==expected


def test_cartoon_short_finger_heuristic_keeps_explicit_mapping_and_names():
    rig,_=cartoon_finger_rig()
    assert identify(rig,{'left_hand':'bone_12'})['roles']['left_hand']==12
    rig.names[12]='LeftHand'
    rig.names[11]='LeftForeArm'
    roles=identify(rig)['roles']
    assert roles['left_hand']==12 and roles['left_forearm']==11


def test_long_distal_segments_do_not_trigger_the_short_finger_heuristic():
    rig,_=cartoon_finger_rig()
    points=rig.worlds[:,:3,3].copy()
    for node in (10,11,12):
        points[node]=points[node-1]+[.14,.02,0]
    rig=skin.oriented_rig(points,rig.parents,rig.names)
    assert identify(rig)['roles']['left_hand']==12


def test_saved_motion_drives_the_actual_cartoon_elbow_instead_of_fingers(tmp_path):
    from hi3d_workbench.studio.animation import Motion,rotations
    from hi3d_workbench.studio.motion_constraints import skeleton_document
    from hi3d_workbench.studio.gltf import Document
    source=human_rig()
    worlds=np.repeat(source.worlds[None],2,axis=0)
    rotation=Rotation.from_euler('z',80,degrees=True).as_matrix()
    pivot=source.worlds[6,:3,3]
    delta=np.eye(4);delta[:3,:3]=rotation;delta[:3,3]=pivot-rotation@pivot
    worlds[1,[6,7]]=delta@worlds[1,[6,7]]
    motion=Motion(source,np.array([0.,1.]),worlds,'elbow raise').validate()
    target,_=cartoon_finger_rig()
    output,details=apply(skeleton_document(target),motion,{'foot_lock':False})
    path=tmp_path/'cartoon-elbow.glb';output.save(path)
    actual=sample_glb(Document.load(path))
    assert details['target_mapping']['left_forearm']=='bone_8'
    before,after=rotations(actual.worlds[[0,-1],8])
    assert Rotation.from_matrix(after@before.T).magnitude()>np.deg2rad(75)
    assert np.linalg.norm(actual.worlds[-1,9,:3,3]-actual.worlds[0,9,:3,3])>.1


def test_two_bone_ik_hits_reachable_target_without_stretching_bones():
    parents=np.array([-1,0,1]);order=[0,1,2]
    translation=np.array([[0.,2,0],[0,-1,.15],[0,-1,-.15]])
    rotations=np.repeat(np.eye(3)[None],3,axis=0);scales=np.ones((3,3))
    before=fk(parents,order,translation,rotations,scales)
    target=np.array([.5,.1,.2])
    after,error=solve_leg([0,1,2],target,np.eye(3),parents,order,translation,rotations,scales,[0,0,1])
    assert error<1e-6
    np.testing.assert_allclose(after[2,:3,3],target,atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(np.diff(after[:,:3,3],axis=0),axis=1),
                               np.linalg.norm(np.diff(before[:,:3,3],axis=0),axis=1),atol=1e-7)


def test_foot_contact_rejects_swing_and_opposite_vectors_have_valid_rotation():
    times=np.arange(20)/30
    points=np.zeros((20,3));points[10:,0]=np.arange(10)*.1;points[10:,1]=.4
    contact=contact_mask(points,times,1)
    assert contact[:8].all() and not contact[11:].any()
    np.testing.assert_allclose(between([0,1,0],[0,-1,0])@[0,1,0],[0,-1,0],atol=1e-7)


def test_neutral_motion_keeps_shoe_bind_rotation_when_auto_rig_toe_pivots_differ():
    from hi3d_workbench.studio.animation import Motion,rotations
    from hi3d_workbench.studio.motion_constraints import skeleton_document
    source=human_rig();points=source.worlds[:,:3,3].copy()
    points[[14,18],1]=.005;points[[14,18],2]=.02
    target=skin.oriented_rig(points,source.parents,source.names)
    document=skeleton_document(target)
    clip=Motion(source,np.array([0.,1.]),np.repeat(source.worlds[None],2,axis=0),'neutral').validate()
    output,_=apply(document,clip,{'foot_lock':False})
    actual=sample_glb(output)
    for foot in [13,14,17,18]:
        np.testing.assert_allclose(rotations(actual.worlds[:,foot]),
                                   np.repeat(rotations(target.worlds[foot])[None],len(actual.times),axis=0),atol=2e-7)


@pytest.mark.parametrize('root_mode',['original','in_place'])
def test_exported_loop_poses_close_under_the_recorded_root_cycle(tmp_path,root_mode):
    source=animated_two(tmp_path)
    output,report=apply(source,sample_glb(source),{'loop':True,'root_motion':root_mode,'foot_lock':True})
    output.save(tmp_path/'loop.glb')
    from hi3d_workbench.studio.gltf import Document
    pose=sample_glb(Document.load(tmp_path/'loop.glb'))
    cycle=np.asarray(report['cycle_transform'])
    np.testing.assert_allclose(pose.worlds[-1],cycle @ pose.worlds[0],atol=3e-7)
    if root_mode=='in_place':
        np.testing.assert_allclose(pose.worlds[-1],pose.worlds[0],atol=3e-7)
