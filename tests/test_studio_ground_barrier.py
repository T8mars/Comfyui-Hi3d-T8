import numpy as np
from scipy.spatial.transform import Rotation

from test_workbench import testroot
from test_studio_foot_support import foot_mesh
from hi3d_workbench.studio.skin import extract
from hi3d_workbench.studio.humanoid import identify
from hi3d_workbench.studio.foot_support import soles
from hi3d_workbench.studio.animation import Motion,sample_glb
from hi3d_workbench.studio.retarget import apply
from hi3d_workbench.studio.gltf import Document


def test_contact_label_gaps_cannot_put_the_sole_underground_and_airtime_is_preserved(tmp_path):
    document=foot_mesh()
    # A rotating toe can lower shoe vertices while its joint origin stays at
    # the same height. Joint-only floor normalization cannot catch this case.
    for entry in document.primitives():
        primitive=entry[2]
        indices=document.accessor(primitive['attributes']['JOINTS_0']).copy()
        indices[indices==13]=14;indices[indices==17]=18
        primitive['attributes']['JOINTS_0']=document.add_accessor(indices)
    rig=extract(document);roles=identify(rig)['roles']
    support=soles(document,rig,roles,.8);floor=min(sole.height(rig.worlds) for sole in support.values())
    times=np.arange(31)/30
    vertical=np.zeros(31);vertical[15:22]=[.1,.2,.3,.4,.3,.2,.1]
    worlds=np.repeat(rig.worlds[None],len(times),axis=0)
    worlds[:,:,1,3]+=vertical[:,None]
    worlds[:,:,0,3]+=np.arange(31)[:,None]*.01
    for frame,angle in zip((5,6,7),(-30,-40,-30)):
        rotation=Rotation.from_euler('x',angle,degrees=True).as_matrix()
        for side in ('left','right'):
            joint=roles[side+'_toe'];worlds[frame,joint,:3,:3]=rotation@worlds[frame,joint,:3,:3]
    contacts=np.zeros((31,4));contacts[-2:]=1
    motion=Motion(rig,times,worlds,'contact gaps',foot_contacts=contacts).validate()
    output,report=apply(document,motion,{'foot_lock':True,'root_motion':'original'})
    path=tmp_path/'actual-output.glb';output.save(path)
    saved=sample_glb(Document.load(path))
    for side,sole in support.items():
        heights=np.array([sole.height(frame) for frame in saved.worlds])
        assert heights.min()>=floor-1e-5
        np.testing.assert_allclose(heights[15:22],floor+vertical[15:22],atol=1e-6)
        assert report['foot_ik'][side]['ground_projection_frames']>=3
        assert report['foot_ik'][side]['contact_frames']==2
    hips=roles['hips']
    np.testing.assert_allclose(saved.worlds[:,hips,:3,3],worlds[:,hips,:3,3],atol=1e-6)


def test_pelvis_reach_correction_cannot_push_the_other_unlabelled_foot_through_floor(tmp_path):
    document=foot_mesh();rig=extract(document);roles=identify(rig)['roles']
    support=soles(document,rig,roles,.8)
    floor=min(sole.height(rig.worlds) for sole in support.values())
    times=np.arange(31)/30;worlds=np.repeat(rig.worlds[None],len(times),axis=0)
    worlds[:,:,0,3]+=np.linspace(0,.4,len(times))[:,None]
    contacts=np.zeros((len(times),4));contacts[:,2:]=1
    motion=Motion(rig,times,worlds,'one planted foot',foot_contacts=contacts).validate()
    output,report=apply(document,motion,{'foot_lock':True,'root_motion':'original'})
    path=tmp_path/'pelvis-compensated.glb';output.save(path)
    saved=sample_glb(Document.load(path))
    assert report['pelvis_max_drop']>.01
    assert report['foot_ik']['left']['contact_frames']==0
    for sole in support.values():
        assert min(sole.height(frame) for frame in saved.worlds)>=floor-1e-5
    assert report['foot_ik']['left']['ground_projection_frames']>0
    np.testing.assert_allclose(saved.worlds[:,roles['hips'],0,3],worlds[:,roles['hips'],0,3],atol=1e-6)
