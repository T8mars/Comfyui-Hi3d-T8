import numpy as np
from scipy.spatial.transform import Rotation

from test_workbench import testroot
from test_studio_retarget import human_rig
from hi3d_workbench.studio.gltf import Document
from hi3d_workbench.studio.skin import bind,extract
from hi3d_workbench.studio.humanoid import identify
from hi3d_workbench.studio.foot_support import soles,surface_contacts


def foot_mesh(morph=False):
    document=Document({'asset':{'version':'2.0'},'scene':0,'scenes':[{'nodes':[0]}],
        'nodes':[{'mesh':0,'translation':[0,.02,0],'scale':[1.5,1,1]}],
        'meshes':[{'primitives':[]}]})
    positions=np.array([[x,y,z] for side in (1,-1) for x in (side*.06,side*.13)
                        for y in (0,.1) for z in (-.08,.25)],dtype=np.float32)
    position=document.add_accessor(positions)
    indices=document.add_accessor(np.array([0,1,2,1,3,2,8,9,10,9,11,10],dtype=np.uint32))
    primitive={'attributes':{'POSITION':position},'indices':indices}
    if morph:
        delta=np.zeros_like(positions);delta[:,1]=.01
        primitive['targets']=[{'POSITION':document.add_accessor(delta)}]
        document.tree['meshes'][0]['weights']=[1.]
    document.tree['meshes'][0]['primitives']=[primitive]
    rig=human_rig()
    def weights(points):
        result=np.zeros((len(points),len(rig.names)),dtype=np.float32)
        result[np.arange(len(points)),np.where(points[:,0]>0,13,17)]=1
        return result
    return bind(document,rig,weights)[0]


def test_sole_uses_inverse_binds_world_transforms_and_default_morphs():
    for morph,expected in ((False,.02),(True,.03)):
        document=foot_mesh(morph);rig=extract(document)
        support=soles(document,rig,identify(rig)['roles'],.8)
        assert set(support)=={'left','right'}
        for sole in support.values():
            assert 1<=len(sole.indices)<=64
            np.testing.assert_allclose(sole.height(rig.worlds),expected,atol=1e-7)
            moved=rig.worlds.copy();moved[:,1,3]+=2
            np.testing.assert_allclose(sole.height(moved),expected+2,atol=1e-7)


def test_sole_follows_foot_roll_without_changing_the_document():
    document=foot_mesh();rig=extract(document);original=bytes(document.binary)
    support=soles(document,rig,identify(rig)['roles'],.8)['left']
    worlds=rig.worlds.copy();rotation=Rotation.from_euler('x',30,degrees=True).as_matrix()
    worlds[13,:3,:3]=rotation @ worlds[13,:3,:3]
    before=support.points(rig.worlds)
    expected=(before-rig.worlds[13,:3,3]) @ rotation.T+rig.worlds[13,:3,3]
    np.testing.assert_allclose(support.points(worlds),expected,atol=1e-7)
    assert bytes(document.binary)==original


def test_imported_contact_estimate_rejects_the_airborne_middle_of_a_jump():
    from hi3d_workbench.studio.animation import Motion
    document=foot_mesh();rig=extract(document)
    supports=soles(document,rig,identify(rig)['roles'],.8)
    times=np.arange(15)/30;worlds=np.repeat(rig.worlds[None],15,axis=0)
    worlds[5:10,:,1,3]+=1
    motion=Motion(rig,times,worlds,'jump').validate()
    contacts=surface_contacts(supports,motion,.8)
    assert contacts[:4].all() and contacts[11:].all()
    assert not contacts[5:10].any()
