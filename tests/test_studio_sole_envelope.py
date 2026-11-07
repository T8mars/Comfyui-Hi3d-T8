import numpy as np
from scipy.spatial.transform import Rotation

from test_workbench import testroot
from hi3d_workbench.studio.gltf import Document
from hi3d_workbench.studio.skin import bind,extract,oriented_rig
from hi3d_workbench.studio.foot_support import Sole,soles


def test_toe_blend_above_the_rest_grid_minimum_can_be_the_true_contact_after_roll():
    positions=np.array([[-.2,-1.1,-.1],[.2,-1.1,.3],[0,-1.1,.12],
                        [.001,-1.099,.121],[-.15,-1.,.27],[.15,-1.,-.08]],dtype=np.float32)
    document=Document({'asset':{'version':'2.0'},'scene':0,'scenes':[{'nodes':[0]}],
        'nodes':[{'mesh':0}],'meshes':[{'primitives':[]}]})
    document.tree['meshes'][0]['primitives']=[{'attributes':{'POSITION':document.add_accessor(positions)},
        'indices':document.add_accessor(np.array([0,1,2,2,1,3,0,4,5],dtype=np.uint32))}]
    rig=oriented_rig([[0,0,0],[0,-1,0],[0,-1,.2]],[-1,0,1],['hips','left_foot','left_toe'])
    def provider(points):
        result=np.zeros((len(points),3));result[:,1]=1
        toe=(abs(points[:,0]-.001)<1e-6)&(abs(points[:,2]-.121)<1e-6)
        result[toe,1]=0;result[toe,2]=1
        return result
    document,_=bind(document,rig,provider);rig=extract(document)
    support=soles(document,rig,{'left_foot':1,'left_toe':2},1.)['left']
    rotation=Rotation.from_euler('x',-35,degrees=True).as_matrix()
    worlds=rig.worlds.copy();worlds[2,:3,:3]=rotation@worlds[2,:3,:3]
    actual=positions.astype(float).copy();origin=rig.worlds[2,:3,3]
    actual[3]=(actual[3]-origin)@rotation.T+origin
    assert actual[3,1]<positions[:,1].min()-.02
    np.testing.assert_allclose(support.height(worlds),actual[:,1].min(),atol=2e-7)


def test_height_keeps_late_vertices_instead_of_truncating_to_a_sample_budget():
    count=20000;indices=np.zeros((count,1),dtype=np.int64)
    weighted=np.zeros((count,1,4));weighted[:,:,3]=1;weighted[:,:,1]=1
    weighted[-1,0,1]=-5
    support=Sole(indices,weighted);worlds=np.eye(4)[None]
    assert support.height(worlds)==-5
    np.testing.assert_allclose(support.height(worlds),support.points(worlds)[:,1].min())
