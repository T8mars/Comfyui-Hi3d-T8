import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from test_studio_skin import rig_two, provider
from hi3d_workbench.studio.gltf import Document
from hi3d_workbench.studio.skin import bind, extract, posed_vertices, validate_preservation
from hi3d_workbench.studio.scene import combined


def morphed_source():
    doc=Document({'asset':{'version':'2.0'},'scene':0,'scenes':[{'nodes':[0]}],
        'nodes':[{'mesh':0,'translation':[2.,-1.,.5],'scale':[2.,3.,4.],
                  'rotation':Rotation.from_euler('z',90,degrees=True).as_quat().tolist()}],
        'meshes':[{'weights':[.6],'primitives':[]}]})
    vertices=np.array([[0,0,0],[1,0,0],[0,1,0]],dtype=np.float32)
    deltas=np.tile([.5,.1,-.2],(3,1)).astype(np.float32)
    doc.tree['meshes'][0]['primitives']=[{'attributes':{'POSITION':doc.add_accessor(vertices)},
        'indices':doc.add_accessor(np.array([0,1,2],dtype=np.uint32)),
        'targets':[{'POSITION':doc.add_accessor(deltas)}]}]
    return doc,vertices.astype(float),deltas.astype(float)


@pytest.mark.parametrize('override',[None,.25])
def test_visible_morph_is_applied_before_node_transform_and_skinning(override):
    source,vertices,deltas=morphed_source()
    if override is not None:source.tree['nodes'][0]['weights']=[override]
    weight=.6 if override is None else override
    rotation=np.array([[0.,-1,0],[1,0,0],[0,0,1.]])
    expected=((vertices+weight*deltas)*[2,3,4])@rotation.T+[2,-1,.5]
    np.testing.assert_allclose(combined(source,apply_morphs=True)[0],expected,atol=1e-7)
    np.testing.assert_allclose(posed_vertices(source,allow_static=True)[0],expected,atol=1e-7)
    observed=[]
    def recording_provider(points):
        observed.append(points.copy())
        return provider(points)
    result,_=bind(source,rig_two(),recording_provider)
    np.testing.assert_allclose(np.concatenate(observed),expected,atol=1e-7)
    np.testing.assert_allclose(posed_vertices(result)[0],expected,atol=1e-7)
    rig=extract(result)
    result.tree['nodes'][rig.nodes[1]]['rotation']=Rotation.from_euler('z',90,degrees=True).as_quat().tolist()
    expected=(expected-[0,1,0])@rotation.T+[0,1,0]
    np.testing.assert_allclose(posed_vertices(result)[0],expected,atol=2e-7)


def test_rebinding_non_bind_pose_preserves_active_morph_and_detects_dropped_weights():
    source,_,_=morphed_source()
    first,_=bind(source,rig_two(),provider)
    rig=extract(first)
    first.tree['nodes'][rig.nodes[1]]['rotation']=Rotation.from_euler('z',35,degrees=True).as_quat().tolist()
    expected=posed_vertices(first)[0]
    result,_=bind(first,extract(first),provider,use_existing=True)
    np.testing.assert_allclose(posed_vertices(result)[0],expected,atol=2e-6)
    assert validate_preservation(first,result)['initial_geometry_preserved']
    result.tree['meshes'][0]['weights']=[0.]
    with pytest.raises(RuntimeError,match='外形'):
        validate_preservation(first,result)


@pytest.mark.parametrize('weights',[[float('nan')],[.2,.3]])
def test_invalid_active_morph_weights_are_rejected(weights):
    source,_,_=morphed_source();source.tree['nodes'][0]['weights']=weights
    with pytest.raises(ValueError,match='形变目标权重'):
        posed_vertices(source,allow_static=True)
