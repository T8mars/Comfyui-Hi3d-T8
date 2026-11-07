import numpy as np
from scipy.spatial.transform import Rotation

from test_studio_contact_skin import contact_asset
from hi3d_workbench.studio.attachment_ground import solve,envelope
from hi3d_workbench.studio.part_attachment import attach
from hi3d_workbench.studio.skin import extract,posed_vertices,validate_existing_rig
from hi3d_workbench.studio.gltf import Document


def test_exact_rigid_envelope_is_lifted_without_translating_the_attachment_pivot():
    points=np.array([[-.1,0,-1],[.1,0,-1],[-.1,.1,0],[.1,.1,0]],dtype=float)
    worlds=np.repeat(np.eye(4)[None],9,axis=0);worlds[:,:3,3]=[0,.2,0]
    worlds[:,:3,:3]=Rotation.from_euler('x',[0,-10,-20,-30,-40,-30,-20,-10,0],degrees=True).as_matrix()
    delta,report=solve(points,worlds,0,loop=True)
    assert report['corrected_frames']>0 and report['max_penetration_before']>.4
    assert report['max_penetration_after']<1e-6
    for matrix,rotation in zip(worlds,delta):
        actual=points@(matrix[:3,:3]@rotation).T+matrix[:3,3]
        assert actual[:,1].min()>-1e-6
        # A rotation keeps every edge length and leaves the origin at the pivot.
        np.testing.assert_allclose(np.linalg.norm(actual[1:]-actual[0],axis=1),
            np.linalg.norm(points[1:]-points[0],axis=1),atol=1e-10)
    np.testing.assert_allclose(delta[0],delta[-1],atol=1e-8)


def test_unreachable_ground_is_reported_without_moving_the_character():
    points=np.array([[0,-1,0],[0,-1,.01],[.01,-1,0.]])
    matrices=np.repeat(np.eye(4)[None],3,axis=0)
    _,report=solve(points,matrices,10.)
    assert report['max_penetration_after']>9 and report['corrected_frames']==0


def test_optional_attachment_joint_keeps_original_rig_animation_mesh_and_pbr(tmp_path):
    source=tmp_path/'source.glb';document=contact_asset(source)
    before=extract(document);visible=posed_vertices(document)
    output,details=attach(document,{7:'tail'},[{'part':7,'bone':0,'avoid_ground':True}])
    after=extract(output)
    assert len(after.nodes)==len(before.nodes)+1
    assert validate_existing_rig(document,output,allow_extra_joints=True)['animation_clips_preserved']==1
    for expected,actual in zip(visible,posed_vertices(output)):np.testing.assert_allclose(actual,expected,atol=1e-6)
    assert output.tree['materials']==document.tree['materials']
    node=details['attachments'][0]['helper_node']
    points=envelope(output,after,node)
    assert len(points)>0 and output.tree['extras']['hi3d_rigid_attachments'][0]['node']==node
    path=tmp_path/'result.glb';output.save(path)
    assert validate_existing_rig(document,Document.load(path),allow_extra_joints=True)['animation_clips_preserved']==1


def test_reassigning_an_attachment_without_ground_avoidance_removes_its_constraint(tmp_path):
    document=contact_asset(tmp_path/'source.glb')
    first,_=attach(document,{7:'tail'},[{'part':7,'bone':0,'avoid_ground':True}])
    second,_=attach(first,{7:'tail'},[{'part':7,'bone':0}])
    assert second.tree['extras']['hi3d_rigid_attachments']==[]
    for a,b in zip(posed_vertices(first),posed_vertices(second)):np.testing.assert_allclose(a,b,atol=1e-6)


def test_replacing_a_skeleton_drops_old_attachment_node_references(tmp_path):
    from test_studio_skin import rig_two,provider
    from hi3d_workbench.studio.skin import bind
    document=contact_asset(tmp_path/'source.glb')
    first,_=attach(document,{7:'tail'},[{'part':7,'bone':0,'avoid_ground':True}])
    second,_=bind(first,rig_two(),provider)
    assert 'hi3d_rigid_attachments' not in second.tree.get('extras',{})


def test_attachment_canonicalizes_reordered_skin_indices_without_changing_other_weights(tmp_path):
    import copy
    from hi3d_workbench.studio.scene import instances
    document=contact_asset(tmp_path/'source.glb');first=instances(document)[0]
    skin_index=document.tree['nodes'][first.node]['skin'];original=copy.deepcopy(document.tree['skins'][skin_index])
    count=len(original['joints']);original['joints'].reverse()
    matrices=document.accessor(original['inverseBindMatrices'])
    original['inverseBindMatrices']=document.add_accessor(matrices[::-1])
    mesh=copy.deepcopy(document.tree['meshes'][first.mesh])
    for primitive in mesh['primitives']:
        primitive['extras']['studio_part']=8
        attr=primitive['attributes'];indices=document.accessor(attr['JOINTS_0'])
        attr['JOINTS_0']=document.add_accessor((count-1-indices).astype(np.uint16))
    document.tree['meshes'].append(mesh);document.tree['skins'].append(original)
    node=len(document.tree['nodes']);document.tree['nodes'].append({'mesh':len(document.tree['meshes'])-1,
        'skin':len(document.tree['skins'])-1});document.tree['scenes'][0]['nodes'].append(node)
    before=posed_vertices(document)
    output,_=attach(document,{7:'tail',8:'body'},[{'part':7,'bone':0,'avoid_ground':True}])
    assert validate_existing_rig(document,output,allow_extra_joints=True)['existing_skeleton_preserved']
    for left,right in zip(before,posed_vertices(output)):np.testing.assert_allclose(left,right,atol=1e-6)
    row=instances(output)[1];after_skin=output.tree['skins'][output.tree['nodes'][row.node]['skin']]
    observed=np.asarray(after_skin['joints'])[output.accessor(row.definition['attributes']['JOINTS_0'])]
    expected=np.asarray(original['joints'])[document.accessor(mesh['primitives'][0]['attributes']['JOINTS_0'])]
    np.testing.assert_array_equal(observed,expected)
