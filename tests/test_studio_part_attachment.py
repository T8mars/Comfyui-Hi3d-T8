import copy
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from test_studio_contact_skin import contact_asset
from test_workbench import client, testroot
from hi3d_workbench.common import atomic_json, digest
from hi3d_workbench.studio.gltf import Document
from hi3d_workbench.studio.part_attachment import attach
from hi3d_workbench.studio.parts import carry_metadata, read_parts
from hi3d_workbench.studio.skin import extract, posed_vertices, validate_existing_rig
from hi3d_workbench.studio.scene import instances


def metadata(path,names):
    counts={part:0 for part in names}
    for row in instances(Document.load(path)):
        counts[row.definition['extras']['studio_part']]+=len(row.faces)
    value={'schema':1,'mesh_sha256':digest(path),'parts':[{'id':part,'name':name,'faces':counts[part]}
        for part,name in names.items()],'faces':sum(counts.values()),'geometry_completed':False}
    atomic_json(path.parent/'parts.json',value)
    return path.parent/'parts.json'


def test_rigid_part_stops_unwanted_limb_motion_and_preserves_morph_pbr_animation(tmp_path):
    source=tmp_path/'source.glb';document=contact_asset(source)
    before=posed_vertices(document)
    result,report=attach(document,{7:'tail'},[{'part':7,'bone':0}])
    assert report['attachments']==[{'part':7,'name':'tail','bone':0,'bone_name':'hips','faces':4}]
    assert validate_existing_rig(document,result)['animation_clips_preserved']==1
    for left,right in zip(before,posed_vertices(result)):np.testing.assert_allclose(left,right,atol=1e-7)
    rig=extract(result)
    # Rotate a leg, rather than mirroring the production weight assignment.
    result.tree['nodes'][rig.nodes[3]]['rotation']=Rotation.from_euler('z',80,degrees=True).as_quat().tolist()
    for left,right in zip(before,posed_vertices(result)):np.testing.assert_allclose(left,right,atol=1e-7)
    target=tmp_path/'result.glb';result.save(target)
    assert Document.load(target).tree['materials']==document.tree['materials']


def test_attachment_preserves_visible_non_bind_pose_and_accepts_unskinned_accessories(tmp_path):
    document=contact_asset(tmp_path/'source.glb')
    rig=extract(document)
    document.tree['nodes'][rig.nodes[1]]['rotation']=Rotation.from_euler('z',35,degrees=True).as_quat().tolist()
    accessory=copy.deepcopy(document.tree['meshes'][0])
    for primitive in accessory['primitives']:
        primitive['extras']['studio_part']=8
        primitive['attributes']={k:v for k,v in primitive['attributes'].items() if not k.startswith(('JOINTS_','WEIGHTS_'))}
    mesh=len(document.tree['meshes']);document.tree['meshes'].append(accessory)
    node=len(document.tree['nodes']);document.tree['nodes'].append({'mesh':mesh,'translation':[2.,.5,0]})
    document.tree['scenes'][0]['nodes'].append(node)
    before=posed_vertices(document,allow_static=True)
    output,_=attach(document,{7:'body',8:'shield'},[{'part':8,'bone':1}])
    for a,b in zip(before,posed_vertices(output,allow_static=True)):np.testing.assert_allclose(a,b,atol=1e-6)
    assert validate_existing_rig(document,output)['animation_clips_preserved']==1


@pytest.mark.parametrize('attachments',[[],[{'part':7,'bone':True}],[{'part':99,'bone':0}],
    [{'part':7,'bone':999}],[{'part':7,'bone':0},{'part':7,'bone':1}]])
def test_invalid_bindings_are_rejected_before_modifying_the_source(tmp_path,attachments):
    document=contact_asset(tmp_path/'source.glb');tree=copy.deepcopy(document.tree)
    with pytest.raises(ValueError):attach(document,{7:'body'},attachments)
    assert document.tree==tree


def test_part_metadata_remains_editable_after_binding_and_cannot_authenticate_another_glb(tmp_path):
    source=tmp_path/'source.glb';document=contact_asset(source);sidecar=metadata(source,{7:'tail'})
    result,_=attach(document,{7:'tail'},[{'part':7,'bone':0}])
    folder=tmp_path/'job';folder.mkdir();target=folder/'output.glb';result.save(target)
    assert carry_metadata(source,target,sidecar)['part_metadata_preserved']
    _,labels,names=read_parts(target,folder/'parts.json')
    assert names=={7:'tail'} and sum(map(len,labels))==4
    with pytest.raises(ValueError,match='不一致'):read_parts(source,folder/'parts.json')


def test_source_api_does_not_return_stale_parts_and_attachment_requires_explicit_choices(client,tmp_path):
    from hi3d_workbench.server import ASSETS
    source=tmp_path/'source.glb';contact_asset(source)
    uploaded=client.post('/api/assets',files={'file':('rig.glb',source.read_bytes())}).json()
    key='asset:'+uploaded['id'];path=ASSETS/uploaded['id']/'mesh.glb';metadata(path,{7:'tail'})
    assert client.get('/api/studio/source',params={'source':key}).json()['parts']['parts'][0]['name']=='tail'
    value=json.loads((path.parent/'parts.json').read_text(encoding='utf-8'));value['mesh_sha256']='0'*64
    atomic_json(path.parent/'parts.json',value)
    assert 'parts' not in client.get('/api/studio/source',params={'source':key}).json()
    missing=client.post('/api/studio/jobs',json={'source':key,'action':'attach_parts'})
    assert missing.status_code==400 and '添加' in missing.json()['detail']
    malformed=client.post('/api/studio/jobs',json={'source':key,'action':'attach_parts','part_attachments':[{'part':7,'bone':True}]})
    assert malformed.status_code==422


def test_api_passes_explicit_assignments_and_part_metadata_to_the_owned_worker(client,tmp_path,monkeypatch):
    from hi3d_workbench import server
    from hi3d_workbench.studio import api
    source=tmp_path/'source.glb';contact_asset(source)
    uploaded=client.post('/api/assets',files={'file':('rig.glb',source.read_bytes())}).json()
    path=server.ASSETS/uploaded['id']/'mesh.glb';sidecar=metadata(path,{7:'tail'})
    monkeypatch.setattr(api.models,'status',lambda:{'features':{'retarget':{'ready':True}}})
    monkeypatch.setattr(api.models,'download_busy',lambda:False)
    calls=[]
    def submit(image,source,options):
        calls.append((image,source,options));return {'id':'fixture','options':options}
    monkeypatch.setattr(server.manager,'create',submit)
    response=client.post('/api/studio/jobs',json={'source':'asset:'+uploaded['id'],'action':'attach_parts',
        'part_attachments':[{'part':7,'bone':0}]})
    assert response.status_code==200,response.text
    assert len(calls)==1 and calls[0][0] is None and calls[0][1]==path
    assert calls[0][2]['part_attachments']==[{'part':7,'bone':0,'avoid_ground':False}] and calls[0][2]['parts_path']==str(sidecar)
