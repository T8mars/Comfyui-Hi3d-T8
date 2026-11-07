import copy
import io
import json

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from test_workbench import testroot
from hi3d_workbench.studio.contact_skin import release,validate_partition,verify_saved_partition
from hi3d_workbench.studio.gltf import Document
from hi3d_workbench.studio.scene import instances
from hi3d_workbench.studio.skin import bind,extract,oriented_rig,posed_vertices,validate_existing_rig


def contact_asset(path,*,contact=True):
    xyz=np.array([[-1,0,0],[0,0,0],[1,0,0],[-1,1,0],[0,1,0],[1,1,0]],dtype=np.float32)
    faces=np.array([[0,1,3],[1,4,3],[1,2,4],[2,5,4]],dtype=np.uint32)
    doc=Document({'asset':{'version':'2.0'},'scene':0,'scenes':[{'nodes':[0]}],
        'nodes':[{'mesh':0}],'meshes':[{'primitives':[]}],'materials':[{'pbrMetallicRoughness':{'metallicFactor':.2}}]})
    attrs={name:doc.add_accessor(value) for name,value in {
        'POSITION':xyz,'NORMAL':np.tile([0.,0,1],(6,1)).astype(np.float32),
        'TEXCOORD_0':np.c_[(xyz[:,0]+1)/2,xyz[:,1]].astype(np.float32),
        'COLOR_0':np.tile([.3,.6,.8,1.],(6,1)).astype(np.float32)}.items()}
    doc.tree['meshes'][0].update(weights=[0.],primitives=[{'attributes':attrs,'material':0,
        'indices':doc.add_accessor(faces.reshape(-1)),
        'targets':[{'POSITION':doc.add_accessor(np.tile([0.,.01,0],(6,1)).astype(np.float32))}],
        'extras':{'studio_part':7,'user_note':'keep this'}}])
    image=io.BytesIO();Image.new('RGB',(2,2),(40,150,190)).save(image,format='PNG')
    doc.tree['images']=[{'bufferView':doc.add_view(image.getvalue()),'mimeType':'image/png'}]
    doc.tree['textures']=[{'source':0}]
    doc.tree['materials'][0]['pbrMetallicRoughness']['baseColorTexture']={'index':0}
    rig=oriented_rig(np.array([[0,0,0],[-1,.8,0],[1,.8,0],[-.4,-.3,0],[.4,-.3,0]]),
        [-1,0,0,0,0],['hips','left_upper_arm','right_upper_arm','left_thigh','right_thigh'])
    def provider(points):
        weights=np.zeros((len(points),5));weights[:,1]=np.clip((1-points[:,0])/2,0,1) if contact else 1
        weights[:,3]=1-weights[:,1];return weights
    bound,_=bind(doc,rig,provider)
    target=extract(bound).nodes[1]
    times=bound.add_accessor(np.array([0.,1.],dtype=np.float32))
    values=bound.add_accessor(Rotation.from_euler('z',[0,30],degrees=True).as_quat().astype(np.float32))
    bound.tree['animations']=[{'name':'preserved clip','samplers':[{'input':times,'output':values}],
        'channels':[{'sampler':0,'target':{'node':target,'path':'rotation'}}]}]
    bound.save(path);return bound


def strain(document):
    before=posed_vertices(document);rig=extract(document);node=document.tree['nodes'][rig.nodes[1]]
    node['rotation']=(Rotation.from_quat(node['rotation'])*Rotation.from_euler('z',60,degrees=True)).as_quat().tolist()
    after=posed_vertices(document);worst=0
    for row,a,b in zip(instances(document),before,after):
        a,b=a[row.faces],b[row.faces]
        old=np.linalg.norm(a[:,[1,2,0]]-a,axis=2);new=np.linalg.norm(b[:,[1,2,0]]-b,axis=2)
        worst=max(worst,float((new/old).max()))
    return worst


def test_contact_release_preserves_mesh_uv_morph_material_rig_animation_and_stops_stretch(tmp_path):
    source,target=tmp_path/'source.glb',tmp_path/'result.glb';before=contact_asset(source)
    original_tree=copy.deepcopy(before.tree)
    result,report=release(source,target,tmp_path/'stage')
    assert report['changed'] and report['corrected_vertex_copies']>0
    assert report['faces']==4 and report['preservation']['all_original_faces_once']
    assert len(instances(result))==2
    assert all(row.definition['extras']['studio_part']==7 and row.definition['extras']['user_note']=='keep this' for row in instances(result))
    assert before.tree==original_tree
    assert validate_existing_rig(before,result)['animation_clips_preserved']==1
    assert (tmp_path/'stage/face-map.npz').is_file()
    assert strain(Document.load(source))>1.1
    assert strain(Document.load(target))<1.00001


def test_clean_character_is_kept_byte_identical(tmp_path):
    source,target=tmp_path/'source.glb',tmp_path/'result.glb';contact_asset(source,contact=False)
    _,report=release(source,target,tmp_path/'stage')
    assert not report['changed'] and source.read_bytes()==target.read_bytes()


def test_contact_audit_rejects_duplicate_source_faces(tmp_path):
    source,target=tmp_path/'source.glb',tmp_path/'result.glb';contact_asset(source)
    result,_=release(source,target,tmp_path/'stage')
    with np.load(tmp_path/'stage/face-map.npz') as saved:
        plans=[(int(row),saved[f'faces_{i}'].copy()) for i,row in enumerate(saved['source_rows'])]
    plans[0][1][1]=plans[0][1][0]
    with pytest.raises(RuntimeError,match='重复'):
        validate_partition(Document.load(source),result,plans)


def test_rigging_runs_optional_contact_stage_and_keeps_raw_binding(tmp_path,monkeypatch):
    from hi3d_workbench.studio import rigging,rig_skintokens
    source=tmp_path/'source.glb';contact_asset(source)
    folder=tmp_path/'job';folder.mkdir();target=folder/'output.glb'
    def generate(document,request,emit,**kwargs):
        return Document(document.tree,document.binary),{'engine':'fixture','existing_skeleton':True}
    monkeypatch.setattr(rig_skintokens,'generate',generate)
    monkeypatch.setattr(rigging,'export_fbx',lambda *a,**k:{'fbx_roundtrip':True})
    report=rigging.run(source,target,{'action':'rig','rig_engine':'skintokens','use_skeleton':True,
        'release_contacts':True},lambda *a,**k:None)
    assert (folder/'rig-original.glb').is_file() and (folder/'rig.json').is_file()
    assert report['contact_release']['changed'] and report['preservation']['contact_split_triangle_preservation']
    assert len(instances(Document.load(target)))==2
    assert validate_existing_rig(Document.load(source),Document.load(target))['animation_clips_preserved']==1


def test_saved_contact_audit_and_full_binding_chain_check_real_files(tmp_path):
    from tools.validate_studio_binding import verify
    source,target=tmp_path/'source.glb',tmp_path/'result.glb';contact_asset(source)
    stage=tmp_path/'stage';release(source,target,stage)
    result=verify(source,target,original_binding=source,contact_stage=stage,existing_skeleton=True)
    assert result['contact_partition']['all_original_faces_once']
    assert result['existing_skeleton_preserved'] and result['animation_clips_preserved']==1
    # A stale pass flag or unchanged file length cannot authenticate a new GLB.
    altered=Document.load(target);altered.tree['asset']['generator']='tampered'
    altered.save(target)
    with pytest.raises(ValueError,match='哈希'):
        verify_saved_partition(source,target,stage)


def test_contact_audit_rejects_changed_part_metadata_even_with_a_valid_face_map(tmp_path):
    from hi3d_workbench.common import atomic_json,digest
    source,target=tmp_path/'source.glb',tmp_path/'result.glb';contact_asset(source)
    stage=tmp_path/'stage';release(source,target,stage)
    altered=Document.load(target)
    instances(altered)[0].definition['extras']['studio_part']=999
    altered.save(target)
    report=json.loads((stage/'report.json').read_text(encoding='utf-8'))
    report['output_sha256']=digest(target);atomic_json(stage/'report.json',report)
    with pytest.raises(RuntimeError,match='部件归属'):
        verify_saved_partition(source,target,stage)


def test_binding_audit_requires_both_contact_inputs_before_reading_files(tmp_path):
    from tools.validate_studio_binding import verify
    with pytest.raises(ValueError,match='同时指定'):
        verify(tmp_path/'absent.glb',tmp_path/'result.glb',original_binding=tmp_path/'raw.glb')
