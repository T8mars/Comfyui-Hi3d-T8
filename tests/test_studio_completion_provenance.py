import json,shutil,zipfile

import numpy as np

from test_workbench import testroot
from test_studio_parts import source_model,textured_triangles
from test_studio_skin import rig_two,provider
from hi3d_workbench.common import atomic_json,digest,read_json
from hi3d_workbench.studio.gltf import Document
from hi3d_workbench.studio.parts import carry_metadata,read_parts,run,write_parts
from hi3d_workbench.studio.scene import instances
from hi3d_workbench.studio.skin import bind


def reconstructed_fixture(tmp_path):
    folder=tmp_path/'completed';folder.mkdir()
    source=tmp_path/'original.glb';document=source_model(source)
    target=folder/'output.glb'
    labels=[np.zeros(len(row.faces),dtype=np.int32) for row in instances(document)]
    meta=write_parts(document,labels,{0:'piece'},target,engine='fixture',source_sha256=digest(source))
    shutil.copy2(target,folder/'original-segmentation.glb')
    atomic_json(folder/'original-parts.json',meta)
    shutil.copy2(folder/'parts.zip',folder/'original-parts.zip')
    meta.update(geometry_completed=True,engine='X-Part reconstruction fixture',original_segmentation='original-segmentation.glb')
    atomic_json(folder/'parts.json',meta)
    return target,folder/'parts.json'


def edit(tmp_path,source,sidecar,edits):
    folder=tmp_path/'edited';folder.mkdir();coarse=folder/'coarse.glb';shutil.copy2(source,coarse)
    output=folder/'output.glb'
    report=run(coarse,output,{'action':'edit_parts','parts_path':str(sidecar),'part_edits':edits},lambda *a,**k:None)
    return output,report,read_json(folder/'parts.json')


def test_rename_retains_reconstruction_and_originals_with_a_sidecar_in_the_previous_job(tmp_path):
    source,sidecar=reconstructed_fixture(tmp_path)
    output,report,meta=edit(tmp_path,source,sidecar,[{'operation':'rename','part':0,'name':'renamed'}])
    assert report['geometry_completed'] and meta['completion_inherited']
    assert meta['completion_origin']['mesh_sha256']==digest(source)
    assert report['original_segmentation_preserved']
    assert meta['parts'][0]['name']=='renamed' and meta['parts'][0]['watertight']
    np.testing.assert_array_equal(textured_triangles(Document.load(source)),textured_triangles(Document.load(output)))
    for name in ('original-segmentation.glb','original-parts.json','original-parts.zip'):
        assert digest(output.parent/name)==digest(source.parent/name)
    read_parts(output.parent/'original-segmentation.glb',output.parent/'original-parts.json')
    with zipfile.ZipFile(output.parent/'parts.zip') as archive:
        assert archive.testzip() is None
        assert json.loads(archive.read('parts.json'))['geometry_completed']
        assert archive.read('original-segmentation.glb')==(source.parent/'original-segmentation.glb').read_bytes()


def test_new_cut_does_not_confuse_reconstruction_history_with_current_closed_parts(tmp_path):
    source,sidecar=reconstructed_fixture(tmp_path);row=instances(Document.load(source))[0]
    output,_,meta=edit(tmp_path,source,sidecar,[{'operation':'create','part':1,'name':'new cut'},
        {'operation':'faces','mesh':row.mesh,'primitive':row.primitive,'faces':[0],'target_part':1}])
    assert meta['geometry_completed'] and meta['completion_inherited']
    assert len(meta['parts'])==2 and all(not part['watertight'] for part in meta['parts'])
    assert sum(part['faces'] for part in meta['parts'])==80
    assert '当前切口' in meta['notice']


def test_binding_keeps_original_partition_artifacts_with_the_reconstruction_metadata(tmp_path):
    source,sidecar=reconstructed_fixture(tmp_path)
    document,_=bind(Document.load(source),rig_two(),provider)
    folder=tmp_path/'rigged';folder.mkdir();output=folder/'output.glb';document.save(output)
    report=carry_metadata(source,output,sidecar)
    meta=read_json(folder/'parts.json')
    assert report['part_metadata_preserved'] and report['original_segmentation_preserved']
    assert meta['geometry_completed'] and meta['mesh_sha256']==digest(output)
    assert meta['completion_origin']['mesh_sha256']==digest(source)
    assert digest(folder/'original-segmentation.glb')==digest(source.parent/'original-segmentation.glb')


def test_invalid_original_attachment_is_not_published_as_an_available_reference(tmp_path):
    source,sidecar=reconstructed_fixture(tmp_path)
    original=read_json(source.parent/'original-parts.json');original['mesh_sha256']='0'*64
    atomic_json(source.parent/'original-parts.json',original)
    output,report,meta=edit(tmp_path,source,sidecar,[{'operation':'rename','part':0,'name':'valid current result'}])
    assert report['geometry_completed'] and not report['original_segmentation_preserved']
    assert 'original_segmentation' not in meta and meta['original_segmentation_error']
    assert not (output.parent/'original-segmentation.glb').exists()
