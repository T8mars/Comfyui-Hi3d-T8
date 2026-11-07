import numpy as np

from test_workbench import client, testroot
from test_studio_bvh import BVH
from test_studio_retarget import animated_two
from test_studio_parts import source_model
from hi3d_workbench.server import ASSETS


def test_motion_upload_parses_bom_bvh_and_axis_changes_without_losing_identity(client):
    response=client.post('/api/studio/upload-motion',files={'file':('walk.bvh',b'\xef\xbb\xbf'+BVH.encode())})
    assert response.status_code==200,response.text
    value=response.json()
    assert value['kind']=='motion' and len(value['bones'])==3
    assert value['clips'][0]['duration']==1/30
    np.testing.assert_allclose(value['bones'][1]['position'],[0,100,0])
    changed=client.get('/api/studio/motion-source',params={'asset':value['id'],'up_axis':'Z'})
    assert changed.status_code==200,changed.text
    assert changed.json()['id']==value['id']
    np.testing.assert_allclose(changed.json()['bones'][1]['position'],[0,0,-100],atol=1e-10)


def test_motion_upload_reads_actual_glb_clips_and_rejects_static_mesh(client,tmp_path):
    document=animated_two(tmp_path);path=tmp_path/'motion.glb';document.save(path)
    response=client.post('/api/studio/upload-motion',files={'file':('action.glb',path.read_bytes())})
    assert response.status_code==200,response.text
    value=response.json()
    assert len(value['bones'])==2 and value['clips']==[{'index':0,'name':'turn','duration':1.}]
    before=set(ASSETS.iterdir())
    static=tmp_path/'static.glb';source_model(static).save(static)
    bad=client.post('/api/studio/upload-motion',files={'file':('static.glb',static.read_bytes())})
    assert bad.status_code==400 and '动画' in bad.json()['detail']
    assert set(ASSETS.iterdir())==before


def test_invalid_motion_cleans_only_its_fresh_directory_and_rejects_path_traversal(client):
    before=set(ASSETS.iterdir())
    for data in (b'',b'not bvh',BVH.replace('Frames: 2','Frames: 5').encode()):
        response=client.post('/api/studio/upload-motion',files={'file':('bad.bvh',data)})
        assert response.status_code==400,response.text
        assert set(ASSETS.iterdir())==before
    assert client.get('/api/studio/motion-source',params={'asset':'../outside'}).status_code==400
    assert client.get('/api/studio/motion-source',params={'asset':'0'*32}).status_code==404


def test_unrigged_target_fails_before_loading_motion_models(client,tmp_path):
    path=tmp_path/'static.glb';source_model(path).save(path)
    uploaded=client.post('/api/assets',files={'file':('model.glb',path.read_bytes())}).json()
    response=client.post('/api/studio/jobs',json={'source':'asset:'+uploaded['id'],'action':'motion'})
    assert response.status_code==400 and '骨架' in response.json()['detail']
