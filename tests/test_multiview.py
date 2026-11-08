"""Named-view API snapshots and official MV inference wiring, without downloads."""
import io
import json
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from test_workbench import client
from hi3d_workbench import pipeline_models, server
from hi3d_workbench.common import read_json
from hi3d_workbench.engine_catalog import engine
from hi3d_workbench.multiview import prepare_views, stage_paths


def image_file(path, color):
    Image.new('RGBA', (40, 60), color).save(path)
    return path


def upload_image(client, color):
    stream = io.BytesIO()
    Image.new('RGBA', (40, 60), color).save(stream, format='PNG')
    response = client.post('/api/assets', files={'file': ('view.png', stream.getvalue(), 'image/png')})
    assert response.status_code == 200
    return response.json()


def test_mv_api_resolves_only_uploaded_images_and_checks_every_background(client, monkeypatch):
    front = upload_image(client, (255, 0, 0, 0))
    left = upload_image(client, (0, 255, 0, 255))
    status = {'engines': {'coarse': {'hunyuan_mv': {'ready': True}}}, 'background_ready': True}
    monkeypatch.setattr(server.pipeline_models, 'status', lambda: status)
    captures = []
    def create(image, mesh, options, view_images=None):
        captures.append((image, options, view_images))
        return {'id': 'mv-fixture', 'options': options}
    monkeypatch.setattr(server.manager, 'create', create)
    payload = {'image': front['id'], 'mode': 'image_to_asset', 'generate_texture': False,
               'geometry_engine': 'hunyuan_mv', 'view_images': {'left': left['id']}}
    response = client.post('/api/jobs', json=payload)
    assert response.status_code == 200, response.text
    assert captures[0][2]['left'].read_bytes() == (server.ASSETS / left['id'] / 'image.png').read_bytes()
    assert response.json()['options']['geometry_steps'] == 30
    assert response.json()['options']['view_images'] == {'left': left['id']}
    status['background_ready'] = False
    assert client.post('/api/jobs', json=payload).status_code == 409
    assert client.post('/api/jobs', json={**payload, 'background': 'keep'}).status_code == 200
    assert client.post('/api/jobs', json={**payload, 'geometry_engine': 'hunyuan_mini'}).status_code == 400
    assert client.post('/api/jobs', json={**payload, 'view_images': {'top': left['id']}}).status_code == 422
    assert client.post('/api/jobs', json={**payload, 'view_images': {'left': front['id'] + 'missing'}}).status_code in {400, 404}


def test_named_views_survive_job_snapshots_download_and_reuse(client, tmp_path, monkeypatch):
    from hi3d_workbench.jobs import JobManager
    manager = JobManager()
    manager.root = tmp_path / 'jobs'
    manager.root.mkdir()
    monkeypatch.setattr(manager, '_run', lambda jid: None)
    front = image_file(tmp_path / 'front.png', 'red')
    left = image_file(tmp_path / 'left.png', 'green')
    back = image_file(tmp_path / 'back.png', 'blue')
    options = {'mode': 'image_to_asset', 'geometry_engine': 'hunyuan_mv', 'view_images': {'left': 'old-left', 'back': 'old-back'}}
    job = manager.create(front, None, options, view_images={'left': left, 'back': back})
    folder = manager.root / job['id']
    left.unlink();back.unlink()
    monkeypatch.setattr(server, 'manager', manager)
    files = client.get('/api/jobs/' + job['id']).json()['available_files']
    assert {'input.png', 'view-left.png', 'view-back.png'} <= set(files)
    for name in ('view-left.png', 'view-back.png'):
        response = client.get(f"/api/jobs/{job['id']}/files/{name}")
        assert response.status_code == 200 and response.content == (folder / name).read_bytes()
    reused = client.post(f"/api/jobs/{job['id']}/reuse")
    assert reused.status_code == 200, reused.text
    value = reused.json()
    assert value['mesh'] is None and set(value['views']) == {'left', 'back'}
    for view, item in value['views'].items():
        assert item['id'] not in {'old-left', 'old-back'}
        assert (server.ASSETS / item['id'] / 'image.png').read_bytes() == (folder / f'view-{view}.png').read_bytes()
    (folder / 'view-back.png').unlink()
    assert client.post(f"/api/jobs/{job['id']}/reuse").status_code == 404


def test_every_view_uses_the_same_foreground_policy_with_direction_intact(tmp_path):
    from hi3d_workbench.engines.run_stage import foreground
    front = image_file(tmp_path / 'front.png', 'red')
    left = image_file(tmp_path / 'left.png', 'green')
    back = image_file(tmp_path / 'back.png', 'blue')
    directory = tmp_path / 'stage'
    directory.mkdir()
    request = {'engine': 'hunyuan_mv', 'image_path': str(front), 'background': 'keep',
               'view_paths': {'back': str(back), 'left': str(left)}}
    images, policies = prepare_views(request, directory, foreground, 'coarse')
    assert list(images) == ['front', 'left', 'back']
    assert images['front'].getpixel((0, 0)) == (255, 0, 0, 255)
    assert images['left'].getpixel((0, 0)) == (0, 128, 0, 255)
    assert images['back'].getpixel((0, 0)) == (0, 0, 255, 255)
    assert set(policies) == set(images)
    assert (directory / 'foreground-left.png').is_file() and (directory / 'foreground-back.png').is_file()
    with pytest.raises(ValueError, match='附加视图'):
        stage_paths({**request, 'engine': 'hunyuan21'}, 'coarse')
    with pytest.raises(ValueError, match='只支持'):
        stage_paths({**request, 'view_paths': {'top': str(front)}}, 'coarse')


def test_mv_uses_its_own_pinned_checkpoint_and_all_images(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    import trimesh
    from hi3d_workbench.engines import run_stage
    seen = {}
    class Pipeline:
        encode_cond = staticmethod(lambda *a, **kw: None)
        _export = staticmethod(lambda *a, **kw: None)
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            seen.update(path=path, loading=kwargs)
            return cls()
        def __call__(self, **kwargs):
            seen.update(sampling=kwargs)
            return [trimesh.creation.box()]
    module = types.ModuleType('hy3dgen.shapegen')
    module.Hunyuan3DDiTFlowMatchingPipeline = Pipeline
    monkeypatch.setitem(sys.modules, 'hy3dgen.shapegen', module)
    monkeypatch.setattr(run_stage, 'source_dir', lambda name: None)
    images = {view: Image.new('RGBA', (32, 32)) for view in ('front', 'left', 'back')}
    report = run_stage.coarse({'engine': 'hunyuan_mv', 'model_root': str(tmp_path), 'offload': False,
                              'geometry_steps': 30}, images, tmp_path / 'output.glb', lambda *a, **kw: None)
    assert Path(seen['path']).name == 'coarse_hunyuan_mv'
    assert seen['loading']['subfolder'] == 'hunyuan3d-dit-v2-mv'
    assert seen['loading']['use_safetensors'] is True
    assert seen['sampling']['image'] is images and seen['sampling']['num_inference_steps'] == 30
    assert report['views'] == ['front', 'left', 'back']
    spec = pipeline_models.SPECS['coarse_hunyuan_mv']
    assert report['model']['revision'] == spec['revision']
    assert 'coarse_hunyuan_mv' in pipeline_models.selection_components('complete')
    assert engine('coarse', 'hunyuan_mv')['views'] == ['front', 'left', 'back', 'right']
