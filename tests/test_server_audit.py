"""Keep status polling responsive during CPU-heavy uploads, with isolated assets."""
import threading
from concurrent.futures import ThreadPoolExecutor
import trimesh
from test_server_operations import isolated_server

def test_upload_inspection_does_not_block_health(isolated_server,monkeypatch):
    server,client=isolated_server
    entered=threading.Event();release=threading.Event();original=server.inspect_mesh
    def inspect(mesh):
        entered.set()
        assert release.wait(8),'Test did not release geometry inspection'
        return original(mesh)
    monkeypatch.setattr(server,'inspect_mesh',inspect)
    with ThreadPoolExecutor(max_workers=2) as pool:
        upload=pool.submit(client.post,'/api/assets',files={'file':('mesh.glb',trimesh.creation.box().export(file_type='glb'))})
        try:
            assert entered.wait(3)
            health=pool.submit(client.get,'/api/health')
            assert health.result(timeout=2).status_code==200
        finally:release.set()
        assert upload.result(timeout=5).status_code==200

def test_live_endpoints_are_not_cached_and_assets_are_versioned(isolated_server):
    server,client=isolated_server
    for path in ['/','/api/session','/api/health']:
        response=client.get(path)
        assert response.status_code==200
        assert response.headers['cache-control']=='no-store'
    html=client.get('/').text
    assert f'/static/app.js?v={server.VERSION}' in html
    assert f'/static/style.css?v={server.VERSION}' in html
    assert '__VERSION__' not in html

def test_corrupt_optional_diagnostics_does_not_disable_health(isolated_server):
    server,client=isolated_server
    path=server.DATA/'diagnostics.json';path.write_text('{incomplete',encoding='utf-8')
    response=client.get('/api/health')
    assert response.status_code==200
    assert response.json()['diagnostics']['kernel_probe']=='not_run'
    assert '诊断记录无法读取' in response.json()['diagnostics']['error']
    assert path.read_text(encoding='utf-8')=='{incomplete'
