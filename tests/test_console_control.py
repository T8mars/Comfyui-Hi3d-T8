import threading
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from fastapi.testclient import TestClient
from hi3d_workbench import console_control as control, lifecycle


def test_duplicate_uses_existing_server_and_stop_is_authenticated(monkeypatch):
    monkeypatch.setattr(control, 'running', lambda port: True)
    calls = []
    def request(port, path='/api/session', token=None):
        calls.append((port, path, token))
        return {'instance_id': lifecycle.instance_id(), 'token': 'session-secret', 'can_shutdown': True}
    monkeypatch.setattr(control, 'request', request)
    assert control.existing(8382) == 0
    assert calls == [(8382, '/api/session', None)]
    assert control.existing(8382, stop=True) == 0
    assert calls[-1] == (8382, '/api/shutdown', 'session-secret')


def test_other_bundle_or_legacy_service_is_never_stopped(monkeypatch):
    monkeypatch.setattr(control, 'running', lambda port: True)
    calls = []
    def request(port, path='/api/session', token=None):
        calls.append(path)
        return {'instance_id': 'another-directory', 'token': 'other-secret', 'can_shutdown': True}
    monkeypatch.setattr(control, 'request', request)
    assert control.existing(8382, stop=True) == 1
    assert calls == ['/api/session']


def test_browser_waits_for_same_directory(monkeypatch):
    monkeypatch.setattr(lifecycle, 'shutdown', threading.Event())
    monkeypatch.setattr(control, 'request', lambda port: {'instance_id': lifecycle.instance_id()})
    opened = []
    monkeypatch.setattr(control.webbrowser, 'open', opened.append)
    control.open_when_ready(8382, deadline=1)
    assert opened == ['http://127.0.0.1:8382/']
    assert lifecycle.instance_id(Path('a')) != lifecycle.instance_id(Path('b'))


def test_shutdown_requires_token_and_managed_server(monkeypatch):
    from hi3d_workbench import server
    stop = threading.Event()
    monkeypatch.setattr(lifecycle, 'shutdown', stop)
    monkeypatch.setattr(lifecycle, 'managed', True)
    client = TestClient(server.app, base_url='http://127.0.0.1:8382')
    assert client.post('/api/shutdown').status_code == 403
    assert not stop.is_set()
    assert client.post('/api/shutdown', headers={'X-Hi3D-Token': server.TOKEN, 'Origin': 'https://example.com'}).status_code == 403
    assert not stop.is_set()
    assert client.post('/api/shutdown', headers={'X-Hi3D-Token': server.TOKEN}).json() == {'stopping': True}
    assert stop.is_set()
    stop.clear()
    monkeypatch.setattr(lifecycle, 'managed', False)
    assert client.post('/api/shutdown', headers={'X-Hi3D-Token': server.TOKEN}).status_code == 409
    assert not stop.is_set()


def test_real_console_duplicate_wrong_directory_and_shutdown(tmp_path):
    root = tmp_path/'instance'; (root/'data/updates').mkdir(parents=True)
    (root/'data/updates/preferences.json').write_text('{"auto_update":false}')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0)); port = listener.getsockname()[1]
    environment = {**os.environ, 'HI3D_WORKBENCH_ROOT': str(root), 'PYTHONIOENCODING': 'utf-8'}
    command = [sys.executable, '-m', 'hi3d_workbench', '--port', str(port), '--no-browser']
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    def get(path):
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(f'http://127.0.0.1:{port}{path}', timeout=2) as response:
            return response.read().decode('utf-8')
    with (tmp_path/'server.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1], env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        try:
            deadline = time.monotonic()+30
            while True:
                try:session = json.loads(get('/api/session')); break
                except OSError:
                    if process.poll() is not None or time.monotonic() > deadline:raise
                    time.sleep(.2)
            assert session['can_shutdown'] and session['instance_id'] == lifecycle.instance_id(root)
            html = get('/')
            for domain in ('space.bilibili.com/385085361', 'youtube.com/@T8star-Aix/',
                           'api.seedance.nz/sign-up?aff=5f4w', 'www.openzhenzhen.com',
                           'inviteCode=rh-v1121', 'pan.quark.cn/s/264edb7e36bd', 'huggingface.co/t8star'):
                assert domain in html
            duplicate = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=environment,
                                       capture_output=True, timeout=15, creationflags=flags)
            assert duplicate.returncode == 0 and process.poll() is None
            wrong = subprocess.run([*command, '--stop'], cwd=Path(__file__).resolve().parents[1],
                                   env={**environment, 'HI3D_WORKBENCH_ROOT': str(tmp_path/'other')},
                                   capture_output=True, timeout=15, creationflags=flags)
            assert wrong.returncode == 1 and process.poll() is None
            stopped = subprocess.run([*command, '--stop'], cwd=Path(__file__).resolve().parents[1], env=environment,
                                     capture_output=True, timeout=15, creationflags=flags)
            assert stopped.returncode == 0
            assert process.wait(timeout=15) == 0
        finally:
            if process.poll() is None:process.kill()
            process.wait(timeout=10)
