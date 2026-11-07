"""Real HTTP upload/process/cancel/recovery against an isolated local service.

All jobs and server processes belong to a newly created validation directory.
Production ports and data are never used. Authentication tokens stay in memory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--motion', type=Path, required=True)
    args = parser.parse_args()
    folder = args.output.resolve()
    if folder.exists() or not folder.is_relative_to(ROOT / 'data'):
        raise ValueError('Use a new validation directory under this checkout data folder')
    folder.mkdir(parents=True)
    (folder / 'config.local.json').write_text(json.dumps({'models_dir': str(ROOT / 'models'),
        'worker_python': str(Path(sys.executable)), 'worker_command': [], 'minimum_free_vram_mb': 1500}),encoding='utf-8')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0)); port = listener.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    env = {**os.environ, 'HI3D_WORKBENCH_ROOT': str(folder),
        'HI3D_STUDIO_RUNTIME': str(ROOT / 'runtime-assets'),
        'HI3D_STUDIO_PYTHON': str(ROOT / 'runtime-assets/python.exe'), 'PYTHONIOENCODING': 'utf-8'}
    server = None; server_log = None
    client = httpx.Client(base_url=base, timeout=90, headers={'origin': base})
    tracked = []; results = {}

    def call(method, path, **kwargs):
        response = client.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()

    def start():
        nonlocal server, server_log
        server_log = (folder / 'server.log').open('ab')
        server = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'hi3d_workbench.server:app',
            '--host', '127.0.0.1', '--port', str(port), '--no-access-log'], cwd=ROOT,
            env=env, stdout=server_log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        (folder / 'server-process.json').write_text(json.dumps({'pid':server.pid,
            'created':psutil.Process(server.pid).create_time(),'port':port}),encoding='utf-8')
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if server.poll() is not None: raise RuntimeError('Validation server exited')
            try:
                value = call('GET', '/api/session')
                client.headers['x-hi3d-token'] = value['token']; return
            except (httpx.TransportError, httpx.HTTPStatusError): time.sleep(.3)
        raise TimeoutError('Validation server did not become ready')

    def stop():
        nonlocal server, server_log
        if server is not None and server.poll() is None:
            server.terminate(); server.wait(timeout=20)
        if server_log is not None: server_log.close()

    def upload(path, endpoint):
        with Path(path).open('rb') as stream:
            return call('POST', endpoint, files={'file':(Path(path).name, stream, 'application/octet-stream')})

    def wait_job(jid, condition, timeout=180):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            value = call('GET', '/api/jobs/' + jid)
            if condition(value): return value
            if value['state'] in {'failed','cancelled','interrupted'}:raise RuntimeError(str(value))
            time.sleep(.5)
        raise TimeoutError('Job did not reach the required observable state: ' + jid)

    def create(body):
        job = call('POST', '/api/studio/jobs', json=body)
        tracked.append(job['id']); return job['id']

    def sampler_running(value):
        progress = value.get('progress') or {}
        return value.get('stage') == 'asset_motion_generate' and progress.get('current',0) >= 1

    try:
        start()
        state = call('GET', '/api/studio/status')
        for name in ('simplify','retarget','motion'):
            if not state['features'][name]['ready']:raise RuntimeError(name + ' runtime not ready')
        asset = upload(args.source, '/api/assets'); source = 'asset:' + asset['id']
        jid = create({'source':source,'action':'simplify','target_faces':2000,'priority':'count'})
        job = wait_job(jid, lambda value:value['state']=='succeeded')
        assert job['report']['output_faces'] <= 2000
        log = client.get('/api/jobs/' + jid + '/log'); log.raise_for_status()
        assert '\ufffd' not in log.text and '资产加工完成' in log.text
        result = client.get('/api/studio/jobs/' + jid + '/files/output.glb'); result.raise_for_status()
        (folder / 'http-simplified.glb').write_bytes(result.content)
        results['simplify'] = {'job':jid,'report':job['report'],'utf8_progress':True,
            'download_sha256':hashlib.sha256(result.content).hexdigest()}
        motion = upload(args.motion, '/api/studio/upload-motion')
        jid = create({'source':source,'action':'retarget','motion_asset':motion['id'],
            'root_motion':'in_place','loop':True,'foot_lock':True})
        job = wait_job(jid, lambda value:value['state']=='succeeded')
        assert job['report']['details']['fbx_roundtrip']
        results['retarget'] = {'job':jid,'report':job['report']}
        body = {'source':source,'action':'motion','text_device':'cuda','duration':10,'diffusion_steps':200,
            'prompt':'A person performs a controlled punch and returns to a fighting stance.','seed':23}
        jid = create(body); running = wait_job(jid,sampler_running)
        identity = {'pid':running['worker_pid'],'created':running['worker_created']}
        call('POST','/api/jobs/' + jid + '/cancel')
        job = wait_job(jid, lambda value:value['state']=='cancelled')
        results['cancel'] = {'job':jid,'state':job['state'],'worker':identity}
        if psutil.pid_exists(identity['pid']):
            proc=psutil.Process(identity['pid'])
            assert abs(proc.create_time()-identity['created'])>.01, 'Cancelled worker is still live'
        jid = create(body); running = wait_job(jid,sampler_running)
        identity = {'pid':running['worker_pid'],'created':running['worker_created']}
        assert psutil.Process(identity['pid']).is_running()
        stop(); start()
        job = call('GET','/api/jobs/' + jid)
        assert job['state']=='interrupted' and not call('GET','/api/health')['recovery_blocked']
        if psutil.pid_exists(identity['pid']):
            assert abs(psutil.Process(identity['pid']).create_time()-identity['created'])>.01
        assert (folder/'data/jobs'/jid/'coarse.glb').is_file()
        results['restart_recovery'] = {'job':jid,'state':job['state'],'worker':identity,'input_preserved':True}
        results['status']='passed'
    finally:
        if server is not None and server.poll() is None:
            for jid in tracked:
                try:
                    value=call('GET','/api/jobs/'+jid)
                    if value['state'] in {'running','queued','cancelling'}:
                        call('POST','/api/jobs/'+jid+'/cancel')
                        wait_job(jid,lambda row:row['state']=='cancelled',timeout=30)
                except Exception:pass
        stop(); client.close()
        (folder/'validation.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'status':results['status'],'evidence':str(folder/'validation.json')}),flush=True)


if __name__=='__main__':main()
