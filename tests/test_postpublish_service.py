"""New release regressions; fake probes and model operations use temporary roots."""
import io
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import psutil
import pytest
from test_server_operations import isolated_server


def test_verification_cannot_enter_while_download_is_starting(isolated_server,monkeypatch):
    server,client=isolated_server
    entered=threading.Event();release=threading.Event()
    def download():
        entered.set();assert release.wait(5);return True
    monkeypatch.setattr(server.models,'download_all',download)
    with ThreadPoolExecutor(max_workers=2) as pool:
        downloading=pool.submit(client.post,'/api/models/download')
        try:
            assert entered.wait(3)
            assert client.post('/api/models/verify').status_code==409
        finally:release.set()
        assert downloading.result(timeout=5).status_code==200

def _write_probe_fixture(server,monkeypatch,exit_parent=False):
    """Every process has a unique temp script path and inherited probe nonce."""
    import json
    script=server.DATA/'fake_probe.py';child_script=server.DATA/'owned_probe_child.py'
    pidfile=server.DATA/'owned_probe_child.json'
    child_script.write_text(
        "import json,os,time,psutil\nfrom pathlib import Path\n"
        "Path("+repr(str(pidfile))+").write_text(json.dumps({'pid':os.getpid(),'created':psutil.Process().create_time(),'marker':os.environ.get('HI3D_DIAGNOSTIC_ID')}))\n"
        "print('fixture child holding stdout',flush=True)\ntime.sleep(120)\n",encoding='utf-8')
    script.write_text(
        "import subprocess,sys,time\np=subprocess.Popen([sys.executable,"+repr(str(child_script))+"])\n"
        "print('fixture probe launcher',flush=True)\n"+("" if exit_parent else "time.sleep(120)\n"),encoding='utf-8')
    monkeypatch.setattr(server,'config',lambda:{'worker_python':sys.executable,'worker_command':[sys.executable,str(script)],'port':8378})
    monkeypatch.setattr(server,'DIAGNOSTIC_TIMEOUT',1.5)
    monkeypatch.setattr(server,'PROBE_WORKERS',{})
    return script,child_script,pidfile


def _cleanup_probe_fixture(script,child_script,started):
    # Exact temp paths and creation times identify our fixtures, including an
    # orphaned Windows venv redirector and its real Python child. No broad
    # Python-name / generic sleep-command cleanup is permitted.
    owned=[]
    for candidate in psutil.process_iter(['cmdline','create_time']):
        try:
            command=candidate.info['cmdline'] or []
            if (str(script) in command or str(child_script) in command) and candidate.info['create_time']>=started:
                proc=psutil.Process(candidate.pid)
                if abs(proc.create_time()-candidate.info['create_time'])<=.01:
                    owned.append(proc)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    for proc in reversed(owned):
        try:proc.kill()
        except psutil.NoSuchProcess:pass
    psutil.wait_procs(owned,timeout=5)


@pytest.mark.parametrize('exit_parent',[False,True],ids=['live-venv-launcher','exited-launcher-inherited-stdout'])
def test_diagnostic_timeout_reaps_descendants(isolated_server,monkeypatch,exit_parent):
    import json,os
    server,client=isolated_server
    script,child_script,pidfile=_write_probe_fixture(server,monkeypatch,exit_parent)
    started=time.time();inherited_marker=os.environ.get('HI3D_DIAGNOSTIC_ID')
    try:
        before=time.monotonic()
        response=client.post('/api/diagnostics')
        assert time.monotonic()-before<10,'Timeout must remain bounded after launcher exit'
        assert response.status_code==200 and response.json()['kernel_probe']=='failed'
        assert pidfile.is_file(),'The fixture must have created a real child'
        identity=json.loads(pidfile.read_text())
        assert identity['marker'] and identity['marker']!=inherited_marker
        assert not psutil.pid_exists(identity['pid']),'Timed-out diagnostic left its Python child alive'
        assert server.probe_recovery_workers()==[]
        health=client.get('/api/health').json()
        assert not health['recovery_blocked'] and health['recovery_workers']==[]
        assert server.operation is None and not server.GPU_OPERATION.locked()
        assert os.environ.get('HI3D_DIAGNOSTIC_ID')==inherited_marker,'The nonce must not mutate global environment'
    finally:_cleanup_probe_fixture(script,child_script,started)


def test_unreaped_owned_probe_keeps_gpu_guard_until_real_exit(isolated_server,monkeypatch):
    import json
    server,client=isolated_server
    script,child_script,pidfile=_write_probe_fixture(server,monkeypatch,True)
    started=time.time();kill=server._kill_probe_process
    def refuse_owned_child(process):
        if str(child_script) in process.cmdline():raise psutil.AccessDenied(process.pid)
        return kill(process)
    monkeypatch.setattr(server,'_kill_probe_process',refuse_owned_child)
    try:
        response=client.post('/api/diagnostics')
        assert response.status_code==200 and response.json()['kernel_probe']=='failed'
        identity=json.loads(pidfile.read_text())
        assert psutil.pid_exists(identity['pid'])
        health=client.get('/api/health').json()
        assert health['recovery_blocked'] and any(w['pid']==identity['pid'] for w in health['recovery_workers'])
        assert health['recovery_errors'] and server.operation is None
        assert client.post('/api/diagnostics').status_code==409
        assert client.post('/api/models/download').status_code==409
    finally:_cleanup_probe_fixture(script,child_script,started)
    health=client.get('/api/health').json()
    assert not health['recovery_blocked'] and health['recovery_workers']==[]


def test_nonzero_probe_exit_cannot_report_passed(isolated_server,monkeypatch):
    server,client=isolated_server
    script=server.DATA/'probe_nonzero.py'
    script.write_text('import sys\nprint(\'{"kernel_probe":"passed","cuda":true}\',flush=True)\nsys.exit(3)\n',encoding='utf-8')
    monkeypatch.setattr(server,'config',lambda:{'worker_python':sys.executable,'worker_command':[sys.executable,str(script)],'port':8378})
    result=client.post('/api/diagnostics')
    assert result.status_code==200 and result.json()['kernel_probe']=='failed'
    assert '3' in result.json()['error'] and not server.GPU_OPERATION.locked()


@pytest.mark.parametrize('vertices',[
    np.array([[0+1j,0,0],[1,0,0],[0,1,0]]),
    np.array([[0,0,0],[1e100,0,0],[0,1e100,1e100]]),
    np.array([[0,0,0],[1e-100,0,0],[0,1e-100,1e-100]])])
def test_npz_rejects_coordinates_glb_cannot_preserve(tmp_path,vertices):
    from hi3d_workbench.geometry import load_mesh
    path=tmp_path/'input.npz';np.savez(path,vertices=vertices,faces=[[0,1,2]])
    with pytest.raises(ValueError):load_mesh(path)

def test_scaled_export_rejects_overflow_without_touching_mesh(tmp_path):
    import trimesh
    from hi3d_workbench.geometry import export_mesh
    mesh=trimesh.creation.box(extents=[1e30,1,1e-11]);before=mesh.vertices.copy()
    with pytest.raises(ValueError):export_mesh(mesh,tmp_path/'bad.glb',100000)
    np.testing.assert_array_equal(mesh.vertices,before)
    assert not (tmp_path/'bad.glb').exists()
