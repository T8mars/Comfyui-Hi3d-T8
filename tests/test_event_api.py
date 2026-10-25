"""Real HTTP progress reads from an isolated mixed-output worker; no models."""
import sys
import time

from test_workbench import client


def test_inline_stderr_stdout_event_reaches_http_and_terminal(client,tmp_path,monkeypatch,capsys):
    from hi3d_workbench import jobs,server
    monkeypatch.setattr(jobs,'ROOT',tmp_path)
    monkeypatch.setattr(jobs,'DATA',tmp_path/'data')
    manager=jobs.JobManager()
    monkeypatch.setattr(server,'manager',manager)
    script=tmp_path/'mixed_output_worker.py'
    script.write_text('from pathlib import Path\nimport json,sys,time\n'
        'folder=Path(sys.argv[-1]).parent\n'
        'sys.stderr.write("\\rSampling 7/12 ");sys.stderr.flush()\n'
        'print(\'HI3D_EVENT {bad JSON\',flush=True)\n'
        'sys.stderr.write("\\rSampling 7/12 ");sys.stderr.flush()\n'
        'print(\'HI3D_EVENT {"stage":"texture_sampling","phase":"material","device":"cuda","step":7,"steps":12} trailing display\',flush=True)\n'
        'while not (folder/"continue.fixture").exists():time.sleep(.01)\n'
        '(folder/"refined.glb").write_bytes(b"fixture")\n'
        '(folder/"report.json").write_text(json.dumps({"status":"succeeded"}))\n',encoding='utf-8')
    monkeypatch.setattr(jobs,'config',lambda:{'worker_command':[sys.executable,str(script)]})
    image=tmp_path/'image.png';image.write_bytes(b'fixture')
    mesh=tmp_path/'coarse.glb';mesh.write_bytes(b'fixture')
    jid=manager.create(image,mesh,{'resolution':512})['id']
    def wait_for(predicate):
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            if predicate():return
            time.sleep(.01)
        assert predicate(),'isolated worker did not deliver its progress'
    received=[]
    def http_has_progress():
        response=client.get('/api/jobs/'+jid)
        assert response.status_code==200,response.text
        received.append(response.json())
        return received[-1].get('progress',{}).get('current')==7
    try:
        wait_for(http_has_progress)
        state=received[-1]
        assert state['state']=='running' and state['progress']['total']==12
        assert state['progress']['phase']=='material' and state['progress']['device']=='cuda'
        output=[]
        def terminal_has_progress():
            output.append(capsys.readouterr().out)
            return '7/12 steps' in ''.join(output)
        wait_for(terminal_has_progress)
        assert 'CUDA / GPU' in ''.join(output)
        (manager.root/jid/'continue.fixture').touch()
        wait_for(lambda:manager.active is None)
        assert client.get('/api/jobs/'+jid).json()['state']=='succeeded'
        log=client.get('/api/jobs/'+jid+'/files/worker.log')
        assert log.status_code==200 and 'HI3D_EVENT {bad JSON' in log.text and 'trailing display' in log.text
    finally:
        if manager.active:
            manager.cancel(jid)
            wait_for(lambda:manager.active is None)
