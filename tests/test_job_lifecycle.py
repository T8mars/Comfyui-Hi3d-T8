"""Exercise process ownership with temporary scripts, never CUDA/model code."""
import json
import sys
import time
from pathlib import Path

import psutil
import pytest


@pytest.fixture
def lifecycle(tmp_path,monkeypatch):
    monkeypatch.setenv('HI3D_WORKBENCH_ROOT',str(tmp_path))
    from hi3d_workbench import jobs
    monkeypatch.setattr(jobs,'ROOT',tmp_path)
    monkeypatch.setattr(jobs,'DATA',tmp_path/'data')
    script=tmp_path/'fixture_worker.py'
    monkeypatch.setattr(jobs,'config',lambda:{'worker_command':[sys.executable,str(script)]})
    manager=jobs.JobManager()
    assert manager.root.is_relative_to(tmp_path)
    image=tmp_path/'input.png';image.write_bytes(b'fixture')
    mesh=tmp_path/'coarse.glb';mesh.write_bytes(b'fixture')
    yield jobs,manager,script,image,mesh
    # A Windows venv launcher can have its own Python child. Clean only exact
    # fixture script/request identities, including children of a dead launcher.
    for proc in psutil.process_iter(['pid','cmdline','create_time']):
        command=proc.info['cmdline'] or []
        if str(script) in command and 'hi3d_workbench.worker' in command and '--request' in command:
            request=Path(command[command.index('--request')+1])
            assert request.is_relative_to(tmp_path)
            jobs._kill_worker_tree(proc.pid,proc.info['create_time'],request)
    if manager.process is not None:manager.process.wait(timeout=5)


def wait_until(predicate,timeout=10):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if predicate():return
        time.sleep(.02)
    assert predicate(),'Timed out waiting for isolated process lifecycle'


def stop_fixture_tree(jobs,process,request):
    if process.poll() is None:
        jobs._kill_worker_tree(process.pid,psutil.Process(process.pid).create_time(),request)
    process.wait(timeout=5)


def test_malformed_progress_preserves_worker_ownership_until_cancel(lifecycle):
    jobs,manager,script,image,mesh=lifecycle
    script.write_text('import time\nprint("HI3D_EVENT invalid-json",flush=True)\n'
                      'print(\'HI3D_EVENT {"stage":"sampling","step":NaN}\',flush=True)\n'
                      'print("HI3D_EVENT []",flush=True)\ntime.sleep(30)\n')
    job=manager.create(image,mesh,{'resolution':512});jid=job['id']
    wait_until(lambda:(manager.root/jid/'worker.log').is_file() and 'invalid-json' in (manager.root/jid/'worker.log').read_text())
    assert manager.active==jid and manager.get(jid)['state']=='running'
    process=manager.process
    assert process is not None and process.poll() is None
    manager.cancel(jid)
    wait_until(lambda:manager.active is None)
    assert process.poll() is not None and manager.get(jid)['state']=='cancelled'
    assert jid not in manager.cancelled


def test_state_write_failure_reaps_worker_before_releasing_gpu_slot(lifecycle,monkeypatch):
    jobs,manager,script,image,mesh=lifecycle
    script.write_text('import time\nprint(\'HI3D_EVENT {"stage":"loading"}\',flush=True)\ntime.sleep(30)\n')
    original=manager._update
    def fail_once(jid,**values):
        if values.get('stage')=='loading':raise OSError('fixture state write failure')
        return original(jid,**values)
    monkeypatch.setattr(manager,'_update',fail_once)
    job=manager.create(image,mesh,{'resolution':512});jid=job['id']
    wait_until(lambda:manager.active is None)
    state=manager.get(jid)
    assert state['state']=='failed' and 'fixture state write failure' in state['message']
    assert not psutil.pid_exists(state['worker_pid'])
    assert (manager.root/jid/'input.png').read_bytes()==b'fixture'


def test_worker_identity_rejects_wrong_timestamp_and_other_request(lifecycle):
    jobs,manager,script,image,mesh=lifecycle
    script.write_text('import time\ntime.sleep(30)\n')
    job=manager.create(image,mesh,{'resolution':512});jid=job['id']
    wait_until(lambda:manager.process_created is not None)
    process=manager.process
    assert not jobs._kill_worker_tree(process.pid,manager.process_created+100,manager.root/jid/'request.json')
    assert process.poll() is None
    assert not jobs._kill_worker_tree(process.pid,manager.process_created,manager.root/'another-job'/'request.json')
    assert process.poll() is None
    manager.cancel(jid)
    wait_until(lambda:manager.active is None)
    assert process.poll() is not None


def test_fast_exit_with_artifacts_is_success_even_if_pid_probe_misses(lifecycle,monkeypatch):
    jobs,manager,script,image,mesh=lifecycle
    script.write_text('from pathlib import Path\nimport json,sys\n'
                      'folder=Path(sys.argv[-1]).parent\n'
                      '(folder/"refined.glb").write_bytes(b"fixture result")\n'
                      '(folder/"report.json").write_text(json.dumps({"status":"succeeded"}))\n')
    real_process=psutil.Process
    # Simulate a child already gone before psutil queries its PID. Popen's own
    # wait handle still gives its real exit code and preserves the artifacts.
    def disappeared(pid=None):
        if pid is not None:raise psutil.NoSuchProcess(pid)
        return real_process()
    monkeypatch.setattr(psutil,'Process',disappeared)
    job=manager.create(image,mesh,{'resolution':512});jid=job['id']
    wait_until(lambda:manager.active is None)
    assert manager.get(jid)['state']=='succeeded'
    assert (manager.root/jid/'refined.glb').read_bytes()==b'fixture result'


def test_spawn_failure_keeps_inputs_and_releases_slot(lifecycle,monkeypatch):
    jobs,manager,script,image,mesh=lifecycle
    monkeypatch.setattr(jobs,'config',lambda:{'worker_command':[str(script.parent/'missing-python.exe')]})
    job=manager.create(image,mesh,{'resolution':512});jid=job['id']
    wait_until(lambda:manager.active is None)
    assert manager.process is None and manager.get(jid)['state']=='failed'
    assert (manager.root/jid/'input.png').read_bytes()==b'fixture'
    assert (manager.root/jid/'coarse.glb').read_bytes()==b'fixture'


def test_recovery_reaps_only_matching_recorded_worker(lifecycle):
    jobs,manager,script,image,mesh=lifecycle
    script.write_text('import time\ntime.sleep(30)\n')
    folder=manager.root/'recover-fixture';folder.mkdir()
    request=folder/'request.json';request.write_text('{}')
    process=jobs.subprocess.Popen([sys.executable,str(script),'-m','hi3d_workbench.worker','--request',str(request)])
    try:
        created=psutil.Process(process.pid).create_time()
        jobs.atomic_json(folder/'state.json',{'id':folder.name,'created':1,'state':'running','worker_pid':process.pid,'worker_created':created})
        manager.recover()
        assert process.wait(timeout=5) is not None
        assert manager.get(folder.name)['state']=='interrupted'
        assert not manager.recovery_blocked and manager.recovery_workers==[]
        assert manager.recovery_errors==[]  # A stale process snapshot is not a live blocker.
        assert request.read_text()=='{}'
    finally:
        stop_fixture_tree(jobs,process,request)


def test_thread_start_failure_does_not_leave_active_slot(lifecycle,monkeypatch):
    jobs,manager,script,image,mesh=lifecycle
    monkeypatch.setattr(jobs.threading.Thread,'start',lambda self:(_ for _ in ()).throw(RuntimeError('fixture thread limit')))
    with pytest.raises(RuntimeError,match='无法启动任务线程'):manager.create(image,mesh,{'resolution':512})
    assert manager.active is None and manager.process is None
    records=manager.list()
    assert len(records)==1 and records[0]['state']=='failed'
    assert (manager.root/records[0]['id']/'input.png').read_bytes()==b'fixture'


def test_cancel_with_persistently_unwritable_state_still_reaps_and_releases(lifecycle,monkeypatch):
    jobs,manager,script,image,mesh=lifecycle
    script.write_text('import time\ntime.sleep(30)\n')
    job=manager.create(image,mesh,{'resolution':512});jid=job['id']
    wait_until(lambda:manager.process_created is not None)
    process=manager.process
    original=jobs.atomic_json
    def disk_full(path,value):
        if path.name=='state.json':raise OSError('fixture disk full')
        return original(path,value)
    monkeypatch.setattr(jobs,'atomic_json',disk_full)
    manager.cancel(jid)
    wait_until(lambda:manager.active is None)
    assert process.poll() is not None and manager.process is None
    assert manager.get(jid)['state']=='cancelled'
    assert manager.list()[0]['state']=='cancelled'


def test_corrupt_history_is_reported_without_breaking_other_jobs(lifecycle):
    jobs,manager,script,image,mesh=lifecycle
    broken=manager.root/'broken-history';broken.mkdir();(broken/'state.json').write_text('{bad JSON')
    valid=manager.root/'good-history';valid.mkdir()
    jobs.atomic_json(valid/'state.json',{'id':valid.name,'state':'succeeded','created':1})
    manager.recover()
    assert [value['id'] for value in manager.list()]==['good-history']
    assert any('broken-history' in value for value in manager.recovery_errors)
    assert not manager.recovery_blocked
    assert (broken/'state.json').read_text()=='{bad JSON'
    with pytest.raises(ValueError,match='记录损坏'):manager.get(broken.name)


@pytest.mark.parametrize('field,value',[('state',[]),('state',{}),('options',[]),('options','invalid'),('created',float('nan'))])
def test_invalid_record_schema_is_skipped_without_breaking_recovery(lifecycle,field,value):
    jobs,manager,script,image,mesh=lifecycle
    folder=manager.root/'invalid-schema';folder.mkdir()
    state={'id':folder.name,'state':'running','created':1,'options':{}}
    state[field]=value
    (folder/'state.json').write_text(json.dumps(state))
    manager.recover()
    assert manager.list()==[] and manager.recovery_errors
    assert not manager.recovery_blocked


def test_corrupt_record_of_live_worker_blocks_new_gpu_without_killing_unverified_identity(lifecycle):
    jobs,manager,script,image,mesh=lifecycle
    script.write_text('import time\ntime.sleep(30)\n')
    folder=manager.root/'corrupt-live';folder.mkdir();request=folder/'request.json';request.write_text('{}')
    (folder/'state.json').write_text('{broken')
    process=jobs.subprocess.Popen([sys.executable,str(script),'-m','hi3d_workbench.worker','--request',str(request)])
    try:
        manager.recover()
        assert process.poll() is None and manager.recovery_blocked
        with pytest.raises(RuntimeError,match='残留工作进程'):manager.create(image,mesh,{'resolution':512})
        stop_fixture_tree(jobs,process,request)
        assert not manager.recovery_blocked
    finally:
        stop_fixture_tree(jobs,process,request)


@pytest.mark.parametrize('state',['queued','running','cancelling'])
def test_live_worker_before_pid_persist_blocks_new_gpu_without_unsafe_kill(lifecycle,state):
    jobs,manager,script,image,mesh=lifecycle
    script.write_text('import time\ntime.sleep(30)\n')
    folder=manager.root/'worker-before-pid';folder.mkdir()
    request=folder/'request.json';request.write_text('{}')
    # Simulate a crash after Popen but before worker_pid / worker_created can
    # replace the valid persisted queued record.
    jobs.atomic_json(folder/'state.json',{'id':folder.name,'created':1,'state':state,'options':{}})
    process=jobs.subprocess.Popen([sys.executable,str(script),'-m','hi3d_workbench.worker','--request',str(request)])
    try:
        manager.recover()
        assert process.poll() is None  # Unrecorded identity is never killed.
        assert manager.recovery_blocked
        assert any(worker['pid']==process.pid for worker in manager.recovery_workers)
        assert any(folder.name in message for message in manager.recovery_errors)
        assert manager.get(folder.name)['state']=='interrupted'
        with pytest.raises(RuntimeError,match='残留工作进程'):manager.create(image,mesh,{'resolution':512})
        # A second restart sees the persisted interrupted state while the same
        # unverified worker is still alive. It must keep the GPU guard.
        restarted=jobs.JobManager();restarted.recover()
        assert restarted.recovery_blocked and process.poll() is None
        assert any(worker['pid']==process.pid for worker in restarted.recovery_workers)
        with pytest.raises(RuntimeError,match='残留工作进程'):restarted.create(image,mesh,{'resolution':512})
        stop_fixture_tree(jobs,process,request)
        assert not manager.recovery_blocked
        assert manager.recovery_workers==[]
        assert not restarted.recovery_blocked and restarted.recovery_workers==[]
    finally:
        stop_fixture_tree(jobs,process,request)


@pytest.mark.parametrize('state',['failed','cancelled','succeeded'])
def test_terminal_state_persisted_before_process_cleanup_blocks_overlap(lifecycle,state):
    jobs,manager,script,image,mesh=lifecycle
    script.write_text('import time\ntime.sleep(30)\n')
    folder=manager.root/'terminal-live';folder.mkdir()
    request=folder/'request.json';request.write_text('{}')
    process=jobs.subprocess.Popen([sys.executable,str(script),'-m','hi3d_workbench.worker','--request',str(request)])
    try:
        jobs.atomic_json(folder/'state.json',{'id':folder.name,'created':1,'state':state,'options':{},
                                             'worker_pid':process.pid,'worker_created':psutil.Process(process.pid).create_time()})
        manager.recover()
        assert process.poll() is None and manager.recovery_blocked
        assert manager.get(folder.name)['state']==state
        with pytest.raises(RuntimeError,match='残留工作进程'):manager.create(image,mesh,{'resolution':512})
        restarted=jobs.JobManager();restarted.recover()
        assert restarted.recovery_blocked and process.poll() is None
        stop_fixture_tree(jobs,process,request)
        assert not manager.recovery_blocked and not restarted.recovery_blocked
    finally:
        stop_fixture_tree(jobs,process,request)


def test_recovery_uses_one_process_snapshot_and_memory_state_on_disk_failure(lifecycle,monkeypatch):
    jobs,manager,script,image,mesh=lifecycle
    for number in range(3):
        folder=manager.root/f'queued-{number}';folder.mkdir()
        jobs.atomic_json(folder/'state.json',{'id':folder.name,'created':number,'state':'queued','options':{}})
    original_iter=psutil.process_iter;calls=[]
    def snapshot(*args,**kwargs):calls.append(True);return original_iter(*args,**kwargs)
    monkeypatch.setattr(psutil,'process_iter',snapshot)
    monkeypatch.setattr(jobs,'atomic_json',lambda *args,**kwargs:(_ for _ in ()).throw(OSError('fixture read-only history')))
    manager.recover()
    assert len(calls)==1
    assert len(manager.list())==3 and all(job['state']=='interrupted' for job in manager.list())
    assert not manager.recovery_blocked
    assert len(manager.recovery_errors)==3


def test_corrupted_active_record_keeps_in_memory_ownership_and_can_cancel(lifecycle):
    jobs,manager,script,image,mesh=lifecycle
    script.write_text('import time\ntime.sleep(30)\n')
    job=manager.create(image,mesh,{'resolution':512});jid=job['id']
    wait_until(lambda:manager.process_created is not None)
    (manager.root/jid/'state.json').write_text('{broken')
    assert manager.get(jid)['state']=='running' and manager.active==jid
    manager.cancel(jid)
    wait_until(lambda:manager.active is None)
    assert manager.get(jid)['state']=='cancelled'
