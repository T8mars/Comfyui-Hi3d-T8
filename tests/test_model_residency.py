import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

from test_workbench import client
from hi3d_workbench.engines import model_loading as loading
from hi3d_workbench.runtime_session import RuntimeSession


class Model:
    def __init__(self):self.weight=42;self.moves=[]
    def to(self,*args,**kwargs):self.moves.append(args);return self
    def cpu(self):self.moves.append(('cpu',));return self
    def __call__(self,x):return self.weight+x
    def parameters(self):return []
    def buffers(self):return []


@pytest.fixture
def cache(monkeypatch,tmp_path):
    cache=loading.ModelCache()
    monkeypatch.setattr(loading,'CACHE',cache)
    cache.configure(tmp_path,'release')
    return cache


def test_lazy_loads_only_when_used_and_releases_after_phase(cache):
    calls=[];events=[]
    model=loading.LazyModel('fixture',('path',),lambda:calls.append(1) or Model(),lambda *a,**kw:events.append(kw))
    model.eval();model.to('cuda')
    assert not calls
    assert model(8)==50 and len(calls)==1
    model.cpu()
    assert model._value is not None  # A phase may call the same model twice.
    model.release()
    assert model._value is None and not cache.entries
    assert model(9)==51 and len(calls)==2


def test_resident_reuses_only_same_fingerprint_and_root(cache,tmp_path):
    cache.configure(tmp_path,'resident')
    calls=[]
    def proxy(key):return loading.LazyModel('fixture',key,lambda:calls.append(1) or Model(),lambda *a,**kw:None)
    first=proxy(('one',));first(0);first.release()
    second=proxy(('one',));second(0);second.release()
    assert len(calls)==1 and cache.hits==1
    third=proxy(('changed-file',));third(0);third.release()
    assert len(calls)==2
    cache.configure(tmp_path/'other-root','resident')
    assert not cache.entries


def test_resident_does_not_cache_under_ram_pressure(cache,tmp_path,monkeypatch):
    cache.configure(tmp_path,'resident')
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(total=64*2**30,available=1*2**30))
    cache.put(('one',),Model())
    assert not cache.entries


def test_checkpoint_fingerprint_invalidates_changed_weights(tmp_path):
    stem=tmp_path/'checkpoint'
    Path(str(stem)+'.json').write_text('{}')
    weight=Path(str(stem)+'.safetensors');weight.write_bytes(b'first')
    before=loading.fingerprint(stem)
    weight.write_bytes(b'changed checkpoint')
    assert loading.fingerprint(stem)!=before


def test_texturing_flow_keeps_condition_and_shape_arguments(cache):
    proxy=loading.LazyModel('flow',('flow',),Model,lambda *a,**kw:None)
    condition,shape=object(),object()
    args,kwargs=loading.flow_arguments((condition,proxy,shape,{'steps':4}),{})
    assert args[0] is condition and args[2] is shape and isinstance(args[1],Model)
    assert args[3]=={'steps':4} and kwargs=={}
    args,kwargs=loading.flow_arguments((condition,),{'flow_model':proxy,'shape_slat':shape})
    assert args==(condition,) and isinstance(kwargs['flow_model'],Model) and kwargs['shape_slat'] is shape


def test_runtime_session_runs_multiple_requests_and_releases_exact_process(tmp_path):
    script="from hi3d_workbench.runtime_session import serve; import os; serve(lambda item: print('PID='+str(os.getpid()),flush=True) or 0)"
    script=f'import sys;sys.path.insert(0,{str(Path(__file__).parents[1])!r});'+script
    session=RuntimeSession([sys.executable,'-u','-c',script,'hi3d_workbench.engines.run_stage'],tmp_path/'control.json')
    pid=session.process.pid
    try:
        one=list(session.stream(tmp_path/'one.json','coarse'))
        two=list(session.stream(tmp_path/'two.json','texture'))
        assert one==two and one[0].startswith('PID=') and session.code==0
        interpreter=int(one[0].strip().split('=')[1])
        assert interpreter in [pid,*[p.pid for p in psutil.Process(pid).children(recursive=True)]]
        assert session.process.poll() is None
    finally:session.close()
    assert not psutil.pid_exists(pid)


def test_runtime_failure_reaches_client_without_becoming_success(tmp_path):
    script="from hi3d_workbench.runtime_session import serve; serve(lambda item: 1)"
    script=f'import sys;sys.path.insert(0,{str(Path(__file__).parents[1])!r});'+script
    session=RuntimeSession([sys.executable,'-u','-c',script,'hi3d_workbench.engines.run_stage'],tmp_path/'control.json')
    try:
        assert list(session.stream(tmp_path/'one.json','coarse'))==[]
        assert session.code==1
    finally:session.close()


def test_runtime_close_cancels_active_request_and_its_child(tmp_path):
    import threading
    script="from hi3d_workbench.runtime_session import serve; import time,subprocess,sys; serve(lambda item: (subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']),print('BUSY',flush=True),time.sleep(120),0)[-1])"
    script=f'import sys;sys.path.insert(0,{str(Path(__file__).parents[1])!r});'+script
    session=RuntimeSession([sys.executable,'-u','-c',script,'hi3d_workbench.engines.run_stage'],tmp_path/'control.json')
    started=threading.Event();errors=[]
    def request():
        try:
            for line in session.stream(tmp_path/'one.json','coarse'):
                if 'BUSY' in line:started.set()
        except (RuntimeError,OSError,ValueError) as error:errors.append(error)
    thread=threading.Thread(target=request);thread.start()
    try:
        assert started.wait(10)
        pids=[session.process.pid,*[p.pid for p in psutil.Process(session.process.pid).children(recursive=True)]]
        session.close();thread.join(5)
        assert not thread.is_alive() and errors
        assert not any(psutil.pid_exists(pid) for pid in pids)
    finally:
        session.close();thread.join(5)


def test_unload_endpoint_is_authenticated_and_refuses_during_job(client,monkeypatch):
    from hi3d_workbench import server
    calls=[]
    monkeypatch.setattr(server.manager,'close_resident',lambda:calls.append('closed'))
    monkeypatch.setattr(server.manager,'runtime_status',lambda:{'resident':False})
    monkeypatch.setattr(server.manager,'active','busy-job')
    assert client.post('/api/runtime/unload').status_code==409 and not calls
    monkeypatch.setattr(server.manager,'active',None)
    assert client.post('/api/runtime/unload').status_code==200 and calls==['closed']

