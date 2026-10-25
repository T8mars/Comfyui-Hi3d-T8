from types import SimpleNamespace
import threading

import numpy as np
import psutil
import pytest

from hi3d_workbench.engines import encoding_cache
from hi3d_workbench.engines.checkpoint_io import ReadAhead
from hi3d_workbench.engines.texture_baking import inpaint_channels
from hi3d_workbench.engines.model_loading import LazyModel, ModelCache


def test_parallel_pbr_preserves_channels_parameters_and_waits_for_all(monkeypatch):
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=16*2**30))
    monkeypatch.setattr('os.cpu_count',lambda:8)
    barrier=threading.Barrier(4)
    calls=[]
    class CV2:
        INPAINT_TELEA=object()
        def inpaint(self,image,mask,radius,flags):
            assert flags is self.INPAINT_TELEA
            calls.append((image,mask,radius))
            barrier.wait(timeout=5)
            return image+radius
    arrays=[np.full((3,4),i,dtype=np.uint8) for i in range(4)]
    mask=np.zeros((3,4),dtype=np.uint8)
    events=[]
    values,report=inpaint_channels(CV2(),arrays,mask,lambda *a,**kw:events.append(kw))
    assert report['workers']==4
    assert all(np.array_equal(value,arrays[i]+(3 if i==0 else 1)) for i,value in enumerate(values))
    assert len(calls)==4 and all(m is mask for _,m,_ in calls)
    assert [e['progress_current'] for e in events]==[0,1,2,3,4]


def test_parallel_pbr_propagates_failure_and_joins_remaining_calls(monkeypatch):
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=16*2**30))
    finished=[]
    class CV2:
        INPAINT_TELEA=0
        def inpaint(self,image,*args):
            index=int(image[0,0]);finished.append(index)
            if index==2:raise ValueError('real inpaint failure')
            return image
    arrays=[np.full((3,4),i,dtype=np.uint8) for i in range(4)]
    with pytest.raises(ValueError,match='real inpaint failure'):
        inpaint_channels(CV2(),arrays,np.zeros((3,4),dtype=np.uint8),lambda *a,**kw:None)
    assert sorted(finished)==[0,1,2,3]


def test_pbr_memory_guard_uses_serial_original_algorithm(monkeypatch):
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=256*2**20))
    threads=set()
    class CV2:
        INPAINT_TELEA=0
        def inpaint(self,image,*args):threads.add(threading.get_ident());return image
    arrays=[np.zeros((3,4),dtype=np.uint8)]*4
    _,report=inpaint_channels(CV2(),arrays,arrays[0],lambda *a,**kw:None)
    assert report['workers']==1 and len(threads)==1


def test_encoding_cache_roundtrip_and_corruption_rejection(tmp_path):
    features=np.arange(32,dtype=np.float32).reshape(4,8)
    coords=np.arange(16,dtype=np.int32).reshape(4,4)
    key='a'*64
    assert encoding_cache.save_arrays(tmp_path,key,features,coords,'float32')
    actual=encoding_cache.load_arrays(tmp_path,key)
    assert actual is not None and np.array_equal(actual[0],features) and np.array_equal(actual[1],coords)
    (tmp_path/(key+'.npz')).write_bytes(b'PK broken zip')
    assert encoding_cache.load_arrays(tmp_path,key) is None


def test_encoding_cache_malformed_metadata_is_a_miss(tmp_path):
    import json
    key='b'*64
    np.savez(tmp_path/(key+'.npz'),features=np.ones((2,3),dtype=np.float32),
        coords=np.zeros((2,4),dtype=np.int32),metadata=np.frombuffer(json.dumps([]).encode(),dtype=np.uint8))
    assert encoding_cache.load_arrays(tmp_path,key) is None


@pytest.mark.parametrize('changed',['mesh','resolution','encoder','backend','runtime'])
def test_encoding_cache_invalidates_every_computational_input(changed):
    mesh=SimpleNamespace(vertices=np.eye(3),faces=np.array([[0,1,2]]))
    inputs=[mesh,512,('weights',1),'cpu',('torch',12)]
    before=encoding_cache.cache_identity(*inputs)
    if changed=='mesh':inputs[0]=SimpleNamespace(vertices=np.eye(3)*2,faces=mesh.faces)
    else:inputs[{'resolution':1,'encoder':2,'backend':3,'runtime':4}[changed]]={'resolution':1024,'encoder':('weights',2),'backend':'cuda','runtime':('torch',13)}[changed]
    assert encoding_cache.cache_identity(*inputs)!=before


def test_checkpoint_read_ahead_has_real_byte_progress_and_closes(tmp_path,monkeypatch):
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(total=64*2**30,available=16*2**30))
    path=tmp_path/'weights';data=b'abcd'*1000000;path.write_bytes(data)
    reader=ReadAhead();events=[]
    try:
        reader.ensure(path,lambda *a,**kw:events.append(kw),'checkpoint')
        assert reader.completed==len(data) and reader.stats['files']==1
        assert events[-1]['progress_current']==events[-1]['progress_total']==len(data)
        path.write_bytes(b'new weight bytes')
        reader.ensure(path,lambda *a,**kw:None,'checkpoint')
        assert reader.completed==16 and reader.stats['files']==2
    finally:reader.close()
    assert reader.executor is None


def test_checkpoint_prefetch_does_not_read_under_ram_pressure(tmp_path,monkeypatch):
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(total=64*2**30,available=1*2**30))
    path=tmp_path/'weights';path.write_bytes(b'weights')
    reader=ReadAhead()
    try:assert reader.start(path) is None and reader.executor is None
    finally:reader.close()


def test_runtime_only_policy_retains_no_model_and_avoids_cpu_copy(monkeypatch,tmp_path):
    from hi3d_workbench.engines import model_loading
    cache=ModelCache();cache.configure(tmp_path,'runtime');monkeypatch.setattr(model_loading,'CACHE',cache)
    moves=[]
    class Model:
        def __call__(self,value):return value
        def cpu(self):moves.append('unnecessary copy');return self
    proxy=LazyModel('model',('identity',),Model,lambda *a,**kw:None)
    assert proxy(5)==5
    proxy.release()
    assert not cache.entries and not moves and proxy._value is None
