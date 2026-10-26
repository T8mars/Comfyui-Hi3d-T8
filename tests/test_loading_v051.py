import json
from pathlib import Path
import threading
from types import SimpleNamespace

import psutil
import pytest

from hi3d_workbench.stage_prefetch import StagePrefetch, candidates, next_stages
from hi3d_workbench.engines import weight_transfer


def make_texture(root, size=4096):
    directory=root/'extensions/texture';directory.mkdir(parents=True)
    names=['shape_slat_encoder','tex_slat_flow_model_512','tex_slat_flow_model_1024','tex_slat_decoder']
    config={'args':{'models':{name:'ckpts/'+name for name in names}}}
    (directory/'texturing_pipeline.json').write_text(json.dumps(config))
    for name in names:
        path=directory/'ckpts'/f'{name}.safetensors';path.parent.mkdir(exist_ok=True);path.write_bytes(b'a'*size)
    encoder=root/'extensions/texture_encoder/model.safetensors';encoder.parent.mkdir();encoder.write_bytes(b'b'*size)
    return config


def test_prefetch_plans_only_requested_successor_stages():
    assert next_stages({'mode':'image_to_asset'})=={'coarse':'texture'}
    assert next_stages({'mode':'image_to_asset','refine_geometry':True})=={'coarse':'refine','refine':'texture'}
    assert next_stages({'mode':'image_to_asset','generate_texture':False})=={}
    assert next_stages({'mode':'refine','generate_texture':True})=={'refine':'texture'}
    assert next_stages({'mode':'texture'})=={}


@pytest.mark.parametrize('resolution',[512,1024])
def test_prefetch_excludes_unused_resolution_and_external_paths(tmp_path,resolution):
    config=make_texture(tmp_path)
    paths=candidates({'texture_resolution':resolution},'texture',tmp_path)
    assert paths[0].name=='model.safetensors'
    assert [path.stem for path in paths[1:]]==['shape_slat_encoder',f'tex_slat_flow_model_{resolution}','tex_slat_decoder']
    external=tmp_path.parent/'external.safetensors';external.write_bytes(b'external')
    config['args']['models']['shape_slat_encoder']=str(external)[:-len('.safetensors')]
    (tmp_path/'extensions/texture/texturing_pipeline.json').write_text(json.dumps(config))
    assert external not in candidates({'texture_resolution':resolution},'texture',tmp_path)


def test_real_prefetch_records_bytes_and_preserves_model_files(tmp_path,monkeypatch):
    make_texture(tmp_path)
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(total=64*2**30,available=16*2**30))
    prefetch=StagePrefetch({'mode':'image_to_asset'},tmp_path)
    try:
        prefetch.observe('coarse',{'stage':'coarse_complete'})
        prefetch.future.result(timeout=5)
        assert prefetch.status()['records'][0]['bytes']==4*4096
        assert all(item['complete'] for item in prefetch.status()['records'][0]['files'])
        prefetch.observe('texture',{'stage':'texture_loading','phase':'model_plan'})
        assert prefetch.status()['records'][0]['stopped_for_model_loading']
    finally:prefetch.close()
    assert prefetch.executor is None
    assert (tmp_path/'extensions/texture_encoder/model.safetensors').read_bytes()==b'b'*4096


def test_prefetch_yields_with_at_most_one_read_buffer_on_model_load(tmp_path,monkeypatch):
    make_texture(tmp_path,size=20*2**20)
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(total=64*2**30,available=16*2**30))
    entered=threading.Event();resume=threading.Event();original=Path.open
    class BlockingRead:
        def __init__(self,stream):self.stream=stream
        def __enter__(self):return self
        def __exit__(self,*args):self.stream.close()
        def readinto(self,buffer):
            entered.set();assert resume.wait(5)
            return self.stream.readinto(buffer)
    def opened(path,*args,**kwargs):
        stream=original(path,*args,**kwargs)
        return BlockingRead(stream) if path.suffix=='.safetensors' and args==('rb',) else stream
    monkeypatch.setattr(Path,'open',opened)
    prefetch=StagePrefetch({'mode':'image_to_asset'},tmp_path)
    try:
        prefetch.observe('coarse',{'stage':'coarse_complete'})
        assert entered.wait(5)
        prefetch.observe('texture',{'stage':'texture_unwrapping','phase':'uv_build'})
        assert not prefetch.stop.is_set()
        prefetch.observe('texture',{'stage':'texture_loading','phase':'model_plan'})
        assert prefetch.stop.is_set()
    finally:resume.set();prefetch.close()
    assert prefetch.status()['records'][0]['bytes']<=8*2**20


def test_prefetch_memory_guard_starts_no_reader(tmp_path,monkeypatch):
    make_texture(tmp_path)
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(total=16*2**30,available=1*2**30))
    prefetch=StagePrefetch({'mode':'image_to_asset'},tmp_path)
    try:
        prefetch.observe('coarse',{'stage':'coarse_complete'})
        assert prefetch.future is None
    finally:prefetch.close()


def test_transfer_fallback_preserves_custom_to_arguments_and_reports(monkeypatch):
    monkeypatch.setattr(weight_transfer.sys,'modules',{**weight_transfer.sys.modules,'torch':None})
    calls=[];events=[]
    class Wrapper:
        def to(self,*args,**kwargs):calls.append((args,kwargs));return self
    value=Wrapper()
    assert weight_transfer.move(value,'cpu',dtype='explicit',emit=lambda *a,**kw:events.append(kw)) is value
    assert calls==[(('cpu',),{'dtype':'explicit'})]
    assert events[-1]['transfer_backend']=='standard'


def test_failed_pipeline_closes_prefetch_and_resets_global(tmp_path,monkeypatch):
    from hi3d_workbench import pipeline_worker
    import hi3d_workbench.stage_prefetch as prefetch_module
    request=tmp_path/'request.json';request.write_text('{"mode":"texture"}')
    closed=[]
    class Prefetch:
        def __init__(self,*args):pass
        def close(self):closed.append(True)
        def status(self):return {'enabled':False,'records':[]}
    monkeypatch.setattr(prefetch_module,'StagePrefetch',Prefetch)
    def fail(path):raise RuntimeError('actual stage failure')
    monkeypatch.setattr(pipeline_worker,'_run',fail)
    with pytest.raises(RuntimeError,match='actual stage failure'):pipeline_worker.run(request)
    assert closed==[True] and pipeline_worker._stage_prefetch is None
