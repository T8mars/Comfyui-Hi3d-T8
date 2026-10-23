"""Post-publication boundary regressions; no production models, jobs or CUDA."""
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image
import pytest


class Tensor:
    def __init__(self, value):self.value=np.asarray(value)
    @property
    def shape(self):return self.value.shape
    def __getitem__(self, index):return Tensor(self.value[index])
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.value


@pytest.fixture
def bridge(monkeypatch):
    spec=importlib.util.spec_from_file_location('postpublish_nodes',Path(__file__).parents[1]/'comfy_nodes.py')
    nodes=importlib.util.module_from_spec(spec);spec.loader.exec_module(nodes)
    calls=[]
    class Interrupted(Exception):pass
    mm=ModuleType('comfy.model_management')
    mm.unload_all_models=lambda:calls.append('unload')
    mm.soft_empty_cache=lambda:calls.append('empty_cache')
    mm.throw_exception_if_processing_interrupted=lambda:None
    comfy=ModuleType('comfy');comfy.model_management=mm
    utils=ModuleType('comfy.utils');utils.ProgressBar=lambda steps:SimpleNamespace(update_absolute=lambda value:None)
    monkeypatch.setitem(sys.modules,'comfy',comfy)
    monkeypatch.setitem(sys.modules,'comfy.model_management',mm)
    monkeypatch.setitem(sys.modules,'comfy.utils',utils)
    monkeypatch.setitem(sys.modules,'torch',ModuleType('torch'))
    uploads=[];requests=[]
    def upload(backend,name,data):
        uploads.append((name,data));return {'id':name}
    def request(base,path,payload=None,**kwargs):
        requests.append((path,payload))
        if path=='/api/jobs':return {'id':'fixture'}
        if path=='/api/jobs/fixture':return {'state':'succeeded','report':{'status':'succeeded'}}
        if path.endswith('/refined.npz'):
            data=io.BytesIO();np.savez(data,vertices=[[0,0,0],[1,0,0],[0,1,0]],faces=[[0,1,2]]);return data.getvalue()
        if path.endswith('/cancel'):return {'state':'cancelled'}
        raise AssertionError(path)
    monkeypatch.setattr(nodes,'upload',upload)
    monkeypatch.setattr(nodes,'request',request)
    monkeypatch.setattr(nodes,'mesh_value',lambda v,f:SimpleNamespace(vertices=v,faces=f))
    mesh=SimpleNamespace(vertices=np.array([[0,0,0],[1,0,0],[0,1,0]]),faces=np.array([[0,1,2]]))
    def run(image=None,mask=None):
        if image is None:image=Tensor(np.ones((1,2,2,3),dtype=np.float32))
        return nodes.Hi3DRefine().refine({'url':'http://127.0.0.1:8378'},image,mesh,512,25,0,True,'sdpa',mask)
    return SimpleNamespace(nodes=nodes,mm=mm,calls=calls,uploads=uploads,requests=requests,run=run,Interrupted=Interrupted)


def test_rgba_transparency_survives_connected_empty_mask(bridge):
    rgba=np.ones((1,2,2,4),dtype=np.float32);rgba[0,0,0,3]=0;rgba[0,0,1,3]=.5
    bridge.run(Tensor(rgba),Tensor(np.zeros((1,2,2))))
    png=Image.open(io.BytesIO(bridge.uploads[0][1]))
    assert np.asarray(png)[...,3].tolist()==[[0,128],[255,255]]


def test_interrupt_during_upload_never_starts_remote_job(bridge,monkeypatch):
    original=bridge.nodes.upload
    def interrupted_upload(*args):
        result=original(*args)
        bridge.mm.throw_exception_if_processing_interrupted=lambda:(_ for _ in ()).throw(bridge.Interrupted('fixture interrupted'))
        return result
    monkeypatch.setattr(bridge.nodes,'upload',interrupted_upload)
    with pytest.raises(bridge.Interrupted):bridge.run()
    assert not bridge.requests and 'unload' not in bridge.calls


def test_comfy_mask_convention_and_mesh_success(bridge):
    output,report,jid=bridge.run(mask=Tensor([[[1,0],[.25,.75]]]))
    alpha=np.asarray(Image.open(io.BytesIO(bridge.uploads[0][1])))[...,3]
    assert alpha.tolist()==[[0,255],[191,64]]
    with np.load(io.BytesIO(bridge.uploads[1][1]),allow_pickle=False) as mesh:
        assert mesh['vertices'].shape==(3,3) and mesh['faces'].tolist()==[[0,1,2]]
    assert output.faces.tolist()==[[0,1,2]] and json.loads(report)['status']=='succeeded' and jid=='fixture'


@pytest.fixture
def native_nodes():
    spec=importlib.util.spec_from_file_location('postpublish_http_nodes',Path(__file__).parents[1]/'comfy_nodes.py')
    nodes=importlib.util.module_from_spec(spec);spec.loader.exec_module(nodes)
    return nodes


@pytest.fixture
def local_http():
    """A real loopback HTTP peer whose body can be held without using GPU."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    release=threading.Event();received=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):self.respond()
        def do_POST(self):self.respond()
        def respond(self):
            received.append((self.command,self.path,self.headers.get('x-hi3d-token')))
            length=int(self.headers.get('Content-Length','0'))
            if length:self.rfile.read(length)
            data=b'{"token":"fixture-token","ok":true}'
            self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers()
            if self.path.startswith('/slow'):release.wait(3)
            try:self.wfile.write(data)
            except OSError:pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    server.daemon_threads=True
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield 'http://127.0.0.1:'+str(server.server_port),received
    finally:
        release.set();server.shutdown();server.server_close();thread.join(timeout=3)


def test_loopback_http_uses_distinct_poll_upload_and_cancel_timeouts(native_nodes,local_http,monkeypatch):
    base,received=local_http;timeouts=[]
    original=native_nodes._LOCAL_HTTP.open
    def open_request(req,**kwargs):timeouts.append(kwargs['timeout']);return original(req,**kwargs)
    monkeypatch.setattr(native_nodes._LOCAL_HTTP,'open',open_request)
    assert native_nodes.request(base,'/api/jobs/fixture')['ok']
    assert native_nodes.request(base,'/api/assets',b'data',token='fixture-token')['ok']
    assert native_nodes.request(base,'/api/jobs/fixture/cancel',{},token='fixture-token')['ok']
    assert timeouts==[15,600,10]
    assert received==[('GET','/api/jobs/fixture',None),('POST','/api/assets','fixture-token'),('POST','/api/jobs/fixture/cancel','fixture-token')]


@pytest.mark.parametrize('cancel',[False,True])
def test_silent_loopback_response_times_out_for_poll_and_cancel(native_nodes,local_http,monkeypatch,cancel):
    import time
    base,received=local_http
    monkeypatch.setattr(native_nodes,'_READ_TIMEOUT',.1)
    monkeypatch.setattr(native_nodes,'_CANCEL_TIMEOUT',.1)
    start=time.monotonic()
    with pytest.raises(TimeoutError):native_nodes.request(base,'/slow/cancel' if cancel else '/slow',{} if cancel else None,token='fixture-token')
    assert time.monotonic()-start<2 and received


@pytest.mark.parametrize('where',['before_upload','after_image','after_mesh','while_running'])
def test_interruption_boundaries_cancel_only_an_existing_remote_job(bridge,monkeypatch,where):
    def interrupt():raise bridge.Interrupted('original interrupt')
    if where=='before_upload':bridge.mm.throw_exception_if_processing_interrupted=interrupt
    elif where in {'after_image','after_mesh'}:
        original=bridge.nodes.upload
        def upload(*args):
            result=original(*args)
            if len(bridge.uploads)==(1 if where=='after_image' else 2):bridge.mm.throw_exception_if_processing_interrupted=interrupt
            return result
        monkeypatch.setattr(bridge.nodes,'upload',upload)
    else:
        original=bridge.nodes.request
        def request(base,path,*args,**kwargs):
            if path.endswith('/cancel'):
                bridge.calls.append('cancel_attempt')
                raise RuntimeError('cancel endpoint unavailable')
            result=original(base,path,*args,**kwargs)
            if path=='/api/jobs':bridge.mm.throw_exception_if_processing_interrupted=interrupt
            return result
        monkeypatch.setattr(bridge.nodes,'request',request)
    with pytest.raises(bridge.Interrupted,match='original interrupt'):bridge.run()
    assert any(path=='/api/jobs' for path,_ in bridge.requests)==(where=='while_running')
    assert len(bridge.uploads)=={'before_upload':0,'after_image':1,'after_mesh':2,'while_running':2}[where]
    assert ('cancel_attempt' in bridge.calls)==(where=='while_running')


@pytest.mark.parametrize('image,mask',[
    (np.ones((1,2,2,1)),None),
    (np.full((1,2,2,3),np.nan),None),
    (np.ones((1,2,2,3)),np.full((1,2,2),np.nan)),
    (np.ones((1,2,2,3)),np.ones((1,1,2,2))),
])
def test_invalid_reference_values_reject_before_upload(bridge,image,mask):
    with pytest.raises(ValueError):bridge.run(Tensor(image),None if mask is None else Tensor(mask))
    assert not bridge.uploads and not bridge.requests and not bridge.calls


def test_matching_soft_image_and_mask_alpha_is_not_applied_twice(bridge):
    rgba=np.ones((1,2,2,4),dtype=np.float32);rgba[...,3]=.5
    bridge.run(Tensor(rgba),Tensor(np.full((1,2,2),.5)))
    assert np.asarray(Image.open(io.BytesIO(bridge.uploads[0][1])))[...,3].tolist()==[[128,128],[128,128]]
