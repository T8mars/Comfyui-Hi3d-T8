"""Backend selection and official-step reporting; no weights or GPU required."""
import sys
import types
import pytest


@pytest.fixture
def backends(monkeypatch):
    from hi3d_workbench.engines import attention
    sparse = types.SimpleNamespace(set_attn_backend=lambda value: None)
    dense = types.SimpleNamespace(set_backend=lambda value: None)
    full, window = types.SimpleNamespace(), types.SimpleNamespace()
    for name, module in {
        'trellis2': types.ModuleType('trellis2'),
        'trellis2.modules': types.ModuleType('trellis2.modules'),
        'trellis2.modules.sparse': types.SimpleNamespace(config=sparse),
        'trellis2.modules.sparse.attention': types.SimpleNamespace(full_attn=full,windowed_attn=window),
        'trellis2.modules.attention': types.SimpleNamespace(config=dense),
    }.items(): monkeypatch.setitem(sys.modules,name,module)
    monkeypatch.setitem(sys.modules,'flash_attn',None)
    return attention


def test_auto_records_real_missing_native_reason(backends):
    events=[]
    report=backends.install('auto',emit=lambda stage,**event:events.append(event))
    assert report['sparse']=='sdpa' and report['dense']=='sdpa'
    assert 'ModuleNotFoundError' in report['fallback_reason']
    assert sys.modules['flash_attn'].__hi3d_sdpa_compat__
    assert events[-1]['attention']==report and events[-1]['device']=='cuda'


def test_explicit_flash_fails_clearly_without_silent_substitution(backends):
    with pytest.raises(RuntimeError,match='FlashAttention'):
        backends.install('flash_attn')


def test_native_retained_only_after_kernel_probe(backends,monkeypatch):
    native=types.SimpleNamespace(__version__='2.8.3')
    monkeypatch.setitem(sys.modules,'flash_attn',native)
    probed=[]
    monkeypatch.setattr(backends,'probe_native',lambda module:probed.append(module) or {'status':'passed'})
    monkeypatch.setattr(backends,'_native_adapter',lambda module,report:module)
    report=backends.install('auto')
    assert probed==[native] and sys.modules['flash_attn'] is native
    assert report['sparse']=='flash_attn' and report['version']=='2.8.3'


def test_native_abi_or_kernel_failure_has_explicit_fallback(backends,monkeypatch):
    monkeypatch.setitem(sys.modules,'flash_attn',types.SimpleNamespace())
    def failed(module):raise RuntimeError('no kernel image')
    monkeypatch.setattr(backends,'probe_native',failed)
    report=backends.install('auto')
    assert report['sparse']=='sdpa' and 'no kernel image' in report['fallback_reason']


def test_explicit_sdpa_does_not_probe_native(backends,monkeypatch):
    def forbidden(module):raise AssertionError('must not probe')
    monkeypatch.setattr(backends,'probe_native',forbidden)
    assert backends.install('sdpa')['sparse']=='sdpa'


@pytest.mark.parametrize('dtype,device,dim,expected',[('half',True,128,'native'),('float',True,128,'sdpa'),('half',False,64,'sdpa'),('half',True,32,'sdpa')])
def test_guard_preserves_dtype_and_window_keyword_protocol(monkeypatch,dtype,device,dim,expected):
    from hi3d_workbench.engines.attention import _native_adapter,TrellisSDPA
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(float16='half',bfloat16='bf16'))
    calls=[]
    def native(*args,**kw):calls.append(('native',args,kw));return 'native'
    def sdpa(*args,**kw):calls.append(('sdpa',args,kw));return 'sdpa'
    native_module=types.SimpleNamespace(flash_attn_varlen_qkvpacked_func=native,
        flash_attn_varlen_kvpacked_func=native,flash_attn_varlen_func=native,__version__='2.8.3',__path__=['original/package'])
    for name in ('flash_attn_varlen_qkvpacked_func','flash_attn_varlen_kvpacked_func','flash_attn_varlen_func'):
        monkeypatch.setattr(TrellisSDPA,name,staticmethod(sdpa))
    value=types.SimpleNamespace(is_cuda=device,dtype=dtype,shape=(7,3,2,dim))
    class Scalar:
        def __int__(self):return 7
    report={};adapter=_native_adapter(native_module,report)
    assert adapter.__path__==['original/package']
    assert adapter.flash_attn_varlen_qkvpacked_func(value,cu_seqlens=[0,7],max_seqlen=Scalar())==expected
    assert adapter.flash_attn_varlen_kvpacked_func(value,value,cu_seqlens_q=[0,7],cu_seqlens_k=[0,7],max_seqlen_q=Scalar(),max_seqlen_k=Scalar())==expected
    assert adapter.flash_attn_varlen_func(value,value,value,cu_seqlens_q=[0,7],cu_seqlens_k=[0,7],max_seqlen_q=Scalar(),max_seqlen_k=Scalar())==expected
    assert all(item[1][0] is value for item in calls)
    assert calls[0][1][-1]==7
    assert report['packed_calls']['flash_attn' if expected=='native' else 'sdpa']==3


class Sampler:
    def sample_once(self,value):return value+1
    def sample(self,value,steps=3):
        for _ in range(steps):value=self.sample_once(value)
        return value


def test_official_sampler_step_results_and_rounds_preserved():
    from hi3d_workbench.engines.sampler_progress import track_sampler
    sampler=Sampler();events=[]
    track_sampler(sampler,lambda stage,**event:events.append(event),'geometry','Geometry')
    assert sampler.sample(0,steps=3)==3 and sampler.sample(4,steps=2)==6
    assert [(e['pass_index'],e['step'],e['num_inference_steps']) for e in events]==[
        (0,0,3),(0,1,3),(0,2,3),(0,3,3),(1,0,2),(1,1,2),(1,2,2)]
    assert all(e['device']=='cuda' for e in events)
    assert 'sample_once' not in sampler.__dict__


def test_failed_sampler_does_not_claim_completed_step_and_restores_override():
    from hi3d_workbench.engines.sampler_progress import track_sampler
    sampler=Sampler();events=[]
    def fail(value):raise ValueError('failed kernel')
    sampler.sample_once=fail
    track_sampler(sampler,lambda stage,**event:events.append(event),'material','PBR')
    with pytest.raises(ValueError,match='failed kernel'):sampler.sample(1)
    assert [event['step'] for event in events]==[0]
    assert sampler.sample_once is fail


def test_checkpoint_observer_preserves_loader_and_counts_only_actual_model_paths(tmp_path):
    from hi3d_workbench.engines.loading_progress import checkpoint_progress
    seen=[];events=[];output=object()
    def load(path,*args,**kwargs):seen.append((path,args,kwargs));return output
    module=types.SimpleNamespace(from_pretrained=load)
    path=str(tmp_path/'model')
    with checkpoint_progress(module,{'flow':path},lambda stage,**kw:events.append(kw)):
        assert module.from_pretrained('unmatched',1,strict=False) is output
        assert module.from_pretrained(path,2,strict=False) is output
    assert module.from_pretrained is load
    assert seen==[('unmatched',(1,),{'strict':False}),(path,(2,),{'strict':False})]
    assert [(event['progress_current'],event['progress_total']) for event in events]==[(0,2),(1,2)]
    assert all(event['device']=='cpu' for event in events)


def test_checkpoint_observer_restores_original_on_loader_error(tmp_path):
    from hi3d_workbench.engines.loading_progress import checkpoint_progress
    def load(path):raise OSError('disk read failed')
    module=types.SimpleNamespace(from_pretrained=load);events=[]
    with pytest.raises(OSError,match='disk read failed'):
        with checkpoint_progress(module,{'flow':'model'},lambda stage,**kw:events.append(kw)):
            module.from_pretrained('model')
    assert module.from_pretrained is load
    assert [event['progress_current'] for event in events]==[0]
