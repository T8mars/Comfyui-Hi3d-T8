"""Local, on-demand TRELLIS checkpoints with bounded optional CPU residency.

The pinned architectures and checkpoint precision are retained. Random
initialization that would immediately be overwritten is skipped; missing
parameters fall back to the original loader rather than uninitialized weights.
"""
from collections import OrderedDict
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import sys
import time


class ModelCache:
    def __init__(self):
        self.entries = OrderedDict()
        self.root = None
        self.policy = 'release'
        self.bytes = 0
        self.hits = self.loads = self.evictions = 0

    def configure(self, root, policy):
        if policy not in {'release','resident'}:raise ValueError('无效模型驻留模式')
        root = str(Path(root).resolve())
        if root != self.root or policy != 'resident':self.clear()
        self.root, self.policy = root, policy

    def clear(self):
        self.entries.clear();self.bytes=0
        gc.collect()

    def take(self, key):
        value=self.entries.pop(key,None)
        if value is not None:
            self.bytes-=value[1];self.hits+=1
            return value[0]

    def put(self, key, value):
        if self.policy!='resident':return
        import psutil
        module=getattr(value,'model',value)
        tensors=list(module.parameters())+list(module.buffers())
        size=sum(t.numel()*t.element_size() for t in tensors)
        memory=psutil.virtual_memory()
        limit=min(24*2**30,int(memory.total*.35))
        reserve=max(4*2**30,int(memory.total*.1))
        while self.entries and (self.bytes+size>limit or memory.available<reserve):
            _,(_,removed)=self.entries.popitem(last=False)
            self.bytes-=removed;self.evictions+=1
            gc.collect();memory=psutil.virtual_memory()
        if size<=limit and memory.available>=reserve:
            self.entries[key]=(value,size);self.bytes+=size

    def status(self):
        return dict(scope='runtime_session_cumulative',policy=self.policy,entries=len(self.entries),bytes=self.bytes,
                    hits=self.hits,loads=self.loads,evictions=self.evictions)


CACHE=ModelCache()
ACTIVE=[]


@contextmanager
def normal_weight_tensors():
    # Deferred loading can happen inside the caller's inference_mode().
    # Weights must remain regular tensors so a later no_grad() pipeline can
    # reuse them (HF embeddings inspect version counters on views).
    torch=sys.modules.get('torch')
    if torch is None:
        yield
    else:
        with torch.inference_mode(False),torch.no_grad():yield


@contextmanager
def preserve_rng():
    torch=sys.modules.get('torch')
    if torch is None:yield
    else:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []):yield


def fingerprint(stem, *, encoder=False):
    stem=Path(stem).resolve()
    paths=[stem/'config.json',*sorted(stem.glob('*.safetensors'))] if encoder else [Path(str(stem)+'.json'),Path(str(stem)+'.safetensors')]
    if not paths or (encoder and len(paths)<2) or any(not p.is_file() for p in paths):raise FileNotFoundError(str(stem))
    return tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns) for p in paths)


@contextmanager
def skip_random_initialization(torch):
    names=('uniform_','normal_','trunc_normal_','xavier_uniform_','xavier_normal_',
           'kaiming_uniform_','kaiming_normal_','orthogonal_','sparse_')
    original={name:getattr(torch.nn.init,name) for name in names}
    try:
        for name in names:setattr(torch.nn.init,name,lambda tensor,*a,**kw:tensor)
        yield
    finally:
        for name,value in original.items():setattr(torch.nn.init,name,value)


def fast_checkpoint(models, original, path, **kwargs):
    import torch
    from safetensors.torch import load_file
    config=json.loads(Path(str(path)+'.json').read_text(encoding='utf-8'))
    constructor=getattr(models,config['name'])
    # Keep computed buffers on real CPU storage. Meta construction would lose
    # non-checkpoint buffers such as official rotary position embeddings.
    with normal_weight_tensors(),torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []):
        with skip_random_initialization(torch):model=constructor(**config['args'],**kwargs)
        weights=load_file(str(path)+'.safetensors',device='cpu')
        expected=model.state_dict()
        parameters=dict(model.named_parameters())
        if any(name not in weights for name in parameters):
            del model,weights,expected,parameters
            gc.collect()
            return original(str(path),**kwargs)
        # assign=True must preserve the constructor's mixed precision, not
        # blindly adopt checkpoint BF16 for the FP32 input/output layers.
        for name,value in tuple(weights.items()):
            if name in expected and value.dtype!=expected[name].dtype:
                weights[name]=value.to(dtype=expected[name].dtype)
        model.load_state_dict(weights,strict=False,assign=True)
        model.eval()
        return model


class LazyModel:
    def __init__(self, label, key, factory, emit):
        object.__setattr__(self,'_label',label)
        object.__setattr__(self,'_key',key)
        object.__setattr__(self,'_factory',factory)
        object.__setattr__(self,'_emit',emit)
        object.__setattr__(self,'_value',None)
        object.__setattr__(self,'_device','cpu')
        object.__setattr__(self,'_settings',{})

    def materialize(self):
        value=self._value
        if value is None:
            started=time.perf_counter()
            value=CACHE.take(self._key)
            hit=value is not None
            self._emit('loading',phase='model_cache' if hit else 'model_read',device='cpu',
                message=f'{"复用常驻模型" if hit else "按需加载"} {self._label}',model_cache_hit=hit)
            if value is None:
                with normal_weight_tensors(),preserve_rng():value=self._factory()
                CACHE.loads+=1
            for name,setting in self._settings.items():setattr(value,name,setting)
            if self._device!='cpu':
                with normal_weight_tensors():value.to(self._device)
            object.__setattr__(self,'_value',value)
            self._emit('loading',phase='model_cache' if hit else 'model_read',device=self._device if self._device in {'cpu','cuda'} else 'cuda',
                message=f'{self._label} 已就绪（{time.perf_counter()-started:.1f} 秒）',model_cache_hit=hit,
                model_load_seconds=round(time.perf_counter()-started,3))
        return value

    def __getattr__(self,name):return getattr(self.materialize(),name)
    def __setattr__(self,name,value):
        if name.startswith('_'):object.__setattr__(self,name,value);return
        self._settings[name]=value
        if self._value is not None:setattr(self._value,name,value)
    def __call__(self,*args,**kwargs):
        value=self.materialize()
        if 'DINOv3' in self._label:
            self._emit('conditioning',device='cuda',message='执行参考图编码')
        elif 'decoder' in self._label:
            self._emit('decoding',device='cuda',message=f'执行 {self._label}')
        return value(*args,**kwargs)
    def eval(self):return self
    def to(self,device,*args,**kwargs):
        object.__setattr__(self,'_device',str(device))
        if self._value is not None:
            with normal_weight_tensors():self._value.to(device,*args,**kwargs)
        return self
    def cuda(self):return self.to('cuda')
    def cpu(self):
        object.__setattr__(self,'_device','cpu')
        if self._value is not None:
            with normal_weight_tensors():self._value.cpu()
        return self
    def release(self):
        object.__setattr__(self,'_device','cpu')
        value=self._value
        if value is not None:
            with normal_weight_tensors():value.cpu()
            object.__setattr__(self,'_value',None)
            CACHE.put(self._key,value)
            self._emit('releasing',phase='model_release',device='cpu',
                message=f'{self._label} 已结束，{"保留在内存缓存" if self._key in CACHE.entries else "释放权重"}')
            del value
            gc.collect()
        return self


def release(value):
    if isinstance(value,LazyModel):value.release()


def flow_arguments(args,kwargs):
    """Unwrap the typed flow_model argument of sample_tex_slat(cond, model, …)."""
    value=kwargs.get('flow_model',args[1] if len(args)>1 else None)
    if isinstance(value,LazyModel):
        if 'flow_model' in kwargs:kwargs={**kwargs,'flow_model':value.materialize()}
        else:args=(args[0],value.materialize(),*args[2:])
    return args,kwargs


def finish_stage(success=True):
    had_active=bool(ACTIVE)
    if not success:
        CACHE.clear();CACHE.policy='release'
    for value in ACTIVE: value.release()
    ACTIVE.clear()
    gc.collect()
    import torch
    if had_active and torch.cuda.is_initialized():torch.cuda.empty_cache()


@contextmanager
def deferred_models(models, expected, image_module, request, emit):
    """Construct the official pipeline with lazy handles, then restore globals."""
    CACHE.configure(request['model_root'],request.get('model_policy','release'))
    original=models.from_pretrained
    encoder=image_module.DinoV3FeatureExtractor
    labels={os.path.normcase(str(Path(path).resolve())):label for label,path in expected.items()}
    proxies=[]
    def checkpoint(path,**kwargs):
        identity=os.path.normcase(str(Path(path).resolve()))
        key=('checkpoint',fingerprint(path),json.dumps(kwargs,sort_keys=True,default=str))
        value=LazyModel(labels.get(identity,Path(path).name),key,
            lambda:fast_checkpoint(models,original,path,**kwargs),emit)
        proxies.append(value)
        ACTIVE.append(value)
        return value
    def image_encoder(*args,**kwargs):
        path=kwargs.get('model_name',args[0] if args else None)
        key=('encoder',fingerprint(path,encoder=True))
        value=LazyModel('DINOv3 图像编码器',key,lambda:encoder(*args,**kwargs),emit)
        proxies.append(value)
        ACTIVE.append(value)
        return value
    models.from_pretrained=checkpoint
    image_module.DinoV3FeatureExtractor=image_encoder
    try:yield proxies
    finally:
        models.from_pretrained=original
        image_module.DinoV3FeatureExtractor=encoder
