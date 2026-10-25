"""Verified native FlashAttention, with an explicit fused SDPA fallback.

Only the isolated workbench TRELLIS process is configured here. Hi3D nodes
and UltraTex's trained sparse-attention path keep their own implementations.
"""
from ..attention import SDPACompat


class TrellisSDPA(SDPACompat):
    @classmethod
    def flash_attn_varlen_kvpacked_func(cls, q, kv, cu_seqlens_q, cu_seqlens_k,
                                      max_seqlen_q, max_seqlen_k, **kwargs):
        return super().flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k,
                                                    max_seqlen_q, max_seqlen_k, **kwargs)

    @classmethod
    def flash_attn_varlen_func(cls, q, k, v, cu_seqlens_q, cu_seqlens_k,
                               max_seqlen_q, max_seqlen_k, **kwargs):
        import torch
        return cls.flash_attn_varlen_kvpacked_func(q, torch.stack([k, v], dim=1),
                                                 cu_seqlens_q, cu_seqlens_k,
                                                 max_seqlen_q, max_seqlen_k, **kwargs)


def probe_native(module):
    """Exercise the actual CUDA packed kernels, not just the Python import."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    if getattr(module, '__hi3d_sdpa_compat__', False):
        raise RuntimeError('flash_attn is an SDPA adapter, not the native package')
    functions = ('flash_attn_varlen_qkvpacked_func', 'flash_attn_varlen_kvpacked_func',
                 'flash_attn_varlen_func')
    if any(not callable(getattr(module, name, None)) for name in functions):
        raise RuntimeError('Native FlashAttention packed API is incomplete')
    # Unequal, non-power-of-two windows and unequal cross attention lengths
    # cover the protocol that upstream sparse layers actually use. Keep RNG
    # state intact so enabling acceleration does not change the requested seed.
    with torch.inference_mode(), torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(317)
        for dtype, head_dim in ((dtype, dim) for dtype in (torch.float16, torch.bfloat16) for dim in (64,128)):
            qkv = torch.randn(98, 3, 2, head_dim, device='cuda', dtype=dtype)
            cu = torch.tensor([0, 1, 34, 98], device='cuda', dtype=torch.int32)
            q = qkv[:, 0].contiguous()
            kv = torch.randn(104, 2, 2, head_dim, device='cuda', dtype=dtype)
            ck = torch.tensor([0, 3, 40, 104], device='cuda', dtype=torch.int32)
            pairs = ((module.flash_attn_varlen_qkvpacked_func(qkv, cu, 64),
                      TrellisSDPA.flash_attn_varlen_qkvpacked_func(qkv, cu, 64)),
                     (module.flash_attn_varlen_kvpacked_func(q, kv, cu, ck, 64, 64),
                      TrellisSDPA.flash_attn_varlen_kvpacked_func(q, kv, cu, ck, 64, 64)),
                     (module.flash_attn_varlen_func(q, kv[:, 0], kv[:, 1], cu, ck, 64, 64),
                      TrellisSDPA.flash_attn_varlen_func(q, kv[:, 0], kv[:, 1], cu, ck, 64, 64)))
            for actual, expected in pairs:
                if not torch.isfinite(actual).all() or not torch.allclose(actual, expected, atol=.025, rtol=.025):
                    raise RuntimeError(f'Native packed attention numerical probe failed ({dtype})')
        torch.cuda.synchronize()
    return {'status': 'passed', 'dtypes': ['float16', 'bfloat16'], 'head_dims':[64,128],
            'device': torch.cuda.get_device_name(), 'capability': list(torch.cuda.get_device_capability())}


def _native_adapter(native, report):
    """Guard native ABI constraints without changing input precision."""
    import importlib.machinery
    import types
    import torch
    adapter=types.ModuleType('flash_attn', 'Hi3D guarded native packed attention; explicit SDPA for unsupported tensor types')
    adapter.__dict__.update(native.__dict__)
    adapter.__hi3d_native_module__=native
    adapter.__version__=getattr(native,'__version__','unknown')
    counts=report.setdefault('packed_calls',{'flash_attn':0,'sdpa':0})

    def dispatch(name, tensors, args, kwargs):
        # The official flow is BF16; never silently downcast an FP32 caller
        # just to satisfy a faster kernel. CUDA math SDPA remains forbidden.
        supported=all(value.is_cuda and value.dtype in (torch.float16,torch.bfloat16)
                      and value.shape[-1] in (64,128) for value in tensors)
        supported=supported and len({value.dtype for value in tensors})==1
        if not supported:
            counts['sdpa']+=1
            if counts['sdpa']==1:
                print('[HI3D] Packed call uses fused SDPA: native tensor dtype/device/head dimension unsupported; input precision retained',flush=True)
            return getattr(TrellisSDPA,name)(*args,**kwargs)
        counts['flash_attn']+=1
        return getattr(native,name)(*args,**kwargs)

    def qkv(qkv,cu_seqlens,max_seqlen,**kw):
        return dispatch('flash_attn_varlen_qkvpacked_func',(qkv,),(qkv,cu_seqlens,int(max_seqlen)),kw)
    def kv(q,kv,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k,**kw):
        return dispatch('flash_attn_varlen_kvpacked_func',(q,kv),(q,kv,cu_seqlens_q,cu_seqlens_k,int(max_seqlen_q),int(max_seqlen_k)),kw)
    def separate(q,k,v,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k,**kw):
        return dispatch('flash_attn_varlen_func',(q,k,v),(q,k,v,cu_seqlens_q,cu_seqlens_k,int(max_seqlen_q),int(max_seqlen_k)),kw)
    adapter.flash_attn_varlen_qkvpacked_func=qkv
    adapter.flash_attn_varlen_kvpacked_func=kv
    adapter.flash_attn_varlen_func=separate
    return adapter


def install(preference='auto', *, emit=None):
    import importlib.machinery
    import sys
    import types
    import importlib
    if preference not in {'auto', 'flash_attn', 'sdpa'}:
        raise ValueError('Unknown TRELLIS attention preference')
    from trellis2.modules.sparse import config
    from trellis2.modules.sparse.attention import full_attn, windowed_attn
    from trellis2.modules.attention import config as dense_config
    dense_config.set_backend('sdpa')
    config.set_attn_backend("flash_attn")
    report = {'requested': preference, 'sparse': 'sdpa', 'dense': 'sdpa', 'device': 'cuda'}
    if preference != 'sdpa':
        try:
            previous = sys.modules.get('flash_attn')
            if getattr(previous, '__hi3d_sdpa_compat__', False):
                sys.modules.pop('flash_attn')
            elif getattr(previous,'__hi3d_native_module__',None) is not None:
                sys.modules['flash_attn']=previous.__hi3d_native_module__
            native = importlib.import_module('flash_attn')
            report['probe'] = probe_native(native)
            report.update(sparse='flash_attn', version=getattr(native, '__version__', 'unknown'))
            sys.modules['flash_attn']=_native_adapter(native,report)
            for attention in (full_attn, windowed_attn):
                attention.__dict__.pop('flash_attn', None)
            message = '[HI3D] TRELLIS sparse attention: native FlashAttention ' + report['version'] + '; CUDA packed-kernel probe passed; dense SDPA'
            print(message, flush=True)
            if emit: emit('attention', phase='attention_probe', device='cuda', message=message, attention=report)
            return report
        except Exception as exc:
            report['fallback_reason'] = f'{type(exc).__name__}: {exc}'
            if preference == 'flash_attn':
                raise RuntimeError('原生 FlashAttention 不可用；请补装匹配轮子或选择自动/SDPA：' + report['fallback_reason']) from exc
    # Upstream imports flash_attn *inside* each function, so injecting its
    # module globals skips that import and leaves an unbound local variable.
    # Register the explicitly declared SDPA protocol shim in this isolated
    # inference process, letting the official lazy import bind its local.
    shim = types.ModuleType("flash_attn", "Hi3D explicit PyTorch SDPA protocol adapter; not native FlashAttention")
    shim.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    shim.__hi3d_sdpa_compat__ = True
    for name in ("flash_attn_varlen_qkvpacked_func", "flash_attn_varlen_kvpacked_func", "flash_attn_varlen_func"):
        setattr(shim, name, getattr(TrellisSDPA, name))
    sys.modules["flash_attn"] = shim
    full_attn.__dict__.pop("flash_attn", None)
    windowed_attn.__dict__.pop("flash_attn", None)
    message = '[HI3D] TRELLIS sparse attention: fused PyTorch SDPA compatibility; dense SDPA'
    if report.get('fallback_reason'):
        message += '; native fallback: ' + report['fallback_reason']
    print(message, flush=True)
    if emit: emit('attention', phase='attention_probe', device='cuda', message=message, attention=report)
    return report
