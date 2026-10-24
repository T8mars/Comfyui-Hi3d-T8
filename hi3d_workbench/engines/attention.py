"""Explicit Windows SDPA adapter for upstream TRELLIS.2 packed attention."""
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


def install():
    import importlib.machinery
    import sys
    import types
    from trellis2.modules.sparse import config
    from trellis2.modules.sparse.attention import full_attn, windowed_attn
    config.set_attn_backend("flash_attn")
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
    print("[HI3D] Sparse attention: explicit PyTorch SDPA compatibility; no native FlashAttention distribution", flush=True)
