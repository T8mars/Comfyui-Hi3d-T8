"""Explicit optional Windows port. Never installed in the official mode.

Implements the two packed FlashAttention calls using fused PyTorch SDPA.
No model/weights/sampler changes. Numerical equivalence still needs validation.
"""
class SDPACompat:
    @staticmethod
    def _run(q, k, v, scale=None):
        import torch
        import torch.nn.functional as F
        from torch.nn.attention import sdpa_kernel, SDPBackend
        # Refuse quadratic math fallback on CUDA, which can allocate enormous N*N.
        backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION] if q.is_cuda else [SDPBackend.MATH]
        with sdpa_kernel(backends):
            return F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], dropout_p=0., scale=scale)[0].transpose(0, 1)

    @classmethod
    def flash_attn_varlen_qkvpacked_func(cls, qkv, cu_seqlens, max_seqlen, dropout_p=0., softmax_scale=None, causal=False, **kw):
        import torch
        if dropout_p or causal or kw:
            raise ValueError("Unsupported attention options")
        spans = cu_seqlens.tolist()
        # VAE has many equal-sized small patches: process them as a batch.
        lengths = [b-a for a,b in zip(spans, spans[1:])]
        if lengths and len(set(lengths)) == 1 and lengths[0] and len(lengths) > 1:
            import torch.nn.functional as F
            from torch.nn.attention import sdpa_kernel, SDPBackend
            packed = qkv.reshape(len(lengths), lengths[0], 3, *qkv.shape[2:])
            q,k,v = [packed[:,:,i].transpose(1,2) for i in range(3)]
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION] if q.is_cuda else [SDPBackend.MATH]):
                return F.scaled_dot_product_attention(q,k,v,dropout_p=0.,scale=softmax_scale).transpose(1,2).reshape(-1,*qkv.shape[2:])
        return torch.cat([cls._run(*qkv[a:b].unbind(1), scale=softmax_scale) for a,b in zip(spans,spans[1:])])

    @classmethod
    def flash_attn_varlen_kvpacked_func(cls, q, kv, cu_q, cu_k, max_q, max_k, dropout_p=0., softmax_scale=None, causal=False, **kw):
        import torch
        if dropout_p or causal or kw:
            raise ValueError("Unsupported attention options")
        qs, ks = cu_q.tolist(), cu_k.tolist()
        return torch.cat([cls._run(q[a:b], *kv[c:d].unbind(1), scale=softmax_scale) for a,b,c,d in zip(qs, qs[1:],ks,ks[1:])])

def install_sdpa():
    from sparc3d.nn.sparse.attention import full_attn
    from sparc3d.models.autoencoders import blocks
    full_attn._flash_attn = lambda: SDPACompat
    blocks._flash_attn = lambda: SDPACompat
