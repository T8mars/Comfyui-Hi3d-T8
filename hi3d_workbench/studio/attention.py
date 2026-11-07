"""Bounded, full-precision attention for packed point-cloud patches."""
from __future__ import annotations


def bounded_sdpa(query,key,value,attn_mask=None,dropout_p=0.,is_causal=False,*,scale=None,enable_gqa=False,matrix_elements=8_000_000):
    """Same noncausal attention, bounded along independent query rows.

    X-Part cross-attends 4096 queries to 81920 surface points. On Windows,
    FP32 SDPA's math backend can otherwise allocate a multi-gigabyte matrix.
    Every chunk still attends to ALL keys; no point or attention is discarded.
    """
    import math
    import torch
    from torch.nn.functional import scaled_dot_product_attention
    if dropout_p or is_causal:
        raise RuntimeError('固定 X-Part 推理需要无 dropout、非因果的注意力')
    rows=query.shape[-2]
    leading=torch.broadcast_shapes(query.shape[:-2],key.shape[:-2],value.shape[:-2])
    per_row=max(1,math.prod(leading)*key.shape[-2])
    size=max(1,min(rows,matrix_elements//per_row)) if rows else 1
    result=[]
    for start in range(0,rows,size):
        mask=attn_mask
        if mask is not None and mask.shape[-2]!=1:
            mask=mask[...,start:start+size,:]
        result.append(scaled_dot_product_attention(query[...,start:start+size,:],key,value,
            attn_mask=mask,dropout_p=0.,is_causal=False,scale=scale,enable_gqa=enable_gqa))
    return torch.cat(result,dim=-2) if result else scaled_dot_product_attention(query,key,value,attn_mask=attn_mask,scale=scale,enable_gqa=enable_gqa)


def install_xpart_attention():
    from partgen.models.autoencoders import attention_blocks,attention_processors
    attention_blocks.scaled_dot_product_attention=bounded_sdpa
    attention_processors.scaled_dot_product_attention=bounded_sdpa


def install_xpart_sonata_attention(model):
    """Preserve Sonata's patch ordering/scale without its forced FP16 cast."""
    from types import MethodType
    from partgen.models.sonata.model import SerializedAttention
    def forward(self,point):
        pad,unpad,offsets=self.get_padding_and_inverse(point)
        order=point.serialized_order[self.order_index][pad]
        inverse=unpad[point.serialized_inverse[self.order_index]]
        qkv=self.qkv(point.feat)[order]
        features=packed_sdpa(qkv.reshape(-1,3,self.num_heads,self.channels//self.num_heads),offsets,scale=self.scale)
        point.feat=self.proj_drop(self.proj(features.reshape(-1,self.channels)[inverse]))
        return point
    changed=0
    for module in model.modules():
        if isinstance(module,SerializedAttention):
            if not module.enable_flash or module.enable_rpe:
                raise RuntimeError('固定 X-Part Sonata 配置与 FP32 分块注意力不兼容')
            module.forward=MethodType(forward,module);changed+=1
    if not changed:raise RuntimeError('未找到固定 X-Part Sonata 注意力层')
    return changed


def packed_sdpa(qkv, offsets, *, scale=None):
    import torch
    from torch.nn.functional import scaled_dot_product_attention
    if qkv.ndim!=4 or qkv.shape[1]!=3:
        raise ValueError('点云注意力需要 [points,3,heads,channels]')
    starts=offsets.detach().cpu().tolist() if isinstance(offsets,torch.Tensor) else list(offsets)
    if not starts or starts[0]!=0 or starts[-1]!=len(qkv) or any(b<=a for a,b in zip(starts,starts[1:])):
        raise ValueError('点云注意力片段边界无效')
    outputs=[];index=0
    while index<len(starts)-1:
        length=starts[index+1]-starts[index]
        # Bound the quadratic temporary even when SDPA selects its math
        # backend. Group only consecutive equal-size patches, including the
        # final short patch which cannot be reshaped to the maximum length.
        limit=max(1,min(16,2_000_000//max(1,qkv.shape[2]*length*length)))
        stop=index+1
        while stop<len(starts)-1 and stop-index<limit and starts[stop+1]-starts[stop]==length:
            stop+=1
        packed=qkv[starts[index]:starts[stop]].reshape(stop-index,length,3,*qkv.shape[2:])
        query,key,value=packed.permute(2,0,3,1,4).unbind(0)
        result=scaled_dot_product_attention(query.float(),key.float(),value.float(),dropout_p=0.,scale=scale)
        outputs.append(result.transpose(1,2).reshape(-1,*qkv.shape[2:]).to(qkv.dtype))
        index=stop
    return torch.cat(outputs,dim=0)


def install_unirig_attention(model):
    """Same pinned PTv3 patch order/QK normalization, without FP16 packing."""
    from types import MethodType
    from src.model.pointcept.models.PTv3Object import SerializedAttention
    def forward(self,point):
        pad,unpad,offsets=self.get_padding_and_inverse(point)
        order=point.serialized_order[self.order_index][pad]
        inverse=unpad[point.serialized_inverse[self.order_index]]
        qkv=self.qkv(point.feat)[order]
        if self.enable_qknorm:
            qkv=self.qknorm(qkv)
        features=packed_sdpa(qkv.reshape(-1,3,self.num_heads,self.channels//self.num_heads),offsets,scale=self.scale)
        features=features.reshape(-1,self.channels)[inverse]
        point.feat=self.proj_drop(self.proj(features))
        return point
    changed=0
    for module in model.modules():
        if isinstance(module,SerializedAttention):
            if not module.enable_flash or module.enable_rpe:
                raise RuntimeError('固定 UniRig PTv3 配置与分块注意力实现不兼容')
            module.forward=MethodType(forward,module);changed+=1
    if not changed:
        raise RuntimeError('没有找到固定 UniRig 点云注意力层')
    return changed
