"""Staged FP32 X-Part sampling with independent conditional CFG groups.

The public pipeline batches CFG as additional parts, although its global part
attention and part embeddings assume one object. Keep both branches separate
and share their part-embedding RNG. No point, latent or layer is omitted.
"""
from __future__ import annotations

import gc
import time
from contextlib import contextmanager

import numpy as np


@contextmanager
def condition_progress(conditioner,emit):
    stages=[('局部几何',conditioner.geo_encoder),('整体几何',conditioner.obj_encoder.encoder),
            ('部件语义特征',conditioner.seg_feat_encoder)]
    hooks=[]
    for index,(label,module) in enumerate(stages):
        def start(module,inputs,index=index,label=label):
            emit('asset_part_condition','正在编码'+label,device='cuda',progress_current=index,
                 progress_total=len(stages),progress_unit='encoders')
        def complete(module,inputs,output,index=index,label=label):
            emit('asset_part_condition',label+'已编码',device='cuda',progress_current=index+1,
                 progress_total=len(stages),progress_unit='encoders')
        hooks.extend([module.register_forward_pre_hook(start),module.register_forward_hook(complete)])
    try:yield
    finally:
        for hook in hooks:hook.remove()


def guided_prediction(model,latents,timestep,condition,*,boxes,num_tokens,guidance,unconditional=None):
    import torch
    if not np.isfinite(guidance) or not -1<=guidance<=15:raise ValueError('X-Part 引导强度无效')
    cpu_before=torch.get_rng_state()
    cuda_before=torch.cuda.get_rng_state(latents.device) if latents.is_cuda else None
    conditional=model(latents,timestep,condition,aabb=boxes,num_tokens=num_tokens,guidance_cond=None)
    if guidance<0:return conditional
    cpu_after=torch.get_rng_state()
    cuda_after=torch.cuda.get_rng_state(latents.device) if latents.is_cuda else None
    torch.set_rng_state(cpu_before)
    if cuda_before is not None:torch.cuda.set_rng_state(cuda_before,latents.device)
    try:
        empty=unconditional if unconditional is not None else {key:torch.zeros_like(value) for key,value in condition.items()}
        baseline=model(latents,timestep,empty,aabb=boxes,num_tokens=num_tokens,guidance_cond=None)
    finally:
        torch.set_rng_state(cpu_after)
        if cuda_after is not None:torch.cuda.set_rng_state(cuda_after,latents.device)
    return baseline+guidance*(conditional-baseline)


def generate(pipe,*,mesh,boxes,steps,resolution,seed,guidance,emit,callback):
    import torch,trimesh
    from partgen.partformer_pipeline import retrieve_timesteps
    if not 1<=guidance<=15:raise ValueError('X-Part 引导强度需在 1–15 之间')
    torch.manual_seed(seed);np.random.seed(seed)
    started=time.monotonic();device=torch.device('cuda')
    pipe.device=device;pipe.dtype=torch.float32
    emit('asset_part_condition','读取原部件与整体形状，计算几何条件',device='cpu')
    obj,bbox,parts,_,center,scale=pipe.check_inputs(mesh=mesh,aabb=boxes,seed=seed)
    if not np.allclose(center,0) or abs(scale-1)>1e-9:
        raise RuntimeError('X-Part 输入未按已记录的坐标归一化')
    count=parts.shape[1]
    if count!=len(boxes):raise RuntimeError('X-Part 部分部件缺少有效表面')
    pipe.conditioner.to(device=device,dtype=torch.float32)
    emit('asset_part_condition','编码部件与整体几何条件',device='cuda')
    with condition_progress(pipe.conditioner,emit),torch.inference_mode(),torch.autocast('cuda',enabled=False):
        part_gpu=parts.to(device=device,dtype=torch.float32)
        obj_gpu=obj.to(device=device,dtype=torch.float32).expand(count,-1,-1)
        condition=pipe.conditioner(part_gpu.reshape(count,*part_gpu.shape[-2:]),obj_gpu)
    if not isinstance(condition,dict) or any(not torch.isfinite(value).all() for value in condition.values()):
        raise RuntimeError('X-Part 几何条件包含无效数值')
    del obj,parts,part_gpu,obj_gpu
    pipe.conditioner=None;gc.collect();torch.cuda.empty_cache()
    condition_seconds=time.monotonic()-started
    emit('asset_model_transfer','几何编码器已释放，加载 X-Part 采样模型',device='cuda')
    pipe.model.to(device=device,dtype=torch.float32)
    bbox=bbox.to(device=device,dtype=torch.float32)
    shape=pipe.vae.latent_shape
    tokens=torch.full((1,count),shape[0],device=device,dtype=torch.int64)
    generator=torch.Generator(device).manual_seed(seed)
    latents=pipe.prepare_latents(count,shape,torch.float32,device,generator)
    timesteps,_=retrieve_timesteps(pipe.scheduler,steps,device,sigmas=np.linspace(0,1,steps))
    unconditional={key:torch.zeros_like(value) for key,value in condition.items()}
    sample_started=time.monotonic()
    with torch.inference_mode(),torch.autocast('cuda',enabled=False):
        for index,t in enumerate(timesteps):
            timestep=t.expand(count).to(latents.dtype)/pipe.scheduler.config.num_train_timesteps
            prediction=guided_prediction(pipe.model,latents,timestep,condition,boxes=bbox,num_tokens=tokens,
                guidance=guidance,unconditional=unconditional)
            if not torch.isfinite(prediction).all():raise RuntimeError('X-Part 几何采样产生无效数值')
            outputs=pipe.scheduler.step(prediction,t,latents);latents=outputs.prev_sample
            callback(index,t,outputs)
    sample_seconds=time.monotonic()-sample_started
    pipe.model=None;del condition,unconditional,prediction,outputs
    gc.collect();torch.cuda.empty_cache()
    emit('asset_part_decode','已释放采样模型，加载几何解码器',device='cuda')
    pipe.vae.to(device=device,dtype=torch.float32)
    decode_started=time.monotonic();scene=trimesh.Scene()
    with torch.inference_mode(),torch.autocast('cuda',enabled=False):
        for index,latent in enumerate(latents):
            emit('asset_part_decode','解码独立部件',device='cuda',progress_current=index,progress_total=count,progress_unit='parts')
            part=pipe._export(latents=latent[None],output_type='trimesh',box_v=1.01,mc_level=-1/512,
                num_chunks=32768,octree_resolution=resolution,mc_algo='mc',enable_pbar=True)[0]
            if not len(part.faces) or not np.isfinite(part.vertices).all():raise RuntimeError('X-Part 没有生成有效部件')
            scene.add_geometry(part,geom_name=f'part_{index:04d}')
    report={'guidance_scale':guidance,'precision':'FP32','sampling':'separate CFG branches with shared part-embedding RNG',
        'staged':True,'seed':seed,'steps':steps,'resolution':resolution,'decode_chunks':32768,
        'condition_seconds':condition_seconds,'sampling_seconds':sample_seconds,'decode_seconds':time.monotonic()-decode_started}
    return scene,report
