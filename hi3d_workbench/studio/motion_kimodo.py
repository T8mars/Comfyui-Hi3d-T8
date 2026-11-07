"""Offline Kimodo inference with staged text encoding and original-rig controls."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from .. import common
from ..common import atomic_json
from . import models
from .animation import Motion,rotations as rotation_frames
from .gltf import Document
from .skin import Rig,extract
from .motion_constraints import validate_controls,transfer_controls

SOURCE_REVISION='e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954'
MODEL_NAME='Kimodo-SOMA-RP-v1.2'


def component(name,emit):
    spec=models.CATALOG[name]
    for index,file in enumerate(spec['files']):
        models.verified_weight(name,file['name'])
        emit('asset_loading',f'校验本地 {name}',device='cpu',progress_current=index+1,
             progress_total=len(spec['files']),progress_unit='files')
    return models.model_root()/name


def text_embedding(prompt,device,emit):
    import torch
    from filelock import FileLock
    from kimodo.model.llm2vec.llm2vec import LLM2Vec
    from transformers import AutoConfig,AutoTokenizer
    from peft import PeftModel
    from kimodo.sanitize import sanitize_text
    prompt=sanitize_text(prompt)
    versions={name:models.CATALOG[name]['revision'] for name in ('text_base','text_adapter','text_adapter_supervised')}
    key=hashlib.sha256(json.dumps({'schema':1,'source':SOURCE_REVISION,'models':versions,
        'prompt':prompt,'device':device,'precision':'bfloat16'},sort_keys=True).encode()).hexdigest()
    folder=common.DATA/'studio-cache'/'text-embeddings';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(key+'.npz')
    started=time.monotonic()
    with FileLock(str(folder/(key+'.lock')),timeout=30):
        if path.is_file():
            try:
                with np.load(path,allow_pickle=False) as cached:
                    vector=cached['embedding'].copy()
                    valid=str(cached['key'])==key and vector.shape==(1,1,4096) and np.isfinite(vector).all()
                if valid:
                    emit('asset_motion_text','复用已编码的动作描述',device='cpu')
                    return vector,{'cache_hit':True,'device':device,'seconds':round(time.monotonic()-started,3)}
            except (OSError,ValueError,KeyError):
                pass
        base=component('text_base',emit);first=component('text_adapter',emit);second=component('text_adapter_supervised',emit)
        emit('asset_motion_text_loading','读取本地 Llama 3 文本编码器与两组适配权重',device=device)
        config=AutoConfig.from_pretrained(str(first),local_files_only=True)
        model_class=LLM2Vec._get_model_class(type(config).__name__,enable_bidirectional=True)
        # Supplying the verified base directly prevents adapter_config.json
        # from resolving its original Hub ID or fetching any remote code.
        from .loading_progress import weight_loading, merge_adapter
        with weight_loading(emit,device):
            model=model_class.from_pretrained(str(base),config=config,dtype=torch.bfloat16,
                device_map=device,local_files_only=True,attn_implementation='sdpa')
        emit('asset_motion_text_adapter','读取第一组文本适配权重',device=device)
        model=PeftModel.from_pretrained(model,str(first),local_files_only=True)
        model=merge_adapter(model,emit,device)
        emit('asset_motion_text_adapter','读取第二组文本适配权重',device=device)
        model=PeftModel.from_pretrained(model,str(second),local_files_only=True)
        model.config._name_or_path='meta-llama/Meta-Llama-3-8B-Instruct'
        tokenizer=AutoTokenizer.from_pretrained(str(first),local_files_only=True)
        tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='left'
        encoder=LLM2Vec(model,tokenizer).eval()
        hooks=[]
        layers=model.get_base_model().layers
        for index,layer in enumerate(layers):
            def progress(module,inputs,output,index=index):
                emit('asset_motion_text','正在编码动作描述',device=device,progress_current=index+1,
                     progress_total=len(layers),progress_unit='layers')
            hooks.append(layer.register_forward_hook(progress))
        try:
            with torch.no_grad():
                encoded=encoder.encode([prompt],batch_size=1,show_progress_bar=False,device=device)
            vector=encoded.detach().float().cpu().numpy()[:,None]
        finally:
            for hook in hooks:hook.remove()
        del encoder,model,tokenizer,encoded,layers,hooks,layer,progress
        gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()
        if vector.shape!=(1,1,4096) or not np.isfinite(vector).all():
            raise RuntimeError('文本编码器产生无效动作向量')
        temporary=folder/(key+'.tmp.npz')
        np.savez_compressed(temporary,key=np.asarray(key),embedding=vector)
        temporary.replace(path)
    emit('asset_motion_text','文本编码完成，编码模型已释放',device='cpu')
    return vector,{'cache_hit':False,'device':device,'seconds':round(time.monotonic()-started,3)}


class CachedTextEncoder:
    def __init__(self,prompt,vector):
        from kimodo.sanitize import sanitize_text
        self.prompt=sanitize_text(prompt);self.vector=vector

    def __call__(self,texts):
        import torch
        from kimodo.sanitize import sanitize_text
        if isinstance(texts,str):texts=[texts]
        if any(sanitize_text(text)!=self.prompt for text in texts):
            raise RuntimeError('动作采样请求了未经本地编码的描述')
        return torch.from_numpy(np.repeat(self.vector,len(texts),axis=0)),[1]*len(texts)


def rig_from_skeleton(skeleton,*,grounded=False):
    points=skeleton.neutral_joints.detach().cpu().numpy().copy()
    if grounded:
        feet=[skeleton.bone_order_names.index(name) for name in
              [*skeleton.left_foot_joint_names,*skeleton.right_foot_joint_names]]
        # SOMA's neutral joints are pelvis-centred, while FullBody constraints
        # are absolute scene positions. Stand the model rig on the ground.
        points[:,1]-=points[feet,1].min()
    worlds=np.repeat(np.eye(4)[None],len(points),axis=0);worlds[:,:3,3]=points
    return Rig(list(skeleton.bone_order_names),skeleton.joint_parents.detach().cpu().numpy().astype(np.int64),worlds)


def export_bvh(motion,skeleton,folder):
    import torch
    from kimodo.exports.bvh import save_motion_bvh
    from kimodo.skeleton import global_rots_to_local_rots
    from .bvh import load
    local=global_rots_to_local_rots(torch.from_numpy(motion.worlds[:,:,:3,:3]),skeleton)
    neutral=skeleton.neutral_joints[skeleton.root_idx].detach().cpu().numpy().copy()
    if np.all(neutral==0):neutral=np.array([0.,1.,0.])
    # Upstream writes both a Hips OFFSET and translation channels. Channels
    # must be relative to that OFFSET, otherwise all BVH joints gain 1 metre.
    root=motion.worlds[:,skeleton.root_idx,:3,3]-neutral
    temporary=folder/'motion.tmp.bvh'
    save_motion_bvh(temporary,local,torch.from_numpy(root),skeleton=skeleton,
                    fps=1/np.median(np.diff(motion.times)),standard_tpose=True)
    checked=load(temporary,scale=.01)
    selected=[checked.rig.names.index(name) for name in motion.rig.names]
    worlds=checked.worlds[:,selected]
    position_error=float(np.linalg.norm(worlds[:,:,:3,3]-motion.worlds[:,:,:3,3],axis=-1).max())
    rotation_error=float(np.max(np.abs(rotation_frames(worlds)-rotation_frames(motion.worlds))))
    if (len(checked.times)!=len(motion.times) or not np.allclose(checked.times,motion.times,atol=1e-5)
            or position_error>1e-4 or rotation_error>1e-4):
        raise RuntimeError(f'BVH 往返验证失败：位置误差 {position_error:.6g}，旋转误差 {rotation_error:.6g}')
    temporary.replace(folder/'motion.bvh')
    return {'position_error_m':position_error,'rotation_matrix_error':rotation_error,'frames':len(checked.times),'units':'centimetres'}


def build_constraints(controls,source,skeleton,request):
    import torch
    from kimodo.constraints import Root2DConstraintSet,FullBodyConstraintSet
    from kimodo.skeleton import SOMASkeleton77
    full=SOMASkeleton77()
    paths,poses=transfer_controls(controls,source,rig_from_skeleton(full,grounded=True),request)
    result=[]
    if paths:
        frames=sorted(paths)
        result.append(Root2DConstraintSet(skeleton,torch.tensor(frames,device='cuda'),
            torch.tensor(np.array([paths[frame] for frame in frames]),device='cuda',dtype=torch.float32)))
    selected=[full.bone_order_names.index(name) for name in skeleton.bone_order_names]
    for pose in poses:
        worlds=pose['worlds'][selected]
        result.append(FullBodyConstraintSet(skeleton,torch.tensor([pose['frame']],device='cuda'),
            torch.tensor(worlds[None,:,:3,3],device='cuda',dtype=torch.float32),
            torch.tensor(worlds[None,:,:3,:3],device='cuda',dtype=torch.float32)))
    return result


def check_constraints(controls,source,motion,request):
    from kimodo.skeleton import SOMASkeleton77,SOMASkeleton30
    paths,poses=transfer_controls(controls,source,rig_from_skeleton(SOMASkeleton77(),grounded=True),request)
    # The upstream 30 -> 77 conversion fills intermediate finger joints with
    # a relaxed hand pose. Fine finger endpoints are not body key-pose targets.
    body=[motion.rig.names.index(name) for name in SOMASkeleton30().bone_order_names if not name.endswith(('ThumbEnd','MiddleEnd'))]
    path_error=max((float(np.linalg.norm(motion.worlds[frame,0,[0,2],3]-point)) for frame,point in paths.items()),default=0.)
    pose_error=max((float(np.linalg.norm(motion.worlds[pose['frame'],body][:,:3,3]-pose['worlds'][body,:3,3],axis=-1).max()) for pose in poses),default=0.)
    rotation_error=max((float(np.abs(motion.worlds[pose['frame'],body][:,:3,:3]-pose['worlds'][body,:3,:3]).max()) for pose in poses),default=0.)
    result={'path_keyframes':len(paths),'body_keyframes':len(poses),'path_error_m':path_error,
            'body_position_error_m':pose_error,'body_rotation_matrix_error':rotation_error,
            'fine_fingers':'upstream relaxed hand pose'}
    if max(path_error,pose_error,rotation_error)>1e-3:
        raise RuntimeError(f'动作约束未通过回读验证：路径 {path_error:.6g}m，姿态 {pose_error:.6g}m，旋转 {rotation_error:.6g}')
    return result


def generate(request,folder,emit):
    folder=Path(folder)
    prompt=str(request.get('prompt','')).strip()
    duration=float(request.get('duration',4));steps=request.get('diffusion_steps',100)
    device=request.get('text_device','cpu')
    if not prompt or not any(char.isalnum() for char in prompt):raise ValueError('请输入动作描述')
    if not np.isfinite(duration) or not .5<=duration<=10 or type(steps) is not int or not 10<=steps<=200:
        raise ValueError('动作时长或采样步数无效')
    if device not in {'cpu','cuda'}:raise ValueError('文本编码设备无效')
    source=extract(Document.load(folder/'coarse.glb'))
    controls=validate_controls(request.get('constraints',[]),duration,len(source.names))
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['TEXT_ENCODER_MODE']='local'
    root=models.runtime_root()/'sources'/'kimodo'
    sys.path.insert(0,str(root))
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Kimodo 需要支持 CUDA 的 NVIDIA 显卡')
    # Check native postprocessing before the much larger text encoder loads.
    # Importing Kimodo alone does not validate this optional upstream package.
    try:
        from motion_correction import motion_postprocess
    except ImportError as exc:
        raise RuntimeError('Kimodo 动作校正扩展未安装或无法加载，请修复资产运行环境') from exc
    from kimodo.skeleton import SOMASkeleton30,SOMASkeleton77
    from kimodo.model.loading import instantiate_from_dict
    from kimodo.tools import seed_everything
    from omegaconf import OmegaConf
    # Validate mappings and controls before allocating the text model.
    transfer_controls(controls,source,rig_from_skeleton(SOMASkeleton77(),grounded=True),request)
    vector,text_report=text_embedding(prompt,device,emit)
    checkpoint=component('kimodo',emit)
    emit('asset_loading','文本编码器已释放，正在加载 Kimodo 动作模型',device='cuda')
    config=OmegaConf.load(checkpoint/'config.yaml')
    config.checkpoint_dir=str(checkpoint)
    config=OmegaConf.to_container(config,resolve=True);config.pop('checkpoint_dir')
    config['text_encoder']=None
    model=instantiate_from_dict(config,overrides={'device':'cuda'}).eval()
    model.text_encoder=CachedTextEncoder(prompt,vector)
    constraints=build_constraints(controls,source,model.skeleton,request)
    def progress(iterable,**kwargs):
        total=len(iterable)
        for index,value in enumerate(iterable):
            yield value
            emit('asset_motion_generate','Kimodo 已完成动作采样步',device='cuda',progress_current=index+1,
                 progress_total=total,progress_unit='steps')
        emit('asset_motion_postprocess','动作采样完成，正在校正足部接触与约束',device='cpu')
    # Model constructors and cache misses consume RNG; reset directly before
    # denoising so identical seeds also reproduce across cache hit/miss runs.
    seed=request.get('seed',0);seed_everything(seed)
    started=time.monotonic();frames=round(duration*model.fps)+1
    output=model([prompt],[frames],num_denoising_steps=steps,num_samples=1,multi_prompt=False,
        constraint_lst=[constraints] if constraints else [],post_processing=True,root_margin=0.,return_numpy=True,progress_bar=progress)
    skeleton=model.output_skeleton.to('cpu')
    rig=rig_from_skeleton(skeleton)
    positions=np.asarray(output['posed_joints'][0]);rotations=np.asarray(output['global_rot_mats'][0])
    worlds=np.repeat(np.eye(4)[None,None],len(positions),axis=0);worlds=np.repeat(worlds,len(rig.names),axis=1)
    worlds[:,:,:3,:3]=rotations;worlds[:,:,:3,3]=positions
    contacts=np.asarray(output['foot_contacts'][0])
    if contacts.shape[-1]==6:contacts=contacts[:,[0,1,3,4]]
    report={'engine':'kimodo-soma-rp-v1.2','source_revision':SOURCE_REVISION,
        'model_revision':models.CATALOG['kimodo']['revision'],'text_encoding':text_report,
        'requested_duration':duration,'fps':float(model.fps),'generated_frames':len(positions),
        'diffusion_steps':steps,'seed':seed,'constraint_count':len(controls),
        'sampling_and_postprocess_seconds':round(time.monotonic()-started,3)}
    motion=Motion(rig,np.arange(len(positions))/model.fps,worlds,prompt[:160],metadata=report,foot_contacts=contacts).validate()
    report['constraints']=check_constraints(controls,source,motion,request)
    emit('asset_motion_export','导出原始 SOMA 动作 BVH',device='cpu')
    report['bvh_roundtrip']=export_bvh(motion,skeleton,folder)
    atomic_json(folder/'kimodo.json',report)
    del model,output,constraints;gc.collect();torch.cuda.empty_cache()
    emit('asset_motion_generate','Kimodo 动作已生成，模型已释放，准备适配角色骨架',device='cpu')
    return motion
