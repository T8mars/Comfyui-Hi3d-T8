"""Pinned UniRig skeleton and skin stages; original geometry stays untouched."""
from __future__ import annotations

import copy
import gc
import os
import random
import sys

import numpy as np

from . import models, skin
from .parts_models import weight
from .rigging import SampledWeights
from .rig_skintokens import Y_UP_TO_Z_UP, local_config
from .scene import combined


class LocalOPTConfig:
    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, **overrides):
        # Architecture constants from facebook/opt-350m/config.json. UniRig
        # supplies its own complete weights, vocabulary and projection size;
        # no OPT pretrained weights or online config lookup are needed.
        if pretrained_model_name_or_path != 'facebook/opt-350m':
            raise RuntimeError('固定 UniRig 权重引用了未支持的语言模型架构')
        from transformers import OPTConfig
        config=dict(activation_dropout=0.,activation_function='relu',attention_dropout=0.,
            bos_token_id=2,do_layer_norm_before=False,dropout=.1,eos_token_id=2,ffn_dim=4096,
            hidden_size=1024,init_std=.02,layerdrop=0.,max_position_embeddings=2048,
            num_attention_heads=16,num_hidden_layers=24,pad_token_id=1,use_cache=True,
            vocab_size=50272,word_embed_proj_dim=512)
        config.update(overrides)
        # Same scaled dot product attention, without an extra FA2 dependency
        # inside Transformers. Other upstream mesh attention stays unchanged.
        config['_attn_implementation']='sdpa'
        return OPTConfig(**config)


def inference_mesh(document, folder):
    import trimesh
    vertices,faces,_=combined(document,apply_morphs=True)
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=True,validate=True)
    if len(mesh.faces)>50000:
        from .gltf import Document
        from .mesh import simplify
        from pathlib import Path
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        proxy=Document({'asset':{'version':'2.0'},'scene':0,'scenes':[{'nodes':[0]}],
            'nodes':[{'mesh':0}],'meshes':[{'primitives':[]} ]})
        positions=proxy.add_accessor(np.asarray(mesh.vertices,dtype=np.float32))
        indices=proxy.add_accessor(np.asarray(mesh.faces,dtype=np.uint32).reshape(-1))
        proxy.tree['meshes'][0]['primitives']=[{'attributes':{'POSITION':positions},'indices':indices}]
        source=folder/'unirig-proxy-source.glb';target=folder/'unirig-proxy.glb';proxy.save(source)
        simplify(source,target,target_faces=50000,priority='count',library_path=models.runtime_root()/
                 ('meshoptimizer.dll' if os.name=='nt' else 'libmeshoptimizer.so'))
        vertices,faces,_=combined(Document.load(target),apply_morphs=True)
        mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=True,validate=True)
    return mesh


def predict_skin(data,root,emit,*,mixed_precision=False):
    """One independently repeatable neural skin stage, with finite checks."""
    import torch,yaml
    from box import Box
    from src.model.unirig_skin import UniRigSkin
    model_config=Box(local_config(yaml.safe_load((root/'configs/model/unirig_skin.yaml').read_text(encoding='utf-8')),root))
    model_config.pop('__target__')
    model=UniRigSkin(**model_config)
    saved=torch.load(weight('unirig','skin/articulation-xl/model.ckpt'),map_location='cpu',weights_only=False,mmap=True)
    model.load_state_dict({key.removeprefix('model.'):value for key,value in saved['state_dict'].items()},strict=True)
    del saved
    model=model.eval().to('cuda')
    from .attention import install_unirig_attention
    install_unirig_attention(model.mesh_encoder)
    count=len(data['vertices']);joints=len(data['joints'])
    batch={name:torch.as_tensor(value,device='cuda')[None] for name,value in data.items()}
    batch.update(num_bones=torch.tensor([joints],device='cuda'),offset=torch.tensor([count],device='cuda'))
    progress=[0]
    def chunk_progress(module,inputs,output):
        progress[0]+=int(output.shape[0])//joints
        if progress[0]==min(count,model.num_train_vertex) or progress[0]%4096==0 or progress[0]>=count:
            emit('asset_rig_skin','UniRig 已预测采样点蒙皮',device='cuda',progress_current=min(count,progress[0]),
                 progress_total=count,progress_unit='points')
    hooks=[model.skinweight_pred.register_forward_hook(chunk_progress)]
    def finite_hook(name):
        def check(module,inputs,output):
            values=output if isinstance(output,(tuple,list)) else [output.get('feat') if isinstance(output,dict) else output]
            if any(isinstance(value,torch.Tensor) and not torch.isfinite(value).all().item() for value in values):
                raise RuntimeError(f'UniRig {name} 产生无效数值')
        return check
    for name in ('global_encoder','mesh_encoder','bone_encoder','feat_map','out_proj','skinweight_pred'):
        hooks.append(getattr(model,name).register_forward_hook(finite_hook(name)))
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=mixed_precision):
            weights=model.predict_step(batch)[0].float().cpu().numpy()
    finally:
        for hook in hooks:hook.remove()
        del model,batch;gc.collect();torch.cuda.empty_cache()
    return weights


def generate(document, request, emit, *, folder=None):
    from pathlib import Path
    if folder is None:
        raise ValueError('UniRig 需要独立任务目录')
    document=skin.freeze_current_pose(document)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('UniRig 需要支持 CUDA 的 NVIDIA 显卡')
    seed=request.get('seed',0);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    root=models.runtime_root()/'sources'/'unirig'
    if 'src' in sys.modules:
        raise RuntimeError('UniRig 必须在独立资产加工进程中运行')
    sys.path.insert(0,str(root))
    from box import Box
    import yaml
    from src.model import unirig_ar
    from src.data.asset import Asset
    from src.data.transform import TransformConfig,transform_asset
    from src.tokenizer.parse import get_tokenizer
    from src.tokenizer.spec import TokenizerConfig
    from transformers import StoppingCriteria,StoppingCriteriaList
    unirig_ar.AutoConfig=LocalOPTConfig

    def config(name):
        value=yaml.safe_load((root/name).read_text(encoding='utf-8'))
        return Box(local_config(value,root))

    emit('asset_rig_prepare','准备 UniRig 官方 5 万面推理副本，保留原成品',device='cpu')
    mesh=inference_mesh(document,folder)
    vertices=np.asarray(mesh.vertices,dtype=np.float32) @ Y_UP_TO_Z_UP[:3,:3].T
    normals=np.asarray(mesh.vertex_normals,dtype=np.float32) @ Y_UP_TO_Z_UP[:3,:3].T
    face_normals=np.asarray(mesh.face_normals,dtype=np.float32) @ Y_UP_TO_Z_UP[:3,:3].T

    def asset_for(rig=None,tails=None):
        value=Asset(cls='articulationxl',path='local.glb',data_name='raw_data.npz',vertices=vertices.copy(),
            vertex_normals=normals.copy(),face_normals=face_normals.copy(),faces=np.asarray(mesh.faces,dtype=np.int64),meta={})
        if rig is not None:
            order=rig.validate()
            reverse={old:new for new,old in enumerate(order)}
            value.names=[rig.names[index] for index in order]
            value.parents=[reverse[int(rig.parents[index])] if rig.parents[index]!=-1 else None for index in order]
            value.joints=rig.worlds[order,:3,3] @ Y_UP_TO_Z_UP[:3,:3].T
            value.tails=(tails[order] if tails is not None else rig.worlds[order,:3,3]+rig.worlds[order,:3,1]*.05) @ Y_UP_TO_Z_UP[:3,:3].T
        return value

    def prepare(asset,configuration):
        first,second=transform_asset(asset,TransformConfig.parse(configuration.predict_transform_config))
        matrix=Y_UP_TO_Z_UP.copy()
        for augment in [*first,*second]:
            if not hasattr(augment,'trans_vertex'):
                raise RuntimeError('固定 UniRig 预测配置包含未支持的非仿射坐标变换')
            matrix=np.asarray(augment.trans_vertex) @ matrix
        return matrix

    original=skin.extract(document) if request.get('use_skeleton') else None
    if original is None:
        ar_asset=asset_for();to_model=prepare(ar_asset,config('configs/transform/inference_ar_transform.yaml'))
        tokenizer=get_tokenizer(TokenizerConfig.parse(config('configs/tokenizer/tokenizer_parts_articulationxl_256.yaml')))
        model_config=config('configs/model/unirig_ar_350m_1024_81920_float32.yaml');model_config.pop('__target__')
        emit('asset_loading','读取 UniRig 骨架预测参数',device='cpu')
        model=unirig_ar.UniRigAR(**model_config,tokenizer=tokenizer)
        saved=torch.load(weight('unirig','skeleton/articulation-xl_quantization_256/model.ckpt'),map_location='cpu',weights_only=False,mmap=True)
        model.load_state_dict({key.removeprefix('model.'):value for key,value in saved['state_dict'].items()},strict=True)
        del saved
        model=model.eval().to('cuda')
        class Progress(StoppingCriteria):
            def __call__(self,input_ids,scores,**kwargs):
                count=int(input_ids.shape[-1])
                if count==1 or count%20==0:
                    emit('asset_rig_skeleton','UniRig 正在生成骨架',device='cuda',progress_current=count,progress_unit='tokens')
                return False
        options=dict(config('configs/system/ar_inference_articulationxl.yaml').generate_kwargs)
        for key in ('no_cls','assign_cls','use_dir_cls'):options.pop(key,None)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            predicted=model.generate(torch.as_tensor(ar_asset.sampled_vertices,device='cuda',dtype=torch.float32),
                torch.as_tensor(ar_asset.sampled_normals,device='cuda',dtype=torch.float32),cls='articulationxl',
                stopping_criteria=StoppingCriteriaList([Progress()]),**options)
        inverse=np.linalg.inv(to_model)
        points=np.asarray(predicted.joints) @ inverse[:3,:3].T+inverse[:3,3]
        tails=(np.asarray(predicted.tails) @ inverse[:3,:3].T+inverse[:3,3]) if predicted.tails is not None else None
        names=list(predicted.names) if predicted.names is not None else [f'bone_{index}' for index in range(len(points))]
        original=skin.oriented_rig(points,[-1 if parent is None else parent for parent in predicted.parents],names)
        del model,ar_asset,predicted;gc.collect();torch.cuda.empty_cache()
    else:
        tails=None
    emit('asset_rig_prepare','计算 UniRig 体素蒙皮先验并采样表面',device='cpu')
    asset=asset_for(original,tails)
    transform=config('configs/transform/inference_skin_transform.yaml')
    transform.predict_transform_config.vertex_group_config.kwargs.voxel_skin.backend='open3d'
    to_model=prepare(asset,transform)
    emit('asset_loading','读取 UniRig 蒙皮参数；骨架模型已释放',device='cpu')
    count=len(asset.sampled_vertices);joints=asset.J
    inputs={
        'vertices':asset.sampled_vertices,'normals':asset.sampled_normals,'joints':asset.joints,
        'tails':asset.tails,'voxel_skin':asset.sampled_vertex_groups['voxel_skin'],
        'parents':np.array([-1 if parent is None else parent for parent in asset.parents],dtype=np.int64)}
    temporary=Path(folder)/'unirig-inputs.tmp.npz'
    np.savez_compressed(temporary,**inputs)
    temporary.replace(Path(folder)/'unirig-inputs.npz')
    weights=predict_skin(inputs,root,emit)
    columns=[asset.names.index(name) for name in original.names];weights=weights[:,columns]
    samples=np.asarray(asset.sampled_vertices).copy()
    # Keep the complete official prediction for independent diagnostics and
    # later correction; never silently replace invalid neural weights.
    temporary=Path(folder)/'unirig-samples.tmp.npz'
    np.savez_compressed(temporary,points=samples,weights=weights,to_model=to_model,
        voxel_skin=asset.sampled_vertex_groups['voxel_skin'][:,columns],
        worlds=original.worlds,parents=original.parents,names=np.asarray(original.names))
    temporary.replace(Path(folder)/'unirig-samples.npz')
    if not np.isfinite(weights).all() or np.any(weights<0):
        raise RuntimeError('UniRig 蒙皮预测含无效权重，已保留诊断样本，未生成伪完成结果')
    emit('asset_rig_transfer','将蒙皮转移回原网格并保留 UV、材质和比例',device='cpu')
    output,details=skin.bind(document,original,SampledWeights(samples,weights,to_model),
                             use_existing=bool(request.get('use_skeleton')))
    details.update(engine='unirig',source_revision='6793c6640ff01c8fb389f3993434124bb43d2933',
        inference_faces=len(mesh.faces),sampled_points=count,use_skeleton=bool(request.get('use_skeleton')),
        skin_precision='float32',point_attention='packed-sdpa-float32')
    return output,details
