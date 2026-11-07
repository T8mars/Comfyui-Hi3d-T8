"""Pinned SkinTokens inference with exact preprocessing-coordinate inversion."""
from __future__ import annotations

import copy
import importlib
import os
import random
import sys
import time

import numpy as np

from . import models, skin
from .parts_models import weight
from .rigging import SampledWeights
from .scene import combined


Y_UP_TO_Z_UP = np.array([[1., 0., 0., 0.], [0., 0., -1., 0.],
                         [0., 1., 0., 0.], [0., 0., 0., 1.]])


def local_config(value, root):
    """Resolve only the upstream skeleton YAMLs; never use the caller's cwd."""
    result = copy.deepcopy(value)
    def visit(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if key == 'skeleton_path':
                    for name, relative in child.items():
                        path = (root / relative).resolve()
                        if not path.is_relative_to(root.resolve()) or not path.is_file():
                            raise RuntimeError('SkinTokens 骨架配置不存在或超出固定源码目录')
                        child[name] = str(path)
                else:
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
    visit(result)
    return result


def generate(document, request, emit, *, folder=None):
    document = skin.freeze_current_pose(document)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('SkinTokens 需要支持 CUDA 的 NVIDIA 显卡')
    torch.manual_seed(request.get('seed', 0))
    np.random.seed(request.get('seed', 0))
    random.seed(request.get('seed', 0))
    emit('asset_loading', '校验 SkinTokens 骨架与蒙皮权重', device='cpu')
    checkpoint = weight('skintokens', 'experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt')
    vae = weight('skintokens', 'experiments/skin_vae_2_10_32768/last.ckpt')
    qwen = weight('qwen_rig_config', 'config.json').parent
    root = models.runtime_root() / 'sources' / 'skintokens'
    if 'src' in sys.modules:
        raise RuntimeError('SkinTokens 必须在独立的资产加工进程中运行')
    sys.path.insert(0, str(root))
    module = importlib.import_module('src.model.tokenrig')
    from src.data.transform import Transform
    from src.rig_package.info.asset import Asset
    from src.tokenizer.spec import TokenizeInput
    from transformers import StoppingCriteria, StoppingCriteriaList
    module.LLM_LOCAL_DIR = qwen
    emit('asset_loading', '读取已校验的 SkinTokens 参数', device='cpu')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = saved['hyper_parameters']
    model_config = copy.deepcopy(config['model_config'])
    model_config['pretrained_vae'] = str(vae)
    model_config['llm']['pretrained_model_name_or_path'] = str(qwen)
    transform_config = local_config(config['transform_config'], root)
    tokenizer_config = local_config(config['tokenizer_config'], root)
    model = module.TokenRig(model_config=model_config, transform_config=transform_config,
                            tokenizer_config=tokenizer_config)
    state = {key.replace('_orig_mod.', '').removeprefix('model.'): value
             for key, value in saved['state_dict'].items()}
    model.load_state_dict(state, strict=True)
    model.on_load_checkpoint(saved)
    del saved, state
    emit('asset_loading', 'SkinTokens 权重移入显卡', device='cuda')
    model = model.eval().to('cuda')
    emit('asset_rig_prepare', '按官方密度采样表面，记录坐标归一化以保留原比例', device='cpu')
    vertices, faces, _ = combined(document, apply_morphs=True)
    original_rig = skin.extract(document) if request.get('use_skeleton') else None

    class TrackedAsset(Asset):
        def transform(self, trans):
            self.to_model = np.asarray(trans) @ self.to_model
            super().transform(trans)

    asset = TrackedAsset(vertices=vertices.astype(np.float32), faces=faces,
                         cls='articulation', path='local.glb', matrix_world=np.eye(4))
    asset.to_model = np.eye(4)
    order = None
    if original_rig is not None:
        roots = np.flatnonzero(original_rig.parents == -1)
        if len(roots) != 1:
            raise ValueError('现有模型包含多个独立骨架，请按单个角色分别绑定')
        order = []
        def visit(index):
            order.append(index)
            for child in np.flatnonzero(original_rig.parents == index):
                visit(int(child))
        visit(int(roots[0]))
        reverse = {old: new for new, old in enumerate(order)}
        asset.parents = np.array([reverse.get(int(original_rig.parents[index]), -1) for index in order])
        asset.joint_names = [original_rig.names[index] for index in order]
        asset.matrix_local = original_rig.worlds[order].copy()
    asset.transform(Y_UP_TO_Z_UP)
    transform = Transform.parse(**transform_config['predict_transform'])
    if any(augment.__class__.__name__ not in {'AugmentTrim', 'AugmentAffine', 'AugmentLinear', 'AugmentNormalize'}
           for augment in transform.augments or []):
        raise RuntimeError('固定权重含未支持的非仿射预测变换，不能保证原比例')
    transform.apply(asset)
    if asset.sampled_vertices is None or asset.sampled_normals is None:
        raise RuntimeError('SkinTokens 官方表面采样没有返回坐标或法线')
    skeleton_tokens = None
    if original_rig is not None:
        if asset.joint_names != [original_rig.names[index] for index in order]:
            raise RuntimeError('SkinTokens 预处理改变已有骨架，已拒绝错误权重映射')
        skeleton_tokens = model.tokenizer.tokenize(TokenizeInput(
            joints=asset.joints, parents=asset.parents.tolist(), cls=asset.cls, joint_names=asset.joint_names))
    class Progress(StoppingCriteria):
        def __init__(self):
            self.last = 0.
        def __call__(self, input_ids, scores, **kwargs):
            if time.monotonic() - self.last > 3:
                emit('asset_rig_generate', f'生成骨架与蒙皮序列：{input_ids.shape[-1]} 个 token', device='cuda')
                self.last = time.monotonic()
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    # Match the public demo defaults; this is not a reduced-beam preview model.
    kwargs = dict(max_length=2048, top_k=5, top_p=.95, temperature=1., repetition_penalty=2.,
                  num_return_sequences=1, num_beams=10, do_sample=True,
                  stopping_criteria=StoppingCriteriaList([Progress()]))
    emit('asset_rig_generate', '预测骨架层级与蒙皮，使用官方 10 束搜索', device='cuda')
    with torch.inference_mode():
        result = model.predict_step({'vertices': torch.from_numpy(asset.sampled_vertices).float().cuda(),
            'normals': torch.from_numpy(asset.sampled_normals).float().cuda(), 'cls': ['articulation'],
            'generate_kwargs': kwargs}, skeleton_tokens=[skeleton_tokens] if skeleton_tokens is not None else None,
            make_asset=False)['results'][0]
    if result.skin_pred is None or result.detokenize_output is None:
        raise RuntimeError('SkinTokens 没有生成完整骨架与蒙皮')
    decoded = result.detokenize_output
    weights = result.skin_pred.detach().float().cpu().numpy()
    inverse = np.linalg.inv(asset.to_model)
    if original_rig is None:
        points = decoded.joints @ inverse[:3, :3].T + inverse[:3, 3]
        rig = skin.oriented_rig(points, decoded.parents, decoded.joint_names)
    else:
        if len(decoded.parents) != original_rig.parents.size or not np.array_equal(decoded.parents, asset.parents):
            raise RuntimeError('已有骨架在量化后发生关节合并，请先修正重合关节再生成蒙皮')
        rig = original_rig
        weights = weights[:, np.argsort(order)]
    points = asset.sampled_vertices.copy()
    to_model = asset.to_model.copy()
    del result, model
    torch.cuda.empty_cache()
    emit('asset_rig_transfer', '已释放推理模型，保存完整权重并向原网格转移蒙皮', device='cpu')
    if folder is not None:
        # Keep the untruncated prediction for quality correction. Once reduced
        # to four influences a later mask cannot recover a valid missing bone.
        from ..common import digest
        cache = folder / 'rig-samples.tmp.npz'
        np.savez_compressed(cache, points=points, weights=weights, world_to_model=to_model,
            joint_worlds=rig.worlds, joint_parents=rig.parents, joint_names=np.asarray(rig.names),
            input_sha256=np.asarray(digest(folder / 'coarse.glb')),
            revision=np.asarray(models.CATALOG['skintokens']['revision']))
        cache.replace(folder / 'rig-samples.npz')
    output, details = skin.bind(document, rig, SampledWeights(points, weights, to_model),
                                use_existing=original_rig is not None)
    details.update(engine='SkinTokens', revision=models.CATALOG['skintokens']['revision'],
                   sampled_points=len(points), generation_beams=10,
                   world_to_inference=to_model.tolist(), seed=request.get('seed', 0))
    return output, details
