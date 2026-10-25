"""Isolated UltraTex PBR engine, without launching a ComfyUI server or graph.

Uses the pinned MIT UltraTex community adapter and GPL Comfy inference library;
their original sources and licenses are distributed by the runtime installer.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path


def generate(request, image, output, emit):
    import numpy as np
    import torch
    import trimesh
    from .run_stage import source_dir
    from ..pipeline_models import MODEL_ROOT, SPECS
    source_dir('comfy_core')
    source_dir('ultratex_bridge')
    import comfy.cli_args
    selected_engine = request.get('engine', 'ultratex')
    int8 = selected_engine == 'ultratex_int8'
    # The pinned workbench uses cu128; use the library's standard offload
    # path rather than CUDA-13-only dynamic allocation kernels.
    comfy.cli_args.args.disable_dynamic_vram = True
    comfy.cli_args.args.lowvram = bool(request.get('offload', True))
    comfy.cli_args.args.highvram = not bool(request.get('offload', True))
    if int8:comfy.cli_args.args.enable_triton_backend = True
    import comfy.sd
    import comfy.utils
    import comfy.model_management as mm
    from ultratex_core import prep as preparation, sampling, lora, bake, optim
    import logging
    logging.getLogger('UltraTex').setLevel(logging.INFO)
    sparse_error=optim.check_sparse_attention()
    actual_attention='ultratex_sparse' if sparse_error is None else 'comfy_default'

    root = Path(request.get('model_root') or MODEL_ROOT)
    adapter = root / 'texture_ultratex'
    backbone_component = 'ultratex_backbone_int8' if int8 else 'ultratex_backbone'
    backbone = root / backbone_component / SPECS[backbone_component]['files'][0]
    encoding = root / 'ultratex_text_encoder' / 'split_files'
    resolution = int(request.get('resolution', 1024))
    texture_size = int(request.get('texture_size', 2048))
    seed = int(request.get('seed', 0))
    steps = int(request.get('texture_steps', 25))
    peaks = {'allocated': 0.0, 'reserved': 0.0}

    def capture_peak():
        peaks['allocated'] = max(peaks['allocated'], torch.cuda.max_memory_allocated()/2**20)
        peaks['reserved'] = max(peaks['reserved'], torch.cuda.max_memory_reserved()/2**20)
    emit('encoding_mesh', message='为 UltraTex 准备六视角法线与 UV')
    from ..asset_pipeline import load_texture_input
    mesh, uv_policy = load_texture_input(Path(request['mesh_path']))
    from .mesh_uv import ensure_uv
    mesh,uv_generation=ensure_uv(mesh)
    # An existing UV map is retained. A new xatlas layout may split vertices
    # along seams but does not simplify, remesh, or change input positions.
    uv_mode = 'keep_existing'
    prep = preparation.run_prep(mesh, np.asarray(image.convert('RGBA')), min(resolution, 1024),
                                texture_size, uv_mode, 'fit_front_silhouette')
    emit('loading', message='加载 FLUX.2 Klein 9B 与 UltraTex 前景解码器')
    text_checkpoint = encoding / 'text_encoders/qwen_3_8b_fp8mixed.safetensors'
    text_options = {}
    device = mm.get_torch_device()
    # LOW_VRAM otherwise puts Qwen entirely on CPU, even on a mostly empty
    # 24 GB card. Encode on CUDA only with generous weight/activation headroom,
    # then unload it before loading the diffusion backbone. Small cards retain
    # the conservative CPU path.
    if mm.get_free_memory(device) >= text_checkpoint.stat().st_size * 1.5 + 2 * 1024**3:
        text_options = {'load_device': device, 'offload_device': mm.text_encoder_offload_device()}
    text_started = time.perf_counter()
    clip = comfy.sd.load_clip([str(text_checkpoint)], clip_type=comfy.sd.CLIPType.FLUX2,
                              model_options=text_options)
    text_device = str(clip.patcher.load_device)
    emit('encoding_reference', message=f'编码参考条件（{text_device}）')
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(''))
    text_seconds = round(time.perf_counter() - text_started, 3)
    capture_peak()
    del clip
    mm.unload_all_models()
    gc.collect()
    torch.cuda.empty_cache()
    state = comfy.utils.load_torch_file(str(encoding / 'vae/flux2-vae.safetensors'), safe_load=True)
    # FLUX.2's distributed checkpoint nests these two convolutions, while the
    # foreground decoder uses Comfy VAE's canonical state names. Apply the
    # same key-only normalization as the pinned VAE loader before replacement.
    if 'decoder.post_quant_conv.weight' in state:
        state = comfy.utils.state_dict_prefix_replace(state, {
            'decoder.post_quant_conv.': 'post_quant_conv.',
            'encoder.quant_conv.': 'quant_conv.',
        })
    decoder = comfy.utils.load_torch_file(str(adapter / 'flux2/decoder.pt'), safe_load=True)
    for key, value in decoder.items():
        target = key if key.startswith('post_quant_conv.') else f'decoder.{key}'
        if target not in state or state[target].shape != value.shape:
            raise RuntimeError(f'UltraTex 前景解码器与 FLUX.2 VAE 不匹配: {target}')
        state[target] = value.to(state[target].dtype)
    vae = comfy.sd.VAE(sd=state)
    vae.throw_exception_if_invalid()
    del state, decoder
    model = comfy.sd.load_diffusion_model(str(backbone))
    if model is None:
        raise RuntimeError('UltraTex FLUX.2 Klein 9B 底座无法加载')
    if sampling.model_family(model) != 'flux2':
        raise RuntimeError('UltraTex 9B LoRA 需要 FLUX.2 Klein 9B base 底座')
    quantized_parameters = 0
    if int8:
        from comfy.quant_ops import QuantizedTensor
        quantized_parameters = sum(isinstance(value, QuantizedTensor) for value in model.model.diffusion_model.parameters())
        if not quantized_parameters:raise RuntimeError('INT8 ConvRot 底座没有按原生量化格式加载，已停止贴图')

    def predict(branch, sample_resolution):
        emit('sampling', message='生成六视角颜色贴图' if branch == 'flux2' else '生成六视角金属度与粗糙度', steps=steps)
        weights = comfy.utils.load_torch_file(str(adapter / branch / 'lora/dit_lora.safetensors'), safe_load=True)
        dm = model.model.diffusion_model
        shapes = {f'diffusion_model.single_blocks.{i}.linear1.weight': (block.linear1.out_features,)
                  for i, block in enumerate(dm.single_blocks) if hasattr(block.linear1, 'out_features')}
        patches, fmt = lora.convert(weights, shapes)
        if fmt != 'peft':
            raise RuntimeError('UltraTex LoRA 不是预期的 FLUX.2 PEFT 发布格式')
        patched, _ = comfy.sd.load_lora_for_models(model, None, patches, 1.0, 0.0)
        expected = len(patches) // 2
        applied = len(patched.patches) - len(model.patches)
        if applied != expected:
            raise RuntimeError(f'UltraTex LoRA 未完整应用: {applied}/{expected}')
        capture_peak()
        previous_hook=comfy.utils.PROGRESS_BAR_HOOK
        message='颜色贴图' if branch=='flux2' else '金属度与粗糙度'
        comfy.utils.set_progress_bar_global_hook(lambda current,total,preview,**kwargs:
            emit('sampling',message=f'{message} · {current}/{total}',step=int(current),steps=int(total)))
        try:
            views, masks = sampling.sample(patched, vae, positive, None, prep, sample_resolution, steps, seed,
                                           guidance=1.0, attention=actual_attention, token_chunk=8192)
        finally:comfy.utils.set_progress_bar_global_hook(previous_hook)
        capture_peak()
        values = views.detach().cpu().numpy().astype(np.float32)
        visible = masks.detach().cpu().numpy() > 0.5
        from PIL import Image
        for i, value in enumerate(values):
            Image.fromarray(np.clip(value * 255, 0, 255).astype(np.uint8)).save(output.parent / f'{branch}-view-{i}.png')
        del views, masks, patched, patches, weights
        mm.unload_all_models()
        gc.collect()
        torch.cuda.empty_cache()
        return values, visible

    albedo_views, albedo_masks = predict('flux2', resolution)
    mr_views, mr_masks = predict('flux2_mr', min(resolution, 1024))
    emit('baking', message='融合多视角细节并烘焙完整 PBR 材质')
    baker = bake.Baker(prep, texture_size)
    try:
        settings = bake.BakeSettings()
        albedo, confidence = baker.bake([(albedo_views, albedo_masks, prep.rig)], settings)
        orm, _ = baker.bake([(mr_views, mr_masks, prep.rig)], settings, data=True)
        orm[..., 0] = 1.0
        albedo_image, orm_image = bake.to_pil(albedo), bake.to_pil(orm)
    finally:
        baker.release()
    albedo_image.save(output.parent / 'albedo.png')
    orm_image.save(output.parent / 'metallic-roughness.png')
    result = bake.textured_mesh(prep, albedo_image, orm_image)
    emit('exporting', message='导出嵌入式 PBR 材质 GLB')
    result.export(output)
    return {'model': SPECS['texture_ultratex'], 'engine': selected_engine, 'backbone': SPECS[backbone_component],
            'backbone_precision': 'int8_convrot' if int8 else 'fp8', 'quantized_weight_tensors': quantized_parameters,
            'text_encoder_device': text_device, 'text_encoder_seconds': text_seconds,
            'geometry_generated': False, 'texture_generated': True, 'inference_steps': steps,
            'uv_policy': uv_policy,
            'uv_generation':uv_generation,
            'attention':actual_attention,'attention_fallback_reason':sparse_error,
            'resolution': resolution, 'mr_resolution': min(resolution, 1024), 'texture_size': texture_size,
            'peak_allocated_mb': round(peaks['allocated'],2), 'peak_reserved_mb': round(peaks['reserved'],2),
            'offload_policy': 'standard CPU offload' if request.get('offload', True) else 'high VRAM residency',
            'vertices': len(result.vertices), 'faces': len(result.faces),
            'pbr_channels': ['baseColor', 'metallic', 'roughness'], 'views_per_channel': 6,
            'bake': {'fusion': 'two_band', 'hole_fill': 'mesh', 'exposure_match': True},
            'transform': {'normalization_restored': True, 'mesh_input_frame_preserved': True,
                          'uv_seam_vertex_splits_possible': True, 'remeshed': False},
            'licenses': {'adapter': 'MIT', 'inference_library': 'GPL-3.0',
                         'base_model': 'FLUX Non-Commercial License'}}
