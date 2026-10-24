"""Pinned Hunyuan3D-Paint 2.1 PBR inference on an existing workbench mesh."""
from __future__ import annotations

import gc
from pathlib import Path
import sys


def pipeline_classes(source):
    """Load inference modules without importing upstream Lightning trainers."""
    import importlib
    import importlib.util
    import types
    package = 'hunyuanpaintpbr'
    if package not in sys.modules:
        namespace = types.ModuleType(package)
        namespace.__path__ = [str(source / 'hy3dpaint' / package)]
        namespace.__package__ = package
        namespace.__spec__ = importlib.util.spec_from_loader(package, loader=None, is_package=True)
        sys.modules[package] = namespace
    pipeline = importlib.import_module(package + '.pipeline')
    modules = importlib.import_module(package + '.unet.modules')
    return pipeline.HunyuanPaintPipeline, modules.UNet2p5DConditionModel, modules.Dino_v2


def load_pretrained_pipeline(pipeline_class, model, unet, torch_dtype):
    """Load exact components without the checkpoint's obsolete class registry.

    Its Diffusers 0.24 metadata names a plain nn.Module as ``modules``. Modern
    Diffusers cannot validate that legacy entry. Construct the official class
    with its unmodified, locally loaded inference components instead.
    """
    from diffusers import AutoencoderKL,DDIMScheduler
    from transformers import CLIPTextModel,CLIPTokenizer,CLIPImageProcessor
    def local(cls,folder,**kwargs):
        return cls.from_pretrained(str(model/folder),local_files_only=True,**kwargs)
    return pipeline_class(
        vae=local(AutoencoderKL,'vae',torch_dtype=torch_dtype,use_safetensors=False),
        text_encoder=local(CLIPTextModel,'text_encoder',torch_dtype=torch_dtype),
        tokenizer=local(CLIPTokenizer,'tokenizer'),unet=unet,
        scheduler=local(DDIMScheduler,'scheduler'),
        feature_extractor=local(CLIPImageProcessor,'feature_extractor'),safety_checker=None,
    )


def generate(request, image, output, emit):
    import numpy as np
    import torch
    import trimesh
    from PIL import Image
    from types import SimpleNamespace
    from .run_stage import source_dir
    from ..pipeline_models import MODEL_ROOT, SPECS
    from .hunyuan_renderer import load_renderer
    source = source_dir('hunyuan21')
    sys.path.insert(0, str(source / 'hy3dpaint'))
    MeshRender = load_renderer(source)
    from utils.pipeline_utils import ViewProcessor
    HunyuanPaintPipeline, UNet2p5DConditionModel, Dino_v2 = pipeline_classes(source)
    from diffusers import UniPCMultistepScheduler

    root = Path(request.get('model_root') or MODEL_ROOT)
    model = root / 'texture_hunyuan21' / 'hunyuan3d-paintpbr-v2-1'
    resolution = int(request.get('resolution', 512))
    texture_size = int(request.get('texture_size', 2048))
    steps = int(request.get('texture_steps', 15))
    render_size = min(max(1024, resolution * 2), 2048)
    emit('encoding_mesh', message='为 Hunyuan Paint 准备网格与多视角条件')
    from ..asset_pipeline import load_texture_input
    mesh, uv_policy = load_texture_input(Path(request['mesh_path']))
    from .mesh_uv import ensure_uv
    mesh,uv_generation=ensure_uv(mesh)
    render = MeshRender(default_resolution=render_size, texture_size=texture_size, bake_mode='back_sample', raster_mode='cr')
    render.load_mesh(mesh=mesh)
    cfg = SimpleNamespace(bake_exp=4)
    processor = ViewProcessor(cfg, render)
    # Official canonical six views. View selection only adds more views when
    # max_num_view exceeds six, so no extra candidate renders are necessary.
    elevations, azimuths, weights = [0,0,0,0,90,-90], [0,90,180,270,0,180], [1,.1,.5,.1,.05,.05]
    normals = processor.render_normal_multiview(elevations, azimuths, use_abs_coor=True)
    positions = processor.render_position_multiview(elevations, azimuths)
    for i, normal in enumerate(normals):normal.save(output.parent / f'normal-view-{i}.png')
    emit('loading', message='加载 Hunyuan3D-Paint 2.1 与 DINOv2')
    unet = UNet2p5DConditionModel.from_pretrained(str(model / 'unet'), torch_dtype=torch.float16)
    pipeline = load_pretrained_pipeline(HunyuanPaintPipeline,model,unet,torch_dtype=torch.float16)
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, timestep_spacing='trailing')
    pipeline.eval()
    pipeline.enable_vae_tiling()
    pipeline.enable_vae_slicing()
    reference = Image.new('RGB', image.size, 'white')
    reference.paste(image, mask=image.getchannel('A'))
    reference = reference.resize((resolution, resolution), Image.Resampling.LANCZOS)
    dino = Dino_v2(str(root / 'paint_dino')).to(dtype=torch.float16, device='cuda')
    with torch.inference_mode():
        features = dino(reference)
    del dino
    gc.collect()
    torch.cuda.empty_cache()
    if request.get('offload', True):
        pipeline.enable_model_cpu_offload(device='cuda')
        original_encode_images=pipeline.encode_images
        def encode_on_execution_device(images):
            # Upstream prepares the reference on vae.device (CPU while
            # offloaded), but the VAE encode hook moves only its weights.
            # Keep the encoder inputs on the hook's CUDA execution device.
            return original_encode_images(images.to('cuda'))
        pipeline.encode_images=encode_on_execution_device
        original_denoise=pipeline.denoise
        def denoise_on_execution_device(*args,**kwargs):
            # Material prompt tensors are read from the offloaded UNet before
            # its forward hook runs. Upstream bypasses encode_prompt, so move
            # those exact learned embeddings to the execution device here.
            for key in ('prompt_embeds','negative_prompt_embeds'):
                if kwargs.get(key) is not None:kwargs[key]=kwargs[key].to('cuda')
            return original_denoise(*args,**kwargs)
        pipeline.denoise=denoise_on_execution_device
    else:
        pipeline.to('cuda')
    emit('sampling', message='生成多视角颜色与金属度 / 粗糙度', steps=steps)
    def sample_progress(pipe,index,timestep,values):
        emit('sampling',message=f'多视角 PBR · {index+1}/{steps}',step=index+1,steps=steps)
        return values
    # Upstream VAE posterior sampling does not use the supplied noise
    # generator. Seed its global RNG too, after component initialization.
    torch.manual_seed(int(request.get('seed',0)))
    with torch.inference_mode():
        result = pipeline([reference], num_inference_steps=steps, prompt='high quality', sync_condition=None,
                          guidance_scale=3.0, generator=torch.Generator(device='cuda').manual_seed(int(request.get('seed',0))),
                          width=resolution, height=resolution, num_in_batch=6,
                          images_normal=[[v.resize((resolution,resolution)) for v in normals]],
                          images_position=[[v.resize((resolution,resolution)) for v in positions]],
                          dino_hidden_states=features,callback_on_step_end=sample_progress,
                          callback_on_step_end_tensor_inputs=[]).images
    if len(result) != 12:
        raise RuntimeError('Hunyuan Paint 没有生成完整的六视角颜色与材质图')
    del pipeline, unet, features
    gc.collect()
    torch.cuda.empty_cache()
    emit('decoding', message='使用 Real-ESRGAN 增强生成的视角贴图')
    import spandrel
    upscaler = spandrel.ModelLoader().load_from_file(str(root / 'paint_upscaler' / 'RealESRGAN_x4plus.pth')).eval().half().cuda()

    def enhance(value):
        data = torch.from_numpy(np.asarray(value.convert('RGB')).copy()).float().div_(255).permute(2,0,1)[None].cuda().half()
        scale, tile, pad = 4, 256, 16
        _, _, height, width = data.shape
        out = torch.empty((1,3,height*scale,width*scale), device='cpu', dtype=torch.float32)
        with torch.inference_mode():
            for y in range(0,height,tile):
                for x in range(0,width,tile):
                    y0,x0,y1,x1=max(0,y-pad),max(0,x-pad),min(height,y+tile+pad),min(width,x+tile+pad)
                    prediction=upscaler(data[:,:,y0:y1,x0:x1])
                    h,w=min(tile,height-y),min(tile,width-x)
                    out[:,:,y*scale:(y+h)*scale,x*scale:(x+w)*scale] = prediction[:,:,(y-y0)*scale:(y-y0+h)*scale,(x-x0)*scale:(x-x0+w)*scale].float().cpu()
        array=out[0].permute(1,2,0).clamp(0,1).mul(255).byte().numpy()
        return Image.fromarray(array).resize((render_size,render_size),Image.Resampling.LANCZOS)

    albedo_views, mr_views = [], []
    for i, value in enumerate(result):
        value.save(output.parent / f'generated-view-{i}.png')
        (albedo_views if i < 6 else mr_views).append(enhance(value))
    del upscaler
    gc.collect()
    torch.cuda.empty_cache()
    emit('baking', message='按原网格烘焙 Hunyuan PBR 材质')
    albedo, mask = processor.bake_from_multiview(albedo_views,elevations,azimuths,weights)
    mr, mr_mask = processor.bake_from_multiview(mr_views,elevations,azimuths,weights)
    albedo_np=render.uv_inpaint(albedo,mask)
    mr_np=render.uv_inpaint(mr,mr_mask)
    # Hunyuan MR uses R=metallic, G=roughness. glTF ORM uses G=roughness, B=metallic.
    orm=np.zeros_like(mr_np);orm[...,0]=255;orm[...,1]=mr_np[...,1];orm[...,2]=mr_np[...,0]
    vertices,faces,uv,uv_faces=render.get_mesh(normalize=False)
    if not np.array_equal(faces,uv_faces):raise RuntimeError('Hunyuan UV 顶点索引与网格不一致')
    albedo_image,orm_image=Image.fromarray(albedo_np),Image.fromarray(orm)
    albedo_image.save(output.parent/'albedo.png');orm_image.save(output.parent/'metallic-roughness.png')
    material=trimesh.visual.material.PBRMaterial(baseColorTexture=albedo_image,metallicRoughnessTexture=orm_image,
                                               metallicFactor=1.0,roughnessFactor=1.0)
    final=trimesh.Trimesh(vertices,faces,vertex_normals=np.asarray(mesh.vertex_normals),process=False,
                         visual=trimesh.visual.TextureVisuals(uv=uv,material=material))
    emit('exporting',message='导出 Hunyuan PBR 材质 GLB')
    final.export(output)
    return {'model':SPECS['texture_hunyuan21'],'engine':'hunyuan21','geometry_generated':False,'texture_generated':True,
            'uv_policy':uv_policy,
            'uv_generation':uv_generation,
            'shading_normals':'input normals preserved; no UV-chart normal recomputation',
            'inference_steps':steps,'texture_size':texture_size,'vertices':len(final.vertices),'faces':len(final.faces),
            'resolution':resolution,'conditioning_render_size':render_size,
            'vae_memory_policy':'batch slicing and spatial tiling',
            'seed_policy':'global VAE posterior RNG and explicit CUDA noise generator',
            'pbr_channels':['baseColor','metallic','roughness'],'views_per_channel':6,
            'renderer':'official camera/conditioning/baking with nvdiffrast Windows raster adapter',
            'inpainting':'OpenCV NS in UV space; optional upstream C++ vertex inpainting is not used',
            'upscaler':SPECS['paint_upscaler'],'transform':{'normalization_restored':True,
            'mesh_input_frame_preserved':True,'uv_seam_vertex_splits_possible':True,'remeshed':False}}
