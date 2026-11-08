"""Idol adapter for the pinned Comfy inference library, in a private process.

Upstream algorithms and GPL sources remain in the verified runtime. No Comfy
server is launched, and model loading never accesses the network.
"""
from __future__ import annotations

import gc
import math
from pathlib import Path

FIELDS = ('idol_gaussians', 'idol_cfg', 'idol_decode_seed', 'idol_extract_mesh',
          'idol_crop', 'idol_level', 'idol_smooth', 'idol_min_component',
          'idol_min_opacity', 'idol_video', 'idol_sampling', 'idol_align_mesh', 'idol_mesh_backend', 'idol_solid', 'idol_fill')


def settings(request):
    values = dict(idol_gaussians=262144, idol_cfg=3., idol_decode_seed=0,
                  idol_extract_mesh=True, idol_crop='synthetic', idol_level=.6,
                  idol_smooth=0, idol_min_component=500, idol_min_opacity=.02,
                  idol_video=True, idol_sampling='native_euler', idol_align_mesh=True,
                  idol_mesh_backend='solid',idol_solid=True,idol_fill=20.)
    values.update({key: request[key] for key in FIELDS if key in request})
    for key, low, high in [('idol_gaussians',32768,1048576), ('idol_decode_seed',0,2**53-1),
                          ('idol_smooth',0,60), ('idol_min_component',0,100000)]:
        value = values[key]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'{key} 应为 {low} 至 {high} 的整数')
    if values['idol_gaussians'] % 32:
        raise ValueError('高斯数量必须是 32 的倍数')
    for key, low, high in [('idol_cfg',0.,20.), ('idol_level',.01,2.), ('idol_min_opacity',0.,1.), ('idol_fill',0.,1000.)]:
        value = values[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{key} 超出有效范围')
    if values['idol_crop'] not in {'synthetic','real'}:
        raise ValueError('Idol 裁切模式应为 synthetic 或 real')
    if values['idol_sampling'] not in {'comfy','native_euler'}:
        raise ValueError('Idol 采样方案应为 comfy 或 native_euler')
    if values['idol_mesh_backend'] not in {'solid','legacy'}:raise ValueError('无效 Idol 网格后端')
    if any(type(values[key]) is not bool for key in ('idol_extract_mesh','idol_video','idol_align_mesh','idol_solid')):
        raise ValueError('Idol 网格提取与展示视频选项应为布尔值')
    return values


def mesher(backend):
    if backend=='legacy':
        from comfy_extras.nodes_gaussian_splat import SplatToMesh
        return SplatToMesh
    import importlib.util
    from .run_stage import source_dir
    path=source_dir('idol_mesh')/'nodes_gaussian_splat.py'
    spec=importlib.util.spec_from_file_location('hi3d_idol_mesh_upstream',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.SplatToMesh


def camera_metadata(values):
    """Decode the original SDK's five camera values (Idol Apache-2.0).

    See licenses/idol/Idol-Apache-2.0.txt and the pinned SDK camera.py.
    The mesh direction also accounts for the upstream SPLAT/glTF rotations.
    """
    import numpy as np
    v=np.asarray(values,dtype=np.float64)
    if v.shape!=(5,) or not np.isfinite(v).all():raise ValueError('Idol 预测相机包含无效数值')
    norm=float(np.linalg.norm(v[:3]));direction=v[:3]/norm if norm>1e-6 else np.array([0.,0.,1.])
    dx,dy,dz=map(float,direction);inverse=float(v[3]);half=float(v[4])
    radius=1000. if inverse<1e-3 else 1/inverse
    return {'yaw':math.atan2(dx,dy),'pitch':math.asin(float(np.clip(dz,-1,1))),
        'radius':radius,'fov':math.degrees(2*math.atan(half/radius)),
        'object_direction':direction.tolist(),'gltf_direction':[dx,dz,-dy],
        'latent':v.tolist(),'units':'yaw/pitch radians; fov degrees; radius canonical units'}


def initialize():
    import sys
    from .run_stage import source_dir
    overlay = Path(sys.prefix) / 'engines' / 'idol'
    if not overlay.is_dir():
        raise RuntimeError('缺少 Idol 私有依赖，请运行 tools/setup_idol_runtime.py')
    if str(overlay) not in sys.path:sys.path.insert(0,str(overlay))
    source_dir('comfy_core')
    import comfy.cli_args
    comfy.cli_args.args.disable_dynamic_vram = True
    # Match the distributed H+ checkpoint's BF16 precision. The generic
    # Comfy text/vision-encoder default is FP16, which is not this checkpoint.
    comfy.cli_args.args.bf16_text_enc = True


def release():
    import torch
    import comfy.model_management as mm
    mm.unload_all_models()
    gc.collect()
    torch.cuda.empty_cache()


def foreground(request, output_dir, stage):
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from .run_stage import emit
    from ..pipeline_models import MODEL_ROOT
    initialize()
    image = ImageOps.exif_transpose(Image.open(request['image_path'])).convert('RGBA')
    alpha = np.asarray(image.getchannel('A'))
    policy = 'input_alpha'
    if not np.any(alpha < 250) and request.get('background','auto') != 'keep':
        emit(stage+'_background',message='BiRefNet 提取 Idol 前景',device='cuda')
        from comfy.bg_removal_model import load
        root = Path(request.get('model_root') or MODEL_ROOT)
        model = load(str(root/'idol_background/background_removal/birefnet.safetensors'))
        if model is None:raise ValueError('BiRefNet 权重类型不正确')
        pixels = torch.from_numpy(np.asarray(image.convert('RGB')).copy()).float()[None]/255
        with torch.inference_mode():mask = model.encode_image(pixels)[0].clamp(0,1).cpu().numpy()
        image.putalpha(Image.fromarray(np.rint(mask*255).astype(np.uint8)))
        del model,pixels
        release()
        policy = 'birefnet'
    elif not np.any(alpha < 250):policy = 'keep_opaque'
    if np.count_nonzero(np.asarray(image.getchannel('A')) > 16) < 32:
        raise ValueError('Idol 参考图片没有有效前景')
    image.save(output_dir/'foreground.png')
    return image,policy


def generate(request, image, output, emit):
    import numpy as np
    import torch
    from PIL import Image, ImageFilter
    from ..common import atomic_json, digest
    from ..pipeline_models import MODEL_ROOT, SPECS
    initialize()
    import comfy.sd
    import comfy.utils
    import comfy.sample
    import comfy.clip_vision
    from comfy_extras.nodes_triposplat import TripoSplatConditioning, VAEDecodeTripoSplat, _preprocess
    from comfy_extras.nodes_gaussian_splat import SplatToFile3D, SplatToMesh, RenderSplat
    from comfy_extras.nodes_save_3d import mesh_item_to_glb_bytes
    options = settings(request)
    root = Path(request.get('model_root') or MODEL_ROOT)/'coarse_idol'
    directory = output.parent
    stages = {}
    rgba = torch.from_numpy(np.asarray(image).copy()).float()/255
    if options['idol_crop'] == 'real':
        # Photo framing is explicitly separate from the official Comfy crop.
        resized=image.resize((round(image.width*1024/min(image.size)),round(image.height*1024/min(image.size))),Image.Resampling.LANCZOS)
        mask=resized.getchannel('A').filter(ImageFilter.MinFilter(3))
        box=mask.getbbox()
        if box is None:raise ValueError('照片裁切没有有效前景')
        x0,y0,x1,y1=box;cx=(x0+x1)/2;cy=(y0+y1)/2;half=max(x1-x0,y1-y0)*1.7/2
        resized.putalpha(mask)
        cropped=resized.crop((int(cx-half),int(cy-half),int(cx+half),int(cy+half))).resize((1024,1024),Image.Resampling.LANCZOS)
        pixels=np.asarray(cropped).astype(np.float32)/255
        prepared=torch.from_numpy(pixels[:,:,:3]*pixels[:,:,3:4])[None]
    else:prepared = _preprocess(rgba[:,:,:3],rgba[:,:,3],1,1024)
    Image.fromarray(np.rint(prepared[0].numpy().clip(0,1)*255).astype(np.uint8)).save(directory/'idol-input.png')
    import time
    previous_progress=comfy.utils.PROGRESS_BAR_HOOK
    progress_phase=['decoding']
    comfy.utils.set_progress_bar_global_hook(lambda current,total,*_,**kwargs:emit(progress_phase[0],
        progress_current=current,progress_total=total,progress_unit='steps',device='cuda'))
    try:
        started=time.perf_counter()
        progress_phase[0]='encoding_reference'
        emit('encoding_reference',message='编码 Idol 图像条件：DINOv3 H+ 与 FLUX.2 VAE',device='cuda')
        clip = comfy.clip_vision.load(str(root/'clip_vision/dino_v3_vit_h_bf16.safetensors'))
        if clip is None:raise ValueError('Idol DINOv3 H+ 权重无法识别')
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(root/'vae/flux2-vae.safetensors'),safe_load=True))
        with torch.inference_mode():positive,negative,latent=TripoSplatConditioning.execute(clip,vae,prepared).result
        del clip,vae
        release()
        stages['conditioning']=round(time.perf_counter()-started,3)
        emit('loading',message='加载 Idol 扩散模型',device='cuda')
        progress_phase[0]='loading'
        model = comfy.sd.load_diffusion_model(str(root/'diffusion_models/idol_fp16.safetensors'))
        if model is None:raise ValueError('Idol 扩散模型无法识别')
        started=time.perf_counter()
        seed=int(request.get('seed',0));steps=int(request.get('geometry_steps',50))
        progress_phase[0]='sampling'
        emit('sampling',message='Idol 高斯形状采样',steps=steps,device='cuda')
        noise = comfy.sample.prepare_noise(latent['samples'],seed)
        callback=lambda step,*_:emit('sampling',step=step+1,num_inference_steps=steps,device='cuda')
        with torch.inference_mode():
            if options['idol_sampling']=='native_euler':
                import comfy.samplers
                t=torch.linspace(1,0,steps+1)
                sigmas=3*t/(1+2*t)
                samples=comfy.sample.sample_custom(model,noise,options['idol_cfg']+1,
                    comfy.samplers.sampler_object('euler'),sigmas,positive,negative,latent['samples'],
                    callback=callback,seed=seed)
            else:samples=comfy.sample.sample(model,noise,steps,options['idol_cfg'],'dpmpp_2m','simple',
                positive,negative,latent['samples'],seed=seed,callback=callback)
        del model,noise,positive,negative,latent
        release()
        stages['sampling']=round(time.perf_counter()-started,3)
        emit('decoding',message='解码 Idol 原始高斯',device='cuda')
        progress_phase[0]='decoding'
        streams=samples.unbind()
        camera=camera_metadata(streams[1][0,0].cpu().float().tolist())
        atomic_json(directory/'idol-camera.json',camera)
        from safetensors.torch import save_file
        save_file({'latent':streams[0].cpu().contiguous(),'camera':streams[1].cpu().contiguous()},str(directory/'idol-latent.partial.safetensors'))
        (directory/'idol-latent.partial.safetensors').replace(directory/'idol-latent.safetensors')
        vae=comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(root/'vae/idol_decoder_fp16.safetensors'),safe_load=True))
        with torch.inference_mode():splat=VAEDecodeTripoSplat.execute({'samples':samples},vae,options['idol_gaussians'],options['idol_decode_seed'])[0]
        del vae,samples
        release()
        for value in (splat.positions,splat.scales,splat.rotations,splat.opacities,splat.sh):
            if not torch.isfinite(value).all():raise ValueError('Idol 生成了非有限高斯属性')
        if splat.positions.shape[1] != options['idol_gaussians']:raise ValueError('Idol 实际高斯数量与请求不一致')
        files={}
        for format in ('ply','spz','ksplat'):
            target=directory/f'splat.{format}'
            data=SplatToFile3D.execute(splat,format)[0].get_bytes()
            temp=target.with_suffix(target.suffix+'.partial');temp.write_bytes(data);temp.replace(target)
            files[target.name]={'bytes':len(data),'sha256':digest(target)}
        report={'engine':'idol','model':SPECS['coarse_idol'],'gaussians':splat.positions.shape[1],
                'options':options,'sampler':'euler' if options['idol_sampling']=='native_euler' else 'dpmpp_2m',
                'sampling_seed':seed,'input_image_sha256':digest(Path(request['image_path'])),
                'conditioning_image_sha256':digest(directory/'idol-input.png'),
                'scheduler':'native 3*t/(1+2*t)' if options['idol_sampling']=='native_euler' else 'simple','inference_steps':steps,
                'files':files,'geometry_generated':False,'texture_generated':False,
                'camera':camera,'encoder_precision':'DINOv3 H+ BF16',
                'representation':'3d_gaussian_splat','splat_frame':'official Comfy SPLAT; mesh rotates Y and Z by -1',
                'source_revision':'7cebca52227c63b4180ceed448cbd7ad22903b6d'}
        atomic_json(directory/'splat.json',report)
        emit('splat_saved',message='原始高斯已保存，可在网格转换完成前查看或下载',device='cpu')
        if options['idol_extract_mesh']:
            progress_phase[0]='extracting'
            emit('extracting',message='从原始高斯提取顶点色网格',device='cuda')
            started=time.perf_counter()
            meshing=mesher(options['idol_mesh_backend'])
            extra={'solid':options['idol_solid'],'fill':options['idol_fill']} if options['idol_mesh_backend']=='solid' else {}
            with torch.inference_mode():mesh=meshing.execute(splat,int(request.get('resolution',384)),5,
                options['idol_smooth'],options['idol_level'],options['idol_min_component'],options['idol_min_opacity'],2.,**extra)[0]
            if options['idol_align_mesh']:
                yaw=math.atan2(camera['gltf_direction'][0],camera['gltf_direction'][2])
                c,s=math.cos(-yaw),math.sin(-yaw)
                matrix=torch.tensor([[c,0,s],[0,1,0],[-s,0,c]],dtype=mesh.vertices.dtype,device=mesh.vertices.device)
                mesh.vertices=mesh.vertices@matrix.T
                report['mesh_alignment']={'rotation_y_radians':-yaw,'matrix':matrix.cpu().tolist(),
                    'policy':'predicted input yaw faces glTF +Z; original splat unchanged; vertical axis retained'}
            data=mesh_item_to_glb_bytes(mesh,0)
            if not data:raise ValueError('高斯未提取出有效表面；原始高斯仍已保存，可调整网格参数')
            output.write_bytes(data)
            report.update(geometry_generated=True,vertices=int(mesh.vertex_counts[0]),faces=int(mesh.face_counts[0]),
                          mesh_material='unlit vertex color; no UV/PBR generated',representation='splat_and_mesh',
                          mesh_source_revision='0decb39ee5b5e17c200334589fc317dbbe98310e' if options['idol_mesh_backend']=='solid' else '7cebca52227c63b4180ceed448cbd7ad22903b6d')
            stages['mesh']=round(time.perf_counter()-started,3)
            del mesh
            release()
        if options['idol_video']:
            progress_phase[0]='rendering'
            emit('rendering',message='渲染高斯环绕展示视频（静态物体）',device='cuda')
            import av
            with torch.inference_mode():frames=RenderSplat.execute(splat,512,512,75,1.,2.,0.,0.,'#848484','color')[0]
            temp=directory/'splat-preview.partial.mp4'
            with av.open(str(temp),'w') as container:
                stream=container.add_stream('libx264',rate=25);stream.width=stream.height=512;stream.pix_fmt='yuv420p'
                for frame in frames:
                    pic=av.VideoFrame.from_ndarray(np.rint(frame.cpu().numpy().clip(0,1)*255).astype(np.uint8),format='rgb24')
                    for packet in stream.encode(pic):container.mux(packet)
                for packet in stream.encode():container.mux(packet)
            temp.replace(directory/'splat-preview.mp4')
            report['video']={'frames':75,'fps':25,'size':512,'type':'static object camera orbit'}
        report['stage_seconds']=stages
        for name in ('idol-input.png','idol-camera.json','idol-latent.safetensors','splat-preview.mp4'):
            path=directory/name
            if path.is_file():files[name]={'bytes':path.stat().st_size,'sha256':digest(path)}
        atomic_json(directory/'splat.json',report)
        return report
    finally:
        comfy.utils.set_progress_bar_global_hook(previous_progress)
        release()
