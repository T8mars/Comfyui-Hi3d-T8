"""Geometry-only calls into the pinned official TRELLIS.2 pipeline."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def generate(request, image, output, emit):
    import torch
    import trimesh
    from .run_stage import source_dir
    from ..pipeline_models import MODEL_ROOT, SPECS
    source_dir('trellis2')
    os.environ['ATTN_BACKEND'] = 'sdpa'
    os.environ['SPARSE_ATTN_BACKEND'] = 'sdpa'
    os.environ['SPARSE_CONV_BACKEND'] = 'flex_gemm'
    from .attention import install
    attention = install(request.get('pipeline_attention', 'auto'), emit=emit)
    emit('loading',phase='runtime_import',device='cpu',message='载入 TRELLIS.2 代码与本地依赖')
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.pipelines import rembg

    class PreparedForeground:
        def __init__(self, **kwargs): pass
        def to(self, device): return self
        def cpu(self): return self
        def __call__(self, value): return value.convert('RGBA')

    rembg.PreparedForeground = PreparedForeground
    root = Path(request.get('model_root') or MODEL_ROOT)
    config = json.loads((root / 'coarse_trellis2' / 'pipeline.json').read_text(encoding='utf-8'))
    resolution = int(request.get('resolution', 512))
    requested_resolution=resolution
    models = config['args']['models']
    for key in list(models):
        if key.startswith('tex_') or resolution == 512 and key == 'shape_slat_flow_model_1024':
            del models[key]
        elif key == 'sparse_structure_decoder':
            models[key] = str(root / 'trellis_structure' / 'ckpts/ss_dec_conv3d_16l8_fp16')
        else:
            models[key] = str(root / 'coarse_trellis2' / models[key])
    config['args']['image_cond_model']['args']['model_name'] = str(root / 'texture_encoder')
    config['args']['rembg_model'] = {'name': 'PreparedForeground', 'args': {}}
    config['args']['low_vram'] = bool(request.get('offload', True))
    local_config = output.parent / 'geometry-runtime.json'
    local_config.write_text(json.dumps(config, ensure_ascii=True, indent=2), encoding='ascii')
    emit('loading', message='加载 TRELLIS.2 图生几何模型')
    from trellis2 import models as official_models
    from .loading_progress import checkpoint_progress
    with checkpoint_progress(official_models,config['args']['models'],emit):
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(output.parent), config_file=local_config.name)
    emit('loading',phase='model_cuda',device='cuda',message='模型已载入，设置 CUDA 执行环境')
    pipeline.cuda()
    from .sampler_progress import track_sampler
    track_sampler(pipeline.sparse_structure_sampler, emit, 'structure', '生成空间结构')
    track_sampler(pipeline.shape_slat_sampler, emit, 'geometry', '生成几何细节')
    with torch.inference_mode():
        torch.manual_seed(int(request.get('seed', 0)))
        image = pipeline.preprocess_image(image)
        image.save(output.parent / 'conditioning.png')
        emit('conditioning', message='编码参考图', device='cuda')
        cond_512 = pipeline.get_cond([image], 512)
        cond = cond_512 if resolution == 512 else pipeline.get_cond([image], 1024)
        emit('sampling', message='生成空间结构')
        sampler_params = {'steps': int(request.get('geometry_steps',12))}
        coords = pipeline.sample_sparse_structure(cond_512, 32, sampler_params=sampler_params)
        if request.get('offload', True): torch.cuda.empty_cache()
        emit('sampling', message='生成几何细节')
        if resolution == 512:
            slat = pipeline.sample_shape_slat(cond, pipeline.models['shape_slat_flow_model_512'], coords, sampler_params)
        else:
            slat, resolution = pipeline.sample_shape_slat_cascade(cond_512,cond,
                pipeline.models['shape_slat_flow_model_512'],pipeline.models['shape_slat_flow_model_1024'],
                512,resolution,coords,sampler_params)
        if request.get('offload', True): torch.cuda.empty_cache()
        emit('decoding', message='解码网格表面', device='cuda')
        meshes, _ = pipeline.decode_shape_slat(slat, resolution)
        mesh = meshes[0]
        raw_vertices = mesh.vertices.detach().cpu().numpy()[:, [0, 2, 1]].copy()
        raw_vertices[:, 2] *= -1
        raw = trimesh.Trimesh(raw_vertices, mesh.faces.detach().cpu().numpy(), process=False)
        raw.export(output.parent / 'raw-geometry.glb')
        processing = {'enabled': bool(request.get('geometry_cleanup', True)), 'raw_faces': len(raw.faces)}
        if processing['enabled']:
            # Follow the official opaque-asset exporter: fill short boundary
            # loops, narrow-band dual-contouring remesh, then simplify. Raw
            # open-surface geometry remains available as a separate artifact.
            import cumesh
            target_faces = int(request.get('target_faces', 1000000))
            if target_faces not in {100000,300000,1000000}:
                raise ValueError('无效几何面数预算')
            emit('postprocessing', message='整理 TRELLIS.2 开放表面并保留原始几何', device='cuda')
            started = time.perf_counter()
            torch.cuda.empty_cache()
            handler = cumesh.CuMesh()
            handler.init(mesh.vertices.contiguous(), mesh.faces.contiguous())
            handler.fill_holes(max_hole_perimeter=.03)
            native_vertices, native_faces = handler.read()
            bvh = cumesh.cuBVH(native_vertices,native_faces)
            remeshed = cumesh.remeshing.remesh_narrow_band_dc(native_vertices,native_faces,
                center=torch.zeros(3,device='cuda'),scale=(resolution+3)/resolution,
                resolution=resolution,band=1,project_back=0,verbose=False,bvh=bvh)
            handler.init(*remeshed)
            handler.simplify(target_faces,verbose=False)
            final_vertices, final_faces = handler.read()
            mesh.vertices, mesh.faces = final_vertices, final_faces
            processing.update(method='official CuMesh opaque-asset narrow-band DC; band=1, project_back=0',
                              target_faces=target_faces,seconds=round(time.perf_counter()-started,3))
        # TRELLIS representation is Z-up. Official texturing export uses
        # glTF Y-up (x,z,-y); keep the same interchange frame for all engines.
        vertices = mesh.vertices.detach().cpu().numpy()
        vertices = vertices[:, [0, 2, 1]].copy()
        vertices[:, 2] *= -1
        result = trimesh.Trimesh(vertices, mesh.faces.detach().cpu().numpy(), process=False)
        if not len(result.faces):
            raise RuntimeError('TRELLIS.2 没有生成有效几何')
        emit('exporting', faces=len(result.faces))
        result.export(output)
    return {'model': SPECS['coarse_trellis2'], 'image_encoder': SPECS['texture_encoder'],
            'engine': 'trellis2', 'geometry_generated': True, 'texture_generated': False,
            'postprocessing': processing, 'raw_geometry_file': 'raw-geometry.glb',
            'pipeline_type': '512' if resolution == 512 else '1024_cascade',
            'requested_resolution':requested_resolution,'actual_resolution':resolution,
            'vertices': len(result.vertices), 'faces': len(result.faces),
            'inference_steps': sampler_params['steps'], 'attention': attention,
            'transform': {'coordinate_system': 'glTF Y-up', 'axis_mapping': '[x,z,-y]',
                          'unit_scale': 'generator canonical; no absolute scale'}}
