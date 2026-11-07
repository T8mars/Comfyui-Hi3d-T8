"""Local, pinned P3-SAM / X-Part adapters; no implicit Hub downloads."""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np

from ..common import atomic_copy, atomic_json, digest, read_json
from . import models


def weight(component, name):
    return models.verified_weight(component, name)


def parameterized_mask_source(source):
    """Fix upstream's two unconditional assignments that ignore caller budgets."""
    assignments = '    point_num = 100000\n    prompt_num = 400\n'
    if source.count(assignments) != 1:
        raise RuntimeError('P3-SAM 源码版本不匹配，不能应用已核对的参数修复')
    return source.replace(assignments, '', 1)


def _p3sam(request, emit):
    import torch
    from safetensors.torch import load_file
    root = models.runtime_root() / 'sources' / 'parts'
    p3 = root / 'P3-SAM'
    sys.path[:0] = [str(p3), str(root / 'XPart' / 'partgen')]
    # Upstream imports `model` and `models` by these names. Each operation has
    # its own interpreter; refusing conflicts avoids silently using another rig.
    if 'model' in sys.modules:
        raise RuntimeError('P3-SAM 独立运行环境出现模块名称冲突')
    module = importlib.import_module('model')
    original_load = module.sonata.load
    sonata_path = weight('sonata', 'sonata.pth')
    module.sonata.load = lambda *args, **kwargs: original_load(str(sonata_path))
    checkpoint = weight('p3sam', 'p3sam/p3sam.safetensors')

    def strict_load(self, ckpt_path=None, state_dict=None, **kwargs):
        if state_dict is None:
            if Path(ckpt_path).resolve() != checkpoint.resolve():
                raise ValueError('P3-SAM 权重来源不匹配')
            state_dict = load_file(str(checkpoint))
        state_dict = {key.removeprefix('dit.'): value for key, value in state_dict.items()}
        # The upstream helper merely prints missing/shape-mismatched weights.
        # Loading through torch itself makes a partial model a concrete error.
        return torch.nn.Module.load_state_dict(self, state_dict, strict=True)

    module.load_state_dict = strict_load
    spec = importlib.util.spec_from_file_location('hi3d_p3sam_auto_mask', p3 / 'demo' / 'auto_mask.py')
    automatic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(automatic)
    exec(compile(parameterized_mask_source(inspect.getsource(automatic.mesh_sam)),
                 str(p3 / 'demo' / 'auto_mask.py'), 'exec'), automatic.__dict__)
    original_timer = automatic.Timer

    class ProgressTimer(original_timer):
        def __enter__(self):
            emit('asset_segmentation', str(self.name), device='cuda' if '特征' in str(self.name) else 'cpu')
            return super().__enter__()

    # Timer's upstream field name is checked below to keep diagnostics from
    # breaking a successful inference when updating the pinned source.
    timer_parameters = inspect.signature(original_timer).parameters
    if 'name' in timer_parameters:
        automatic.Timer = ProgressTimer
    emit('asset_loading', '加载 P3-SAM 与本地 Sonata 权重', device='cpu')
    predictor = automatic.AutoMask(str(checkpoint), point_num=request.get('segment_points', 100000),
        prompt_num=request.get('segment_prompts', 400), threshold=request.get('segment_threshold', .95))
    original_mask = automatic.get_mask
    completed = 0
    total = request.get('segment_prompts', 400)

    def tracked_mask(model, feats, points, prompts, *args, **kwargs):
        nonlocal completed
        emit('asset_segmentation', '预测三维部件边界', device='cuda',
             progress_current=completed, progress_total=total, progress_unit='prompts')
        result = original_mask(model, feats, points, prompts, *args, **kwargs)
        completed += len(prompts)
        emit('asset_segmentation', '部件提示点推理完成', device='cuda',
             progress_current=min(completed, total), progress_total=total, progress_unit='prompts')
        return result

    automatic.get_mask = tracked_mask
    return predictor


def segment(vertices, faces, request, emit, folder):
    import torch
    import trimesh
    if not torch.cuda.is_available():
        raise RuntimeError('P3-SAM 需要 CUDA 显卡')
    # UV seams often duplicate vertices. Weld only exact positions in this
    # inference copy so connectivity-based label repair sees geometric edges.
    # The triangle ordering and the exported source attributes stay untouched.
    inference_vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
    inference_faces = inverse[faces]
    mesh = trimesh.Trimesh(inference_vertices, inference_faces, process=False)
    torch.manual_seed(request.get('seed', 0))
    predictor = _p3sam(request, emit)
    with torch.inference_mode():
        _, labels, returned = predictor.predict_aabb(mesh,
            point_num=request.get('segment_points', 100000),
            prompt_num=request.get('segment_prompts', 400),
            threshold=request.get('segment_threshold', .95), post_process=True,
            clean_mesh_flag=False, save_mid_res=False, show_info=True,
            save_path=str(folder), seed=request.get('seed', 0), is_parallel=False, prompt_bs=4)
    if (not np.array_equal(returned.faces, inference_faces)
            or not np.array_equal(returned.vertices, inference_vertices)):
        raise RuntimeError('P3-SAM 改变了原始面顺序，已拒绝错误材质映射')
    return np.asarray(labels)


def _xpart(emit,*,staged=False):
    import torch
    from safetensors.torch import load_file
    from easydict import EasyDict
    root = models.runtime_root() / 'sources' / 'parts' / 'XPart'
    sys.path.insert(0, str(root))
    from .attention import install_xpart_attention
    install_xpart_attention()
    from partgen.partformer_pipeline import PartFormerPipeline
    from partgen.utils.misc import instantiate_from_config
    values = {}
    for index,(name, argument) in enumerate([('model', 'model'), ('shapevae', 'vae'), ('conditioner', 'conditioner')]):
        emit('asset_loading', f'加载 X-Part {name}', device='cpu',progress_current=index,progress_total=3,progress_unit='models')
        # The official constructor relies on attribute access to nested config
        # fields (notably projection dimensions). Plain dicts skip those layers.
        config = EasyDict(json.loads(weight('parts_full', name + '/config.json').read_text(encoding='utf-8')))
        instance = instantiate_from_config(config)
        state = load_file(str(weight('parts_full', name + '/' + name + '.safetensors')))
        instance.load_state_dict(state, strict=True)
        instance.eval().requires_grad_(False)
        if name=='conditioner':
            from .attention import install_xpart_sonata_attention
            layers=install_xpart_sonata_attention(instance.seg_feat_encoder.sonata)
            emit('asset_loading',f'Sonata {layers} 层使用有界 FP32 点云注意力',device='cpu')
        del state
        # Publish each verified model to its execution device before building
        # the next one, avoiding simultaneous CPU copies of all FP32 models.
        if not staged:emit('asset_model_transfer',f'上传 X-Part {name} 到显存',device='cuda')
        values[argument] = instance.to(device='cpu' if staged else 'cuda',dtype=torch.float32)
        emit('asset_loading',f'X-Part {name} 已读取',device='cpu' if staged else 'cuda',progress_current=index+1,progress_total=3,progress_unit='models')
    config = EasyDict(json.loads(weight('parts_full', 'scheduler/config.json').read_text(encoding='utf-8')))
    values['scheduler'] = instantiate_from_config(config)
    # User-edited boxes are always provided; constructing another P3-SAM would
    # waste RAM/VRAM and could trigger upstream's unfrozen network download.
    pipeline = PartFormerPipeline(**values, bbox_predictor=None, verbose=False)
    if not staged:pipeline.to(device='cuda', dtype=torch.float32)
    def finite(value,label):
        if isinstance(value,torch.Tensor):
            if not torch.isfinite(value).all():raise RuntimeError('X-Part '+label+' 产生非有限数值，已停止，未导出无效网格')
        elif isinstance(value,dict):
            for key,item in value.items():finite(item,label+'/'+str(key))
        elif isinstance(value,(list,tuple)):
            for index,item in enumerate(value):finite(item,label+'/'+str(index))
    for label,module in [('conditioner',pipeline.conditioner),('diffusion',pipeline.model),
                         ('vae',pipeline.vae),('geometry_decoder',pipeline.vae.geo_decoder),
                         ('part_encoder',pipeline.conditioner.geo_encoder),
                         ('part_point_encoder',pipeline.conditioner.geo_encoder.local_encoder.encoder),
                         ('object_point_encoder',pipeline.conditioner.obj_encoder.encoder),
                         ('sonata_feature',pipeline.conditioner.seg_feat_encoder),
                         ('sonata_backbone',pipeline.conditioner.seg_feat_encoder.sonata)]:
        module.register_forward_hook(lambda module,inputs,output,label=label:finite(output,label))
    return pipeline


def validate_completion_coverage(coverage):
    samples = coverage.get('source_vertex_samples', 0)
    missing = coverage.get('vertices_over_3_percent_extent', 0)
    if type(samples) is not int or samples <= 0 or type(missing) is not int or not 0 <= missing <= samples:
        raise RuntimeError('X-Part 原表面覆盖记录无效，已拒绝输出')
    if missing > samples * .05:
        raise RuntimeError('X-Part 明显改变了原部件外形，已停止材质转移并保留原拆件；请调整部件划分后重试。')


def complete(source, target, document, labels, names, request, emit):
    import torch
    import trimesh
    from .scene import combined
    from .parts import write_parts
    from .gltf import Document
    from .join import join_static
    folder = Path(target).parent
    original = folder / 'segmentation'
    original.mkdir(exist_ok=True)
    segmented = original / 'output.glb'
    metadata = write_parts(document, labels, names, segmented,
                           engine='original face partition', source_sha256=digest(source))
    # Publish the unchanged partition before any model load or generation can
    # fail. These flat artifacts remain available through the normal job API.
    for filename,name in [('output.glb','original-segmentation.glb'),
                          ('parts.json','original-parts.json'),('parts.zip','original-parts.zip')]:
        atomic_copy(original/filename,folder/name)
    vertices, faces, _ = combined(document)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    scale = np.ptp(vertices, axis=0).max() / 1.6
    if not np.isfinite(scale) or scale <= 1e-10:
        raise ValueError('模型尺度无效，不能补全部件')
    mesh = trimesh.Trimesh((vertices - center) / scale, faces, process=False)
    boxes = (np.asarray([part['bounds'] for part in metadata['parts']]) - center) / scale
    if np.any(np.diff(boxes, axis=1) <= 1e-8):
        raise ValueError('部件缺少三维体积，请合并薄片或修正拆件后再补全')
    steps = request.get('part_steps', 50)
    pipeline = _xpart(emit,staged=True)
    emit('asset_part_completion', '正在生成独立部件及隐藏表面', device='cuda')
    def callback(index, timestep, outputs):
        emit('asset_part_completion', 'X-Part 部件几何采样', device='cuda',
             progress_current=index + 1, progress_total=steps, progress_unit='steps')
        if index+1==steps:
            temporary=folder/'xpart-latents.tmp.npz'
            np.savez_compressed(temporary,latents=outputs.prev_sample.detach().float().cpu().numpy(),
                source_sha256=np.asarray(digest(source)),boxes=boxes,steps=np.asarray(steps),seed=np.asarray(request.get('seed',0)))
            temporary.replace(folder/'xpart-latents.npz')
            # The remainder of the pinned pipeline only calls VAE decoding.
            # Release diffusion and conditioning weights before extracting parts.
            emit('asset_part_decode','几何采样完成，完整潜变量已保存',device='cuda')
    with torch.inference_mode():
        from .xpart_sampling import generate
        result,generation=generate(pipeline,mesh=mesh,boxes=boxes,steps=steps,
            resolution=request.get('part_resolution',512),seed=request.get('seed',0),
            guidance=request.get('part_guidance',7.5),emit=emit,callback=callback)
    atomic_json(folder/'xpart-generation.json',generation)
    parts = list(result.geometry.values())
    if len(parts) != len(metadata['parts']) or any(not len(part.faces) for part in parts):
        raise RuntimeError('X-Part 部分部件未生成，已保留原分割结果并拒绝不完整成品')
    del pipeline
    import gc
    gc.collect(); torch.cuda.empty_cache()
    paths = []
    transfer_reports=[]
    from .material_transfer import run as transfer_material
    from .retopology import geometry_coverage, visible_mesh
    geometries=[];coverage_reports=[];surface_reports=[]
    # Check every original surface before spending minutes baking a failed
    # reconstruction. New hidden surfaces remain allowed; lost outer surfaces
    # do not. Candidate geometry and the untouched partition remain on failure.
    for index, (part, info) in enumerate(zip(parts, metadata['parts'])):
        part.vertices = part.vertices * scale + center
        stage = folder / 'stages' / 'studio-blender'
        stage.mkdir(parents=True, exist_ok=True)
        geometry = stage / f'completed_{index:04d}.obj'
        # Retain a diagnostic candidate even if float32 preparation rejects it.
        part.export(geometry, include_color=False)
        from .generated_geometry import prepare
        prepared,surface=prepare(part)
        surface.update(part=info['id'],name=info['name'])
        if surface['removed_collapsed_faces'] or surface['merged_float32_vertices']:
            raw=stage/f'completed_{index:04d}.raw.obj'
            atomic_copy(geometry,raw)
            surface.update(raw_geometry=raw.name,raw_sha256=digest(raw))
            emit('asset_part_validation',f'整理新生成部件的 GLB 精度：合并 {surface["merged_float32_vertices"]} 个重合顶点，移除 {surface["removed_collapsed_faces"]} 个坍缩面；原拆件不变',device='cpu')
        part=prepared
        part.export(geometry, include_color=False)
        surface['prepared_sha256']=digest(geometry);surface_reports.append(surface)
        atomic_json(folder/'xpart-surface-preparation.json',surface_reports)
        donor=original/'parts'/f'part_{info["id"]:04d}.glb'
        coverage=geometry_coverage(visible_mesh(Document.load(donor)),part,emit,stage='asset_part_validation')
        record={'part':info['id'],'name':info['name'],'coverage':coverage,
                'source_sha256':digest(donor),'candidate_sha256':digest(geometry),'status':'checking'}
        coverage_reports.append(record)
        try:
            validate_completion_coverage(coverage)
        except RuntimeError:
            record['status']='rejected'
            atomic_json(folder/'part-coverage.json',coverage_reports)
            raise
        record['status']='passed'
        atomic_json(folder/'part-coverage.json',coverage_reports)
        geometries.append((geometry,donor,info))
    for index,(geometry,donor,info) in enumerate(geometries):
        output = stage / f'completed_{index:04d}.glb'
        emit('asset_material_transfer', f'转移部件 {index + 1}/{len(parts)} 的 PBR 材质', device='cpu')
        # The original segmented part is the only material donor; adjacent
        # body/head pieces must not paint each other's newly closed surfaces.
        transfer_reports.append(transfer_material(donor,geometry,output,stage/f'part_{index:04d}',
            obj_output=stage/f'baked_{index:04d}.obj',texture_size=request.get('texture_size',2048),
            projection=request.get('material_projection','nearest'),uv_backend=request.get('uv_backend','auto'),owner=folder,emit=emit))
        paths.append((output, info['id'], info['name']))
    merged = folder / 'completed-parts.glb'
    join_static(paths, merged)
    completed_document = Document.load(merged)
    from .scene import instances
    labels = [np.full(len(row.faces), row.definition['extras']['studio_part'], dtype=np.int32)
              for row in instances(completed_document)]
    final = write_parts(completed_document, labels, names, target,
                        engine='X-Part public light + CPU surface PBR transfer', source_sha256=digest(source))
    projection_name='法线投射' if request.get('material_projection','nearest')=='cage' else '最近表面'
    final.update(geometry_completed=True, original_segmentation='original-segmentation.glb',
        notice=f'X-Part 已重新生成部件并推测隐藏表面；UV 已重建，PBR 通过原部件{projection_name}转移。内部材质和装配边界需要检查。',
        material_transfers=transfer_reports)
    atomic_json(folder / 'parts.json', final)
    # Refresh the archive to include both the untouched segmentation and the
    # reconstruction notice, rather than silently replacing the former result.
    archive_path = folder / 'parts.zip'
    with zipfile_for_update(archive_path) as archive:
        archive.write(segmented, 'original-segmentation.glb')
        archive.write(original / 'parts.json', 'original-parts.json')
    return {'engine': final['engine'], 'parts': len(parts), 'geometry_completed': True,
            'original_segmentation_preserved': True, 'material_transfer': 'CPU '+request.get('material_projection','nearest')+' surface PBR projection',
            'material_transfers':transfer_reports,'geometry_coverage':coverage_reports,
            'surface_preparation':surface_reports,'generation':generation,'notice': final['notice']}


class zipfile_for_update:
    """Replace archive metadata without duplicate entries or partial publication."""
    def __init__(self, path):
        self.path = Path(path)
    def __enter__(self):
        import zipfile
        self.temp = self.path.with_suffix('.zip.tmp')
        self.archive = zipfile.ZipFile(self.temp, 'w', zipfile.ZIP_DEFLATED)
        with zipfile.ZipFile(self.path) as old:
            for info in old.infolist():
                if info.filename != 'parts.json':
                    self.archive.writestr(info, old.read(info))
        self.archive.write(self.path.parent / 'parts.json', 'parts.json')
        return self.archive
    def __exit__(self, kind, value, traceback):
        import os
        self.archive.close()
        try:
            if kind is None:
                os.replace(self.temp, self.path)
        finally:
            self.temp.unlink(missing_ok=True)
