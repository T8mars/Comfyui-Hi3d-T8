"""Parallelize the four independent, unchanged upstream Telea operations.

GPU sampling and mesh transforms follow Microsoft's TRELLIS.2 texturing
postprocess_mesh at 75fbf0183001ed9876c8dbb35de6b68552ee08bd (MIT).
Only independent CPU inpainting is scheduled concurrently; no global OpenCV
thread settings, interpolation, radii or channel quantization are changed.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time


def inpaint_channels(cv2, arrays, mask, emit, *, workers=None):
    import psutil
    labels = ('颜色', '金属度', '粗糙度', '透明度')
    if len(arrays) != 4:
        raise ValueError('PBR baking requires four channels')
    # Extra output/work buffers scale with the atlas; fall back to serial on
    # a constrained host instead of making memory pressure or paging worse.
    pixels = int(mask.size)
    reserve = max(2 * 2**30, pixels * 64)
    available = psutil.virtual_memory().available
    maximum = min(4, os.cpu_count() or 1) if available >= reserve else 1
    count = min(maximum, max(1, workers)) if workers is not None else maximum
    started = time.perf_counter()
    emit('baking', phase='texture_inpaint', device='cpu', progress_current=0,
         progress_total=4, progress_unit='textures', inpaint_workers=count,
         message=f'CPU PBR 贴图补边：{count} 路并行 · 0/4')
    results = [None] * 4
    def work(index):
        return cv2.inpaint(arrays[index], mask, 3 if index == 0 else 1, cv2.INPAINT_TELEA)
    with ThreadPoolExecutor(max_workers=count, thread_name_prefix='pbr-inpaint') as executor:
        pending = {executor.submit(work, index): index for index in range(4)}
        for completed, future in enumerate(as_completed(pending), 1):
            index = pending[future]
            results[index] = future.result()  # Real errors propagate; executor joins every worker.
            emit('baking', phase='texture_inpaint', device='cpu', progress_current=completed,
                 progress_total=4, progress_unit='textures', texture_channel=labels[index],
                 inpaint_workers=count, message=f'{labels[index]}补边完成 · {completed}/4')
    return results, {'workers': count, 'seconds': round(time.perf_counter() - started, 3),
                     'algorithm': 'original OpenCV INPAINT_TELEA; radii 3/1/1/1'}


def bake(pipeline, mesh, pbr_voxel, resolution, texture_size, emit):
    import cv2
    import flex_gemm
    import numpy as np
    import nvdiffrast.torch as dr
    from PIL import Image
    import torch
    import trimesh
    if getattr(mesh.visual, 'uv', None) is None:
        raise ValueError('Parallel PBR baking requires validated existing UVs')
    # Own writable arrays: Trimesh cache copies may share read-only normal
    # arrays, and rotating a cache view would also mutate another mesh.
    vertices = np.array(mesh.vertices, copy=True)
    faces = np.asarray(mesh.faces)
    normals = np.array(mesh.vertex_normals, copy=True)
    vertices_torch = torch.from_numpy(vertices).float().cuda()
    faces_torch = torch.from_numpy(faces).int().cuda()
    uvs = mesh.visual.uv.copy()
    uvs[:, 1] = 1 - uvs[:, 1]
    uvs_torch = torch.from_numpy(uvs).float().cuda()
    emit('baking', phase='texture_raster', device='cuda', message='GPU UV 光栅化与材质采样')
    ctx = dr.RasterizeCudaContext()
    uvs_torch = torch.cat([uvs_torch * 2 - 1, torch.zeros_like(uvs_torch[:, :1]),
                           torch.ones_like(uvs_torch[:, :1])], dim=-1).unsqueeze(0)
    rast, _ = dr.rasterize(ctx, uvs_torch, faces_torch, resolution=[texture_size, texture_size])
    mask = rast[0, ..., 3] > 0
    pos = dr.interpolate(vertices_torch.unsqueeze(0), rast, faces_torch)[0][0]
    attrs = torch.zeros(texture_size, texture_size, pbr_voxel.shape[1], device=pipeline.device)
    attrs[mask] = flex_gemm.ops.grid_sample.grid_sample_3d(
        pbr_voxel.feats, pbr_voxel.coords,
        shape=torch.Size([*pbr_voxel.shape, *pbr_voxel.spatial_shape]),
        grid=((pos[mask] + 0.5) * resolution).reshape(1, -1, 3), mode='trilinear')
    # Preserve each original per-channel cast/quantization, including opacity.
    arrays = [np.clip(attrs[..., pipeline.pbr_attr_layout[name]].cpu().numpy() * 255, 0, 255).astype(np.uint8)
              for name in ('base_color', 'metallic', 'roughness', 'alpha')]
    inpaint_mask = (~mask.cpu().numpy()).astype(np.uint8)
    (base_color, metallic, roughness, alpha), report = inpaint_channels(cv2, arrays, inpaint_mask, emit)
    metallic, roughness, alpha = (value[..., None] for value in (metallic, roughness, alpha))
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
        metallicFactor=1.0, roughnessFactor=1.0, alphaMode='OPAQUE', doubleSided=True)
    vertices[:, 1], vertices[:, 2] = vertices[:, 2], -vertices[:, 1]
    normals[:, 1], normals[:, 2] = normals[:, 2], -normals[:, 1]
    uvs[:, 1] = 1 - uvs[:, 1]
    result = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, process=False,
                            visual=trimesh.visual.TextureVisuals(uv=uvs, material=material))
    return result, report
