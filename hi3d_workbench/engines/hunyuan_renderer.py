"""Windows raster adapter for the pinned Hunyuan Paint renderer.

Retains its camera, normal encoding and baking code. nvdiffrast replaces the
unavailable custom CUDA extension; export uses trimesh, with no Blender import.
"""
from __future__ import annotations


def load_renderer(source):
    import sys
    import types
    import numpy as np
    import torch
    import nvdiffrast.torch as dr
    sys.path.insert(0, str(source / 'hy3dpaint'))
    context = dr.RasterizeCudaContext()
    raster = types.ModuleType('custom_rasterizer')

    def rasterize(pos, tri, resolution):
        # Official CR maps [-1,1] to [0.5,size-0.5], rather than [0,size].
        height, width = resolution
        corrected = pos.float().clone()
        corrected[..., 0] *= (width - 1) / width
        corrected[..., 1] *= (height - 1) / height
        result, _ = dr.rasterize(context, corrected.contiguous(), tri.int().contiguous(), resolution=resolution)
        result = result[0]
        valid = result[..., 3] > 0
        barycentric = torch.stack([result[..., 0], result[..., 1], 1 - result[..., 0] - result[..., 1]], -1)
        barycentric[~valid] = 0
        return result[..., 3].int(), barycentric

    def interpolate(attributes, faces, barycentric, triangles):
        index = (faces.long() - 1).clamp_min(0)
        colors = attributes[0, triangles.long()[index]]
        return (colors * barycentric[..., None]).sum(-2).unsqueeze(0)

    raster.rasterize, raster.interpolate = rasterize, interpolate
    sys.modules['custom_rasterizer'] = raster
    mesh_utils = types.ModuleType('DifferentiableRenderer.mesh_utils')

    def load_mesh(mesh):
        return (np.array(mesh.vertices, dtype=np.float32, copy=True),
                np.array(mesh.faces, dtype=np.int32, copy=True),
                np.array(mesh.visual.uv, dtype=np.float32, copy=True),
                np.array(mesh.faces, dtype=np.int32, copy=True), None)

    def save_mesh(*args, **kwargs):
        raise RuntimeError('Use the workbench PBR GLB exporter')

    mesh_utils.load_mesh, mesh_utils.save_mesh = load_mesh, save_mesh
    sys.modules['DifferentiableRenderer.mesh_utils'] = mesh_utils
    from DifferentiableRenderer.MeshRender import MeshRender
    # This port uses OpenCV's UV inpainting, rather than the optional C++
    # vertex-inpainting extension. The report records that difference.
    original = MeshRender.uv_inpaint

    def inpaint(self, texture, mask, **kwargs):
        kwargs['vertex_inpaint'] = False
        return original(self, texture, mask, **kwargs)

    MeshRender.uv_inpaint = inpaint
    return MeshRender
