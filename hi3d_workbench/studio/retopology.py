"""Quad remeshing followed by actual PBR transfer onto the new topology."""
from __future__ import annotations

import sys
import ctypes
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh

from .. import common
from ..common import atomic_json, digest, read_json
from .models import runtime_root
from .processes import run_stage, blender_command
from .gltf import Document
from .scene import instances


def visible_mesh(document):
    """Give geometry and its material donor the same visible initial pose."""
    if document.tree.get('skins'):
        from .skin import freeze_current_pose
        document = freeze_current_pose(document)
    vertices, triangles, offset = [], [], 0
    for row in instances(document):
        positions = row.positions.astype(np.float64).copy()
        mesh = document.tree['meshes'][row.mesh]
        weights = document.tree['nodes'][row.node].get('weights', mesh.get('weights', []))
        for weight, morph in zip(weights, row.definition.get('targets', [])):
            if weight and 'POSITION' in morph:
                positions += weight * document.accessor(morph['POSITION'])
        vertices.append(positions @ row.world[:3, :3].T + row.world[:3, 3])
        faces = row.faces[:, [0, 2, 1]] if np.linalg.det(row.world[:3, :3]) < 0 else row.faces
        triangles.append(faces.astype(np.int64, copy=False) + offset)
        offset += len(positions)
    vertices = np.concatenate(vertices)
    triangles = np.concatenate(triangles)
    # glTF duplicates positions at UV/material seams. Geometry-only tools see
    # those as open islands unless the identical coordinates are welded first.
    # No tolerance welding: nearby, distinct surfaces must stay distinct.
    unique, inverse = np.unique(vertices, axis=0, return_inverse=True)
    return trimesh.Trimesh(unique, inverse[triangles], process=False)


def remesh_geometry(payload, stage, emit):
    """Retry material/UV work without repeating a successful native solve."""
    from filelock import FileLock
    quadriflow=payload.get('operation')=='quadriflow_remesh'
    if quadriflow:
        command=blender_command()
        identity={key:payload.get(key) for key in ('target_quads','seed','preserve_sharp','solid_surface')}
        identity.update(schema=1,engine='Blender QuadriFlow',input_sha256=digest(Path(payload['input'])),
            executable_sha256=digest(Path(command[0])),script_sha256=digest(Path(__file__).with_name('blender_ops.py')))
    else:
        identity = {key: payload[key] for key in ('target_quads', 'sharp_angle', 'adaptivity')}
        identity.update(schema=1, input_sha256=digest(Path(payload['input'])),
                        executable_sha256=digest(Path(payload['executable'])),
                        edge_scaling=payload.get('edge_scaling', 1), smooth_normal=0, anisotropy=1)
        command=[sys.executable, '-u', '-m', 'hi3d_workbench.studio.native']
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = common.DATA / 'studio-cache' / 'retopology' / key
    cache.mkdir(parents=True, exist_ok=True)
    cached, output = cache / 'quads.obj', Path(payload['output'])
    hit = False
    def valid_geometry(path):
        mesh = trimesh.load(path, force='mesh', process=False)
        return bool(len(mesh.faces) and np.isfinite(mesh.vertices).all()
                    and np.any(mesh.area_faces > 0))
    with FileLock(str(cache / '.lock'), timeout=30):
        try:
            saved = read_json(cache / 'manifest.json', {})
            hit = (isinstance(saved, dict) and saved.get('identity') == identity and cached.is_file()
                   and saved.get('sha256') == digest(cached) and valid_geometry(cached))
        except (ValueError, TypeError, KeyError, OSError):
            hit = False
        if hit:
            shutil.copy2(cached, output)
            if quadriflow and isinstance(saved.get('geometry_report'),dict):
                atomic_json(Path(payload['report']),saved['geometry_report'])
            emit('asset_retopology', '复用已校验的四边面几何，继续材质与 UV 阶段', device='cpu')
        else:
            run_stage(command, payload, stage)
            if not output.is_file() or not valid_geometry(output):
                raise RuntimeError('重拓扑后端没有生成有效网格')
            temporary = cache / 'quads.obj.tmp'
            shutil.copy2(output, temporary)
            temporary.replace(cached)
            manifest={'identity': identity, 'sha256': digest(cached)}
            if quadriflow:
                manifest['geometry_report']=read_json(Path(payload['report']))
            atomic_json(cache / 'manifest.json', manifest)
    atomic_json(Path(stage) / 'native-cache.json', {'cache_hit': bool(hit), 'identity': identity})
    return bool(hit)


def geometry_proxy(mesh, emit, *, face_limit=200000, max_error=.0005):
    """Bound native geometry work; full-resolution PBR remains the donor."""
    before = len(mesh.faces)
    report = {'source_faces': before, 'inference_faces': before,
              'target_faces': face_limit, 'normalized_error': 0., 'max_normalized_error': max_error}
    if before <= face_limit:
        return mesh, report
    from .mesh import _library
    emit('asset_retopology_prepare', '为高面数模型准备受误差限制的几何副本；原高模保留用于材质转移', device='cpu')
    span = float(np.ptp(mesh.vertices, axis=0).max())
    if not np.isfinite(span) or span <= 1e-10:
        raise ValueError('重拓扑模型尺寸无效')
    vertices = np.ascontiguousarray((mesh.vertices - mesh.bounds.mean(0)) / span, dtype=np.float32)
    indices = np.ascontiguousarray(mesh.faces.reshape(-1), dtype=np.uint32)
    output = np.empty_like(indices)
    error = ctypes.c_float(0)
    library = _library(runtime_root() / ('meshoptimizer.dll' if sys.platform == 'win32' else 'libmeshoptimizer.so'))
    count = library.meshopt_simplifyWithAttributes(output.ctypes.data, indices.ctypes.data, len(indices),
        vertices.ctypes.data, len(vertices), vertices.strides[0], None, 0, None, 0, None,
        face_limit * 3, max_error, 0, ctypes.byref(error))
    if not count or count > len(indices) or count % 3 or not np.isfinite(error.value) or error.value > max_error * 1.001:
        raise RuntimeError('高面数几何副本没有通过面数与误差检查')
    used, remap = np.unique(output[:count], return_inverse=True)
    proxy = trimesh.Trimesh(mesh.vertices[used], remap.reshape(-1, 3), process=False)
    report.update(inference_faces=len(proxy.faces), normalized_error=float(error.value))
    emit('asset_retopology_prepare', f'重拓扑几何副本 {before} → {len(proxy.faces)} 三角面，算法报告归一化误差 {error.value:.6f}', device='cpu')
    return proxy, report


def geometry_coverage(source, target, emit=lambda *a, **kw: None, *, stage='asset_retopology_validation'):
    """Check the original surface too, so a missing part cannot look perfect."""
    import open3d as o3d
    center = source.bounds.mean(0)
    span = float(np.ptp(source.vertices, axis=0).max())
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(np.asarray((target.vertices - center) / span, dtype=np.float32)),
                        o3d.core.Tensor(np.asarray(target.faces, dtype=np.uint32)))
    used = np.unique(source.faces)
    distances = []
    emit(stage, '检查原始表面覆盖', device='cpu',
         progress_current=0, progress_total=len(used), progress_unit='vertices')
    for start in range(0, len(used), 65536):
        positions = np.asarray((source.vertices[used[start:start + 65536]] - center) / span, dtype=np.float32)
        closest = scene.compute_closest_points(o3d.core.Tensor(positions), nthreads=4)
        distances.append(np.linalg.norm(closest['points'].numpy() - positions, axis=1))
        emit(stage, '检查原始表面覆盖', device='cpu',
             progress_current=min(start + 65536, len(used)), progress_total=len(used), progress_unit='vertices')
    distances = np.concatenate(distances)
    if not np.isfinite(distances).all():
        raise RuntimeError('重拓扑表面覆盖检查产生无效距离')
    return {'source_vertex_samples': len(used), 'normalized_distance_p95': float(np.percentile(distances, 95)),
            'normalized_distance_max': float(distances.max()),
            'vertices_over_3_percent_extent': int((distances > .03).sum())}


def validate_coverage(coverage):
    if coverage['vertices_over_3_percent_extent'] > coverage['source_vertex_samples'] * .05:
        raise RuntimeError('重拓扑缺失或明显偏移了较多原表面，已拒绝不完整成品；原模型和原生日志已保留，请检查网格或调整设置。')


def run(source, target, request, emit):
    root = runtime_root()
    engine=request.get('retopology_engine','autoremesher')
    if engine not in {'autoremesher','quadriflow'}:raise ValueError('重拓扑引擎无效')
    surface=request.get('retopology_surface','original')
    if surface not in {'original','solid'}:raise ValueError('重拓扑表面处理方式无效')
    executables=[]
    if engine=='autoremesher':
        executables=list((root/'autoremesher').rglob('autoremesher-core.exe' if sys.platform=='win32' else 'autoremesher-core'))
        if not executables:
            executables=list((root/'autoremesher').rglob('autoremesher.exe' if sys.platform=='win32' else 'autoremesher'))
        if len(executables)!=1:raise RuntimeError('AutoRemesher 环境未就绪，请安装资产加工组件')
    blender_command()  # Validate the shared tessellation environment before processing a dense mesh.
    folder = Path(target).parent
    stage = folder / 'stages' / 'studio-remesh'
    stage.mkdir(parents=True, exist_ok=True)
    document = Document.load(source)
    original_mesh = visible_mesh(document)
    mesh, preparation = geometry_proxy(original_mesh, emit,
        face_limit=60000 if engine=='quadriflow' and surface=='original' else 200000,
        max_error=.002 if engine=='quadriflow' and surface=='original' else .0005)
    input_mesh = stage / 'input.obj'
    mesh.export(input_mesh)
    if surface=='solid':
        prepare=folder/'studio-blender'/'prepare-remesh';prepare.mkdir(parents=True,exist_ok=True)
        solid=prepare/'surface.obj'
        run_stage(blender_command(),{'operation':'prepare_remesh_surface','input':str(input_mesh),
            'output':str(solid),'report':str(prepare/'report.json')},prepare)
        mesh,proxy=geometry_proxy(trimesh.load(solid,force='mesh',process=False),emit,
            face_limit=60000,max_error=.002)
        preparation.update(surface='solid',surface_preparation=read_json(prepare/'report.json'),solid_proxy=proxy)
    if engine=='quadriflow' or surface=='solid':
        from .topology import split_fans
        mesh,topology=split_fans(mesh,emit)
        preparation['topology']=topology
        mesh.export(input_mesh)
    output_mesh = stage / 'quads.obj'
    emit('asset_retopology', '正在重新整理四边面布线', device='cpu')
    if engine=='quadriflow':
        native_stage=folder/'studio-blender'/'quadriflow';native_stage.mkdir(parents=True,exist_ok=True)
        payload={'operation':'quadriflow_remesh','input':str(input_mesh),'output':str(output_mesh),
            'report':str(native_stage/'report.json'),'target_quads':request.get('target_quads',15000),
            'seed':request.get('seed',0),'preserve_sharp':request.get('preserve_sharp',False),
            'solid_surface':surface=='solid'}
        label='Blender QuadriFlow'
    else:
        payload = {'tool': 'autoremesher', 'executable': str(executables[0]), 'input': str(input_mesh),
                   'output': str(output_mesh), 'report': str(stage / 'native-report.txt'),
                   'target_quads': request.get('target_quads', 15000),
                   'sharp_angle': request.get('sharp_angle', 60), 'adaptivity': request.get('adaptivity', 1),
                   'edge_scaling': request.get('edge_scaling', 1)}
        native_stage=stage;label='AutoRemesher'
    geometry_cache_hit = remesh_geometry(payload, native_stage, emit)
    tessellate=folder/'studio-blender'/'tessellate';tessellate.mkdir(parents=True,exist_ok=True)
    triangles=tessellate/'triangles.obj'
    emit('asset_retopology_validation','按多边形形状划分预览三角面；四边面 OBJ 保留',device='cpu')
    run_stage(blender_command(),{'operation':'tessellate_retopology','input':str(output_mesh),
        'output':str(triangles),'report':str(tessellate/'report.json')},tessellate)
    emit('asset_retopology_validation', '检查原始表面是否有部件遗漏或明显偏移', device='cpu')
    geometry=trimesh.load(triangles,force='mesh',process=False)
    coverage = geometry_coverage(original_mesh, geometry, emit)
    atomic_json(stage / 'coverage.json', coverage)
    validate_coverage(coverage)
    emit('asset_material_transfer', '正在展开新 UV 并转移原材质与表面细节', device='cpu')
    bake = folder / 'stages' / 'studio-material'
    from .material_transfer import run as transfer_material
    details=transfer_material(source,output_mesh,target,bake,texture_size=request.get('texture_size',2048),
        obj_output=folder/'output.obj',projection=request.get('material_projection','nearest'),
        triangulated_geometry=triangles,uv_backend=request.get('uv_backend','auto'),owner=folder,emit=emit)
    if surface=='solid':details['warnings'].append(preparation['surface_preparation']['notice'])
    edge_counts=np.bincount(geometry.edges_unique_inverse)
    area_epsilon=float(np.ptp(geometry.vertices,axis=0).max())**2*1e-14
    topology={'boundary_edges':int((edge_counts==1).sum()),'nonmanifold_edges':int((edge_counts>2).sum()),
        'degenerate_triangles':int((geometry.area_faces<=area_epsilon).sum()),'winding_consistent':bool(geometry.is_winding_consistent)}
    if topology['boundary_edges'] or topology['nonmanifold_edges']:
        details['warnings'].append(f"结果有 {topology['boundary_edges']} 条边界边、{topology['nonmanifold_edges']} 条非流形边，打印或动画制作前请检查。")
    if topology['degenerate_triangles']:
        details['warnings'].append(f"结果仍有 {topology['degenerate_triangles']} 个退化三角面，请检查细薄结构。")
    if document.tree.get('skins') or document.tree.get('animations'):
        details['warnings'].append('已按模型初始可见姿态重拓扑；新拓扑需重新绑定骨骼，原骨架和动画保留在原文件。')
    if coverage['vertices_over_3_percent_extent']:
        details['warnings'].append(f"{coverage['vertices_over_3_percent_extent']} 个原始顶点距离新表面超过模型尺寸的 3%，请检查小部件或局部形状是否丢失。")
    if details['quad_faces'] > request.get('target_quads', 15000) * 1.25:
        details['warnings'].append(f"实际四边面数 {details['quad_faces']} 高于目标 {request.get('target_quads', 15000)}；原生重拓扑没有达到面数预算，请检查形状后调整设置或继续减面。")
    return {'engine': label+' + CPU surface PBR transfer', **details,
            'target_quads': request.get('target_quads', 15000), 'geometry_cache_hit': geometry_cache_hit,
            'geometry_preparation': preparation, 'geometry_coverage': coverage,'output_topology':topology,
            'retopology_engine':engine,'retopology_surface':surface,
            'geometry_tessellation':read_json(tessellate/'report.json'),
            'edge_scaling': request.get('edge_scaling', 1),
            'target_quad_budget_reached': details['quad_faces'] <= request.get('target_quads', 15000)}
