"""Material-aware GLB validation/export, independent of the geometry-only bridge."""
from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import numpy as np
import trimesh


def inspect_asset_geometry(path: Path) -> dict:
    """Measure topology without mistaking coincident UV seam vertices for holes.

    Only exactly equal positions are combined in a temporary diagnostic mesh.
    There is no tolerance welding, repair, or modification of the deliverable.
    """
    from .geometry import load_mesh, inspect_mesh
    mesh = load_mesh(path)
    positions, inverse = np.unique(np.asarray(mesh.vertices), axis=0, return_inverse=True)
    topology = trimesh.Trimesh(positions, inverse[np.asarray(mesh.faces)], process=False)
    info = inspect_mesh(topology)
    info.update(vertices=len(mesh.vertices), topology_vertices=len(positions),
                topology_basis='exact duplicate positions combined for diagnosis only; no tolerance or repair',
                warnings=['拓扑统计临时合并完全相同坐标的 UV 接缝顶点；未修改交付网格，也不代表可打印。'])
    return info


def read_glb_container(path: Path) -> dict:
    """Read a self-contained GLB without resolving external images or buffers."""
    with path.open('rb') as stream:
        magic, version, size = struct.unpack('<4sII', stream.read(12))
        if magic != b'glTF' or version != 2 or size != path.stat().st_size:
            raise ValueError('贴图结果不是完整的 GLB 2 文件')
        length, kind = struct.unpack('<I4s', stream.read(8))
        if kind != b'JSON' or length > size - 20:
            raise ValueError('贴图 GLB 缺少有效场景描述')
        tree = json.loads(stream.read(length).decode('utf-8').rstrip('\x00 \t\n\r'))
    for buffer in tree.get('buffers', []):
        if buffer.get('uri') and not buffer['uri'].startswith('data:'):
            raise ValueError('贴图 GLB 仍依赖外部缓冲区')
    images, textures = tree.get('images', []), tree.get('textures', [])
    for im in images:
        if im.get('uri') and not str(im['uri']).startswith('data:image/'):
            raise ValueError('贴图 GLB 仍依赖外部图片')
        if 'bufferView' not in im and not str(im.get('uri','')).startswith('data:image/'):
            raise ValueError('贴图 GLB 包含未嵌入的图片')
    return tree


def usable_texture_uv(mesh) -> bool:
    """The painters need non-degenerate triangles inside one unit atlas."""
    uv=getattr(mesh.visual,'uv',None)
    if uv is None:return False
    uv=np.asarray(uv)
    if uv.shape!=(len(mesh.vertices),2) or not np.isfinite(uv).all():return False
    # These bakers do not implement UDIM/tiled wrapping.
    if np.min(uv)<-1e-6 or np.max(uv)>1+1e-6:return False
    triangles=uv[np.asarray(mesh.faces)]
    edge1=triangles[:,1]-triangles[:,0];edge2=triangles[:,2]-triangles[:,0]
    area=np.abs(edge1[:,0]*edge2[:,1]-edge1[:,1]*edge2[:,0])
    if float(area.sum())*.5>1.001:return False
    # Exact mirrored/reused triangles paint unrelated faces into the same pixels.
    # This is a fast guard, not an exhaustive arbitrary-polygon overlap solver.
    order=np.lexsort((triangles[:,:,1],triangles[:,:,0]),axis=1)
    canonical=np.take_along_axis(triangles,order[:,:,None],axis=1).reshape(-1,6)
    significant=canonical[area>1e-12]
    if len(np.unique(significant,axis=0))!=len(significant):return False
    surface=np.asarray(mesh.area_faces)
    total=float(surface.sum())
    return total>0 and float(surface[area>1e-16].sum())/total>=.999


def _texture_scene_mesh(scene):
    """Bake scene instances without discarding authored shading normals.

    Scene.dump/to_mesh use an uncached copy, which loses glTF NORMAL data.
    Transform normals explicitly as inverse-transposed column vectors (or
    row vectors multiplied by inverse(linear)); mirror winding separately.
    """
    if isinstance(scene,trimesh.Trimesh):
        scene=trimesh.Scene(scene)
    if not isinstance(scene,trimesh.Scene):raise ValueError('文件必须包含三角网格')
    parts=[]
    shading=[]
    for node in scene.graph.nodes_geometry:
        transform,name=scene.graph[node]
        geometry=scene.geometry[name]
        if not isinstance(geometry,trimesh.Trimesh):continue
        transform=np.asarray(transform,dtype=np.float64)
        if (transform.shape!=(4,4) or not np.isfinite(transform).all() or
                not np.allclose(transform[3],[0,0,0,1],rtol=0,atol=1e-12)):
            raise ValueError('贴图输入包含无效或非仿射节点变换')
        linear=transform[:3,:3]
        try:inverse=np.linalg.inv(linear)
        except np.linalg.LinAlgError as exc:
            raise ValueError('贴图输入包含不可逆节点变换，无法保留原始法线') from exc
        if not np.isfinite(inverse).all():raise ValueError('贴图输入节点变换无法可靠变换法线')
        normals=np.array(geometry.vertex_normals,dtype=np.float64,copy=True)
        if normals.shape!=geometry.vertices.shape or not np.isfinite(normals).all():
            raise ValueError('贴图输入包含无效顶点法线')
        if not np.array_equal(linear,np.eye(3)):
            normals=normals@inverse
            # Stable normalization also preserves zero normals on unused
            # vertices instead of fabricating a direction for them.
            largest=np.max(np.abs(normals),axis=1,keepdims=True)
            normals=np.divide(normals,largest,out=np.zeros_like(normals),where=largest>0)
            length=np.linalg.norm(normals,axis=1,keepdims=True)
            normals=np.divide(normals,length,out=np.zeros_like(normals),where=length>0)
        if not np.isfinite(normals).all():raise ValueError('贴图输入节点变换产生无效法线')
        part=geometry.copy(include_cache=True)
        part.vertices=trimesh.transformations.transform_points(geometry.vertices,transform)
        if np.linalg.slogdet(linear)[0]<0:
            part.faces=np.ascontiguousarray(geometry.faces[:,::-1])
        # Cached face normals belong to the previous frame. Rebuild those
        # from transformed geometry while restoring the authored vertex data.
        part._cache.clear()
        part.vertex_normals=normals
        parts.append(part)
        shading.append(normals)
    if not parts:raise ValueError('文件必须包含三角网格')
    mesh=parts[0] if len(parts)==1 else trimesh.util.concatenate(parts)
    # Concatenation's one-mesh shortcut and future Trimesh versions may copy
    # without caches. Keep the per-node normal ordering explicit.
    mesh.vertex_normals=np.vstack(shading)
    return mesh


def load_texture_input(path: Path):
    """Retain valid single-material UVs; rebuild ambiguous multi-material atlases."""
    from .geometry import validate
    if Path(path).suffix.lower()=='.glb':read_glb_container(Path(path))
    scene=trimesh.load(path,force='scene',process=False,resolver={},allow_remote=False)
    mesh=_texture_scene_mesh(scene)
    validate(mesh)
    uv=getattr(mesh.visual,'uv',None)
    material=getattr(mesh.visual,'material',None)
    multiple=len(getattr(material,'materials',[]))>1
    valid=usable_texture_uv(mesh)
    if valid and not multiple:
        return mesh,'input_single_material_uv'
    if uv is not None:
        mesh.visual=trimesh.visual.ColorVisuals(mesh=mesh)
    return mesh,'rebuild_multimaterial_uv' if multiple else 'generate_uv'


def inspect_textured_glb(path: Path) -> dict:
    """Require actual embedded UV-based base-color and metallic/roughness maps."""
    tree=read_glb_container(path)
    images,textures=tree.get('images',[]),tree.get('textures',[])
    materials = tree.get('materials', [])
    def mapped(info):
        index = info.get('index') if isinstance(info, dict) else None
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(textures):
            return False
        source = textures[index].get('source')
        if not isinstance(source, int) or not 0 <= source < len(images):
            return False
        im = images[source]
        return 'bufferView' in im or str(im.get('uri', '')).startswith('data:image/')
    count = 0
    for mesh in tree.get('meshes', []):
        for primitive in mesh.get('primitives', []):
            count += 1
            material = primitive.get('material', -1)
            if not isinstance(material, int) or not 0 <= material < len(materials):
                raise ValueError('贴图结果有网格缺少材质')
            pbr = materials[material].get('pbrMetallicRoughness', {})
            if not mapped(pbr.get('baseColorTexture')) or not mapped(pbr.get('metallicRoughnessTexture')):
                raise ValueError('贴图结果缺少嵌入式颜色或金属度/粗糙度贴图')
            for info in (pbr['baseColorTexture'], pbr['metallicRoughnessTexture']):
                if f'TEXCOORD_{info.get("texCoord", 0)}' not in primitive.get('attributes', {}):
                    raise ValueError('贴图结果缺少材质使用的 UV 坐标')
    if not count:
        raise ValueError('贴图结果没有可显示的网格')
    # Decode image bytes and geometry as well as validating JSON references.
    scene = trimesh.load(path, force='scene', process=False, resolver={}, allow_remote=False)
    if not scene.geometry or not np.isfinite(scene.bounds).all():
        raise ValueError('贴图结果几何无效')
    for mesh in scene.geometry.values():
        uv = getattr(mesh.visual, 'uv', None)
        material = getattr(mesh.visual, 'material', None)
        if uv is None or not np.isfinite(uv).all() or len(uv) != len(mesh.vertices):
            raise ValueError('贴图结果 UV 无效')
        for key in ('baseColorTexture', 'metallicRoughnessTexture'):
            image = getattr(material, key, None)
            if image is None:
                raise ValueError('贴图图片无法解码')
            image.load()
    return {'texture_generated': True, 'uv': True, 'embedded': True,
            'images': len(images), 'materials': len(materials), 'primitives': count,
            'channels': ['base_color', 'metallic', 'roughness']}


def export_textured_glb(source: Path, target: Path, height_mm: float | None = None, height_axis: str = 'Z') -> dict:
    if height_axis not in {'X','Y','Z'}:raise ValueError('尺寸轴应为 X、Y 或 Z')
    info = inspect_textured_glb(source)
    scene = trimesh.load(source, force='scene', process=False, resolver={}, allow_remote=False)
    if height_mm is not None:
        height = float(scene.extents['XYZ'.index(height_axis)])
        if not np.isfinite(height_mm) or not 0 < height_mm <= 100000 or height <= 1e-12:
            raise ValueError('目标尺寸或所选轴无效')
        scene.apply_scale(height_mm / 1000 / height)
        scene.export(target)
        inspect_textured_glb(target)
    else:
        shutil.copy2(source, target)
    return {'format': '.glb', 'height_axis': height_axis, 'height_mm': height_mm,
            'units': 'm' if height_mm else 'normalized/unknown', 'bounds': scene.bounds.tolist(), **info}


def export_asset_glb(source: Path, target: Path, height_mm: float | None = None, height_axis: str = 'Z') -> dict:
    """Export uploaded GLB materials without calling them AI-generated textures."""
    if height_axis not in {'X','Y','Z'}:raise ValueError('尺寸轴应为 X、Y 或 Z')
    tree=read_glb_container(source)
    scene=trimesh.load(source,force='scene',process=False,resolver={},allow_remote=False)
    if not scene.geometry or not np.isfinite(scene.bounds).all():raise ValueError('模型几何无效')
    if height_mm is not None:
        height=float(scene.extents['XYZ'.index(height_axis)])
        if not np.isfinite(height_mm) or not 0<height_mm<=100000 or height<=1e-12:raise ValueError('目标尺寸或所选轴无效')
        scene.apply_scale(height_mm/1000/height)
        scene.export(target)
        read_glb_container(target)
    else:shutil.copy2(source,target)
    return {'format':'.glb','height_axis':height_axis,'height_mm':height_mm,
            'units':'m' if height_mm else 'normalized/unknown','bounds':scene.bounds.tolist(),
            'material_preserved':bool(tree.get('materials')),'texture_generated':False}
