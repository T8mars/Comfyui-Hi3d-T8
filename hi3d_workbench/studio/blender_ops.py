"""Blender side of material transfer and animation interchange.

Executed with Blender's Python, outside the model inference interpreter.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import bpy
import numpy as np


def emit(stage, message, **details):
    print('HI3D_EVENT ' + json.dumps({'stage': stage, 'message': message, **details},
                                   ensure_ascii=False, allow_nan=False), flush=True)


def load_mesh(path):
    bpy.ops.import_scene.gltf(filepath=str(path))
    objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not objects:
        raise ValueError('模型不包含可烘焙网格')
    return objects


def create_quad_mesh(path):
    vertices, faces = [], []
    with Path(path).open(encoding='utf-8', errors='strict') as stream:
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == 'v':
                values = [float(value) for value in fields[1:4]]
                if len(values) != 3 or not all(math.isfinite(value) for value in values):
                    raise ValueError('重拓扑顶点无效')
                # glTF Y-up -> Blender Z-up, matching the glTF importer.
                vertices.append((values[0], -values[2], values[1]))
            elif fields[0] == 'f':
                face = [int(value.split('/')[0]) - 1 for value in fields[1:]]
                if len(face) < 3 or any(index < 0 or index >= len(vertices) for index in face):
                    raise ValueError('重拓扑面索引无效')
                faces.append(face)
    if not vertices or not faces:
        raise ValueError('重拓扑输出为空')
    mesh = bpy.data.meshes.new('Retopology')
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new('Retopology', mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def save_quad_mesh(obj, path, *, triangles=False):
    """Write glTF-axis geometry without an implicit OBJ axis conversion."""
    path=Path(path);temporary=path.with_suffix('.partial.obj')
    with temporary.open('w',encoding='utf-8',newline='\n') as stream:
        stream.write('# Hi3D geometry in original world coordinates, Y up\n')
        for vertex in obj.data.vertices:
            x,y,z=vertex.co
            if not all(math.isfinite(float(v)) for v in (x,y,z)):
                raise ValueError('重拓扑产生无效坐标')
            stream.write(f'v {x:.9g} {z:.9g} {-y:.9g}\n')
        if triangles:obj.data.calc_loop_triangles()
        for face in obj.data.loop_triangles if triangles else obj.data.polygons:
            if len(face.vertices)<3:raise ValueError('重拓扑产生无效多边形')
            stream.write('f '+' '.join(str(v+1) for v in face.vertices)+'\n')
    temporary.replace(path)


def tessellate_retopology(request):
    obj=create_quad_mesh(request['input'])
    obj.data.calc_loop_triangles()
    save_quad_mesh(obj,request['output'],triangles=True)
    return {'backend':'Blender polygon tessellation','polygons':len(obj.data.polygons),
        'triangles':len(obj.data.loop_triangles),'original_quad_obj_preserved':True,
        'blender':bpy.app.version_string}


def activate(obj):
    bpy.ops.object.select_all(action='DESELECT');obj.select_set(True)
    bpy.context.view_layer.objects.active=obj


def prepare_remesh_surface(request):
    obj=create_quad_mesh(request['input']);activate(obj)
    started=time.monotonic();before=len(obj.data.polygons)
    span=max(obj.dimensions)
    if not math.isfinite(span) or span<=1e-10:raise ValueError('模型尺寸无效')
    emit('asset_topology','融合重叠表面并闭合开口；原模型保留用于材质转移',device='cpu')
    obj.data.remesh_voxel_size=span*.0025
    obj.data.remesh_voxel_adaptivity=0
    obj.data.use_remesh_preserve_volume=True
    result=bpy.ops.object.voxel_remesh()
    if 'FINISHED' not in result or not len(obj.data.polygons):raise RuntimeError('表面融合没有生成有效网格')
    if len(obj.data.polygons)>4_000_000:raise RuntimeError('融合副本网格规模过大，请先减面')
    save_quad_mesh(obj,request['output'])
    return {'input_polygons':before,'output_polygons':len(obj.data.polygons),
        'voxel_size':span*.0025,'seconds':round(time.monotonic()-started,3),
        'notice':'融合会闭合开口、连接接近的表面，并可能移除细小结构；原高模保留。'}


def fill_generated_holes(obj,maximum=.075):
    """Only used after the user explicitly requested a solid inference copy."""
    import bmesh
    bm=bmesh.new();bm.from_mesh(obj.data)
    span=max(obj.dimensions);boundary={e for e in bm.edges if e.is_boundary}
    loops=[];added=0
    while boundary:
        edges={boundary.pop()};pending=list(edges)
        while pending:
            for vertex in pending.pop().verts:
                for edge in vertex.link_edges:
                    if edge in boundary:
                        boundary.remove(edge);edges.add(edge);pending.append(edge)
        vertices={v for edge in edges for v in edge.verts}
        # A bounding-box diagonal safely bounds the pairwise diameter without
        # quadratic work on unexpectedly large boundaries.
        bounds=np.asarray([tuple(v.co) for v in vertices])
        diameter=float(np.linalg.norm(np.ptp(bounds,axis=0)))
        closed=all(sum(e in edges for e in v.link_edges)==2 for v in vertices)
        if not closed or diameter>span*maximum:
            loops.append({'edges':len(edges),'filled':False,'diameter':diameter});continue
        faces=bmesh.ops.holes_fill(bm,edges=list(edges),sides=0)['faces']
        if not faces and len(vertices)==3:
            faces=[bm.faces.new(list(vertices))]
        added+=len(faces)
        loops.append({'edges':len(edges),'filled':bool(faces),'diameter':diameter})
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
    report={'added_polygons':added,'loops':loops,
        'boundary_edges':sum(e.is_boundary for e in bm.edges),
        'nonmanifold_edges':sum(len(e.link_faces)>2 for e in bm.edges)}
    bm.to_mesh(obj.data);bm.free();obj.data.update()
    return report


def clean_degenerate_remesh(obj):
    """Collapse only degenerate face neighbourhoods, keeping valid quads."""
    import bmesh
    obj.data.calc_loop_triangles()
    positions=np.asarray([tuple(v.co) for v in obj.data.vertices])
    triangles=np.asarray([tuple(t.vertices) for t in obj.data.loop_triangles])
    xyz=positions[triangles]
    area=np.linalg.norm(np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),axis=1)*.5
    span=max(obj.dimensions);epsilon=span*1e-8
    bad=np.flatnonzero(area<=span*span*1e-14)
    if not len(bad):return {'degenerate_triangles_before':0,'changed':False}
    polygons={obj.data.loop_triangles[int(i)].polygon_index for i in bad}
    bm=bmesh.new();bm.from_mesh(obj.data);bm.faces.ensure_lookup_table()
    before=(sum(e.is_boundary for e in bm.edges),sum(len(e.link_faces)>2 for e in bm.edges))
    vertices={v for index in polygons for v in bm.faces[index].verts}
    edges={e for v in vertices for e in v.link_edges}
    bmesh.ops.remove_doubles(bm,verts=list(vertices),dist=epsilon)
    bmesh.ops.dissolve_degenerate(bm,edges=[e for e in edges if e.is_valid],dist=epsilon)
    after=(sum(e.is_boundary for e in bm.edges),sum(len(e.link_faces)>2 for e in bm.edges))
    accepted=after[0]<=before[0] and after[1]<=before[1] and len(bm.faces)>0
    report={'degenerate_triangles_before':len(bad),'changed':accepted,
        'merge_distance':epsilon,'boundary_edges_before':before[0],'boundary_edges_after':after[0],
        'nonmanifold_edges_before':before[1],'nonmanifold_edges_after':after[1]}
    if accepted:
        bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(obj.data);obj.data.update()
    bm.free()
    return report


def quadriflow_remesh(request):
    obj=create_quad_mesh(request['input']);activate(obj)
    before=len(obj.data.polygons);started=time.monotonic()
    validated=bool(obj.data.validate(verbose=False,clean_customdata=True))
    emit('asset_retopology','QuadriFlow 正在求解四边面；此阶段没有可量化总进度',device='cpu')
    result=bpy.ops.object.quadriflow_remesh(target_faces=int(request['target_quads']),
        seed=int(request.get('seed',0)),use_mesh_symmetry=False,
        use_preserve_sharp=bool(request.get('preserve_sharp',False)),
        use_preserve_boundary=True,preserve_attributes=False,smooth_normals=False)
    if 'FINISHED' not in result or not any(len(p.vertices)==4 for p in obj.data.polygons):
        raise RuntimeError('QuadriFlow 没有生成四边面，请检查非流形连接及法线')
    repair=fill_generated_holes(obj) if request.get('solid_surface',False) else None
    cleanup=clean_degenerate_remesh(obj)
    save_quad_mesh(obj,request['output'])
    counts={'input_polygons':before,'output_polygons':len(obj.data.polygons),
        'quads':sum(len(p.vertices)==4 for p in obj.data.polygons),
        'input_validation_changed_copy':validated,'hole_repair':repair,
        'degenerate_cleanup':cleanup,
        'seconds':round(time.monotonic()-started,3),'blender':bpy.app.version_string}
    emit('asset_retopology','QuadriFlow 几何已生成，开始检查原表面覆盖',device='cpu',
        progress_current=counts['output_polygons'],progress_total=counts['output_polygons'],progress_unit='polygons')
    return counts


def image_pixels(image):
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return pixels.reshape(-1, 4)


def new_image(name, size, *, color=False):
    image = bpy.data.images.new(name, width=size, height=size, alpha=True, float_buffer=False)
    image.colorspace_settings.name = 'sRGB' if color else 'Non-Color'
    return image


def bake_retopology(request):
    sources = load_mesh(request['source'])
    low = create_quad_mesh(request['quads'])
    quads = sum(len(face.vertices) == 4 for face in low.data.polygons)
    polygon_count = len(low.data.polygons)
    bpy.ops.object.select_all(action='DESELECT')
    low.select_set(True)
    bpy.context.view_layer.objects.active = low
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=.008)
    bpy.ops.object.mode_set(mode='OBJECT')
    material = bpy.data.materials.new('Transferred PBR')
    material.use_nodes = True
    low.data.materials.append(material)
    active_image_node = material.node_tree.nodes.new('ShaderNodeTexImage')
    material.node_tree.nodes.active = active_image_node
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 8
    scene.render.bake.use_selected_to_active = True
    scene.render.bake.use_clear = True
    scene.render.bake.margin = 12
    bounds = np.array([low.matrix_world @ vertex.co for vertex in low.data.vertices])
    span = float(np.ptp(bounds, axis=0).max())
    scene.render.bake.cage_extrusion = max(span * .02, 1e-6)
    source_materials = set()
    for obj in sources:
        if not obj.data.materials:
            mat = bpy.data.materials.new('Source neutral')
            mat.diffuse_color = (.65, .65, .65, 1)
            obj.data.materials.append(mat)
        for mat in obj.data.materials:
            if mat:
                mat.use_nodes = True
                source_materials.add(mat)
    bpy.ops.object.select_all(action='DESELECT')
    for obj in sources:
        obj.select_set(True)
    low.select_set(True)
    bpy.context.view_layer.objects.active = low
    size = request.get('texture_size', 2048)
    textures = {}
    passes = [('base_color', 'Base Color', True), ('roughness', 'Roughness', False),
              ('metallic', 'Metallic', False), ('alpha', 'Alpha', False),
              ('emission', 'Emission Color', True), ('normal', None, False)]
    for pass_index, (name, socket_name, is_color) in enumerate(passes):
        emit('asset_material_transfer', f'烘焙 {name}', device='cpu',
             progress_current=pass_index, progress_total=len(passes), progress_unit='textures')
        image = new_image('Transferred ' + name, size, color=is_color)
        active_image_node.image = image
        restored = []
        if socket_name is not None:
            for mat in source_materials:
                nodes, links = mat.node_tree.nodes, mat.node_tree.links
                output = next((node for node in nodes if node.type == 'OUTPUT_MATERIAL' and node.is_active_output), None)
                principled = next((node for node in nodes if node.type == 'BSDF_PRINCIPLED'), None)
                if output is None:
                    output = nodes.new('ShaderNodeOutputMaterial')
                previous = [(link.from_socket, link.to_socket) for link in list(output.inputs['Surface'].links)]
                for link in list(output.inputs['Surface'].links):
                    links.remove(link)
                emission = nodes.new('ShaderNodeEmission')
                socket = principled.inputs.get(socket_name) if principled else None
                if socket is not None and socket.is_linked:
                    links.new(socket.links[0].from_socket, emission.inputs['Color'])
                else:
                    value = socket.default_value if socket is not None else (1 if name == 'alpha' else .5 if name in {'base_color', 'roughness'} else 0)
                    if hasattr(value, '__len__'):
                        color = tuple(value)
                    else:
                        color = (value, value, value, 1)
                    emission.inputs['Color'].default_value = color
                if name == 'emission' and principled is not None:
                    strength = principled.inputs.get('Emission Strength')
                    if strength is not None:
                        if strength.is_linked:
                            links.new(strength.links[0].from_socket, emission.inputs['Strength'])
                        else:
                            emission.inputs['Strength'].default_value = strength.default_value
                links.new(emission.outputs[0], output.inputs['Surface'])
                restored.append((mat, emission, previous))
        try:
            bpy.ops.object.bake(type='NORMAL' if name == 'normal' else 'EMIT')
        finally:
            for mat, emission, previous in restored:
                mat.node_tree.nodes.remove(emission)
                for from_socket, to_socket in previous:
                    mat.node_tree.links.new(from_socket, to_socket)
        textures[name] = image
        emit('asset_material_transfer', f'{name} 已烘焙', device='cpu',
             progress_current=pass_index + 1, progress_total=len(passes), progress_unit='textures')
    base = textures['base_color']
    rgba = image_pixels(base)
    # Empty UV atlas pixels are transparent even for an opaque source. They
    # must not turn an opaque asset into an alpha-blended material; newly
    # reconstructed interior surfaces also remain visible when rays miss.
    opaque = request.get('source_opaque', False)
    rgba[:, 3] = 1 if opaque else image_pixels(textures['alpha'])[:, 0]
    base.pixels.foreach_set(rgba.reshape(-1)); base.update()
    mr = new_image('Metallic Roughness', size)
    mr_pixels = np.ones((size * size, 4), dtype=np.float32)
    mr_pixels[:, 1] = image_pixels(textures['roughness'])[:, 0]
    mr_pixels[:, 2] = image_pixels(textures['metallic'])[:, 0]
    mr.pixels.foreach_set(mr_pixels.reshape(-1)); mr.update()
    material.node_tree.nodes.remove(active_image_node)
    nodes, links = material.node_tree.nodes, material.node_tree.links
    principled = next(node for node in nodes if node.type == 'BSDF_PRINCIPLED')
    color_node = nodes.new('ShaderNodeTexImage'); color_node.image = base
    links.new(color_node.outputs['Color'], principled.inputs['Base Color'])
    if not opaque:
        links.new(color_node.outputs['Alpha'], principled.inputs['Alpha'])
    mr_node = nodes.new('ShaderNodeTexImage'); mr_node.image = mr
    separate = nodes.new('ShaderNodeSeparateColor')
    links.new(mr_node.outputs['Color'], separate.inputs[0])
    links.new(separate.outputs['Green'], principled.inputs['Roughness'])
    links.new(separate.outputs['Blue'], principled.inputs['Metallic'])
    normal_node = nodes.new('ShaderNodeTexImage'); normal_node.image = textures['normal']
    normal = nodes.new('ShaderNodeNormalMap')
    links.new(normal_node.outputs['Color'], normal.inputs['Color'])
    links.new(normal.outputs['Normal'], principled.inputs['Normal'])
    emission_node = nodes.new('ShaderNodeTexImage'); emission_node.image = textures['emission']
    links.new(emission_node.outputs['Color'], principled.inputs['Emission Color'])
    principled.inputs['Emission Strength'].default_value = 1
    for image in (base, mr, textures['normal'], textures['emission']):
        image.file_format = 'PNG'
        image.pack()
    if not opaque and float(rgba[:, 3].min()) < .99:
        material.surface_render_method = 'DITHERED'
    bpy.ops.object.select_all(action='DESELECT')
    low.select_set(True)
    bpy.context.view_layer.objects.active = low
    bpy.ops.export_scene.gltf(filepath=request['output'], export_format='GLB', use_selection=True,
                              export_yup=True, export_normals=True, export_texcoords=True,
                              export_materials='EXPORT', export_animations=False)
    shutil.copy2(request['quads'], request['obj_output'])
    return {'quad_faces': quads, 'polygons': polygon_count, 'texture_size': size,
            'material_transfer': 'Cycles selected-to-active baking', 'channels': [row[0] for row in passes],
            'cage_extrusion': scene.render.bake.cage_extrusion,
            'warnings': ['重拓扑改变布线与 UV，材质已经重新烘焙；细小重叠表面请检查投射结果。']}


def activate_action(obj, action, slot=None):
    data = obj.animation_data_create()
    for track in data.nla_tracks:
        track.mute = True
    data.action = action
    if action is not None and hasattr(action, 'slots') and len(action.slots):
        candidates = [item for item in action.slots if item.target_id_type == obj.id_type]
        if slot is not None:
            data.action_slot = slot
        elif candidates:
            data.action_slot = candidates[0]


def animation_snapshot(scene, armatures, meshes, frame):
    """Joint heads and evaluated surface bounds, independent of FBX bone axes."""
    scene.frame_set(int(math.floor(frame)), subframe=float(frame - math.floor(frame)))
    bpy.context.view_layer.update()
    heads = {bone.name: list(obj.matrix_world @ bone.head)
             for obj in armatures for bone in obj.pose.bones}
    bounds = []
    graph = bpy.context.evaluated_depsgraph_get()
    for obj in meshes:
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        try:
            if not len(mesh.vertices):
                continue
            vertices = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
            mesh.vertices.foreach_get('co', vertices)
            matrix = np.asarray(evaluated.matrix_world)
            vertices = vertices.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]
            bounds.append(np.array([vertices.min(axis=0), vertices.max(axis=0)]))
        finally:
            evaluated.to_mesh_clear()
    if not bounds:
        raise RuntimeError('动画导出验证缺少有效表面')
    return {'heads': heads, 'bounds': np.array([np.min(np.array(bounds)[:, 0], axis=0),
                                                np.max(np.array(bounds)[:, 1], axis=0)])}


def export_fbx(request):
    scene = bpy.context.scene
    scene.render.fps = 30
    bpy.ops.import_scene.gltf(filepath=request['source'])
    armatures = [obj for obj in scene.objects if obj.type == 'ARMATURE']
    # The glTF importer creates an 80-triangle custom bone display mesh. It is
    # viewport scaffolding, not user geometry, and must never enter the FBX.
    shapes = {bone.custom_shape for obj in armatures for bone in obj.pose.bones if bone.custom_shape}
    meshes = [obj for obj in scene.objects if obj.type == 'MESH' and obj not in shapes]
    if not armatures or not meshes:
        raise ValueError('GLB 缺少角色骨架或网格，不能导出动画 FBX')
    texture_folder = Path(request['output']).parent / 'fbx-textures'
    texture_folder.mkdir(exist_ok=True)
    for index, image in enumerate(list(bpy.data.images)):
        if image.source == 'FILE' and image.packed_file:
            image.filepath_raw = str(texture_folder / f'texture_{index:04d}.png')
            image.file_format = 'PNG'
            image.save()
    expected_bones = sum(len(obj.data.bones) for obj in armatures)
    expected_faces = sum(sum(len(face.vertices) - 2 for face in obj.data.polygons) for obj in meshes)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from hi3d_workbench.studio.gltf import Document
    document = Document.load(request['source'])
    tree = document.tree
    source_faces = sum(tree['accessors'][primitive.get('indices', primitive['attributes']['POSITION'])]['count'] // 3
        for node in tree.get('nodes', []) if 'mesh' in node
        for primitive in tree['meshes'][node['mesh']]['primitives'])
    if expected_faces != source_faces:
        raise RuntimeError('Blender 导入后的用户网格与 GLB 三角面数量不一致')
    clips = []
    for index, animation in enumerate(tree.get('animations', [])):
        duration = max((float(document.accessor(sampler['input'])[-1])
                        for sampler in animation.get('samplers', [])), default=0.)
        clips.append({'index': index, 'duration': duration})
    # Blender's FBX baker truncates fractional action ranges. glTF time starts
    # at zero, even when the first key is later: retain that held first pose.
    # NLA track names are the glTF importer's canonical animation identities.
    expected_clips = []
    from io_scene_gltf2.blender.imp.blender_gltf import BlenderGlTF
    track_names = set()
    for clip in clips:
        desired_name = tree['animations'][clip['index']].get('name') or f"Anim_{clip['index']}"
        animation_name = BlenderGlTF.find_unused_name(track_names, desired_name)
        track_names.add(animation_name)
        if clip['duration'] <= 0:
            continue
        assignments = []
        for obj in [*armatures, *meshes]:
            if obj.animation_data:
                for track in obj.animation_data.nla_tracks:
                    if track.name == animation_name:
                        for strip in track.strips:
                            if strip.action:
                                assignments.append((obj, strip.action, getattr(strip, 'action_slot', None)))
        if not assignments:
            raise RuntimeError(f'Blender 未找到 GLB 动作片段：{animation_name}')
        # FBX import prefixes action names with the object name; Blender then
        # truncates the combined ID to 63 UTF-8 bytes. Long prompt names can
        # otherwise lose their identity before the pose validation runs.
        # These are temporary export-scene IDs; the GLB retains full labels.
        unique_actions={action.as_pointer():action for _,action,_ in assignments}
        for action_index,action in enumerate(unique_actions.values()):
            action.name=f'T8Clip_{clip["index"]:04d}_{action_index:04d}'
        end_frame = math.ceil(clip['duration'] * scene.render.fps - 1e-5)
        for obj in [*armatures, *meshes]:
            if obj.animation_data:
                activate_action(obj, None)
        for obj, action, slot in assignments:
            action.use_frame_range = True
            action.frame_start = 0
            action.frame_end = end_frame
            activate_action(obj, action, slot)
        times = [0., clip['duration'] / 2, clip['duration']]
        snapshots = [animation_snapshot(scene, armatures, meshes, time * scene.render.fps) for time in times]
        expected_clips.append({'name': animation_name, 'source_name':desired_name, 'duration': clip['duration'], 'times': times,
                               'snapshots': snapshots, 'actions': list({action.name for _, action, _ in assignments})})
    emit('asset_export', '导出骨架、蒙皮与动画 FBX', device='cpu')
    for obj in scene.objects:
        obj.select_set(False)
    for obj in [*armatures, *meshes]:
        obj.select_set(True)
    bpy.ops.export_scene.fbx(filepath=request['output'], use_selection=True,
        object_types={'ARMATURE', 'MESH'}, add_leaf_bones=False, use_armature_deform_only=True,
        bake_anim=True, bake_anim_use_all_actions=True, bake_anim_use_nla_strips=False,
        bake_anim_simplify_factor=0, path_mode='COPY', embed_textures=True,
        axis_forward='-Z', axis_up='Y', apply_unit_scale=True)
    if Path(request['output']).stat().st_size < 1024:
        raise RuntimeError('FBX 输出为空')
    # Operator selection excludes hidden custom shapes; removing only selected
    # objects leaves them in the scene and would invalidate a roundtrip check.
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action, do_unlink=True)
    bpy.ops.import_scene.fbx(filepath=request['output'], use_anim=True, use_image_search=False)
    actual_armatures = [obj for obj in scene.objects if obj.type == 'ARMATURE']
    actual_meshes = [obj for obj in scene.objects if obj.type == 'MESH']
    actual_bones = sum(len(obj.data.bones) for obj in actual_armatures)
    actual_faces = sum(sum(len(face.vertices) - 2 for face in obj.data.polygons) for obj in actual_meshes)
    if actual_bones != expected_bones or actual_faces != expected_faces:
        raise RuntimeError(f'FBX 重导入后骨骼或三角面数量不一致：'
            f'骨骼 {expected_bones} → {actual_bones}；三角面 {expected_faces} → {actual_faces}；'
            f'网格 {[obj.name for obj in actual_meshes]}')
    verified_clips = []
    for clip in expected_clips:
        imported = [action for action in bpy.data.actions if any(action.name == name or action.name.endswith('|' + name)
                    for name in clip['actions'])]
        if not imported:
            raise RuntimeError(f"FBX 重导入后缺少动作：{clip['name']}")
        starts = [float(action.frame_range[0]) for action in imported]
        duration = max(float(action.frame_range[1] - action.frame_range[0]) / scene.render.fps for action in imported)
        if abs(duration - clip['duration']) > 1 / scene.render.fps + 1e-5:
            raise RuntimeError(f"FBX 动作时长发生变化：{clip['name']} {clip['duration']:.4f} → {duration:.4f} 秒")
        for obj in actual_armatures:
            action = next((action for action in imported if action.name.startswith(obj.name + '|')), imported[0])
            activate_action(obj, action)
        max_head_error = max_surface_error = 0.
        for time, expected in zip(clip['times'], clip['snapshots']):
            actual = animation_snapshot(scene, actual_armatures, actual_meshes, min(starts) + time * scene.render.fps)
            if set(actual['heads']) != set(expected['heads']):
                raise RuntimeError('FBX 重导入后骨骼名称发生变化')
            head_error = max(np.linalg.norm(np.asarray(actual['heads'][name]) - value)
                             for name, value in expected['heads'].items())
            surface_error = float(np.max(np.abs(actual['bounds'] - expected['bounds'])))
            span = float(np.ptp(expected['bounds'], axis=0).max())
            tolerance = max(span * 2e-4, 2e-5)
            if head_error > tolerance or surface_error > tolerance:
                raise RuntimeError(f"FBX 动作姿态校验失败：{clip['name']}，关节误差 {head_error:.6g}，表面误差 {surface_error:.6g}")
            max_head_error = max(max_head_error, float(head_error))
            max_surface_error = max(max_surface_error, surface_error)
        verified_clips.append({'name': clip['name'], 'duration': duration, 'pose_samples': len(clip['times']),
                               'source_name':clip['source_name'],'fbx_action_names':clip['actions'],
                               'max_joint_error': max_head_error, 'max_surface_bounds_error': max_surface_error})
    return {'fbx_roundtrip': True, 'fbx_armatures': len(actual_armatures), 'fbx_bones': actual_bones,
            'fbx_faces': actual_faces, 'fbx_source_faces': source_faces, 'fbx_actions': len(bpy.data.actions),
            'fbx_animation_checks': verified_clips,
            'notice': 'PBR 材质以 GLB 为准，FBX 的材质兼容性取决于目标软件。'}


def import_motion(request):
    scene=bpy.context.scene
    bpy.ops.import_scene.fbx(filepath=request['source'],use_anim=True,use_image_search=False,
                             automatic_bone_orientation=False)
    armatures=[obj for obj in scene.objects if obj.type=='ARMATURE']
    if not armatures:
        raise ValueError('FBX 没有骨架，不能用作角色动作')
    selected=request.get('armature')
    arm=next((obj for obj in armatures if obj.name==selected),None) if selected else armatures[0]
    if arm is None:
        raise ValueError('所选 FBX 骨架不存在')
    bones=list(arm.data.bones)
    if not 1<=len(bones)<=512:
        raise ValueError('FBX 骨骼数量需在 1–512 之间')
    names=[bone.name for bone in bones]
    parents=[names.index(bone.parent.name) if bone.parent else -1 for bone in bones]
    # Blender has already interpreted FBX axes and units. This final matrix
    # converts Blender Z-up world space to the workbench's glTF Y-up space.
    conversion=np.array([[1.,0,0,0],[0,0,1.,0],[0,-1.,0,0],[0,0,0,1.]])
    rest=np.array([conversion @ np.array(arm.matrix_world) @ np.array(bone.matrix_local) for bone in bones])
    actions=[]
    if arm.animation_data:
        if arm.animation_data.action:
            actions.append(arm.animation_data.action)
        for track in arm.animation_data.nla_tracks:
            for strip in track.strips:
                if strip.action and strip.action not in actions:
                    actions.append(strip.action)
    # Blender's FBX importer activates only the first take and leaves the rest
    # as fake-user actions, without NLA strips. Include every take belonging
    # to this armature instead of silently hiding all but the first clip.
    for action in bpy.data.actions:
        if action in actions:
            continue
        paths=[curve.data_path for curve in action.fcurves]
        bone_paths=[path for path in paths if path.startswith('pose.bones[')]
        compatible=bool(bone_paths) and all(any(path.startswith('pose.bones['+json.dumps(name)+']')
                                              for name in names) for path in bone_paths)
        if action.name.startswith(arm.name+'|') or ('|' not in action.name and compatible):
            actions.append(action)
    if not actions:
        actions=list(bpy.data.actions)
    fps=float(scene.render.fps)/scene.render.fps_base
    clips=[{'index':index,'name':action.name,'duration':float(action.frame_range[1]-action.frame_range[0])/fps}
           for index,action in enumerate(actions) if action.frame_range[1]>action.frame_range[0]]
    actions=[action for action in actions if action.frame_range[1]>action.frame_range[0]]
    if not actions:
        raise ValueError('FBX 没有可用的骨骼动画片段')
    report={'rig':{'names':names,'parents':parents,'rest_worlds':rest.tolist()},'clips':clips,
        'armature':arm.name,'armatures':[obj.name for obj in armatures],'source_fps':fps,'axis':'Y','units':'metres'}
    if request.get('inspect_only',False):
        return report
    clip=request.get('clip',0)
    if type(clip) is not int or not 0<=clip<len(actions):
        raise ValueError('FBX 动作片段编号无效')
    action=actions[clip]
    activate_action(arm,action)
    start,end=map(float,action.frame_range)
    duration=(end-start)/fps
    target_fps=request.get('fps',30)
    frames=max(2,int(np.ceil(duration*target_fps))+1)
    if duration>600 or frames*len(bones)>2_000_000:
        raise ValueError('FBX 动作过长或骨架规模过大，请先裁剪片段')
    times=np.linspace(0,duration,frames)
    poses=np.empty((frames,len(bones),4,4),dtype=np.float64)
    for frame,time in enumerate(times):
        value=start+time*fps
        scene.frame_set(int(np.floor(value)),subframe=float(value-np.floor(value)))
        bpy.context.view_layer.update()
        for index,name in enumerate(names):
            poses[frame,index]=conversion @ np.array(arm.matrix_world) @ np.array(arm.pose.bones[name].matrix)
        if frame%30==0 or frame==frames-1:
            emit('asset_motion_import','读取 FBX 动作帧',device='cpu',progress_current=frame+1,
                 progress_total=frames,progress_unit='frames')
    output=Path(request['output']);temporary=output.with_suffix('.tmp.npz')
    np.savez_compressed(temporary,times=times,worlds=poses,rest_worlds=rest,parents=np.array(parents),
                        names=np.array(names),name=np.asarray(action.name))
    temporary.replace(output)
    report.update(frames=frames,selected_clip=clip,duration=duration)
    return report


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    request = json.loads(Path(args.request).read_text(encoding='utf-8'))
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    if request['operation'] == 'bake_retopology':
        report = bake_retopology(request)
    elif request['operation'] == 'prepare_remesh_surface':
        report = prepare_remesh_surface(request)
    elif request['operation'] == 'quadriflow_remesh':
        report = quadriflow_remesh(request)
    elif request['operation'] == 'tessellate_retopology':
        report = tessellate_retopology(request)
    elif request['operation'] == 'export_fbx':
        report = export_fbx(request)
    elif request['operation'] == 'import_motion':
        report = import_motion(request)
    else:
        raise ValueError('无效 Blender 资产操作')
    Path(request['report']).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
