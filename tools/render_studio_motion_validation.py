"""Blender QA renders with fixed projection and a stationary world grid.

Optional horizontal camera following makes long root-motion clips inspectable.
It never reframes each pose, changes vertical motion, or moves the floor.
"""
import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector,Matrix


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--times',default='0,.5,1,1.5,2,2.5,3,3.5,4')
    parser.add_argument('--ground',type=float,default=0)
    parser.add_argument('--follow-root',action='store_true')
    parser.add_argument('--camera-from',help='Reuse fixed framing from an earlier validation render.json')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=False)
    for obj in list(bpy.data.objects):bpy.data.objects.remove(obj,do_unlink=True)
    scene=bpy.context.scene;scene.render.fps=30
    bpy.ops.import_scene.gltf(filepath=args.model)
    arms=[obj for obj in scene.objects if obj.type=='ARMATURE']
    if len(arms)!=1:raise ValueError('Expected one character armature')
    arm=arms[0];roots=[bone for bone in arm.pose.bones if bone.parent is None]
    if len(roots)!=1:raise ValueError('Expected one root bone')
    shapes={bone.custom_shape for bone in arm.pose.bones if bone.custom_shape}
    meshes=[obj for obj in scene.objects if obj.type=='MESH' and obj not in shapes]

    def root_position():return np.array(arm.matrix_world @ roots[0].head)
    def bounds():
        graph=bpy.context.evaluated_depsgraph_get();points=[]
        for obj in meshes:
            evaluated=obj.evaluated_get(graph);mesh=evaluated.to_mesh()
            try:
                xyz=np.empty(len(mesh.vertices)*3);mesh.vertices.foreach_get('co',xyz)
                matrix=np.array(evaluated.matrix_world)
                points.append(xyz.reshape(-1,3) @ matrix[:3,:3].T+matrix[:3,3])
            finally:evaluated.to_mesh_clear()
        xyz=np.concatenate(points);return np.array([xyz.min(axis=0),xyz.max(axis=0)])

    times=[float(value) for value in args.times.split(',')]
    scene.frame_set(0);origin=root_position()
    offsets=[];boxes=[]
    for time in times:
        scene.frame_set(round(time*30))
        offset=root_position()-origin;offset[2]=0
        offsets.append(offset if args.follow_root else np.zeros(3));boxes.append(bounds())
    relative=np.asarray(boxes)-np.asarray(offsets)[:,None,:]
    total=np.array([relative[:,0].min(axis=0),relative[:,1].max(axis=0)])
    center=Vector(total.mean(axis=0));span=float(np.ptp(total,axis=0).max())
    reference=None
    if args.camera_from:
        reference=json.loads(Path(args.camera_from).read_text(encoding='utf-8'))
        if args.follow_root or not reference.get('fixed_camera'):
            raise ValueError('Comparison framing requires a fixed-camera reference')
        if abs(float(reference['ground'])-args.ground)>1e-9:raise ValueError('Comparison ground must match')
        boxes_reference=np.asarray([row['bounds'] for row in reference['records']],dtype=float)
        framing=np.array([boxes_reference[:,0].min(axis=0),boxes_reference[:,1].max(axis=0)])
        center=Vector(reference.get('framing_center',framing.mean(axis=0)))
        span=float(reference.get('framing_span',np.ptp(framing,axis=0).max()))
    if not span>0:raise ValueError('Invalid animated geometry bounds')
    bpy.ops.mesh.primitive_plane_add(size=max(span*30,float(np.ptp(np.array(offsets),axis=0).max())*3),
        location=(center.x,center.y,args.ground))
    floor=bpy.context.object;mat=bpy.data.materials.new('stationary world grid');mat.use_nodes=True
    nodes=mat.node_tree.nodes;checker=nodes.new('ShaderNodeTexChecker');geometry=nodes.new('ShaderNodeNewGeometry')
    checker.inputs['Color1'].default_value=(.13,.15,.18,1);checker.inputs['Color2'].default_value=(.2,.22,.25,1)
    checker.inputs['Scale'].default_value=4/span
    mat.node_tree.links.new(geometry.outputs['Position'],checker.inputs['Vector'])
    bsdf=nodes.get('Principled BSDF');bsdf.inputs['Roughness'].default_value=.9
    mat.node_tree.links.new(checker.outputs['Color'],bsdf.inputs['Base Color']);floor.data.materials.append(mat)
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=12
    scene.render.resolution_x=576;scene.render.resolution_y=640;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.world.color=(.2,.2,.2)
    scene.view_settings.view_transform='AgX'
    bpy.ops.object.camera_add(location=center+Vector((.85,-1.8,.6))*span)
    camera=bpy.context.object;camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler()
    camera.data.type='ORTHO';camera.data.ortho_scale=span*1.35;scene.camera=camera
    if reference:
        camera.matrix_world=Matrix(reference['records'][0]['camera_matrix'])
        camera.data.ortho_scale=float(reference.get('camera_ortho_scale',span*1.35))
    base_camera=camera.location.copy();lights=[]
    for pos,energy,size in [((1,-1,2),600,3),((-1,-.5,1),350,2),((0,1,2),700,2)]:
        bpy.ops.object.light_add(type='AREA',location=center+Vector(pos)*span)
        light=bpy.context.object;light.data.energy=energy*span**2;light.data.size=size*span
        light.rotation_euler=(center-light.location).to_track_quat('-Z','Y').to_euler()
        lights.append((light,light.location.copy()))
    records=[]
    for index,(time,offset,box) in enumerate(zip(times,offsets,boxes)):
        scene.frame_set(round(time*30));camera.location=base_camera+Vector(offset)
        for light,base in lights:light.location=base+Vector(offset)
        scene.render.filepath=str(folder/f'frame-{index}.png');bpy.ops.render.render(write_still=True)
        records.append({'time':time,'bounds':box.tolist(),'ground_clearance':float(box[0,2]-args.ground),
            'camera_matrix':[list(row) for row in camera.matrix_world],'horizontal_camera_offset':offset.tolist()})
    (folder/'render.json').write_text(json.dumps({'model':args.model,'follow_horizontal_root':args.follow_root,
        'framing_center':list(center),'framing_span':span,'camera_ortho_scale':float(camera.data.ortho_scale),
        'camera_reference':args.camera_from,
        'fixed_projection':True,'fixed_camera':not args.follow_root,'ground':args.ground,
        'fixed_world_grid':True,'records':records},indent=2),encoding='utf-8')


if __name__=='__main__':main()
