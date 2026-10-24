"""Blender CPU renders of the actual exported GLB, preserving its PBR materials.

Run: blender -b -t 8 --python tools/render_asset_validation.py -- --model ... --output ...
"""
import argparse
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector

args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
parser=argparse.ArgumentParser()
parser.add_argument('--model',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
options=parser.parse_args(args)
output=options.output.resolve()
if output.exists():raise RuntimeError('Use a new visual validation directory')
output.mkdir(parents=True)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.gltf(filepath=str(options.model.resolve()))
objects=[obj for obj in bpy.context.scene.objects if obj.type=='MESH']
if not objects:raise RuntimeError('The delivered GLB has no renderable geometry')
corners=[obj.matrix_world@Vector(point) for obj in objects for point in obj.bound_box]
minimum=Vector([min(point[i] for point in corners) for i in range(3)])
maximum=Vector([max(point[i] for point in corners) for i in range(3)])
center=(minimum+maximum)/2
extent=max(maximum-minimum)
if not math.isfinite(extent) or extent<=0:raise ValueError('Invalid asset bounds')
scene=bpy.context.scene
scene.render.engine='CYCLES'
scene.cycles.device='CPU'
scene.cycles.samples=24
scene.cycles.use_denoising=True
scene.render.resolution_x=768
scene.render.resolution_y=768
scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG'
scene.render.film_transparent=False
scene.world=bpy.data.worlds.new('QA neutral world')
scene.world.use_nodes=True
scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(.72,.72,.72,1)
scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.35
def aim(obj,point):obj.rotation_euler=(point-obj.location).to_track_quat('-Z','Y').to_euler()
for name,position,power,size in [('key',(-2,-3,4),650,4),('fill',(3,-1,2),400,3),('rim',(0,3,3),750,3)]:
    data=bpy.data.lights.new('QA '+name,'AREA')
    data.energy=power*extent**2
    data.shape='DISK';data.size=size*extent
    obj=bpy.data.objects.new(data.name,data);scene.collection.objects.link(obj)
    obj.location=center+Vector(position)*extent;aim(obj,center)
data=bpy.data.cameras.new('QA camera')
camera=bpy.data.objects.new(data.name,data);scene.collection.objects.link(camera)
scene.camera=camera
data.type='ORTHO';data.ortho_scale=extent*1.35
views=[('front',0,12),('quarter',35,18),('right',90,12),('back',180,12),('left',270,12),('top',35,75)]
for name,azimuth,elevation in views:
    a,e=math.radians(azimuth),math.radians(elevation)
    camera.location=center+Vector((math.sin(a)*math.cos(e),-math.cos(a)*math.cos(e),math.sin(e)))*extent*3
    aim(camera,center)
    scene.render.filepath=str(output/(name+'.png'))
    bpy.ops.render.render(write_still=True)
print('Actual GLB PBR renders saved:',output)
