"""Blender CPU renders of the actual exported GLB, preserving its PBR materials.

Run: blender -b -t 8 --python-exit-code 1 --python tools/render_asset_validation.py -- --model ... --output ...
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector

args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
parser=argparse.ArgumentParser()
parser.add_argument('--model',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--environment',type=Path,help='Optional HDR environment for a relighting comparison')
parser.add_argument('--geometry',action='store_true',help='Neutral grey material, for geometry inspection only')
parser.add_argument('--views',default='front,quarter,right,back,left,top')
parser.add_argument('--exposure',type=float,default=0)
options=parser.parse_args(args)
if not -4<=options.exposure<=4:parser.error('exposure must be between -4 and 4')
if not options.model.is_file():parser.error('model file is not ready or does not exist')
if options.environment and not options.environment.is_file():parser.error('environment file does not exist')
output=options.output.resolve()
if output.exists():raise RuntimeError('Use a new visual validation directory')
output.mkdir(parents=True)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.gltf(filepath=str(options.model.resolve()))
objects=[obj for obj in bpy.context.scene.objects if obj.type=='MESH']
if not objects:raise RuntimeError('The delivered GLB has no renderable geometry')
if options.geometry:
    material=bpy.data.materials.new('QA neutral geometry');material.use_nodes=True
    surface=material.node_tree.nodes.get('Principled BSDF')
    surface.inputs['Base Color'].default_value=(.22,.22,.22,1)
    surface.inputs['Roughness'].default_value=.75
    for obj in objects:
        obj.data.materials.clear();obj.data.materials.append(material)
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
scene.view_settings.exposure=options.exposure
scene.world=bpy.data.worlds.new('QA neutral world')
scene.world.use_nodes=True
scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(.72,.72,.72,1)
scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.35
if options.environment:
    environment=scene.world.node_tree.nodes.new('ShaderNodeTexEnvironment')
    environment.image=bpy.data.images.load(str(options.environment.resolve()))
    scene.world.node_tree.links.new(environment.outputs['Color'],scene.world.node_tree.nodes['Background'].inputs['Color'])
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=1
def aim(obj,point):obj.rotation_euler=(point-obj.location).to_track_quat('-Z','Y').to_euler()
for name,position,power,size in [('key',(-2,-3,4),650,4),('fill',(3,-1,2),400,3),('rim',(0,3,3),750,3)]:
    if options.environment:continue
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
requested=set(options.views.split(','))
if not requested or requested-{name for name,_,_ in views}:raise ValueError('Unknown render view')
for name,azimuth,elevation in views:
    if name not in requested:continue
    a,e=math.radians(azimuth),math.radians(elevation)
    camera.location=center+Vector((math.sin(a)*math.cos(e),-math.cos(a)*math.cos(e),math.sin(e)))*extent*3
    aim(camera,center)
    scene.render.filepath=str(output/(name+'.png'))
    bpy.ops.render.render(write_still=True)
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
manifest={'model':str(options.model.resolve()),'model_sha256':sha(options.model),'geometry_only':options.geometry,
          'environment':str(options.environment.resolve()) if options.environment else 'neutral world + three area lights',
          'environment_sha256':sha(options.environment) if options.environment else None,
          'engine':scene.render.engine,'samples':scene.cycles.samples,'blender':bpy.app.version_string,
          'view_transform':scene.view_settings.view_transform,'look':scene.view_settings.look,'exposure':options.exposure,
          'images':{name:sha(output/(name+'.png')) for name,_,_ in views if name in requested}}
(output/'render-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Actual GLB PBR renders saved:',output)
