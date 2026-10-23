"""Procedural in-house smoke sample; no third-party sample media."""
import sys
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image,ImageDraw

out=Path(sys.argv[1] if len(sys.argv)>1 else 'samples')
out.mkdir(parents=True,exist_ok=True)
mesh=trimesh.creation.icosphere(subdivisions=3)
v=mesh.vertices.copy()
angle=np.arctan2(v[:,1],v[:,0]);z=v[:,2]
radius=1+.12*np.cos(angle*8)*(1-z*z)+.04*np.cos(z*18)
v*=radius[:,None];v[:,2]*=1.2
mesh=trimesh.Trimesh(v,mesh.faces,process=False);mesh.export(out/'sculpted-vase.glb')
# Camera front at +Y with Z up. Orthographic shaded reference from same mesh.
rot=trimesh.transformations.rotation_matrix(np.deg2rad(18),[0,0,1])[:3,:3]
v=mesh.vertices@rot.T
im=Image.new('RGBA',(768,768),(0,0,0,0));draw=ImageDraw.Draw(im)
light=np.array([-.4,.7,.8]);light/=np.linalg.norm(light)
for index in np.argsort(v[mesh.faces].mean(axis=1)[:,1]):
    tri=v[mesh.faces[index]];normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);normal/=max(np.linalg.norm(normal),1e-12)
    shade=.3+.7*max(0,float(normal@light));color=tuple(int(c*shade) for c in [150,205,176])+(255,)
    draw.polygon([(float(384+x*250),float(384-z*250)) for x,y,z in tri],fill=color)
im.save(out/'sculpted-vase.png')
print(out.resolve())
