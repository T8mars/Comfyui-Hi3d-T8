"""Owned CPU/GPU UV and CPU material stages, with verified layout reuse."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import time

from .. import common
from ..common import atomic_json,digest,read_json
from .gltf import Document
from .uv_atlas import create,select_backend,validate_cached
from .material_projection import transfer


def run(source,geometry,output,directory,*,texture_size=2048,obj_output=None,projection='nearest',
        triangulated_geometry=None,uv_backend='auto',owner=None,emit=lambda *a,**k:None):
    source,geometry,output,directory=map(Path,(source,geometry,output,directory))
    directory.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();uv_geometry=Path(triangulated_geometry) if triangulated_geometry is not None else geometry
    fingerprint=digest(uv_geometry)
    identity={'schema':1,'geometry_sha256':fingerprint,'xatlas':'0.0.11','texture_size':texture_size,
              'chart_iterations':4,'padding':4,'smooth_normals':True}
    backend=select_backend(uv_geometry,uv_backend)
    if backend=='gpu':
        identity.update(schema=2,backend='gpu',adapter_sha256=digest(Path(__file__).parents[1]/'engines/community_uv.py'),
            uv_adapter_sha256=digest(Path(__file__).with_name('uv_atlas.py')),
            worker_sha256=digest(Path(__file__).with_name('gpu_uv.py')))
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    cache=common.DATA/'studio-cache'/'uv'/key;cache.mkdir(parents=True,exist_ok=True)
    from filelock import FileLock
    with FileLock(str(cache/'.lock'),timeout=30):
        try:saved=read_json(cache/'manifest.json',{})
        except (ValueError,OSError):saved={}
        cached=cache/'atlas.glb';valid=False
        if isinstance(saved,dict) and saved.get('identity')==identity and cached.is_file():
            valid=saved.get('sha256')==digest(cached)
        if valid:
            try:
                validate_cached(cached)
                valid = isinstance(saved.get('uv'), dict)
            except (ValueError, KeyError, TypeError, OSError, RuntimeError):
                valid = False
        if valid:
            uv=saved['uv'];emit('asset_uv','复用已校验的 UV 图集',device='cpu')
        else:
            uv=create(uv_geometry,cached,size=texture_size,backend=backend,owner=owner or directory,emit=emit)
            atomic_json(cache/'manifest.json',{'identity':identity,'uv':uv,'sha256':digest(cached)})
        # A private stage copy remains inspectable if material transfer fails.
        atlas=directory/'atlas.glb';shutil.copy2(cached,atlas)
    material=transfer(source,atlas,output,size=texture_size,projection=projection,emit=emit)
    if obj_output is not None:
        obj_output=Path(obj_output);obj_output.parent.mkdir(parents=True,exist_ok=True)
        if obj_output.resolve()!=geometry.resolve():shutil.copy2(geometry,obj_output)
    quads=polygons=0
    with geometry.open(encoding='utf-8') as stream:
        for line in stream:
            if line.startswith('f '):
                polygons+=1;quads+=len(line.split())==5
    report={'quad_faces':quads,'polygons':polygons,'material_transfer':material['material_transfer'],'uv_backend':backend,
        'texture_size':texture_size,'channels':['base_color','roughness','metallic','alpha','emission','normal','occlusion'],
        'uv':{**uv,'cache_hit':valid},'projection':material,'seconds':round(time.monotonic()-started,3),
        'warnings':[material['notice']]}
    atomic_json(directory/'report.json',report)
    return report
