"""Validated CPU/GPU UV layout, retaining every oriented source triangle."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .gltf import Document
from .material_projection import normalize
from ..common import digest


class UVLayoutError(RuntimeError):
    """A layout failed acceptance; source geometry validation remains separate."""


def select_backend(source,backend='auto'):
    if backend not in {'auto','cpu','gpu'}:raise ValueError('UV 展开方式无效')
    if backend=='cpu':return 'cpu'
    if backend=='auto':
        with Path(source).open('rb') as stream:
            if sum(line.startswith(b'f ') for line in stream)<50000:return 'cpu'
    from .processes import pipeline_command
    try:pipeline_command()
    except RuntimeError:
        if backend=='gpu':raise
        return 'cpu'
    import torch
    if torch.cuda.is_available():return 'gpu'
    if backend=='gpu':raise RuntimeError('GPU UV 需要可用 CUDA 设备；可选择 CPU UV')
    return 'cpu'


def layout_metrics(faces,uv,size):
    from ..engines.community_uv import _zero_uv_faces,_duplicate_uv_faces
    uv=np.asarray(uv,dtype=np.float64)
    if uv.ndim!=2 or uv.shape[1]!=2 or not np.isfinite(uv).all() or np.any(uv<0) or np.any(uv>1):
        raise RuntimeError('UV 图集坐标无效或越界')
    zero=len(_zero_uv_faces(faces,uv));duplicates=len(_duplicate_uv_faces(faces,uv))
    triangle=uv[faces];a,b=triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]
    area=np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])/2
    return {'zero_uv_faces':zero,'duplicate_uv_faces':duplicates,'uv_area':float(area.sum()),
            'subpixel_triangles':int((area*size*size<.5).sum())}


def validate_cached(path):
    document=Document.load(path)
    found=False
    for mesh in document.tree.get('meshes',[]):
        for primitive in mesh.get('primitives',[]):
            if primitive.get('mode',4)!=4:raise RuntimeError('缓存 UV 不是三角网格')
            faces=document.accessor(primitive['indices']).reshape(-1,3)
            if not len(faces):raise RuntimeError('缓存 UV 网格为空')
            uv=document.accessor(primitive['attributes']['TEXCOORD_0'])
            metrics=layout_metrics(faces,uv,1024)
            if metrics['zero_uv_faces'] or metrics['duplicate_uv_faces']:
                raise RuntimeError('缓存 UV 含退化或复用三角面')
            found=True
    if not found:raise RuntimeError('缓存 UV 缺少网格')
    return document


def oriented_keys(faces):
    first=faces.argmin(axis=1)
    return np.take_along_axis(faces,(first[:,None]+np.arange(3))%3,axis=1)


def tangent_vectors(positions,faces,uv,normals):
    points=positions[faces];tex=uv[faces]
    ea,eb=points[:,1]-points[:,0],points[:,2]-points[:,0]
    ua,ub=tex[:,1]-tex[:,0],tex[:,2]-tex[:,0]
    determinant=ua[:,0]*ub[:,1]-ua[:,1]*ub[:,0]
    safe=np.where(np.abs(determinant)>1e-14,determinant,1.)
    tangent=(ea*ub[:,1,None]-eb*ua[:,1,None])/safe[:,None]
    bitangent=(-ea*ub[:,0,None]+eb*ua[:,0,None])/safe[:,None]
    weight=np.linalg.norm(np.cross(ea,eb),axis=1)[:,None]
    t=np.zeros_like(positions,dtype=np.float64);b=np.zeros_like(t)
    for corner in range(3):
        np.add.at(t,faces[:,corner],tangent*weight)
        np.add.at(b,faces[:,corner],bitangent*weight)
    t=normalize(t-normals*(t*normals).sum(1)[:,None])
    empty=np.linalg.norm(t,axis=1)<1e-8
    if empty.any():
        axis=np.zeros_like(normals[empty]);axis[:,0]=1
        axis[np.abs(normals[empty,0])>.9]=[0,1,0]
        t[empty]=normalize(np.cross(axis,normals[empty]))
    sign=np.where((np.cross(normals,t)*b).sum(1)<0,-1.,1.)
    return np.c_[t,sign].astype(np.float32)


def create(source,output,*,size=2048,backend='cpu',owner=None,emit=lambda *a,**k:None):
    import trimesh
    if backend not in {'cpu','gpu'}:raise ValueError('UV 展开方式无效')
    mesh=trimesh.load(Path(source),force='mesh',process=False)
    positions=np.asarray(mesh.vertices,dtype=np.float32);faces=np.asarray(mesh.faces,dtype=np.uint32)
    if not len(faces) or not np.isfinite(positions).all():raise ValueError('UV 网格为空或坐标无效')
    normals=np.asarray(mesh.vertex_normals,dtype=np.float32)
    details={}
    if backend=='gpu':
        from .processes import pipeline_command,run_stage
        stage=Path(owner if owner is not None else Path(output).parent)/'stages/studio-uv'
        stage.mkdir(parents=True,exist_ok=True)
        arrays=stage/'uv-input.npz';result=stage/'uv-output.npz'
        np.savez_compressed(arrays,vertices=positions,faces=faces)
        identity=digest(arrays)
        emit('asset_uv','使用 GPU 展开 UV；保持原三角面、坐标和法线',device='cuda')
        run_stage(pipeline_command(),{'arrays':str(arrays),'output':str(result),
            'input_sha256':identity,'texture_size':size},stage)
        import json
        with np.load(result,allow_pickle=False) as saved:
            mapping=saved['mapping'];new_faces=saved['faces'];uv=saved['uv']
            details=json.loads(str(saved['metadata'].item()))
        if details.get('input_sha256')!=identity:raise RuntimeError('GPU UV 输出与当前输入不匹配')
        chart_count=details.get('uv_diagnostics',{}).get('charts')
        label='community GPU PEC / original indices'
    else:
        import xatlas
        emit('asset_uv','CPU 展开新网格 UV，按纹理像素保留边距',device='cpu')
        atlas=xatlas.Atlas();atlas.add_mesh(positions,faces,normals)
        charts=xatlas.ChartOptions();charts.max_iterations=4
        packing=xatlas.PackOptions();packing.resolution=size;packing.padding=4;packing.bilinear=True
        atlas.generate(charts,packing)
        if atlas.atlas_count!=1:raise UVLayoutError('网格需要多个图集，请提高贴图分辨率或先减面')
        mapping,new_faces,uv=atlas[0];chart_count=atlas.chart_count;label='xatlas 0.0.11 CPU'
        metrics=layout_metrics(new_faces,np.asarray(uv,dtype=np.float32),size)
        if metrics['zero_uv_faces'] or metrics['duplicate_uv_faces']:
            from ..engines.community_uv import _repair_local_faces
            pixels=np.column_stack((uv[:,0],1.-uv[:,1])).astype(np.float64)*size
            try:
                mapping,new_faces,uv,repair=_repair_local_faces(positions,mapping,new_faces,pixels,float(size),size,4,
                    lambda phase,message,**values:emit('asset_uv',message,device=values.get('device','cpu')))
            except RuntimeError as error:
                raise UVLayoutError(str(error)) from error
            details['uv_repair']=repair
    mapping=np.asarray(mapping);new_faces=np.asarray(new_faces)
    if (mapping.ndim!=1 or not np.issubdtype(mapping.dtype,np.integer) or np.any(mapping<0) or np.any(mapping>=len(positions))
            or new_faces.ndim!=2 or new_faces.shape[1]!=3 or not np.issubdtype(new_faces.dtype,np.integer)
            or np.any(new_faces<0) or np.any(new_faces>=len(mapping)) or np.shape(uv)!=(len(mapping),2)):
        raise RuntimeError('UV 展开返回无效的原始索引映射')
    expected=oriented_keys(faces)
    actual=oriented_keys(np.asarray(mapping)[new_faces])
    if not np.array_equal(expected,actual):
        expected=expected[np.lexsort(expected.T[::-1])];actual=actual[np.lexsort(actual.T[::-1])]
        if not np.array_equal(expected,actual):raise RuntimeError('UV 展开改变或丢失了原始有向三角面')
    new_positions=positions[mapping];new_normals=normals[mapping]
    uv=np.asarray(uv,dtype=np.float32)
    metrics=layout_metrics(new_faces,uv,size)
    if metrics['zero_uv_faces'] or metrics['duplicate_uv_faces']:
        raise UVLayoutError('UV 展开后仍有退化或复用三角面，请提高贴图分辨率或更换展开方式')
    document=Document({'asset':{'version':'2.0'},'scene':0,'scenes':[{'nodes':[0]}],
        'nodes':[{'mesh':0}],'meshes':[{'primitives':[]}],
        'materials':[{'name':'UV preparation','pbrMetallicRoughness':{'metallicFactor':0,'roughnessFactor':.5}}]})
    attrs={'POSITION':document.add_accessor(new_positions,target=34962),
           'NORMAL':document.add_accessor(new_normals,target=34962),
           'TEXCOORD_0':document.add_accessor(uv,target=34962),
           'TANGENT':document.add_accessor(tangent_vectors(new_positions,new_faces,uv,new_normals),target=34962)}
    document.tree['meshes'][0]['primitives']=[{'attributes':attrs,
        'indices':document.add_accessor(np.asarray(new_faces,dtype=np.uint32).reshape(-1),target=34963),'material':0}]
    report={'backend':label,'faces':len(faces),'charts':chart_count,
        'texture_size':size,'padding_pixels':4,'oriented_geometry_preserved':True,**metrics,'stage':details}
    document.tree['extras']={'studio_uv':report};document.save(output)
    emit('asset_uv','UV 图集已生成并验证原三角面保持',device='cpu')
    return report
