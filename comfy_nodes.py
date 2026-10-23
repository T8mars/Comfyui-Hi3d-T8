from __future__ import annotations
import io
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from urllib.parse import urlparse
import uuid
import os

_LOCAL_HTTP=urllib.request.build_opener(urllib.request.ProxyHandler({}))

def request(base,path,payload=None,token=None,raw=False,content_type=None):
    parsed=urlparse(base)
    if parsed.scheme!="http" or parsed.hostname not in {"127.0.0.1","localhost","::1"}:
        raise ValueError("Hi3D 后端只允许本机 HTTP 地址")
    headers={}
    if payload is not None and token is None:
        token=request(base,'/api/session')['token']
    if token:headers["x-hi3d-token"]=token
    if payload is not None and not isinstance(payload,bytes):
        payload=json.dumps(payload).encode();headers["content-type"]="application/json"
    if content_type:headers["content-type"]=content_type
    req=urllib.request.Request(base.rstrip('/')+path,data=payload,headers=headers)
    try:
        with _LOCAL_HTTP.open(req,timeout=600) as r:
            data=r.read()
            return data if raw else json.loads(data)
    except urllib.error.HTTPError as e:
        raise RuntimeError(e.read().decode('utf-8',errors='replace')) from e

def upload(backend,name,data):
    boundary=uuid.uuid4().hex
    body=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\nContent-Type: application/octet-stream\r\n\r\n').encode()+data+f'\r\n--{boundary}--\r\n'.encode()
    return request(backend['url'],'/api/assets',body,content_type='multipart/form-data; boundary='+boundary)

def arrays(mesh):
    import numpy as np
    v,f=mesh.vertices,mesh.faces
    if hasattr(v,'detach'):v=v.detach().cpu().numpy()
    if hasattr(f,'detach'):f=f.detach().cpu().numpy()
    v,f=np.asarray(v),np.asarray(f)
    if v.ndim==3:
        if len(v)!=1 or f.ndim!=3 or len(f)!=1:raise ValueError('Hi3D 目前只支持单物体 batch，顶点和面须一一对应')
        v,f=v[0],f[0]
        vc,fc=getattr(mesh,'vertex_counts',None),getattr(mesh,'face_counts',None)
        def unpad(value,counts):
            if counts is None:return value
            count=int(counts[0])
            if count!=counts[0] or not 0<count<=len(value):raise ValueError('网格的有效顶点 / 面数量无效')
            return value[:count]
        v,f=unpad(v,vc),unpad(f,fc)
    if v.ndim!=2 or v.shape[1:]!=(3,) or not len(v) or not np.isrealobj(v):raise ValueError('网格顶点必须是非空三维坐标')
    if f.ndim!=2 or f.shape[1:]!=(3,) or not len(f) or not np.issubdtype(f.dtype,np.integer):raise ValueError('三角面索引必须是整数，不能自动截断浮点值')
    if f.min()<0 or f.max()>=len(v):raise ValueError('三角面索引超出有效顶点范围')
    with np.errstate(over='ignore'):v=np.asarray(v,dtype=np.float32)
    if not np.isfinite(v).all():raise ValueError('网格顶点包含无效或超出精度范围的数值')
    return v,np.asarray(f,dtype=np.int64)

def mesh_value(v,f):
    import torch
    try:
        from comfy_api.latest._util.geometry_types import MESH
    except ImportError:
        from types import SimpleNamespace
        MESH=SimpleNamespace
    return MESH(vertices=torch.as_tensor(v,dtype=torch.float32).unsqueeze(0),faces=torch.as_tensor(f,dtype=torch.int64).unsqueeze(0))

class Hi3DBackend:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'url':('STRING',{'default':'http://127.0.0.1:8378'})}}
    RETURN_TYPES=('HI3D_BACKEND',)
    FUNCTION='connect'
    CATEGORY='Hi3D'
    def connect(self,url):
        request(url,'/api/session')
        return ({'url':url},)
    @classmethod
    def IS_CHANGED(cls,**kw):return float('nan')

class Hi3DLoadMesh:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'backend':('HI3D_BACKEND',),'path':('STRING',{'default':'','tooltip':'GLB/PLY/STL 本机路径；不包含其他模型'})}}
    RETURN_TYPES=('MESH',)
    FUNCTION='load'
    CATEGORY='Hi3D'
    def load(self,backend,path):
        import numpy as np
        p=Path(path).expanduser()
        if p.suffix.lower() not in {'.glb','.ply','.stl','.npz'}:raise ValueError('请选择 GLB/PLY/STL 网格')
        item=upload(backend,'coarse'+p.suffix.lower(),p.read_bytes())
        # Preserve the parsed mesh in a stable binary format returned by server.
        # Download GLB and parse using ComfyUI's native Get3DComponents when possible.
        data=request(backend['url'],item['url'],raw=True)
        try:
            from comfy_extras.nodes_mesh_io import Get3DComponents
            from comfy_api.latest._util.geometry_types import File3D
            return (Get3DComponents.execute(File3D(io.BytesIO(data),'glb'))[0],)
        except ImportError:
            # Legacy Comfy installations can install the lightweight trimesh extra.
            import trimesh
            m=trimesh.load(io.BytesIO(data),file_type='glb',force='mesh',process=False)
            return (mesh_value(np.asarray(m.vertices),np.asarray(m.faces)),)
    @classmethod
    def IS_CHANGED(cls,**kw):
        p=Path(kw['path'])
        return (p.stat().st_size,p.stat().st_mtime_ns) if p.is_file() else 'missing'

class Hi3DRefine:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'backend':('HI3D_BACKEND',),'image':('IMAGE',),'mesh':('MESH',),'resolution':([512,1024,1536],{'default':512}),'steps':('INT',{'default':25,'min':1,'max':100}),'seed':('INT',{'default':0,'min':0,'max':2**53-1}),'cpu_offload':('BOOLEAN',{'default':True}),'attention':(['flash','sdpa'],{'default':'sdpa' if os.name=='nt' else 'flash'})},'optional':{'mask':('MASK',)}}
    RETURN_TYPES=('MESH','STRING','STRING')
    RETURN_NAMES=('refined_mesh','report_json','job_id')
    FUNCTION='refine'
    CATEGORY='Hi3D'
    def refine(self,backend,image,mesh,resolution,steps,seed,cpu_offload,attention,mask=None):
        import numpy as np
        import torch
        from PIL import Image
        import comfy.model_management as mm
        from comfy.utils import ProgressBar
        if image.shape[0]!=1:raise ValueError('Hi3D 只支持单张参考图')
        pixels=(image[0].detach().cpu().numpy().clip(0,1)*255).round().astype('uint8')
        im=Image.fromarray(pixels).convert('RGBA')
        if mask is not None:
            m=mask.detach().cpu().numpy()
            if m.ndim==3:
                if m.shape[0]!=1:raise ValueError('Hi3D 只支持单个蒙版')
                m=m[0]
            # ComfyUI MASK: 1 means masked-out, so alpha is its inverse.
            alpha=Image.fromarray(((1-m.clip(0,1))*255).round().astype('uint8'))
            if alpha.size!=im.size:alpha=alpha.resize(im.size,Image.Resampling.BILINEAR)
            im.putalpha(alpha)
        imagebuf=io.BytesIO();im.save(imagebuf,format='PNG')
        v,f=arrays(mesh);meshbuf=io.BytesIO();np.savez_compressed(meshbuf,vertices=v,faces=f)
        imageitem=upload(backend,'reference.png',imagebuf.getvalue());meshitem=upload(backend,'coarse.npz',meshbuf.getvalue())
        mm.unload_all_models();mm.soft_empty_cache()
        job=request(backend['url'],'/api/jobs',{'image':imageitem['id'],'mesh':meshitem['id'],'resolution':resolution,'steps':steps,'seed':seed,'offload':cpu_offload,'attention':attention})
        jid=job['id'];pbar=ProgressBar(steps*({512:1,1024:2,1536:3}[resolution]))
        try:
            while True:
                mm.throw_exception_if_processing_interrupted()
                j=request(backend['url'],'/api/jobs/'+jid)
                if j['state']=='succeeded':break
                if j['state'] in {'failed','cancelled','interrupted'}:raise RuntimeError(j.get('message','Hi3D 未完成'))
                event=j.get('event',{})
                if event.get('stage')=='sampling' and 'step' in event and 'pass_index' in event:pbar.update_absolute(event['pass_index']*steps+event['step'])
                time.sleep(.5)
        except BaseException:
            try:request(backend['url'],f'/api/jobs/{jid}/cancel',{})
            except Exception:pass
            raise
        result=request(backend['url'],f'/api/jobs/{jid}/files/refined.npz',raw=True)
        with np.load(io.BytesIO(result),allow_pickle=False) as data:output=mesh_value(data['vertices'].copy(),data['faces'].copy())
        return (output,json.dumps(j['report'],ensure_ascii=False),jid)
    @classmethod
    def IS_CHANGED(cls,**kw):return float('nan')  # External artifacts never silently reuse Comfy's cache.

NODE_CLASS_MAPPINGS={'Hi3DBackend':Hi3DBackend,'Hi3DLoadMesh':Hi3DLoadMesh,'Hi3DRefine':Hi3DRefine}
NODE_DISPLAY_NAME_MAPPINGS={'Hi3DBackend':'Hi3D · 本地后端','Hi3DLoadMesh':'Hi3D · 加载粗网格','Hi3DRefine':'Hi3D V1.0 · 几何精修'}
