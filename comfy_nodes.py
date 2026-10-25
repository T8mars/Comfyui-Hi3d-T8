from __future__ import annotations
import json
from pathlib import Path
import os


def native_package():
    import importlib
    return importlib.import_module((__package__+'.' if __package__ else '')+'hi3d_native')

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


def image_to_pil(image,mask=None):
    import numpy as np
    from PIL import Image
    if len(image.shape)!=4 or image.shape[0]!=1 or image.shape[-1] not in (3,4) or min(image.shape[1:3])<=0:
        raise ValueError('Hi3D 只支持单张 RGB / RGBA 参考图，形状须为 1×H×W×3/4')
    pixels=image[0].detach().cpu().numpy()
    if not np.isfinite(pixels).all():raise ValueError('参考图包含无效数值')
    im=Image.fromarray((pixels.clip(0,1)*255).round().astype('uint8')).convert('RGBA')
    if mask is not None:
        m=mask.detach().cpu().numpy()
        if m.ndim==3:
            if m.shape[0]!=1:raise ValueError('Hi3D 只支持单个蒙版')
            m=m[0]
        if m.ndim!=2 or min(m.shape)<=0 or not np.isfinite(m).all():raise ValueError('蒙版须为有效的 H×W 或 1×H×W 数值')
        alpha=Image.fromarray(((1-m.clip(0,1))*255).round().astype('uint8'))
        if alpha.size!=im.size:alpha=alpha.resize(im.size,Image.Resampling.BILINEAR)
        im.putalpha(Image.fromarray(np.minimum(np.asarray(im.getchannel('A')),np.asarray(alpha))))
    return im


class Hi3DModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'model_root':('STRING',{'default':'','tooltip':'留空使用 ComfyUI/models/Hi3D；目录下包含 hi3d、dino、clip。支持自定义绝对路径与 extra_model_paths.yaml 的 hi3d 项。'}),
            'cpu_offload':('BOOLEAN',{'default':True}),
            'attention':(['sdpa','flash'],{'default':'sdpa' if os.name=='nt' else 'flash'}),
            'keep_model_loaded':('BOOLEAN',{'default':False,'tooltip':'连续任务保留模型内存；关闭时每次精修后释放模型。'}),
            'download_missing':('BOOLEAN',{'default':False,'tooltip':'缺少权重时，从固定版本的 t8star/Hi3d-Comfy 仅下载 Hi3D 核心与必需编码器。'})}}
    RETURN_TYPES=('HI3D_MODEL',)
    RETURN_NAMES=('model',)
    FUNCTION='load'
    CATEGORY='Hi3D'
    def load(self,model_root='',cpu_offload=True,attention='sdpa',keep_model_loaded=False,download_missing=False):
        import importlib
        package=native_package().__name__
        models=importlib.import_module(package+'.models')
        runtime=importlib.import_module(package+'.runtime')
        if attention not in {'sdpa','flash'}:raise ValueError('无效 Hi3D 注意力选项')
        return (runtime.ModelHandle(models.model_root(model_root),cpu_offload,attention,keep_model_loaded,download_missing),)
    @classmethod
    def IS_CHANGED(cls,model_root='',**kwargs):
        import importlib
        models=importlib.import_module(native_package().__name__+'.models')
        return models.identity(models.model_root(model_root))


class Hi3DLoadMesh:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        root=Path(folder_paths.get_input_directory())
        files=sorted(path.relative_to(root).as_posix() for path in root.rglob('*')
            if path.is_file() and path.suffix.lower() in {'.glb','.ply','.stl','.npz'})
        return {'required':{'mesh_file':(files or ['请先上传粗模'],{'tooltip':'点击上传粗模按钮或把 GLB / PLY / STL 文件拖到节点；文件保存在 ComfyUI/input/hi3d。'})}}
    RETURN_TYPES=('MESH',)
    RETURN_NAMES=('mesh',)
    FUNCTION='load'
    CATEGORY='Hi3D'
    @classmethod
    def file_path(cls,mesh_file):
        import folder_paths
        root=Path(folder_paths.get_input_directory()).resolve()
        path=Path(folder_paths.get_annotated_filepath(mesh_file)).resolve()
        if not path.is_relative_to(root) or path.suffix.lower() not in {'.glb','.ply','.stl','.npz'} or not path.is_file():
            raise ValueError('请上传 GLB / PLY / STL 粗模，并在下拉列表选择文件')
        return path
    def load(self,mesh_file):
        import numpy as np
        import trimesh
        path=self.file_path(mesh_file)
        if path.suffix.lower()=='.npz':
            with np.load(path,allow_pickle=False) as data:
                from types import SimpleNamespace
                vertices,faces=arrays(SimpleNamespace(vertices=data['vertices'],faces=data['faces']))
        else:
            mesh=trimesh.load(path,force='mesh',process=False,resolver={},allow_remote=False,skip_materials=True)
            if isinstance(mesh,trimesh.Scene):
                mesh=mesh.to_geometry() if hasattr(mesh,'to_geometry') else mesh.dump(concatenate=True)
            if not isinstance(mesh,trimesh.Trimesh):raise ValueError('文件必须包含三角网格')
            vertices,faces=arrays(mesh)
        if not np.any(np.ptp(vertices,axis=0)>1e-12):raise ValueError('粗模尺寸无效')
        return (mesh_value(vertices,faces),)
    @classmethod
    def IS_CHANGED(cls,mesh_file):
        path=cls.file_path(mesh_file)
        return path.stat().st_size,path.stat().st_mtime_ns,path.stat().st_ctime_ns
    @classmethod
    def VALIDATE_INPUTS(cls,mesh_file):
        try:cls.file_path(mesh_file);return True
        except ValueError as error:return str(error)


class Hi3DRefine:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'model':('HI3D_MODEL',),'image':('IMAGE',),'mesh':('MESH',),
            'resolution':([512,1024,1536],{'default':512}),
            'steps':('INT',{'default':25,'min':1,'max':100}),
            'seed':('INT',{'default':0,'min':0,'max':2**53-1})},'optional':{'mask':('MASK',)}}
    RETURN_TYPES=('MESH','STRING')
    RETURN_NAMES=('refined_mesh','report_json')
    FUNCTION='refine'
    CATEGORY='Hi3D'
    def refine(self,model,image,mesh,resolution,steps,seed,mask=None):
        import trimesh
        if resolution not in {512,1024,1536} or not 1<=steps<=100 or not 0<=seed<=2**53-1:
            raise ValueError('无效 Hi3D 推理配置')
        im=image_to_pil(image,mask)
        vertices,faces=arrays(mesh)
        coarse=trimesh.Trimesh(vertices,faces,process=False)
        refined,report=model.refine(im,coarse,resolution,steps,seed)
        return mesh_value(*arrays(refined)),json.dumps(report,ensure_ascii=False)


NODE_CLASS_MAPPINGS={'Hi3DModelLoader':Hi3DModelLoader,'Hi3DLoadMesh':Hi3DLoadMesh,'Hi3DRefine':Hi3DRefine}
NODE_DISPLAY_NAME_MAPPINGS={'Hi3DModelLoader':'Hi3D · 加载模型（独立）','Hi3DLoadMesh':'Hi3D · 上传粗模','Hi3DRefine':'Hi3D V1.0 · 几何精修'}
