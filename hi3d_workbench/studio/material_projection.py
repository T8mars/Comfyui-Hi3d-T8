"""CPU nearest-surface transfer of glTF metallic/roughness PBR onto a new atlas.

All atlas texels sample the original surface; ray misses never become black.
Geometry/UVs are read, never moved. Unobserved interiors are explicitly reported
as extrapolation. Texture coordinates, factors and encodings follow glTF 2.0.
"""
from __future__ import annotations

import copy
import io
import math
from pathlib import Path

import numpy as np

from .gltf import Document
from .scene import instances,make_meshes_unique


def normalize(value):
    return value/np.maximum(np.linalg.norm(value,axis=-1,keepdims=True),1e-12)


def linear(value):
    return np.where(value<=.04045,value/12.92,((value+.055)/1.055)**2.4)


def srgb(value):
    value=np.maximum(value,0)
    return np.where(value<=.0031308,value*12.92,1.055*value**(1/2.4)-.055)


def _wrap_index(index,length,mode):
    if mode==33071:return np.clip(index,0,length-1)
    if mode==33648:
        index=index%(2*length)
        return np.where(index<length,index,2*length-1-index)
    if mode==10497:return index%length
    raise ValueError('不支持的纹理环绕方式')


def texture_sample(image,uv,sampler=None,*,color=False):
    sampler=sampler or {};height,width=image.shape[:2]
    def texels(y,x):
        values=image[y,x].astype(np.float32)
        if image.dtype==np.uint8:values/=255
        if color:values[:,:3]=linear(values[:,:3])
        return values
    # glTF (0,0) is the upper-left image corner. Linear filtering is performed
    # after sRGB decoding; alpha and data textures remain linear.
    x=uv[:,0]*width-.5;y=uv[:,1]*height-.5
    wrap_s=sampler.get('wrapS',10497);wrap_t=sampler.get('wrapT',10497)
    if sampler.get('magFilter',9729)==9728:
        ix=_wrap_index(np.floor(x+.5).astype(np.int64),width,wrap_s)
        iy=_wrap_index(np.floor(y+.5).astype(np.int64),height,wrap_t)
        return texels(iy,ix)
    x0=np.floor(x).astype(np.int64);y0=np.floor(y).astype(np.int64)
    tx=(x-x0)[:,None];ty=(y-y0)[:,None]
    xa,xb=_wrap_index(x0,width,wrap_s),_wrap_index(x0+1,width,wrap_s)
    ya,yb=_wrap_index(y0,height,wrap_t),_wrap_index(y0+1,height,wrap_t)
    return ((1-ty)*((1-tx)*texels(ya,xa)+tx*texels(ya,xb))
            +ty*((1-tx)*texels(yb,xa)+tx*texels(yb,xb)))


def _raster_impl(triangles,size):
    face=np.full((size,size),-1,dtype=np.int32)
    bary=np.zeros((size,size,2),dtype=np.float32)
    for i in range(len(triangles)):
        a,b,c=triangles[i]*size
        determinant=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(determinant)<1e-14:continue
        x0=max(0,int(math.floor(min(a[0],b[0],c[0])-.5)))
        x1=min(size-1,int(math.ceil(max(a[0],b[0],c[0])-.5)))
        y0=max(0,int(math.floor(min(a[1],b[1],c[1])-.5)))
        y1=min(size-1,int(math.ceil(max(a[1],b[1],c[1])-.5)))
        count=0
        for y in range(y0,y1+1):
            for x in range(x0,x1+1):
                u=((b[1]-c[1])*(x+.5-c[0])+(c[0]-b[0])*(y+.5-c[1]))/determinant
                v=((c[1]-a[1])*(x+.5-c[0])+(a[0]-c[0])*(y+.5-c[1]))/determinant
                if min(u,v,1-u-v)>=-1e-6:
                    face[y,x]=i;bary[y,x,0]=u;bary[y,x,1]=v;count+=1
        # Preserve a sample of a subpixel triangle when the pixel centre misses
        # it. Existing samples win, so shared edges do not overwrite a chart.
        if not count:
            x=min(size-1,max(0,int((a[0]+b[0]+c[0])/3)))
            y=min(size-1,max(0,int((a[1]+b[1]+c[1])/3)))
            if face[y,x]<0:
                face[y,x]=i;bary[y,x,0]=1/3;bary[y,x,1]=1/3
    return face,bary


_RASTER=None
def rasterize(triangles,size):
    global _RASTER
    triangles=np.asarray(triangles,dtype=np.float32)
    if (triangles.ndim!=3 or triangles.shape[1:]!=(3,2) or not np.isfinite(triangles).all()
            or triangles.min()<-1e-5 or triangles.max()>1+1e-5):
        raise ValueError('目标模型需要有效且位于 0–1 范围内的新 UV 图集')
    if _RASTER is None:
        from numba import njit
        _RASTER=njit(cache=True)(_raster_impl)
    return _RASTER(np.clip(triangles,0,1),size)


def frame(points,uv,normals):
    edge_a,edge_b=points[:,1]-points[:,0],points[:,2]-points[:,0]
    ua,ub=uv[:,1]-uv[:,0],uv[:,2]-uv[:,0]
    determinant=ua[:,0]*ub[:,1]-ua[:,1]*ub[:,0]
    safe=np.where(np.abs(determinant)>1e-12,determinant,1.)
    tangent=(edge_a*ub[:,1,None]-edge_b*ua[:,1,None])/safe[:,None]
    bitangent=(-edge_a*ub[:,0,None]+edge_b*ua[:,0,None])/safe[:,None]
    tangent=normalize(tangent-normals*(tangent*normals).sum(1)[:,None])
    bad=(np.linalg.norm(tangent,axis=1)<1e-8)|(np.abs(determinant)<1e-12)
    if bad.any():
        axis=np.zeros_like(normals[bad]);axis[:,0]=1
        axis[np.abs(normals[bad,0])>.9]=[0,1,0]
        tangent[bad]=normalize(np.cross(axis,normals[bad]))
    sign=np.where((np.cross(normals,tangent)*bitangent).sum(1)<0,-1.,1.)
    return tangent,np.cross(normals,tangent)*sign[:,None]


class Surface:
    def __init__(self,document):
        import open3d as o3d
        if document.tree.get('skins'):
            from .skin import freeze_current_pose
            document=freeze_current_pose(document)
        self.document=document;self.images={};self.accessors={};self.rows={};self.o3d=o3d
        self.cage_samples=0;self.cage_fallbacks=0
        source_rows=instances(document)
        coordinates=np.concatenate([row.world_positions() for row in source_rows])
        self.center=(coordinates.min(0)+coordinates.max(0))/2
        self.scale=float(np.ptp(coordinates,axis=0).max())
        if self.scale<=1e-10:raise ValueError('材质来源模型尺寸无效')
        self.scene=o3d.t.geometry.RaycastingScene(nthreads=4)
        allowed={'KHR_materials_emissive_strength'}
        for row in source_rows:
            definition=row.definition
            position_accessor=document.tree['accessors'][definition['attributes']['POSITION']]
            if position_accessor.get('normalized') and position_accessor['componentType']!=5126:
                raise ValueError('请先展开来源模型的量化坐标，再进行材质转移')
            material=document.tree.get('materials',[])[definition['material']] if 'material' in definition else {}
            unsupported=set(material.get('extensions',{}))-allowed
            if unsupported:raise ValueError('最近表面 PBR 转移暂不支持材质扩展：'+', '.join(sorted(unsupported)))
            positions=row.positions.astype(np.float64).copy()
            mesh=document.tree['meshes'][row.mesh]
            morph_weights=document.tree['nodes'][row.node].get('weights',mesh.get('weights',[]))
            for weight,morph in zip(morph_weights,definition.get('targets',[])):
                if weight and 'POSITION' in morph:positions+=weight*document.accessor(morph['POSITION'])
            world=positions @ row.world[:3,:3].T+row.world[:3,3]
            faces=row.faces[:,[0,2,1]] if np.linalg.det(row.world[:3,:3])<0 else row.faces
            vertex=o3d.core.Tensor(np.asarray((world-self.center)/self.scale,dtype=np.float32))
            triangle=o3d.core.Tensor(np.asarray(faces,dtype=np.uint32))
            identifier=self.scene.add_triangles(vertex,triangle)
            attrs=definition['attributes'];normals=None
            if 'NORMAL' in attrs:
                normals=document.accessor(attrs['NORMAL'],normalized=True).astype(np.float64)
                for weight,morph in zip(morph_weights,definition.get('targets',[])):
                    if weight and 'NORMAL' in morph:normals+=weight*document.accessor(morph['NORMAL'])
                normals=normalize(normals @ np.linalg.inv(row.world[:3,:3]))
            tangents=None
            if 'TANGENT' in attrs:
                tangents=self.attribute(attrs['TANGENT'],normalized=True).astype(np.float64)
                for weight,morph in zip(morph_weights,definition.get('targets',[])):
                    if weight and 'TANGENT' in morph:
                        tangents[:,:3]+=weight*self.attribute(morph['TANGENT'])
            self.rows[identifier]={'row':row,'positions':world,'faces':faces,'normals':normals,
                                   'tangents':tangents,'material':material}

    def attribute(self,index,*,normalized=False):
        key=(index,normalized)
        if key not in self.accessors:
            self.accessors[key]=self.document.accessor(index,normalized=normalized)
        return self.accessors[key]

    def texture(self,info,record,faces,bary,*,color=False):
        if not info:return np.ones((len(faces),4))
        transform=info.get('extensions',{}).get('KHR_texture_transform',{})
        coordinate=int(transform.get('texCoord',info.get('texCoord',0)))
        attrs=record['row'].definition['attributes'];key=f'TEXCOORD_{coordinate}'
        if key not in attrs:raise ValueError('来源材质缺少 '+key)
        uv=self.attribute(attrs[key],normalized=True)[record['faces'][faces]]
        uv=np.einsum('ni,nij->nj',bary,uv)
        angle=float(transform.get('rotation',0));scale=np.asarray(transform.get('scale',[1.,1.]))
        rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        uv=(uv*scale) @ rotation.T+np.asarray(transform.get('offset',[0.,0.]))
        texture=self.document.tree['textures'][info['index']];index=texture['source']
        if index not in self.images:
            from PIL import Image
            image=self.document.tree['images'][index]
            self.images[index]=np.asarray(Image.open(io.BytesIO(bytes(self.document.view(image['bufferView'])))).convert('RGBA'))
        sampler=self.document.tree.get('samplers',[])[texture['sampler']] if 'sampler' in texture else {}
        return texture_sample(self.images[index],uv,sampler,color=color)

    def sample(self,points,*,normals=None):
        coordinates=np.asarray((points-self.center)/self.scale,dtype=np.float32)
        query=self.o3d.core.Tensor(coordinates)
        closest=self.scene.compute_closest_points(query,nthreads=4)
        identifiers=closest['geometry_ids'].numpy();faces=closest['primitive_ids'].numpy()
        uv=closest['primitive_uvs'].numpy();bary=np.c_[1-uv.sum(1),uv]
        hits=closest['points'].numpy()
        if normals is not None:
            directions=normalize(np.asarray(normals,dtype=np.float32))
            if directions.shape!=coordinates.shape or not np.isfinite(directions).all():
                raise ValueError('材质投射需要有效的目标法线')
            # A nearest point can land on the underside of overlapping fur,
            # armour or cloth. A small outward cage selects the visible skin
            # along the target normal; a miss retains the nearest surface.
            radius=.03
            rays=np.c_[coordinates+directions*radius,-directions].astype(np.float32)
            cast=self.scene.cast_rays(self.o3d.core.Tensor(rays),nthreads=4)
            depth=cast['t_hit'].numpy()
            alignment=(cast['primitive_normals'].numpy()*directions).sum(1)
            valid=np.isfinite(depth)&(depth>=0)&(depth<=2*radius)&(alignment>.05)
            selected=np.flatnonzero(valid)
            offset=directions[valid]*(radius-depth[valid,None])
            identifiers[valid]=cast['geometry_ids'].numpy()[selected]
            faces[valid]=cast['primitive_ids'].numpy()[selected]
            ray_uv=cast['primitive_uvs'].numpy()[selected]
            bary[valid]=np.c_[1-ray_uv.sum(1),ray_uv]
            hits[valid]=coordinates[valid]+offset
            self.cage_samples+=int(valid.sum());self.cage_fallbacks+=int((~valid).sum())
        distance=np.linalg.norm(hits-coordinates,axis=1)*self.scale
        result={name:np.zeros((len(points),width),dtype=np.float32) for name,width in
                [('base',4),('mr',3),('emission',3),('normal',3)]}
        for identifier in np.unique(identifiers):
            record=self.rows[int(identifier)];mask=identifiers==identifier;selected=faces[mask];weights=bary[mask]
            material=record['material'];pbr=material.get('pbrMetallicRoughness',{})
            base=self.texture(pbr.get('baseColorTexture'),record,selected,weights,color=True)*np.array(pbr.get('baseColorFactor',[1.,1.,1.,1.]))
            attrs=record['row'].definition['attributes']
            if 'COLOR_0' in attrs:
                vertex_color=self.attribute(attrs['COLOR_0'],normalized=True)[record['faces'][selected]]
                interpolated=np.einsum('ni,nij->nj',weights,vertex_color)
                base[:,:interpolated.shape[1]]*=interpolated
            mode=material.get('alphaMode','OPAQUE')
            if mode=='OPAQUE':base[:,3]=1
            elif mode=='MASK':base[:,3]=(base[:,3]>=material.get('alphaCutoff',.5)).astype(float)
            mr=self.texture(pbr.get('metallicRoughnessTexture'),record,selected,weights)
            rough=mr[:,1]*pbr.get('roughnessFactor',1.)
            metal=mr[:,2]*pbr.get('metallicFactor',1.)
            ao_info=material.get('occlusionTexture')
            ao=self.texture(ao_info,record,selected,weights)[:,0]
            ao=1+(ao-1)*(ao_info or {}).get('strength',1.)
            emission=self.texture(material.get('emissiveTexture'),record,selected,weights,color=True)[:,:3]
            emission*=np.array(material.get('emissiveFactor',[0.,0.,0.]))
            emission*=material.get('extensions',{}).get('KHR_materials_emissive_strength',{}).get('emissiveStrength',1.)
            triangle=record['positions'][record['faces'][selected]]
            if record['normals'] is not None:
                normals=normalize(np.einsum('ni,nij->nj',weights,record['normals'][record['faces'][selected]]))
            else:normals=normalize(np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]))
            normal_info=material.get('normalTexture')
            if normal_info:
                if normal_info.get('extensions',{}).get('KHR_texture_transform'):
                    raise ValueError('含变换的法线纹理需要先应用变换再进行材质转移')
                texture_normal=self.texture(normal_info,record,selected,weights)[:,:3]*2-1
                texture_normal[:,:2]*=normal_info.get('scale',1.)
                texture_normal=normalize(texture_normal)
                if 'TANGENT' in attrs and normal_info.get('texCoord',0)==0:
                    tangent=record['tangents'][record['faces'][selected]]
                    tangent=np.einsum('ni,nij->nj',weights,tangent)
                    t=normalize(tangent[:,:3] @ record['row'].world[:3,:3].T)
                    t=normalize(t-normals*(t*normals).sum(1)[:,None])
                    sign=np.where(tangent[:,3]<0,-1.,1.)*np.sign(np.linalg.det(record['row'].world[:3,:3]))
                    b=np.cross(normals,t)*sign[:,None]
                else:
                    uv_key=f"TEXCOORD_{normal_info.get('texCoord',0)}"
                    texcoords=self.document.accessor(attrs[uv_key],normalized=True)[record['faces'][selected]]
                    t,b=frame(triangle,texcoords,normals)
                normals=normalize(t*texture_normal[:,0,None]+b*texture_normal[:,1,None]+normals*texture_normal[:,2,None])
            result['base'][mask]=base;result['mr'][mask]=np.c_[ao,rough,metal]
            result['emission'][mask]=emission;result['normal'][mask]=normals
        return result,distance


def _image(document,array,name):
    from PIL import Image
    stream=io.BytesIO();Image.fromarray(np.asarray(np.clip(array,0,1)*255+.5,dtype=np.uint8)).save(stream,format='PNG')
    view=document.add_view(stream.getvalue())
    images=document.tree.setdefault('images',[]);images.append({'name':name,'bufferView':view,'mimeType':'image/png'})
    textures=document.tree.setdefault('textures',[]);textures.append({'source':len(images)-1,'sampler':0})
    return {'index':len(textures)-1,'texCoord':0}


def transfer(source,target,output,*,size=2048,projection='nearest',emit=lambda *a,**k:None):
    if size not in {256,512,1024,2048,4096}:raise ValueError('材质图集尺寸无效')
    if projection not in {'nearest','cage'}:raise ValueError('材质投射方式无效')
    from scipy.ndimage import distance_transform_edt
    document=Document.load(target);make_meshes_unique(document)
    original=Document.load(source);surface=Surface(original)
    document.tree['images']=[];document.tree['textures']=[];document.tree['materials']=[]
    document.tree['samplers']=[{'magFilter':9729,'minFilter':9987,'wrapS':33071,'wrapT':33071}]
    supersample=2 if size<=2048 else 1;resolution=size*supersample
    rows=instances(document);reports=[]
    for row_index,row in enumerate(rows):
        attrs=row.definition['attributes']
        if 'TEXCOORD_0' not in attrs:raise ValueError('新网格缺少 UV 图集')
        uv=document.accessor(attrs['TEXCOORD_0'],normalized=True)[row.faces]
        face,bary=rasterize(uv,resolution);flat=np.flatnonzero(face.reshape(-1)>=0)
        if not len(flat):raise ValueError('新 UV 图集没有可采样表面')
        world=row.world_positions();triangle=world[row.faces]
        normals=(normalize(document.accessor(attrs['NORMAL']) @ np.linalg.inv(row.world[:3,:3])) if 'NORMAL' in attrs else None)
        tangents=document.accessor(attrs['TANGENT'],normalized=True) if 'TANGENT' in attrs else None
        maps={name:np.zeros((resolution*resolution,width),dtype=np.float32) for name,width in [('base',4),('mr',3),('emission',3),('normal',3)]}
        distances=[];opposed=0
        for start in range(0,len(flat),32768):
            pixels=flat[start:start+32768];indices=face.reshape(-1)[pixels]
            weights2=bary.reshape(-1,2)[pixels];weights=np.c_[weights2,1-weights2.sum(1)]
            points=np.einsum('ni,nij->nj',weights,triangle[indices])
            n=(normalize(np.einsum('ni,nij->nj',weights,normals[row.faces[indices]])) if normals is not None
               else normalize(np.cross(triangle[indices,1]-triangle[indices,0],triangle[indices,2]-triangle[indices,0])))
            values,distance=surface.sample(points,normals=n if projection=='cage' else None);distances.append(distance)
            if tangents is not None:
                tangent=tangents[row.faces[indices]]
                tangent=np.einsum('ni,nij->nj',weights,tangent)
                t=normalize(tangent[:,:3] @ row.world[:3,:3].T)
                t=normalize(t-n*(t*n).sum(1)[:,None])
                sign=np.where(tangent[:,3]<0,-1.,1.)*np.sign(np.linalg.det(row.world[:3,:3]))
                b=np.cross(n,t)*sign[:,None]
            else:t,b=frame(triangle[indices],uv[indices],n)
            wrong=(values['normal']*n).sum(1)<0
            # A newly closed interior can face away from its nearest original
            # surface. Retain its geometric normal instead of inverting light.
            opposed+=int(wrong.sum());values['normal'][wrong]=n[wrong]
            values['normal']=normalize(np.c_[(values['normal']*t).sum(1),(values['normal']*b).sum(1),(values['normal']*n).sum(1)])
            for name,array in maps.items():array[pixels]=values[name]
            method={'cage':'法线投射','nearest':'最近表面'}[projection]
            emit('asset_material_transfer',f"{method} PBR 转移 {row_index+1}/{len(rows)}",device='cpu',
                 progress_current=min(start+len(pixels),len(flat)),progress_total=len(flat),progress_unit='texels')
        covered=(face>=0).reshape(size,supersample,size,supersample)
        count=covered.sum((1,3));valid=count>0
        maps['base'][:,:3]*=maps['base'][:,3,None]
        for name,array in maps.items():
            channels=array.shape[1]
            array=array.reshape(size,supersample,size,supersample,channels).sum((1,3))/np.maximum(count[:,:,None],1)
            maps[name]=array.astype(np.float32)
        maps['base'][:,:,:3]/=np.maximum(maps['base'][:,:,3,None],1e-12)
        maps['normal']=normalize(maps['normal'])*.5+.5
        nearest=distance_transform_edt(~valid,return_distances=False,return_indices=True)
        for array in maps.values():array[~valid]=array[nearest[0,~valid],nearest[1,~valid]]
        base=maps['base'];base[:,:,:3]=srgb(base[:,:,:3])
        material={'name':f'Transferred PBR {row_index+1}','doubleSided':True,
            'pbrMetallicRoughness':{'baseColorTexture':_image(document,base,'Transferred base color'),
                'metallicRoughnessTexture':_image(document,maps['mr'],'Transferred occlusion roughness metallic'),
                'metallicFactor':1.,'roughnessFactor':1.},
            'normalTexture':_image(document,maps['normal'],'Transferred normal')}
        material['occlusionTexture']=copy.deepcopy(material['pbrMetallicRoughness']['metallicRoughnessTexture'])
        strength=max(1.,float(maps['emission'].max()))
        material['emissiveFactor']=[1.,1.,1.]
        material['emissiveTexture']=_image(document,srgb(maps['emission']/strength),'Transferred emission')
        if strength>1:
            material['extensions']={'KHR_materials_emissive_strength':{'emissiveStrength':strength}}
            document.tree.setdefault('extensionsUsed',[])
            if 'KHR_materials_emissive_strength' not in document.tree['extensionsUsed']:document.tree['extensionsUsed'].append('KHR_materials_emissive_strength')
        if np.any(base[valid,3]<.999):material['alphaMode']='BLEND'
        row.definition['material']=len(document.tree['materials']);document.tree['materials'].append(material)
        attrs.pop('COLOR_0',None)  # Source vertex colour is already in the atlas.
        distance=np.concatenate(distances)
        reports.append({'atlas_texels':int(valid.sum()),'surface_samples':len(flat),'unassigned_texels':0,
            'max_projection_distance':float(distance.max()),'p95_projection_distance':float(np.percentile(distance,95)),
            'extrapolated_samples':int((distance>surface.scale*.03).sum()),'opposed_surface_samples':opposed})
    method={'cage':'normal cage with nearest fallback','nearest':'nearest surface'}[projection]
    report={'material_transfer':'CPU '+method+' PBR projection',
        'projection':projection,'cage_samples':surface.cage_samples,'nearest_fallback_samples':surface.cage_fallbacks,
        'texture_size':size,'supersample':supersample,
        'source_has_materials':bool(original.tree.get('materials')),'atlases':reports,
        'notice':('保留新网格与 UV；沿目标法线投射原表面，未命中时回退最近表面。内部新表面属于推测转移，需要检查或重新贴图。'
                  if projection=='cage' else '保留新网格与 UV；内部新表面的材质取自最近原表面，属于推测转移，需要检查或重新贴图。')}
    document.tree.setdefault('extras',{})['studio_material_transfer']=report
    document.save(output)
    return report
