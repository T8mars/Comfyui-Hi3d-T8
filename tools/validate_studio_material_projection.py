"""Real CPU PBR transfer oracle: rotated atlas, instancing, alpha and HDR light."""
from pathlib import Path
import io,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from hi3d_workbench.studio.gltf import Document
from hi3d_workbench.studio.material_projection import Surface,transfer
from hi3d_workbench.common import atomic_json


def texture(document,pixels):
    stream=io.BytesIO();Image.fromarray(pixels.astype(np.uint8)).save(stream,format='PNG')
    view=document.add_view(stream.getvalue());images=document.tree.setdefault('images',[])
    images.append({'bufferView':view,'mimeType':'image/png'})
    textures=document.tree.setdefault('textures',[]);textures.append({'source':len(images)-1,'sampler':0})
    return {'index':len(textures)-1}


def plane(positions,uv,tangent):
    d=Document({'asset':{'version':'2.0'},'scene':0,'scenes':[{'nodes':[0]}],
        'nodes':[{'mesh':0}],'meshes':[{'primitives':[]}],
        'samplers':[{'magFilter':9728,'wrapS':33071,'wrapT':33071}]})
    attrs={'POSITION':d.add_accessor(np.array(positions,dtype=np.float32)),
        'NORMAL':d.add_accessor(np.tile([0.,0,1],(4,1))),
        'TANGENT':d.add_accessor(np.asarray(np.tile(tangent,(4,1)),dtype=np.float32)),
        'TEXCOORD_0':d.add_accessor(np.array(uv,dtype=np.float32))}
    d.tree['meshes'][0]['primitives']=[{'attributes':attrs,'indices':d.add_accessor(np.array([0,1,2,0,2,3],dtype=np.uint32))}]
    return d


def main():
    folder=ROOT/'data/asset-studio-validation/material-projection-oracle';folder.mkdir(exist_ok=True)
    positions=np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0]])
    uv=np.array([[0.,0],[1,0],[1,1],[0,1]])
    source=plane(positions,uv,[1,0,0,1]);source.tree['nodes'][0].update(translation=[-1,.5,2],scale=[2,.5,1])
    color=np.array([[[255,0,0,128],[0,255,0,128]],[[0,0,255,128],[255,255,255,128]]],dtype=np.uint8)
    base=texture(source,color);mr=texture(source,np.array([[[64,128,192,255]]]))
    normal=texture(source,np.array([[[191,128,238,255]]]))
    emission=texture(source,np.array([[[255,255,255,255]]]))
    source.tree['materials']=[{'alphaMode':'BLEND','pbrMetallicRoughness':{
        'baseColorTexture':base,'baseColorFactor':[.5,1,.25,.75],
        'metallicRoughnessTexture':mr,'roughnessFactor':.8,'metallicFactor':.7},
        'normalTexture':normal,'occlusionTexture':{**mr,'strength':.5},
        'emissiveTexture':emission,'emissiveFactor':[1,.25,.1],
        'extensions':{'KHR_materials_emissive_strength':{'emissiveStrength':4}}}]
    source.tree['extensionsUsed']=['KHR_materials_emissive_strength']
    primitive=source.tree['meshes'][0]['primitives'][0];primitive['material']=0
    primitive['attributes']['COLOR_0']=source.add_accessor(np.tile([.5,.5,.5,1.],(4,1)))
    source.save(folder/'source.glb')
    world=positions*np.array([2,.5,1])+np.array([-1,.5,2])
    target=plane(world,np.c_[1-uv[:,1],uv[:,0]],[0,-1,0,1]);target.save(folder/'target.glb')
    start=time.monotonic();report=transfer(folder/'source.glb',folder/'target.glb',folder/'output.glb',size=256)
    points=np.array([[.25,.25,0],[.75,.25,0],[.25,.75,0],[.75,.75,0]])*np.array([2,.5,1])+np.array([-1,.5,2])
    before,_=Surface(Document.load(folder/'source.glb')).sample(points)
    after,_=Surface(Document.load(folder/'output.glb')).sample(points)
    expected_base=np.c_[np.array([[1.,0,0],[0,1,0],[0,0,1],[1,1,1]])*np.array([.25,.5,.125]),np.full(4,.75*128/255)]
    expected_mr=np.tile([1+(.25098039215686274-1)*.5,.8*128/255,.7*192/255],(4,1))
    expected_normal=np.array([191,128,238])/255*2-1;expected_normal/=np.linalg.norm(expected_normal)
    np.testing.assert_allclose(before['base'],expected_base,atol=1e-6)
    np.testing.assert_allclose(before['mr'],expected_mr,atol=1e-6)
    np.testing.assert_allclose(before['normal'],np.tile(expected_normal,(4,1)),atol=1e-6)
    np.testing.assert_allclose(before['emission'],np.tile([4.,1,.4],(4,1)),atol=1e-6)
    morphed=Document(source.tree,source.binary)
    morphed.tree['nodes'][0]['weights']=[1.]
    morphed.tree['meshes'][0]['primitives'][0]['targets']=[{
        'TANGENT':morphed.add_accessor(np.tile([-1.,1.,0.],(4,1)))}]
    morphed.save(folder/'source-tangent-morph.glb')
    morphed_values,_=Surface(morphed).sample(points)
    rotated_normal=expected_normal[[1,0,2]]*np.array([-1.,1.,1.])
    np.testing.assert_allclose(morphed_values['normal'],np.tile(rotated_normal,(4,1)),atol=1e-6)
    errors={name:float(np.abs(after[name]-before[name]).max()) for name in before}
    assert errors['base']<.006 and errors['mr']<.003 and errors['normal']<.01 and errors['emission']<.03,errors
    output=Document.load(folder/'output.glb')
    np.testing.assert_array_equal(output.accessor(output.tree['meshes'][0]['primitives'][0]['attributes']['POSITION']),world.astype(np.float32))
    result={'status':'passed','seconds':time.monotonic()-start,'max_channel_errors':errors,
            'morphed_tangent_normals_verified':True,'geometry_preserved':True,'transfer':report}
    # The underside of layered geometry can be closer than the visible skin.
    # Verify a real Open3D ray query, rather than mirroring the implementation.
    layered=plane(positions+np.array([0,0,.01]),uv,[1,0,0,1])
    layered.tree['materials']=[{'pbrMetallicRoughness':{'baseColorFactor':[1,0,0,1]}},
                               {'pbrMetallicRoughness':{'baseColorFactor':[0,0,1,1]}}]
    layered.tree['meshes'][0]['primitives'][0]['material']=0
    attrs={'POSITION':layered.add_accessor((positions+np.array([0,0,-.001])).astype(np.float32)),
           'NORMAL':layered.add_accessor(np.tile([0.,0,-1],(4,1)))}
    layered.tree['meshes'][0]['primitives'].append({'attributes':attrs,'material':1,
        'indices':layered.add_accessor(np.array([0,2,1,0,3,2],dtype=np.uint32))})
    surface=Surface(layered);query=np.array([[.4,.6,0.],[.6,.4,0.]])
    nearest,_=surface.sample(query)
    cage,distance=surface.sample(query,normals=np.tile([0.,0,1.],(len(query),1)))
    np.testing.assert_allclose(nearest['base'],np.tile([0,0,1,1.],(2,1)))
    np.testing.assert_allclose(cage['base'],np.tile([1,0,0,1.],(2,1)))
    np.testing.assert_allclose(distance,.01,atol=1e-7)
    assert surface.cage_samples==2 and surface.cage_fallbacks==0
    fallback,_=surface.sample(query,normals=np.tile([1.,0,0],(len(query),1)))
    np.testing.assert_allclose(fallback['base'],nearest['base'])
    assert surface.cage_fallbacks==2
    result['cage_visible_layer_and_miss_fallback_verified']=True
    atomic_json(folder/'check.json',result);print(json.dumps(result))


if __name__=='__main__':main()
