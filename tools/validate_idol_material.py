"""Independent colour oracle for unlit vertex-colour -> UV atlas transfer."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hi3d_workbench.studio.gltf import Document
from hi3d_workbench.studio.material_projection import Surface,transfer
from hi3d_workbench.common import atomic_json,digest


def main():
    folder=ROOT/'data/asset-studio-validation/idol-unlit-material';folder.mkdir(exist_ok=True)
    positions=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],dtype=np.float32)
    faces=np.array([0,1,2,0,2,3],dtype=np.uint32)
    uv=np.array([[0,0],[1,0],[1,1],[0,1]],dtype=np.float32)
    def plane():
        d=Document({'asset':{'version':'2.0'},'scene':0,'scenes':[{'nodes':[0]}],
            'nodes':[{'mesh':0}],'meshes':[{'primitives':[]}]})
        d.tree['meshes'][0]['primitives']=[{'attributes':{'POSITION':d.add_accessor(positions),
            'TEXCOORD_0':d.add_accessor(uv)},'indices':d.add_accessor(faces)}]
        return d
    source=plane();color=np.tile([.1,.4,.8,1.],(4,1)).astype(np.float32)
    source.tree['meshes'][0]['primitives'][0]['attributes']['COLOR_0']=source.add_accessor(color)
    source.tree['meshes'][0]['primitives'][0]['material']=0
    source.tree['materials']=[{'extensions':{'KHR_materials_unlit':{}},'doubleSided':True}]
    source.tree['extensionsUsed']=['KHR_materials_unlit'];source.save(folder/'source.glb')
    target=plane();target.save(folder/'target.glb')
    report=transfer(folder/'source.glb',folder/'target.glb',folder/'output.glb',size=256)
    output=Document.load(folder/'output.glb')
    assert 'KHR_materials_unlit' in output.tree['materials'][0]['extensions']
    points=np.array([[.1,.2,0],[.8,.7,0],[.5,.5,0]])
    actual,_=Surface(output).sample(points)
    np.testing.assert_allclose(actual['base'],np.tile([.1,.4,.8,1.],(3,1)),atol=.006)
    attrs=output.tree['meshes'][0]['primitives'][0]['attributes']
    np.testing.assert_array_equal(output.accessor(attrs['POSITION']),positions)
    atomic_json(folder/'validation.json',{'status':'passed','preserves_unlit':True,'geometry_preserved':True,
        'max_color_error':float(np.abs(actual['base']-np.tile([.1,.4,.8,1.],(3,1))).max()),
        'output_sha256':digest(folder/'output.glb'),'report':report})
    print('Unlit vertex-colour transfer oracle passed')


if __name__=='__main__':main()
