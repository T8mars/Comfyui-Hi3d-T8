"""Native ComfyUI boundaries: no workbench, HTTP or CUDA dependency in these tests."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

class Tensor:
    def __init__(self,value):self.value=np.asarray(value)
    @property
    def shape(self):return self.value.shape
    def __getitem__(self,index):return Tensor(self.value[index])
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.value

@pytest.fixture
def native(monkeypatch):
    spec=importlib.util.spec_from_file_location('native_boundary_nodes',Path(__file__).parents[1]/'comfy_nodes.py')
    nodes=importlib.util.module_from_spec(spec);spec.loader.exec_module(nodes)
    monkeypatch.setattr(nodes,'mesh_value',lambda v,f:SimpleNamespace(vertices=v,faces=f))
    mesh=SimpleNamespace(vertices=np.array([[0,0,0],[1,0,0],[0,1,0]]),faces=np.array([[0,1,2]]))
    calls=[]
    class Model:
        def refine(self,image,coarse,resolution,steps,seed):
            calls.append((image,coarse,resolution,steps,seed));return coarse,{'status':'succeeded','runtime':'ComfyUI in-process'}
    def run(image=None,mask=None):
        if image is None:image=Tensor(np.ones((1,2,2,3)))
        return nodes.Hi3DRefine().refine(Model(),image,mesh,512,25,23,mask)
    return SimpleNamespace(nodes=nodes,calls=calls,run=run,mesh=mesh)

def test_rgba_transparency_survives_connected_empty_mask(native):
    rgba=np.ones((1,2,2,4));rgba[0,0,0,3]=0;rgba[0,0,1,3]=.5
    native.run(Tensor(rgba),Tensor(np.zeros((1,2,2))))
    assert np.asarray(native.calls[0][0])[...,3].tolist()==[[0,128],[255,255]]

def test_comfy_mask_convention_mesh_and_seed_reach_native_model(native):
    result,report=native.run(mask=Tensor([[[1,0],[.25,.75]]]))
    image,mesh,resolution,steps,seed=native.calls[0]
    assert np.asarray(image)[...,3].tolist()==[[0,255],[191,64]]
    assert result.faces.tolist()==[[0,1,2]] and mesh.faces.tolist()==[[0,1,2]]
    assert (resolution,steps,seed)==(512,25,23)
    assert json.loads(report)['runtime']=='ComfyUI in-process'

@pytest.mark.parametrize('image,mask',[
    (np.ones((1,2,2,1)),None),(np.full((1,2,2,3),np.nan),None),
    (np.ones((1,2,2,3)),np.full((1,2,2),np.nan)),
    (np.ones((1,2,2,3)),np.ones((1,1,2,2))),
    (np.ones((2,2,2,3)),None),
])
def test_invalid_reference_rejected_before_native_inference(native,image,mask):
    with pytest.raises(ValueError):native.run(Tensor(image),None if mask is None else Tensor(mask))
    assert not native.calls

def test_matching_soft_image_mask_not_applied_twice(native):
    rgba=np.ones((1,2,2,4));rgba[...,3]=.5
    native.run(Tensor(rgba),Tensor(np.full((1,2,2),.5)))
    assert np.asarray(native.calls[0][0])[...,3].tolist()==[[128,128],[128,128]]

def test_native_interruption_is_propagated_without_fabricated_output(native):
    class Interrupted(Exception):pass
    class Model:
        def refine(self,*args):raise Interrupted('native cancellation')
    with pytest.raises(Interrupted,match='native cancellation'):
        native.nodes.Hi3DRefine().refine(Model(),Tensor(np.ones((1,2,2,3))),native.mesh,512,25,0)
