"""Comfy client boundaries, with no ComfyUI, torch, HTTP or GPU process."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def nodes():
    spec=importlib.util.spec_from_file_location('hi3d_contract_nodes',Path(__file__).parents[1]/'comfy_nodes.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_node_interface_is_standalone_without_backend_or_http(nodes):
    assert set(nodes.NODE_CLASS_MAPPINGS)=={'Hi3DModelLoader','Hi3DLoadMesh','Hi3DRefine'}
    assert 'backend' not in nodes.Hi3DRefine.INPUT_TYPES()['required']
    assert not hasattr(nodes,'request') and not hasattr(nodes,'upload')
    assert nodes.Hi3DRefine.RETURN_TYPES==('MESH','STRING')


def test_float_face_indices_are_rejected_without_silent_geometry_change(nodes):
    mesh=SimpleNamespace(vertices=np.array([[0,0,0],[1,0,0],[0,1,0]]),faces=np.array([[0.,1.9,2.]]))
    with pytest.raises(ValueError,match='整数'):nodes.arrays(mesh)


@pytest.mark.parametrize('counts',[-1,0,5,1.5])
def test_invalid_padding_counts_are_rejected(nodes,counts):
    mesh=SimpleNamespace(vertices=np.zeros((1,4,3)),faces=np.zeros((1,2,3),dtype=np.int64),vertex_counts=[counts],face_counts=[1])
    with pytest.raises(ValueError,match='数量无效'):nodes.arrays(mesh)


def test_out_of_range_faces_and_non_finite_vertices_are_rejected(nodes):
    mesh=SimpleNamespace(vertices=np.zeros((3,3)),faces=np.array([[0,1,3]]))
    with pytest.raises(ValueError,match='索引超出'):nodes.arrays(mesh)
    mesh.faces=np.array([[0,1,2]]);mesh.vertices[0,0]=np.nan
    with pytest.raises(ValueError,match='无效'):nodes.arrays(mesh)
