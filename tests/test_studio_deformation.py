import numpy as np
import pytest

from test_workbench import testroot
from test_studio_parts import source_model
from test_studio_skin import rig_two,provider
from hi3d_workbench.studio.skin import bind,extract
from hi3d_workbench.studio.gltf import Document
from tools.validate_studio_deformation import analyze


def fixture(tmp_path,stretch=False):
    def split(points):
        side=(points[:,0]>0).astype(np.float32)
        return np.c_[1-side,side]
    document,_=bind(source_model(tmp_path/'colored.glb'),rig_two(),split if stretch else provider)
    source=tmp_path/'source.glb';document.save(source)
    rig=extract(document)
    time=document.add_accessor(np.array([0.,1.],dtype=np.float32))
    values=([[0,1,0],[20,1,0]] if stretch else [[0,0,0],[0,2,0]])
    position=document.add_accessor(np.asarray(values,dtype=np.float32))
    document.tree['animations']=[{'name':'actual saved motion',
        'samplers':[{'input':time,'output':position}],
        'channels':[{'sampler':0,'target':{'node':rig.nodes[1 if stretch else 0],'path':'translation'}}]}]
    target=tmp_path/'output.glb';document.save(target)
    return source,target


def test_rigid_motion_is_measured_over_every_saved_frame_without_false_stretch(tmp_path):
    source,target=fixture(tmp_path)
    progress=[];report=analyze(source,target,progress=lambda current,total:progress.append((current,total)))
    assert report['frames']==31 and progress[-1]==(31,31)
    assert report['max_severely_stretched_faces']==0
    assert report['max_ground_penetration']<1e-7
    assert report['records'][-1]['minimum_y']-report['records'][0]['minimum_y']==pytest.approx(2)
    assert report['geometry_attributes_materials_preserved'] and not report['quality_accepted']


def test_large_relative_joint_displacement_reports_actual_torn_surface(tmp_path):
    source,target=fixture(tmp_path,True)
    report=analyze(source,target)
    assert report['records'][0]['severely_stretched_faces']==0
    assert report['records'][-1]['severely_stretched_faces']>0
    assert report['records'][-1]['maximum_edge_ratio']>4
    assert report['records'][-1]['examples'] and not report['quality_accepted']


def test_same_accessor_number_cannot_hide_changed_inverse_bind_matrix(tmp_path):
    source,target=fixture(tmp_path)
    document=Document.load(target);index=document.tree['skins'][0]['inverseBindMatrices']
    values=document.accessor(index).copy();values[0,0]=2
    # Replace buffer bytes behind the existing accessor rather than changing its ID.
    accessor=document.tree['accessors'][index];view=document.tree['bufferViews'][accessor['bufferView']]
    start=view.get('byteOffset',0)+accessor.get('byteOffset',0)
    document.binary[start:start+values.nbytes]=values.astype('<f4').tobytes();document.save(target)
    with pytest.raises(ValueError,match='逆绑定矩阵数值'):analyze(source,target)
