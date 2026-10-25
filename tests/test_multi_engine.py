"""Model choice boundaries and persistence; fixtures are not inference evidence."""
import json
import shutil

import pytest

from test_asset_pipeline import backend, tiny_extensions, _saved_job, _textured_fixture
from test_workbench import client, upload_pair
from hi3d_workbench import server, pipeline_models, pipeline_worker
from hi3d_workbench.common import read_json, atomic_json
from hi3d_workbench.engine_catalog import ENGINES, engine, is_ready


@pytest.mark.parametrize('field,value', [
    ('geometry_engine','../../injected'), ('texture_engine','custom_command'),
    ('geometry_resolution',0), ('geometry_resolution',999),
    ('geometry_steps',0), ('texture_steps',101),
])
def test_invalid_engine_options_never_launch_workers(backend, field, value):
    client, ready, captures, checks = backend
    image, _ = upload_pair(client)
    result = client.post('/api/jobs', json={'image':image['id'], 'mode':'image_to_asset', field:value})
    assert result.status_code in {400,422}
    assert captures == []


def test_uninstalled_optional_engine_does_not_fall_back_to_baseline(backend):
    client, ready, captures, checks = backend
    image, _ = upload_pair(client)
    result = client.post('/api/jobs', json={'image':image['id'], 'mode':'image_to_asset', 'geometry_engine':'hunyuan21'})
    assert result.status_code == 409
    assert captures == []


def test_selected_model_settings_persist_with_engine_defaults(backend, monkeypatch):
    client, ready, captures, checks = backend
    monkeypatch.setattr(server.pipeline_models, 'status', lambda: {'background_ready':True, 'engines':{
        'coarse':{'hunyuan21':{'ready':True}}, 'texture':{'ultratex':{'ready':True}}}})
    image, _ = upload_pair(client)
    result = client.post('/api/jobs',json={'image':image['id'], 'mode':'image_to_asset',
                    'geometry_engine':'hunyuan21', 'texture_engine':'ultratex'})
    assert result.status_code == 200, result.text
    options = captures[0][2]
    assert options['geometry_engine'] == 'hunyuan21' and options['geometry_resolution'] == 384
    assert options['geometry_steps'] == 50
    assert options['texture_engine'] == 'ultratex' and options['texture_resolution'] == 1024
    assert options['texture_steps'] == 25


def test_only_selected_engine_readiness_can_authorize_a_job(backend, monkeypatch):
    client, ready, captures, checks = backend
    state={'background_ready':True, 'coarse_ready':False, 'texture_ready':False, 'engines':{
        'coarse':{'hunyuan21':{'ready':True}, 'hunyuan_mini':{'ready':False}},
        'texture':{'trellis2':{'ready':True},'ultratex':{'ready':False}}}}
    monkeypatch.setattr(server.pipeline_models,'status',lambda:state)
    image, _ = upload_pair(client)
    result=client.post('/api/jobs',json={'image':image['id'],'mode':'image_to_asset','geometry_engine':'hunyuan21'})
    assert result.status_code == 200, result.text
    captures.clear()
    result=client.post('/api/jobs',json={'image':image['id'],'mode':'image_to_asset','geometry_engine':'hunyuan21','texture_engine':'ultratex'})
    assert result.status_code == 409 and captures == []


def test_optional_missing_models_never_invalidate_baseline(tiny_extensions, monkeypatch):
    monkeypatch.setitem(pipeline_models.SPECS,'optional_fixture',{
        'repo':'fixture/optional','revision':'fixed','license':'test','files':['missing.bin'],'optional':True})
    value=pipeline_models.status()
    assert value['ready']
    assert not value['components']['optional_fixture']['ready']
    assert pipeline_models.verify()['verification_errors'] == []
    assert 'optional_fixture' not in pipeline_models.selection_components('all')


def test_engine_downloads_only_include_declared_dependencies():
    assert pipeline_models.selection_components('engine:coarse:hunyuan21') == ['background','coarse_hunyuan21']
    assert pipeline_models.selection_components('engine:texture:ultratex') == ['background','texture_ultratex','ultratex_backbone','ultratex_text_encoder']
    assert pipeline_models.selection_components('engine:texture:ultratex_int8') == ['background','texture_ultratex','ultratex_backbone_int8','ultratex_text_encoder']
    with pytest.raises(ValueError):pipeline_models.selection_components('engine:coarse:unknown')


def test_complete_bundle_covers_every_registered_engine_and_component():
    from hi3d_workbench.engine_catalog import all_engine_selections,covers_all_engines
    components={component for selection in all_engine_selections() for component in pipeline_models.selection_components(selection)}
    assert components==set(pipeline_models.SPECS)
    assert set(pipeline_models.selection_components('complete'))==components
    assert covers_all_engines(components)
    assert not covers_all_engines({'background','coarse','texture','texture_encoder'})
    assert not covers_all_engines(components-{'ultratex_backbone_int8'})


def test_int8_texture_routes_to_ultratex_with_original_engine_identity(monkeypatch,tmp_path):
    from hi3d_workbench.engines import run_stage,ultratex_texture
    monkeypatch.setattr(ultratex_texture,'generate',lambda request,*args:{'engine':request['engine']})
    for choice in ('ultratex','ultratex_int8'):
        assert run_stage.texture({'engine':choice},None,tmp_path/'result.glb',lambda *a,**k:None)=={'engine':choice}


def test_comparison_records_include_only_existing_models_and_input_identity(client):
    folder=_saved_job(mesh=True)
    state=read_json(folder/'state.json')
    state['report']={'input_sha256':'a'*64,'texture_mesh_sha256':'b'*64}
    atomic_json(folder/'state.json',state)
    result=client.get(f'/api/jobs/{folder.name}/comparisons')
    assert result.status_code==200
    item=next(item for item in result.json() if item['id']==folder.name)
    assert item['available_files']==['coarse.glb']
    assert item['input_sha256']=='a'*64 and item['texture_mesh_sha256']=='b'*64
    assert not {'worker_command','worker_pid','request_path','model_root'} & set(item)
    (folder/'coarse.glb').unlink()
    assert all(item['id']!=folder.name for item in client.get(f'/api/jobs/{folder.name}/comparisons').json())


def test_comparison_endpoint_rejects_unknown_job(client):
    assert client.get('/api/jobs/no-such-quality-job/comparisons').status_code==404


def test_uploaded_glb_keeps_original_uv_materials_and_bytes(client,tmp_path):
    source=tmp_path/'artist.glb';_textured_fixture(source)
    import numpy as np
    import trimesh
    mesh=trimesh.load(source,force='mesh',process=False)
    material=mesh.visual.material
    mesh.unmerge_vertices()
    # A real per-face atlas, rather than projecting a box onto one plane.
    uv=np.zeros((len(mesh.vertices),2))
    for index,face in enumerate(mesh.faces):
        cell=np.array([index%4,index//4])/4
        uv[face]=cell+np.array([[.01,.01],[.22,.01],[.01,.22]])
    mesh.visual=trimesh.visual.TextureVisuals(uv=uv,material=material)
    mesh.export(source)
    result=client.post('/api/assets',files={'file':('artist.glb',source.read_bytes(),'model/gltf-binary')})
    assert result.status_code==200,result.text
    asset=result.json()
    assert asset['has_materials']
    assert client.get(asset['url']).content==source.read_bytes()
    from hi3d_workbench.asset_pipeline import load_texture_input
    mesh,policy=load_texture_input(source)
    assert policy=='input_single_material_uv' and mesh.visual.uv is not None


def _authored_normal_mesh():
    import numpy as np
    import trimesh
    from PIL import Image
    material=trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.new('RGB',(2,2),(30,100,190)),
        metallicRoughnessTexture=Image.new('RGB',(2,2),(255,140,70)),
        metallicFactor=.75,roughnessFactor=.6)
    return trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]],faces=[[0,1,2]],
        vertex_normals=[[.6,0,.8],[-.6,0,.8],[0,.6,.8]],process=False,
        visual=trimesh.visual.TextureVisuals(uv=[[.1,.1],[.9,.1],[.1,.9]],material=material))


@pytest.mark.parametrize('kind',['identity','translation','nonuniform','mirror'])
def test_real_glb_authored_normals_survive_node_transform_uv_and_pbr(tmp_path,kind):
    import numpy as np
    import trimesh
    from hi3d_workbench.asset_pipeline import load_texture_input,read_glb_container
    transform=np.eye(4)
    if kind=='translation':transform[:3,3]=[3,-5,7]
    if kind in {'nonuniform','mirror'}:
        rotation=trimesh.transformations.rotation_matrix(.37,[0,0,1])[:3,:3]
        transform[:3,:3]=rotation@np.diag([-2 if kind=='mirror' else 2,.5,1.5])
        transform[:3,3]=[3,-5,7]
    path=tmp_path/'authored.glb'
    scene=trimesh.Scene()
    scene.add_geometry(_authored_normal_mesh(),geom_name='authored',node_name='node',transform=transform)
    scene.export(path)
    tree=read_glb_container(path)
    assert 'NORMAL' in tree['meshes'][0]['primitives'][0]['attributes']
    # Read the actual serialized NORMAL accessor through the scene loader;
    # comparing to a separately reconstructed normal would conceal this bug.
    raw=trimesh.load(path,force='scene',process=False)
    node=raw.graph.nodes_geometry[0]
    matrix,name=raw.graph[node]
    original=raw.geometry[name]
    original_normals=original.vertex_normals.copy()
    expected=original_normals@np.linalg.inv(matrix[:3,:3])
    if kind!='identity' and kind!='translation':expected/=np.linalg.norm(expected,axis=1,keepdims=True)
    loaded,policy=load_texture_input(path)
    assert policy=='input_single_material_uv'
    assert np.allclose(loaded.vertices,trimesh.transformations.transform_points(original.vertices,matrix),atol=1e-12)
    if kind in {'identity','translation'}:assert np.array_equal(loaded.vertex_normals,original_normals)
    else:assert np.allclose(loaded.vertex_normals,expected,atol=1e-12)
    assert np.array_equal(loaded.faces,original.faces[:,::-1] if kind=='mirror' else original.faces)
    assert np.array_equal(loaded.visual.uv,original.visual.uv)
    for channel in ('baseColorTexture','metallicRoughnessTexture'):
        assert np.array_equal(np.asarray(getattr(loaded.visual.material,channel)),
                              np.asarray(getattr(original.visual.material,channel)))
    assert loaded.visual.material.metallicFactor==original.visual.material.metallicFactor
    assert loaded.visual.material.roughnessFactor==original.visual.material.roughnessFactor
    assert not np.allclose(loaded.vertex_normals,loaded.face_normals[loaded.vertex_faces[:,0]])
    assert np.array_equal(original.vertex_normals,original_normals)


def test_texture_frame_restore_preserves_file_normals_after_real_glb_roundtrip(tmp_path):
    import numpy as np
    import trimesh
    from hi3d_workbench.engines.run_stage import restore_texture_frame
    original=_authored_normal_mesh()
    normals=original.vertex_normals.copy()
    vertices=original.vertices.copy()
    uv=original.visual.uv.copy()
    center=np.array([8.,-3.,2.]);scale=.4
    restore_texture_frame(original,center,scale)
    assert np.array_equal(original.visual.uv,uv)
    target=tmp_path/'restored.glb';original.export(target)
    loaded=next(iter(trimesh.load(target,force='scene',process=False).geometry.values()))
    assert np.allclose(loaded.vertices,vertices/scale+center,atol=1e-6)
    assert np.allclose(loaded.vertex_normals,normals,atol=1e-7)
    assert np.allclose(loaded.visual.uv,uv,rtol=0,atol=1e-7)
    assert np.array_equal(loaded.faces,original.faces)
    assert np.array_equal(loaded.visual.material.baseColorTexture,original.visual.material.baseColorTexture)


def test_real_glb_shared_geometry_instances_keep_per_node_inverse_transpose_normals(tmp_path):
    import numpy as np
    import trimesh
    from hi3d_workbench.asset_pipeline import load_texture_input
    path=tmp_path/'instances.glb'
    scene=trimesh.Scene()
    scene.add_geometry(_authored_normal_mesh(),geom_name='authored',node_name='one')
    second=np.eye(4);second[:3,:3]=np.diag([-2,.5,1.5]);second[:3,3]=[3,4,5]
    scene.graph.update(frame_to='two',frame_from=scene.graph.base_frame,matrix=second,geometry='authored')
    scene.export(path)
    raw=trimesh.load(path,force='scene',process=False)
    assert len(raw.geometry)==1 and len(raw.graph.nodes_geometry)==2
    vertices=[];normals=[];faces=[];offset=0
    for node in raw.graph.nodes_geometry:
        transform,name=raw.graph[node];mesh=raw.geometry[name]
        vertices.append(trimesh.transformations.transform_points(mesh.vertices,transform))
        transformed=mesh.vertex_normals@np.linalg.inv(transform[:3,:3])
        if not np.array_equal(transform[:3,:3],np.eye(3)):
            transformed/=np.linalg.norm(transformed,axis=1,keepdims=True)
        normals.append(transformed)
        faces.append((mesh.faces[:,::-1] if np.linalg.det(transform[:3,:3])<0 else mesh.faces)+offset)
        offset+=len(mesh.vertices)
    loaded,policy=load_texture_input(path)
    assert np.allclose(loaded.vertices,np.vstack(vertices),atol=1e-12)
    assert np.allclose(loaded.vertex_normals,np.vstack(normals),atol=1e-12)
    assert np.array_equal(loaded.faces,np.vstack(faces))
    # Both instances reuse one UV region, so the existing overlap guard
    # requests a new atlas without discarding their authored normals.
    assert policy=='generate_uv'


def test_real_glb_singular_node_transform_is_rejected_without_fabricating_normals(tmp_path):
    import numpy as np
    import trimesh
    from hi3d_workbench.asset_pipeline import load_texture_input
    transform=np.eye(4);transform[0,0]=0
    scene=trimesh.Scene()
    scene.add_geometry(_authored_normal_mesh(),transform=transform)
    path=tmp_path/'singular.glb';scene.export(path)
    with pytest.raises(ValueError,match='不可逆'):
        load_texture_input(path)


def test_exporting_original_uploaded_glb_preserves_materials_without_claiming_generation(client,tmp_path):
    source=tmp_path/'artist.glb';_textured_fixture(source)
    folder=_saved_job(mode='texture',mesh=True)
    shutil.copy2(source,folder/'coarse.glb')
    result=client.post(f'/api/jobs/{folder.name}/export',json={'source':'coarse','format':'glb'})
    assert result.status_code==200,result.text
    info=result.json()
    assert info['material_preserved'] and not info['texture_generated']
    assert client.get(info['url']).content==source.read_bytes()


def test_raw_generated_geometry_can_be_exported_independently(client,tmp_path):
    folder=_saved_job(mesh=True)
    shutil.copy2(folder/'coarse.glb',folder/'generated.glb')
    result=client.post(f'/api/jobs/{folder.name}/export',json={'source':'generated','format':'glb'})
    assert result.status_code==200,result.text
    assert client.get(result.json()['url']).content==(folder/'generated.glb').read_bytes()


@pytest.mark.parametrize('source_kind,format_kind',[('coarse','glb'),('coarse','stl'),('textured','glb')])
def test_export_selected_y_dimension_preserves_source_and_materials(client,tmp_path,source_kind,format_kind):
    import numpy as np
    import trimesh
    from hi3d_workbench.geometry import load_mesh
    from hi3d_workbench.asset_pipeline import inspect_textured_glb
    source=tmp_path/'artist.glb';_textured_fixture(source)
    folder=_saved_job(mesh=True)
    shutil.copy2(source,folder/(source_kind+'.glb'))
    original=source.read_bytes()
    result=client.post(f'/api/jobs/{folder.name}/export',json={
        'source':source_kind,'format':format_kind,'height_mm':80,'height_axis':'Y'})
    assert result.status_code==200,result.text
    info=result.json();assert info['height_axis']=='Y'
    target=tmp_path/('export.'+format_kind);target.write_bytes(client.get(info['url']).content)
    expected=.08 if format_kind=='glb' else 80
    assert abs(load_mesh(target).extents[1]-expected)<1e-5
    assert (folder/(source_kind+'.glb')).read_bytes()==original
    if format_kind=='glb':assert inspect_textured_glb(target)['embedded']


def test_compacted_uv_vertices_recover_the_corresponding_input_shading_normals():
    import numpy as np
    import trimesh
    from hi3d_workbench.engines.mesh_uv import match_source_vertices
    source=trimesh.creation.box()
    permutation=np.array([5,0,3,7,5,1])  # reordered native vertices and a UV seam duplicate
    positions,normals=match_source_vertices(source,source.vertices[permutation])
    assert np.array_equal(positions,source.vertices[permutation])
    assert np.array_equal(normals,source.vertex_normals[permutation])
    with pytest.raises(RuntimeError,match='改变'):
        match_source_vertices(source,np.array([[40.,50.,60.]]))


def test_coincident_hard_edge_normals_never_use_position_only_matching():
    import trimesh
    from hi3d_workbench.engines.mesh_uv import match_source_vertices,has_split_normals
    source=trimesh.creation.box();source.unmerge_vertices()
    assert has_split_normals(source)
    with pytest.raises(RuntimeError,match='硬边'):
        match_source_vertices(source,source.vertices)


def test_compacted_hard_edges_use_exact_incident_topology_after_reordering():
    import numpy as np
    import trimesh
    from hi3d_workbench.engines.mesh_uv import compacted_source_indices
    source=trimesh.creation.box();source.unmerge_vertices()
    source.vertex_normals=np.repeat(source.face_normals,3,axis=0)
    permutation=np.random.default_rng(42).permutation(len(source.vertices))
    inverse=np.argsort(permutation)
    native_faces=np.roll(inverse[source.faces][::-1],1,axis=1)
    mapping=compacted_source_indices(source,source.vertices[permutation],native_faces)
    assert np.array_equal(mapping,permutation)
    assert np.array_equal(source.vertex_normals[mapping],source.vertex_normals[permutation])


def test_compacted_mapping_accepts_deleted_faces_and_unused_vertices_only():
    import numpy as np
    import trimesh
    from hi3d_workbench.engines.mesh_uv import compacted_source_indices
    source=trimesh.creation.box();source.unmerge_vertices()
    # Native compaction is allowed to delete a degenerate triangle and its
    # now-unused vertices; it cannot identify vertices by their old offsets.
    source=trimesh.Trimesh(np.vstack([source.vertices,[[8,9,10]]]),
        np.vstack([source.faces,[[len(source.vertices)]*3]]),process=False)
    permutation=np.arange(len(source.vertices)-1)[::-1]
    inverse=np.argsort(permutation)
    mapping=compacted_source_indices(source,source.vertices[permutation],inverse[source.faces[:-1]])
    assert np.array_equal(mapping,permutation)


def test_compacted_mapping_refuses_unprovable_overlapping_shell_identity():
    import numpy as np
    import trimesh
    from hi3d_workbench.engines.mesh_uv import compacted_source_indices
    box=trimesh.creation.box()
    source=trimesh.Trimesh(np.vstack([box.vertices,box.vertices]),
        np.vstack([box.faces,box.faces+len(box.vertices)]),process=False)
    source.vertex_normals=np.vstack([box.vertex_normals,-box.vertex_normals])
    with pytest.raises(RuntimeError,match='唯一对应'):
        compacted_source_indices(source,source.vertices,source.faces)


@pytest.mark.parametrize('mutation',['position','winding','duplicate_face'])
def test_compacted_mapping_never_accepts_modified_geometry(mutation):
    import numpy as np
    import trimesh
    from hi3d_workbench.engines.mesh_uv import compacted_source_indices
    source=trimesh.creation.box()
    positions=source.vertices.copy();faces=source.faces.copy()
    if mutation=='position':positions[0,0]+=1e-5
    if mutation=='winding':faces[0]=faces[0,::-1]
    if mutation=='duplicate_face':faces=np.vstack([faces,faces[0]])
    with pytest.raises(RuntimeError,match='改变'):
        compacted_source_indices(source,positions,faces)


def test_native_clustered_unwrap_keeps_hard_edge_normals_without_whole_mesh_atlas(monkeypatch):
    import sys
    from types import SimpleNamespace
    import numpy as np
    import trimesh
    from hi3d_workbench.engines.mesh_uv import ensure_uv
    source=trimesh.creation.box();source.unmerge_vertices()
    source.vertex_normals=np.repeat(source.face_normals,3,axis=0)
    # Keep the two triangles of each planar side connected, while retaining
    # distinct hard-edge normals between sides. Fully disconnected triangles
    # intentionally use the CPU atlas because the native adjacency grid is empty.
    source.merge_vertices(merge_norm=False)
    assert len(source.vertices)==24
    original=[np.asarray(value).copy() for value in (source.vertices,source.faces,source.vertex_normals)]
    permutation=np.random.default_rng(11).permutation(len(source.vertices))
    inverse=np.argsort(permutation)
    compact_faces=inverse[source.faces]
    # The native xatlas result may introduce additional UV seam vertices.
    vmap=compact_faces.ravel()
    uv_faces=np.arange(len(vmap)).reshape(-1,3)
    uv=np.concatenate([np.array([[.01,.01],[.08,.01],[.01,.08]]) + [i%4*.2,i//4*.2] for i in range(len(uv_faces))])
    class Tensor:
        def __init__(self,value):self.value=np.asarray(value)
        def cuda(self):return self
        def detach(self):return self
        def cpu(self):return self
        def numpy(self):return self.value
    class Native:
        def init(self,*args):pass
        def uv_unwrap(self,**kwargs):
            return tuple(Tensor(value) for value in
                (source.vertices[permutation][vmap],uv_faces,uv,vmap))
        def read(self):return Tensor(source.vertices[permutation]),Tensor(compact_faces)
    def forbidden_atlas():raise AssertionError('A few hard edges must not select unclustered whole-mesh xatlas')
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(from_numpy=Tensor,cuda=SimpleNamespace(empty_cache=lambda:None)))
    monkeypatch.setitem(sys.modules,'cumesh',SimpleNamespace(CuMesh=Native,Atlas=forbidden_atlas))
    from hi3d_workbench.engines import native_uv
    monkeypatch.setattr(native_uv,'unwrap',lambda handler,**kw:handler.uv_unwrap(return_vmaps=True))
    result,policy=ensure_uv(source)
    expected=permutation[vmap]
    assert policy=='native_CuMesh_topology_mapped'
    assert np.array_equal(result.vertices,source.vertices[expected])
    assert np.array_equal(result.vertex_normals,source.vertex_normals[expected])
    assert np.array_equal(result.faces,uv_faces)
    assert all(np.array_equal(before,after) for before,after in
        zip(original,(source.vertices,source.faces,source.vertex_normals)))


def test_unused_zero_normals_do_not_trigger_hard_edge_atlas_fallback():
    import numpy as np
    import trimesh
    from hi3d_workbench.engines.mesh_uv import has_split_normals
    source=trimesh.creation.box()
    mesh=trimesh.Trimesh(np.vstack([source.vertices,[8,9,10]]),source.faces,process=False)
    assert np.array_equal(mesh.vertex_normals[-1],[0,0,0])
    assert not has_split_normals(mesh)


@pytest.mark.parametrize('kind',['collapsed','partially_collapsed','tiled','overlapping'])
def test_unusable_uv_is_rebuilt_before_ai_baking(tmp_path,kind):
    import numpy as np
    import trimesh
    from hi3d_workbench.asset_pipeline import load_texture_input
    source=tmp_path/'bad-uv.glb'
    mesh=trimesh.creation.icosphere(subdivisions=1)
    uv=np.zeros((len(mesh.vertices),2))
    if kind=='partially_collapsed':uv[0]=[.5,.5];uv[1]=[1,0]
    if kind=='tiled':uv=np.asarray(mesh.vertices)[:,:2]*2
    if kind=='overlapping':
        mesh.unmerge_vertices()
        uv=np.tile([[0.,0.],[1.,0.],[0.,1.]],(len(mesh.faces),1))
    mesh.visual=trimesh.visual.TextureVisuals(uv=uv)
    mesh.export(source)
    prepared,policy=load_texture_input(source)
    assert policy=='generate_uv' and getattr(prepared.visual,'uv',None) is None


def test_release_bundle_declarations_require_the_exact_engine_dependency_closure():
    from tools.audit_release import verify_bundle
    from hi3d_workbench.common import VERSION
    baseline={'background','coarse','texture','texture_encoder'}
    components=baseline|{'coarse_trellis2','trellis_structure'}
    bundle={'version':VERSION,'models_included':True,'pipeline_runtime':True,
        'extension_components':list(components),'extra_engine_selections':['engine:coarse:trellis2']}
    state={'ready':True,'features':['trellis_geometry']}
    assert verify_bundle(bundle,components,state)['declared_components']==sorted(components)
    with pytest.raises(AssertionError,match='覆盖声明'):
        verify_bundle({**bundle,'full_engine_coverage':True},components,state)
    with pytest.raises(AssertionError,match='组件'):
        verify_bundle(bundle,baseline,state)
    with pytest.raises(AssertionError,match='版本'):
        verify_bundle({**bundle,'version':'0.0.0'},components,state)
    with pytest.raises(AssertionError,match='特性'):
        verify_bundle(bundle,components,{'ready':True,'features':[]})
