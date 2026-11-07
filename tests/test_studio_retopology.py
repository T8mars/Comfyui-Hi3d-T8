import copy
import json
import shutil

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from test_workbench import client, testroot
from test_studio_parts import source_model
from test_studio_skin import rig_two, provider
from hi3d_workbench.common import digest
from hi3d_workbench.studio import material_transfer
from hi3d_workbench.studio.retopology import visible_mesh
from hi3d_workbench.studio import retopology
from hi3d_workbench.studio.scene import instances
from hi3d_workbench.studio.skin import bind, extract, posed_vertices
from hi3d_workbench.studio.uv_atlas import tangent_vectors


def test_missing_surface_guard_rejects_major_loss_but_reports_small_deviations():
    retopology.validate_coverage({'source_vertex_samples': 2338, 'vertices_over_3_percent_extent': 3})
    with pytest.raises(RuntimeError, match='原表面'):
        retopology.validate_coverage({'source_vertex_samples': 446986, 'vertices_over_3_percent_extent': 152247})


def test_retopology_uses_visible_skin_pose_without_modifying_original(tmp_path):
    source = source_model(tmp_path / 'source.glb', instance=True)
    rigged, _ = bind(source, rig_two(), provider)
    joint = extract(rigged).nodes[1]
    rigged.tree['nodes'][joint]['rotation'] = Rotation.from_euler('z', 37, degrees=True).as_quat().tolist()
    original_tree, original_binary = copy.deepcopy(rigged.tree), bytes(rigged.binary)
    expected = np.concatenate(posed_vertices(rigged))
    mesh = visible_mesh(rigged)
    offset, faces = 0, []
    for row in instances(rigged):
        faces.append(row.faces + offset)
        offset += len(row.positions)
    np.testing.assert_allclose(mesh.vertices[mesh.faces], expected[np.concatenate(faces)], atol=2e-6)
    assert rigged.tree == original_tree and bytes(rigged.binary) == original_binary


def test_retopology_retains_instance_morphs_and_mirrored_winding(tmp_path):
    source = source_model(tmp_path / 'source.glb', instance=True)
    rows = instances(source)
    source.tree['nodes'][rows[-1].node]['scale'] = [-2, .5, 3]
    for mesh in source.tree['meshes']:
        mesh['weights'] = [.6]
        for primitive in mesh['primitives']:
            count = source.tree['accessors'][primitive['attributes']['POSITION']]['count']
            primitive['targets'] = [{'POSITION': source.add_accessor(np.tile([.1, .2, -.1], (count, 1)))}]
    expected_vertices, expected_faces, offset = [], [], 0
    for row in instances(source):
        expected_vertices.append((row.positions + [.06, .12, -.06]) @ row.world[:3, :3].T + row.world[:3, 3])
        expected_faces.append((row.faces[:, [0, 2, 1]] if np.linalg.det(row.world[:3, :3]) < 0 else row.faces) + offset)
        offset += len(row.positions)
    mesh = visible_mesh(source)
    np.testing.assert_allclose(mesh.vertices[mesh.faces],
        np.concatenate(expected_vertices)[np.concatenate(expected_faces)], atol=1e-7)


def test_retopology_welds_uv_seams_without_merging_nearby_distinct_surfaces():
    from hi3d_workbench.studio.gltf import Document
    document = Document({'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0]}],
                         'nodes': [{'mesh': 0}], 'meshes': [{'primitives': []}]})
    positions = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0],
                          [0, 0, 0], [1, 1, 0], [0, 1, 0],
                          [0, 0, .000001], [1, 0, .000001], [0, 1, .000001]], dtype=np.float32)
    primitive = {'attributes': {'POSITION': document.add_accessor(positions)},
                 'indices': document.add_accessor(np.arange(9, dtype=np.uint32))}
    document.tree['meshes'][0]['primitives'] = [primitive]
    mesh = visible_mesh(document)
    assert len(mesh.vertices) == 7 and len(mesh.faces) == 3
    assert len(set(mesh.faces[0]) & set(mesh.faces[1])) == 2
    assert not (set(mesh.faces[0]) & set(mesh.faces[2]))
    np.testing.assert_array_equal(mesh.vertices[mesh.faces].reshape(-1, 3), positions)


def test_atlas_tangents_are_orthogonal_and_keep_mirrored_handedness():
    positions = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2]])
    normals = np.tile([0., 0, 1], (3, 1))
    uv = positions[:, :2].copy()
    normal = tangent_vectors(positions, faces, uv, normals)
    uv[:, 1] *= -1
    mirrored = tangent_vectors(positions, faces, uv, normals)
    np.testing.assert_allclose(normal, np.tile([1., 0, 0, 1], (3, 1)))
    np.testing.assert_allclose(mirrored, np.tile([1., 0, 0, -1], (3, 1)))
    degenerate = tangent_vectors(positions, faces, np.zeros((3, 2)), normals)
    assert np.isfinite(degenerate).all()
    np.testing.assert_allclose((degenerate[:, :3] * normals).sum(1), 0, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(degenerate[:, :3], axis=1), 1, atol=1e-7)


def test_native_geometry_retry_reuses_only_matching_valid_output(tmp_path, monkeypatch):
    monkeypatch.setattr(retopology.common, 'DATA', tmp_path / 'data')
    geometry = 'v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n'
    source, executable = tmp_path / 'input.obj', tmp_path / 'autoremesher.exe'
    source.write_text(geometry)
    executable.write_bytes(b'fixture executable')
    invocations = []
    def produce(command, payload, directory):
        invocations.append(payload)
        __import__('pathlib').Path(payload['output']).write_text(geometry)
    monkeypatch.setattr(retopology, 'run_stage', produce)
    def run(index, target=100, edge=1):
        stage = tmp_path / f'stage-{index}'
        stage.mkdir()
        payload = {'input': str(source), 'output': str(stage / 'quads.obj'),
                   'executable': str(executable), 'target_quads': target, 'sharp_angle': 60, 'adaptivity': 1,
                   'edge_scaling': edge}
        return retopology.remesh_geometry(payload, stage, lambda *a, **kw: None)
    assert not run(0) and run(1) and len(invocations) == 1
    cached = next((tmp_path / 'data/studio-cache/retopology').glob('*/quads.obj'))
    cached.write_text('v 0 0 0\n')
    manifest = cached.parent / 'manifest.json'
    saved = json.loads(manifest.read_text()); saved['sha256'] = digest(cached)
    manifest.write_text(json.dumps(saved))
    assert not run(2) and run(3) and len(invocations) == 2
    assert not run(4, target=200) and len(invocations) == 3
    executable.write_bytes(b'new native version')
    assert not run(5) and len(invocations) == 4
    assert not run(6, edge=2) and len(invocations) == 5


def test_quadriflow_cache_tracks_blender_seed_surface_and_sharp_edges(tmp_path,monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(retopology.common,'DATA',tmp_path/'data')
    source=tmp_path/'input.obj';source.write_text('v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n')
    executable=tmp_path/'blender.exe';executable.write_bytes(b'fixture blender')
    monkeypatch.setattr(retopology,'blender_command',lambda:[str(executable),'fixture args'])
    invocations=[]
    def produce(command,payload,directory):
        assert command[0]==str(executable) and payload['operation']=='quadriflow_remesh'
        shutil.copy2(source,payload['output']);Path(payload['report']).write_text('{"quads":1}')
        invocations.append(payload)
    monkeypatch.setattr(retopology,'run_stage',produce)
    def run(index,**options):
        stage=tmp_path/f'stage-{index}';stage.mkdir()
        payload={'operation':'quadriflow_remesh','input':str(source),'output':str(stage/'quads.obj'),
            'report':str(stage/'report.json'),'target_quads':1000,'seed':0,'preserve_sharp':False,'solid_surface':False,**options}
        hit=retopology.remesh_geometry(payload,stage,lambda *a,**k:None)
        assert json.loads(Path(payload['report']).read_text())['quads']==1
        return hit
    assert not run(0) and run(1) and len(invocations)==1
    assert not run(2,seed=23) and len(invocations)==2
    assert not run(3,preserve_sharp=True) and len(invocations)==3
    assert not run(4,solid_surface=True) and len(invocations)==4
    executable.write_bytes(b'updated blender')
    assert not run(5) and len(invocations)==5


def test_real_blender_tessellation_keeps_a_concave_polygon_opening(tmp_path,monkeypatch):
    import os
    import subprocess
    import trimesh
    from pathlib import Path
    from hi3d_workbench.studio.processes import blender_command
    if not os.environ.get('HI3D_STUDIO_RUNTIME'):
        monkeypatch.setenv('HI3D_STUDIO_RUNTIME',str(Path(__file__).resolve().parents[1]/'runtime-assets'))
    try:command=blender_command()
    except RuntimeError:pytest.skip('Optional Blender runtime is not installed')
    source=tmp_path/'concave.obj';target=tmp_path/'triangles.obj';request=tmp_path/'request.json'
    points=np.array([[0,0,0],[3,0,0],[3,3,0],[2,3,0],[2,1,0],[1,1,0],[1,3,0],[0,3,0.]])
    source.write_text(''.join('v '+' '.join(map(str,p))+'\n' for p in points)+'f 1 2 3 4 5 6 7 8\n')
    request.write_text(json.dumps({'operation':'tessellate_retopology','input':str(source),
        'output':str(target),'report':str(tmp_path/'report.json')}),encoding='utf-8')
    subprocess.run([*command,'--request',str(request)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True,timeout=60)
    mesh=trimesh.load(target,force='mesh',process=False)
    assert len(mesh.faces)==6
    np.testing.assert_allclose(mesh.area,7,atol=1e-8)
    centers=mesh.triangles_center
    assert not np.any((centers[:,0]>1)&(centers[:,0]<2)&(centers[:,1]>1))
    np.testing.assert_array_equal(np.unique(mesh.vertices,axis=0),np.unique(points,axis=0))


def test_api_forwards_retopology_engine_and_surface_choices(client,tmp_path,monkeypatch):
    from hi3d_workbench import server
    from hi3d_workbench.studio import api
    source=tmp_path/'source.glb';source_model(source)
    upload=client.post('/api/assets',files={'file':('source.glb',source.read_bytes())})
    assert upload.status_code==200,upload.text
    monkeypatch.setattr(api.models,'status',lambda:{'features':{'retopology':{'ready':True}}})
    monkeypatch.setattr(api.models,'download_busy',lambda:False)
    calls=[]
    def submit(image,path,options):
        assert image is None
        calls.append((path,options));return {'id':'fixture','options':options}
    monkeypatch.setattr(server.manager,'create',submit)
    body={'source':'asset:'+upload.json()['id'],'action':'retopology','retopology_engine':'quadriflow',
        'retopology_surface':'solid','material_projection':'cage','preserve_sharp':True,'target_quads':15000,'uv_backend':'gpu'}
    response=client.post('/api/studio/jobs',json=body)
    assert response.status_code==200,response.text
    for key,value in body.items():assert calls[0][1][key]==value
    assert calls[0][1]['mode']=='studio' and calls[0][0].is_file()
    for key,value in [('retopology_engine','unknown'),('retopology_surface','unknown'),('material_projection','unknown'),('uv_backend','unknown'),('edge_scaling',.1)]:
        bad=client.post('/api/studio/jobs',json={**body,key:value})
        assert bad.status_code==422 and len(calls)==1


@pytest.mark.skipif(not __import__('pathlib').Path('runtime-assets/meshoptimizer.dll').is_file(), reason='native backend is optional')
def test_native_proxy_bounds_error_keeps_full_resolution_donor_and_small_inputs(monkeypatch):
    import trimesh
    runtime = __import__('pathlib').Path('runtime-assets').resolve()
    monkeypatch.setattr(retopology, 'runtime_root', lambda: runtime)
    mesh = trimesh.creation.icosphere(subdivisions=4)
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    proxy, report = retopology.geometry_proxy(mesh, lambda *a, **kw: None, face_limit=2000, max_error=.01)
    assert 0 < len(proxy.faces) <= 2000
    assert 0 < report['normalized_error'] <= .01
    np.testing.assert_array_equal(mesh.vertices, vertices)
    np.testing.assert_array_equal(mesh.faces, faces)
    assert len(mesh.faces) == report['source_faces'] and len(proxy.faces) == report['inference_faces']
    small, report = retopology.geometry_proxy(proxy, lambda *a, **kw: None, face_limit=2000)
    assert small is proxy and report['normalized_error'] == 0


@pytest.mark.parametrize('damage', ['truncated_glb', 'missing_metadata', 'invalid_glb_with_matching_hash', 'zero_uv_with_matching_hash'])
def test_uv_cache_reuses_only_valid_content_and_recovers_corrupt_state(tmp_path, monkeypatch, damage):
    source_path = tmp_path / 'source.glb'
    source_model(source_path)
    geometry = tmp_path / 'quads.obj'
    geometry.write_text('v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n')
    monkeypatch.setattr(material_transfer.common, 'DATA', tmp_path / 'data')
    layouts = []
    def create_geometry(source, output, **kwargs):
        layouts.append(output)
        shutil.copy2(source_path, output)
        return {'faces': 2, 'charts': 1}
    def project(source, target, output, **kwargs):
        shutil.copy2(target, output)
        return {'material_transfer': 'fixture', 'notice': 'fixture'}
    monkeypatch.setattr(material_transfer, 'create', create_geometry)
    monkeypatch.setattr(material_transfer, 'transfer', project)
    def run(index):
        return material_transfer.run(source_path, geometry, tmp_path / f'output-{index}.glb',
                                     tmp_path / f'stage-{index}', texture_size=256)
    assert not run(0)['uv']['cache_hit']
    assert run(1)['uv']['cache_hit'] and len(layouts) == 1
    cached = layouts[0]
    manifest = cached.parent / 'manifest.json'
    saved = json.loads(manifest.read_text())
    if damage == 'missing_metadata':
        saved.pop('uv')
    elif damage == 'zero_uv_with_matching_hash':
        from hi3d_workbench.studio.gltf import Document
        document=Document.load(cached)
        primitive=next(document.primitives())[2]
        old=document.accessor(primitive['attributes']['TEXCOORD_0'])
        primitive['attributes']['TEXCOORD_0']=document.add_accessor(np.zeros_like(old))
        document.save(cached)
        saved['sha256']=digest(cached)
    else:
        cached.write_bytes(b'incomplete atlas')
        if damage == 'invalid_glb_with_matching_hash':
            saved['sha256'] = digest(cached)
    manifest.write_text(json.dumps(saved))
    assert not run(2)['uv']['cache_hit'] and len(layouts) == 2
    assert run(3)['uv']['cache_hit'] and len(layouts) == 2
