"""TRELLIS settings propagation without weights, runtime mutation, or CUDA."""
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def pipeline_settings(monkeypatch):
    # Keep browser-only checks independent of native scientific-library imports.
    from test_asset_pipeline import _fake_stages,_pipeline_request,pipeline_worker
    return _fake_stages(monkeypatch),_pipeline_request,pipeline_worker


@pytest.mark.parametrize('attention',['auto','flash_attn','sdpa'])
@pytest.mark.parametrize('faces',[100000,300000,1000000])
def test_pipeline_passes_attention_and_generated_face_budget_only(pipeline_settings,tmp_path,attention,faces):
    calls,request_for,pipeline_worker=pipeline_settings
    request=request_for(tmp_path,geometry_engine='trellis2',texture_engine='trellis2',
        geometry_target_faces=faces,pipeline_attention=attention)
    pipeline_worker.run(request)
    assert [stage for stage,_ in calls]==['coarse','texture']
    coarse,texture=(value for _,value in calls)
    assert coarse['target_faces']==faces
    assert coarse['pipeline_attention']==texture['pipeline_attention']==attention
    assert 'target_faces' not in texture and 'geometry_target_faces' not in texture
    assert texture['mesh_path']==str(tmp_path/'coarse.glb')


def test_pipeline_defaults_preserve_original_face_budget_and_auto_attention(pipeline_settings,tmp_path):
    calls,request_for,pipeline_worker=pipeline_settings
    pipeline_worker.run(request_for(tmp_path))
    assert calls[0][1]['target_faces']==1000000
    assert all(value['pipeline_attention']=='auto' for _,value in calls)


def test_existing_mesh_texture_request_never_receives_reduction_budget(pipeline_settings,tmp_path):
    import trimesh
    calls,request_for,pipeline_worker=pipeline_settings
    trimesh.creation.box().export(tmp_path/'coarse.glb')
    original=(tmp_path/'coarse.glb').read_bytes()
    request=request_for(tmp_path,mode='texture',geometry_target_faces=100000,
        geometry_cleanup=True,pipeline_attention='sdpa')
    pipeline_worker.run(request)
    assert len(calls)==1 and calls[0][0]=='texture'
    payload=calls[0][1]
    assert payload['pipeline_attention']=='sdpa'
    assert 'target_faces' not in payload and 'geometry_target_faces' not in payload
    assert (tmp_path/'coarse.glb').read_bytes()==original


def test_browser_settings_scope_restore_and_submission_are_independent(tmp_path):
    node=shutil.which('node')
    if not node:pytest.skip('Node.js is required for browser settings checks')
    web=Path(__file__).parents[1]/'hi3d_workbench/web'
    app=(web/'app.js').read_text(encoding='utf-8')
    scopes=app[app.index('function needsMesh('):app.index('function inputsReady(')]
    restore=app[app.index('function applyOptions('):app.index('function renderComparison(')]
    submit=app[app.index('function generationOptions('):app.index("$('#run').onclick =")]
    script=tmp_path/'settings.cjs'
    script.write_text(r"""
const assert=require('node:assert/strict');
const fields={};
const $=selector=>fields[selector] ||= {value:'',checked:false};
const modeLabels={image_to_asset:'图片建模',texture:'贴图',refine:'精修'};
let selectedMode='image_to_asset';
let inputImage={id:'image-fixture'},inputMesh={id:'mesh-fixture'};
function syncModeUI(){}
function syncEngines(){}
"""+scopes+restore+submit+r"""
applyOptions({mode:'image_to_asset',geometry_engine:'trellis2',texture_engine:'ultratex_int8',generate_texture:true,attention:'flash',geometry_cleanup:true});
assert.equal($('#geometry-target-faces').value,'1000000');
assert.equal($('#pipeline-attention').value,'auto');
assert.equal(usesTrellisPipeline(),true);
assert.equal(usesTrellisFaceBudget(),true);
$('#trellis-cleanup').checked=false;
assert.equal(usesTrellisFaceBudget(),false);
$('#trellis-cleanup').checked=true;
$('#pipeline-attention').value='sdpa';
$('#geometry-target-faces').value='300000';
let request=generationOptions();
assert.equal(request.pipeline_attention,'sdpa');
assert.equal(request.attention,'flash');
assert.equal(request.geometry_target_faces,300000);
assert.equal(request.mesh,null);
applyOptions({mode:'texture',texture_engine:'trellis2',pipeline_attention:'flash_attn',attention:'sdpa',geometry_target_faces:100000});
assert.equal(usesTrellisPipeline(),true);
assert.equal(usesTrellisFaceBudget(),false);
request=generationOptions();
assert.equal(request.mesh,'mesh-fixture');
assert.equal(request.pipeline_attention,'flash_attn');
assert.equal(request.attention,'sdpa');
applyOptions({mode:'texture',texture_engine:'ultratex_int8',pipeline_attention:'bad',geometry_target_faces:1});
assert.equal(usesTrellisPipeline(),false);
assert.equal($('#pipeline-attention').value,'auto');
assert.equal($('#geometry-target-faces').value,'1000000');
applyOptions({mode:'refine',generate_texture:false,geometry_engine:'trellis2'});
assert.equal(usesTrellisPipeline(),false);
assert.equal(usesTrellisFaceBudget(),false);
console.log('browser settings checks passed');
""",encoding='utf-8')
    result=subprocess.run([node,str(script)],capture_output=True,text=True,encoding='utf-8',check=False)
    assert result.returncode==0,result.stdout+result.stderr
    html=(web/'index.html').read_text(encoding='utf-8')
    assert 'id="pipeline-attention"' in html and 'id="geometry-target-faces"' in html
    assert '已有模型贴图不会减面' in html
    assert 'value="1000000" selected' in html and 'value="auto" selected' in html
