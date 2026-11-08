"""Execute actual named-view draft, history and submission handlers in Node."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_view_drafts_submission_and_history_never_mix_directions(tmp_path):
    node=shutil.which('node')
    if not node:pytest.skip('Node.js required')
    source=(Path(__file__).parents[1]/'hi3d_workbench/web/app.js').read_text(encoding='utf-8')
    generation=source[source.index('function generationOptions('):source.index("$('#run').onclick")]
    draft=source[source.index('function draftWith('):source.index("$('#new-task').onclick")]
    views=source[source.index('function renderViewInputs('):source.index('function stageName(')]
    script=tmp_path/'multiview-ui.cjs'
    script.write_text(r"""
const assert=require('node:assert/strict');const fields={};
const $=id=>fields[id] ||= {value:'',checked:false,src:null,
  getAttribute(name){return this[name]},removeAttribute(name){delete this[name]}};
let selectedMode='image_to_asset',inputImage={id:'front'},inputMesh=null,currentJob=null;
let epoch=0,shownJob=null,inputViews={left:{id:'left',url:'/left'},back:{id:'back',url:'/back'},right:null};
const viewLabels={left:'左侧',back:'背面',right:'右侧'};
const inputBusy=()=>false,needsMesh=()=>selectedMode!=='image_to_asset',wantsRefine=()=>false,wantsTexture=()=>true;
const hasJobFile=(job,name)=>job.available_files.includes(name),resetResult=()=>{},renderDraft=()=>{};
"""+generation+draft+views+r"""
$('#geometry-engine').value='hunyuan_mv';
assert.deepEqual(generationOptions().view_images,{left:'left',back:'back'});
draftWith({id:'edited-front'},null);assert.equal(inputViews.left.id,'left');
renderViewInputs();assert.equal($('#multiview-inputs').hidden,false);
assert.equal($('#view-left-thumb').src,'/left');assert.equal($('#view-left-download').hidden,true);
$('#geometry-engine').value='hunyuan21';assert.deepEqual(generationOptions().view_images,{});
renderViewInputs();assert.equal($('#multiview-inputs').hidden,true);
$('#geometry-engine').value='hunyuan_mv';selectedMode='texture';inputMesh={id:'mesh'};
assert.deepEqual(generationOptions().view_images,{});
selectedMode='image_to_asset';currentJob={id:'saved',options:{view_images:{left:'old',back:'old'}},available_files:['view-left.png']};
renderViewInputs();assert.equal($('#view-left-thumb').src,'/api/jobs/saved/files/view-left.png');
assert.equal($('#view-left-download').hidden,false);assert.equal($('#view-left-file').disabled,true);
assert.equal($('#view-back-thumb').hidden,true,'a missing saved view must not show a draft from another object');
assert.equal($('#view-back-name').textContent,'原图已缺失');
draftWith({id:'reused-front'},null,{right:{id:'reused-right',url:'/new-right'}});
assert.deepEqual(generationOptions().view_images,{right:'reused-right'});
draftWith(null,null);assert.deepEqual(inputViews,{left:null,back:null,right:null});
""",encoding='utf-8')
    result=subprocess.run([node,str(script)],capture_output=True,text=True,encoding='utf-8',timeout=10)
    assert result.returncode==0,result.stdout+result.stderr
