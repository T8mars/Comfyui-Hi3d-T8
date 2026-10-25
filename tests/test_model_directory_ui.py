"""Actual model-directory UI handlers with mocked network and DOM, no server mutation."""
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest


def test_model_directory_controls_preserve_drafts_guard_busy_and_use_real_contract(tmp_path):
    node=shutil.which('node')
    if not node:pytest.skip('Node.js is required for browser handler checks')
    web=Path(__file__).parents[1]/'hi3d_workbench/web'
    source=(web/'app.js').read_text(encoding='utf-8')
    handlers=source[source.index('function modelPathBusy('):source.index('function availability(')]
    script=tmp_path/'model-directory.cjs'
    script.write_text(r"""
const assert=require('node:assert/strict');
const fields={};
const $=selector=>fields[selector] ||= {value:'',textContent:'',disabled:false};
let online=true,pending=null,modelPath=null,modelPathDirty=false,modelPathMessage='',modelPathUnsupported=false;
let modelPathRevision=0;
let gpu=false,checking=false,downloading=false;
const gpuBusy=()=>gpu,checkingModels=()=>checking,downloadingModels=()=>downloading;
let calls=[],notices=[],errors=[],nextFailure=null;
const defaultRoot=String.raw`E:\Workbench\models`;
const meta={models_dir:String.raw`E:\Shared\Hi3D`,resolved_path:String.raw`E:\Shared\Hi3D`,default_path:defaultRoot,extensions_path:String.raw`E:\Shared\Hi3D\extensions`,is_default:false,exists:true,restart_required:false,busy:false,busy_reason:null};
let serverMeta={...meta};
let deferredRead=null;
async function api(path){assert.equal(path,'/api/model-path');return deferredRead || {...serverMeta};}
async function post(path,body){
  calls.push({path,body});
  if(nextFailure){const error=nextFailure;nextFailure=null;throw error;}
  assert.equal(path,'/api/model-path');
  assert.deepEqual(Object.keys(body),['models_dir']);
  serverMeta=body.models_dir===null ? {...meta,models_dir:'',resolved_path:defaultRoot,is_default:true} : {...meta,models_dir:body.models_dir,resolved_path:body.models_dir};
  return {...serverMeta};
}
function availability(){renderModelPathControls();}
async function refresh(){renderModelPath(serverMeta);}
function toast(text){notices.push(text);}
function showError(error){errors.push(error.message);}
"""+handlers+r"""
async function main(){
  renderModelPath(meta);
  assert.equal(calls.length,0,'reading settings must not save or reset a path');
  assert.equal($('#models-directory-current').textContent,meta.resolved_path);
  assert.equal($('#models-directory-configured').textContent,meta.models_dir);
  assert.equal($('#models-directory-input').value,meta.models_dir);
  assert.equal($('#models-directory-save').disabled,true);
  assert.match($('#models-directory-state').textContent,/立即生效/);
  const draft=String.raw`E:\中文 模型\Hi3D`;
  $('#models-directory-input').value=draft;
  $('#models-directory-input').oninput();
  renderModelPath({...meta,resolved_path:String.raw`E:\Other\Models`});
  assert.equal($('#models-directory-input').value,draft,'polling must preserve an edited path');
  assert.equal($('#models-directory-current').textContent,String.raw`E:\Other\Models`);
  assert.equal($('#models-directory-save').disabled,false);
  for(const reason of ['gpu','checking','downloading','backend']){
    gpu=reason==='gpu';checking=reason==='checking';downloading=reason==='downloading';
    modelPath={...meta,busy:reason==='backend',busy_reason:'已有任务运行'};
    renderModelPathControls();
    assert.equal($('#models-directory-save').disabled,true,reason);
    await saveModelPath();await saveModelPath(true);
    assert.equal(calls.length,0,'busy state must never POST');
  }
  gpu=checking=downloading=false;modelPath={...meta};
  nextFailure=Object.assign(new Error('已有任务运行，目录未变'),{status:409});
  await saveModelPath();
  assert.equal(calls.length,1);
  assert.equal(calls[0].body.models_dir,draft);
  assert.equal(modelPath.models_dir,meta.models_dir,'a rejected save must not replace saved metadata');
  assert.equal($('#models-directory-input').value,draft);
  assert.equal(modelPathDirty,true);
  assert.match($('#models-directory-state').textContent,/目录未变/);
  await saveModelPath();
  assert.equal(modelPath.models_dir,draft);
  assert.equal(modelPathDirty,false);
  assert.equal(pending,null);
  assert.match(notices.at(-1),/未移动/);
  let finishRead;
  deferredRead=new Promise(resolve=>{finishRead=resolve;});
  const staleRead=refreshModelPath();
  $('#models-directory-input').value='newer models';
  $('#models-directory-input').oninput();
  await saveModelPath();
  finishRead({...meta});
  await staleRead;deferredRead=null;
  assert.equal($('#models-directory-input').value,'newer models','an older in-flight read must not undo a successful save');
  assert.equal(modelPath.models_dir,'newer models');
  await saveModelPath(true);
  assert.deepEqual(calls.at(-1),{path:'/api/model-path',body:{models_dir:null}});
  assert.equal($('#models-directory-current').textContent,defaultRoot);
  assert.equal($('#models-directory-input').value,'models');
  assert.equal($('#models-directory-reset').disabled,true);
  assert.equal($('#models-directory-configured').textContent,'默认（models）');
  renderModelPath({...meta,models_dir:'future models',resolved_path:String.raw`E:\Workbench\future models`,exists:false},true);
  assert.match($('#models-directory-state').textContent,/尚不存在/);
  $('#models-directory-input').value='  future models  ';
  $('#models-directory-input').oninput();
  await saveModelPath();
  assert.equal(calls.at(-1).body.models_dir,'future models');
  const before=calls.length;
  $('#models-directory-input').value='   ';
  $('#models-directory-input').oninput();
  assert.equal($('#models-directory-save').disabled,true);
  await saveModelPath();
  assert.equal(calls.length,before,'blank input uses the explicit reset action instead');
  console.log('model-directory UI checks passed');
}
main().catch(error=>{console.error(error);process.exitCode=1;});
""",encoding='utf-8')
    result=subprocess.run([node,str(script)],capture_output=True,text=True,encoding='utf-8',check=False)
    assert result.returncode==0,result.stdout+result.stderr


def test_model_directory_markup_has_unique_labeled_controls():
    web=Path(__file__).parents[1]/'hi3d_workbench/web'
    class Inventory(HTMLParser):
        def __init__(self):super().__init__();self.ids=[];self.label_targets=[]
        def handle_starttag(self,tag,attrs):
            values=dict(attrs)
            if 'id' in values:self.ids.append(values['id'])
            if tag=='label' and 'for' in values:self.label_targets.append(values['for'])
    parser=Inventory();html=(web/'index.html').read_text(encoding='utf-8');parser.feed(html)
    assert len(parser.ids)==len(set(parser.ids))
    for field in ('models-directory-current','models-directory-configured','models-directory-input','models-directory-save','models-directory-reset','models-directory-state'):
        assert field in parser.ids
    assert 'models-directory-input' in parser.label_targets
    assert '不会自动移动现有模型' in html and '相对工作台根目录' in html
