"""Run the actual studio result/selection handlers with controlled DOM and I/O."""
import shutil
import subprocess
from pathlib import Path

import pytest


WEB = Path(__file__).parents[1] / 'hi3d_workbench/web/studio.js'


def run_node(tmp_path, source, checks):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js required for studio handler regressions')
    script = tmp_path / 'studio-results.cjs'
    script.write_text(r"""
const assert=require('node:assert/strict');
const watchdog=setTimeout(()=>{console.error('Handler test did not finish');process.exitCode=1;},5000);
const fields={};
const $=id=>fields[id] ||= {value:'',textContent:'',hidden:false,children:[],
  replaceChildren(...children){this.children=children;}};
const document={createElement:()=>({})};
function Option(text,value){return {text,value};}
const activeStates=new Set(['queued','running','cancelling']);
const actions={complete_parts:'部件补全'};
let currentJob=null,resultUrl=null,sourceUrl=null,sourceInfo=null,loadedUrl=null;
let selectionRevision=0,sourceRevision=0,comparing=false,busy=false,polling=false,state=null;
let partEdits=[],boneEdits=new Map(),partAttachments=new Map(),constraints=[];
let renderCalls=[],models=[];
const availability=()=>{},message=()=>{},renderAttachments=()=>{},disposeModel=()=>{},
  drawBones=()=>{},renderParts=()=>{},renderMapping=()=>{},editCount=()=>{},renderConstraints=()=>{};
let api=async()=>{throw new Error('Unexpected API call');};
let fetch=async()=>({ok:true,text:async()=>''});
const showModel=async url=>{models.push(url);loadedUrl=url;return true;};
const setComparing=value=>{comparing=value;};
function job(id,state,available_files=[]){return {id,state,available_files,created:1,updated:2,
  message:state,options:{action:'complete_parts'},report:{input_faces:12,output_faces:24}};}
""" + source + '\n' + checks, encoding='utf-8')
    result = subprocess.run([node, str(script)], capture_output=True, text=True, encoding='utf-8', timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr


def result_handler():
    source = WEB.read_text(encoding='utf-8')
    return source[source.index('async function renderJob('):source.index('async function history(')]


def test_failed_cancelled_and_running_jobs_expose_only_published_artifacts(tmp_path):
    run_node(tmp_path, result_handler(), r"""
async function main(){
  const files=['original-segmentation.glb','original-parts.zip','worker.log','request.json'];
  for(const state of ['failed','cancelled','interrupted','running']){
    await renderJob(job('retained',state,files));
    const links=$('downloads').children;
    assert.equal(links.length,2,state+' must offer already-published original models');
    assert.deepEqual(links.map(link=>link.href),[
      '/api/studio/jobs/retained/files/original-segmentation.glb',
      '/api/studio/jobs/retained/files/original-parts.zip']);
    assert.equal(links.some(link=>link.textContent==='下载 GLB'),false);
    if(state!=='running')assert.doesNotMatch($('result-summary').textContent,/完成/);
  }
  resultUrl='/api/jobs/completed/files/output.glb';
  await renderJob(job('completed','succeeded',['output.glb','report.json']));
  assert.equal($('downloads').children.length,2);
  assert.match($('result-summary').textContent,/部件补全完成/);
  await renderJob(job('empty','failed'));
  assert.equal($('downloads').children.length,0,'opening a different failed task clears stale links');
}
main().catch(error=>{console.error(error);process.exitCode=1;}).finally(()=>clearTimeout(watchdog));
""")


def test_failed_completion_exposes_actual_diagnostics_without_a_finished_model(tmp_path):
    run_node(tmp_path, result_handler(), r"""
async function main(){
  const files=['part-coverage.json','xpart-generation.json','xpart-surface-preparation.json'];
  await renderJob(job('failed-parts','failed',files));
  const links=$('downloads').children;
  assert.deepEqual(links.map(link=>link.textContent),['部件外形检查','补全推理报告','补全表面检查']);
  assert.deepEqual(links.map(link=>link.href),files.map(name=>'/api/studio/jobs/failed-parts/files/'+name));
  assert.equal(links.some(link=>link.textContent==='下载 GLB'),false);
  await renderJob(job('no-report','failed'));
  assert.equal($('downloads').children.length,0);
}
main().catch(error=>{console.error(error);process.exitCode=1;}).finally(()=>clearTimeout(watchdog));
""")


def test_new_model_selection_invalidates_completed_result_and_inflight_poll(tmp_path):
    source = WEB.read_text(encoding='utf-8')
    selection = source[source.index('async function sourceChanged('):source.index("$('source').onchange=")]
    poll = source[source.index('async function poll('):source.index("window.addEventListener('hi3d:health'")]
    run_node(tmp_path, selection + poll, r"""
async function renderJob(value){renderCalls.push(value.id);currentJob=value;}
async function main(){
  const previous=job('previous','succeeded',['output.glb']);currentJob=previous;
  let finishPoll,notifyPoll;
  const pollStarted=new Promise(resolve=>notifyPoll=resolve);
  api=async path=>{
    if(path==='/api/studio/status')return {};
    if(path==='/api/jobs/previous')return new Promise(resolve=>{finishPoll=resolve;notifyPoll();});
    if(path.startsWith('/api/studio/source?'))return {url:'/new.glb',faces:12,materials:1,bones:[],parts:null};
    throw new Error('Unexpected API '+path);
  };
  const oldPoll=poll();await pollStarted;
  assert.equal(typeof finishPoll,'function');
  $('source').value='asset:new';await sourceChanged();
  assert.equal(currentJob,null,'an explicit new input must detach the completed result');
  assert.equal(loadedUrl,'/new.glb');
  finishPoll(previous);await oldPoll;
  assert.deepEqual(renderCalls,[],'a late poll cannot reopen the old completed task');
  assert.equal(currentJob,null);assert.equal(loadedUrl,'/new.glb');
  assert.equal($('progress-box').hidden,true);
}
main().catch(error=>{console.error(error);process.exitCode=1;}).finally(()=>clearTimeout(watchdog));
""")


def test_late_log_response_does_not_replace_the_selected_job(tmp_path):
    run_node(tmp_path, result_handler(), r"""
async function main(){
  const pending=new Map();
  fetch=path=>new Promise(resolve=>pending.set(path,resolve));
  const old=renderJob(job('old','failed'));
  const next=renderJob(job('new','cancelled'));
  pending.get('/api/jobs/new/log')({ok:true,text:async()=>'new task log'});await next;
  assert.equal($('log').textContent,'new task log');
  pending.get('/api/jobs/old/log')({ok:true,text:async()=>'obsolete log'});await old;
  assert.equal(currentJob.id,'new');
  assert.equal($('log').textContent,'new task log');
}
main().catch(error=>{console.error(error);process.exitCode=1;}).finally(()=>clearTimeout(watchdog));
""")


def test_downloads_do_not_wait_for_preview_and_obsolete_preview_is_ignored(tmp_path):
    run_node(tmp_path, result_handler(), r"""
async function main(){
  let finishPreview;
  api=()=>new Promise(resolve=>finishPreview=resolve);
  const pending=renderJob(job('saved','failed',['output.glb','original-parts.zip']));
  assert.equal($('downloads').children.length,2,'saved downloads must not wait for GLB parsing');
  currentJob=null;selectionRevision++;
  finishPreview({url:'/obsolete.glb',bones:[]});await pending;
  assert.deepEqual(models,[],'a model selected after the request must remain selected');
}
main().catch(error=>{console.error(error);process.exitCode=1;}).finally(()=>clearTimeout(watchdog));
""")


def test_old_source_list_response_preserves_a_new_selection(tmp_path):
    source = WEB.read_text(encoding='utf-8')
    refresh = source[source.index('async function refreshSources('):source.index('async function sourceChanged(')]
    run_node(tmp_path, refresh, r"""
async function main(){
  let finishList;
  api=()=>new Promise(resolve=>finishList=resolve);
  $('source').value='asset:old';
  const pending=refreshSources('asset:old');
  selectionRevision++;$('source').value='asset:new';
  finishList([{value:'asset:old',name:'old'},{value:'asset:new',name:'new'}]);await pending;
  assert.equal($('source').value,'asset:new');
}
main().catch(error=>{console.error(error);process.exitCode=1;}).finally(()=>clearTimeout(watchdog));
""")
