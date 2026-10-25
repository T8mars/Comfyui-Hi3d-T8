import threading
from pathlib import Path
import shutil
import subprocess

import pytest
from test_workbench import client
from hi3d_workbench import common


@pytest.mark.parametrize('previous',[None,b'previous-complete-glb'])
def test_reader_never_observes_partially_copied_artifact(tmp_path,monkeypatch,previous):
    source=tmp_path/'source';source.write_bytes(b'new complete model')
    target=tmp_path/'coarse.glb'
    if previous is not None:target.write_bytes(previous)
    copying=threading.Event();resume=threading.Event();errors=[]
    def slow_copy(src,dst):
        Path(dst).write_bytes(b'partial');copying.set()
        assert resume.wait(5)
        Path(dst).write_bytes(Path(src).read_bytes())
    monkeypatch.setattr(common.shutil,'copyfile',slow_copy)
    def publish():
        try:common.atomic_copy(source,target)
        except Exception as exc:errors.append(exc)
    thread=threading.Thread(target=publish);thread.start()
    try:
        assert copying.wait(5)
        assert target.read_bytes()==previous if previous is not None else not target.exists()
    finally:resume.set();thread.join(5)
    assert not errors and target.read_bytes()==source.read_bytes()
    assert not list(tmp_path.glob('*.copy-tmp'))


def test_completed_same_volume_immutable_artifact_needs_no_second_copy(tmp_path,monkeypatch):
    source=tmp_path/'stage.glb';source.write_bytes(b'complete immutable stage')
    target=tmp_path/'served.glb'
    monkeypatch.setattr(common.shutil,'copyfile',lambda *args:pytest.fail('unexpected full artifact copy'))
    common.atomic_copy(source,target,immutable=True)
    assert target.read_bytes()==source.read_bytes()


def test_preview_retry_does_not_revert_url_or_reload_every_poll(tmp_path):
    node=shutil.which('node')
    if not node:pytest.skip('Node required')
    app=(Path(__file__).parents[1]/'hi3d_workbench/web/app.js').read_text(encoding='utf-8')
    function=app[app.index('function setView('):app.index('function resetResult(')]
    script=tmp_path/'preview.cjs'
    script.write_text("""
const assert=require('node:assert/strict');
const view={dataset:{},src:null,getAttribute(){return this.src},setAttribute(k,v){this.src=v},removeAttribute(){this.src=null}};
const empty={hidden:true,querySelector(){return {textContent:''}}};
const $=id=>id==='#before'?view:empty;
"""+function+"""
setView('before','/model.glb');
view.setAttribute('src','/model.glb?preview_retry=123');
setView('before','/model.glb');
assert.equal(view.src,'/model.glb?preview_retry=123');
setView('before','/another.glb');assert.equal(view.src,'/another.glb');
setView('before',null);assert.equal(view.src,null);assert.equal(view.dataset.sourceUrl,undefined);
""",encoding='utf-8')
    result=subprocess.run([node,str(script)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_preview_errors_retry_bounded_and_ignore_obsolete_sources(tmp_path):
    node=shutil.which('node')
    if not node:pytest.skip('Node required')
    app=(Path(__file__).parents[1]/'hi3d_workbench/web/app.js').read_text(encoding='utf-8')
    handlers=app[app.index("for (const view of [$('#before'), $('#after')]) {"):app.index('function syncView(')]
    script=tmp_path/'errors.cjs'
    script.write_text("""
const assert=require('node:assert/strict');let timers=[];let toasts=[];
const setTimeout=f=>timers.push(f),toast=text=>toasts.push(text),post=()=>Promise.resolve({});
const elements={};
for(const id of ['before','after']){
 const labels={b:{textContent:''},p:{textContent:''}};
 elements['#'+id+'-empty']={hidden:true,querySelector:s=>labels[s]};
 elements['#'+id]={id,dataset:{sourceUrl:'/first.glb'},src:'/first.glb',events:{},loaded:false,
  hasAttribute(){return !!this.src},getAttribute(){return this.src},setAttribute(k,v){this.src=v},
  addEventListener(k,f){this.events[k]=f}};
}
const $=selector=>elements[selector];
"""+handlers+"""
const view=$('#before'),error={detail:{type:'loadfailure'}};
view.events.error(error);assert.equal(timers.length,1);timers.shift()();assert.match(view.src,/preview_retry=/);
view.events.error(error);timers.shift()();view.events.error(error);
assert.equal(view.dataset.previewRetries,'2');assert.equal(timers.length,0);assert.equal(toasts.length,1);
view.dataset.previewRetries='0';view.events.error(error);view.dataset.sourceUrl='/new.glb';view.src='/new.glb';
timers.shift()();assert.equal(view.src,'/new.glb');
""",encoding='utf-8')
    result=subprocess.run([node,str(script)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_invalid_normals_are_repaired_without_changing_geometry_or_valid_normals():
    import numpy as np
    import trimesh
    from hi3d_workbench.asset_pipeline import repair_invalid_normals
    mesh=trimesh.creation.icosphere(subdivisions=1)
    vertices=mesh.vertices.copy();faces=mesh.faces.copy();normals=mesh.vertex_normals.copy()
    original=normals.copy();normals[0]=0;normals[1]=float('nan')
    mesh.vertex_normals=normals
    result=repair_invalid_normals(mesh)
    assert result['invalid_repaired']==2
    assert np.array_equal(mesh.vertices,vertices) and np.array_equal(mesh.faces,faces)
    assert np.array_equal(mesh.vertex_normals[2:],original[2:])
    assert np.allclose(np.linalg.norm(mesh.vertex_normals,axis=1),1)


def test_preview_status_reports_only_local_models_and_is_authenticated(client):
    from hi3d_workbench import server
    server.PREVIEW_EVENTS.clear()
    payload={'view':'before','source':'/api/jobs/example/files/coarse.glb','status':'loaded'}
    assert client.post('/api/preview-status',json=payload,headers={'x-hi3d-token':''}).status_code==403
    assert client.post('/api/preview-status',json={**payload,'source':'https://outside.example/model.glb'}).status_code==400
    assert client.post('/api/preview-status',json=payload).status_code==200
    events=client.get('/api/preview-status').json()
    assert len(events)==1 and events[0]['status']=='loaded'
