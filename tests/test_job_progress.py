"""Observed task progress, independent of CUDA and the user's active jobs."""
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

@pytest.fixture
def progress_manager(tmp_path,monkeypatch):
    from hi3d_workbench import jobs
    monkeypatch.setattr(jobs,'DATA',tmp_path/'data')
    manager=jobs.JobManager()
    jid='progress-fixture'
    state={'id':jid,'created':90.,'state':'running','stage':'queued','options':{}}
    jobs.atomic_json(manager.root/jid/'state.json',state)
    clock=[100.]
    monkeypatch.setattr(jobs.time,'time',lambda:clock[0])
    monkeypatch.setattr(jobs.time,'monotonic',lambda:clock[0])
    return jobs,manager,jid,clock


@pytest.mark.parametrize('event,expected',[
    ({'step':0,'steps':25},{'current':0,'total':25,'unit':'steps'}),
    ({'step':2,'num_inference_steps':12},{'current':2,'total':12,'unit':'steps'}),
    ({'progress_current':25.5,'progress_total':100,'progress_unit':'percent'},{'current':25.5,'total':100,'unit':'percent'}),
    ({'progress_current':3,'progress_total':8,'progress_unit':'charts'},{'current':3,'total':8,'unit':'charts'}),
    ({'progress_current':17,'progress_unit':'merges'},{'current':17,'unit':'merges'}),
    ({'step':True,'steps':10},{}),({'step':2,'steps':0},{}),
    ({'step':11,'steps':10},{}),({'step':float('nan'),'steps':10},{}),
    ({'step':2,'steps':float('inf')},{}),({'progress_current':-1,'progress_total':100},{}),
    ({'step':10**400,'steps':10**401},{}),
])
def test_observed_counters_accept_real_work_only(event,expected):
    from hi3d_workbench import jobs
    assert jobs._event_progress(event)==expected


def test_phase_boundaries_clear_previous_progress_and_device(progress_manager,capsys):
    jobs,manager,jid,clock=progress_manager
    manager._record_event(jid,{'stage':'texture_sampling','phase':'color','device':'cuda','step':25,'steps':25})
    clock[0]+=4
    manager._record_event(jid,{'stage':'texture_unwrapping','phase':'charts','device':'cpu','progress_current':0,'progress_total':100,'progress_unit':'percent'})
    state=manager.get(jid)
    assert state['progress']=={'stage':'texture_unwrapping','phase':'charts','device':'cpu','current':0,'total':100,'unit':'percent'}
    assert state['stage_started_at']==state['phase_started_at']==state['last_progress_at']==104
    clock[0]+=2
    manager._record_event(jid,{'stage':'texture_unwrapping','phase':'packing','message':'pack'})
    state=manager.get(jid)
    assert state['progress']=={'stage':'texture_unwrapping','phase':'packing'}
    assert state['stage_started_at']==104 and state['phase_started_at']==106
    output=capsys.readouterr().out
    assert 'CUDA / GPU' in output and 'CPU' in output and '0%' in output and 'packing' in output


def test_repeated_heartbeat_is_not_effective_progress(progress_manager):
    jobs,manager,jid,clock=progress_manager
    event={'stage':'texture_unwrapping','phase':'chart','message':'running','device':'cpu'}
    manager._record_event(jid,event)
    clock[0]+=10
    manager._record_event(jid,event)
    state=manager.get(jid)
    assert state['last_event_at']==110 and state['last_progress_at']==100
    assert 'current' not in state['progress']
    manager._record_event(jid,{**event,'progress_current':1,'progress_total':5,'progress_unit':'charts'})
    assert manager.get(jid)['last_progress_at']==110


def test_invalid_metadata_does_not_claim_a_device_or_break_worker_ownership(progress_manager):
    jobs,manager,jid,clock=progress_manager
    manager._record_event(jid,{'stage':'sampling','device':['cuda'],'pipeline_stage':{},'step':0,'steps':2})
    assert manager.get(jid)['progress']=={'stage':'sampling','current':0,'total':2,'unit':'steps'}


def test_absolute_counts_update_effective_progress_without_an_invented_total(progress_manager,capsys):
    jobs,manager,jid,clock=progress_manager
    event={'stage':'texture_unwrapping','phase':'uv_charts','device':'cuda','progress_current':17,'progress_unit':'merges'}
    manager._record_event(jid,event)
    clock[0]+=2
    manager._record_event(jid,{**event,'progress_current':29})
    state=manager.get(jid)
    assert state['progress']['current']==29 and 'total' not in state['progress']
    assert state['last_progress_at']==102
    output=capsys.readouterr().out
    assert '已合并 29' in output and '%' not in output


def test_baking_fallback_without_device_clears_preceding_cuda_label(progress_manager):
    jobs,manager,jid,clock=progress_manager
    manager._record_event(jid,{'stage':'texture_baking','phase':'texture_raster','device':'cuda'})
    clock[0]+=1
    manager._record_event(jid,{'stage':'texture_baking','phase':'texture_bake','message':'GPU 光栅化与 CPU 贴图补边'})
    assert 'device' not in manager.get(jid)['progress']
    assert 'current' not in manager.get(jid)['progress']
    clock[0]+=1
    manager._record_event(jid,{'stage':'texture_baking','phase':'texture_inpaint','device':'cpu','progress_current':0,'progress_total':4,'progress_unit':'textures'})
    assert manager.get(jid)['progress']['device']=='cpu'
    assert manager.get(jid)['progress']['current']==0


def test_rapid_callbacks_coalesce_disk_and_terminal_but_keep_live_state(progress_manager,monkeypatch,capsys):
    jobs,manager,jid,clock=progress_manager
    original=jobs.atomic_json;writes=[]
    def record(path,state):
        writes.append(state.copy());original(path,state)
    monkeypatch.setattr(jobs,'atomic_json',record)
    for step in range(10):
        clock[0]=100+step*.01
        manager._record_event(jid,{'stage':'sampling','device':'cuda','step':step,'steps':10})
    assert len(writes)==1
    assert manager.get(jid)['progress']['current']==9
    assert len(capsys.readouterr().out.splitlines())==1
    clock[0]=100.1
    manager._record_event(jid,{'stage':'sampling','step':10,'steps':10})
    assert len(writes)==2
    assert jobs.read_json(manager.root/jid/'state.json')['progress']['current']==10
    assert '10/10 steps' in capsys.readouterr().out


def test_sampling_pass_restart_never_keeps_old_100_percent(progress_manager):
    jobs,manager,jid,clock=progress_manager
    manager._record_event(jid,{'stage':'sampling','step':25,'steps':25,'device':'cuda'})
    clock[0]+=1
    manager._record_event(jid,{'stage':'sampling','steps':25,'message':'second branch'})
    assert manager.get(jid)['progress']=={'stage':'sampling'}
    assert manager.get(jid)['phase_started_at']==101
    clock[0]+=1
    manager._record_event(jid,{'stage':'sampling','step':0,'steps':25})
    assert manager.get(jid)['progress']['current']==0


def test_console_escape_sequences_and_closed_output_do_not_interrupt_job(progress_manager,monkeypatch,capsys):
    jobs,manager,jid,clock=progress_manager
    manager._record_event(jid,{'stage':'loading','message':'file\n\x1b[2Jname'})
    output=capsys.readouterr().out
    assert '\x1b' not in output and len(output.splitlines())==1
    class Closed:
        def write(self,_):raise BrokenPipeError('closed console')
        def flush(self):pass
    monkeypatch.setattr(jobs.sys,'stdout',Closed())
    clock[0]+=1
    manager._record_event(jid,{'stage':'sampling','step':1,'steps':2})
    assert manager.get(jid)['stage']=='sampling'


@pytest.mark.parametrize('nested',[False,True])
def test_worker_stdout_reaches_api_state_and_visible_terminal(tmp_path,monkeypatch,capsys,nested):
    from hi3d_workbench import jobs
    monkeypatch.setattr(jobs,'ROOT',tmp_path)
    monkeypatch.setattr(jobs,'DATA',tmp_path/'data')
    script=tmp_path/'progress_worker.py'
    script.write_text('from pathlib import Path\nimport json,sys,time\n'
        'folder=Path(sys.argv[-1]).parent\n'
        +('folder=folder.parent.parent\n' if nested else '')+
        'print("HI3D_EVENT invalid-json",flush=True)\n'
        'print(\'HI3D_EVENT {"stage":"sampling","step":NaN}\',flush=True)\n'
        'sys.stderr.write("\\rSampling 4/8 [progress display] ");sys.stderr.flush()\n'
        'print(\'HI3D_EVENT {"stage":"texture_unwrapping","phase":"charts","device":"cpu","progress_current":4,"progress_total":8,"progress_unit":"charts","message":"chart fixture"} [tail display]\\x1b[0m\',flush=True)\n'
        'while not (folder/"continue.fixture").exists():time.sleep(.01)\n'
        '(folder/"refined.glb").write_bytes(b"fixture result")\n'
        '(folder/"report.json").write_text(json.dumps({"status":"succeeded"}))\n',encoding='utf-8')
    command_script=script
    if nested:
        command_script=tmp_path/'pipeline_supervisor.py'
        command_script.write_text('from pathlib import Path\nimport json,sys\n'
            f'sys.path.insert(0,{str(Path(__file__).parents[1])!r})\n'
            'from hi3d_workbench.pipeline_worker import run_child\n'
            'folder=Path(sys.argv[-1]).parent\n'
            'stage=folder/"stages"/"texture";stage.mkdir(parents=True)\n'
            '(stage/"request.json").write_text("{}")\n'
            f'run_child([sys.executable,{str(script)!r}],stage/"request.json",stage,"texture")\n',encoding='utf-8')
    monkeypatch.setattr(jobs,'config',lambda:{'worker_command':[sys.executable,str(command_script)]})
    manager=jobs.JobManager()
    image=tmp_path/'image.png';image.write_bytes(b'fixture')
    mesh=tmp_path/'coarse.glb';mesh.write_bytes(b'fixture')
    jid=manager.create(image,mesh,{'resolution':512})['id']
    def wait_for(predicate):
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            if predicate():return
            time.sleep(.01)
        assert predicate(),'isolated fixture did not report progress'
    try:
        wait_for(lambda:manager.get(jid).get('progress',{}).get('current')==4)
        assert manager.active==jid and manager.process.poll() is None
        state=manager.get(jid)
        assert state['progress']['device']=='cpu' and state['progress']['total']==8
        if nested:assert state['progress']['pipeline_stage']=='texture'
        console=[]
        def console_received():
            console.append(capsys.readouterr().out)
            return '4/8 charts' in ''.join(console)
        # The state becomes visible just before its console write. Wait for
        # that same event rather than racing the worker thread's final print.
        wait_for(console_received)
        (manager.root/jid/'continue.fixture').touch()
        wait_for(lambda:manager.active is None)
        assert manager.get(jid)['state']=='succeeded'
        assert manager.get(jid)['progress']=={}
        log=(manager.root/jid/'worker.log').read_text(encoding='utf-8')
        assert 'HI3D_EVENT invalid-json' in log and '[progress display]' in log and '[tail display]' in log
        from hi3d_workbench.events import iter_events
        assert len(list(iter_events(log)))==1
        if nested:
            raw=(manager.root/jid/'stages/texture/stage.log').read_text(encoding='utf-8')
            assert '[progress display] HI3D_EVENT {' in raw and '} [tail display]' in raw
            assert '"pipeline_stage"' not in raw
    finally:
        if manager.active:
            manager.cancel(jid)
            wait_for(lambda:manager.active is None)


def test_browser_progress_renders_supported_counters_without_stale_phase_or_eta(tmp_path):
    node=shutil.which('node')
    if not node:pytest.skip('Node.js is required for browser progress logic')
    app=(Path(__file__).parents[1]/'hi3d_workbench/web/app.js').read_text(encoding='utf-8')
    functions=app[app.index('function duration('):app.index('function renderStages(')]
    script=tmp_path/'progress.cjs'
    script.write_text("""
const assert=require('node:assert/strict');
const elements={};
const $=key=>elements[key] ||= {hidden:false,textContent:'',setAttribute(name,value){this[name]=value;}};
const activeStates=new Set(['running','queued','cancelling']);
function validMetric(value){return typeof value==='number' && Number.isFinite(value) && value>=0;}
"""+functions+r"""
const base={state:'running',stage:'texture_sampling',created:100,stage_started_at:120,phase_started_at:130,last_progress_at:150};
renderJobProgress({...base,event:{stage:'texture_sampling',step:0,steps:25}},160);
assert.equal(elements['#progress'].hidden,false);
assert.equal(elements['#progress'].value,0);
assert.match(elements['#progress-count'].textContent,/0 \/ 25 步/);
renderJobProgress({...base,progress:{stage:'texture_unwrapping',phase:'packing',current:12,total:100,unit:'percent',device:'cpu'}},160);
assert.equal(elements['#progress-count'].textContent,'12%');
assert.match(elements['#progress-meta'].textContent,/当前执行：CPU/);
assert.match(elements['#progress-meta'].textContent,/本阶段 40 秒/);
assert.match(elements['#progress-meta'].textContent,/距上次有效进展 10 秒/);
renderJobProgress({...base,progress:{stage:'texture_unwrapping',phase:'uv_charts',current:129,unit:'merges',device:'cuda'}},160);
assert.equal(elements['#progress'].hidden,true);
assert.equal(elements['#progress-count'].textContent,'已合并 129');
assert.match(elements['#progress-meta'].textContent,/合并 UV 分片/);
assert.match(elements['#progress-note'].textContent,/尚无总量/);
renderJobProgress({...base,progress:{stage:'texture_sampling',phase:'structure',current:2,total:12,unit:'steps',pass_index:0,device:'cuda'}},160);
assert.match(elements['#progress-count'].textContent,/第 1 次采样 · 2 \/ 12 步/);
assert.match(elements['#progress-meta'].textContent,/生成空间结构/);
renderJobProgress({...base,progress:{stage:'texture_encoding_mesh',phase:'voxelize',device:'cpu'}},160);
assert.equal(elements['#progress'].hidden,true);
assert.match(elements['#progress-meta'].textContent,/CPU 网格体素化与编码准备/);
assert.doesNotMatch(elements['#progress-meta'].textContent,/当前执行：CUDA/);
renderJobProgress({...base,progress:{stage:'texture_encoding_mesh',phase:'shape_encoding',device:'cuda'}},160);
assert.match(elements['#progress-meta'].textContent,/CUDA 模型表面编码/);
renderJobProgress({...base,progress:{stage:'texture_baking',phase:'texture_raster',device:'cuda'}},160);
assert.match(elements['#progress-meta'].textContent,/GPU UV光栅化/);
assert.match(elements['#progress-meta'].textContent,/当前执行：CUDA/);
renderJobProgress({...base,progress:{stage:'texture_baking',phase:'texture_inpaint',device:'cpu',current:0,total:4,unit:'textures'}},160);
assert.equal(elements['#progress-count'].textContent,'0 / 4 张贴图');
assert.match(elements['#progress-meta'].textContent,/CPU贴图补边/);
assert.match(elements['#progress-meta'].textContent,/当前执行：CPU/);
renderJobProgress({...base,progress:{stage:'texture_baking',phase:'texture_bake'}},160);
assert.match(elements['#progress-meta'].textContent,/材质烘焙与补边/);
assert.doesNotMatch(elements['#progress-meta'].textContent,/当前执行：CPU|当前执行：CUDA/);
assert.equal(elements['#progress'].hidden,true);
renderJobProgress({...base,progress:{},event:{stage:'texture_sampling',step:25,steps:25}},160);
assert.equal(elements['#progress'].hidden,true);
assert.equal(elements['#progress-count'].textContent,'');
assert.doesNotMatch(elements['#progress-meta'].textContent,/CPU|GPU/);
assert.match(elements['#progress-note'].textContent,/未提供可量化进度/);
renderJobProgress({...base,state:'succeeded',updated:160,progress:{current:25,total:25,unit:'steps',device:'cuda'}},160);
assert.equal(elements['#progress'].hidden,true);
assert.equal(elements['#progress-note'].hidden,true);
assert.doesNotMatch(Object.values(elements).map(e=>e.textContent).join(' '),/剩余|预计|ETA/);
console.log('progress browser checks passed');
""",encoding='utf-8')
    result=subprocess.run([node,str(script)],capture_output=True,text=True,encoding='utf-8',check=False)
    assert result.returncode==0,result.stdout+result.stderr
    html=(Path(__file__).parents[1]/'hi3d_workbench/web/index.html').read_text(encoding='utf-8')
    for element in ('progress-count','progress-meta','progress-note'):
        assert f'id="{element}"' in html
