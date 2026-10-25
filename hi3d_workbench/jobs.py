from __future__ import annotations
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from .common import ROOT, DATA, atomic_json, read_json, config, child_path
from .events import iter_events

WORKER_MODULES={'hi3d_workbench.worker','hi3d_workbench.pipeline_worker','hi3d_workbench.engines.run_stage'}

def _finite_number(value):
    try:return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
    except OverflowError:return False

def _event_progress(event):
    """Read observed work, never estimate overall completion or remaining time."""
    if 'progress_current' in event or 'progress_total' in event:
        current,total=event.get('progress_current'),event.get('progress_total')
        unit=event.get('progress_unit','items')
        if total is None and _finite_number(current) and current>=0:
            if not isinstance(unit,str) or not unit or len(unit)>32:unit='items'
            return {'current':current,'unit':unit}
    else:
        current,total=event.get('step'),event.get('num_inference_steps',event.get('steps'))
        unit='steps'
    if not (_finite_number(current) and _finite_number(total) and total>0 and 0<=current<=total):return {}
    if not isinstance(unit,str) or not unit or len(unit)>32:unit='items'
    return {'current':current,'total':total,'unit':unit}

def _terminal_text(value):
    # Worker messages may include uploaded filenames. Keep them on one console
    # line and do not permit ANSI / terminal control sequences.
    return ''.join(char if char.isprintable() else ' ' for char in str(value))[:500]

def _worker_request(command):
    if not any(module in command for module in WORKER_MODULES):return None
    try:return command[command.index('--request')+1]
    except (ValueError,IndexError):return None

def _job_requests(request_path):
    return {str(request_path),*(str(request_path.parent/'stages'/stage/'request.json') for stage in ('coarse','refine','texture'))}

def _kill_worker_tree(pid, created, request_path):
    """Kill only the recorded worker identity, never an unrelated reused PID."""
    import psutil
    if not isinstance(pid,int) or isinstance(pid,bool) or pid<=0 or not isinstance(created,(int,float)) or isinstance(created,bool) or not math.isfinite(created):return False
    try:
        proc=psutil.Process(pid)
        command=proc.cmdline()
        if not created or abs(proc.create_time()-created)>.01 or _worker_request(command)!=str(request_path):
            return False
        children=proc.children(recursive=True)
        for child in children:
            try:child.kill()
            except psutil.NoSuchProcess:pass
        proc.kill()
        _,alive=psutil.wait_procs([*children,proc],timeout=5)
        return not alive
    except psutil.NoSuchProcess:
        return True

class JobManager:
    def __init__(self):
        self.root=DATA/"jobs"
        self.root.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock()
        self.active=None
        self.process=None
        self.process_created=None
        self.cancelled=set()
        self._memory_states={}
        self._dirty=set()
        self.recovery_errors=[]
        self._recovery_workers=[]
        self._progress_writes={}
        self._console_progress={}

    def _record_error(self,path,exc):
        message=f'任务 {path.parent.name} 的记录无法读取，已跳过历史条目：{exc}'
        if message not in self.recovery_errors:self.recovery_errors.append(message)

    @staticmethod
    def _valid_state(value,jid):
        if not isinstance(value,dict):raise ValueError('任务记录不是 JSON 对象')
        if value.get('id')!=jid or not isinstance(value.get('created'),(int,float)) or isinstance(value.get('created'),bool) or not math.isfinite(value['created']):raise ValueError('缺少有效的任务 ID / 创建时间')
        if not isinstance(value.get('state'),str) or value['state'] not in {'queued','running','cancelling','succeeded','failed','cancelled','interrupted'}:raise ValueError('任务状态无效')
        if not isinstance(value.get('options',{}),dict):raise ValueError('任务参数不是 JSON 对象')
        # Old imported records may omit options; an empty object is readable
        # and lets the UI show its documented defaults without crashing.
        return {**value,'options':value.get('options',{})}

    def _remember_unowned_worker(self,request_path,snapshot=None):
        import psutil
        found=False
        for proc in snapshot if snapshot is not None else psutil.process_iter(['pid','cmdline','create_time']):
            try:
                command=proc.info['cmdline'] or []
                if _worker_request(command) in _job_requests(request_path) and isinstance(proc.info['create_time'],(int,float)):
                    identity=(proc.pid,proc.info['create_time'])
                    # A startup snapshot can still contain a worker that we
                    # have just safely killed. Recheck its live identity before
                    # reporting a blocker, and reject a reused PID.
                    try:
                        current=psutil.Process(proc.pid)
                        if abs(current.create_time()-identity[1])>.01 or not current.is_running() or current.status()==psutil.STATUS_ZOMBIE:continue
                    except psutil.NoSuchProcess:continue
                    except psutil.AccessDenied:pass  # Conservatively retain a known matching snapshot.
                    if identity not in self._recovery_workers:self._recovery_workers.append(identity)
                    found=True
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        return found

    def _cleanup_stage_workers(self,request_path,snapshot=None):
        """Reap recorded stages even when their supervisor has already exited."""
        import psutil
        for stage in ('coarse','refine','texture'):
            stage_dir=request_path.parent/'stages'/stage
            try:
                identity=read_json(stage_dir/'process.json',{})
                if not isinstance(identity,dict):continue
                _kill_worker_tree(identity.get('pid'),identity.get('created'),stage_dir/'request.json')
            except psutil.AccessDenied:
                # A known identity which becomes unreadable cannot safely free
                # the GPU slot. Do not guess permission to kill another PID.
                pid,created=identity.get('pid'),identity.get('created')
                if isinstance(pid,int) and not isinstance(pid,bool) and pid>0 and isinstance(created,(int,float)) and not isinstance(created,bool) and math.isfinite(created):
                    if (pid,created) not in self._recovery_workers:self._recovery_workers.append((pid,created))
            except (OSError,ValueError,psutil.Error):pass
        return self._remember_unowned_worker(request_path,snapshot)

    @property
    def recovery_blocked(self):
        import psutil
        alive=[]
        for pid,created in self._recovery_workers:
            try:
                if abs(psutil.Process(pid).create_time()-created)<=.01:alive.append((pid,created))
            except psutil.AccessDenied:alive.append((pid,created))
            except psutil.NoSuchProcess:pass
        self._recovery_workers=alive
        return bool(alive)

    @property
    def recovery_workers(self):
        self.recovery_blocked
        return [{'pid':pid,'created':created} for pid,created in self._recovery_workers]
    def recover(self):
        import psutil
        records=list(self.root.glob('*/state.json'))
        snapshot=list(psutil.process_iter(['pid','cmdline','create_time'])) if records else []
        for p in records:
            try:
                value=self._valid_state(read_json(p,{}),p.parent.name)
            except (OSError,ValueError) as exc:
                self._record_error(p,exc)
                self._remember_unowned_worker(p.parent/'request.json',snapshot)
                continue
            incomplete=value['state'] in {'running','queued','cancelling'}
            pid=value.get('worker_pid')
            if incomplete:
                if pid:
                    try:
                        _kill_worker_tree(pid,value.get('worker_created'),p.parent/'request.json')
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                # Pipeline children have independent pinned runtimes. A parent
                # may exit before restart, so also reap recorded stage identities.
                self._cleanup_stage_workers(p.parent/'request.json',snapshot)
            # Popen may succeed before worker_pid can be persisted. An earlier
            # restart can already have marked that record interrupted, and a
            # terminal write can precede final process cleanup. In every such
            # window, block a live exact-request worker without guessing that a
            # terminal/missing identity authorizes killing it.
            if incomplete or value['state']=='interrupted' or pid:
                if self._remember_unowned_worker(p.parent/'request.json',snapshot):
                    message=f'任务 {p.parent.name} 的残留工作进程无法安全确认 / 关闭，暂时阻止新的 GPU 操作'
                    if message not in self.recovery_errors:self.recovery_errors.append(message)
            if incomplete:
                value.update(state="interrupted",message="工作台重新启动，未完成任务没有被自动续跑")
                try:atomic_json(p,value)
                except (OSError,ValueError) as exc:
                    self._memory_states[p.parent.name]=value
                    self._dirty.add(p.parent.name)
                    self._record_error(p,exc)

    def get(self,jid):
        if jid in self._memory_states:return self._memory_states[jid].copy()
        path=child_path(self.root,jid)/"state.json"
        try:
            value=read_json(path)
            return self._valid_state(value,jid) if value is not None else None
        except (OSError,ValueError) as exc:
            self._record_error(path,exc)
            raise ValueError('此任务的记录损坏或不可读，请查看工作台诊断信息') from exc

    def list(self):
        values={}
        for path in self.root.glob('*/state.json'):
            try:
                value=self.get(path.parent.name)
                if value is None:raise ValueError('任务记录不存在')
                values[path.parent.name]=value
            except (OSError,ValueError) as exc:self._record_error(path,exc)
        values.update(self._memory_states)
        return sorted(values.values(),key=lambda value:value['created'],reverse=True)

    def create(self,image,mesh,options):
        with self.lock:
            if self.recovery_blocked:raise RuntimeError('检测到记录损坏或身份不明的残留工作进程，请先关闭它，再创建新任务')
            if self.active:raise RuntimeError("已有推理任务运行，请等待或取消")
            jid=uuid.uuid4().hex
            folder=self.root/jid
            folder.mkdir()
            try:
                shutil.copy2(image,folder/"input.png")
                if mesh is not None:shutil.copy2(mesh,folder/"coarse.glb")
                atomic_json(folder/"request.json",options)
                state={"id":jid,"created":time.time(),"state":"queued","stage":"queued","options":options,"message":"任务已保存"}
                atomic_json(folder/"state.json",state)
            except Exception:
                # No thread or worker owns this unpublished directory yet.
                # Resolve the fresh UUID child before recursive cleanup; never
                # remove a preexisting job or a redirected path.
                if not folder.is_symlink() and folder.resolve().parent==self.root.resolve():
                    shutil.rmtree(folder,ignore_errors=True)
                raise
            self.active=jid
            self._memory_states[jid]=state.copy()
            try:threading.Thread(target=self._run,args=(jid,),daemon=True).start()
            except Exception as exc:
                try:self._update(jid,state='failed',stage='failed',message=f'无法启动任务线程：{exc}')
                except (OSError,ValueError):pass
                finally:
                    self.active=None
                    if jid not in self._dirty:self._memory_states.pop(jid,None)
                raise RuntimeError('无法启动任务线程，请重试或查看系统资源') from exc
            return state

    def _update(self,jid,_persist=True,**kw):
        with self.lock:
            state=self.get(jid)
            now=time.time()
            if 'stage' in kw and kw['stage']!=state.get('stage'):
                state.update(stage_started_at=now,phase_started_at=now,last_progress_at=now,progress={})
            state.update(kw,updated=now)
            self._memory_states[jid]=state.copy()
            if _persist:
                try:atomic_json(self.root/jid/"state.json",state)
                except (OSError,ValueError):
                    self._dirty.add(jid)
                    raise
                else:self._dirty.discard(jid)
            self._show_progress(jid,state)

    def _record_event(self,jid,event):
        """Coalesce frequent callbacks while keeping the latest API state live."""
        with self.lock:
            state=self.get(jid)
            now=time.time();monotonic=time.monotonic()
            old=state.get('progress') or {}
            phase=event.get('phase')
            phase=phase if isinstance(phase,str) and phase else None
            progress={'stage':event['stage'],**_event_progress(event)}
            if isinstance(event.get('pipeline_stage'),str) and event['pipeline_stage'] in {'coarse','refine','texture'}:progress['pipeline_stage']=event['pipeline_stage']
            if phase:progress['phase']=phase
            for field in ('pass_index','num_passes'):
                if _finite_number(event.get(field)) and event[field]>=0:progress[field]=event[field]
            boundary=any(old.get(field)!=progress.get(field) for field in ('stage','phase','pass_index'))
            # Legacy samplers announce a second pass with a total but no step.
            # This is a phase boundary rather than permission to reuse 100%.
            boundary=boundary or ('current' in old and 'current' not in progress and ('steps' in event or 'num_inference_steps' in event))
            if isinstance(event.get('device'),str) and event['device'] in {'cpu','cuda'}:progress['device']=event['device']
            elif not boundary and old.get('device') in {'cpu','cuda'}:progress['device']=old['device']
            changed=boundary or any(old.get(key)!=progress.get(key) for key in ('current','total','unit','device')) or event.get('message')!=(state.get('event') or {}).get('message')
            values={'stage':event['stage'],'event':event,'progress':progress,'last_event_at':now,
                    'message':event.get('message',event['stage'])}
            if boundary:values['phase_started_at']=now
            if changed:values['last_progress_at']=now
            finished='total' in progress and progress['current']==progress['total']
            persist=boundary or finished or monotonic-self._progress_writes.get(jid,float('-inf'))>=.5
            self._update(jid,_persist=persist,**values)
            if persist:self._progress_writes[jid]=monotonic

    def _show_progress(self,jid,state):
        """Mirror meaningful job progress to the visible launcher console."""
        progress=state.get('progress') or {}
        identity=(state.get('state'),state.get('stage'),progress.get('phase'),progress.get('pass_index'))
        previous=self._console_progress.get(jid)
        now=time.monotonic()
        number=(progress.get('current'),progress.get('total'),progress.get('unit'))
        message=state.get('message','')
        signature=(identity,number,message,progress.get('device'))
        if previous and previous[2]==signature:return
        finished=progress.get('total') is not None and progress.get('current')==progress['total']
        if previous and previous[1]==identity and not finished and now-previous[0]<1:return
        pieces=[f'[Hi3D {jid[:8]}]',str(state.get('stage') or state.get('state',''))]
        if progress.get('phase'):pieces.append(str(progress['phase']))
        if isinstance(progress.get('pass_index'),int):pieces.append(f'第 {progress["pass_index"]+1} 次采样')
        if progress.get('device'):pieces.append('CPU' if progress['device']=='cpu' else 'CUDA / GPU')
        if progress.get('total') is not None:
            current,total=progress['current'],progress['total']
            pieces.append(f'{current:g}%' if progress['unit']=='percent' and total==100 else f'{current:g}/{total:g} {progress["unit"]}')
        elif progress.get('current') is not None:
            pieces.append(f'已合并 {progress["current"]:g}' if progress['unit']=='merges' else f'已处理 {progress["current"]:g} {progress["unit"]}')
        elapsed=max(0,time.time()-state.get('stage_started_at',state['created']))
        pieces.extend((f'阶段 {elapsed:.0f}s',str(message)))
        try:
            if sys.stdout is not None:print(' · '.join(_terminal_text(piece) for piece in pieces),flush=True)
        except (OSError,UnicodeError,ValueError):pass  # Closing a console must not corrupt a live job.
        self._console_progress[jid]=(now,identity,signature)

    def _run(self,jid):
        folder=self.root/jid
        process=None
        created=None
        pipeline=False
        try:
            cfg=config()
            cmd=cfg["worker_command"] or [str((ROOT/cfg["worker_python"]).resolve())]
            options=self.get(jid)['options']
            pipeline=options.get('mode','refine')!='refine' or options.get('generate_texture',False)
            module='hi3d_workbench.pipeline_worker' if pipeline else 'hi3d_workbench.worker'
            # A custom command is a trusted local configuration, never HTTP input.
            with self.lock:
                if jid in self.cancelled:return
                process=subprocess.Popen([*cmd,"-u","-m",module,"--request",str(folder/"request.json")],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",env={**os.environ,"PYTHONIOENCODING":"utf-8"},creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
                self.process=process
                import psutil
                try:created=psutil.Process(process.pid).create_time()
                except psutil.NoSuchProcess:pass  # A short process may already have exited.
                self.process_created=created
                self._update(jid,state="running",message="正在启动推理进程",worker_pid=process.pid,worker_created=created)
            with (folder/"worker.log").open("w",encoding="utf-8") as log:
                for line in process.stdout:
                    log.write(line);log.flush()
                    for event in iter_events(line):
                        self._record_event(jid,event)
            code=process.wait()
            if jid in self.cancelled:
                self._update(jid,state="cancelled",stage="cancelled",message="已停止；输入和日志已保留")
            elif code==0 and (folder/"report.json").is_file():
                report=read_json(folder/'report.json')
                if not isinstance(report,dict) or report.get('status')!='succeeded':
                    raise ValueError('进程未提供有效的成功报告，请查看阶段日志')
                expected='textured.glb' if options.get('mode')=='texture' or options.get('generate_texture',options.get('mode')=='image_to_asset') else 'refined.glb' if options.get('mode','refine')=='refine' or options.get('refine_geometry',False) else 'coarse.glb'
                if not (folder/expected).is_file():raise RuntimeError('进程退出但缺少预期结果文件')
                if expected=='textured.glb':
                    from .asset_pipeline import inspect_textured_glb
                    inspect_textured_glb(folder/expected)
                    if not report.get('texture_generated'):raise ValueError('贴图报告未确认生成成功')
                self._update(jid,state="succeeded",stage="complete",report=report,message="模型与贴图已完成" if expected=='textured.glb' else "模型已完成")
            else:
                try:fail=read_json(folder/"failure.json",{})
                except (OSError,ValueError):fail={}
                if not isinstance(fail,dict):fail={}
                message=fail.get('error')
                if not isinstance(message,str) or not message.strip():message=f"进程退出 {code}，请查看日志"
                fail={**fail,'error':message}
                self._update(jid,state="failed",stage="failed",failure=fail,message=message)
        except Exception as e:
            try:self._update(jid,state="failed",stage="failed",message=str(e))
            except (OSError,ValueError):pass
        finally:
            # A parser, log or state write can fail while CUDA is still running.
            # Keep the active slot until the actual child has exited.
            if process is not None:
                if process.poll() is None:
                    try:_kill_worker_tree(process.pid,created,folder/'request.json')
                    except Exception:pass
                    if process.poll() is None:
                        try:self._update(jid,state='cancelling',message='等待工作进程退出后才会释放任务；可再次点击停止')
                        except Exception:pass
                process.wait()
                if process.stdout:process.stdout.close()
            # Supervisor crashes and cancellation between stage transitions
            # can leave a child outside the earlier tree snapshot. Never free
            # the GPU gate until this final exact-request scan is recorded.
            if pipeline:
                try:
                    if self._cleanup_stage_workers(folder/'request.json'):
                        message=f'任务 {jid} 的阶段进程尚未退出，已阻止新的 GPU 操作'
                        if message not in self.recovery_errors:self.recovery_errors.append(message)
                except Exception as exc:
                    self.recovery_errors.append(f'任务 {jid} 的阶段清理未完成：{exc}')
                    self._remember_unowned_worker(folder/'request.json')
            with self.lock:
                try:
                    if jid in self.cancelled and self.get(jid)["state"] not in {"cancelled","succeeded"}:
                        self._update(jid,state="cancelled",message="已停止")
                except (OSError,ValueError):pass
                finally:
                    self.process=None
                    self.process_created=None
                    self.active=None
                    self.cancelled.discard(jid)
                    self._progress_writes.pop(jid,None)
                    self._console_progress.pop(jid,None)
                    if jid not in self._dirty:self._memory_states.pop(jid,None)

    def cancel(self,jid):
        with self.lock:
            if jid!=self.active:return self.get(jid)
            self.cancelled.add(jid)
            try:self._update(jid,state="cancelling",message="已请求取消，正在释放进程")
            except (OSError,ValueError):pass  # A full/read-only disk must not prevent stopping CUDA.
            if self.process and self.process.poll() is None:
                _kill_worker_tree(self.process.pid,self.process_created,self.root/jid/'request.json')
            return self.get(jid)
