from __future__ import annotations
import json
import math
import os
import shutil
import subprocess
import threading
import time
import uuid
from .common import ROOT, DATA, atomic_json, read_json, config, child_path

def _kill_worker_tree(pid, created, request_path):
    """Kill only the recorded worker identity, never an unrelated reused PID."""
    import psutil
    if not isinstance(pid,int) or isinstance(pid,bool) or pid<=0 or not isinstance(created,(int,float)) or isinstance(created,bool) or not math.isfinite(created):return False
    try:
        proc=psutil.Process(pid)
        command=proc.cmdline()
        if not created or abs(proc.create_time()-created)>.01 or 'hi3d_workbench.worker' not in command or str(request_path) not in command:
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
                if 'hi3d_workbench.worker' in command and str(request_path) in command and isinstance(proc.info['create_time'],(int,float)):
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
            shutil.copy2(image,folder/"input.png")
            shutil.copy2(mesh,folder/"coarse.glb")
            atomic_json(folder/"request.json",options)
            state={"id":jid,"created":time.time(),"state":"queued","stage":"queued","options":options,"message":"任务已保存"}
            atomic_json(folder/"state.json",state)
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

    def _update(self,jid,**kw):
        with self.lock:
            state=self.get(jid)
            state.update(kw,updated=time.time())
            self._memory_states[jid]=state.copy()
            try:atomic_json(self.root/jid/"state.json",state)
            except (OSError,ValueError):
                self._dirty.add(jid)
                raise
            else:self._dirty.discard(jid)

    def _run(self,jid):
        folder=self.root/jid
        process=None
        created=None
        try:
            cfg=config()
            cmd=cfg["worker_command"] or [str((ROOT/cfg["worker_python"]).resolve())]
            # A custom command is a trusted local configuration, never HTTP input.
            with self.lock:
                if jid in self.cancelled:return
                process=subprocess.Popen([*cmd,"-u","-m","hi3d_workbench.worker","--request",str(folder/"request.json")],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",env={**os.environ,"PYTHONIOENCODING":"utf-8"},creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
                self.process=process
                import psutil
                try:created=psutil.Process(process.pid).create_time()
                except psutil.NoSuchProcess:pass  # A short process may already have exited.
                self.process_created=created
                self._update(jid,state="running",message="正在启动推理进程",worker_pid=process.pid,worker_created=created)
            with (folder/"worker.log").open("w",encoding="utf-8") as log:
                for line in process.stdout:
                    log.write(line);log.flush()
                    if line.startswith("HI3D_EVENT "):
                        try:
                            event=json.loads(line[len("HI3D_EVENT "):])
                            if not isinstance(event,dict) or not isinstance(event.get('stage'),str):continue
                            json.dumps(event,allow_nan=False)
                        except (ValueError,TypeError):continue  # Preserve malformed output in the log without losing the worker.
                        self._update(jid,stage=event["stage"],event=event,message=event.get("message",event["stage"]))
            code=process.wait()
            if jid in self.cancelled:
                self._update(jid,state="cancelled",stage="cancelled",message="已停止；输入和日志已保留")
            elif code==0 and (folder/"report.json").is_file() and (folder/"refined.glb").is_file():
                self._update(jid,state="succeeded",stage="complete",report=read_json(folder/"report.json"),message="精修完成")
            else:
                fail=read_json(folder/"failure.json",{"error":f"进程退出 {code}，请查看日志"})
                self._update(jid,state="failed",stage="failed",failure=fail,message=fail["error"])
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
