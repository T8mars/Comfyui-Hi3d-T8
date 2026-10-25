"""Owned, sequential subprocess sessions; no socket, pickle or external service."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid

from .common import ROOT, atomic_json

RESULT='HI3D_SESSION_RESULT '


class RuntimeSession:
    def __init__(self, command, control):
        import psutil
        self.control=Path(control).resolve()
        self.process=subprocess.Popen([*command,'--session','--request',str(self.control)],cwd=ROOT,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,encoding='utf-8',errors='replace',bufsize=1,
            env={**os.environ,'PYTHONIOENCODING':'utf-8'},
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:self.created=psutil.Process(self.process.pid).create_time()
        except psutil.NoSuchProcess:self.created=None  # Preserve the real exit status of a very short process.
        self.code=None

    def stream(self, request, stage=None):
        self.code=None
        exited=self.process.poll()
        if exited is not None:
            self.code=exited
            raise RuntimeError(f'常驻运行时已退出（{exited}），请重新提交任务')
        identity=uuid.uuid4().hex
        try:
            self.process.stdin.write(json.dumps({'id':identity,'request':str(Path(request).resolve()),'stage':stage})+'\n')
            self.process.stdin.flush()
        except OSError:
            try:self.code=self.process.wait(timeout=.2)
            except subprocess.TimeoutExpired:raise
            raise RuntimeError(f'运行时意外退出（{self.code}），请查看任务日志') from None
        for line in self.process.stdout:
            if not line.strip():continue
            if line.startswith(RESULT):
                result=json.loads(line[len(RESULT):])
                if result.get('id')!=identity:raise RuntimeError('运行时任务回执不匹配')
                self.code=int(result['code'])
                return
            yield line
        raise RuntimeError(f'运行时意外退出（{self.process.wait()}），请查看任务日志')

    def close(self):
        if self.process.poll() is None:
            from .jobs import _kill_worker_tree
            # The fixed control path and creation time identify this exact
            # owned interpreter, including its inference descendants.
            _kill_worker_tree(self.process.pid,self.created,self.control)
            self.process.wait(timeout=10)
        for stream in (self.process.stdin,self.process.stdout):
            if stream:
                try:stream.close()
                except OSError:pass


def serve(execute):
    import psutil
    parent=psutil.Process(os.getppid())
    parent_created=parent.create_time()
    def watch_parent():
        while True:
            threading.Event().wait(1)
            try:
                if not parent.is_running() or abs(parent.create_time()-parent_created)>.01:os._exit(1)
            except psutil.NoSuchProcess:os._exit(1)
            except psutil.AccessDenied:pass
    threading.Thread(target=watch_parent,daemon=True).start()
    for line in sys.stdin:
        identity=None
        try:
            if len(line)>65536:raise ValueError('运行时指令过长')
            item=json.loads(line)
            identity=item['id']
            code=execute(item)
        except Exception:
            import traceback
            traceback.print_exc()
            code=1
        print('\n'+RESULT+json.dumps({'id':identity,'code':code}),flush=True)
    return 0
