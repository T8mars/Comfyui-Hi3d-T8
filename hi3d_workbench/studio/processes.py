"""Persist identities for external asset tools and stream their real progress."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from ..common import atomic_json
from .ownership import ProcessContainment


def run_stage(command, payload, directory, *, parser=None):
    import psutil
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    request = directory / 'request.json'
    atomic_json(request, payload)
    process = subprocess.Popen([*map(str, command), '--request', str(request)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    containment = ProcessContainment(process)
    created = None
    try:
        try:
            created = psutil.Process(process.pid).create_time()
        except psutil.NoSuchProcess:
            pass
        atomic_json(directory / 'process.json', {'pid': process.pid, 'created': created,
                                                'request': str(request)})
        with (directory / 'stage.log').open('w', encoding='utf-8') as log:
            for line in process.stdout:
                log.write(line); log.flush()
                if parser:
                    parser(line)
                print(line, end='', flush=True)
        code = process.wait()
        if code:
            raise RuntimeError(f'资产工具退出 {code}；详情见阶段日志')
    finally:
        try:
            if process.poll() is None:
                from ..jobs import _kill_worker_tree
                _kill_worker_tree(process.pid, created, request)
                process.wait()
        finally:
            if process.stdout:
                process.stdout.close()
            containment.close()


def blender_command():
    from .models import runtime_root, _read
    root = runtime_root()
    candidates = [root / 'blender' / 'blender.exe', root / 'blender' / 'blender']
    configured = _read(root / 'studio-runtime.json').get('blender')
    if configured:
        configured=Path(configured)
        candidates.insert(0, configured if configured.is_absolute() else root/configured)
    blender = next((path for path in candidates if path.is_file()), None)
    if blender is None:
        raise RuntimeError('Blender 运行环境未安装')
    return [str(blender), '-b', '--factory-startup', '-t', '8', '--python-exit-code', '1', '--python',
            str(Path(__file__).with_name('blender_ops.py')), '--']


def pipeline_command():
    """Use the existing private pipeline interpreter, never system/Comfy Python."""
    from ..common import ROOT
    override=os.environ.get('HI3D_PIPELINE_PYTHON')
    candidates=[Path(override)] if override else [
        root/name for root in (ROOT/'runtime-pipeline',ROOT/'.pipelineenv')
        for name in ('Scripts/python.exe','python.exe','bin/python')]
    executable=next((path.resolve() for path in candidates if path.is_file()),None)
    if executable is None:raise RuntimeError('GPU UV 需要整合包的独立推理运行环境；可选择 CPU UV')
    return [str(executable),'-u','-m','hi3d_workbench.studio.gpu_uv']
