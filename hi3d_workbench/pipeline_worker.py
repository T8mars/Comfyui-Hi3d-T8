"""Sequential workbench supervisor. No changes to the ComfyUI bridge worker."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .common import ROOT, atomic_json, atomic_copy, config, digest, read_json
from .events import EVENT_MARKER, event_spans
from .worker import emit

_engine_session=None


def close_engine_session():
    global _engine_session
    if _engine_session is not None:
        _engine_session.close();_engine_session=None


def run_child(command, request_path, directory, stage):
    """Persist child identity before consuming output so restart recovery is safe."""
    import psutil
    global _engine_session
    session=None
    if 'hi3d_workbench.engines.run_stage' in command and read_json(request_path).get('engine')=='trellis2':
        from .runtime_session import RuntimeSession
        if _engine_session is None or _engine_session.process.poll() is not None:
            _engine_session=RuntimeSession(command[:command.index('--stage')],request_path)
        session=_engine_session
        process=session.process
        lines=session.stream(request_path,stage)
    else:
        close_engine_session()
        process = subprocess.Popen([*command, '--request', str(request_path)], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        lines=process.stdout
    created = None
    try:
        try:
            created = psutil.Process(process.pid).create_time()
        except psutil.NoSuchProcess:
            created = None
        atomic_json(directory / 'process.json', {'pid': process.pid, 'created': created,
                    'request': str(request_path), 'stage': stage})
        with (directory / 'stage.log').open('w', encoding='utf-8') as log:
            for line in lines:
                log.write(line)
                log.flush()
                spans=list(event_spans(line))
                if not spans:
                    print(line,end='',flush=True)
                    continue
                cursor=0
                for start,end,event in spans:
                    display=line[cursor:start]
                    if display:print(display,end='' if display.endswith('\n') else '\n',flush=True)
                    # Keep real events on independent lines. Forwarding the
                    # original mixed line too would make Jobs apply it twice.
                    event.setdefault('pipeline_stage',stage)
                    if event['stage']=='complete':
                        event.update(stage=stage+'_complete',message=f'{stage} 阶段已保存，继续处理后续阶段')
                    print(EVENT_MARKER+json.dumps(event,ensure_ascii=False,allow_nan=False),flush=True)
                    cursor=end
                tail=line[cursor:]
                if tail.strip():print(tail,end='' if tail.endswith('\n') else '\n',flush=True)
        code = session.code if session else process.wait()
        if code:
            try:
                failure = read_json(directory / 'failure.json', {})
            except (OSError, ValueError):
                failure = {}
            detail = failure.get('error') if isinstance(failure, dict) else None
            if not isinstance(detail, str) or not detail.strip():
                detail = f'进程退出 {code}，请查看日志'
            raise RuntimeError(f'{stage} 阶段失败：{detail}')
    finally:
        if session:
            if session.code!=0:close_engine_session()
        else:
            if process.poll() is None:
                from .jobs import _kill_worker_tree
                _kill_worker_tree(process.pid, created, request_path)
                process.wait()
            if process.stdout:process.stdout.close()


def run(request_path):
    from . import pipeline_models
    from .asset_pipeline import inspect_textured_glb, inspect_asset_geometry
    from .geometry import inspect_mesh, load_mesh

    started = time.time()
    request_path = request_path.resolve()
    folder = request_path.parent
    req = read_json(request_path)
    mode = req.get('mode', 'refine')
    if mode not in {'image_to_asset', 'texture', 'refine'}:
        raise ValueError('任务模式无效')
    stages = []
    image = folder / 'input.png'
    mesh = folder / 'coarse.glb'
    want_refine = mode == 'refine' or mode == 'image_to_asset' and req.get('refine_geometry', False)
    want_texture = mode == 'texture' or req.get('generate_texture', mode == 'image_to_asset')
    pipeline_python = str(pipeline_models.runtime_python())

    def extension_stage(stage, source_mesh=None):
        stage_dir = folder / 'stages' / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        out = stage_dir / ('coarse.glb' if stage == 'coarse' else 'textured.glb')
        payload = {'image_path': str(image), 'output_path': str(out),
                   'seed': req['seed'], 'resolution': req.get('geometry_resolution',256) if stage == 'coarse' else req.get('texture_resolution', 512),
                   'engine': req.get('geometry_engine','hunyuan_mini') if stage == 'coarse' else req.get('texture_engine','trellis2'),
                   'geometry_steps': req.get('geometry_steps',50), 'texture_steps': req.get('texture_steps',12),
                   'geometry_cleanup': req.get('geometry_cleanup',True),
                   'pipeline_attention': req.get('pipeline_attention','auto'),
                   'model_policy': req.get('model_policy','runtime'),
                   'voxel_backend': req.get('voxel_backend','auto'), 'encoding_cache': req.get('encoding_cache',True),
                   'texture_size': req.get('texture_size', 2048), 'offload': req['offload'],
                   'background': req.get('background', 'auto'), 'model_root': str(pipeline_models.MODEL_ROOT)}
        # Face budgets apply to generated geometry only. Existing mesh
        # texturing must never inherit a simplification request.
        if stage == 'coarse':payload['target_faces']=req.get('geometry_target_faces',1000000)
        if source_mesh:
            payload['mesh_path'] = str(source_mesh)
        stage_request = stage_dir / 'request.json'
        atomic_json(stage_request, payload)
        emit(stage + '_starting', message='正在生成基础模型' if stage == 'coarse' else '正在为最终模型生成 PBR 材质')
        run_child([pipeline_python, '-u', '-m', 'hi3d_workbench.engines.run_stage', '--stage', stage], stage_request, stage_dir, stage)
        if not out.is_file():
            raise RuntimeError(f'{stage} 阶段退出但没有输出模型')
        stage_report = read_json(out.with_suffix('.stage.json'), {})
        if not isinstance(stage_report, dict) or stage_report.get('status') != 'succeeded':
            raise ValueError('模型阶段没有提供有效的成功报告')
        stages.append({'stage': stage, **stage_report})
        if stage == 'coarse':atomic_copy(out, mesh, immutable=True)
        if stage == 'coarse' and (stage_dir/'raw-geometry.glb').is_file():
            atomic_copy(stage_dir/'raw-geometry.glb',folder/'generated.glb',immutable=True)
        for candidate in (stage_dir / 'foreground.png', out.with_name('foreground.png')):
            if candidate.is_file():
                atomic_copy(candidate, folder / 'foreground.png')
                break
        atomic_json(folder / 'pipeline-progress.json', {'stages': stages})
        return out

    if mode == 'image_to_asset':
        extension_stage('coarse')
    if not mesh.is_file():
        raise ValueError('此模式需要已有模型')
    inspect_mesh(load_mesh(mesh))
    final = mesh
    if want_refine:
        stage_dir = folder / 'stages' / 'refine'
        stage_dir.mkdir(parents=True, exist_ok=True)
        foreground = folder / 'foreground.png'
        shutil.copy2(foreground if foreground.is_file() else image, stage_dir / 'input.png')
        shutil.copy2(mesh, stage_dir / 'coarse.glb')
        stage_request = stage_dir / 'request.json'
        atomic_json(stage_request, req)
        cfg = config()
        command = cfg['worker_command'] or [str((ROOT / cfg['worker_python']).resolve())]
        run_child([*command, '-u', '-m', 'hi3d_workbench.worker'], stage_request, stage_dir, 'refine')
        stage_report = read_json(stage_dir / 'report.json')
        if not isinstance(stage_report, dict) or stage_report.get('status') != 'succeeded' or not (stage_dir / 'refined.glb').is_file():
            raise RuntimeError('Hi3D 阶段缺少有效的输出或报告')
        stages.append({'stage': 'refine', **stage_report})
        for name in ('refined.glb', 'refined.ply', 'refined.stl', 'refined.npz', 'conditioning.png'):
            if (stage_dir / name).is_file():
                atomic_copy(stage_dir / name, folder / name,immutable=True)
        atomic_json(folder / 'pipeline-progress.json', {'stages': stages})
        final = folder / 'refined.glb'
    material = {'texture_generated': False}
    if want_texture:
        output = extension_stage('texture', final)
        material = inspect_textured_glb(output)
        atomic_copy(output, folder / 'textured.glb',immutable=True)
        final = folder / 'textured.glb'
    report = {'status': 'succeeded', 'mode': mode, 'stages': stages, 'seconds': round(time.time() - started, 2),
              'peak_allocated_mb': max((s.get('peak_allocated_mb', s.get('peak_vram_mb', 0)) for s in stages), default=0),
              'actual_resolution': req['resolution'] if want_refine else stages[0].get('actual_resolution',req.get('geometry_resolution',256)) if mode=='image_to_asset' else 0,
              'geometry_engine': req.get('geometry_engine','hunyuan_mini') if mode=='image_to_asset' else None,
              'texture_engine': req.get('texture_engine','trellis2') if want_texture else None,
              'texturing_resolution': req.get('texture_resolution',512) if want_texture else None,
              'texture_size': req.get('texture_size',2048) if want_texture else None,
              'geometry': inspect_asset_geometry(final) if want_texture else inspect_mesh(load_mesh(final)), 'final_file': final.name,
              'input_sha256': digest(image), 'final_sha256': digest(final),
              'mesh_input_sha256': digest(mesh),
              'texture_mesh_sha256': next((stage.get('input_mesh_sha256') for stage in stages if stage.get('stage')=='texture'),None),
              'normalized': bool(want_refine or mode == 'image_to_asset'), **material}
    atomic_json(folder / 'report.json', report)
    emit('complete', message='模型与贴图已保存' if want_texture else '模型已保存')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--session', action='store_true')
    args = parser.parse_args()
    if args.session:
        from .runtime_session import serve
        def execute(item):
            try:run(Path(item['request']));return 0
            except Exception as exc:
                path=Path(item['request'])
                atomic_json(path.parent/'failure.json',{'status':'failed','error':str(exc),'traceback':traceback.format_exc()})
                close_engine_session()
                emit('failed',message=str(exc))
                return 1
        try:return serve(execute)
        finally:close_engine_session()
    try:
        run(args.request)
    except Exception as exc:
        atomic_json(args.request.parent / 'failure.json', {'status': 'failed', 'error': str(exc),
                    'traceback': traceback.format_exc(), 'partial_results_preserved': True})
        emit('failed', message=str(exc))
        raise SystemExit(1)
    finally:close_engine_session()


if __name__ == '__main__':
    main()
