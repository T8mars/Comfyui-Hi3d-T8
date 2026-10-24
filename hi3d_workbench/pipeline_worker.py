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

from .common import ROOT, atomic_json, config, digest, read_json
from .worker import emit


def run_child(command, request_path, directory, stage):
    """Persist child identity before consuming output so restart recovery is safe."""
    import psutil
    process = subprocess.Popen([*command, '--request', str(request_path)], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    created = None
    try:
        try:
            created = psutil.Process(process.pid).create_time()
        except psutil.NoSuchProcess:
            created = None
        atomic_json(directory / 'process.json', {'pid': process.pid, 'created': created,
                    'request': str(request_path), 'stage': stage})
        with (directory / 'stage.log').open('w', encoding='utf-8') as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith('HI3D_EVENT '):
                    try:
                        event=json.loads(line[len('HI3D_EVENT '):])
                        if event.get('stage')=='complete':
                            event.update(stage=stage+'_complete',message=f'{stage} 阶段已保存，继续处理后续阶段')
                            line='HI3D_EVENT '+json.dumps(event,ensure_ascii=False)+'\n'
                    except (ValueError,AttributeError):pass
                print(line, end='', flush=True)
        code = process.wait()
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
        if process.poll() is None:
            from .jobs import _kill_worker_tree
            _kill_worker_tree(process.pid, created, request_path)
            process.wait()
        if process.stdout:
            process.stdout.close()


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
                   'texture_size': req.get('texture_size', 2048), 'offload': req['offload'],
                   'background': req.get('background', 'auto'), 'model_root': str(pipeline_models.MODEL_ROOT)}
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
        if stage == 'coarse' and (stage_dir/'raw-geometry.glb').is_file():
            shutil.copy2(stage_dir/'raw-geometry.glb',folder/'generated.glb')
        for candidate in (stage_dir / 'foreground.png', out.with_name('foreground.png')):
            if candidate.is_file():
                shutil.copy2(candidate, folder / 'foreground.png')
                break
        atomic_json(folder / 'pipeline-progress.json', {'stages': stages})
        return out

    if mode == 'image_to_asset':
        shutil.copy2(extension_stage('coarse'), mesh)
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
                shutil.copy2(stage_dir / name, folder / name)
        atomic_json(folder / 'pipeline-progress.json', {'stages': stages})
        final = folder / 'refined.glb'
    material = {'texture_generated': False}
    if want_texture:
        output = extension_stage('texture', final)
        material = inspect_textured_glb(output)
        shutil.copy2(output, folder / 'textured.glb')
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
    args = parser.parse_args()
    try:
        run(args.request)
    except Exception as exc:
        atomic_json(args.request.parent / 'failure.json', {'status': 'failed', 'error': str(exc),
                    'traceback': traceback.format_exc(), 'partial_results_preserved': True})
        emit('failed', message=str(exc))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
