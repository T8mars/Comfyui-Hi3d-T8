"""Resumable real-engine comparisons. CPU fixtures never count as quality evidence.

Geometry comparisons share the input image and seed. Texture comparisons share
one supplied mesh; changing the mesh requires a separate comparison directory.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hi3d_workbench.common import atomic_json, digest
from hi3d_workbench import pipeline_models
from hi3d_workbench.engine_catalog import engine


def run_stage(stage, name, reference, mesh, directory, resolution, texture_size, seed):
    choice = engine(stage, name)
    output = directory / ('geometry.glb' if stage == 'coarse' else 'textured.glb')
    directory.mkdir(parents=True, exist_ok=True)
    request = {'engine': name, 'image_path': str(reference), 'output_path': str(output),
               'seed': seed, 'resolution': resolution or choice['default_resolution'],
               'texture_size': texture_size, 'offload': True, 'background': 'keep',
               'model_root': str(pipeline_models.MODEL_ROOT),
               'geometry_steps': choice['default_steps'] if stage == 'coarse' else 50,
               'texture_steps': choice['default_steps'] if stage == 'texture' else 12}
    if mesh is not None:
        request['mesh_path'] = str(mesh)
    evidence = {'stage': stage, 'engine': name, 'request': request,
                'input_image_sha256': digest(reference), 'input_mesh_sha256': digest(mesh) if mesh else None}
    implementation = ['hi3d_workbench/engines/run_stage.py', 'hi3d_workbench/engine_catalog.py',
                      'hi3d_workbench/asset_pipeline.py']
    if name == 'trellis2':implementation.append('hi3d_workbench/engines/attention.py')
    specialized = {('coarse','trellis2'):'trellis_geometry.py',
                   ('texture','ultratex'):'ultratex_texture.py', ('texture','hunyuan21'):'hunyuan_texture.py'}
    if (stage,name) in specialized:
        implementation.append('hi3d_workbench/engines/' + specialized[(stage,name)])
    if stage == 'texture' and name == 'hunyuan21':
        implementation.append('hi3d_workbench/engines/hunyuan_renderer.py')
    if stage=='texture':
        implementation.append('hi3d_workbench/engines/mesh_uv.py')
    evidence['implementation_sha256'] = {path:digest(ROOT/path) for path in implementation}
    evidence['model_revisions'] = {key:{field:pipeline_models.SPECS[key][field] for field in ('repo','revision')}
                                  for key in choice['components']}
    receipt = directory / 'benchmark.json'
    if receipt.is_file():
        old = json.loads(receipt.read_text(encoding='utf-8'))
        if any(old.get(key) != value for key, value in evidence.items()):
            raise RuntimeError(f'Comparison inputs changed: use a new directory: {directory}')
        if old.get('status') == 'succeeded' and output.is_file() and old.get('output_sha256') == digest(output):
            return old
    if not pipeline_models.status()['engines'][stage][name]['ready']:
        evidence.update(status='unavailable', reason='Required model or runtime is not installed and verified')
        atomic_json(receipt, evidence)
        return evidence
    atomic_json(directory / 'request.json', request)
    atomic_json(receipt, {**evidence, 'status': 'running', 'started': time.time()})
    command = [str(pipeline_models.runtime_python()), '-u', '-m', 'hi3d_workbench.engines.run_stage',
               '--stage', stage, '--request', str(directory / 'request.json')]
    measurement = directory / f'memory-{time.time_ns()}.json'
    stop = measurement.with_suffix('.stop')
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    with measurement.with_suffix('.log').open('w',encoding='utf-8') as monitor_log:
        monitor = subprocess.Popen([sys.executable,'-u',str(ROOT/'tools/monitor_gpu_memory.py'),
            '--output',str(measurement),'--stop-file',str(stop),'--seconds','14400'],
            cwd=ROOT,stdout=monitor_log,stderr=subprocess.STDOUT,creationflags=flags)
        try:
            with (directory / 'worker.log').open('w', encoding='utf-8') as log:
                code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                      env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},creationflags=flags).returncode
        finally:
            stop.touch()
            try:monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:monitor.terminate();monitor.wait(timeout=5)
    if monitor.returncode == 0 and measurement.is_file():
        measured=json.loads(measurement.read_text(encoding='utf-8'))
        evidence['resource_measurement']={key:value for key,value in measured.items() if key!='samples'}
        evidence['resource_measurement']['file']=measurement.name
    else:evidence['resource_measurement']={'status':'unavailable','log':measurement.with_suffix('.log').name}
    report_file = output.with_suffix('.stage.json')
    report = json.loads(report_file.read_text(encoding='utf-8')) if report_file.is_file() else {}
    succeeded = code == 0 and report.get('status') == 'succeeded' and output.is_file()
    if succeeded and stage == 'texture':
        from hi3d_workbench.asset_pipeline import inspect_textured_glb
        evidence['material_validation'] = inspect_textured_glb(output)
    evidence.update(status='succeeded' if succeeded else 'failed', report=report, exit_code=code,
                    output_sha256=digest(output) if succeeded else None,
                    quality_review='pending actual rendered front/back/closeup and relighting review')
    atomic_json(receipt, evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--mesh', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['coarse', 'texture'], required=True)
    parser.add_argument('--engines', nargs='+', required=True)
    parser.add_argument('--resolution', type=int)
    parser.add_argument('--texture-size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    reference = args.reference.resolve()
    mesh = args.mesh.resolve() if args.mesh else None
    if args.stage == 'texture' and mesh is None:
        parser.error('Texture comparisons require one shared --mesh')
    results = []
    for name in args.engines:
        results.append(run_stage(args.stage, name, reference, mesh, args.output.resolve() / name,
                                 args.resolution, args.texture_size, args.seed))
        print(json.dumps({'engine': name, 'status': results[-1]['status']}, ensure_ascii=False), flush=True)
    atomic_json(args.output.resolve() / 'comparison.json', {'stage': args.stage, 'results': results,
                'quality_selected': False, 'note': 'Successful inference alone does not prove visual quality.'})
    return 0 if all(result['status'] == 'succeeded' for result in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
