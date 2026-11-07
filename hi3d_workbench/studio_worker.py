"""One owned process for an asset operation; no permanently hidden GPU server."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from .common import atomic_json, read_json, digest


def emit(stage, message, **details):
    print('HI3D_EVENT ' + json.dumps({'stage': stage, 'message': message, **details},
                                   ensure_ascii=False, allow_nan=False), flush=True)


def run(request_path):
    request_path = Path(request_path).resolve()
    folder = request_path.parent
    request = read_json(request_path)
    source = folder / 'coarse.glb'
    target = folder / 'output.glb'
    action = request['action']
    started = time.monotonic()
    from .studio import models
    # Kimodo requires Transformers 5; the two rigging backends use separately
    # pinned versions. A fresh operation process keeps their imports isolated.
    if action == 'motion':
        sys.path.insert(0, str(models.runtime_root() / 'engines' / 'kimodo-v510'))
    elif action == 'rig' and request.get('rig_engine') == 'unirig':
        sys.path.insert(0, str(models.runtime_root() / 'engines' / 'unirig'))
    emit('asset_loading', '正在读取模型', device='cpu')
    from .studio.gltf import Document
    from .studio.scene import instances
    document = Document.load(source)
    before = sum(len(item.faces) for item in instances(document))
    if action == 'simplify':
        from .studio.mesh import simplify
        emit('asset_simplify', '正在保护材质、UV 与模型轮廓并减面', device='cpu')
        details = simplify(source, target, target_faces=request.get('target_faces'), ratio=request.get('ratio'),
            max_error=request.get('max_error', .01), priority=request.get('priority', 'quality'),
            library_path=models.runtime_root() / ('meshoptimizer.dll' if os.name == 'nt' else 'libmeshoptimizer.so'),
            progress=lambda current, total: emit('asset_simplify', '已处理材质分组', device='cpu',
                progress_current=current, progress_total=total, progress_unit='primitives'))
    elif action == 'retopology':
        from .studio.retopology import run as retopology
        details = retopology(source, target, request, emit)
    elif action in {'segment', 'edit_parts', 'complete_parts'}:
        from .studio.parts import run as parts
        details = parts(source, target, request, emit)
    elif action in {'rig', 'edit_rig'}:
        from .studio.rigging import run as rig
        details = rig(source, target, request, emit)
    elif action == 'attach_parts':
        from .studio.part_attachment import run as attach
        details = attach(source, target, request, emit)
    elif action in {'retarget', 'motion'}:
        from .studio.motion import run as motion
        details = motion(source, target, request, emit)
    else:
        raise ValueError('无效资产加工任务')
    emit('asset_validation', '正在检查输出网格、材质与动画数据', device='cpu')
    result = Document.load(target)
    faces = sum(len(item.faces) for item in instances(result))
    if not faces:
        raise ValueError('结果网格为空')
    if action in {'rig', 'edit_rig', 'attach_parts', 'retarget', 'motion'} and not result.tree.get('skins'):
        raise ValueError('结果缺少骨架或蒙皮')
    if action in {'retarget', 'motion'} and not result.tree.get('animations'):
        raise ValueError('结果缺少动画')
    if request.get('parts_path') and action in {'rig', 'edit_rig', 'attach_parts', 'retarget', 'motion'}:
        from .studio.parts import carry_metadata
        details.update(carry_metadata(source, target, request['parts_path']))
    report = {'status': 'succeeded', 'mode': 'studio', 'action': action, 'engine': details.get('engine'),
              'seconds': round(time.monotonic() - started, 3), 'input_faces': before, 'output_faces': faces,
              'input_sha256': digest(source), 'output_sha256': digest(target), 'details': details,
              'materials': len(result.tree.get('materials', [])), 'skins': len(result.tree.get('skins', [])),
              'animations': [item.get('name', str(index)) for index, item in enumerate(result.tree.get('animations', []))]}
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        report['torch_peak_allocated_mib'] = round(torch.cuda.max_memory_allocated() / 2**20, 2)
        report['torch_peak_reserved_mib'] = round(torch.cuda.max_memory_reserved() / 2**20, 2)
    atomic_json(folder / 'report.json', report)
    emit('complete', '资产加工完成，输出已保存', report=report)
    return report


def main():
    from .common import configure_stdio
    configure_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    request = Path(args.request).resolve()
    try:
        run(request)
    except BaseException as exc:
        if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
            traceback.print_exc()
        atomic_json(request.parent / 'failure.json', {'error': str(exc), 'type': type(exc).__name__})
        emit('failed', str(exc))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
