"""Owned command wrapper for AutoRemesher with a durable request identity."""
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import time
from collections import deque
from pathlib import Path

from ..common import read_json, atomic_json
from ..studio_worker import emit
from .ownership import ProcessContainment

NATIVE_STEPS = {
    'Computing voxel size': '计算目标网格密度',
    'Splitting mesh into islands': '识别连通部件',
    'Building island contexts': '准备连通部件',
    'Remeshing uniformly': '重新采样网格',
    'Solving frame field': '求解布线方向场',
    'Eliminating cover constraints': '求解参数化约束',
    'Rounding cover to integers': '求解四边面参数坐标',
    'Projecting vertices': '将顶点投影回原表面',
    'Smoothing and projecting': '平滑并保持原表面',
    'Removing non-manifold faces': '整理非流形连接',
    'Cleaning up triangles': '整理残余三角面',
    'Merging shared five edge faces': '合并相邻五边面',
    'Switching high valence edges': '整理高连接度顶点',
    'Merging islands': '合并重拓扑部件',
}


def stream_output(stream, directory, *, clock=time.monotonic, report=emit, output=None):
    """Drain verbose native output with bounded lines, storage and console I/O."""
    write = output or (lambda value: print(value, flush=True))
    tail = deque(maxlen=200)
    records = printed = 0
    last_printed = last_saved = float('-inf')
    last_progress = None
    directory = Path(directory)
    def save():
        atomic_json(directory / 'native-output.json',
            {'read_records': records, 'printed_records': printed, 'tail': list(tail)})
    for line in iter(lambda: stream.readline(16384), ''):
        records += 1
        value = ''.join(char if char.isprintable() else ' ' for char in line.rstrip())[:500]
        tail.append(value)
        now = clock()
        fatal = re.search(r'error|exception|bad allocation|invalid remesh|access violation', value, re.I)
        if records <= 20 or fatal or now - last_printed >= 1:
            write(value); printed += 1; last_printed = now
        match = re.search(r'(?<!\d)(\d+(?:\.\d+)?)\s*%', line)
        if match:
            progress = float(match.group(1))
            status = re.sub(r'^\s*(?:done\.?\s*)?', '', line[match.end():]).strip()[:180]
            observation = (progress, status)
            if 0 <= progress <= 100 and observation != last_progress:
                message = NATIVE_STEPS.get(status, '原生步骤：' + status) if status else '正在重新整理四边面布线'
                report('asset_retopology', message, device='cpu',
                       progress_current=progress, progress_total=100, progress_unit='percent')
                last_progress = observation
        if now - last_saved >= 2:
            save(); last_saved = now
    save()
    return {'read_records': records, 'printed_records': printed}


def main():
    from ..common import configure_stdio
    configure_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    request = read_json(__import__('pathlib').Path(args.request))
    if request.get('tool') != 'autoremesher':
        raise ValueError('无效原生工具')
    edge_scaling = float(request.get('edge_scaling', 1))
    if not math.isfinite(edge_scaling) or not 1 <= edge_scaling <= 4:
        raise ValueError('重拓扑边长倍率必须为 1 到 4')
    command = [request['executable'], '--input', request['input'], '--output', request['output'],
               '--report', request['report'], '--target-quads', str(request['target_quads']),
               '--edge-scaling', str(edge_scaling), '--sharp-edge', str(request['sharp_angle']),
               '--smooth-normal', '0', '--adaptivity', str(request['adaptivity']), '--anisotropy', '1']
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace',
        env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'},
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    containment = ProcessContainment(child)
    try:
        stream_output(child.stdout, Path(args.request).resolve().parent)
        if child.wait():
            raise RuntimeError('AutoRemesher 未成功完成，请查看原生日志')
    finally:
        try:
            if child.poll() is None:
                child.kill(); child.wait()
        finally:
            child.stdout.close()
            containment.close()


if __name__ == '__main__':
    main()
