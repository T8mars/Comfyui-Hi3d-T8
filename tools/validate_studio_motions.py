"""Run real isolated single-character motion cases, never parallel GPU models."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    'walk': 'A person walks forward at a steady pace.',
    'run': 'A person runs forward at a steady pace.',
    'wave': 'A person stands still and waves with the right hand.',
    'punch': 'A person performs a controlled punch and returns to a fighting stance.',
    'kick': 'A person in a fighting stance performs a front kick and returns to the starting stance.',
    'dodge': 'A person in a fighting stance ducks and sidesteps to dodge an attack, then returns to the starting stance.',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--text-device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--cases', nargs='+', choices=list(CASES), default=list(CASES))
    args = parser.parse_args()
    if not 0 <= args.seed <= 2**32-1:
        parser.error('--seed must be an unsigned 32-bit integer')
    source, output = args.source.resolve(), args.output.resolve()
    if not source.is_file() or output.exists():
        raise ValueError('Choose an existing rigged source and a new validation directory')
    output.mkdir(parents=True)
    results = {}
    for index, case in enumerate(args.cases):
        folder = output / case; folder.mkdir()
        shutil.copy2(source, folder / 'coarse.glb')
        request = dict(action='motion', prompt=CASES[case], duration=4, diffusion_steps=100,
            seed=args.seed, text_device=args.text_device, foot_lock=True, loop=False, root_motion='original')
        path = folder / 'request.json'
        path.write_text(json.dumps(request, indent=2), encoding='utf-8')
        print(f'{index + 1}/{len(args.cases)}: {case}', flush=True)
        with (folder / 'worker.log').open('wb') as log:
            process = subprocess.Popen([str(ROOT / 'runtime-assets/python.exe'), '-u', '-m',
                'hi3d_workbench.studio_worker', '--request', str(path)], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            try:
                created = psutil.Process(process.pid).create_time()
            except psutil.NoSuchProcess:
                created = None
            (folder / 'process.json').write_text(json.dumps(dict(pid=process.pid,created=created,
                request=str(path))), encoding='utf-8')
            code = process.wait()
        report = json.loads((folder / 'report.json').read_text(encoding='utf-8')) if (folder / 'report.json').is_file() else {}
        result = dict(exit_code=code, status=report.get('status'), seconds=report.get('seconds'),
            output_sha256=report.get('output_sha256'), visual_quality_accepted=False)
        results[case] = result
        (output / 'suite.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(json.dumps({case: result}), flush=True)
        if code or report.get('status') != 'succeeded':
            raise RuntimeError(case + ' failed; inspect ' + str(folder / 'worker.log'))


if __name__ == '__main__':
    main()
