"""Build a Qt-free console adapter around the pinned upstream remeshing core.

The source checkout must be clean. This creates a validation build and a hash
receipt; it does not replace an installed tool or mark the runtime ready.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

REVISION = '4a49f82f720f647691c09a7d8a16fa63be208aad'
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as stream:
        value = hashlib.sha256()
        while block := stream.read(1024 * 1024):
            value.update(block)
        return value.hexdigest()


def build(source, folder):
    source, folder = Path(source).resolve(), Path(folder).resolve()
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=all'], text=True)
    if revision != REVISION or dirty:
        raise RuntimeError('AutoRemesher source must be the clean pinned revision ' + REVISION)
    if folder.is_relative_to(source):
        raise ValueError('Build output must remain outside the verified source checkout')
    folder.mkdir(parents=True, exist_ok=True)
    project = ROOT / 'tools' / 'native'
    command = ['cmake', '-S', str(project), '-B', str(folder / 'build'),
        '-DCMAKE_POLICY_VERSION_MINIMUM=3.5', '-DAUTO_REMESHER_SOURCE=' + source.as_posix()]
    if os.name == 'nt':
        command += ['-G', 'Visual Studio 17 2022', '-A', 'x64']
    else:
        command += ['-DCMAKE_BUILD_TYPE=Release']
    with (folder / 'configure.log').open('wb') as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    with (folder / 'build.log').open('wb') as stream:
        subprocess.run(['cmake', '--build', str(folder / 'build'), '--config', 'Release', '--parallel', '4'],
            check=True, stdout=stream, stderr=subprocess.STDOUT)
    executable = folder / 'build' / ('Release/autoremesher-core.exe' if os.name == 'nt' else 'autoremesher-core')
    if not executable.is_file():
        raise RuntimeError('Build did not produce the console tool')
    receipt = {'schema': 1, 'upstream': 'https://github.com/huxingyi/autoremesher',
        'source_revision': REVISION, 'native_algorithm_modified': False,
        'adapter': {str(path.relative_to(ROOT)): sha(path) for path in sorted(project.glob('*')) if path.is_file()},
        'executable': str(executable.relative_to(folder)), 'sha256': sha(executable),
        'quality_accepted': False, 'runtime_installed': False}
    (folder / 'build-receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    licenses = folder / 'licenses'; licenses.mkdir(exist_ok=True)
    for name, path in [('AutoRemesher-MIT.txt', source / 'LICENSE'),
                       ('TBB-Apache-2.0.txt', source / 'thirdparty/tbb/LICENSE'),
                       ('Eigen-MPL2.txt', source / 'thirdparty/eigen/COPYING.MPL2')]:
        shutil.copy2(path, licenses / name)
    for candidate in ('LICENSE.md', 'LICENSE'):
        path = source / 'thirdparty/meshoptimizer' / candidate
        if path.is_file():
            shutil.copy2(path, licenses / 'meshoptimizer-MIT.txt'); break
    print(json.dumps({'executable': str(executable), 'sha256': receipt['sha256']}), flush=True)
    return executable


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.build)
