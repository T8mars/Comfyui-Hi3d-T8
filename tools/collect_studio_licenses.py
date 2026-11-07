"""Collect notices for the private studio runtime without copying model weights.

Run after the pinned sources and model catalogue files have been verified. The
manifest records the actual notice bytes; it does not grant additional rights.
Python package metadata and Blender's own license directory remain in the bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCE_PINS = {
    'parts': ('Tencent-Hunyuan/Hunyuan3D-Part', 'e96be065375438962375b55326416291342958a7'),
    'skintokens': ('VAST-AI-Research/SkinTokens', '273b691d35989d71cd17ff2895fdc735097b92d1'),
    'unirig': ('VAST-AI-Research/UniRig', '6793c6640ff01c8fb389f3993434124bb43d2933'),
    'kimodo': ('nv-tlabs/kimodo', 'e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954'),
}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def collect(native_source, eigen_source, *, root=ROOT):
    from hi3d_workbench.studio.models import CATALOG, verified_weight
    root, native_source, eigen_source = map(lambda p: Path(p).resolve(), (root, native_source, eigen_source))
    output = root / 'licenses/asset-studio'
    runtime = root / 'runtime-assets'
    from tools.package_studio_runtime import validate_sources
    shutil.copy2(root/'runtime_requirements/studio-sources.json',runtime/'sources/source-manifest.json')
    validate_sources(runtime/'sources')
    output.mkdir(parents=True, exist_ok=True)
    records = []

    def add(component, upstream, revision, paths, *, scope='code'):
        files = []
        for source, relative in paths:
            source, target = Path(source), output / relative
            if not source.is_file() or not source.stat().st_size:
                raise FileNotFoundError('Missing upstream notice: ' + str(source))
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            files.append({'path': target.relative_to(output).as_posix(), 'bytes': target.stat().st_size,
                          'sha256': sha(target)})
        if not files:
            raise ValueError('Component has no notices: ' + component)
        records.append({'component': component, 'scope': scope, 'upstream': upstream,
                        'revision': revision, 'files': files})

    for name, (repo, revision) in SOURCE_PINS.items():
        source = runtime / 'sources' / name
        names = ['LICENSE']
        if name == 'parts': names += ['NOTICE']
        if name == 'kimodo': names += ['ATTRIBUTIONS.MD']
        if name == 'unirig': names += ['src/model/michelangelo/LICENSE', 'src/model/pointcept/LICENSE']
        add(name + '-code', 'https://github.com/' + repo, revision,
            [(source / name_, 'code/' + name + '/' + name_) for name_ in names])

    add('meshoptimizer', 'https://github.com/zeux/meshoptimizer',
        'f0a684f11ead064e176775c9b796f82a55910e92',
        [(output / 'upstream/meshoptimizer-LICENSE.md', 'upstream/meshoptimizer-LICENSE.md')])
    add('sonata', 'https://github.com/facebookresearch/sonata',
        '18c09ff8d713494f78a8213792262b910977a65d',
        [(output / 'upstream/sonata-LICENSE.txt', 'upstream/sonata-LICENSE.txt')])
    native_names = ['LICENSE', 'AUTHORS', 'CONTRIBUTORS', 'ACKNOWLEDGEMENTS.html',
                    'thirdparty/isotropicremesher/LICENSE', 'thirdparty/tbb/LICENSE',
                    'thirdparty/meshoptimizer/LICENSE.md', 'thirdparty/eigen/LICENSE']
    native_names += [path.relative_to(native_source).as_posix()
                     for path in sorted((native_source / 'thirdparty/eigen').glob('COPYING.*'))]
    add('autoremesher-core', 'https://github.com/huxingyi/autoremesher',
        '4a49f82f720f647691c09a7d8a16fa63be208aad',
        [(native_source / name, 'native/autoremesher/' + name) for name in native_names],
        scope='Qt-free console core and its compiled dependencies; legacy Qt GUI is not covered here')
    add('fpsample', 'https://pypi.org/project/fpsample/1.0.1/', '1.0.1',
        [(runtime / 'licenses/fpsample.txt', 'native/fpsample-LICENSE.txt')])
    add('motion-correction-eigen', 'https://gitlab.com/libeigen/eigen/-/tree/3.4.0', '3.4.0',
        [(path, 'native/motion-correction/eigen/' + path.name) for path in sorted(eigen_source.glob('COPYING.*'))])

    for component, spec in CATALOG.items():
        names = [entry['name'] for entry in spec['files']
                 if Path(entry['name']).name.lower() in {'readme.md', 'license', 'license.txt', 'notice', 'notice.txt'}]
        add('model-' + component, 'https://huggingface.co/' + spec['repo'], spec['revision'],
            [(verified_weight(component, name), 'models/' + component + '/' + name) for name in names],
            scope='model card and supplied license; ' + spec.get('license', 'see upstream'))
    manifest = {'schema': 1, 'components': records,
                'notice': 'These notices retain their upstream terms. They do not relicense models or dependencies. '
                          'Python distributions retain their dist-info licenses; Blender retains its bundled licenses. '
                          'Modified workbench adapters and build recipes are supplied in tools/ and hi3d_workbench/studio/.'}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    target = runtime / 'licenses/asset-studio'
    shutil.copytree(output, target, dirs_exist_ok=True)
    return {'components': len(records), 'files': sum(len(row['files']) for row in records),
            'manifest_sha256': sha(output / 'manifest.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-source', type=Path, required=True)
    parser.add_argument('--eigen-source', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.native_source, args.eigen_source)), flush=True)
