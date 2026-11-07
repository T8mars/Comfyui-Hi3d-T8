"""Build the pinned native FPS sampler for the isolated Windows asset runtime.

The two compatibility changes only supply C++ declarations missing on MSVC;
the upstream farthest-point sampling algorithms remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

VERSION = '1.0.1'
URL = 'https://files.pythonhosted.org/packages/da/03/cc68f58cfe4ee4e66e5d7971cf5a6b9ec42d421838cabec29ee48c3722d2/fpsample-1.0.1.tar.gz'
ARCHIVE_SHA256 = '9f565fc44b5bd36b3cc50c449678c28ef79c506fd12a0d5e41d76751b7b13e47'
BUILD_REQUIRES = ['scikit-build-core==0.11.6', 'pybind11==3.0.1']


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def patch_cpp(source):
    source = source.replace('\r\n', '\n')
    if source.count('namespace py = pybind11;') != 1 or source.startswith('#include <optional>'):
        raise ValueError('fpsample 源码与已核对版本不匹配')
    source = '#include <optional>\n' + source
    return source.replace('namespace py = pybind11;',
        'namespace py = pybind11;\n#ifdef _WIN32\nusing ssize_t = py::ssize_t;\n#endif', 1)


def extract(archive, directory):
    prefix = 'fpsample-' + VERSION
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream.getmembers():
            name = PurePosixPath(member.name)
            if (not name.parts or name.parts[0] != prefix or name.is_absolute()
                    or any(part in {'..', '.'} or '\\' in part or ':' in part for part in name.parts)
                    or not (member.isdir() or member.isfile())):
                raise ValueError('fpsample 压缩包包含不支持的路径或链接')
        stream.extractall(directory, filter='data')
    return directory / prefix


def build(runtime, cache, python):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('当前资产运行环境使用 Python 3.12，请用同版本构建')
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / ('fpsample-' + VERSION + '.tar.gz')
    if not archive.is_file() or sha256(archive) != ARCHIVE_SHA256:
        print('Downloading pinned fpsample source archive', flush=True)
        temporary = archive.with_suffix('.download')
        try:
            with urllib.request.urlopen(URL, timeout=60) as response, temporary.open('wb') as output:
                count = 0
                while block := response.read(256 * 1024):
                    count += len(block)
                    if count > 5 * 1024 * 1024:
                        raise ValueError('fpsample 源码包超过预期长度')
                    output.write(block)
            if sha256(temporary) != ARCHIVE_SHA256:
                raise ValueError('fpsample 官方源码包 SHA256 不匹配')
            os.replace(temporary, archive)
        finally:
            temporary.unlink(missing_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='fpsample-build-', dir=cache))
    source = extract(archive, folder)
    cpp = source / 'src' / 'lib.cpp'
    original_sha = sha256(cpp)
    cpp.write_text(patch_cpp(cpp.read_text(encoding='utf-8')), encoding='utf-8', newline='\n')
    project = source / 'pyproject.toml'
    text = project.read_text(encoding='utf-8')
    previous = 'requires = ["scikit-build-core>=0.11", "pybind11>=2.6"]'
    if text.count(previous) != 1:
        raise ValueError('fpsample 构建元数据版本不匹配')
    project.write_text(text.replace(previous, 'requires = ' + json.dumps(BUILD_REQUIRES), 1), encoding='utf-8')
    wheels = folder / 'wheels'
    wheels.mkdir()
    options = {'check': True, 'env': {**os.environ, 'CMAKE_BUILD_PARALLEL_LEVEL': '4'}}
    if os.name == 'nt':
        options['creationflags'] = subprocess.CREATE_NO_WINDOW
    print('Building native fpsample wheel', flush=True)
    subprocess.run([str(python), '-m', 'pip', 'wheel', '--no-deps', '--wheel-dir', str(wheels), str(source)], **options)
    candidates = list(wheels.glob('fpsample-*.whl'))
    if len(candidates) != 1:
        raise RuntimeError('构建没有产生唯一的 fpsample wheel')
    wheel = candidates[0]
    destination = runtime / 'Lib' / 'site-packages' if os.name == 'nt' else runtime / 'lib' / 'python3.12' / 'site-packages'
    subprocess.run([str(python), '-m', 'pip', 'install', '--no-deps', '--upgrade', '--target', str(destination), str(wheel)], **options)
    receipt = {'version': VERSION, 'source': URL, 'archive_sha256': ARCHIVE_SHA256,
        'original_cpp_sha256': original_sha, 'patched_cpp_sha256': sha256(cpp),
        'changes': ['include <optional>', 'Windows ssize_t = pybind11::ssize_t'],
        'build_requires': BUILD_REQUIRES, 'wheel': wheel.name, 'wheel_sha256': sha256(wheel),
        'abi': sys.implementation.cache_tag}
    licenses = runtime / 'licenses'
    licenses.mkdir(parents=True, exist_ok=True)
    license_path = next(path for path in (source / 'LICENSE', source / 'LICENSE.txt') if path.is_file())
    (licenses / 'fpsample.txt').write_bytes(license_path.read_bytes())
    record = runtime / 'fpsample-build.json'
    record.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    build(args.runtime.resolve(), args.cache.resolve(), args.python.resolve())
