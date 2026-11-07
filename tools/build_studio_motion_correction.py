"""Build Kimodo's pinned motion correction code for the private cp312 runtime.

Only build configuration differs from upstream: dependencies are local and
hash-verified rather than fetched by CMake. Native correction code is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

SOURCE_REVISION = 'e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954'
EIGEN_URL = 'https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2'
EIGEN_SHA256 = 'b4c198460eba6f28d34894e3a5710998818515104d6e74e5cc331ce31e46e626'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def eigen_headers(cache):
    archive = cache / 'eigen-3.4.0.tar.bz2'
    if not archive.is_file() or sha256(archive) != EIGEN_SHA256:
        temporary = archive.with_suffix('.download')
        with urllib.request.urlopen(EIGEN_URL, timeout=60) as response, temporary.open('wb') as output:
            count = 0
            while block := response.read(256 * 1024):
                count += len(block)
                if count > 5 * 1024 * 1024:
                    raise ValueError('Eigen archive exceeds expected size')
                output.write(block)
        if sha256(temporary) != EIGEN_SHA256:
            raise ValueError('Eigen archive SHA256 mismatch')
        os.replace(temporary, archive)
    directory = Path(tempfile.mkdtemp(prefix='eigen-', dir=cache))
    with tarfile.open(archive, 'r:bz2') as stream:
        for member in stream.getmembers():
            name = PurePosixPath(member.name)
            if (not name.parts or name.parts[0] != 'eigen-3.4.0' or name.is_absolute()
                    or any(part in {'..', '.'} or '\\' in part or ':' in part for part in name.parts)
                    or not (member.isdir() or member.isfile())):
                raise ValueError('Unsupported Eigen archive entry')
        stream.extractall(directory, filter='data')
    return directory / 'eigen-3.4.0'


def build(runtime, cache, python_sdk):
    if os.name != 'nt' or sys.version_info[:2] != (3, 12):
        raise RuntimeError('This build targets Windows CPython 3.12')
    runtime, cache = runtime.resolve(), cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    source = runtime / 'sources' / 'kimodo' / 'MotionCorrection'
    if not (source / 'src/cpp/BindingsPython.cpp').is_file():
        raise RuntimeError('Install the pinned Kimodo source first')
    import pybind11
    if pybind11.__version__ != '3.0.1':
        raise RuntimeError('Build requires pybind11==3.0.1')
    eigen = eigen_headers(cache)
    work = Path(tempfile.mkdtemp(prefix='motion-correction-', dir=cache))
    project, output = work / 'project', work / 'output'
    project.mkdir(); output.mkdir()
    # The original CMake project fetches dependencies online. Reference the
    # identical .cpp files with verified local headers and no FetchContent.
    (project / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.15)
project(hi3d_motion_correction LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
file(GLOB MATH_SOURCES "${MOTION_SOURCE}/src/cpp/Math/*.cpp")
file(GLOB ANIM_SOURCES "${MOTION_SOURCE}/src/cpp/AnimProcessing/*.cpp")
add_library(motion_correction_cpp_base STATIC ${MATH_SOURCES} ${ANIM_SOURCES})
set_target_properties(motion_correction_cpp_base PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(motion_correction_cpp_base PUBLIC "${MOTION_SOURCE}/src/cpp" "${EIGEN_HEADERS}")
target_compile_definitions(motion_correction_cpp_base PUBLIC EIGEN_MPL2_ONLY)
target_compile_options(motion_correction_cpp_base PRIVATE /W4 /arch:AVX)
pybind11_add_module(_motion_correction "${MOTION_SOURCE}/src/cpp/BindingsPython.cpp")
target_link_libraries(_motion_correction PRIVATE motion_correction_cpp_base)
''', encoding='utf-8')
    command = ['cmake', '-S', str(project), '-B', str(work / 'build'),
        '-G', 'Visual Studio 17 2022', '-A', 'x64',
        '-DPython3_EXECUTABLE=' + Path(sys.executable).as_posix(),
        '-Dpybind11_DIR=' + Path(pybind11.get_cmake_dir()).as_posix(),
        '-DMOTION_SOURCE=' + source.as_posix(), '-DEIGEN_HEADERS=' + eigen.as_posix(),
        '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE=' + output.as_posix()]
    if python_sdk:
        command += ['-DPython3_INCLUDE_DIR=' + (python_sdk / 'include').as_posix(),
                    '-DPython3_LIBRARY=' + (python_sdk / 'libs/python312.lib').as_posix()]
    subprocess.run(command, check=True)
    subprocess.run(['cmake', '--build', str(work / 'build'), '--config', 'Release', '--parallel', '4'], check=True)
    extensions = list(output.glob('_motion_correction*.pyd'))
    if len(extensions) != 1:
        raise RuntimeError('Build did not produce a unique native extension')
    target = runtime / 'engines' / 'kimodo-v510' / 'motion_correction'
    target.mkdir(parents=True, exist_ok=True)
    for path in (source / 'python/motion_correction').glob('*.py'):
        shutil.copy2(path, target / path.name)
    shutil.copy2(extensions[0], target / extensions[0].name)
    receipt = {'schema': 1, 'source_revision': SOURCE_REVISION,
        'eigen': {'url': EIGEN_URL, 'sha256': EIGEN_SHA256}, 'pybind11': pybind11.__version__,
        'source_files': {str(path.relative_to(source)): sha256(path) for path in sorted((source / 'src').rglob('*')) if path.is_file()},
        'extension': extensions[0].name, 'extension_sha256': sha256(extensions[0]),
        'native_code_modified': False, 'abi': 'cpython-312', 'build_project': str(project)}
    (runtime / 'motion-correction-build.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    licenses = runtime / 'licenses' / 'motion-correction'; licenses.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source.parent / 'LICENSE', licenses / 'LICENSE-Kimodo.txt')
    shutil.copy2(eigen / 'COPYING.MPL2', licenses / 'LICENSE-Eigen.txt')
    print(json.dumps({'extension': str(target / extensions[0].name), 'sha256': receipt['extension_sha256']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, default=Path('runtime-assets'))
    parser.add_argument('--cache', type=Path, default=Path('data/studio-native-build'))
    parser.add_argument('--python-sdk', type=Path, help='Matching CPython 3.12 development headers and import library')
    args = parser.parse_args()
    build(args.runtime, args.cache, args.python_sdk)
