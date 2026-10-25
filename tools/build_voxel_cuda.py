"""Build an isolated CUDA FDG add-on; never replace the installed o_voxel.

Source: quantaji/o-voxel-gpu, e477c69457c650dc680c02c2a1fb2ca58d3429ad.
Only the full flexible-dual-grid CUDA kernels are linked. Original CPU,
serialization, rasterization and other engines retain their existing wheels.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REVISION = 'e477c69457c650dc680c02c2a1fb2ca58d3429ad'
BUILD_ID = REVISION + '-t8-two-pass-fpstrict-v2'


def patched_sources(source, build):
    """Avoid waiting for a producer lane inside a divergent CUDA warp.

    First occupancy pass installs every brick key/value. A stream-ordered
    second pass sets all occupancy bits after publication has completed.
    Duplicate atomicOr is idempotent; geometry/QEF calculations are unchanged.
    """
    target = build / 'kernels'
    folder = target / 'convert/mesh_to_flexible_dual_grid_gpu'
    folder.mkdir(parents=True, exist_ok=True)
    def write(path, value):
        if not path.exists() or path.read_text(encoding='utf-8') != value:
            path.write_text(value, encoding='utf-8')
    write(target/'convert/api.h',(source/'src/convert/api.h').read_text(encoding='utf-8'))
    for path in (source/'src/convert/mesh_to_flexible_dual_grid_gpu').iterdir():
        if path.suffix not in {'.cu','.cuh'}:
            continue
        content = path.read_text(encoding='utf-8')
        if path.name == 'intersect_qef.cu':
            wait = 'while (val == kEmptyBrickVal)\n                        val = *val_ptr;'
            if content.count(wait) != 1:
                raise ValueError('Pinned CUDA publication sequence changed')
            content = content.replace(wait, '// A pending value is retried by the second stream-ordered pass.')
            start = content.index('            mark_occupied_voxel_bits_kernel<<<')
            end = content.index('            C10_CUDA_KERNEL_LAUNCH_CHECK();',start)+len('            C10_CUDA_KERNEL_LAUNCH_CHECK();')
            launch = content[start:end]
            content = content[:end]+'\n\n            // Complete bits skipped before key/value publication.\n'+launch+content[end:]
        write(folder/path.name,content)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arch-list', default='8.6;8.9;12.0+PTX')
    args = parser.parse_args()
    source = args.source.resolve()
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError('GPU voxel source revision does not match the audited build')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('GPU voxel source has local modifications')
    build = source.parent / 'voxel-cuda-build-v049'
    build.mkdir(exist_ok=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    kernels = patched_sources(source,build)
    binding = build / 'binding.cpp'
    binding.write_text('''#include <torch/extension.h>
#include <pybind11/stl.h>
#include "convert/api.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mesh_to_flexible_dual_grid_cuda", &o_voxel::fdg::mesh_to_flexible_dual_grid_cuda,
          py::call_guard<py::gil_scoped_release>());
    m.attr("source_revision") = "''' + BUILD_ID + '''";
    m.attr("architectures") = ''' + json.dumps(args.arch_list) + ''';
}
''', encoding='utf-8')
    files = ['intersect_qef.cu', 'face_qef.cu', 'boundary_qef.cu', 'mesh_to_flexible_dual_grid.cu']
    sources = [str(binding), *[str(kernels / 'convert/mesh_to_flexible_dual_grid_gpu' / name) for name in files]]
    includes = [str(kernels)]
    license_path = Path(__file__).resolve().parents[1] / 'licenses/TRELLIS2-LICENSE.txt'
    (build/'LICENSE.txt').write_text(license_path.read_text(encoding='utf-8'),encoding='utf-8')
    (build/'NOTICE.txt').write_text(
        'CUDA flexible dual grid kernels: https://github.com/quantaji/o-voxel-gpu\n'
        'Revision: '+REVISION+'\n'
        'T8 build: '+BUILD_ID+'; non-blocking two-pass occupancy publication.\n'
        'Based on O-Voxel/TRELLIS.2 by Microsoft and Jianfeng Xiang (MIT).\n'
        'CUDA implementation by quantaji, with reference to yuyujunjun/o-voxel-gpu.\n'
        'Build adapter by T8star. Original CPU o_voxel is not replaced.\n',encoding='utf-8')
    setup = build / 'setup.py'
    setup.write_text('''from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
setup(name='hi3d-voxel-cuda', version='0.1.3', license='MIT', license_files=['LICENSE.txt','NOTICE.txt'],
      description='Pinned CUDA flexible dual grid acceleration for Hi3D Workbench',
      ext_modules=[CUDAExtension('hi3d_voxel_cuda',
          sources=''' + repr(sources) + ''', include_dirs=''' + repr(includes) + ''',
          extra_compile_args={'cxx': ['/O2', '/std:c++17'] if os.name=='nt' else ['-O3','-std=c++17'],
                              'nvcc': ['-O3', '-std=c++17', '--fmad=false']})],
      cmdclass={'build_ext': BuildExtension})
''', encoding='utf-8')
    environment = dict(os.environ)
    environment.setdefault('MAX_JOBS', '1')
    environment['TORCH_CUDA_ARCH_LIST'] = args.arch_list
    temporary = build / 'temp'
    temporary.mkdir(exist_ok=True)
    environment.update(TMP=str(temporary), TEMP=str(temporary))
    subprocess.run([sys.executable, str(setup), 'bdist_wheel', '--dist-dir', str(output)],
                   cwd=build, env=environment, check=True)
    (output / 'build-provenance.json').write_text(json.dumps({
        'repository': 'https://github.com/quantaji/o-voxel-gpu', 'revision': revision,
        'build_id':BUILD_ID,'patch':'two stream-ordered occupancy passes; no divergent lane spin wait',
        'architectures': environment['TORCH_CUDA_ARCH_LIST'], 'python': sys.version,
        'isolation': 'separate hi3d_voxel_cuda extension; existing o_voxel unchanged',
    }, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
