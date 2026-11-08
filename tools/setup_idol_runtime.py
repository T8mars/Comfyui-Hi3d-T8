"""Prepare Idol's private dependency overlay without upgrading Torch or Comfy.

Readiness is only recorded after validating a real successful generation
receipt. Imports alone never count as model-effect acceptance.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import base64
import hashlib
import urllib.request
import uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hi3d_workbench.common import atomic_json,digest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--validation',type=Path)
    parser.add_argument('--no-install',action='store_true')
    parser.add_argument('--validate-image',type=Path,help='Run a real local image-to-mesh validation before recording readiness')
    args=parser.parse_args()
    if args.validation and args.validate_image:parser.error('Choose --validation or --validate-image')
    from hi3d_workbench.pipeline_models import runtime_python
    python=runtime_python()
    runtime=python.parent.parent if python.parent.name.lower() in {'scripts','bin'} else python.parent
    overlay=runtime/'engines/idol'
    mesh_source=runtime/'sources/idol_mesh'
    if not args.no_install:
        revision='0decb39ee5b5e17c200334589fc317dbbe98310e'
        url=f'https://api.github.com/repos/Comfy-Org/ComfyUI/contents/comfy_extras/nodes_gaussian_splat.py?ref={revision}'
        with urllib.request.urlopen(url,timeout=60) as response:remote=json.load(response)
        data=base64.b64decode(remote['content'])
        blob=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
        if blob!=remote['sha']:raise ValueError('Upstream geometry source blob mismatch')
        mesh_source.mkdir(parents=True,exist_ok=True)
        (mesh_source/'nodes_gaussian_splat.py').write_bytes(data)
        license_data=(runtime/'sources/comfy_core/LICENSE').read_bytes()
        (mesh_source/'LICENSE').write_bytes(license_data)
        atomic_json(mesh_source/'source-manifest.json',{'revision':revision,'repo':'Comfy-Org/ComfyUI',
            'scope':'isolated upstream Gaussian meshing module; pinned existing Comfy dependencies retained',
            'files':{'nodes_gaussian_splat.py':hashlib.sha256(data).hexdigest(),'LICENSE':hashlib.sha256(license_data).hexdigest()}})
    if not args.no_install:
        # A clean overlay prevents stale files from older wheel versions.
        overlay.parent.mkdir(parents=True,exist_ok=True)
        staged=overlay.parent/('idol-install-'+uuid.uuid4().hex)
        subprocess.run([str(python),'-m','pip','install','--no-deps','--target',str(staged),
                        '-r',str(ROOT/'runtime_requirements/idol-windows.txt')],check=True)
        subprocess.run([str(python),'-c',f'import sys;sys.path.insert(0,{str(staged)!r});import av,aiohttp;assert av.__version__=="17.0.0"'],check=True)
        if overlay.exists():
            backup=ROOT/'data/runtime-backups'/('idol-'+uuid.uuid4().hex)
            backup.parent.mkdir(parents=True,exist_ok=True)
            if not overlay.resolve().is_relative_to(runtime.resolve()):raise ValueError('Private overlay path escapes runtime')
            overlay.rename(backup)
        staged.rename(overlay)
    subprocess.run([str(python),'-c',
        'from hi3d_workbench.engines.idol import initialize; initialize(); '
        'from comfy_extras.nodes_triposplat import TripoSplatConditioning; '
        'from comfy_extras.nodes_gaussian_splat import SplatToMesh, RenderSplat; '
        'import av; print("Idol private runtime imports passed")'],cwd=ROOT,check=True)
    if args.validate_image:
        image=args.validate_image.resolve()
        if not image.is_file():raise ValueError('Validation image does not exist')
        folder=ROOT/'data/idol-runtime-validation'/uuid.uuid4().hex
        output=folder/'coarse.glb';request=folder/'request.json'
        atomic_json(request,{'engine':'idol','image_path':str(image),'output_path':str(output),
            'seed':23,'geometry_steps':50,'resolution':384,'idol_video':False})
        subprocess.run([str(python),'-u','-m','hi3d_workbench.engines.run_stage','--stage','coarse',
                        '--request',str(request)],cwd=ROOT,check=True)
        args.validation=output.with_suffix('.stage.json')
    if args.validation:
        report=json.loads(args.validation.read_text(encoding='utf-8'))
        output=Path(report.get('output_path',''))
        if not (report.get('engine')=='idol' and report.get('status')=='succeeded' and
                report.get('gaussians',0)>=32768 and report.get('geometry_generated') and
                output.is_file() and digest(output)==report.get('output_sha256')):
            raise ValueError('A real successful Idol mesh generation with matching output hash is required')
        files={p.relative_to(overlay).as_posix():digest(p) for p in overlay.rglob('*')
               if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
        atomic_json(runtime/'idol-runtime.json',{'ready':True,'source_revision':report['source_revision'],
            'requirements_sha256':digest(ROOT/'runtime_requirements/idol-windows.txt'),'files':files,
            'validation':report,'validation_sha256':digest(args.validation)})
        print('Recorded real Idol generation and private dependency hashes')


if __name__=='__main__':main()
