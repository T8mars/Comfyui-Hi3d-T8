"""Probe the packaged runtime with fresh caches and no host CUDA/Python paths."""
import json
import argparse
import os
import subprocess
import uuid
from pathlib import Path

root=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(root))
from hi3d_workbench.common import VERSION
parser=argparse.ArgumentParser();parser.add_argument('release',nargs='?',type=Path,default=root/f'dist/Hi3D-Workbench-Windows-{VERSION}');args=parser.parse_args()
release=args.release.resolve()
env={k:v for k,v in os.environ.items() if k not in {'PYTHONPATH','PYTHONHOME','CUDA_PATH','CUDA_HOME','CUDA_PATH_V12_8','CUDA_PATH_V13_0'}}
env.update(PATH=str(release/'runtime')+os.pathsep+str(Path(os.environ['SystemRoot'])/'System32'),
           HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
           TRITON_CACHE_DIR=str(root/'data/qa'/('fresh-triton-'+uuid.uuid4().hex)))
p=subprocess.run([str(release/'runtime/python.exe'),'-m','hi3d_workbench.worker','--diagnose','--probe'],cwd=release,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
try:result=json.loads(p.stdout.strip().splitlines()[-1])
except Exception:raise RuntimeError(p.stdout+p.stderr)
result['isolated_from_host_paths']=True
result['fresh_kernel_cache']=True
(root/'data/qa/portable-clean-probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
assert result['kernel_probe']=='passed'
