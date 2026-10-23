"""Serve the current UI with disposable data and no installed model files."""
import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8379);args=parser.parse_args()
sandbox=Path(tempfile.mkdtemp(prefix='hi3d-ui-audit-'))
(sandbox/'samples').mkdir()
for name in ['sculpted-vase.png','sculpted-vase.glb']:shutil.copy2(root/'samples'/name,sandbox/'samples'/name)
os.environ['HI3D_WORKBENCH_ROOT']=str(sandbox)
print(f'Disposable audit data: {sandbox}',flush=True)
import uvicorn
uvicorn.run('hi3d_workbench.server:app',host='127.0.0.1',port=args.port,access_log=False)
