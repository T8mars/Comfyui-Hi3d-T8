import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hi3d_workbench.common import ROOT,MODELS
path=ROOT/'data/download-files.json'
if not path.exists():
    from huggingface_hub import HfApi
    from hi3d_workbench.models import SPECS
    files=[]
    for key,spec in SPECS.items():
        info=HfApi().model_info(spec['repo'],revision=spec.get('revision'),files_metadata=True)
        for f in info.siblings:
            if f.rfilename in spec['files'] and f.rfilename.endswith('.safetensors'):
                files.append({'path':key+'/'+f.rfilename,'sha256':f.lfs.sha256,'bytes':f.size})
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(files),encoding='utf-8')
files=json.loads(path.read_text(encoding='utf-8'))
values=[]
for f in files:
    final=MODELS/f['path'];parts=MODELS/'.downloads'/(f['sha256']+'-4m')
    count=f['bytes'] if final.is_file() and final.stat().st_size==f['bytes'] else sum(p.stat().st_size for p in parts.glob('*') if p.is_file())
    values.append({'file':f['path'],'received':min(count,f['bytes']),'total':f['bytes'],'materialized':final.is_file()})
print(json.dumps({'files':values,'received':sum(f['received'] for f in values),'total':sum(f['total'] for f in values)}))
