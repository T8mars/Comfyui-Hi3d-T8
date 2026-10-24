"""Package a prepared portable edition without local/private or obsolete files."""
import argparse
import json
import hashlib
import sys
import zipfile
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.build_release import ALLOWED

ROOTS={Path(name).parts[0] for name in ALLOWED}|{'runtime','runtime-pipeline','upstream','models','runtime-packages.txt','runtime-pipeline-packages.txt','config.local.json','bundle-manifest.json'}

def included(rel):
    return (rel.parts[0] in ROOTS and
        not any(p in {'.git','__pycache__','.secrets','.cache','.downloads','.pytest_cache'} or p=='.env' or p.startswith('.env.') or '.parts' in p or p.endswith('.assembled') for p in rel.parts) and
        rel.name not in {'download-state.json','.download.lock'} and
        rel.suffix not in {'.pyc','.tmp','.incomplete','.bak'} and rel.name!='hi3d_refine_api.json')

def build(release,output_dir=None):
    release=release.resolve()
    if not (release/'runtime/python.exe').is_file() or not (release/'models/manifest.json').is_file():raise ValueError('不是完整运行目录')
    parent=Path(output_dir).resolve() if output_dir else release.parent
    parent.mkdir(parents=True,exist_ok=True)
    target=parent/(release.name+'.zip');count=0
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for f in release.rglob('*'):
            rel=f.relative_to(release)
            if not f.is_file() or not included(rel):continue
            method=zipfile.ZIP_STORED if f.suffix=='.safetensors' else zipfile.ZIP_DEFLATED
            z.write(f,Path(release.name)/rel,compress_type=method);count+=1
            if count%2000==0:print(f'{count} files archived',flush=True)
    h=hashlib.sha256()
    with target.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    record={'file':target.name,'files':count,'bytes':target.stat().st_size,'sha256':h.hexdigest()}
    target.with_name(target.name+'.sha256.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    catalog=target.parent/'SHA256.json'
    values=json.loads(catalog.read_text(encoding='utf-8')) if catalog.exists() else {}
    values[target.name]={key:record[key] for key in ['bytes','sha256']}
    catalog.write_text(json.dumps(values,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2),flush=True)
    return target

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('release',type=Path);p.add_argument('--output-dir',type=Path);args=p.parse_args();build(args.release,args.output_dir)
