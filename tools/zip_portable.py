import argparse
import json
import hashlib
import zipfile
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('release',type=Path);args=p.parse_args()
release=args.release.resolve()
if not (release/'runtime/python.exe').is_file() or not (release/'models/manifest.json').is_file():raise SystemExit('不是完整运行目录')
target=release.with_name(release.name+'.zip')
count=0
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
    for f in release.rglob('*'):
        rel=f.relative_to(release)
        if not f.is_file() or rel.parts[0] in {'data','.git'} or '__pycache__' in rel.parts or f.suffix in {'.pyc','.tmp','.incomplete'}:continue
        # Tensor weights compress poorly; store them directly for faster packaging/extraction.
        method=zipfile.ZIP_STORED if f.suffix=='.safetensors' else zipfile.ZIP_DEFLATED
        z.write(f,Path(release.name)/rel,compress_type=method);count+=1
        if count%2000==0:print(f'{count} files archived',flush=True)
h=hashlib.sha256()
with target.open('rb') as f:
    for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
manifest={'file':target.name,'files':count,'bytes':target.stat().st_size,'sha256':h.hexdigest()}
target.with_name(target.name+'.sha256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
catalog=target.parent/'SHA256.json'
values=json.loads(catalog.read_text(encoding='utf-8')) if catalog.exists() else {}
values[target.name]={key:manifest[key] for key in ['bytes','sha256']}
catalog.write_text(json.dumps(values,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2),flush=True)
