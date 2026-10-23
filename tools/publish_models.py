"""Upload the exact verified offline-model files with the official Hub client."""
import hashlib
import json
import argparse
import sys
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hi3d_workbench.common import VERSION
REPO = 't8star/Hi3d-Comfy'

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--offline-release',type=Path,default=ROOT/'dist'/f'Hi3D-Workbench-Windows-{VERSION}');args=parser.parse_args()
    api=HfApi()
    assert api.whoami()['name']=='t8star','Unexpected Hugging Face identity'
    manifest=json.loads((ROOT/'models/manifest.json').read_text(encoding='utf-8'))
    offline=json.loads((args.offline_release/'models/manifest.json').read_text(encoding='utf-8'))
    assert manifest['components']==offline['components'],'Models differ from offline edition'
    files={}
    for key,component in manifest['components'].items():
        for name,entry in component['files'].items():
            path=ROOT/'models'/key/name
            assert path.stat().st_size==entry['bytes'] and sha256(path)==entry['sha256'],str(path)
            files[f'{key}/{name}']=path
    files.update({'manifest.json':ROOT/'models/manifest.json',
                  'README.md':ROOT/'docs/HUGGINGFACE.md',
                  'README.zh-CN.md':ROOT/'docs/HUGGINGFACE.zh-CN.md',
                  'LICENSE':ROOT/'models/hi3d/LICENSE',
                  'NOTICE':ROOT/'models/hi3d/NOTICE'})
    for path in (ROOT/'licenses').glob('*'):
        if path.is_file():files[f'licenses/{path.name}']=path
    print(f'Verified 16 model artifacts; uploading {len(files)} files to {REPO}',flush=True)
    info=api.model_info(REPO)
    commit=api.create_commit(repo_id=REPO,repo_type='model',parent_commit=info.sha,
        operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=path) for name,path in files.items()],
        commit_message='Publish verified Hi3D V1.0 model bundle and bilingual documentation',num_threads=2)
    print('Commit:',commit.oid,flush=True)
    remote={x.rfilename:x for x in api.model_info(REPO,revision=commit.oid,files_metadata=True).siblings}
    checked=0
    for name,path in files.items():
        entry=remote[name]
        assert entry.size==path.stat().st_size,name
        if entry.lfs:
            assert entry.lfs.sha256==sha256(path),name
        else:
            actual=Path(hf_hub_download(REPO,name,revision=commit.oid))
            assert sha256(actual)==sha256(path),name
        checked+=1
    out=ROOT/'data/qa/huggingface-publication.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'repo':REPO,'commit':commit.oid,'verified_files':checked,
        'model_artifacts':16,'identical_to_offline_edition':True,'url':str(commit.commit_url)},indent=2),encoding='utf-8')
    print(out.read_text(encoding='utf-8'),flush=True)

if __name__=='__main__':main()
