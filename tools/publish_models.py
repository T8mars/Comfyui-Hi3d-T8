"""Publish verified core/extension models in resumable per-component commits."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('HF_XET_CHUNK_CACHE_SIZE_BYTES', '0')
os.environ.setdefault('HF_XET_SHARD_CACHE_SIZE_LIMIT', '1073741824')
os.environ.setdefault('HF_XET_HIGH_PERFORMANCE', '1')
from huggingface_hub import HfApi, CommitOperationAdd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hi3d_workbench.common import atomic_json
from hi3d_workbench.pipeline_models import SPECS

REPO = 't8star/Hi3d-Comfy'


def hashes(path):
    sha, git = hashlib.sha256(), hashlib.sha1()
    git.update(f'blob {path.stat().st_size}\0'.encode('ascii'))
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(block)
            git.update(block)
    return sha.hexdigest(), git.hexdigest()


def matches(remote, meta):
    if remote is None or remote.size != meta['bytes']:
        return False
    return remote.lfs.sha256 == meta['sha256'] if remote.lfs else remote.blob_id == meta['git_sha1']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-extensions', action='store_true')
    parser.add_argument('--models-only', action='store_true')
    parser.add_argument('--preupload', action='store_true', help='Preupload missing weights concurrently before per-component commits')
    args = parser.parse_args()
    api = HfApi()
    assert api.whoami()['name'] == 't8star', 'Unexpected Hugging Face identity'
    groups, catalogue = {}, {}
    manifests = [('core', ROOT / 'models/manifest.json', ROOT / 'models', '')]
    if args.include_extensions:
        manifests.append(('extensions', ROOT / 'models/extensions/manifest.json', ROOT / 'models/extensions', 'extensions/'))
    output = ROOT / 'data/model-publication-v041'
    output.mkdir(parents=True, exist_ok=True)
    for kind, manifest_path, base, prefix in manifests:
        original = json.loads(manifest_path.read_text(encoding='utf-8'))
        clean = {'components': {}}
        for key, component in original['components'].items():
            items = []
            public = {name: value for name, value in component.items() if name != 'files'}
            public['files'] = {}
            if kind == 'extensions':
                for field in ('upstream_repo', 'upstream_revision', 'source_evidence', 'upstream_identical'):
                    if field in SPECS[key]:
                        public[field] = SPECS[key][field]
            for name, entry in component['files'].items():
                path = base / key / name
                assert path.stat().st_size == entry['bytes'], str(path)
                sha, git = hashes(path)
                expected = entry.get('sha256', entry.get('hash'))
                actual = git if entry.get('algorithm') == 'git-sha1' else sha
                assert expected == actual, str(path)
                remote_name = f'{prefix}{key}/{name}'
                meta = {'bytes': path.stat().st_size, 'sha256': sha, 'git_sha1': git,
                        'repo': component['repo'], 'revision': component['revision'], 'license': component['license']}
                catalogue[remote_name] = meta
                items.append((remote_name, path, meta))
                public['files'][name] = {k: v for k, v in entry.items() if k not in {'signature', 'verified'}}
            clean['components'][key] = public
            groups[f'{kind}/{key}'] = items
            print(f'Verified {kind}/{key}: {len(items)} files', flush=True)
        target = output / (kind + '-manifest.json')
        atomic_json(target, clean)
        groups[kind + '/manifest'] = [(prefix + 'manifest.json', target, {})]
    atomic_json(output / 'model-catalogue.json', {'schema': 1, 'files': catalogue})
    groups['catalogue'] = [('model-catalogue.json', output / 'model-catalogue.json', {})]
    licenses = [(f'licenses/{p.name}', p, {}) for p in (ROOT / 'licenses').iterdir() if p.is_file()]
    licenses += [('LICENSE', ROOT / 'models/hi3d/LICENSE', {}), ('NOTICE', ROOT / 'models/hi3d/NOTICE', {})]
    if not args.models_only:
        licenses += [('README.md', ROOT / 'docs/HUGGINGFACE.md', {}),
                     ('README.zh-CN.md', ROOT / 'docs/HUGGINGFACE.zh-CN.md', {})]
    groups = {'licenses-and-docs': licenses, **groups}
    preuploaded = {}
    if args.preupload:
        remote = {x.rfilename: x for x in api.model_info(REPO, files_metadata=True).siblings}
        for files in groups.values():
            for name, path, meta in files:
                if meta and meta['bytes'] > 10 * 1024 * 1024 and not matches(remote.get(name), meta):
                    preuploaded[name] = CommitOperationAdd(path_in_repo=name, path_or_fileobj=path)
        print(f'Preuploading {len(preuploaded)} missing weight files with Xet high-performance mode', flush=True)
        if preuploaded:
            api.preupload_lfs_files(REPO, list(preuploaded.values()), repo_type='model', num_threads=6)
        print('All weight preuploads completed', flush=True)
    proof = {'repo': REPO, 'verified_model_files': len(catalogue), 'model_bytes': sum(v['bytes'] for v in catalogue.values()), 'commits': []}
    for label, files in groups.items():
        info = api.model_info(REPO, files_metadata=True)
        remote = {x.rfilename: x for x in info.siblings}
        changed = []
        expected_files = []
        for name, path, meta in files:
            if not meta:
                sha, git = hashes(path)
                meta = {'sha256': sha, 'git_sha1': git, 'bytes': path.stat().st_size}
            expected_files.append((name, meta))
            if not matches(remote.get(name), meta):
                changed.append(preuploaded.get(name) or CommitOperationAdd(path_in_repo=name, path_or_fileobj=path))
        if changed:
            print(f'Uploading {label}: {len(changed)} changed files', flush=True)
            commit = api.create_commit(repo_id=REPO, repo_type='model', parent_commit=info.sha,
                operations=changed, commit_message=f'Publish verified {label} for Hi3D T8 workbench', num_threads=4)
            proof['commits'].append({'group': label, 'commit': commit.oid})
            remote = {x.rfilename: x for x in api.model_info(REPO, revision=commit.oid, files_metadata=True).siblings}
        for name, meta in expected_files:
            assert matches(remote.get(name), meta), name
        proof['last_verified_group'] = label
        atomic_json(output / 'upload-progress.json', proof)
        print(f'Remote verified: {label}', flush=True)
    final = api.model_info(REPO, files_metadata=True)
    remote = {x.rfilename: x for x in final.siblings}
    assert all(matches(remote.get(name), meta) for name, meta in catalogue.items())
    proof.update(commit=final.sha, all_remote_model_hashes_match=True, completed=True)
    atomic_json(output / 'publication.json', proof)
    print(json.dumps(proof, indent=2), flush=True)


if __name__ == '__main__':
    main()
