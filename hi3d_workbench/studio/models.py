"""Pinned, verified and separately managed asset-studio components."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

from .. import common
from ..common import ROOT, DATA, atomic_json, read_json

CATALOG = json.loads(Path(__file__).with_name('model-catalog.json').read_text(encoding='utf-8'))
FEATURES = {
    'simplify': {'name': '保材质减面', 'components': [], 'runtime': 'meshoptimizer'},
    'retopology': {'name': '四边面重拓扑', 'components': [], 'runtime': 'retopology'},
    'segment': {'name': 'P3-SAM 智能拆件', 'components': ['p3sam', 'sonata'], 'runtime': 'parts'},
    'complete_parts': {'name': 'X-Part 公开轻量部件补全', 'components': ['parts_full', 'p3sam', 'sonata'], 'runtime': 'parts'},
    'rig_skintokens': {'name': 'SkinTokens 自动绑定', 'components': ['skintokens', 'qwen_rig_config'], 'runtime': 'skintokens'},
    'rig_unirig': {'name': 'UniRig 自动绑定', 'components': ['unirig'], 'runtime': 'unirig'},
    'retarget': {'name': '单人动作导入与重定向', 'components': [], 'runtime': 'animation'},
    'motion': {'name': 'Kimodo 单人文本动作', 'components': ['kimodo', 'text_adapter', 'text_adapter_supervised', 'text_base'], 'runtime': 'kimodo'},
}
LOCK = threading.Lock()
_STATE = {'state': 'idle', 'message': ''}


def model_root():
    return common.MODELS / 'studio'


def runtime_root():
    return Path(os.environ.get('HI3D_STUDIO_RUNTIME', ROOT / 'runtime-assets')).resolve()


def runtime_python():
    override = os.environ.get('HI3D_STUDIO_PYTHON')
    if override:
        return Path(override).resolve()
    for name in ('python.exe', 'Scripts/python.exe', 'bin/python'):
        path = runtime_root() / name
        if path.is_file():
            return path
    return runtime_root() / 'python.exe'


def _read(path):
    try:
        value = read_json(path, {})
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _mapping(value):
    return value if isinstance(value, dict) else {}


def download_busy(root=None):
    """The OS file lock, rather than a stale progress JSON, proves ownership."""
    from filelock import FileLock, Timeout
    root = Path(root) if root is not None else model_root()
    if LOCK.locked():
        return True
    if not root.is_dir():
        return False
    guard = FileLock(str(root / '.download.lock'), thread_local=False)
    try:
        guard.acquire(timeout=0)
    except (Timeout, OSError):
        return True
    else:
        guard.release()
        return False


def download_state(root, busy):
    saved = _read(root / '.download-state.json')
    value = saved if saved.get('model_root') == str(root.resolve()) else dict(_STATE)
    if not busy and value.get('state') in {'downloading', 'verifying'}:
        return {**value, 'state': 'interrupted', 'message': '上次资产模型准备已中断，可继续下载或重新校验'}
    if busy and value.get('state') not in {'downloading', 'verifying'}:
        return {'state': 'busy', 'message': '资产模型下载或校验进程正在运行'}
    return value


def _path(root, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or not rel.parts or any(part in {'..', '.'} or ':' in part or '\\' in part for part in rel.parts):
        raise ValueError('模型文件路径无效')
    path = root.joinpath(*rel.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('模型文件路径超出组件目录')
    return path


def status():
    from ..pipeline_models import _signature
    root = model_root()
    manifest = _read(root / 'manifest.json')
    runtime = _read(runtime_root() / 'studio-runtime.json')
    components = {}
    for key, spec in CATALOG.items():
        entry = _mapping(_mapping(manifest.get('components')).get(key))
        valid = entry.get('revision') == spec['revision'] and entry.get('repo') == spec['repo']
        ready = 0
        for file in spec['files']:
            path = _path(root / key, file['name'])
            item = _mapping(_mapping(entry.get('files')).get(file['name']))
            signature = _signature(path)
            if (signature is not None and valid and item.get('hash') == file['hash']
                    and item.get('signature') == signature and item.get('verified') is True):
                ready += 1
        components[key] = {'name': key, 'repo': spec['repo'], 'revision': spec['revision'],
            'license': spec['license'], 'gated': spec['gated'], 'ready': ready == len(spec['files']),
            'verified_files': ready, 'total_files': len(spec['files']),
            'bytes': sum(file['size'] for file in spec['files']), 'path': str(root / key)}
    features = {}
    for key, spec in FEATURES.items():
        missing = [name for name in spec['components'] if not components[name]['ready']]
        runtime_ready = bool(runtime_python().is_file() and
            _mapping(_mapping(runtime.get('features')).get(spec['runtime'])).get('ready') is True)
        features[key] = {**spec, 'ready': runtime_ready and not missing,
                         'runtime_ready': runtime_ready, 'missing_components': missing}
    busy = download_busy(root)
    return {'root': str(root), 'components': components, 'features': features,
            'runtime': runtime, 'download': download_state(root, busy), 'busy': busy}


def _verify(path, file):
    from ..pipeline_models import _verified_signature
    return _verified_signature(path, file['hash'], file['algorithm'] == 'git-sha1', file['size'])


def verified_weight(component, name):
    """Reuse a pinned hash receipt only while the full file signature matches.

    Explicit verification still hashes every byte. Inference should not reread
    gigabytes solely for hashing after a verified download, before reading the
    same file again to load it. A missing/stale receipt always hashes afresh.
    """
    from ..pipeline_models import _signature
    spec = CATALOG[component]
    file = next(item for item in spec['files'] if item['name'] == name)
    path = _path(model_root() / component, name)
    manifest = _read(model_root() / 'manifest.json')
    entry = _mapping(_mapping(manifest.get('components')).get(component))
    receipt = _mapping(_mapping(entry.get('files')).get(name))
    signature = _signature(path)
    if (signature is not None and entry.get('revision') == spec['revision'] and entry.get('repo') == spec['repo']
            and receipt.get('verified') is True and receipt.get('hash') == file['hash']
            and receipt.get('signature') == signature):
        return path
    if _verify(path, file) is None:
        raise RuntimeError(f'资产模型尚未完整下载或已损坏：{component}/{name}')
    return path


def _transfer(spec, file, destination, callback):
    """Resume bounded ranges; verify the complete file before publishing it."""
    import requests
    from huggingface_hub import get_token, hf_hub_url
    if destination.is_file() and _verify(destination, file) is not None:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = hf_hub_url(spec['repo'], file['name'], revision=spec['revision'])
    token = get_token()
    auth = {'Authorization': 'Bearer ' + token} if token else {}
    size = file['size']
    temporary = destination.with_name(destination.name + '.download')
    if size < 32 * 1024 * 1024:
        with requests.get(url, headers=auth, stream=True, timeout=(20, 90)) as response:
            response.raise_for_status()
            done = 0
            with temporary.open('wb') as stream:
                for block in response.iter_content(256 * 1024):
                    done += len(block)
                    if done > size:
                        raise ValueError('下载响应超出固定文件长度')
                    stream.write(block)
                stream.flush(); os.fsync(stream.fileno())
    else:
        chunk = 4 * 1024 * 1024
        parts = destination.parent / ('.' + destination.name + '.parts')
        parts.mkdir(exist_ok=True)
        if parts.resolve().parent != destination.parent.resolve():
            raise ValueError('下载分片目录不能指向组件目录以外的位置')
        total = (size + chunk - 1) // chunk
        def fetch(index):
            start = index * chunk
            stop = min(size, start + chunk)
            part = parts / str(index)
            if part.is_file() and part.stat().st_size == stop - start:
                return stop - start
            temp = part.with_suffix('.tmp')
            for attempt in range(5):
                try:
                    headers = {**auth, 'Range': f'bytes={start}-{stop - 1}'}
                    with requests.get(url + f'?download=true&part={index}', headers=headers,
                                      stream=True, timeout=(20, 90)) as response:
                        response.raise_for_status()
                        if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{stop - 1}/{size}':
                            raise ValueError('服务器未返回指定下载区间')
                        received = 0
                        with temp.open('wb') as stream:
                            for block in response.iter_content(256 * 1024):
                                received += len(block)
                                if received > stop - start:
                                    raise ValueError('下载区间响应过长')
                                stream.write(block)
                            stream.flush(); os.fsync(stream.fileno())
                        if received != stop - start:
                            raise ValueError('下载区间不完整')
                        os.replace(temp, part)
                    return stop - start
                except Exception:
                    temp.unlink(missing_ok=True)
                    if attempt == 4:
                        raise
                    time.sleep(min(8, 2 ** attempt))
        done = 0
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(fetch, index) for index in range(total)]
            for future in as_completed(futures):
                done += future.result()
                callback(done, size)
        with temporary.open('wb') as stream:
            for index in range(total):
                with (parts / str(index)).open('rb') as part:
                    for block in iter(lambda: part.read(chunk), b''):
                        stream.write(block)
            stream.flush(); os.fsync(stream.fileno())
    if _verify(temporary, file) is None:
        temporary.unlink(missing_ok=True)
        if size >= 32 * 1024 * 1024:
            # Correct byte lengths are not hashes: an interrupted/bad cached
            # part must not poison every subsequent resume of the same file.
            for index in range(total):
                (parts / str(index)).unlink(missing_ok=True)
        raise RuntimeError(f'模型内容校验失败：{file["name"]}')
    os.replace(temporary, destination)
    if size >= 32 * 1024 * 1024:
        for index in range(total):
            (parts / str(index)).unlink(missing_ok=True)
        try:
            parts.rmdir()
        except OSError:
            pass
    callback(size, size)


def selection_components(selection):
    if selection == 'all':
        return list(CATALOG)
    if selection in FEATURES:
        return FEATURES[selection]['components']
    if selection in CATALOG:
        return [selection]
    raise ValueError('无效资产模型选择')


def download(selection='all', callback=None):
    from filelock import FileLock
    keys = selection_components(selection)
    root = model_root()
    root.mkdir(parents=True, exist_ok=True)
    with LOCK, FileLock(str(root / '.download.lock'), timeout=0):
        manifest = _read(root / 'manifest.json')
        entries = manifest['components'] = _mapping(manifest.get('components'))
        _STATE.clear()
        _STATE.update(state='downloading', selection=selection, model_root=str(root.resolve()),
                      message='正在下载并校验资产组件')
        last = [0.0]
        def notify(current=0, total=0):
            _STATE.update(file_bytes=current, file_total=total)
            if time.monotonic() - last[0] > 1 or current == total:
                atomic_json(DATA / 'studio-download.json', _STATE)
                atomic_json(root / '.download-state.json', _STATE)
                if callback:
                    callback(dict(_STATE))
                last[0] = time.monotonic()
        try:
            for key in keys:
                spec = CATALOG[key]
                entry = entries.setdefault(key, {'repo': spec['repo'], 'revision': spec['revision'], 'files': {}})
                if not isinstance(entry, dict):
                    entry = entries[key] = {}
                if entry.get('repo') != spec['repo'] or entry.get('revision') != spec['revision']:
                    entry = entries[key] = {'repo': spec['repo'], 'revision': spec['revision'], 'files': {}}
                entry['files'] = _mapping(entry.get('files'))
                for file in spec['files']:
                    _STATE.update(component=key, file=file['name'], message=f'下载 / 校验 {key}/{file["name"]}')
                    path = _path(root / key, file['name'])
                    _transfer(spec, file, path, notify)
                    signature = _verify(path, file)
                    if signature is None:
                        raise RuntimeError('模型下载后发生变化，请重新校验')
                    entry['files'][file['name']] = {'hash': file['hash'], 'signature': signature, 'verified': True}
                    atomic_json(root / 'manifest.json', manifest)
            _STATE.update(state='complete', message='资产模型下载与校验完成')
        except Exception as exc:
            _STATE.update(state='failed', message=str(exc))
            raise
        finally:
            notify()
    return status()


def pending_verification():
    """Installed copies need fresh receipts; absent weights never auto-download."""
    root=model_root();current=status()['components']
    return [key for key,spec in CATALOG.items() if not current[key]['ready']
            and all(_path(root/key,file['name']).is_file() for file in spec['files'])]


def verify(selection='all', *, installed_only=False, callback=None):
    from filelock import FileLock
    root = model_root()
    root.mkdir(parents=True, exist_ok=True)
    errors = []
    with LOCK, FileLock(str(root / '.download.lock'), timeout=0):
        manifest = _read(root / 'manifest.json')
        entries = manifest['components'] = _mapping(manifest.get('components'))
        keys=selection_components(selection)
        if installed_only:
            pending=set(pending_verification())
            keys=[key for key in keys if key in pending]
        if keys:
            _STATE.clear();_STATE.update(state='verifying',model_root=str(root.resolve()),
                verified_files=0,total_files=sum(len(CATALOG[key]['files']) for key in keys))
            def notify(message):
                _STATE['message']=message
                atomic_json(root/'.download-state.json',_STATE)
                if callback:callback(dict(_STATE))
            try:
                for key in keys:
                    spec = CATALOG[key]
                    entry = entries[key] = {'repo': spec['repo'], 'revision': spec['revision'], 'files': {}}
                    for file in spec['files']:
                        notify(f'校验本地资产模型 {_STATE["verified_files"]+1}/{_STATE["total_files"]} · {key}/{file["name"]}')
                        signature = _verify(_path(root / key, file['name']), file)
                        if signature is None:errors.append(key + '/' + file['name'])
                        entry['files'][file['name']] = {'hash': file['hash'], 'signature': signature, 'verified': signature is not None}
                        _STATE['verified_files']+=1
                atomic_json(root / 'manifest.json', manifest)
                _STATE['state']='failed' if errors else 'complete'
                notify(f'资产本地文件校验失败：{len(errors)} 项' if errors else '已安装资产模型校验完成')
            except Exception as exc:
                _STATE['state']='failed';notify(str(exc));raise
    return {**status(), 'verification_errors': errors}


if __name__ == '__main__':
    common.configure_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['status', 'download', 'verify'])
    parser.add_argument('--selection', default='all')
    args = parser.parse_args()
    if args.action == 'download':
        result = download(args.selection, callback=lambda value: print(json.dumps(value, ensure_ascii=False), flush=True))
    elif args.action == 'verify':
        result = verify(args.selection)
    else:
        result = status()
    print(json.dumps(result, ensure_ascii=False, indent=2))
