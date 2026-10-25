"""Code-only GitHub Release updates, staged online and installed while stopped."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path, PurePosixPath

import httpx
from filelock import FileLock, Timeout
from .common import ROOT, VERSION, atomic_json, config

REPOSITORY = 'T8mars/Comfyui-Hi3d-T8'
API = f'https://api.github.com/repos/{REPOSITORY}/releases/latest'
DOWNLOAD = f'https://github.com/{REPOSITORY}/releases/download/'
RUNTIME_API = 1
MAX_ARCHIVE = 128 * 1024 * 1024
MAX_EXPANDED = 512 * 1024 * 1024
CODE_DIRS = {'hi3d_workbench', 'scripts', 'examples', 'docs', 'licenses', 'samples', 'tools', 'tests', 'validation'}
CODE_FILES = {'__init__.py', 'comfy_nodes.py', 'pyproject.toml', 'Start-Workbench.cmd', 'Update-Workbench.cmd',
              'requirements.txt', 'requirements-worker-windows.txt', 'requirements-pipeline-windows.txt',
              'README.md', 'README.zh-CN.md', 'LICENSE', 'NOTICE.md', 'VALIDATION.md', 'roadmap.md',
              'upstream-source-manifest.json', 'config.example.json'}
WEIGHT_SUFFIXES = {'.safetensors', '.ckpt', '.pt', '.pth', '.onnx', '.bin', '.exe', '.dll', '.pyd', '.whl'}
_WORKER = threading.Lock()
_STOP = threading.Event()
_SCHEDULER = None


def version(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d+\.\d+\.\d+', value):
        raise ValueError('更新版本号无效')
    return tuple(map(int, value.split('.')))


def owned_name(name):
    if not isinstance(name, str) or not name or '\\' in name or ':' in name or '\x00' in name:
        raise ValueError('更新包含无效路径')
    parts = PurePosixPath(name).parts
    if str(PurePosixPath(name)) != name or name.startswith('/') or any(part in {'.', '..'} or part.endswith((' ', '.')) for part in name.split('/')):
        raise ValueError('更新路径超出代码目录')
    if any(part.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(10)), *(f'LPT{i}' for i in range(10))} for part in parts):
        raise ValueError('更新包含保留文件名')
    if name not in CODE_FILES and parts[0] not in CODE_DIRS and not name.startswith('upstream/Hi3Dv1.0/'):
        raise ValueError('更新不能包含模型、运行时或个人配置')
    if Path(name).suffix.lower() in WEIGHT_SUFFIXES or any(part.startswith('.') or part == '__pycache__' for part in parts):
        raise ValueError('代码更新不能包含模型或 Python 运行时')
    return name


def target_path(root, name):
    owned_name(name)
    path = root.joinpath(*PurePosixPath(name).parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('更新路径包含目录外的链接')
    return path


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def directory(root):
    return Path(root) / 'data/updates'


def read(path, default=None):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else (default or {})
    except (OSError, ValueError):
        return default or {}


def preferences(root=ROOT):
    value = read(directory(root) / 'preferences.json')
    # A development checkout must not silently overwrite tracked edits.
    return {'auto_update': value.get('auto_update', True) is True and not (Path(root) / '.git').exists()}


def set_preferences(enabled, root=ROOT):
    if type(enabled) is not bool:
        raise ValueError('自动更新设置必须为布尔值')
    if enabled and (Path(root) / '.git').exists():
        raise ValueError('源码 Git 目录请使用 Git 更新；自动安装适用于独立整合包')
    atomic_json(directory(root) / 'preferences.json', {'auto_update': enabled})
    return status(root)


def status(root=ROOT):
    value = read(directory(root) / 'state.json', {'phase': 'idle', 'message': '尚未检查软件更新'})
    value.update(current_version=VERSION, **preferences(root), source_checkout=(Path(root) / '.git').exists())
    return value


def state(root, phase, message, **fields):
    value = {'phase': phase, 'message': message, 'checked_at': time.time(), **fields}
    atomic_json(directory(root) / 'state.json', value)
    return value


def client():
    return httpx.Client(timeout=httpx.Timeout(20, read=60), follow_redirects=True,
                        trust_env=config().get('download_trust_env', True),
                        headers={'Accept': 'application/vnd.github+json', 'User-Agent': 'Hi3D-T8-Updater'})


def get_json(url, limit=4 * 1024 * 1024):
    with client() as transport, transport.stream('GET', url) as response:
        response.raise_for_status()
        if response.url.scheme != 'https':
            raise ValueError('更新地址必须使用 HTTPS')
        content = bytearray()
        for block in response.iter_bytes():
            content.extend(block)
            if len(content) > limit:
                raise ValueError('更新元数据过大')
    return json.loads(content)


def validate_manifest(value):
    if not isinstance(value, dict) or value.get('schema') != 1 or value.get('repository') != REPOSITORY:
        raise ValueError('更新清单来源或格式不符')
    release = value.get('version')
    version(release)
    if value.get('runtime_api') != RUNTIME_API:
        raise ValueError('新版需要不同的 Python/依赖环境，请获取作者提供的新版完整版')
    if version(VERSION) < version(value.get('minimum_version', '0.4.0')):
        raise ValueError('当前整合包太旧，请先获取作者提供的新版完整版')
    if value.get('archive') != f'Hi3D-Workbench-update-{release}.zip':
        raise ValueError('更新压缩包名称不符')
    if not re.fullmatch('[0-9a-f]{64}', str(value.get('sha256', ''))):
        raise ValueError('更新缺少 SHA256')
    size = value.get('bytes')
    if type(size) is not int or not 0 < size <= MAX_ARCHIVE:
        raise ValueError('更新包大小无效')
    files = value.get('files')
    if not isinstance(files, dict) or not 1 <= len(files) <= 3000:
        raise ValueError('更新文件清单无效')
    seen, total = set(), 0
    for name, meta in files.items():
        owned_name(name)
        if name.casefold() in seen:
            raise ValueError('更新包含大小写冲突路径')
        seen.add(name.casefold())
        if not isinstance(meta, dict) or type(meta.get('bytes')) is not int or meta['bytes'] < 0:
            raise ValueError('更新文件大小无效')
        if not re.fullmatch('[0-9a-f]{64}', str(meta.get('sha256', ''))):
            raise ValueError('更新文件缺少校验值')
        total += meta['bytes']
    if total > MAX_EXPANDED:
        raise ValueError('更新解压大小过大')
    for name in files:
        if any(str(parent).casefold() in seen for parent in PurePosixPath(name).parents if str(parent) != '.'):
            raise ValueError('更新文件与目录冲突')
    return value


def download(archive_url, manifest, target):
    expected = DOWNLOAD + f"v{manifest['version']}/{manifest['archive']}"
    if archive_url != expected:
        raise ValueError('只允许本项目 GitHub Release 更新包')
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix('.part')
    size = 0
    with client() as transport, transport.stream('GET', archive_url) as response:
        response.raise_for_status()
        if response.url.scheme != 'https':
            raise ValueError('更新下载必须使用 HTTPS')
        with partial.open('wb') as stream:
            for block in response.iter_bytes(1024 * 1024):
                size += len(block)
                if size > manifest['bytes']:
                    raise ValueError('更新下载大小不符')
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
    if size != manifest['bytes'] or sha256(partial) != manifest['sha256']:
        raise ValueError('更新下载 SHA256 校验失败，原程序未修改')
    os.replace(partial, target)


def check(root=ROOT, prepare=True):
    root = Path(root)
    updates = directory(root)
    updates.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(updates / 'download.lock', timeout=0):
            pending = read(updates / 'pending.json')
            if pending and version(pending.get('version', '0.0.0')) > version(VERSION):
                try:
                    validate_manifest(pending)
                    validate_archive(updates / pending['archive'], pending, root)
                except (OSError, ValueError, zipfile.BadZipFile):
                    os.replace(updates / 'pending.json', updates / f'invalid-pending-{uuid.uuid4().hex}.json')
                else:
                    return state(root, 'staged', '更新已下载；关闭工作台后，下次启动自动安装', latest_version=pending['version'])
            state(root, 'checking', '正在检查 GitHub Release')
            release = get_json(API)
            tag = release.get('tag_name', '')
            latest = tag.removeprefix('v')
            if release.get('draft') or release.get('prerelease'):
                raise ValueError('只安装正式发行版')
            version(latest)
            if version(latest) <= version(VERSION):
                return state(root, 'current', '当前已是最新程序版本', latest_version=latest)
            prefix = DOWNLOAD + tag + '/'
            asset = next((a for a in release.get('assets', []) if a.get('name') == 'update-manifest.json'), None)
            if not asset or asset.get('browser_download_url') != prefix + 'update-manifest.json':
                raise ValueError('新版尚未提供代码更新包，请稍后再试')
            manifest = validate_manifest(get_json(asset['browser_download_url']))
            if manifest['version'] != latest:
                raise ValueError('Release 与更新版本不一致')
            archive_asset = next((a for a in release['assets'] if a.get('name') == manifest['archive']), None)
            if not archive_asset or archive_asset.get('size') != manifest['bytes']:
                raise ValueError('Release 更新附件不完整')
            if archive_asset.get('digest') and archive_asset['digest'] != 'sha256:' + manifest['sha256']:
                raise ValueError('GitHub 附件摘要与更新清单不一致')
            if not prepare:
                return state(root, 'available', '发现新版，可以下载代码更新', latest_version=latest)
            state(root, 'downloading', '正在下载程序更新；不下载模型或 Python', latest_version=latest)
            archive = updates / manifest['archive']
            download(archive_asset['browser_download_url'], manifest, archive)
            validate_archive(archive, manifest, root)
            atomic_json(updates / 'pending.json', manifest)
            return state(root, 'staged', '更新已校验；关闭工作台后，下次启动自动安装', latest_version=latest)
    except Timeout:
        return status(root)
    except Exception as exc:
        return state(root, 'error', str(exc))


def validate_archive(archive, manifest, root):
    validate_manifest(manifest)
    if archive.stat().st_size != manifest['bytes'] or sha256(archive) != manifest['sha256']:
        raise ValueError('待安装更新包校验失败')
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        if len(entries) != len(manifest['files']) or {entry.filename for entry in entries} != set(manifest['files']):
            raise ValueError('更新包内容与清单不符')
        for entry in entries:
            target_path(root, entry.filename)
            if entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16) or entry.flag_bits & 1:
                raise ValueError('更新不能包含符号链接、目录条目或加密文件')
            expected = manifest['files'][entry.filename]
            if entry.file_size != expected['bytes']:
                raise ValueError('更新文件大小不符')
            with bundle.open(entry) as stream:
                value = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    value.update(block)
                if value.hexdigest() != expected['sha256']:
                    raise ValueError('更新文件 SHA256 不符')


def replace_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix='.update-tmp')
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def replace_code(root, name, content):
    target = target_path(root, name)
    replace_bytes(target, content)
    # A same-size edit within one timestamp second must not reuse stale bytecode.
    if target.suffix == '.py':
        for cached in (target.parent / '__pycache__').glob(target.stem + '.*.pyc'):
            if cached.resolve().is_relative_to(root.resolve()):
                cached.unlink()


def rollback(root, journal):
    if not re.fullmatch('[0-9a-f]{32}', str(journal.get('id', ''))) or not isinstance(journal.get('before'), dict):
        raise ValueError('更新恢复记录损坏，需要人工检查备份')
    backup = directory(root) / 'backups' / journal['id']
    for name, entry in reversed(list(journal['before'].items())):
        target = target_path(root, name)
        if entry['existed']:
            source = target_path(backup, name)
            if sha256(source) != entry['sha256']:
                raise ValueError('更新备份校验失败，需要人工恢复')
            replace_code(root, name, source.read_bytes())
        elif target.is_file():
            target.unlink()
    journal['phase'] = 'rolled_back'
    atomic_json(directory(root) / 'transaction.json', journal)


def require_no_workers(root):
    """A crashed server can leave a GPU worker alive after its lock is released."""
    import psutil
    modules = {'hi3d_workbench.worker', 'hi3d_workbench.pipeline_worker', 'hi3d_workbench.engines.run_stage'}
    data = (root / 'data').resolve()
    for process in psutil.process_iter(['cmdline']):
        try:
            command = process.info.get('cmdline') or []
            if '-m' not in command or command[command.index('-m') + 1] not in modules:
                continue
            if any(Path(arg).is_absolute() and Path(arg).resolve().is_relative_to(data) for arg in command):
                raise ValueError('本目录仍有推理或诊断工作进程运行，请等待其退出后再安装更新')
        except (psutil.NoSuchProcess, psutil.AccessDenied, IndexError):
            continue


def apply_pending(root=ROOT):
    root = Path(root).resolve()
    updates = directory(root)
    if not (updates / 'pending.json').is_file() and not (updates / 'transaction.json').is_file():
        return False
    if (root / '.git').exists():
        raise ValueError('源码 Git 目录不自动覆盖，请使用 Git 更新')
    with FileLock(root / 'data/server.lock', timeout=0), FileLock(updates / 'download.lock', timeout=0):
        require_no_workers(root)
        journal = read(updates / 'transaction.json')
        if (updates / 'transaction.json').is_file() and journal.get('phase') not in {'installing', 'complete', 'rolled_back'}:
            raise ValueError('更新事务记录损坏，请检查 data/updates/backups 后恢复')
        if journal.get('phase') == 'installing':
            rollback(root, journal)
            # common.py may have been imported from the interrupted new version.
            # Reload the process before comparing versions or serving requests.
            return True
        manifest = read(updates / 'pending.json')
        if not manifest:
            return False
        validate_manifest(manifest)
        if version(manifest['version']) <= version(VERSION):
            (updates / 'pending.json').unlink(missing_ok=True)
            return False
        archive = updates / manifest['archive']
        validate_archive(archive, manifest, root)
        identity = uuid.uuid4().hex
        backup = updates / 'backups' / identity
        before = {}
        for name in manifest['files']:
            target = target_path(root, name)
            if target.exists() and not target.is_file():
                raise ValueError('代码文件路径被目录占用')
            before[name] = {'existed': target.is_file()}
            if target.is_file():
                content = target.read_bytes()
                before[name]['sha256'] = hashlib.sha256(content).hexdigest()
                replace_bytes(target_path(backup, name), content)
        journal = {'id': identity, 'phase': 'installing', 'version': manifest['version'], 'before': before}
        atomic_json(updates / 'transaction.json', journal)
        try:
            with zipfile.ZipFile(archive) as bundle:
                for name in manifest['files']:
                    replace_code(root, name, bundle.read(name))
            journal['phase'] = 'complete'
            atomic_json(updates / 'transaction.json', journal)
            (updates / 'pending.json').unlink(missing_ok=True)
            state(root, 'installed', '程序更新已安装，模型、Python、配置与历史任务保留', latest_version=manifest['version'])
            return True
        except BaseException:
            rollback(root, journal)
            raise


def recovery_required(root=ROOT):
    path = directory(root) / 'transaction.json'
    return path.is_file() and read(path).get('phase') not in {'complete', 'rolled_back'}


def apply_on_start(root=ROOT):
    if not preferences(root)['auto_update'] and not recovery_required(root):
        return False
    return apply_pending(root)


def start_check(root=ROOT):
    if not _WORKER.acquire(blocking=False):
        return False
    def run():
        try:
            check(root)
        finally:
            _WORKER.release()
    threading.Thread(target=run, daemon=True, name='hi3d-update').start()
    return True


def start_background(is_busy=lambda: False):
    global _SCHEDULER
    if _SCHEDULER is not None and _SCHEDULER.is_alive():
        return
    _STOP.clear()
    def run():
        while not _STOP.is_set():
            if preferences()['auto_update'] and not is_busy():
                current = status()
                if current.get('phase') != 'staged' and time.time() - current.get('checked_at', 0) >= 6 * 3600:
                    start_check()
            _STOP.wait(60)
    _SCHEDULER = threading.Thread(target=run, daemon=True, name='hi3d-update-scheduler')
    _SCHEDULER.start()


def stop_background():
    _STOP.set()
    if _SCHEDULER is not None and _SCHEDULER is not threading.current_thread():
        _SCHEDULER.join(timeout=1)


def main():
    parser = argparse.ArgumentParser(description='Hi3D code-only updater; close the workbench before installing')
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    result = check(prepare=not args.check_only)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    if result.get('phase') == 'error':
        raise SystemExit(1)
    if args.install:
        apply_pending()


if __name__ == '__main__':
    main()
