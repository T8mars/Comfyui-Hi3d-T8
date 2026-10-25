from __future__ import annotations
import json
import threading
import time
import os
import math
import hashlib
import stat
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from .common import MODELS, DEFAULT_MODELS, DATA, MODEL_REVISION, atomic_json, read_json, digest, config

SPECS = {
    "hi3d": {"repo": "hi3d-official/Hi3Dv1.0", "revision": MODEL_REVISION,
             "files": ["model_index.json", "scheduler/scheduler_config.json", "vae/config.json", "vae/model.safetensors", "transformer_512/config.json", "transformer_512/model.safetensors", "transformer_1024/config.json", "transformer_1024/model.safetensors"],
             "license": "Hi3D Open License v1.0"},
    "dino": {"repo": "facebook/dinov2-large", "revision":"47b73eefe95e8d44ec3623f8890bd894b6ea2d6c", "files": ["config.json", "model.safetensors"], "license": "Apache-2.0"},
    "clip": {"repo": "openai/clip-vit-large-patch14", "revision":"32bd64288804d66eefd0ccbe215aa642df71cc41", "files": ["config.json", "model.safetensors"], "license": "MIT"}
}
LOCK = threading.Lock()
STATE = {"state": "idle", "message": "模型仅包含 Hi3D 及其必需的图像编码器"}
FILE_LOCK = None


@contextmanager
def verification_read_guard(path):
    """Prevent writes/deletion while hashing on Windows, where timestamps can coalesce."""
    if os.name!='nt':
        yield
        return
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,
        ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    kernel.CreateFileW.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    kernel.CloseHandle.restype=wintypes.BOOL
    handle=kernel.CreateFileW(str(path),0x80000000,1,None,3,0x80,None)
    if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    try:yield
    finally:kernel.CloseHandle(handle)

def _read_model_state(path,kind,errors=None):
    """Only disposable model metadata gets recovery; application JSON stays strict."""
    try:
        value=read_json(path,{})
        if not isinstance(value,dict):raise ValueError('应为 JSON 对象')
        if kind in {'manifest','verification'}:
            components=value.get('components',{})
            if not isinstance(components,dict) or any(not isinstance(entry,dict) for entry in components.values()):raise ValueError('components 格式无效')
            for entry in components.values():
                if kind=='manifest':
                    files=entry.get('files',{})
                    if not isinstance(files,dict) or any(not isinstance(item,dict) for item in files.values()):raise ValueError('files 格式无效')
                else:
                    if not isinstance(entry.get('snapshot',{}),dict) or not isinstance(entry.get('errors',[]),list) or any(not isinstance(item,str) for item in entry.get('errors',[])):raise ValueError('校验记录格式无效')
        elif value and (value.get('state') not in {'idle','downloading','interrupted','complete','failed'} or not isinstance(value.get('message',''),str)):
            raise ValueError('下载状态格式无效')
        return value
    except (OSError,ValueError,TypeError) as exc:
        if errors is not None:errors.append(f'{path.name} 损坏或不可读：{exc}')
        return {}

def _backup_invalid_state(path,kind):
    errors=[]
    _read_model_state(path,kind,errors)
    if errors and path.is_file():
        import shutil
        folder=DATA/'model-state-backups';folder.mkdir(parents=True,exist_ok=True)
        backup=folder/f'{path.name}.{digest(path)[:12]}.invalid.json'
        if not backup.exists():shutil.copy2(path,backup)

def _json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')).hexdigest()

def _metadata_errors(entry,spec):
    errors=[]
    if entry.get('repo')!=spec['repo']:errors.append('manifest:repo')
    if entry.get('revision')!=spec['revision']:errors.append('manifest:revision')
    return errors

def _git_blob_digest(path):
    """Hub non-LFS files are Git blobs, whose IDs include a length header."""
    value=hashlib.sha1()
    value.update(f'blob {path.stat().st_size}\0'.encode('ascii'))
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()

def _windows_change_time(path):
    # Windows st_ctime is the creation time on supported Python versions. Use
    # NTFS ChangeTime too, so replacing content then restoring mtime invalidates
    # a previous verification without reading gigabytes during health polling.
    import ctypes
    from ctypes import wintypes
    class BasicInfo(ctypes.Structure):
        _fields_=[('creation',ctypes.c_longlong),('access',ctypes.c_longlong),
                  ('write',ctypes.c_longlong),('change',ctypes.c_longlong),
                  ('attributes',wintypes.DWORD)]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,
                                ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    kernel.CreateFileW.restype=wintypes.HANDLE
    kernel.GetFileInformationByHandleEx.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
    kernel.GetFileInformationByHandleEx.restype=wintypes.BOOL
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    kernel.CloseHandle.restype=wintypes.BOOL
    handle=kernel.CreateFileW(str(path),0x80,7,None,3,0,None)
    if handle==ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        info=BasicInfo()
        if not kernel.GetFileInformationByHandleEx(handle,0,ctypes.byref(info),ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        return info.change
    finally:
        kernel.CloseHandle(handle)

def _file_signature(path):
    try:
        info=path.stat()
        if not stat.S_ISREG(info.st_mode):return None
        value={'bytes':info.st_size,'mtime_ns':info.st_mtime_ns,'ctime_ns':info.st_ctime_ns,
               'inode':info.st_ino,'device':info.st_dev}
        if os.name=='nt':value['change_time']=_windows_change_time(path)
        return value
    except FileNotFoundError:
        return None
    except OSError as exc:
        return {'error':str(exc)}

def _snapshot(key,spec,manifest):
    entry=manifest.get('components',{}).get(key,{})
    return {'manifest_sha256':_json_hash({'spec':spec,'entry':entry}),
            'files':{name:_file_signature(MODELS/key/name) for name in spec['files']}}

def _download_lock_held():
    if LOCK.locked():return True
    if not MODELS.is_dir():return False
    from filelock import FileLock,Timeout
    probe=FileLock(MODELS/'.download.lock',thread_local=False)
    try:probe.acquire(timeout=0)
    except Timeout:return True
    else:
        probe.release()
        return False

def _download_status(errors=None):
    local_errors=[]
    stored=_read_model_state(DATA/'model-download.json','download',local_errors)
    if not _state_matches_root(stored):
        stored={}
    value={**STATE,**stored}
    if local_errors:
        value.update(state='interrupted',message='下载状态记录损坏，可重新下载；完整文件仍须通过校验')
        if errors is not None:errors.extend(local_errors)
    if _download_lock_held():
        if value.get('state')!='downloading':
            value={**value,'state':'downloading','message':'另一个本地进程正在下载并校验模型，请等待完成'}
    elif value.get('state')=='downloading':
        # Do not overwrite disk here: a CLI may acquire the lock immediately
        # after our probe. Its real lock, rather than an old JSON state, wins.
        value={**value,'state':'interrupted','message':'上次下载已中断，可重新下载；完整分片会继续复用'}
    return value

def _state_matches_root(value):
    recorded=value.get('model_root')
    if recorded is not None:
        return recorded == str(MODELS.resolve())
    # Old bundles recorded no root. Keep their default-directory metadata,
    # but do not carry it into a newly selected custom directory.
    return not config().get('models_dir') or MODELS.resolve() == DEFAULT_MODELS.resolve()

def set_model_root(root):
    """Called only while the server owns all model-operation guards."""
    global MODELS, FILE_LOCK
    MODELS=Path(root)
    FILE_LOCK=None
    STATE.clear()
    STATE.update(state='idle',message='模型仅包含 Hi3D 及其必需的图像编码器',model_root=str(MODELS.resolve()))

def ranged_download(repo, revision, name, destination, size, expected):
    """Resumable bounded HTTP ranges. Validate every range and final LFS hash."""
    import requests
    from huggingface_hub import hf_hub_url
    if destination.is_file() and destination.stat().st_size==size and digest(destination)==expected:
        return destination
    chunk_size=4*1024*1024
    count=math.ceil(size/chunk_size)
    parts=MODELS/'.downloads'/(expected+'-4m')
    parts.mkdir(parents=True,exist_ok=True)
    url=hf_hub_url(repo,name,revision=revision)
    local=threading.local()
    def fetch(i):
        if not hasattr(local,'session'):
            local.session=requests.Session()
            local.session.trust_env=config().get('download_trust_env',True)
        start=i*chunk_size;end=min(size-1,start+chunk_size-1)
        path=parts/str(i)
        if path.is_file() and path.stat().st_size==end-start+1:return
        for attempt in range(4):
            try:
                # Distinct query avoids stale CDN caches serving a different range.
                with local.session.get(url+f'?download=true&part={i}',headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,60)) as r:
                    r.raise_for_status()
                    if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':
                        raise ValueError('服务器未返回请求的字节区间')
                    temp=path.with_suffix('.tmp')
                    with temp.open('wb') as out:
                        for block in r.iter_content(65536):out.write(block)
                    if temp.stat().st_size!=end-start+1:raise ValueError('下载片段不完整')
                    os.replace(temp,path)
                return
            except Exception:
                if attempt==3:raise
                time.sleep(2**attempt)
    with ThreadPoolExecutor(max_workers=max(1,min(16,config().get('download_workers',6)))) as pool:
        futures=[pool.submit(fetch,i) for i in range(count)]
        done=0
        for future in as_completed(futures):
            future.result();done+=1
            STATE.update(message=f'{destination.relative_to(MODELS)} · {done}/{count} 个片段',completed_chunks=done,total_chunks=count)
            atomic_json(DATA/'model-download.json',STATE.copy())
    destination.parent.mkdir(parents=True,exist_ok=True)
    temp=destination.with_suffix('.assembling')
    with temp.open('wb') as out:
        for i in range(count):
            with (parts/str(i)).open('rb') as source:
                import shutil
                shutil.copyfileobj(source,out,8*1024*1024)
    if digest(temp)!=expected:
        temp.unlink(missing_ok=True)
        # A right-sized chunk can still contain corrupt bytes. Without per-
        # chunk upstream hashes we cannot identify it, so clear these bounded
        # cache files once and allow the next attempt to fetch fresh bytes.
        for i in range(count):
            (parts/str(i)).unlink(missing_ok=True)
            (parts/str(i)).with_suffix('.tmp').unlink(missing_ok=True)
        raise ValueError('完整权重 SHA256 校验失败；已清理该文件的下载分片，请重试')
    os.replace(temp,destination)
    # Keep chunk cache until a separate explicit cleanup, allowing failed transfers to resume.
    return destination

def status():
    state_errors=[]
    manifest = _read_model_state(MODELS / 'manifest.json','manifest',state_errors)
    verification=_read_model_state(DATA/'model-verification.json','verification',state_errors)
    components = {}
    for key, spec in SPECS.items():
        snapshot=_snapshot(key,spec,manifest)
        missing = [n for n in spec['files'] if snapshot['files'][n] is None]
        entries = manifest.get("components", {}).get(key, {})
        metadata_errors=_metadata_errors(entries,spec)
        checked=verification.get('components',{}).get(key,{}) if verification.get('schema')==1 and _state_matches_root(verification) else {}
        current=checked.get('snapshot')==snapshot
        verified=current and checked.get('ok') is True and not missing and not metadata_errors
        state='verified' if verified else ('failed' if current else 'required')
        if missing:message=f'缺少 {len(missing)} 个必需文件，请下载 / 补齐'
        elif metadata_errors:message='模型清单的仓库 / 版本与固定官方组件不一致，请重新下载'
        elif verified:message='完整 SHA256 校验通过，当前文件可用'
        elif current:message='完整 SHA256 校验失败：'+', '.join(checked.get('errors',[]))
        elif checked:message='文件或清单已变化，需重新完整校验'
        else:message='文件齐全，尚需完整 SHA256 校验'
        components[key] = {"repo": spec["repo"], "license": spec["license"], "missing": missing,
                           "present": not missing, "revision": entries.get("revision"),
                           "verified": verified,'verification':state,'message':message,
                           'errors':checked.get('errors',[]) if current else []}
    download_state=_download_status(state_errors)
    if state_errors and download_state['state']!='downloading':download_state['message']+='；'+ '；'.join(state_errors)
    return {"components": components, "ready": all(x["present"] and x["verified"] for x in components.values()) and download_state['state']!='downloading', "download": download_state,'state_errors':state_errors,'model_root':str(MODELS)}

def download_all():
    global FILE_LOCK
    if not LOCK.acquire(blocking=False):
        return False
    from filelock import FileLock, Timeout
    try:
        MODELS.mkdir(parents=True,exist_ok=True)
        FILE_LOCK=FileLock(MODELS/'.download.lock',thread_local=False)
        FILE_LOCK.acquire(timeout=0)
    except Timeout:
        LOCK.release()
        return False
    except Exception:
        LOCK.release()
        raise
    try:
        STATE.clear()
        STATE.update(state='downloading',message='开始下载并校验 Hi3D 与必需图像编码器',model_root=str(MODELS.resolve()))
        _backup_invalid_state(DATA/'model-download.json','download')
        atomic_json(DATA/'model-download.json',STATE.copy())
    except Exception:
        FILE_LOCK.release()
        LOCK.release()
        raise
    def run():
        try:
            from huggingface_hub import HfApi, hf_hub_download
            api = HfApi()
            manifest = _read_model_state(MODELS/'manifest.json','manifest')
            manifest.setdefault('components',{})
            manifest['created']=time.time()
            for key, spec in SPECS.items():
                STATE.update(state="downloading", component=key, message=f"读取 {spec['repo']} 文件清单")
                info = api.model_info(spec["repo"], revision=spec.get("revision"), files_metadata=True)
                if info.sha!=spec['revision']:raise ValueError(f"固定模型版本不匹配：{key}")
                siblings = {x.rfilename: x for x in info.siblings}
                entry = {"repo": spec["repo"], "revision": info.sha, "license": spec["license"], "files": {}}
                # Only official inference artifacts and notices, never model-repository code.
                notice_names = [n for n in siblings if n.lower() in {"license", "license.txt", "notice", "authors", "third_party_licenses.md"}]
                for name in list(dict.fromkeys(spec["files"] + notice_names)):
                    STATE.update(file=name, message=f"下载并校验 {key}/{name}")
                    lfs = getattr(siblings[name], "lfs", None)
                    expected = (lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)) if lfs else None
                    expected_git=None if expected else getattr(siblings[name],'blob_id',None)
                    if not expected and not expected_git:raise ValueError(f'缺少上游文件校验值：{key}/{name}')
                    size=siblings[name].size or 0
                    if size>64*1024*1024 and expected:
                        path=ranged_download(spec['repo'],info.sha,name,MODELS/key/name,size,expected)
                    else:
                        atomic_json(DATA/'model-download.json',STATE.copy())
                        path = Path(hf_hub_download(spec["repo"], name, revision=info.sha, local_dir=MODELS / key))
                    sha = digest(path)
                    matches=(expected==sha) if expected else (_git_blob_digest(path)==expected_git)
                    if not matches:
                        # huggingface_hub may reuse a corrupt, already-present
                        # small artifact. Force a real transfer once on retry.
                        path=Path(hf_hub_download(spec['repo'],name,revision=info.sha,local_dir=MODELS/key,force_download=True))
                        sha=digest(path)
                        matches=(expected==sha) if expected else (_git_blob_digest(path)==expected_git)
                        if not matches:raise ValueError(f"上游文件校验不匹配：{key}/{name}；重新下载后仍不匹配")
                    entry["files"][name] = {"sha256": sha, "bytes": path.stat().st_size}
                manifest["components"][key] = entry
                _backup_invalid_state(MODELS/'manifest.json','manifest')
                atomic_json(MODELS / "manifest.json", manifest)
            result=verify_all()
            if not result['ok']:raise ValueError('下载后的完整校验失败：'+', '.join(result['errors']))
            STATE.update(state="complete", message="三个必需组件下载与 SHA256 校验完成")
        except Exception as e:
            STATE.update(state="failed", message=str(e))
        finally:
            try:atomic_json(DATA / "model-download.json", STATE)
            finally:
                FILE_LOCK.release()
                LOCK.release()
    try:threading.Thread(target=run, daemon=True).start()
    except Exception:
        FILE_LOCK.release()
        LOCK.release()
        raise
    return True

def verify_all():
    errors = []
    manifest = _read_model_state(MODELS / 'manifest.json','manifest',errors)
    checked={}
    for key, spec in SPECS.items():
        entry = manifest.get("components", {}).get(key, {})
        before=_snapshot(key,spec,manifest)
        failures=_metadata_errors(entry,spec)
        for name in spec["files"]:
            path = MODELS / key / name
            expected = entry.get("files", {}).get(name, {}).get("sha256")
            signature=before['files'][name]
            try:
                with verification_read_guard(path):
                    if not signature or 'error' in signature or not expected or digest(path)!=expected:
                        failures.append(name)
            except OSError:
                failures.append(name)
        after=_snapshot(key,spec,_read_model_state(MODELS/'manifest.json','manifest'))
        if before!=after:
            failures.extend(name for name in spec['files'] if name not in failures)
        checked[key]={'snapshot':before,'ok':not failures,'errors':failures}
        errors.extend(f'{key}/{name}' for name in failures)
    result = {'schema':1,"ok": not errors, "errors": errors, "time": time.time(),'components':checked,'model_root':str(MODELS.resolve())}
    _backup_invalid_state(DATA/'model-verification.json','verification')
    atomic_json(DATA / "model-verification.json", result)
    return result
