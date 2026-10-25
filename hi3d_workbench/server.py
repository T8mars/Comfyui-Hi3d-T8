from __future__ import annotations
import json
import os
import secrets
import shutil
import subprocess
import threading
import time
import sys
import uuid
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from contextlib import asynccontextmanager, contextmanager, ExitStack
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field, StrictStr
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.concurrency import run_in_threadpool
from PIL import Image, ImageOps
from .common import ROOT, DATA, MODELS, config, atomic_json, read_json, child_path, SOURCE_REVISION, VERSION
from .geometry import load_mesh, inspect_mesh, export_mesh
from .asset_pipeline import inspect_asset_geometry
from .jobs import JobManager
from . import common, models, pipeline_models, updater, lifecycle

TOKEN=secrets.token_urlsafe(32)
WEB=Path(__file__).parent/"web"
GPU_OPERATION=threading.Lock()
VERIFY_LOCK=threading.Lock()
operation=None
verification_error=None
DIAGNOSTIC_TIMEOUT=180
PROBE_WORKERS={}
PROBE_WORKERS_LOCK=threading.Lock()
PIPELINE_MODELS_LOCK=threading.Lock()
MODEL_PATH_LOCK=threading.RLock()
PREVIEW_LOCK=threading.Lock()
PREVIEW_EVENTS=[]
JOB_FILES={"coarse.glb","generated.glb","input.png","foreground.png","conditioning.png","refined.glb","refined.ply","refined.stl","refined.npz","textured.glb","report.json","pipeline-progress.json","request.json","worker.log","failure.json"}

@contextmanager
def gpu_operation(kind):
    global operation
    if not GPU_OPERATION.acquire(blocking=False):raise HTTPException(409,"正在提交任务或检查 GPU，请稍后再试")
    try:
        if manager.active:raise HTTPException(409,"已有精修任务或建模/贴图任务运行，请先等待或停止该任务")
        if PIPELINE_MODELS_LOCK.locked():raise HTTPException(409,"扩展模型正在下载或校验，请稍候")
        if pipeline_models.status().get('download',{}).get('state')=='downloading':raise HTTPException(409,'扩展模型下载进程仍在运行，请稍候')
        if manager.recovery_blocked or probe_recovery_workers():raise HTTPException(409,"仍有残留工作或诊断进程运行，请等待该进程退出")
        operation=kind
        yield
    finally:
        operation=None
        GPU_OPERATION.release()

def verify_models():
    global verification_error
    if not VERIFY_LOCK.acquire(blocking=False):raise HTTPException(409,"正在校验模型文件，请稍候")
    try:
        if models.status()['download']['state']=='downloading':raise HTTPException(409,"模型正在下载，完成后再校验")
        verification_error=None
        return models.verify_all()
    except HTTPException:raise
    except Exception as exc:
        verification_error=str(exc)
        raise
    finally:VERIFY_LOCK.release()

def initial_verification():
    try:verify_models()
    except Exception:pass  # The health endpoint exposes failure without preventing the UI from opening.

def initial_pipeline_verification():
    if not PIPELINE_MODELS_LOCK.acquire(blocking=False):return
    try:pipeline_models.verify()
    except Exception:pass
    finally:PIPELINE_MODELS_LOCK.release()

@asynccontextmanager
async def lifespan(app):
    from filelock import FileLock,Timeout
    DATA.mkdir(parents=True,exist_ok=True)
    server_lock=FileLock(DATA/'server.lock',thread_local=False)
    try:server_lock.acquire(timeout=0)
    except Timeout:raise RuntimeError('该工作台目录已有服务运行，请打开现有窗口。')
    try:
        manager.recover()
        state=models.status()
        if not state['ready'] and all(c['present'] for c in state['components'].values()) and state['download']['state']!='downloading':
            threading.Thread(target=initial_verification,daemon=True).start()
        extension=pipeline_models.status()
        if extension['runtime']['ready'] and not extension['ready'] and all((pipeline_models.MODEL_ROOT/key/name).is_file() for key,spec in pipeline_models.SPECS.items() if not spec.get('optional') for name in spec['files']):
            threading.Thread(target=initial_pipeline_verification,daemon=True).start()
        updater.start_background(lambda: bool(manager.active or operation or PIPELINE_MODELS_LOCK.locked()))
        yield
    finally:
        updater.stop_background()
        if manager.active:manager.cancel(manager.active)
        manager.close_resident()
        server_lock.release()

app=FastAPI(title="Hi3D Workbench",docs_url=None,redoc_url=None,lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware,allowed_hosts=["127.0.0.1","localhost","[::1]"])
manager=JobManager()
ASSETS=DATA/"assets"
ASSETS.mkdir(parents=True,exist_ok=True)

@app.exception_handler(ValueError)
async def bad_input(request,exc):
    return JSONResponse(status_code=400,content={"detail":str(exc)})

@app.middleware("http")
async def guard(request: Request,call_next):
    if request.method not in {"GET","HEAD","OPTIONS"}:
        origin=request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host"):
            return HTMLResponse("跨站请求被拒绝",status_code=403)
        if not secrets.compare_digest(request.headers.get("x-hi3d-token",""),TOKEN):
            return HTMLResponse("本地会话令牌已失效，请刷新页面后重试",status_code=403)
    response=await call_next(request)
    response.headers["X-Content-Type-Options"]="nosniff"
    if request.url.path=='/' or request.url.path.startswith('/api/'):
        response.headers['Cache-Control']='no-store'
    return response

@app.get("/",response_class=HTMLResponse)
def index():return (WEB/"index.html").read_text(encoding="utf-8").replace("__TOKEN__",TOKEN).replace('__VERSION__',VERSION)

@app.get("/api/session")
def session():return {"token":TOKEN,"version":VERSION,"source_revision":SOURCE_REVISION,"instance_id":lifecycle.instance_id(ROOT),"can_shutdown":lifecycle.managed,"default_attention":"sdpa" if os.name=="nt" and not config()["worker_command"] else "flash"}

@app.get('/api/runtime')
def runtime_status():return manager.runtime_status()

class PreviewStatus(BaseModel):
    view:StrictStr=Field(max_length=16)
    source:StrictStr=Field(max_length=512)
    status:StrictStr=Field(max_length=16)
    detail:StrictStr=Field(default='',max_length=500)

@app.post('/api/preview-status')
def preview_status(body:PreviewStatus):
    if body.view not in {'before','after'} or body.status not in {'loaded','failed'} or not body.source.startswith(('/api/jobs/','/api/assets/')):
        raise HTTPException(400,'无效预览记录')
    with PREVIEW_LOCK:
        PREVIEW_EVENTS.append({**body.model_dump(),'time':time.time()})
        del PREVIEW_EVENTS[:-20]
    return {'ok':True}

@app.get('/api/preview-status')
def preview_status_events():
    with PREVIEW_LOCK:return list(PREVIEW_EVENTS)

@app.post('/api/runtime/unload')
def unload_runtime():
    with gpu_operation('unloading'):
        manager.close_resident()
    return manager.runtime_status()

@app.post('/api/shutdown')
def shutdown_workbench():
    if not lifecycle.managed:raise HTTPException(409,'请从原启动窗口关闭此服务')
    return JSONResponse({'stopping':True},background=BackgroundTask(lifecycle.shutdown.set))

@app.get('/api/updates')
def update_status():return updater.status()

@app.post('/api/updates/check')
def update_check():return {'started':updater.start_check()}

class UpdatePreferences(BaseModel):
    auto_update: bool

@app.post('/api/updates/preferences')
def update_preferences(value:UpdatePreferences):return updater.set_preferences(value.auto_update)

@app.get('/api/sample/{name}')
def sample(name:str):
    if name not in {'sculpted-vase.png','sculpted-vase.glb'}:raise HTTPException(404)
    path=ROOT/'samples'/name
    if not path.is_file():raise HTTPException(404)
    return FileResponse(path)

@app.get("/api/health")
def health(request: Request):
    probe_workers=probe_recovery_workers()
    blocked=manager.recovery_blocked or bool(probe_workers)
    errors=manager.recovery_errors.copy()
    if probe_workers:errors.append('上次 GPU 诊断仍有进程未退出，暂时阻止新的 GPU 操作')
    try:diagnostics=read_json(DATA/'diagnostics.json')
    except (OSError,ValueError) as exc:diagnostics={'error':f'诊断记录无法读取，请重新运行 GPU 检查：{exc}','kernel_probe':'not_run'}
    address=request.scope.get('server')
    with MODEL_PATH_LOCK:
        return {"models":models.status(),"pipeline_models":pipeline_models.status(),"model_path":model_path(),"pipeline_model_checking":PIPELINE_MODELS_LOCK.locked(),"active":manager.active,"operation":operation,"model_checking":VERIFY_LOCK.locked(),"verification_error":verification_error,"recovery_errors":errors,"recovery_blocked":blocked,"recovery_workers":manager.recovery_workers+probe_workers,"diagnostics":diagnostics,"port":address[1] if address else config()["port"],"version":VERSION}

def _model_path_busy_reason(*, include_local_locks=True):
    if manager.active:
        return '已有任务运行，请先等待或停止该任务'
    if operation or (include_local_locks and GPU_OPERATION.locked()):
        return '正在提交任务、下载或检查 GPU，请稍候'
    if manager.recovery_blocked or probe_recovery_workers():
        return '仍有残留工作或诊断进程运行，请等待该进程退出'
    if include_local_locks:
        if VERIFY_LOCK.locked():return '正在校验 Hi3D 模型，请稍候'
        if models.LOCK.locked():return '正在下载 Hi3D 模型，请稍候'
        if PIPELINE_MODELS_LOCK.locked() or pipeline_models._LOCK.locked():
            return '扩展模型正在下载或校验，请稍候'
        if models._download_lock_held():return '模型下载或校验进程仍在运行，请稍候'
        if pipeline_models.status().get('download',{}).get('state')=='downloading':
            return '扩展模型下载或校验进程仍在运行，请稍候'
    return None

@app.get('/api/model-path')
def model_path():
    with MODEL_PATH_LOCK:
        value=config().get('models_dir') or ''
        reason=_model_path_busy_reason()
        return {'models_dir':value,'resolved_path':str(MODELS),
                'default_path':str(common.DEFAULT_MODELS),
                'extensions_path':str(pipeline_models.MODEL_ROOT),
                'is_default':MODELS.resolve()==common.DEFAULT_MODELS.resolve(),
                'exists':MODELS.is_dir(),'restart_required':False,
                'busy':reason is not None,'busy_reason':reason}

class ModelPathInput(BaseModel):
    models_dir:StrictStr | None=Field(default='',max_length=4096)

@app.post('/api/model-path')
def save_model_path(body:ModelPathInput):
    """Commit one idle weight root; no worker or existing weight is modified."""
    global MODELS, verification_error
    value=(body.models_dir or '').strip()
    target=common.resolve_model_dir(value)
    if not GPU_OPERATION.acquire(blocking=False):
        raise HTTPException(409,'正在提交任务、下载或检查 GPU，请稍候再修改模型目录')
    try:
        # The same order as submission (GPU -> manager) excludes a queued job
        # until its worker has been assigned the new persisted configuration.
        with manager.lock, ExitStack() as guards:
            reason=_model_path_busy_reason(include_local_locks=False)
            if reason:raise HTTPException(409,reason)
            for lock in (VERIFY_LOCK,PIPELINE_MODELS_LOCK,models.LOCK,pipeline_models._LOCK):
                if not lock.acquire(blocking=False):
                    raise HTTPException(409,'模型正在下载或校验，请稍候再修改模型目录')
                guards.callback(lock.release)
            # Hold actual CLI locks on both the old and selected directories.
            # Nonexistent roots stay nonexistent until a download is requested.
            from filelock import FileLock, Timeout
            roots={MODELS.resolve(),pipeline_models.MODEL_ROOT.resolve(),target,(target/'extensions').resolve()}
            for root in sorted(roots,key=str):
                if not root.is_dir():continue
                lock=FileLock(str(root/'.download.lock'),thread_local=False)
                try:lock.acquire(timeout=0)
                except Timeout:
                    raise HTTPException(409,'所选或当前目录有模型下载/校验进程运行，请稍候') from None
                guards.callback(lock.release)
            with MODEL_PATH_LOCK:
                manager.close_resident()
                cfg=read_json(ROOT/'config.local.json',{})
                if not isinstance(cfg,dict):raise ValueError('config.local.json 必须为 JSON 对象')
                cfg['models_dir']=value
                atomic_json(ROOT/'config.local.json',cfg)
                common.MODELS=MODELS=target
                models.set_model_root(target)
                pipeline_models.set_model_root(target)
                verification_error=None
    finally:
        GPU_OPERATION.release()
    return model_path()

@app.post("/api/diagnostics")
def diagnose():
    with gpu_operation('diagnosing'):
        cfg=config()
        cmd=cfg["worker_command"] or [str((ROOT/cfg["worker_python"]).resolve())]
        try:
            output,returncode=run_probe([*cmd,"-m","hi3d_workbench.worker","--diagnose","--probe"])
            result=json.loads(output.strip().splitlines()[-1])
            if returncode and result.get('kernel_probe')=='passed':
                result.update(kernel_probe='failed',error=f'诊断进程异常退出：{returncode}')
        except Exception as e:result={"error":str(e),"kernel_probe":"failed"}
        atomic_json(DATA/"diagnostics.json",result)
        return result

def probe_recovery_workers():
    """Keep the GPU guard until previously identified diagnostic workers exit."""
    import psutil
    alive={}
    with PROBE_WORKERS_LOCK:
        for pid,created in PROBE_WORKERS.items():
            try:
                proc=psutil.Process(pid)
                if abs(proc.create_time()-created)>.01 or not proc.is_running() or proc.status()==psutil.STATUS_ZOMBIE:continue
            except psutil.NoSuchProcess:continue
            except psutil.AccessDenied:pass  # Retain a known identity if it becomes unreadable.
            alive[pid]=created
        PROBE_WORKERS.clear();PROBE_WORKERS.update(alive)
    return [{'pid':pid,'created':created} for pid,created in alive.items()]


def _marked_probe_workers(marker,started):
    """The inherited nonce also identifies children whose launcher has exited."""
    import psutil
    found={}
    for candidate in psutil.process_iter(['create_time']):
        try:
            # Ignore older processes using the iterator's cached metadata;
            # recreate only recent candidates before reading their environment.
            if candidate.info['create_time'] is None or candidate.info['create_time']<started:continue
            proc=psutil.Process(candidate.pid)
            created=proc.create_time()
            if created>=started and proc.environ().get('HI3D_DIAGNOSTIC_ID')==marker:
                found[proc.pid]=(proc,created)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return found


def _kill_probe_process(process):
    process.kill()


def run_probe(command):
    """Reap inherited diagnostic workers even after a Windows launcher exits."""
    import psutil
    marker=secrets.token_urlsafe(32);started=time.time()
    process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        text=True,encoding='utf-8',errors='replace',env={**os.environ,'HI3D_DIAGNOSTIC_ID':marker},
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    known={}
    try:
        identity=psutil.Process(process.pid)
        known[process.pid]=(identity,identity.create_time())
    except psutil.NoSuchProcess:pass
    try:
        output,_=process.communicate(timeout=DIAGNOSTIC_TIMEOUT)
        return output,process.returncode
    except subprocess.TimeoutExpired:
        # A parent can disappear before children(recursive=True) runs. Match
        # only this invocation's inherited nonce and creation time, then make
        # a second pass for descendants spawned during the first snapshot.
        try:
            for _ in range(2):
                known.update(_marked_probe_workers(marker,started))
                ordered=sorted(known.values(),key=lambda item:item[0].pid==process.pid)
                for owned,created in ordered:
                    try:
                        if abs(owned.create_time()-created)<=.01:_kill_probe_process(owned)
                    except (psutil.Error,OSError):pass
            try:process.communicate(timeout=2)
            except subprocess.TimeoutExpired:pass
            psutil.wait_procs([owned for owned,_ in known.values()],timeout=2)
        finally:
            # Retain every confirmed identity even if pipe cleanup itself
            # fails, so releasing the operation lock cannot permit overlap.
            try:known.update(_marked_probe_workers(marker,started))
            finally:
                with PROBE_WORKERS_LOCK:PROBE_WORKERS.update({pid:created for pid,(_,created) in known.items()})
        if probe_recovery_workers():raise RuntimeError('GPU 诊断超时且部分进程尚未退出，已阻止新的 GPU 操作') from None
        raise RuntimeError('GPU 诊断超时，诊断进程及子进程已停止') from None

@app.post("/api/models/download")
def download():
    with gpu_operation('downloading'):
        if not VERIFY_LOCK.acquire(blocking=False):raise HTTPException(409,"正在校验模型，请完成后再下载")
        try:return {"started":models.download_all()}
        finally:VERIFY_LOCK.release()

@app.post("/api/models/verify")
def verify():return verify_models()

class PipelineDownloadInput(BaseModel):
    selection:str='all'

@app.post('/api/pipeline/models/download')
def download_pipeline(body:PipelineDownloadInput):
    try:pipeline_models.selection_components(body.selection)
    except ValueError as exc:raise HTTPException(400,str(exc)) from exc
    with gpu_operation('downloading'):
        if not PIPELINE_MODELS_LOCK.acquire(blocking=False):raise HTTPException(409,'扩展模型正在下载或校验')
        def run():
            try:pipeline_models.download(body.selection)
            except Exception:pass  # Model status exposes the concrete Hub/download error.
            finally:PIPELINE_MODELS_LOCK.release()
        try:threading.Thread(target=run,daemon=True).start()
        except Exception:
            PIPELINE_MODELS_LOCK.release()
            raise
    return {'started':True}

@app.post('/api/pipeline/models/verify')
def verify_pipeline():
    with gpu_operation('verifying'):
        if not PIPELINE_MODELS_LOCK.acquire(blocking=False):raise HTTPException(409,'扩展模型正在下载或校验')
        try:return pipeline_models.verify()
        finally:PIPELINE_MODELS_LOCK.release()

def prepare_upload(temp, suffix, folder, filename):
    """Parsing and geometry inspection can be CPU-heavy; keep the event loop free."""
    if suffix in {".png",".jpg",".jpeg",".webp"}:
        with Image.open(temp) as im:
            if im.width*im.height>40_000_000:raise ValueError("图像超过 4000 万像素")
            im=ImageOps.exif_transpose(im).convert("RGBA");im.save(folder/"image.png")
            transparent=im.getchannel('A').getextrema()[0]<250
            im.thumbnail((1024,1024));im.save(folder/"preview.png")
        meta={"id":folder.name,"kind":"image","name":filename,"has_transparency":transparent,"url":f"/api/assets/{folder.name}/preview.png"}
    else:
        if suffix==".npz":
            with zipfile.ZipFile(temp) as z:
                if sum(i.file_size for i in z.infolist())>512*1024*1024:raise ValueError("解压后网格过大")
        mesh=load_mesh(temp)
        if len(mesh.faces)>5_000_000:raise ValueError("输入超过 500 万面，请先精简")
        has_materials=False
        if suffix=='.glb':
            from .asset_pipeline import read_glb_container
            tree=read_glb_container(temp)
            has_materials=bool(tree.get('materials'))
            shutil.copy2(temp,folder/'mesh.glb')
        else:mesh.export(folder/"mesh.glb")
        meta={"id":folder.name,"kind":"mesh","name":filename,"has_materials":has_materials,"url":f"/api/assets/{folder.name}/mesh.glb","geometry":inspect_asset_geometry(folder/"mesh.glb")}
    atomic_json(folder/"meta.json",meta)
    return meta

@app.post("/api/assets")
async def upload(file: UploadFile=File(...)):
    suffix=Path(file.filename or "").suffix.lower()
    if suffix not in {".png",".jpg",".jpeg",".webp",".glb",".ply",".stl",".npz"}:
        raise HTTPException(400,"支持 PNG/JPG/WebP 和 GLB/PLY/STL/NPZ；GLTF/OBJ 请先转为自包含 GLB")
    aid=uuid.uuid4().hex
    folder=ASSETS/aid;folder.mkdir()
    temp=folder/("upload"+suffix)
    total=0
    try:
        with temp.open("wb") as f:
            while chunk:=await file.read(1024*1024):
                total+=len(chunk)
                if total>256*1024*1024:raise ValueError("文件超过 256MB")
                f.write(chunk)
        return await run_in_threadpool(prepare_upload,temp,suffix,folder,file.filename)
    except Exception as e:
        # This UUID directory belongs only to the failed upload. A conversion
        # may already have written a large mesh or image before inspection fails.
        for partial in folder.iterdir():partial.unlink(missing_ok=True)
        folder.rmdir()
        raise HTTPException(400,str(e)) from e
    finally:
        temp.unlink(missing_ok=True)

@app.get("/api/assets/{aid}/{name}")
def asset(aid:str,name:str):
    if name not in {"preview.png","image.png","mesh.glb","meta.json"}:raise HTTPException(404)
    path=child_path(ASSETS,aid)/name
    if not path.is_file():raise HTTPException(404)
    return FileResponse(path)

class JobInput(BaseModel):
    image:str
    mesh:str | None=None
    mode:str='refine'
    refine_geometry:bool=False
    generate_texture:bool | None=None
    geometry_engine:str='hunyuan_mini'
    geometry_resolution:int | None=None
    geometry_steps:int | None=Field(None,ge=1,le=100)
    geometry_cleanup:bool=True
    geometry_target_faces:int=1000000
    pipeline_attention:str='auto'
    model_policy:str='release'
    texture_engine:str='trellis2'
    texture_steps:int | None=Field(None,ge=1,le=100)
    texture_resolution:int | None=None
    texture_size:int=2048
    background:str='auto'
    resolution:int=512
    steps:int=Field(25,ge=1,le=100)
    seed:int=Field(0,ge=0,le=2**53-1)
    offload:bool=True
    attention:str="flash"

@app.post("/api/jobs")
def create_job(body:JobInput):
    from .engine_catalog import engine, is_ready
    if (body.resolution not in {512,1024,1536} or body.attention not in {"flash","sdpa"}
            or body.pipeline_attention not in {'auto','flash_attn','sdpa'}
            or body.model_policy not in {'release','resident'}
            or body.geometry_target_faces not in {100000,300000,1000000}):raise HTTPException(400,"无效推理配置")
    try:
        geometry_choice=engine('coarse',body.geometry_engine)
        texture_choice=engine('texture',body.texture_engine)
    except ValueError as exc:raise HTTPException(400,str(exc)) from exc
    geometry_resolution=geometry_choice['default_resolution'] if body.geometry_resolution is None else body.geometry_resolution
    texture_resolution=texture_choice['default_resolution'] if body.texture_resolution is None else body.texture_resolution
    if body.mode=='image_to_asset' and body.geometry_engine=='hunyuan_mini' and body.geometry_steps not in {None,5}:
        raise HTTPException(400,'mini-Turbo 是蒸馏模型，固定使用5步；请选择其他引擎调整几何步数')
    if (body.mode not in {'refine','image_to_asset','texture'} or body.background not in {'auto','keep'}
            or texture_resolution not in texture_choice['resolutions'] or body.texture_size not in {1024,2048,4096}
            or geometry_resolution not in geometry_choice['resolutions']):raise HTTPException(400,'无效任务模式或所选引擎的采样配置')
    wants_refine=body.mode=='refine' or body.mode=='image_to_asset' and body.refine_geometry
    wants_texture=body.mode=='texture' or (body.generate_texture if body.generate_texture is not None else body.mode=='image_to_asset')
    image=child_path(ASSETS,body.image)/"image.png"
    mesh=child_path(ASSETS,body.mesh)/"mesh.glb" if body.mesh else None
    if not image.is_file() or body.mode!='image_to_asset' and (mesh is None or not mesh.is_file()):raise HTTPException(400,"请上传参考图；已有模型模式还需要同物体网格")
    if body.mode=='image_to_asset':mesh=None
    with gpu_operation('preparing'):
        if models.LOCK.locked():raise HTTPException(409,"模型正在下载")
        if wants_refine:
            if not models.status()["ready"]:raise HTTPException(409,"Hi3D 模型未就绪，请先下载完整组件并校验")
            # status() binds the last full SHA256 receipt to current file
            # identities, NTFS change times and the configured root. Rehashing
            # unchanged multi-GB weights on every submission only repeats I/O.
        extension=pipeline_models.status()
        if body.mode=='image_to_asset' and not is_ready(extension,'coarse',body.geometry_engine):raise HTTPException(409,f'{geometry_choice["name"]} 模型或运行环境未就绪，请查看「模型与环境」')
        if wants_texture and not is_ready(extension,'texture',body.texture_engine):raise HTTPException(409,f'{texture_choice["name"]} 模型或运行环境未就绪，请查看「模型与环境」')
        if (body.mode=='image_to_asset' or wants_texture) and body.background=='auto':
            with Image.open(image) as reference:
                opaque=reference.convert('RGBA').getchannel('A').getextrema()[0]>=250
            if opaque and not extension.get('background_ready',False):raise HTTPException(409,'自动前景处理模型未就绪，请补齐扩展组件，或提供透明图片/选择保留背景')
        options={**body.model_dump(exclude={"image","mesh"}),"generate_texture":wants_texture,"image_asset":body.image,"mesh_asset":body.mesh if mesh is not None else None}
        options['input_has_materials']=bool(mesh is not None and read_json(mesh.parent/'meta.json',{}).get('has_materials'))
        options.update(geometry_resolution=geometry_resolution, texture_resolution=texture_resolution, geometry_steps=body.geometry_steps or geometry_choice['default_steps'],
                       texture_steps=body.texture_steps or texture_choice['default_steps'])
        try:return manager.create(image,mesh,options)
        except RuntimeError as e:raise HTTPException(409,str(e)) from e

@app.get("/api/jobs")
def jobs():return manager.list()

@app.get('/api/jobs/{jid}/comparisons')
def comparisons(jid:str):
    if manager.get(jid) is None:raise HTTPException(404)
    values=[]
    for value in manager.list():
        folder=child_path(manager.root,value['id'])
        files=[name for name in ('coarse.glb','refined.glb','textured.glb','generated.glb') if (folder/name).is_file()]
        if not files:continue
        report=value.get('report') if isinstance(value.get('report'),dict) else {}
        values.append({'id':value['id'],'created':value['created'],'state':value['state'],
                       'options':value.get('options',{}),'available_files':files,
                       'input_sha256':report.get('input_sha256'),
                       'texture_mesh_sha256':report.get('texture_mesh_sha256')})
    return values

@app.get("/api/jobs/{jid}")
def job(jid:str):
    value=manager.get(jid)
    if value is None:raise HTTPException(404)
    return {**value,"available_files":sorted(name for name in JOB_FILES if (manager.root/jid/name).is_file())}

@app.post('/api/jobs/{jid}/reuse')
def reuse_job(jid:str):
    folder=child_path(manager.root,jid)
    value=manager.get(jid)
    image_only=bool(value and value['options'].get('mode')=='image_to_asset')
    if not value or not (folder/'input.png').is_file() or not image_only and not (folder/'coarse.glb').is_file():raise HTTPException(404,"此任务的输入文件已不存在")
    created=[]
    try:
        for kind,filename in ([('image','input.png')] if image_only else [('image','input.png'),('mesh','coarse.glb')]):
            aid=uuid.uuid4().hex;dest=ASSETS/aid;dest.mkdir();created.append(dest)
            if kind=='image':
                shutil.copy2(folder/filename,dest/'image.png')
                with Image.open(dest/'image.png') as im:
                    transparent=im.convert('RGBA').getchannel('A').getextrema()[0]<250
                    im.thumbnail((1024,1024));im.save(dest/'preview.png')
                meta={"id":aid,"kind":kind,"name":"复用参考图.png","has_transparency":transparent,"url":f"/api/assets/{aid}/preview.png"}
            else:
                shutil.copy2(folder/filename,dest/'mesh.glb')
                from .asset_pipeline import read_glb_container
                tree=read_glb_container(dest/'mesh.glb')
                meta={"id":aid,"kind":kind,"name":"复用粗模型.glb","has_materials":bool(tree.get('materials')),"url":f"/api/assets/{aid}/mesh.glb","geometry":inspect_asset_geometry(dest/'mesh.glb')}
            atomic_json(dest/'meta.json',meta)
        return {"image":read_json(created[0]/'meta.json'),"mesh":read_json(created[1]/'meta.json') if len(created)>1 else None,"options":value['options']}
    except Exception:
        # Only remove files in the newly allocated asset directories, never task originals.
        for directory in created:
            for path in directory.iterdir():path.unlink()
            directory.rmdir()
        raise

@app.post("/api/jobs/{jid}/cancel")
def cancel(jid:str):return manager.cancel(jid)

@app.get("/api/jobs/{jid}/files/{name}")
def job_file(jid:str,name:str):
    if name not in JOB_FILES:raise HTTPException(404)
    path=child_path(manager.root,jid)/name
    if not path.is_file():raise HTTPException(404)
    return FileResponse(path,filename=name)

class ExportInput(BaseModel):
    source:str="refined"
    format:str="glb"
    height_mm:float | None=Field(None,gt=0,le=100000)
    height_axis:str=Field('Z',pattern='^[XYZ]$')

@app.post("/api/jobs/{jid}/export")
def export(jid:str,body:ExportInput):
    if body.source not in {"coarse","refined","textured","generated"} or body.format not in {"glb","ply","stl"}:raise HTTPException(400)
    if body.source=='textured' and body.format!='glb':raise HTTPException(400,'带材质结果请导出 GLB；STL/PLY 请另选几何版本')
    folder=child_path(manager.root,jid)
    src=folder/(body.source+".glb")
    if not src.is_file():raise HTTPException(404)
    exportid=uuid.uuid4().hex
    out=DATA/"exports"/exportid;out.mkdir(parents=True)
    try:
        if body.source=='textured':
            from .asset_pipeline import export_textured_glb
            meta=export_textured_glb(src,out/'model.glb',body.height_mm,body.height_axis)
        elif body.format=='glb':
            from .asset_pipeline import export_asset_glb
            meta=export_asset_glb(src,out/'model.glb',body.height_mm,body.height_axis)
        else:meta=export_mesh(load_mesh(src),out/("model."+body.format),body.height_mm,body.height_axis)
        atomic_json(out/"dimensions.json",meta)
    except Exception:
        # A failed export must not leave a partial model or a large orphaned
        # copy. Only this freshly allocated export directory is discarded.
        for partial in out.iterdir():partial.unlink(missing_ok=True)
        out.rmdir()
        raise
    return {"url":f"/api/exports/{exportid}/model.{body.format}","dimensions":f"/api/exports/{exportid}/dimensions.json",**meta}

@app.get("/api/exports/{eid}/{name}")
def exported(eid:str,name:str):
    if name not in {"model.glb","model.ply","model.stl","dimensions.json"}:raise HTTPException(404)
    p=child_path(DATA/"exports",eid)/name
    if not p.is_file():raise HTTPException(404)
    return FileResponse(p,filename=name)

app.mount("/static",StaticFiles(directory=WEB),name="static")
