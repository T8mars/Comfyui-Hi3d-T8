"""Asset-studio endpoints sharing the workbench's single-job ownership."""
from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path
from typing import Literal

from fastapi import HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, StrictStr

from ..common import atomic_json, read_json, child_path, digest
from . import models

FILES = {'output.glb', 'rig-original.glb', 'output.fbx', 'output.obj', 'parts.zip', 'parts.json', 'rig.json',
         'original-segmentation.glb', 'original-parts.json', 'original-parts.zip',
         'part-coverage.json', 'xpart-generation.json', 'xpart-surface-preparation.json',
         'motion.bvh', 'motion.npz', 'motion.json', 'kimodo.json', 'report.json', 'worker.log'}
SOURCE_FILES = {'coarse.glb', 'refined.glb', 'textured.glb', 'generated.glb', 'output.glb', 'rig-original.glb', 'original-segmentation.glb'}


def source_parts_path(source):
    name='original-parts.json' if source.name=='original-segmentation.glb' else 'parts.json'
    path=source.parent/name
    if not path.is_file():return None
    metadata=read_json(path,{})
    return path if (isinstance(metadata,dict) and metadata.get('schema')==1
                    and metadata.get('mesh_sha256')==digest(source)) else None


class PartAttachment(BaseModel):
    part: int = Field(ge=0, le=999999, strict=True)
    bone: int = Field(ge=0, le=999999, strict=True)
    avoid_ground: bool = False


class StudioInput(BaseModel):
    source: StrictStr = Field(max_length=150)
    action: Literal['simplify', 'retopology', 'segment', 'edit_parts', 'complete_parts', 'rig', 'edit_rig', 'attach_parts', 'retarget', 'motion']
    target_faces: int | None = Field(None, ge=4, le=5_000_000)
    ratio: float | None = Field(None, gt=0, le=1)
    max_error: float = Field(.01, ge=0, le=1)
    priority: Literal['quality', 'count'] = 'quality'
    target_quads: int = Field(15000, ge=100, le=1_000_000)
    retopology_engine: Literal['autoremesher','quadriflow'] = 'autoremesher'
    retopology_surface: Literal['original','solid'] = 'original'
    material_projection: Literal['nearest','cage'] = 'nearest'
    preserve_sharp: bool = False
    edge_scaling: float = Field(1,ge=1,le=4)
    sharp_angle: float = Field(60, ge=0, le=180)
    adaptivity: float = Field(1, ge=0, le=2)
    texture_size: Literal[1024, 2048, 4096] = 2048
    uv_backend: Literal['auto','gpu','cpu'] = 'auto'
    segment_points: int = Field(100000, ge=10000, le=200000)
    segment_prompts: int = Field(400, ge=16, le=400)
    segment_threshold: float = Field(.95, ge=.5, le=.999)
    part_steps: int = Field(50, ge=10, le=100)
    part_guidance: float = Field(7.5,ge=1,le=15)
    part_resolution: Literal[256, 512] = 512
    part_edits: list[dict] = Field(default_factory=list, max_length=500)
    rig_engine: Literal['skintokens', 'unirig'] = 'skintokens'
    use_skeleton: bool = False
    release_contacts: bool = False
    bone_edits: list[dict] = Field(default_factory=list, max_length=512)
    part_attachments: list[PartAttachment] = Field(default_factory=list, max_length=512)
    motion_asset: StrictStr | None = Field(None, max_length=32)
    motion_clip: int = Field(0,ge=0,le=999)
    motion_armature: StrictStr | None = Field(None,max_length=200)
    motion_up: Literal['Y','Z'] = 'Y'
    source_mapping: dict[str,str] = Field(default_factory=dict,max_length=512)
    joint_mapping: dict[str, str] = Field(default_factory=dict, max_length=512)
    prompt: StrictStr = Field('A person performs a controlled punch and returns to a fighting stance.', max_length=2000)
    duration: float = Field(4, ge=.5, le=10)
    diffusion_steps: int = Field(100, ge=10, le=200)
    seed: int = Field(0, ge=0, le=2**32-1)
    text_device: Literal['cpu', 'cuda'] = 'cpu'
    speed: float = Field(1, ge=.1, le=4)
    loop: bool = False
    root_motion: Literal['original', 'in_place'] = 'original'
    foot_lock: bool = True
    motion_scale: float | None = Field(None, gt=0, le=1000)
    rest_pose: Literal['auto', 'T', 'A'] = 'auto'
    constraints: list[dict] = Field(default_factory=list, max_length=80)


class Selection(BaseModel):
    selection: StrictStr = Field(max_length=50)


def feature(body):
    if body.action == 'rig':
        return 'rig_' + body.rig_engine
    if body.action in {'edit_rig', 'attach_parts'}:
        return 'retarget'
    if body.action == 'edit_parts':
        return 'simplify'
    return body.action


def resolve_source(source, assets, manager):
    fields = source.split(':')
    if len(fields) < 2 or not re.fullmatch('[0-9a-f]{32}', fields[1]):
        raise ValueError('请选择已上传模型或历史任务成品')
    if fields[0] == 'asset' and len(fields) == 2:
        path = child_path(assets, fields[1]) / 'mesh.glb'
    elif fields[0] == 'job' and len(fields) == 3 and fields[2] in SOURCE_FILES:
        if manager.get(fields[1]) is None:
            raise ValueError('来源任务不存在')
        if fields[1] == manager.active:
            raise ValueError('请等待来源任务保存完成后再加工')
        path = child_path(manager.root, fields[1]) / fields[2]
    else:
        raise ValueError('模型来源无效')
    if not path.is_file():
        raise ValueError('来源模型文件不存在')
    return path


def mount(app, manager, gpu_operation, assets):
    @app.get('/api/studio/status')
    def status():
        return models.status()

    @app.get('/api/studio/sources')
    def sources():
        values = []
        for folder in assets.iterdir():
            if folder.is_dir() and (folder / 'mesh.glb').is_file():
                try:
                    meta = read_json(folder / 'meta.json', {})
                    values.append({'value': 'asset:' + folder.name, 'name': meta.get('name', folder.name),
                                   'kind': 'upload', 'created': folder.stat().st_mtime})
                except (OSError, ValueError):
                    continue
        for job in manager.list():
            for name in sorted(SOURCE_FILES):
                path = manager.root / job['id'] / name
                if path.is_file() and job['id'] != manager.active:
                    values.append({'value': f'job:{job["id"]}:{name}', 'name': f'{job["id"][:8]} · {name}',
                                   'kind': 'job', 'created': job['created']})
        return sorted(values, key=lambda row: row['created'], reverse=True)

    @app.get('/api/studio/source')
    def source_info(source: str):
        from .skeleton import inspect
        path = resolve_source(source, assets, manager)
        value = inspect(path)
        value['url'] = '/api/studio/source-file?source=' + __import__('urllib.parse', fromlist=['quote']).quote(source, safe='')
        metadata_path=source_parts_path(path)
        if metadata_path is not None:value['parts']=read_json(metadata_path)
        return value

    @app.get('/api/studio/source-file')
    def source_file(source: str):
        return FileResponse(resolve_source(source, assets, manager))

    @app.post('/api/studio/jobs')
    def create(body: StudioInput):
        source = resolve_source(body.source, assets, manager)
        if body.action == 'simplify' and (body.target_faces is None) == (body.ratio is None):
            raise HTTPException(400, '目标面数和保留比例请任选一种')
        if body.action == 'retarget' and not body.motion_asset:
            raise HTTPException(400, '请上传 BVH / FBX 动作文件')
        if body.action == 'attach_parts' and not body.part_attachments:
            raise HTTPException(400, '请添加需要固定的部件与目标骨骼')
        if body.action in {'edit_rig', 'attach_parts', 'retarget', 'motion'} or (body.action == 'rig' and body.use_skeleton):
            from .gltf import Document
            from .skin import extract,validate
            try:
                document=Document.load(source)
                extract(document).validate()
                if body.action in {'retarget','motion'}:
                    validate(document,allow_static=True)
            except ValueError as exc:
                raise HTTPException(400, '目标模型缺少有效骨架，请先执行自动骨骼与蒙皮：' + str(exc)) from exc
        with gpu_operation('asset_processing'):
            state = models.status()
            if models.download_busy():
                raise HTTPException(409, '资产模型正在下载或校验')
            if not state['features'][feature(body)]['ready']:
                raise HTTPException(409, '所选资产功能的模型或环境未就绪，请在资产加工页面准备组件')
            options = {**body.model_dump(), 'mode': 'studio'}
            if body.motion_asset:
                if not re.fullmatch('[0-9a-f]{32}', body.motion_asset):
                    raise HTTPException(400, '动作文件标识无效')
                meta = read_json(child_path(assets, body.motion_asset) / 'meta.json', {})
                if meta.get('kind') != 'motion' or meta.get('filename') not in {'motion.bvh', 'motion.fbx', 'motion.glb'}:
                    raise HTTPException(400, '动作文件不存在')
                options['motion_path'] = str(child_path(assets, body.motion_asset) / meta['filename'])
            metadata_path=source_parts_path(source)
            if body.action in {'edit_parts', 'complete_parts', 'attach_parts'}:
                if metadata_path is None:
                    raise HTTPException(400, '请先拆件并保存部件，再进行此操作')
                options['parts_path'] = str(metadata_path)
            elif body.action in {'rig', 'edit_rig', 'retarget', 'motion'} and metadata_path is not None:
                options['parts_path'] = str(metadata_path)
            try:
                return manager.create(None, source, options)
            except RuntimeError as exc:
                raise HTTPException(409, str(exc)) from exc

    @app.get('/api/studio/jobs/{jid}/files/{name}')
    def file(jid: str, name: str):
        if name not in FILES or manager.get(jid) is None:
            raise HTTPException(404)
        path = child_path(manager.root, jid) / name
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    @app.post('/api/studio/upload-motion')
    async def upload_motion(file: UploadFile = File(...)):
        suffix = Path(file.filename or '').suffix.lower()
        if suffix not in {'.bvh', '.fbx', '.glb'}:
            raise HTTPException(400, '请上传 BVH、FBX 或带动画的 GLB')
        folder = assets / uuid.uuid4().hex
        folder.mkdir()
        target = folder / ('motion' + suffix)
        try:
            size = 0
            with target.open('wb') as output:
                while block := await file.read(1024 * 1024):
                    size += len(block)
                    if size > 128 * 1024 * 1024:
                        raise ValueError('动作文件超过 128MB')
                    output.write(block)
            if not size:
                raise ValueError('动作文件为空')
            if suffix == '.bvh':
                with target.open('rb') as stream:
                    header=stream.read(200).removeprefix(b'\xef\xbb\xbf').lstrip()
                if not header.startswith(b'HIERARCHY'):
                    raise ValueError('BVH 文件缺少骨骼层级头')
            from starlette.concurrency import run_in_threadpool
            from .motion import inspect_motion
            information=await run_in_threadpool(inspect_motion,target,folder)
            meta = {'id': folder.name, 'kind': 'motion', 'name': file.filename,
                    'filename': target.name, 'bytes': size}
            atomic_json(folder / 'meta.json', meta)
            return {**meta,**information}
        except Exception as exc:
            # This is the fresh UUID directory created by this request. Verify
            # its resolved parent before removing nested native-stage files.
            if folder.resolve().parent==assets.resolve() and re.fullmatch('[0-9a-f]{32}',folder.name):
                shutil.rmtree(folder)
            raise HTTPException(400,str(exc)) from exc
        finally:
            await file.close()

    @app.get('/api/studio/motion-source')
    def motion_source(asset:str,up_axis:Literal['Y','Z']='Y',armature:str|None=None):
        if not re.fullmatch('[0-9a-f]{32}',asset) or (armature is not None and len(armature)>200):
            raise HTTPException(400,'动作文件标识或骨架名称无效')
        folder=child_path(assets,asset)
        meta=read_json(folder/'meta.json',{})
        if meta.get('kind')!='motion' or meta.get('filename') not in {'motion.bvh','motion.fbx','motion.glb'}:
            raise HTTPException(404,'动作文件不存在')
        from .motion import inspect_motion
        try:
            return {**meta,**inspect_motion(folder/meta['filename'],folder,up_axis=up_axis,armature=armature)}
        except (ValueError,RuntimeError,OSError) as exc:
            raise HTTPException(400,str(exc)) from exc

    @app.post('/api/studio/models/download')
    def download(body: Selection):
        models.selection_components(body.selection)
        # Share the existing GPU gate for the full operation, including idle
        # model-root changes and a concurrent generation submission.
        import threading
        from contextlib import ExitStack
        with ExitStack() as guards:
            guards.enter_context(gpu_operation('studio_models_download'))
            manager.close_resident()
            transferred = guards.pop_all()
            def run():
                try:
                    models.download(body.selection)
                except Exception:
                    pass  # Concrete error is retained in the model status.
                finally:
                    transferred.close()
            try:
                threading.Thread(target=run, daemon=False).start()
            except Exception:
                transferred.close()
                raise
        return {'started': True}

    @app.post('/api/studio/models/verify')
    def verify(body: Selection):
        with gpu_operation('studio_models_verify'):
            return models.verify(body.selection)
