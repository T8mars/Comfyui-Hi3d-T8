"""Run the pinned official Hi3D pipeline inside ComfyUI's own interpreter."""
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

from . import models

VENDOR=Path(__file__).parent/'vendor'
SOURCE_REVISION='23dcc684bc6e1fab86ecafb99ec5270edad122f9'
_RUN_LOCK=threading.Lock()
_SOURCE_CHECKED=False


def source():
    global _SOURCE_CHECKED
    if not _SOURCE_CHECKED:
        manifest=json.loads((VENDOR/'source-manifest.json').read_text(encoding='utf-8'))
        if manifest.get('revision')!=SOURCE_REVISION:raise RuntimeError('Hi3D 固定源码版本不匹配')
        for name,expected in manifest['files'].items():
            path=(VENDOR/name).resolve()
            if not path.is_relative_to(VENDOR.resolve()) or hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=expected:
                raise RuntimeError('Hi3D 源码校验失败：'+name)
        _SOURCE_CHECKED=True
    existing=sys.modules.get('sparc3d')
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(VENDOR.resolve()):
        raise RuntimeError('另一个节点已载入不同版本的 sparc3d；请隔离冲突节点后重启 ComfyUI')
    if str(VENDOR) not in sys.path:sys.path.insert(0,str(VENDOR))


@contextmanager
def attention(backend):
    from sparc3d.nn.sparse.attention import full_attn
    from sparc3d.models.autoencoders import blocks
    previous=(full_attn._flash_attn,blocks._flash_attn)
    try:
        if backend=='sdpa':
            from .attention import SDPACompat
            full_attn._flash_attn=lambda:SDPACompat
            blocks._flash_attn=lambda:SDPACompat
        elif backend=='flash':
            try:import flash_attn  # noqa: F401
            except ImportError as error:raise RuntimeError('缺少 FlashAttention，请在 Hi3D 加载器选择 sdpa') from error
        else:raise ValueError('无效 Hi3D 注意力选项')
        yield
    finally:full_attn._flash_attn,blocks._flash_attn=previous


@contextmanager
def loading_progress(check):
    from sparc3d.pipelines import loader
    originals={name:getattr(loader,name) for name in ('load_vae','load_conditioner','load_transformer')}
    try:
        for name,original in originals.items():
            def observed(*args,_name=name,_original=original,**kwargs):
                check()
                label=str(args[1]) if _name=='load_transformer' and len(args)>1 else _name.removeprefix('load_')
                began=time.perf_counter();print('[Hi3D] Loading '+label+' on CPU',flush=True)
                value=_original(*args,**kwargs)
                print(f'[Hi3D] Loaded {label}: {time.perf_counter()-began:.1f}s',flush=True)
                check()
                return value
            setattr(loader,name,observed)
        yield
    finally:
        for name,original in originals.items():setattr(loader,name,original)


class ModelHandle:
    def __init__(self,root,offload=True,backend='sdpa',keep_loaded=False,download_missing=False):
        self.root=Path(root);self.offload=offload;self.backend=backend
        self.keep_loaded=keep_loaded;self.download_missing=download_missing
        self.pipeline=None;self.fingerprint=None

    def release(self):
        pipeline=self.pipeline;self.pipeline=None;self.fingerprint=None
        if pipeline is not None:
            from accelerate.hooks import remove_hook_from_module
            for module in [pipeline.conditioner,pipeline.transformer_512,pipeline._transformer_1024,pipeline.vae]:
                if module is not None:
                    remove_hook_from_module(module,recurse=True)
                    module.cpu()
            del pipeline
        gc.collect()

    def _load(self,progress,device):
        import torch
        import comfy.model_management as mm
        root=models.prepare(self.root,self.download_missing)
        signature=models.identity(root)
        if self.pipeline is not None and signature!=self.fingerprint:self.release()
        if self.pipeline is None:
            from sparc3d import Sparc3DPipeline
            print('[Hi3D] Loading local weights in ComfyUI: '+str(root),flush=True)
            # ComfyUI executes nodes in inference_mode. Keep persistent weights
            # normal tensors and preserve the user's CPU random state while constructing them.
            with loading_progress(mm.throw_exception_if_processing_interrupted),torch.inference_mode(False),torch.no_grad(),torch.random.fork_rng(devices=[]):
                self.pipeline=Sparc3DPipeline.from_pretrained(root/'hi3d',local_files_only=True,
                    dino_model_id=root/'dino',clip_model_id=root/'clip',progress_callback=progress)
                if self.offload:self.pipeline.enable_sequential_cpu_offload(device)
                else:self.pipeline.to(device)
            self.fingerprint=signature
        self.pipeline.progress_callback=progress
        return self.pipeline

    def refine(self,image,mesh,resolution,steps,seed):
        import torch
        import comfy.model_management as mm
        from comfy.utils import ProgressBar
        if not torch.cuda.is_available():raise RuntimeError('Hi3D 原模型需要 NVIDIA CUDA 显卡')
        device=mm.get_torch_device()
        if device.type!='cuda' or torch.cuda.get_device_capability(device)[0]<8:
            raise RuntimeError('Hi3D 需要计算能力 8.0 或以上的 NVIDIA CUDA 显卡')
        with _RUN_LOCK,torch.cuda.device(device):
            mm.throw_exception_if_processing_interrupted()
            mm.unload_all_models();mm.soft_empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            source()
            total=steps*({512:1,1024:2,1536:3}[resolution]);pbar=ProgressBar(total)
            pbar.update_absolute(0,total)
            def progress(event):
                mm.throw_exception_if_processing_interrupted()
                pbar.update_absolute(event.pass_index*steps+event.step,total)
                print(f'[Hi3D] Pass {event.pass_index+1}/{event.num_passes}: {event.step}/{event.num_inference_steps}',flush=True)
            started=time.perf_counter();success=False;originals=[]
            try:
                with attention(self.backend):
                    pipeline=self._load(progress,device)
                    # Check cancellation before and after every native stage.
                    for method in ['_adaptive_voxelize','_sample_latents','_decode_latents','_extract_mesh','_postprocess_mesh']:
                        original=getattr(pipeline,method);originals.append((method,original))
                        def observed(*a,_original=original,_method=method,**kw):
                            mm.throw_exception_if_processing_interrupted()
                            print('[Hi3D] '+_method.lstrip('_'),flush=True)
                            result=_original(*a,**kw)
                            mm.throw_exception_if_processing_interrupted()
                            return result
                        setattr(pipeline,method,observed)
                    result=pipeline(image,mesh,resolution=resolution,num_inference_steps=steps,seed=seed)
                    mm.throw_exception_if_processing_interrupted()
                    report={'status':'succeeded','runtime':'ComfyUI in-process','source_revision':SOURCE_REVISION,
                        'requested_resolution':resolution,'actual_resolution':result.resolution,'steps':steps,'seed':seed,
                        'offload':self.offload,'attention':self.backend,'seconds':round(time.perf_counter()-started,3),
                        'torch':str(torch.__version__),'vertices':len(result.mesh.vertices),'faces':len(result.mesh.faces),
                        'peak_allocated_mb':round(torch.cuda.max_memory_allocated(device)/2**20,2),
                        'memory_metric':'PyTorch CUDA allocator; native allocations excluded',
                        'texture_generated':False,'normalized':True}
                    print('[Hi3D] REPORT '+json.dumps(report,ensure_ascii=False),flush=True)
                    output=result.mesh
                    success=True
                    return output,report
            finally:
                if self.pipeline is not None:
                    for method,original in originals:setattr(self.pipeline,method,original)
                    self.pipeline.progress_callback=None
                originals.clear()
                # Do not mask the original inference/cancellation exception.
                if not success or not self.keep_loaded:
                    try:self.release()
                    except Exception as error:print('[Hi3D] Model cleanup: '+str(error),flush=True)
                mm.soft_empty_cache()
