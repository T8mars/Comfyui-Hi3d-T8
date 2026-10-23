from __future__ import annotations
import argparse
import dataclasses
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from .common import ROOT, MODELS, SOURCE_REVISION, atomic_json, read_json, digest, config

def emit(stage, **kw):
    print("HI3D_EVENT " + json.dumps({"stage": stage, "time": time.time(), **kw}, ensure_ascii=False), flush=True)

def source_path():
    path = ROOT / "upstream" / "Hi3Dv1.0"
    if not (path / "sparc3d").is_dir():
        raise FileNotFoundError("缺少固定版本 Hi3D 源码，请运行安装脚本")
    manifest=read_json(ROOT/'upstream-source-manifest.json',{})
    if manifest.get('revision')!=SOURCE_REVISION or not manifest.get('files'):
        raise RuntimeError('缺少有效的固定版本 Hi3D 源码校验清单')
    for relative,expected in manifest.get('files',{}).items():
        if not (path/relative).is_file() or digest(path/relative)!=expected:
            raise RuntimeError(f'Hi3D 源码校验失败：{relative}')
    sys.path.insert(0, str(path))
    return path

def diagnostics(probe=False):
    report = {"python": sys.version, "platform": platform.platform(), "source_revision": SOURCE_REVISION,
              "packages": {}, "cuda": False, "kernel_probe": "not_run", "official_validated_environment": False}
    for name in ["torch", "torchvision", "triton", "triton-windows", "spconv-cu126", "spconv-cu120", "flash-attn", "transformers", "accelerate", "numpy"]:
        try: report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    try:
        import torch
        report["cuda"] = torch.cuda.is_available()
        if report["cuda"]:
            free,total = torch.cuda.mem_get_info()
            report.update(gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()), free_vram_mb=round(free/2**20), total_vram_mb=round(total/2**20), torch_cuda=torch.version.cuda)
        if probe:
            source_path()
            import spconv.pytorch as spconv
            from sparc3d.ops.voxelize import adaptive_vox
            import trimesh
            mesh = trimesh.creation.icosphere(subdivisions=1)
            coords = adaptive_vox(mesh, resolution=64, target_num=80000)
            report["probe_voxels"] = len(coords)
            # Verify a real sparse convolution, not just importability.
            indices = torch.tensor([[0,1,1,1],[0,1,1,2]],dtype=torch.int32,device="cuda")
            x = spconv.SparseConvTensor(torch.randn(2,8,device="cuda"),indices,[4,4,4],1)
            layer = spconv.SubMConv3d(8,8,3,padding=1).cuda()
            y = layer(x)
            assert torch.isfinite(y.features).all()
            from sparc3d.ops import sparse_backend
            official_layer=sparse_backend.Conv3d(8,8,3).cuda()
            for dtype in [torch.float32,torch.float16]:
                actual=official_layer(x.replace_feature(x.features.to(dtype)))
                assert actual.features.dtype==dtype and torch.isfinite(actual.features).all()
            report["official_sparse_conv_dtypes"]=["float32","float16"]
            from .attention import SDPACompat
            q = torch.randn(32,3,2,32,device="cuda",dtype=torch.float16)
            SDPACompat.flash_attn_varlen_qkvpacked_func(q,torch.tensor([0,32],device="cuda",dtype=torch.int32),32)
            torch.cuda.synchronize()
            report["kernel_probe"] = "passed"
    except Exception as e:
        report["error"] = str(e)
        report["traceback"] = traceback.format_exc()
        if probe: report["kernel_probe"] = "failed"
    return report

def run(request_path):
    started = time.time()
    req = read_json(request_path)
    jobdir = request_path.parent
    source_path()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import torch
    import numpy as np
    from PIL import Image
    from .geometry import load_mesh, inspect_mesh
    if not torch.cuda.is_available(): raise RuntimeError("需要 NVIDIA CUDA GPU")
    if torch.cuda.get_device_capability()[0]<8:raise RuntimeError("Hi3D 需要计算能力 8.0 或以上的 NVIDIA GPU")
    free,total = torch.cuda.mem_get_info()
    if free / 2**20 < config()["minimum_free_vram_mb"]:
        raise RuntimeError("当前空闲显存不足，请释放其他 GPU 任务后重试")
    if req["attention"] == "sdpa":
        from .attention import install_sdpa
        install_sdpa()
    from sparc3d import Sparc3DPipeline
    from sparc3d.utils.image import prepare_image
    emit("loading", message="从本地文件加载固定版本权重")
    torch.cuda.reset_peak_memory_stats()
    def progress(event):
        emit("sampling", **dataclasses.asdict(event))
    pipe = Sparc3DPipeline.from_pretrained(MODELS / "hi3d", local_files_only=True,
        dino_model_id=MODELS/"dino", clip_model_id=MODELS/"clip", progress_callback=progress)
    if req["offload"]: pipe.enable_sequential_cpu_offload("cuda")
    else: pipe.to("cuda")
    # Observe stages without modifying sampling, voxel budgets or postprocessing.
    import psutil
    metrics=[]
    peaks={"allocated":torch.cuda.max_memory_allocated(),"reserved":torch.cuda.max_memory_reserved()}
    for method,stage in [("_adaptive_voxelize","voxelizing"),("_sample_latents","sampling"),("_decode_latents","decoding"),("_extract_mesh","extracting"),("_postprocess_mesh","postprocessing")]:
        original = getattr(pipe,method)
        def observe(*a,_original=original,_stage=stage,**kw):
            emit(_stage)
            peaks['allocated']=max(peaks['allocated'],torch.cuda.max_memory_allocated())
            peaks['reserved']=max(peaks['reserved'],torch.cuda.max_memory_reserved())
            torch.cuda.reset_peak_memory_stats()
            stage_start=time.time()
            value = _original(*a,**kw)
            torch.cuda.synchronize()
            measured={"stage":_stage,"seconds":round(time.time()-stage_start,3),"peak_vram_mb":round(torch.cuda.max_memory_allocated()/2**20),"rss_mb":round(psutil.Process().memory_info().rss/2**20)}
            peaks['allocated']=max(peaks['allocated'],torch.cuda.max_memory_allocated())
            peaks['reserved']=max(peaks['reserved'],torch.cuda.max_memory_reserved())
            metrics.append(measured)
            emit(_stage, completed=True, **{k:v for k,v in measured.items() if k!='stage'})
            return value
        setattr(pipe,method,observe)
    image = Image.open(jobdir/"input.png").convert("RGBA")
    prepare_image(image).save(jobdir/"conditioning.png")
    coarse = load_mesh(jobdir/"coarse.glb")
    emit("conditioning")
    result = pipe(image,coarse,resolution=req["resolution"],num_inference_steps=req["steps"],seed=req["seed"])
    emit("exporting")
    result.mesh.export(jobdir/"refined.glb")
    result.mesh.export(jobdir/"refined.ply")
    result.mesh.export(jobdir/"refined.stl")
    np.savez_compressed(jobdir/"refined.npz",vertices=result.mesh.vertices,faces=result.mesh.faces)
    report = {"status":"succeeded", "requested_resolution":req["resolution"],"actual_resolution":result.resolution,
        "steps":req["steps"],"seed":req["seed"],"offload":req["offload"],"attention":req["attention"],
        "compatibility_port":req["attention"]=="sdpa", "seconds":round(time.time()-started,2),
        "peak_allocated_mb":round(max(peaks['allocated'],torch.cuda.max_memory_allocated())/2**20), "peak_reserved_mb":round(max(peaks['reserved'],torch.cuda.max_memory_reserved())/2**20),"stage_metrics":metrics,
        "gpu":torch.cuda.get_device_name(), "environment":diagnostics(), "model_manifest_sha256":digest(MODELS/"manifest.json"),
        "source_revision":SOURCE_REVISION,"input_sha256":digest(jobdir/"input.png"),"coarse_sha256":digest(jobdir/"coarse.glb"),
        "geometry":inspect_mesh(result.mesh),"normalized":True,"texture_generated":False}
    atomic_json(jobdir/"report.json",report)
    emit("complete",report=report)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--request",type=Path)
    p.add_argument("--diagnose",action="store_true")
    p.add_argument("--probe",action="store_true")
    args=p.parse_args()
    if args.diagnose:
        report=diagnostics(args.probe)
        print(json.dumps(report,ensure_ascii=False))
        if args.probe and report['kernel_probe']!='passed':raise SystemExit(1)
        return
    try:
        run(args.request)
    except Exception as e:
        text=str(e)
        report={"status":"failed","error":text,"error_type":"out_of_memory" if "out of memory" in text.lower() else type(e).__name__,"traceback":traceback.format_exc()}
        atomic_json(args.request.parent/"failure.json",report)
        emit("failed",**report)
        raise SystemExit(1)

if __name__=="__main__":main()
