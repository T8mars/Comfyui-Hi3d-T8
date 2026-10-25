"""One isolated process per coarse/texture stage; no ComfyUI node registration."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

from ..common import atomic_json, ROOT, digest
from ..pipeline_models import MODEL_ROOT, SPECS
from ..engine_catalog import engine


def emit(stage, **values):
    print("HI3D_EVENT " + json.dumps({"stage": stage, "time": time.time(), **values}, ensure_ascii=False), flush=True)


def restore_texture_frame(mesh, center, scale):
    """Undo uniform bbox normalization without losing authored normals."""
    import numpy as np
    # Assigning vertices invalidates Trimesh's normal cache, including the
    # shading normals explicitly returned by the upstream PBR baker.
    normals = np.array(mesh.vertex_normals, copy=True)
    mesh.vertices = np.asarray(mesh.vertices) / scale + center
    mesh.vertex_normals = normals


def source_dir(name):
    # venv launchers resolve to the venv interpreter on Windows.
    runtime = Path(sys.prefix)
    directory = runtime / "sources" / name
    if not directory.is_dir():
        raise FileNotFoundError(f"缺少独立推理源码: {directory}; 请先运行 setup_pipeline_runtime.py")
    import hashlib
    from ..pipeline_models import _read
    manifest = _read(directory / "source-manifest.json")
    expected = {"hunyuan": "f8db63096c8282cb27354314d896feba5ba6ff8a", "trellis2": "75fbf0183001ed9876c8dbb35de6b68552ee08bd",
                "hunyuan21": "82920d643c0dc2f7bfd7255f45f62d386edfe60c",
                "ultratex_bridge": "8cc29849dd076ddec0fe5a1ec6207945dedbbbf1",
                "comfy_core": "7cebca52227c63b4180ceed448cbd7ad22903b6d"}[name]
    if manifest.get("revision") != expected or not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise RuntimeError(f"推理源码清单无效: {name}")
    for relative, digest in manifest["files"].items():
        candidate = (directory / relative).resolve()
        if not candidate.is_relative_to(directory.resolve()) or not candidate.is_file() or hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"推理源码校验失败: {name}/{relative}")
    sys.path.insert(0, str(directory))
    return directory


def foreground(request, output_dir, stage):
    import numpy as np
    from PIL import Image, ImageOps
    image = ImageOps.exif_transpose(Image.open(request["image_path"])).convert("RGBA")
    alpha = np.asarray(image.getchannel("A"))
    used = "input_alpha"
    # Match upload metadata/model gating: near-opaque alpha noise is not a mask.
    if not np.any(alpha < 250):
        if request.get("background", "auto") == "keep":
            used = "keep_opaque"
        else:
            emit(stage + "_background", message="CPU U2Net 前景提取")
            import onnxruntime as ort
            from ..pipeline_models import _hash
            background_path = Path(request.get("model_root") or MODEL_ROOT) / "background" / "u2netp.onnx"
            expected = SPECS["background"]["artifacts"]["u2netp.onnx"]["sha256"]
            if not background_path.is_file() or _hash(background_path) != expected:
                raise RuntimeError("CPU前景模型未就绪，请先在工作台下载扩展模型（background）；推理阶段不会下载权重")
            # U2netp preprocessing/mask reconstruction follows rembg's MIT
            # U2netpSession/BaseSession. Its license is bundled alongside the
            # weight. Direct ONNX execution needs no download/session registry,
            # alpha-matting, numba, or llvmlite dependencies.
            session = ort.InferenceSession(str(background_path), providers=["CPUExecutionProvider"])
            pixels = np.asarray(image.convert("RGB").resize((320, 320), Image.Resampling.LANCZOS)).astype(np.float64)
            pixels /= max(float(pixels.max()), 1e-6)
            pixels = (pixels - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            value = np.expand_dims(pixels.transpose(2, 0, 1), 0).astype(np.float32)
            prediction = session.run(None, {session.get_inputs()[0].name: value})[0][0, 0]
            span = float(prediction.max() - prediction.min())
            if not np.isfinite(prediction).all() or span <= 1e-8:
                raise ValueError("CPU前景提取没有识别到有效主体，请使用透明PNG或保留背景")
            mask = Image.fromarray(((prediction - prediction.min()) / span * 255).astype(np.uint8)).resize(image.size, Image.Resampling.LANCZOS)
            image.putalpha(mask)
            used = "u2netp_cpu"
    alpha = np.asarray(image.getchannel("A"))
    if np.count_nonzero(alpha > 16) < 32:
        raise ValueError("参考图没有有效前景；请提供主体清楚的图片或透明PNG")
    image.save(output_dir / "foreground.png")
    return image, used


def coarse(request, image, output, emit_fn):
    import torch
    choice = engine('coarse', request.get('engine'))
    if choice['id'] == 'trellis2':
        from .trellis_geometry import generate
        return generate(request, image, output, emit_fn)
    full = choice['id'] == 'hunyuan21'
    if full:
        directory = source_dir('hunyuan21')
        sys.path.insert(0, str(directory / 'hy3dshape'))
        from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
    else:
        source_dir("hunyuan")
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    component = 'coarse_hunyuan21' if full else 'coarse'
    model_dir = Path(request.get("model_root") or MODEL_ROOT) / component
    emit_fn("loading", message=f"加载 {choice['name']} 基础几何模型")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        str(model_dir), subfolder="hunyuan3d-dit-v2-1" if full else "hunyuan3d-dit-v2-mini-turbo", use_safetensors=not full,
        device="cpu" if request.get("offload", True) else "cuda", dtype=torch.float16,
    )
    # Keep the checkpoint's VAE and official default dense volume decoder.
    # The pinned optional hierarchical decoder casts world-space query steps
    # to int64 at refinement levels, losing fractional coordinates. Dense
    # chunked decoding avoids that upstream bug and preserves full-grid detail.
    if request.get("offload", True):
        # The pinned standalone upstream class does not inherit Diffusers'
        # components property, although its official offload helper reads it.
        pipeline.components = {name: getattr(pipeline, name) for name in ("conditioner", "model", "vae")}
        pipeline.enable_model_cpu_offload(device="cuda")
        # The upstream offload helper moves modules to CPU but its sampler
        # still reads pipeline.device rather than _execution_device.
        # Keep allocations on the hooks' execution device without moving all
        # models onto the GPU at once.
        pipeline.device = torch.device("cuda")
    # The fixed mini-Turbo checkpoint is step distilled. Use the upstream
    # mini-Turbo example's five steps rather than a base-model step count.
    steps = int(request.get('geometry_steps', 50)) if full else 5
    original_condition = pipeline.encode_cond
    def conditioned(*args, **kwargs):
        value = original_condition(*args, **kwargs)
        emit_fn("sampling", message="生成基础几何", steps=steps,device='cuda')
        return value
    pipeline.encode_cond = conditioned
    original_export = pipeline._export
    def decoded(*args, **kwargs):
        emit_fn("decoding", message="解码网格表面",device='cuda')
        return original_export(*args, **kwargs)
    pipeline._export = decoded
    emit_fn("conditioning", message="编码参考图片",device='cuda')
    mesh = pipeline(image=image, num_inference_steps=steps, octree_resolution=int(request.get("resolution", 256)),
                    num_chunks=8000, generator=torch.Generator(device="cpu").manual_seed(int(request.get("seed", 0))),
                    callback=lambda step, timestep, values: emit_fn("sampling", step=int(step)+1, num_inference_steps=steps,device='cuda'), callback_steps=1,
                    output_type="trimesh")[0]
    if mesh is None or len(mesh.faces) == 0:
        raise RuntimeError("基础模型没有生成有效网格")
    emit_fn("exporting", faces=len(mesh.faces))
    mesh.export(output)
    return {"model": SPECS[component], "engine": choice['id'], "geometry_generated": True, "texture_generated": False,
            "vertices": len(mesh.vertices), "faces": len(mesh.faces), "inference_steps": steps,
            "volume_decoder": "official dense VanillaVolumeDecoder; checkpoint VAE preserved",
            "runtime_compatibility": "explicit component registry and CUDA execution device for upstream CPU-offload helper" if request.get("offload", True) else "standard CUDA residency",
            "transform": {"coordinate_system": "Hunyuan canonical", "unit_scale": "generator canonical; no absolute scale"}}


def texture(request, image, output, emit_fn):
    choice = engine('texture', request.get('engine'))
    if choice['id'] != 'trellis2':
        if choice['id'] in {'ultratex', 'ultratex_int8'}:
            from .ultratex_texture import generate
        else:
            from .hunyuan_texture import generate
        return generate(request, image, output, emit_fn)
    import torch
    import numpy as np
    import trimesh
    source_dir("trellis2")
    os.environ["ATTN_BACKEND"] = "sdpa"
    os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"
    os.environ["SPARSE_CONV_BACKEND"] = "flex_gemm"
    emit_fn('loading',phase='runtime_import',device='cpu',message='载入 TRELLIS.2 代码与本地依赖')
    from trellis2.pipelines import Trellis2TexturingPipeline
    from trellis2.pipelines import rembg as upstream_rembg
    from .attention import install as install_attention
    attention = install_attention(request.get('pipeline_attention', 'auto'), emit=emit_fn)

    # UV construction can use substantial native CUDA memory. Run it before
    # loading the diffusion models, including when CPU offloading is disabled.
    from ..asset_pipeline import load_texture_input
    loaded, uv_policy = load_texture_input(Path(request['mesh_path']))
    from .uv_runtime import prepare
    loaded,uv_generation,uv_runtime=prepare(loaded,request,emit_fn)

    class PreparedForeground:
        def __init__(self, **kwargs): pass
        def to(self, device): return self
        def cpu(self): return self
        def __call__(self, value): return value.convert("RGBA")

    # The official default uses gated/noncommercial RMBG-2.0. We load no such
    # weights: foreground is prepared above, or explicitly kept by the user.
    upstream_rembg.PreparedForeground = PreparedForeground
    root = Path(request.get("model_root") or MODEL_ROOT)
    model_dir = root / "texture"
    config = json.loads((model_dir / "texturing_pipeline.json").read_text())
    requested_resolution = int(request.get("resolution", 512))
    unused_flow = "tex_slat_flow_model_1024" if requested_resolution == 512 else "tex_slat_flow_model_512"
    config["args"]["models"].pop(unused_flow, None)
    config["args"]["image_cond_model"]["args"]["model_name"] = str(root / "texture_encoder")
    config["args"]["rembg_model"] = {"name": "PreparedForeground", "args": {}}
    config["args"]["low_vram"] = bool(request.get("offload", True))
    local_config = output.parent / "texturing-runtime.json"
    emit_fn("loading", message="加载 TRELLIS.2 官方网格贴图模型")
    try:
        config_name = os.path.relpath(local_config, model_dir)
        load_root = model_dir
    except ValueError:
        # Windows can place custom weights and jobs on different drives. The
        # official loader's fallback accepts absolute checkpoint stems.
        config["args"]["models"] = {key: str((model_dir / value).resolve()) for key, value in config["args"]["models"].items()}
        load_root, config_name = output.parent, local_config.name
    # Upstream opens this JSON without an explicit encoding. ASCII escapes
    # preserve Chinese/non-ASCII portable paths under Windows legacy locales.
    config_temp = local_config.with_name(local_config.name + ".tmp")
    config_temp.write_text(json.dumps(config, ensure_ascii=True, indent=2), encoding="ascii")
    os.replace(config_temp, local_config)
    from trellis2 import models as official_models
    from trellis2.modules import image_feature_extractor
    from .model_loading import deferred_models, release, flow_arguments
    expected_models={key:(Path(value) if Path(value).is_absolute() else load_root/value)
                     for key,value in config['args']['models'].items()}
    with deferred_models(official_models,expected_models,image_feature_extractor,request,emit_fn):
        pipeline = Trellis2TexturingPipeline.from_pretrained(str(load_root), config_file=config_name)
    emit_fn('loading',phase='model_plan',device='cpu',message='按需加载计划已就绪；各组件将在使用前载入')
    pipeline.cuda()
    from .sampler_progress import track_sampler
    track_sampler(pipeline.tex_slat_sampler, emit_fn, 'material', '生成 PBR 材质')
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.faces):
        raise ValueError("贴图输入不是有效三角网格")
    if not np.isfinite(loaded.vertices).all():
        raise ValueError("贴图输入包含无效顶点坐标")
    mesh = loaded.copy(include_cache=True)
    original_preprocess_mesh = pipeline.preprocess_mesh
    def preprocess_with_uv(value):
        prepared = original_preprocess_mesh(value)
        uv = getattr(value.visual, 'uv', None)
        if uv is not None:
            # The upstream normalizer constructs a new geometry-only mesh,
            # although its baker explicitly supports existing UVs. Retain the
            # validated single-material atlas through that coordinate change.
            if len(prepared.vertices) != len(value.vertices) or not np.array_equal(prepared.faces, value.faces):
                raise RuntimeError('TRELLIS 网格预处理改变索引，无法安全保留原 UV')
            prepared.visual = trimesh.visual.TextureVisuals(uv=np.asarray(uv).copy())
            # Carry the input shading normals through the same axis rotation;
            # recomputing on UV seam vertices would create hard chart borders.
            normal=np.asarray(value.vertex_normals)[:,[0,2,1]].copy()
            normal[:,1]*=-1
            prepared.vertex_normals=normal
        return prepared
    pipeline.preprocess_mesh = preprocess_with_uv
    minimum = np.asarray(mesh.vertices).min(axis=0)
    maximum = np.asarray(mesh.vertices).max(axis=0)
    center = (minimum + maximum) / 2
    extent = float((maximum - minimum).max())
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("贴图输入网格尺寸无效")
    scale = 0.99999 / extent
    # Wrap official methods only for progress; inference remains official.
    for method, stage_name in (("get_cond", "conditioning"), ("encode_shape_slat", "encoding_mesh"),
                               ("sample_tex_slat", "sampling"), ("decode_tex_slat", "decoding"),
                               ("postprocess_mesh", "baking")):
        original = getattr(pipeline, method)
        def wrapped(*args, _original=original, _stage=stage_name, **kwargs):
            messages = {"conditioning": "分析参考图的颜色与材质", "encoding_mesh": "分析模型表面",
                        "sampling": "生成 PBR 材质", "decoding": "重建材质细节", "baking": "展开 UV 并烘焙贴图"}
            if _stage not in {'encoding_mesh', 'baking'}:
                emit_fn(_stage, message=messages[_stage], device='cuda')
            if _stage=='sampling':
                # Upstream distinguishes nn.Module from a list of cascaded
                # flows. Pass the real module at that typed API boundary.
                args,kwargs=flow_arguments(args,kwargs)
            if _stage == 'baking' and args and getattr(args[0].visual,'uv',None) is not None:
                # With pre-existing UVs the official baker rotates the cached
                # normal array in place. Trimesh exposes that cache read-only;
                # make this temporary normalized mesh's array writable, as in
                # the upstream UV-unwrapping branch which indexes a fresh copy.
                args[0].vertex_normals.setflags(write=True)
            if _stage == 'encoding_mesh':
                from .encoding_progress import observe_shape_encoding
                with observe_shape_encoding(pipeline, emit_fn):
                    value = _original(*args, **kwargs)
            elif _stage == 'baking':
                from .baking_progress import call_baking_with_progress
                value = call_baking_with_progress(_original, emit_fn, *args, **kwargs)
            else:
                value = _original(*args, **kwargs)
            if _stage == 'conditioning':release(pipeline.image_cond_model)
            for name in {'encoding_mesh':('shape_slat_encoder',),'sampling':('tex_slat_flow_model_512','tex_slat_flow_model_1024'),
                         'decoding':('tex_slat_decoder',)}.get(_stage,()):
                if name in pipeline.models:release(pipeline.models[name])
            if request.get("offload", True):
                # Native CuMesh/nvdiffrast allocations cannot reuse memory
                # reserved by PyTorch after a model was offloaded to CPU.
                torch.cuda.empty_cache()
            return value
        setattr(pipeline, method, wrapped)
    textured = pipeline.run(mesh, image, seed=int(request.get("seed", 0)), resolution=int(request.get("resolution", 512)),
                            texture_size=int(request.get("texture_size", 2048)),
                            tex_slat_sampler_params={'steps': int(request.get('texture_steps',12))})
    # Official preprocess and postprocess apply inverse axis rotations. Restore
    # only the bbox normalization, so users retain their Hi3D geometry frame.
    restore_texture_frame(textured, center, scale)
    emit_fn("exporting", message="导出嵌入式 BaseColor 与 Metallic/Roughness 贴图 GLB")
    material = textured.visual.material
    if getattr(material, "baseColorTexture", None) is None or getattr(material, "metallicRoughnessTexture", None) is None:
        raise RuntimeError("PBR贴图输出不完整")
    from ..asset_pipeline import repair_invalid_normals
    normal_repair=repair_invalid_normals(textured)
    textured.export(output, file_type="glb")
    return {"model": SPECS["texture"], "engine": 'trellis2', "inference_steps": int(request.get('texture_steps',12)), "image_encoder": SPECS["texture_encoder"],
            "geometry_generated": False, "texture_generated": True, "texture_condition": "official mesh-to-shape-SLAT encoder on supplied mesh",
            'uv_policy': uv_policy,
            'uv_generation':uv_generation,'uv_runtime':uv_runtime,'shading_normals':'input normals preserved through UV and frame conversion',
            'normal_repair':normal_repair,
            "vertices": len(textured.vertices), "faces": len(textured.faces), "texture_size": int(request.get("texture_size", 2048)),
            "pbr_channels": ["baseColor", "metallic", "roughness", "alpha"],
            "attention": attention,
            "offload_cache_policy": "release unused PyTorch cache at official stage boundaries" if request.get("offload", True) else "standard CUDA allocator caching",
            "transform": {"normalization_center": center.tolist(), "normalization_scale": scale,
                          "normalization_restored": True, "mesh_input_frame_preserved": True,
                          "uv_seam_vertex_splits_possible": True}}


def run_request(request_path, stage):
    from types import SimpleNamespace
    args=SimpleNamespace(request=Path(request_path),stage=stage)
    request_path = args.request.resolve()
    started = time.perf_counter()
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise ValueError("阶段请求必须为JSON对象")
        output = Path(request["output_path"]).resolve()
        if output.suffix.lower() != ".glb":
            raise ValueError("output_path must be a GLB path")
        output.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        failure = {"stage": args.stage, "request_path": str(request_path), "status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
        request_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(request_path.parent / "failure.json", failure)
        emit(args.stage + "_failed", error=str(exc))
        traceback.print_exc()
        return 1
    report_path = output.with_suffix(".stage.json")
    base = {"stage": args.stage, "request_path": str(request_path), "output_path": str(output), "status": "running"}
    (output.parent / "failure.json").unlink(missing_ok=True)
    atomic_json(report_path, base)
    temp = output.with_name(output.stem + ".partial.glb")
    try:
        if request.get("background", "auto") not in {"auto", "keep"}:
            raise ValueError("background must be auto or keep")
        if not Path(request["image_path"]).is_file():
            raise ValueError("参考图片不存在")
        if args.stage == "texture" and not Path(request["mesh_path"]).is_file():
            raise ValueError("贴图网格不存在")
        choice = engine(args.stage, request.get('engine'))
        resolution = request.get("resolution", choice['default_resolution'])
        if resolution not in choice['resolutions']:
            label = '基础几何分辨率' if args.stage == 'coarse' else '贴图采样分辨率'
            raise ValueError(f"{label}不适用于 {choice['name']}")
        request['resolution'] = resolution
        if not isinstance(request.get('geometry_steps', 50), int) or not 1 <= request.get('geometry_steps', 50) <= 100:
            raise ValueError('几何步数应为1至100的整数')
        if isinstance(request.get('geometry_steps'), bool):
            raise ValueError('几何步数不能为布尔值')
        if ('texture_steps' in request and (not isinstance(request['texture_steps'], int)
                or isinstance(request['texture_steps'], bool) or not 1 <= request['texture_steps'] <= 100)):
            raise ValueError('贴图步数应为1至100的整数')
        if request.get("texture_size", 2048) not in {1024, 2048, 4096}:
            raise ValueError("贴图尺寸应为1024、2048或4096")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if os.name != "nt":
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        emit(args.stage + "_initializing", phase='runtime_import', device='cpu', message="读取独立推理运行时（磁盘 → 内存）", request_path=str(request_path))
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("需要 NVIDIA CUDA GPU")
        if os.name == 'nt' and (args.stage == 'texture' or choice['id']=='trellis2') and torch.cuda.get_device_capability() < (8, 6):
            raise RuntimeError('当前 Windows 资产运行时的预编译扩展需要 CUDA 计算能力 8.6 或以上；该扩展包不支持 RTX 20 系列或 A100（8.0）。这不是模型本身的理论硬件下限。')
        torch.cuda.reset_peak_memory_stats()
        image, background = foreground(request, output.parent, args.stage)
        callback = lambda stage, **kw: emit(args.stage + "_" + stage, **kw)
        result = globals()[args.stage](request, image, temp, callback)
        from .model_loading import finish_stage, CACHE
        finish_stage()
        result['model_loading']=CACHE.status()
        if not temp.is_file() or temp.stat().st_size < 100:
            raise RuntimeError("推理没有生成有效GLB")
        torch.cuda.synchronize()
        result['input_image_sha256'] = digest(Path(request['image_path']))
        if args.stage == 'texture':
            result['input_mesh_sha256'] = digest(Path(request['mesh_path']))
        result['output_sha256'] = digest(temp)
        result.update(base, status="succeeded", background=background, seconds=round(time.perf_counter()-started, 3),
                      peak_allocated_mb=max(result.get('peak_allocated_mb',0), round(torch.cuda.max_memory_allocated()/2**20, 2)),
                      peak_reserved_mb=max(result.get('peak_reserved_mb',0), round(torch.cuda.max_memory_reserved()/2**20, 2)),
                      memory_metric="PyTorch CUDA allocator; external native allocations are not included",
                      runtime={"torch": str(torch.__version__), "cuda": torch.version.cuda,
                               "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability())})
        atomic_json(report_path, result)
        # A stage report is consumed only after this process exits successfully.
        # Finish CUDA/error checks and persist it before replacing a previous
        # deliverable; a failed synchronization or full disk must preserve it.
        os.replace(temp, output)
        emit(args.stage + "_complete", **{k:v for k,v in result.items() if k not in {"stage", "model", "image_encoder"}})
    except BaseException as exc:
        if 'torch' in locals():
            try:
                from .model_loading import finish_stage
                finish_stage(success=False)
            except Exception:pass
        failed_choice = locals().get('choice', {})
        component = failed_choice.get('components', [args.stage])[0]
        failure = {**base, "status": "failed", "model": SPECS.get(component, {}), "engine": failed_choice.get('id', request.get('engine')),
                   "geometry_generated": False, "texture_generated": False,
                   "error": str(exc), "seconds": round(time.perf_counter()-started, 3), "traceback": traceback.format_exc()}
        if "torch" in locals() and torch.cuda.is_initialized():
            try:
                failure["peak_allocated_mb"] = round(torch.cuda.max_memory_allocated()/2**20, 2)
                failure["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved()/2**20, 2)
            except Exception:
                pass  # Preserve the inference error if the CUDA context failed.
        try:
            atomic_json(report_path, failure)
            atomic_json(output.parent / "failure.json", failure)
            emit(args.stage + "_failed", error=str(exc))
            traceback.print_exc()
        finally:
            # A full disk may also prevent saving the failure report. Cleanup
            # must not depend on that second write succeeding.
            temp.unlink(missing_ok=True)
        return 1
    return 0


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--stage',choices=['coarse','texture'])
    parser.add_argument('--session',action='store_true')
    args=parser.parse_args()
    if args.session:
        from ..runtime_session import serve
        return serve(lambda item:run_request(item['request'],item['stage']))
    if not args.stage:parser.error('--stage is required')
    return run_request(args.request,args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
