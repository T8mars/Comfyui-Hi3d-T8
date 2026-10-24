"""One isolated process per coarse/texture stage; no ComfyUI node registration."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

from ..common import atomic_json, ROOT
from ..pipeline_models import MODEL_ROOT, SPECS


def emit(stage, **values):
    print("HI3D_EVENT " + json.dumps({"stage": stage, "time": time.time(), **values}, ensure_ascii=False), flush=True)


def source_dir(name):
    # venv launchers resolve to the venv interpreter on Windows.
    runtime = Path(sys.prefix)
    directory = runtime / "sources" / name
    if not directory.is_dir():
        raise FileNotFoundError(f"缺少独立推理源码: {directory}; 请先运行 setup_pipeline_runtime.py")
    import hashlib
    from ..pipeline_models import _read
    manifest = _read(directory / "source-manifest.json")
    expected = {"hunyuan": "f8db63096c8282cb27354314d896feba5ba6ff8a", "trellis2": "75fbf0183001ed9876c8dbb35de6b68552ee08bd"}[name]
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
    source_dir("hunyuan")
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    model_dir = Path(request.get("model_root") or MODEL_ROOT) / "coarse"
    emit_fn("loading", message="加载 Hunyuan3D-2mini Turbo 基础几何模型")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        str(model_dir), subfolder="hunyuan3d-dit-v2-mini-turbo", use_safetensors=True,
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
    steps = 5
    original_condition = pipeline.encode_cond
    def conditioned(*args, **kwargs):
        value = original_condition(*args, **kwargs)
        emit_fn("sampling", message="生成基础几何", steps=steps)
        return value
    pipeline.encode_cond = conditioned
    original_export = pipeline._export
    def decoded(*args, **kwargs):
        emit_fn("decoding", message="解码网格表面")
        return original_export(*args, **kwargs)
    pipeline._export = decoded
    emit_fn("conditioning", message="编码参考图片")
    mesh = pipeline(image=image, num_inference_steps=steps, octree_resolution=int(request.get("resolution", 256)),
                    num_chunks=8000, generator=torch.Generator(device="cpu").manual_seed(int(request.get("seed", 0))),
                    callback=lambda step, timestep, values: emit_fn("sampling", step=int(step)+1, total_steps=steps), callback_steps=1,
                    output_type="trimesh")[0]
    if mesh is None or len(mesh.faces) == 0:
        raise RuntimeError("基础模型没有生成有效网格")
    emit_fn("exporting", faces=len(mesh.faces))
    mesh.export(output)
    return {"model": SPECS["coarse"], "geometry_generated": True, "texture_generated": False,
            "vertices": len(mesh.vertices), "faces": len(mesh.faces), "inference_steps": steps,
            "volume_decoder": "official dense VanillaVolumeDecoder; checkpoint VAE preserved",
            "runtime_compatibility": "explicit component registry and CUDA execution device for upstream CPU-offload helper" if request.get("offload", True) else "standard CUDA residency",
            "transform": {"coordinate_system": "Hunyuan canonical", "unit_scale": "generator canonical; no absolute scale"}}


def texture(request, image, output, emit_fn):
    import torch
    import numpy as np
    import trimesh
    source_dir("trellis2")
    os.environ["ATTN_BACKEND"] = "sdpa"
    os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"
    os.environ["SPARSE_CONV_BACKEND"] = "flex_gemm"
    from trellis2.pipelines import Trellis2TexturingPipeline
    from trellis2.pipelines import rembg as upstream_rembg
    from .attention import install as install_sdpa
    install_sdpa()

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
    pipeline = Trellis2TexturingPipeline.from_pretrained(str(load_root), config_file=config_name)
    pipeline.cuda()
    loaded = trimesh.load(request["mesh_path"], force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.faces):
        raise ValueError("贴图输入不是有效三角网格")
    if not np.isfinite(loaded.vertices).all():
        raise ValueError("贴图输入包含无效顶点坐标")
    mesh = loaded.copy()
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
            emit_fn(_stage, message=messages[_stage])
            value = _original(*args, **kwargs)
            if request.get("offload", True):
                # Native CuMesh/nvdiffrast allocations cannot reuse memory
                # reserved by PyTorch after a model was offloaded to CPU.
                torch.cuda.empty_cache()
            return value
        setattr(pipeline, method, wrapped)
    textured = pipeline.run(mesh, image, seed=int(request.get("seed", 0)), resolution=int(request.get("resolution", 512)),
                            texture_size=int(request.get("texture_size", 2048)))
    # Official preprocess and postprocess apply inverse axis rotations. Restore
    # only the bbox normalization, so users retain their Hi3D geometry frame.
    textured.vertices = np.asarray(textured.vertices) / scale + center
    emit_fn("exporting", message="导出嵌入式 BaseColor 与 Metallic/Roughness 贴图 GLB")
    material = textured.visual.material
    if getattr(material, "baseColorTexture", None) is None or getattr(material, "metallicRoughnessTexture", None) is None:
        raise RuntimeError("PBR贴图输出不完整")
    textured.export(output, file_type="glb")
    return {"model": SPECS["texture"], "image_encoder": SPECS["texture_encoder"],
            "geometry_generated": False, "texture_generated": True, "texture_condition": "official mesh-to-shape-SLAT encoder on supplied mesh",
            "vertices": len(textured.vertices), "faces": len(textured.faces), "texture_size": int(request.get("texture_size", 2048)),
            "pbr_channels": ["baseColor", "metallic", "roughness", "alpha"],
            "attention": "PyTorch fused SDPA; explicit packed sparse-attention Windows adapter",
            "offload_cache_policy": "release unused PyTorch cache at official stage boundaries" if request.get("offload", True) else "standard CUDA allocator caching",
            "transform": {"normalization_center": center.tolist(), "normalization_scale": scale,
                          "normalization_restored": True, "mesh_input_frame_preserved": True,
                          "uv_seam_vertex_splits_possible": True}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--stage", choices=["coarse", "texture"], required=True)
    args = parser.parse_args()
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
        resolution = request.get("resolution", 256 if args.stage == "coarse" else 512)
        if resolution not in ({256} if args.stage == "coarse" else {512, 1024}):
            raise ValueError("基础几何分辨率应为256；贴图分辨率应为512或1024")
        if request.get("texture_size", 2048) not in {1024, 2048, 4096}:
            raise ValueError("贴图尺寸应为1024、2048或4096")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if os.name != "nt":
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        emit(args.stage + "_initializing", message="启动独立 CUDA 推理运行时", request_path=str(request_path))
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("需要 NVIDIA CUDA GPU")
        if os.name == 'nt' and args.stage == 'texture' and torch.cuda.get_device_capability() < (8, 6):
            raise RuntimeError('当前 Windows 预编译贴图扩展需要 CUDA 计算能力 8.6 或以上；该扩展包不支持 RTX 20 系列或 A100（8.0）。这不是模型本身的理论硬件下限。')
        torch.cuda.reset_peak_memory_stats()
        image, background = foreground(request, output.parent, args.stage)
        callback = lambda stage, **kw: emit(args.stage + "_" + stage, **kw)
        result = globals()[args.stage](request, image, temp, callback)
        if not temp.is_file() or temp.stat().st_size < 100:
            raise RuntimeError("推理没有生成有效GLB")
        torch.cuda.synchronize()
        result.update(base, status="succeeded", background=background, seconds=round(time.perf_counter()-started, 3),
                      peak_allocated_mb=round(torch.cuda.max_memory_allocated()/2**20, 2),
                      peak_reserved_mb=round(torch.cuda.max_memory_reserved()/2**20, 2),
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
        failure = {**base, "status": "failed", "model": SPECS[args.stage],
                   "geometry_generated": False, "texture_generated": False,
                   "error": str(exc), "seconds": round(time.perf_counter()-started, 3), "traceback": traceback.format_exc()}
        if "torch" in locals() and torch.cuda.is_initialized():
            try:
                failure["peak_allocated_mb"] = round(torch.cuda.max_memory_allocated()/2**20, 2)
                failure["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved()/2**20, 2)
            except Exception:
                pass  # Preserve the inference error if the CUDA context failed.
        atomic_json(report_path, failure)
        atomic_json(output.parent / "failure.json", failure)
        emit(args.stage + "_failed", error=str(exc))
        temp.unlink(missing_ok=True)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
