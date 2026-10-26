---
license: other
license_name: multiple-upstream-licenses
license_link: https://huggingface.co/t8star/Hi3d-Comfy/blob/main/README.md#licenses
language:
  - en
  - zh
tags:
  - hi3d
  - comfyui
  - 3d
  - geometry-refinement
  - image-to-3d
  - pbr
  - safetensors
---

# Hi3D T8 — Core and Workbench Models

**English** | [简体中文](README.zh-CN.md)

**By T8star.** Verified redistribution of the shared Hi3D core and all model components supported by the [standalone Hi3D T8 workbench](https://github.com/T8mars/Comfyui-Hi3d-T8). Original weights, configurations, pinned revisions and individual licenses are retained. UltraTex uses an upstream-published fp8 base; this project did not quantize it.

**ComfyUI only uses the Hi3D core for reference-image + coarse-mesh geometry refinement.** Image-to-3D and PBR extensions belong to the standalone workbench. This model repository contains neither the application nor Python. GitHub Releases contain code only; T8star will provide the complete package with Python and models separately.

Native nodes infer directly inside ComfyUI's own Python, without a workbench service. Upload the reference and coarse mesh to refine with the same original core weights. Replace legacy bridge canvases with the new example.

**Workbench v0.5.1 retains all existing model weights.** The optional Windows CUDA converter is code in `runtime-wheels/v0.4.9/`, installed separately by the workbench with a pinned hash; it is not a checkpoint or a Python distribution. The GPU converter keeps the upstream MIT notices.

## Included models

| Directories | Purpose | License |
|---|---|---|
| `hi3d/` | Hi3D V1.0 geometry refinement; 512 and 1024 checkpoints, shared VAE; 1536 uses 1024 weights | Hi3D Open License v1.0 |
| `dino/`, `clip/` | Required DINOv2 Large and CLIP ViT-L/14 core encoders | Apache-2.0 / MIT |
| `extensions/coarse/` | Hunyuan3D-2mini Turbo image-to-geometry | Tencent Hunyuan 3D 2.0 Community |
| `extensions/coarse_hunyuan21/` | Hunyuan3D 2.1 image-to-geometry | Tencent Hunyuan 3D 2.1 Community |
| `extensions/coarse_trellis2/`, `extensions/trellis_structure/` | TRELLIS.2 geometry and structure decoder | MIT |
| `extensions/texture/`, `extensions/texture_encoder/` | TRELLIS.2 PBR and DINOv3 encoder | MIT / DINOv3 License |
| `extensions/texture_hunyuan21/`, `extensions/paint_dino/`, `extensions/paint_upscaler/` | Hunyuan3D-Paint 2.1, DINOv2 Giant and Real-ESRGAN | Tencent / Apache-2.0 / BSD-3-Clause |
| `extensions/texture_ultratex/`, `extensions/ultratex_backbone/`, `extensions/ultratex_text_encoder/` | UltraTex adapters/decoder, FLUX.2 Klein 9B fp8, Qwen3 and VAE | MIT / FLUX Non-Commercial / Apache-2.0 |
| `extensions/ultratex_backbone_int8/` | tsolful Klein Base 9B INT8 ConvRot, pinned CivitAI version 3080042/file 2959286 | Original FLUX Non-Commercial |
| `extensions/background/` | Optional CPU U2Netp foreground extraction | MIT / Apache-2.0 |

`manifest.json` records the shared core. `extensions/manifest.json` records 14 extension components, original repositories and fixed revisions. `model-catalogue.json` lists all **95 model/config/license artifacts (79.3GB)** with byte counts and SHA256. ComfyUI and the workbench use identical Hi3D weights; no separate ComfyUI checkpoint exists. The INT8 file is the community-published **Base** quantization, not distilled/KV; it was not quantized by this project. FP8 remains available separately.

**Inventory checked on 2026-10-26:** all 95 local files match their manifests and this repository's hashes; no missing or changed checkpoints require uploading. Runtime wheels are counted separately from the model catalogue.

## Download

For native ComfyUI, use `ComfyUI/models/Hi3D/`. For the workbench, use its `models/` directory. The same core weights can also be selected through a custom model root. Replace `models` below with the desired target.

Core only (10.66GB), sufficient for the independent Hi3D ComfyUI nodes:

```bash
hf download t8star/Hi3d-Comfy --local-dir models --include "manifest.json" "hi3d/*" "dino/*" "clip/*" "licenses/*"
```

All core and extension files (about 79.3GB; allow additional runtime and working space):

```bash
hf download t8star/Hi3d-Comfy --local-dir models
```

Download the **complete component set in one row** for each workbench engine you need:

| Workbench selection | Directories under `extensions/` | Download size |
|---|---|---:|
| Hunyuan3D-2mini Turbo geometry | `coarse` | 3.82GB |
| Hunyuan3D 2.1 geometry | `coarse_hunyuan21` | 7.37GB |
| TRELLIS.2 geometry | `coarse_trellis2`, `trellis_structure`, `texture_encoder` | 10.06GB |
| TRELLIS.2 PBR | `texture`, `texture_encoder` | 8.04GB |
| Hunyuan3D-Paint 2.1 | `texture_hunyuan21`, `paint_dino`, `paint_upscaler` | 11.50GB |
| UltraTex FP8 | `texture_ultratex`, `ultratex_backbone`, `ultratex_text_encoder` | 19.62GB |
| UltraTex INT8 ConvRot | `texture_ultratex`, `ultratex_backbone_int8`, `ultratex_text_encoder` | 19.46GB |

Sizes use decimal GB, include each row's shared dependencies, and exclude Python, temporary files and optional accelerators. Shared directories download once, so do not add all rows to estimate the full repository. Hi3D refinement is optional in the workbench; add the core only if you enable it. Background removal adds `extensions/background/` (about 4.6MB).

For example, TRELLIS.2 geometry **and** PBR, with their shared encoder:

```bash
hf download t8star/Hi3d-Comfy --local-dir models --include "extensions/manifest.json" "extensions/coarse_trellis2/*" "extensions/trellis_structure/*" "extensions/texture/*" "extensions/texture_encoder/*" "licenses/*"
```

The workbench's Models page can also download only the required engine components from pinned original/public sources. Missing optional engines do not disable installed ones. Verify files in the workbench before inference.

To use another drive or shared location, select its root in **Models and environment → Model directory** and use that same path with `hf download --local-dir`. Keep `hi3d/`, `dino/`, `clip/` and `extensions/` directly under the selected root. The idle switch takes effect without restarting; it does not move existing files. Relative paths resolve against the workbench root, and code updates preserve the saved setting.

Existing core files at a newly selected location still require **Reverify all core models** before that location is reported ready.

## Program updates

The standalone bundle checks stable GitHub Releases, downloads code updates and installs them on the next launch. Models, Python runtimes, configuration and history are preserved. Code updates do not download this entire model repository. Runtime-incompatible versions require a new complete package. [Update instructions](https://github.com/T8mars/Comfyui-Hi3d-T8/blob/main/docs/UPDATES.md).

Acceleration packages are code, not model weights. The optional, hash-pinned `hi3d-voxel-cuda` wheel is stored under `runtime-wheels/v0.4.9/`; the workbench installs the compatible build separately. FlashAttention and Numba/llvmlite are obtained from the external sources pinned by the installer. SDPA and the official CPU converter remain available. Models never include a bundled Python environment. See the [performance guide](https://github.com/T8mars/Comfyui-Hi3d-T8/blob/main/docs/PERFORMANCE.md).

## Validation and limits

Real model runs were tested on Windows / RTX 5090 Laptop 24GB / 64GB RAM. Smaller GPUs have not been validated. Hi3D refinement is optional, rebuilds topology and does not generate textures; the separate PBR stage exports embedded-material GLB. Geometry, back views, UV seams and materials require inspection; printability and universal quality improvements are not guaranteed. [Engine evidence](https://github.com/T8mars/Comfyui-Hi3d-T8/blob/main/docs/ENGINES.md).

## Licenses

Each component retains its own license and attribution; no single MIT license covers the collection. Hi3D's LICENSE/NOTICE/AUTHORS are retained. DINOv3 retains the Meta agreement. UltraTex's **FLUX.2 Klein 9B base uses the FLUX Non-Commercial License**. See component directories, `licenses/` and the source manifests. [Powered by Hi3D 1.0](https://www.hi3d.ai/).

## T8star

[Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [Free Gallery](https://www.openzhenzhen.com) · [Online AI Apps](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI Distribution](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
