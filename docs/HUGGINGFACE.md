---
license: other
license_name: hi3d-open-license-v1.0
license_link: https://huggingface.co/t8star/Hi3d-Comfy/blob/main/LICENSE
language:
  - en
  - zh
tags:
  - hi3d
  - comfyui
  - 3d
  - geometry-refinement
  - safetensors
---

# Hi3D V1.0 — T8 Model Bundle

**English** | [简体中文](README.zh-CN.md)

**By T8star.** Verified, unmodified Hi3D V1.0 weights and required image encoders for [ComfyUI-Hi3D T8 and the local workbench](https://github.com/T8mars/Comfyui-Hi3d-T8). This is a community redistribution, not a new model, fine-tune or quantization.

Hi3D refines **a reference image and a coarse mesh of the same object**. It does not generate the initial coarse mesh, UVs or textures. No other 3D generation models are included.

## Files and provenance

| Directory | Original model | Pinned revision | License |
|---|---|---|---|
| `hi3d/` | [hi3d-official/Hi3Dv1.0](https://huggingface.co/hi3d-official/Hi3Dv1.0) | `705fc0901c5d896cdcbf1dbf1feaa173e62bdd3a` | Hi3D Open License v1.0 |
| `dino/` | [facebook/dinov2-large](https://huggingface.co/facebook/dinov2-large) | `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c` | Apache-2.0 |
| `clip/` | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) | `32bd64288804d66eefd0ccbe215aa642df71cc41` | MIT |

`manifest.json` records the SHA256 and byte size of every distributed model artifact. The model/configuration files are **identical to the offline workbench**. DINOv2 and CLIP are required image encoders, not additional 3D models. `transformer_512` serves 512 refinement; `transformer_1024` serves both 1024 and 1536. There is no separate 1536 checkpoint.

## Download and use

Run from the **workbench root**:

```bash
hf download t8star/Hi3d-Comfy --local-dir models
```

This produces `models/hi3d/`, `models/dino/`, `models/clip/` and `models/manifest.json`. Start the configured workbench backend and wait for SHA256 verification. In ComfyUI, load the [canvas workflow](https://github.com/T8mars/Comfyui-Hi3d-T8/blob/main/examples/Hi3D-Refine.json). Weights do not belong in `ComfyUI/models` for this integration.

**ComfyUI and the standalone workbench use the same weights and inference backend.** The Registry package contains node code and the canvas workflow; the complete offline workbench also contains Python, dependencies and models. This repository contains models and licenses only, not a standalone application. Setup: [English](https://github.com/T8mars/Comfyui-Hi3d-T8#local-workbench) / [中文](https://github.com/T8mars/Comfyui-Hi3d-T8/blob/main/README.zh-CN.md).

## Validation and limits

Real 512 ComfyUI and 512/1024/1536 workbench runs were validated on Windows / RTX 5090 Laptop 24GB / 64GB RAM. Smaller GPUs have not been validated. Start with 512, 25 steps and CPU offload. Windows uses an explicit SDPA compatibility adapter; the official Linux path uses FlashAttention. See [test evidence and limitations](https://github.com/T8mars/Comfyui-Hi3d-T8/blob/main/VALIDATION.md).

Outputs are geometry only. Hi3D normalizes scale and rebuilds topology; watertightness, printability and quality improvements are not guaranteed.

## License

Hi3D files retain the [Hi3D Open License v1.0](LICENSE), including its attribution, acceptable-use and scale conditions. Original `hi3d/LICENSE`, `hi3d/NOTICE`, `hi3d/AUTHORS` and `hi3d/THIRD_PARTY_LICENSES.md` are preserved. Encoder licenses are in `licenses/`. No MIT license is asserted for the bundle as a whole. [Powered by Hi3D 1.0](https://www.hi3d.ai/).

## T8star

[Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [Free Gallery](https://www.openzhenzhen.com) · [Online AI Apps](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI Distribution](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
