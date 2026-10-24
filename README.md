# Hi3D T8

**English** | [简体中文](README.zh-CN.md)

**By T8star** — Hi3D V1.0 geometry refinement for ComfyUI, with a separate local 3D workbench.

The **ComfyUI nodes** refine a **reference image + coarse mesh of the same object** using the official Hi3D model. Their inputs, outputs and canvas workflow remain geometry-only. The **standalone workbench v0.3** also generates an initial mesh from one image and creates UV/PBR materials on the final mesh. See [validated scope](roadmap.md).

[Models](https://huggingface.co/t8star/Hi3d-Comfy) · [Canvas workflow](examples/Hi3D-Refine.json) · [Validation](VALIDATION.md) · [Upstream Hi3D](https://github.com/hi3d-official/Hi3Dv1.0)

![Hi3D canvas workflow in ComfyUI](docs/images/comfyui-canvas.png)

## ComfyUI

1. Install **Hi3D T8** (`hi3d-t8`, publisher `t8star`) through ComfyUI-Manager once the Registry version is available. Alternatively clone this repository into `ComfyUI/custom_nodes/Comfyui-Hi3d-T8` and install `requirements.txt` with ComfyUI's Python.
2. Start the **separate local workbench backend** and wait for models to be verified at `http://127.0.0.1:8378`. See [backend setup](#local-workbench).
3. Drag [Hi3D-Refine.json](examples/Hi3D-Refine.json) onto the ComfyUI **canvas**. This is a visual workflow, not an API prompt. Upload the reference image in **Load Image** and set the absolute coarse-mesh path in **Hi3D Load Mesh**.
4. Start with **512 / 25 steps / CPU offload**. Select `sdpa` for Windows or `flash` for the official Linux stack, then click **Run**. Native **SaveGLB** writes the output to ComfyUI's output directory.

The three custom nodes are **Hi3D Backend**, **Hi3D Load Mesh** and **Hi3D Refine**. They exchange native ComfyUI `MESH` values. The backend runs on the same computer; the node package does not install or replace ComfyUI's PyTorch/CUDA. See [node instructions](docs/COMFYUI.md).

## Models and edition differences

| Component | Purpose | Used by |
|---|---|---|
| Hi3D transformer 512 + VAE | 512 refinement | ComfyUI and workbench |
| Hi3D transformer 1024 + the same VAE | 1024 and 1536 refinement; no separate 1536 checkpoint | ComfyUI and workbench |
| DINOv2 Large + CLIP ViT-L/14 | Required image encoders, not extra 3D models | ComfyUI and workbench |
| Hunyuan3D-2mini-Turbo | Generate an initial mesh from an image | Workbench only |
| TRELLIS.2 TexturingPipeline + DINOv3 | Generate UV/PBR materials on the final mesh | Workbench only |
| CPU U2Netp | Optional foreground extraction for opaque images | Workbench only |

The [Hugging Face repository](https://huggingface.co/t8star/Hi3d-Comfy) contains the **shared, unmodified Hi3D core weights and configurations**. There are no ComfyUI-specific checkpoints. The workbench's additional models are managed separately under `models/extensions`, downloaded from pinned upstream/public redistribution repositories with their own licenses. The Registry/node download contains **code and the canvas workflow only**. Model files alone do not provide an inference environment.

Download into the **workbench root**, not `ComfyUI/models`:

```bash
hf download t8star/Hi3d-Comfy --local-dir models
```

Expected layout: `models/manifest.json`, `models/hi3d/`, `models/dino/`, `models/clip/`. The manifest records pinned upstream revisions and SHA256 hashes. The backend also supports downloading directly from the original upstream repositories.

## Local workbench

The Windows edition starts with **Start-Workbench.cmd** and opens `http://127.0.0.1:8378`, independently of ComfyUI. The workbench has three entry points: image → initial mesh → optional Hi3D → PBR; reference + existing mesh → PBR; and reference + coarse mesh → Hi3D, optionally followed by PBR. Image mode defaults to **initial mesh + PBR**, with Hi3D optional: a paired cartoon test retained a cleaner mouth without refinement. It retains intermediate models, logs and reports. Material results export as self-contained **GLB**; **STL/PLY contain geometry only**.

`Hi3D-Workbench-Windows-0.3.2.zip` is the complete Windows edition, with two isolated runtimes and verified models (about **34GiB unpacked**). The node and source archives contain code; source installations must prepare their runtimes and models separately. All three workbench entry modes have passed real HTTP inference and material-preserving export checks. See the [workbench guide](docs/WORKBENCH.zh-CN.md).

For a Windows source installation, use Python 3.10 in a separate environment:

```powershell
git clone https://github.com/T8mars/Comfyui-Hi3d-T8.git
cd Comfyui-Hi3d-T8
powershell -ExecutionPolicy Bypass -File scripts/Setup.ps1 -Python C:/Python310/python.exe -DownloadModels
```

For Linux Hi3D refinement, run `bash scripts/setup-linux.sh`, then `.venv/bin/python -m hi3d_workbench.model_cli download` and `.venv/bin/python -m hi3d_workbench --open`. Linux setup follows the official Ampere/Hopper stack and has not been tested on this machine. The additional image/PBR runtime is currently validated on Windows. See [detailed workbench guide (Chinese)](docs/WORKBENCH.zh-CN.md).

## Validation and limits

Tested on **Windows / RTX 5090 Laptop 24GB / 64GB RAM**: real ComfyUI 512 refinement and standalone 512/1024/1536 runs. The current suite passes **154 tests**; evidence for the asset pipeline is in [its validation record](validation/asset-pipeline-progress.md), with earlier refinement checks in [VALIDATION.md](VALIDATION.md). Lower-VRAM GPUs have not been validated. PyTorch tensor peaks are not total GPU memory requirements.

Windows uses an explicitly identified **SDPA compatibility adapter**; it is not the official Linux FlashAttention environment. Hi3D normalizes scale and rebuilds topology; its refinement stage produces geometry only. Results are not guaranteed watertight, printable or more accurate than the input; inspect both versions before export.

The bundled Windows texturing extensions require **CUDA compute capability 8.6+**; this excludes RTX 20-series and A100 (8.0) for that stage. Passing this architecture check is not a VRAM or device-validation guarantee.

## License and attribution

Original integration code: [MIT](LICENSE). Hi3D weights and upstream code: **Hi3D Open License v1.0**, with their original notices; encoder and runtime licenses remain separate. This is a community integration. See [NOTICE.md](NOTICE.md). [Powered by Hi3D 1.0](https://www.hi3d.ai/).

## T8star

[Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [Free Gallery](https://www.openzhenzhen.com) · [Online AI Apps](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI Distribution](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
