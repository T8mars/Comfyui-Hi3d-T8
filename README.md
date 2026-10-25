# Hi3D T8

**English** | [简体中文](README.zh-CN.md)

**By T8star** — Hi3D V1.0 geometry refinement for ComfyUI, with a separate local 3D workbench.

The **ComfyUI nodes** refine a **reference image + coarse mesh of the same object** using the official Hi3D model. Their inputs, outputs and canvas workflow remain geometry-only. The **standalone workbench v0.4** also generates geometry from one image, offers multiple geometry/PBR engines, and compares saved results under shared lighting. See [validated scope](roadmap.md).

[Models](https://huggingface.co/t8star/Hi3d-Comfy) · [Canvas workflow](examples/Hi3D-Refine.json) · [Validation](VALIDATION.md) · [Upstream Hi3D](https://github.com/hi3d-official/Hi3Dv1.0)

![Hi3D canvas workflow in ComfyUI](docs/images/comfyui-canvas.png)

## Downloads and automatic updates

**GitHub Releases contain code only: source, ComfyUI nodes, program updates and the small Windows launcher. They include no model weights, Python interpreter or runtime. T8star will provide the complete package with Python and models separately and announce its download location.** Its `bundle-manifest.json` identifies the included components.

[GitHub Releases](https://github.com/T8mars/Comfyui-Hi3d-T8/releases/latest) · [Update instructions](docs/UPDATES.md)

From v0.4.1, standalone bundles check and download code updates while idle, verify them, and install on the next launch. Models, Python, configuration and history are preserved. Disable automatic downloads in **Models and environment → Program updates**, or close the app and run `Update-Workbench.cmd`. Old v0.4.0 bundles need one manual code overlay first. A runtime-incompatible release requests a new complete bundle instead of replacing Python dependencies. Git source checkouts update through Git.

### Launch and close the Windows bundle

**Complete model bundles (v0.4.8):** three geometry choices and four texture entries, including separate UltraTex **INT8 ConvRot Base 9B** and FP8 Base 9B options. Full coverage needs 14 extension components plus the Hi3D core, approximately 79.3 GB of model/config files. A baseline-only package is not a complete multi-engine bundle. **Models → Complete all engines** prepares every registered dependency; the page reports ready/total engine counts. A complete package declares `full_engine_coverage: true` in `bundle-manifest.json`.

Builders must use `--include-models --include-pipeline --all-engines` with `tools/build_release.py`. The INT8 checkpoint is tsolful's pinned community quantization of **Base 9B**, not distilled/KV Klein; its original FLUX non-commercial license remains applicable. This project does not re-quantize it. INT8 is not guaranteed to use less VRAM or run faster than FP8 on every GPU.

**v0.4.8 loading:** TRELLIS.2 loads components when each phase needs them. Choose per-phase release or bounded CPU residency for repeated jobs, independently of GPU offloading. Geometry and texturing share a runtime; **Release resident models and runtime** frees it. Checkpoint precision and sampling seeds are preserved. Preview artifacts are published only when complete, with retry controls. See [validation](validation/v048-loading.json). No new weights or Python packages are required.

Double-click **Hi3D-Workbench.exe** or **Start-Workbench.cmd** and keep its console open to view logs. Stop with **Ctrl+C**, the console window's **X**, the webpage's **Close workbench** button, or **Stop-Workbench.cmd**. Closing only the browser leaves the service running. The EXE owns the server and its inference subprocesses, so closing it also releases its workers. Startup errors remain visible. A second launch reuses the same directory's service; it never stops another bundle sharing the port.

**v0.4.7 launcher fix:** existing bundles need both the code update and the matching launcher ZIP to restore Python logs. Startup displays the model directory, URL and confirmed readiness; slow initialization keeps reporting its status.

The separate `Hi3D-Workbench-launcher-<version>.zip` contains a small Windows launcher and start/stop scripts, **not Python or models**. Apply the matching code update first, then extract the launcher beside `Start-Workbench.cmd`. A missing launcher can also be built by `scripts/Build-Launcher.ps1` using Windows' .NET Framework 4 compiler. Launcher source is included; the executable is unsigned. New complete bundles include it already. Community links are available in the workbench sidebar and footer.

## Progress and acceleration

The workbench console and webpage show the active phase, its elapsed time, real counters/steps, time since meaningful progress, and the explicitly reported CPU/CUDA device. UV packing and file/model loading may use the CPU or disk; low GPU usage during those phases does not mean inference fell back to CPU.

TRELLIS.2 offers **Auto / native FlashAttention / fused SDPA** for packed sparse attention, independently of Hi3D. Auto selects native Flash only after a real CUDA packed-kernel probe; an incompatible/missing wheel reports its reason and uses SDPA. UltraTex retains its trained sparse attention. New UVs prefer CuMesh GPU clustering with CPU atlas packing, verified source-index/normal recovery and a reusable cache; the pinned community GPU algorithm and bounded CPU partitions provide reported fallbacks. A usable existing atlas is retained. Optional **100K / 300K / 1M** face budgets apply only to generated TRELLIS geometry; the raw mesh remains available. Existing-model texturing never simplifies the input.

Code updates preserve existing Python environments. To add the optional Windows cp312 / Torch 2.8 cu128 Flash wheel to an older full bundle, run `runtime/python.exe tools/setup_pipeline_runtime.py --runtime runtime-pipeline --attention-only` from the bundle root. This verifies a fixed SHA256 and installs with no dependencies; it does not upgrade Torch. Other systems retain SDPA. The optional CPU UV packer can be added with the same command and `--uv-only` (Numba 0.61.2 / llvmlite 0.44.0); it keeps the existing NumPy and Torch. First use includes JIT compilation. See [performance evidence and limits](docs/PERFORMANCE.md).

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
| Hunyuan3D-2mini-Turbo / Hunyuan3D 2.1 / TRELLIS.2 | Selectable image-to-geometry engines | Workbench only |
| TRELLIS.2 PBR / Hunyuan3D-Paint 2.1 / UltraTex | Selectable mesh-conditioned PBR engines | Workbench only |
| DINOv3, DINOv2 Giant, FLUX.2 Klein 9B and Qwen3 | Dependencies of the selected extension engine | Workbench only |
| CPU U2Netp | Optional foreground extraction for opaque images | Workbench only |

The [Hugging Face repository](https://huggingface.co/t8star/Hi3d-Comfy) contains the **shared Hi3D core weights and configurations plus every standalone extension component**. There are no ComfyUI-specific Hi3D checkpoints. Extension models live under `models/extensions`, retaining pinned sources, hashes and their individual licenses. UltraTex uses an upstream-published fp8 base, not a new quantization by this project. The Registry/node download contains **code and the canvas workflow only**. Model files alone do not provide an inference environment.

Download into the **workbench root**, not `ComfyUI/models`:

```bash
hf download t8star/Hi3d-Comfy --local-dir models --include "manifest.json" "hi3d/*" "dino/*" "clip/*" "licenses/*"
```

The command above downloads the core only. Omit `--include` to download every geometry/PBR extension (about 79.3GB); see the [model repository](https://huggingface.co/t8star/Hi3d-Comfy) for selective downloads. Layout: `models/manifest.json`, `models/hi3d/`, `models/dino/`, `models/clip/`, with extensions under `models/extensions/`. The manifest records pinned upstream revisions and SHA256 hashes. The backend also supports downloading directly from the original upstream repositories.

**Custom model directory:** choose a root in **Models and environment → Model directory**, such as `E:/AI-Models/Hi3D`, or a path relative to the workbench root. It contains `hi3d/`, `dino/`, `clip/` and optional `extensions/`. Save applies while idle without restarting; **Restore default** returns to `models/`. Files are not moved automatically, and code updates preserve the setting. Existing core files still need **Reverify all core models** before the new location is ready.

<details>
<summary>Actual textured example</summary>

![Actual textured example](docs/images/quality-fox-v04.png)

Standalone workbench example: TRELLIS.2 geometry + UltraTex PBR; no Hi3D refinement in this example.

</details>

## Local workbench

The Windows edition starts with **Start-Workbench.cmd** and opens `http://127.0.0.1:8378`, independently of ComfyUI. The workbench has three entry points: image → initial mesh → optional Hi3D → PBR; reference + existing mesh → PBR; and reference + coarse mesh → Hi3D, optionally followed by PBR. Image mode defaults to **initial mesh + PBR**, with Hi3D optional: a paired cartoon test retained a cleaner mouth without refinement. It retains intermediate models, logs and reports. Geometry and texture engines are selected independently; install optional engines from **Models**. Existing self-contained GLBs retain their materials and usable single-material UVs. Saved jobs can be compared side by side with shared camera, HDR lighting and exposure. See [engine choices and quality evidence](docs/ENGINES.md). Material results export as self-contained **GLB**; **STL/PLY contain geometry only**.

The v0.4.8 full-model offline package includes two isolated runtimes, Hi3D core and all 14 extension components, including Hunyuan 2.1, Hunyuan Paint and both UltraTex FP8/INT8 ConvRot bases. Exact contents are recorded in `bundle-manifest.json`. Older baseline packages can complete their dependencies in Models. Unavailable engines are identified explicitly. Node and source archives contain code; source installations prepare runtimes and weights separately.

Three starting presets balance cost and detail: **Light** uses Mini Turbo 256 + TRELLIS PBR 512 / 2K; **Balanced** uses TRELLIS geometry 512 + PBR 1024 / 4K; **Detail** uses TRELLIS geometry 1024 + UltraTex INT8 ConvRot 1024 / 4K. UltraTex's FLUX.2 Klein 9B base has a **Non-Commercial** license. A larger atlas does not add detail missing from the generated texture. See [engine choices](docs/ENGINES.md) and the [workbench guide](docs/WORKBENCH.zh-CN.md).

For a Windows source installation, use Python 3.10 in a separate environment:

```powershell
git clone https://github.com/T8mars/Comfyui-Hi3d-T8.git
cd Comfyui-Hi3d-T8
powershell -ExecutionPolicy Bypass -File scripts/Setup.ps1 -Python C:/Python310/python.exe -DownloadModels
```

For the standalone image/PBR stages, also prepare the separate Python 3.12 runtime with `C:/Python312/python.exe tools/setup_pipeline_runtime.py`, then run `.venv/Scripts/python.exe -m hi3d_workbench.pipeline_models download` for the baseline models. Additional engines download independently in **Models**; see [engine choices](docs/ENGINES.md).

For Linux Hi3D refinement, run `bash scripts/setup-linux.sh`, then `.venv/bin/python -m hi3d_workbench.model_cli download` and `.venv/bin/python -m hi3d_workbench --open`. Linux setup follows the official Ampere/Hopper stack and has not been tested on this machine. The additional image/PBR runtime is currently validated on Windows. See [detailed workbench guide (Chinese)](docs/WORKBENCH.zh-CN.md).

## Validation and limits

Tested on **Windows / RTX 5090 Laptop 24GB / 64GB RAM**. Current multi-engine inference, portable-runtime checks and remaining acceptance work are in [VALIDATION.md](VALIDATION.md); earlier real ComfyUI 512 and standalone 512/1024/1536 refinement runs remain historical evidence. Automated tests, HTTP inference and browser checks are recorded separately. Lower-VRAM GPUs have not been validated. PyTorch tensor peaks are not total GPU memory requirements.

Windows Hi3D uses an explicitly identified **SDPA compatibility adapter**; workbench TRELLIS.2 can independently select a validated native FlashAttention wheel. These are not the official Linux environment. Hi3D normalizes scale and rebuilds topology; its refinement stage produces geometry only. Results are not guaranteed watertight, printable or more accurate than the input; inspect both versions before export.

The bundled TRELLIS Windows extensions require **CUDA compute capability 8.6+**; this excludes RTX 20-series and A100 (8.0) for that stage. Passing this architecture check is not a VRAM or device-validation guarantee.

## License and attribution

Original integration code: [MIT](LICENSE). The derived community GPU UV adapter retains [GPL-3.0](licenses/ComfyUI-GPL-3.0.txt). UltraTex additionally uses the GPL-3.0 Comfy inference library in the isolated workbench runtime and a **FLUX Non-Commercial** base model; these licenses remain separate. Hi3D weights and upstream code: **Hi3D Open License v1.0**, with their original notices; encoder and runtime licenses remain separate. This is a community integration. See [NOTICE.md](NOTICE.md). [Powered by Hi3D 1.0](https://www.hi3d.ai/).

## T8star

[Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [Free Gallery](https://www.openzhenzhen.com) · [Online AI Apps](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI Distribution](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
