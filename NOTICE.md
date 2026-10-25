# Hi3D Workbench third-party notices

Workbench integration and design: By T8star. Upstream authorship and licenses remain as stated below.

This is a community integration, not an official Hi3D product. The ComfyUI nodes and workbench call the unmodified Hi3D v1.0 inference implementation at commit 23dcc684bc6e1fab86ecafb99ec5270edad122f9. The standalone ComfyUI node archive includes the pinned official Python source and upstream notices under hi3d_native/vendor, plus the integration and mesh-upload extension. It runs directly in ComfyUI, without a workbench service. Model weights and Python runtimes are not included.

Hi3D is licensed under the Hi3D Open License Agreement Version 1.0. Copyright © 2026 OCEANO HK HOLDING LIMITED. All rights not expressly granted are reserved. The Hi3D trademarks are owned by OCEANO HK HOLDING LIMITED or its applicable affiliate(s).

The accompanying upstream LICENSE, NOTICE, AUTHORS, THIRD_PARTY_LICENSES.md and sparc3d/ops/flexicubes/LICENSE.txt are retained. They govern their respective components. No license is asserted for the entire distribution as a single MIT work. Upstream assets/example.png and comparison figures are excluded from release archives. Procedural smoke-test samples are created by this project.

The optional SDPA compatibility adapter is implemented by this integration in hi3d_workbench/attention.py and hi3d_native/attention.py. It is not part of the official measured inference stack. It adapts only packed attention calls; the native ComfyUI adapter scopes and restores its Hi3D-only hooks after execution; no model weights are changed. It requires independent end-to-end quality evaluation.

The local viewer bundles @google/model-viewer 4.0.0, copyright Google LLC, under Apache-2.0; its full LICENSE is in hi3d_workbench/web/vendor/model-viewer.LICENSE. Source: https://github.com/google/model-viewer .

Required image encoders: facebook/dinov2-large (Apache-2.0), openai/clip-vit-large-patch14 (MIT). Their licenses are retained in licenses/ and pinned file hashes in models/manifest.json. These encoders are required by Hi3D and are not additional 3D generation models.

The standalone workbench asset pipeline adds Hunyuan3D-2mini-Turbo (Tencent Hunyuan 3D 2.0 Community License), TRELLIS.2 mesh texturing (MIT), DINOv3 (DINOv3 License), and CPU U2Netp foreground extraction (rembg MIT / U-2-Net Apache-2.0). These are not added to the ComfyUI nodes or canvas workflow. Each downloaded component retains its license and fixed source revision in models/extensions/manifest.json. DINOv3's public redistribution source is recorded separately from its original Meta model; redistribution does not change its license.

The v0.4 standalone workbench also offers TRELLIS.2 geometry (MIT), Hunyuan3D 2.1 geometry and Hunyuan3D-Paint 2.1 (Tencent Hunyuan 3D 2.1 Community License), DINOv2 Giant (Apache-2.0), and Real-ESRGAN (BSD-3-Clause). The Hunyuan Paint Windows adapter retains the official cameras and baking with nvdiffrast rasterization and OpenCV UV inpainting instead of the optional upstream C++ vertex inpainting. This integration is provided by T8star independently; Tencent does not sponsor or endorse it. Original sources and notices remain in the isolated runtime.

UltraTex by Yibo Zhang et al. (MIT) is integrated through the pinned visualbruno/ComfyUI-UltraTex adapter (MIT) and the ComfyUI inference library (GPL-3.0). The pinned corresponding library source and license accompany runtime-pipeline/sources/comfy_core. The FLUX.2 Klein 9B fp8 base retains the FLUX Non-Commercial License. UltraTex's foreground decoder and albedo/MR LoRAs are verified byte-for-byte against the original ModelScope release; the public FLUX mirror is verified against the pinned official SHA256. Qwen3 and FLUX.2 VAE retain their respective Apache-2.0 notices. These dependencies are used only by the standalone runtime; no UltraTex or other model node is registered in the ComfyUI package. The integration's MIT license does not relicense these third-party components or the combined distribution.

Preview lighting uses the unmodified Poly Haven studio_small_08 and studio_small_09 HDR environments under CC0. Source URLs and content hashes are retained in hi3d_workbench/web/environments/sources.json. These are lighting assets, not generated 3D models.

The optional UltraTex INT8 ConvRot Base 9B checkpoint is the community quantization published by tsolful at https://civitai.com/models/2738890?modelVersionId=3080042 (file 2959286). It is distributed unchanged with a pinned SHA256 and retains the original FLUX Non-Commercial License, included with the UltraTex adapter component. It is not the distilled or KV checkpoint and was not quantized by this project. The FP8 base remains a separate option.

The additional Windows runtime offers integration fused SDPA and an optional ABI-pinned native FlashAttention adapter for packed sparse attention. It does not replace model weights and is not the official Linux inference environment. Runtime/source/package versions and component licenses must accompany a full-chain distribution. Validation and measured limits are recorded in docs/PERFORMANCE.md.

Windows runtime distributions include CPython, PyTorch, Triton Windows, spconv/cumm, Transformers, Accelerate, NumPy, SciPy, trimesh and other Python dependencies. Their package license files remain in the runtime. Exact installed versions are recorded in runtime-packages.txt. Python's LICENSE is retained in the runtime directory.

Powered by Hi3D 1.0 — http://hi3d.ai/

The original-index GPU UV adapter in hi3d_workbench/engines/community_uv.py is derived from Comfy-Org/ComfyUI nodes_mesh_postprocess._uv_unwrap at 7cebca52227c63b4180ceed448cbd7ad22903b6d and retains GPL-3.0. Its license is included as licenses/ComfyUI-GPL-3.0.txt; the full pinned corresponding source accompanies runtime-pipeline/sources/comfy_core. The MIT scope excludes that derived file. The adapter omits automatic welding to retain source hard-edge normals.

The progress adapter in hi3d_workbench/engines/native_uv.py follows the MIT-licensed JeffreyXiang/CuMesh CuMesh.uv_unwrap flow. It reuses the installed native kernels and per-instance xatlas callbacks; the SDK Python source and its license remain in runtime-pipeline/Lib/site-packages/cumesh and cumesh-1.0.dist-info/licenses/LICENSE.

Optional native FlashAttention 2.8.3 for the cp312 / Torch 2.8 cu128 pipeline uses the pinned kingbri1 Windows community wheel under BSD-3-Clause (licenses/FlashAttention-LICENSE.txt). Its size and SHA256 are fixed in engines/acceleration_wheels.py. CUDA packed-kernel validation gates automatic selection; unsupported systems retain explicit fused SDPA. Hi3D refinement and UltraTex trained sparse attention are configured separately.

The optional CPU bitmap UV packer uses the original Numba 0.61.2 and llvmlite 0.44.0 Windows wheels (BSD licenses retained in the runtime). Fixed PyPI sizes and SHA256 values are recorded in acceleration_wheels.py. Runtime installation does not upgrade NumPy or Torch; machine-specific JIT caches are excluded from release archives.
