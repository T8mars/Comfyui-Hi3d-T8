# Hi3D Workbench third-party notices

Workbench integration and design: By T8star. Upstream authorship and licenses remain as stated below.

This is a community integration, not an official Hi3D product. The ComfyUI nodes and workbench call the unmodified Hi3D v1.0 inference implementation at commit 23dcc684bc6e1fab86ecafb99ec5270edad122f9. The node-only archive contains only integration code; Hi3D sources, weights and their accompanying licenses are supplied by the separate workbench distribution.

Hi3D is licensed under the Hi3D Open License Agreement Version 1.0. Copyright © 2026 OCEANO HK HOLDING LIMITED. All rights not expressly granted are reserved. The Hi3D trademarks are owned by OCEANO HK HOLDING LIMITED or its applicable affiliate(s).

The accompanying upstream LICENSE, NOTICE, AUTHORS, THIRD_PARTY_LICENSES.md and sparc3d/ops/flexicubes/LICENSE.txt are retained. They govern their respective components. No license is asserted for the entire distribution as a single MIT work. Upstream assets/example.png and comparison figures are excluded from release archives. Procedural smoke-test samples are created by this project.

The optional SDPA compatibility adapter is implemented by this integration in hi3d_workbench/attention.py. It is not part of the official measured inference stack. It replaces only packed attention calls inside the isolated process; no model weights are changed. It requires independent end-to-end quality evaluation.

The local viewer bundles @google/model-viewer 4.0.0, copyright Google LLC, under Apache-2.0; its full LICENSE is in hi3d_workbench/web/vendor/model-viewer.LICENSE. Source: https://github.com/google/model-viewer .

Required image encoders: facebook/dinov2-large (Apache-2.0), openai/clip-vit-large-patch14 (MIT). Their licenses are retained in licenses/ and pinned file hashes in models/manifest.json. These encoders are required by Hi3D and are not additional 3D generation models.

The standalone workbench asset pipeline adds Hunyuan3D-2mini-Turbo (Tencent Hunyuan 3D 2.0 Community License), TRELLIS.2 mesh texturing (MIT), DINOv3 (DINOv3 License), and CPU U2Netp foreground extraction (rembg MIT / U-2-Net Apache-2.0). These are not added to the ComfyUI nodes or canvas workflow. Each downloaded component retains its license and fixed source revision in models/extensions/manifest.json. DINOv3's public redistribution source is recorded separately from its original Meta model; redistribution does not change its license.

The v0.4 standalone workbench also offers TRELLIS.2 geometry (MIT), Hunyuan3D 2.1 geometry and Hunyuan3D-Paint 2.1 (Tencent Hunyuan 3D 2.1 Community License), DINOv2 Giant (Apache-2.0), and Real-ESRGAN (BSD-3-Clause). The Hunyuan Paint Windows adapter retains the official cameras and baking with nvdiffrast rasterization and OpenCV UV inpainting instead of the optional upstream C++ vertex inpainting. This integration is provided by T8star independently; Tencent does not sponsor or endorse it. Original sources and notices remain in the isolated runtime.

UltraTex by Yibo Zhang et al. (MIT) is integrated through the pinned visualbruno/ComfyUI-UltraTex adapter (MIT) and the ComfyUI inference library (GPL-3.0). The pinned corresponding library source and license accompany runtime-pipeline/sources/comfy_core. The FLUX.2 Klein 9B fp8 base retains the FLUX Non-Commercial License. UltraTex's foreground decoder and albedo/MR LoRAs are verified byte-for-byte against the original ModelScope release; the public FLUX mirror is verified against the pinned official SHA256. Qwen3 and FLUX.2 VAE retain their respective Apache-2.0 notices. These dependencies are used only by the standalone runtime; no UltraTex or other model node is registered in the ComfyUI package. The integration's MIT license does not relicense these third-party components or the combined distribution.

Preview lighting uses the unmodified Poly Haven studio_small_08 and studio_small_09 HDR environments under CC0. Source URLs and content hashes are retained in hi3d_workbench/web/environments/sources.json. These are lighting assets, not generated 3D models.

The additional Windows runtime uses an integration SDPA compatibility adapter for packed sparse attention. It does not replace model weights and is not the official Linux FlashAttention environment. Runtime/source/package versions and component licenses must accompany a full-chain distribution. Real inference and memory validation for the new pipeline are tracked separately in roadmap.md.

Windows runtime distributions include CPython, PyTorch, Triton Windows, spconv/cumm, Transformers, Accelerate, NumPy, SciPy, trimesh and other Python dependencies. Their package license files remain in the runtime. Exact installed versions are recorded in runtime-packages.txt. Python's LICENSE is retained in the runtime directory.

Powered by Hi3D 1.0 — http://hi3d.ai/
