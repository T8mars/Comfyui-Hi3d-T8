# Hi3D Workbench third-party notices

Workbench integration and design: By T8star. Upstream authorship and licenses remain as stated below.

This is a community integration, not an official Hi3D product. The ComfyUI nodes and workbench call the unmodified Hi3D v1.0 inference implementation at commit 23dcc684bc6e1fab86ecafb99ec5270edad122f9. The node-only archive contains only integration code; Hi3D sources, weights and their accompanying licenses are supplied by the separate workbench distribution.

Hi3D is licensed under the Hi3D Open License Agreement Version 1.0. Copyright © 2026 OCEANO HK HOLDING LIMITED. All rights not expressly granted are reserved. The Hi3D trademarks are owned by OCEANO HK HOLDING LIMITED or its applicable affiliate(s).

The accompanying upstream LICENSE, NOTICE, AUTHORS, THIRD_PARTY_LICENSES.md and sparc3d/ops/flexicubes/LICENSE.txt are retained. They govern their respective components. No license is asserted for the entire distribution as a single MIT work. Upstream assets/example.png and comparison figures are excluded from release archives. Procedural smoke-test samples are created by this project.

The optional SDPA compatibility adapter is implemented by this integration in hi3d_workbench/attention.py. It is not part of the official measured inference stack. It replaces only packed attention calls inside the isolated process; no model weights are changed. It requires independent end-to-end quality evaluation.

The local viewer bundles @google/model-viewer 4.0.0, copyright Google LLC, under Apache-2.0; its full LICENSE is in hi3d_workbench/web/vendor/model-viewer.LICENSE. Source: https://github.com/google/model-viewer .

Required image encoders: facebook/dinov2-large (Apache-2.0), openai/clip-vit-large-patch14 (MIT). Their licenses are retained in licenses/ and pinned file hashes in models/manifest.json. These encoders are required by Hi3D and are not additional 3D generation models.

Windows runtime distributions include CPython, PyTorch, Triton Windows, spconv/cumm, Transformers, Accelerate, NumPy, SciPy, trimesh and other Python dependencies. Their package license files remain in the runtime. Exact installed versions are recorded in runtime-packages.txt. Python's LICENSE is retained in the runtime directory.

Powered by Hi3D 1.0 — http://hi3d.ai/
