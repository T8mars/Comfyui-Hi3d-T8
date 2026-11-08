# Idol image to splats and meshes

Idol is integrated into the standalone v0.6 workbench with a private runtime. Real generation, processing, downloads, cancellation and recovery have passed, along with native Gaussian browser interaction, mesh/PBR export and complete playback of one four-second character clip. See the linked receipts for the tested scope and quality limits. The ComfyUI-Hi3D custom nodes retain their Hi3D refinement scope.

Select **Idol** in image modeling and upload one complete subject. Existing alpha is retained; automatic foreground extraction uses BiRefNet. Generated/rendered images use the official 1.2 crop margin; the photo option uses 1.7. The actual conditioning image and predicted input camera are saved.

The default follows the native Euler 50-step time schedule and guidance formula using the pinned Comfy backend for encoding and random numbers. It is not claimed to reproduce the SDK bit for bit. The official Comfy template's DPM++ 2M/20-step option is also available. Guidance values have different meanings between these options. In the current character comparison, Euler preserves the body and arms substantially better; failed comparisons are retained.

The default output has 262,144 Gaussians. Higher counts increase density and resource use, without guaranteeing additional detail. Splat-only mode skips mesh extraction, Hi3D refinement and PBR. Download the original PLY, SPZ or KSPLAT and use the independent Gaussian viewer to rotate, zoom and pan. Optional orbit video contains 75 frames at 25 fps and 512×512; it shows a static object rather than character animation.

Optional mesh extraction produces an unlit vertex-color GLB. The newer official surface extraction module is isolated from the existing Comfy core; the older density surface remains available for comparison. Interior processing changes the surface and may close intentional openings. Original splats are preserved even if conversion fails. Mesh yaw can be aligned to the predicted input camera, with the applied matrix recorded; original splats retain their coordinates.

PBR is a separate selected texturing stage that generates UVs and material maps. Vertex-color GLB is not presented as generated metallic/roughness PBR. Valid meshes can continue through simplification, segmentation, rigging and single-person motion in Asset Studio; each operation's quality remains subject to actual inspection.

Five model files require approximately 3.78 GB, excluding Python, dependencies, caches and outputs. An existing VAE with the same SHA256 is reused with a hard link when available. DINOv3 H+ is distinct from the existing TRELLIS.2 DINO-L encoder.

```powershell
.\.pipelineenv\Scripts\python.exe -m hi3d_workbench.pipeline_models download --selection engine:coarse:idol
.\.pipelineenv\Scripts\python.exe tools/setup_idol_runtime.py --validate-image samples/sculpted-vase.png
```

Dependencies live in a private `engines/idol` overlay; the additional upstream meshing source lives in `sources/idol_mesh`. The base Torch, Diffusers and UltraTex Comfy core are preserved. Inference loads local files without network access and uses the existing controlled task lifecycle. A readiness receipt requires real successful generation and a matching output hash.

These commands prepare a development checkout. Complete bundles ship the verified private runtime. `--validate-image` performs an actual mesh generation, records readiness and keeps its output in `data/idol-runtime-validation`; installation validation does not imply visual-quality acceptance.

Idol-only installation uses BiRefNet without requiring U2Net. PBR retains its existing foreground/model dependencies, checked for the selected workflow; those weights are additional.

Real checks on the RTX 5090 Laptop 24GB include:

| Scope | Observed result |
| --- | --- |
| Default 262,144 splats, 384 mesh, 2K PBR and orbit video | 158.97 seconds; front and back renders inspected |
| 1,048,576 splats, BiRefNet, photo framing, splats only | HTTP success in 87.163 seconds; native downloads and hashes, history and input/settings reuse passed; the input was a rendered vase |
| Idol → Hi3D → 2K PBR | HTTP success in 274.09 seconds; all 333,795 refined oriented triangles preserved through texturing, no zero-area/duplicate UV triangles, two 2K images, 180 mm export read back as 0.18 m |
| Native mesh simplification and segmentation | 223,848→49,998 triangles; a separate six-part HTTP result preserved all 223,852 triangles, unlit material and ZIP integrity |
| Binding and single-character punch | 52 joints, preserved geometry/materials and FBX roundtrip; a four-second, 121-frame motion passed saved-frame readback and complete IAB/Chrome playback, pause and timeline seeking |
| Cancellation, deliberate process interruption/restart, empty foreground | Original splats/inputs retained, downloads verified, invalid results withheld and workers reaped |
| Renamed Unicode model/runtime paths, fresh caches, host paths removed, Python socket connections blocked | 32,768 splats, a 99,652-triangle mesh and 75-frame video generated in 87.626 seconds with zero connection attempts; original paths restored |

These scopes differ and do not form an equal-settings speed comparison. The default mesh still has 77 nonmanifold edges. Hi3D can alter the face and clothing; compare its result with the retained original colored mesh, and keep refinement optional. A few punch triangles reach the deformation diagnostic threshold; natural motion and collision quality are not broadly accepted.

The release suite passed 935 tests; [publication checks](../validation/v060-publication-checks.json) also record isolated updater validation. Browser checks covered the million-splat viewer, rotation/zoom/reset, settings reuse, all three native downloads with matching hashes, mesh/PBR stage switching and a downloaded 180 mm GLB. The missing upstream Three.js r180 HDR loader was added with an actual module-import and HDR-decoding regression. Chrome automation still requires its file-URL permission for uploads, and its download-event capture timed out; uploads and downloads were verified through IAB. See the [browser receipt](../validation/idol-browser-2026-11-08.json) and [integration receipt](../validation/idol-integration-2026-11-08.json) for the tested scope and remaining acceptance.

The offline check moved the model and private runtime folders, preserving the user's original log console; it did not repeat moving the entire bundle root. Earlier complete-bundle relocation checks remain separately documented.

Idol, DINOv3, the FLUX.2 autoencoder, BiRefNet, TripoSplat and Comfy code retain their individual licenses. See the [original notices and hashes](../licenses/idol/manifest.json). Source releases exclude model weights and Python runtimes. See the [research report](IDOL_RESEARCH.zh-CN.md) and [roadmap](../roadmap.md) for evidence and acceptance status.
