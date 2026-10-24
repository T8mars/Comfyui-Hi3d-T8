# Workbench engines and quality

**English** | [简体中文](ENGINES.zh-CN.md)

The ComfyUI package remains Hi3D-only. All choices below belong to the separate local workbench. Geometry and texturing are independent stages; a texture model cannot fix an incorrect silhouette, missing part or distorted face.

| Stage | Engine | Sampling choices | Purpose |
|---|---|---|---|
| Geometry | Hunyuan3D-2mini Turbo | 256 / 384 / 512, five steps | Lower-cost first draft |
| Geometry | Hunyuan3D 2.1 | 256 / 384 / 512 | Optional full geometry model |
| Geometry | TRELLIS.2 | 512 / 1024 cascade | Detailed image-conditioned geometry; optional surface reconstruction |
| Texture | TRELLIS.2 PBR | 512 / 1024 | Mesh-conditioned color, metallic and roughness |
| Texture | Hunyuan3D-Paint 2.1 | 512 / 1024 | Six-view PBR generation and baking |
| Texture | UltraTex, FLUX.2 Klein 9B | 1024 / 2048 albedo; MR up to 1024 | Six-view color and material generation; FLUX Non-Commercial base |

Output atlases can be 1024, 2048 or 4096. Atlas size and generation resolution control different stages: a 4K atlas does not turn a low-resolution prediction into 4K detail. More steps or larger models are not an automatic quality improvement.

Three starting presets are available: light uses Mini Turbo 256 + TRELLIS PBR 512 / 2K; balanced uses TRELLIS geometry 512 + PBR 1024 / 4K; detail uses TRELLIS geometry 1024 + UltraTex 1024 / 4K. UltraTex 2048 albedo is an optional higher-cost setting, with MR capped at 1024 and a Non-Commercial base license. Hunyuan 2.1 geometry and Paint remain independent alternatives. These are starting settings, not a universal quality ranking; select a painter by comparing the same geometry under the same lighting.

Open **Models** to download an engine's pinned dependencies. Missing optional models do not disable installed engines. The v0.4 Windows package includes Hi3D core, Mini Turbo, TRELLIS.2 geometry/PBR, DINOv3 and CPU U2Netp. Hunyuan 2.1 geometry/Paint and UltraTex weights are optional downloads; `bundle-manifest.json` records the exact included components. Weights remain unmodified. UltraTex's fp8 base is the original upstream fp8 release, not a quantization made by this project.

Use **Existing model texturing** with exactly the same GLB and reference image to compare painters. A usable single-material UV layout is preserved; ambiguous multi-material layouts are rebuilt. The texture report records the input mesh hash. History comparisons identify when only the image matches and the geometry differs. Inspect front, back, seams and at least two lighting environments before selecting a result. Lighting and exposure controls only change the preview.

Version 0.4.0 fixes two ways of losing shading normals. Native UV compaction/reordering now recovers original indices using exact coordinates and oriented incident faces, including coincident hard edges. Unprovable mappings fall back to original-index CPU Atlas; fully disconnected triangles take that route immediately to avoid a native empty-adjacency kernel. GLB scene imports also retain authored `NORMAL` data, transform normals by the inverse transpose and reverse face winding for mirrored instances. The TRELLIS painting copy retains that data through export. CPU regressions cover these cases; final inference reruns after the importer correction are tracked separately in [validation](../VALIDATION.md).

Painters need one non-overlapping unit atlas. Degenerate, tiled, repeated-triangle UVs and obvious overlaps exceeding the atlas area are rebuilt; arbitrary partial overlap is not exhaustively detected. Previously affected GLBs need repainting from their source geometry when incorrect normals influenced texture conditioning. `tools/restore_shading_normals.py` can correct an export when its source geometry is available and verifies unchanged positions, faces, UVs and material pixels; this is an export correction, not a new texture inference.

TRELLIS geometry can retain `generated.glb`, the raw generated surface, alongside the reconstructed `coarse.glb`. Reconstruction follows the official opaque-surface narrow-band exporter and targets approximately one million faces. It can alter thin/open structures; compare the retained versions. Hi3D refinement is independently optional and rebuilds topology.

Current measurements and remaining acceptance work are in [the quality record](../validation/quality-upgrade-progress.json). Real comparisons use original fox, mechanical owl, explorer chest and observatory references, fixed seeds and recorded output hashes. Hunyuan 2.1's first fox result contained a connected floor plane, while its owl result did not; it has not been selected as a universal upgrade. TRELLIS 1024 improved fox facial/armor detail, while rear fur and tail structure still showed artifacts. These are limited examples, not a general benchmark ranking.

Measured hardware: Windows, RTX 5090 Laptop 24GB, 64GB RAM. The 1024 fox geometry run took 378.718 seconds and reached 13034.965MiB whole-device NVML usage, including driver/desktop allocations. A one-million-face owl with 1024 TRELLIS PBR took 212.728 seconds; its PyTorch peaks were 11021.04MiB allocated and 17892MiB reserved. Neither measurement establishes minimum VRAM. Large CPU-offloaded models also need system RAM.

Original sources: [TRELLIS.2](https://github.com/microsoft/TRELLIS.2), [Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1), [UltraTex](https://github.com/yiboz2001/UltraTex), [Windows UltraTex adapter](https://github.com/visualbruno/ComfyUI-UltraTex). Hunyuan Paint uses official camera/conditioning/baking code with a Windows nvdiffrast raster adapter and OpenCV UV inpainting; its optional upstream C++ vertex inpainting is not used. UltraTex uses a pinned Comfy inference library without launching a Comfy server or adding nodes. See [notices and licenses](../NOTICE.md).
