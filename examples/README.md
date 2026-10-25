# Hi3D native canvas workflow

[English](../README.md#comfyui) · [中文](../README.zh-CN.md#comfyui)

1. Install the node dependencies with ComfyUI's own Python and prepare the shared Hi3D/DINOv2/CLIP core under `ComfyUI/models/Hi3D`, or select a custom root in the loader.
2. Drag `Hi3D-Refine.json` into the ComfyUI canvas. No workbench service is required.
3. Upload the reference in Load Image. Click **选择文件上传粗模** in Hi3D Load Mesh, or drop a GLB / PLY / STL onto it; the uploaded file is selected automatically.
4. Run 512 / 25 steps / CPU offload / SDPA first. Native SaveGLB saves refined geometry to ComfyUI's output directory.

This is a canvas JSON with nodes, positions and links, not an API-format prompt. Placeholder image/mesh filenames must be replaced by uploads. Legacy bridge workflows must be replaced; Hi3D Backend is no longer used. These nodes perform Hi3D refinement only, without additional geometry/PBR models.
