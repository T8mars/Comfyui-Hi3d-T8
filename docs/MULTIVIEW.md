# Multi-view modeling

**English** | [简体中文](MULTIVIEW.zh-CN.md)

Available in the standalone workbench and complete bundle. ComfyUI nodes continue to run Hi3D geometry refinement only.

1. Select **Hunyuan3D-2mv** under image modeling and upload the front image in the main image slot.
2. Upload the same object's left, back or right views separately. Front + left + back is a three-view input; other subsets are supported, with the front as the main reference.
3. Keep the shape, scale and direction consistent across views. Each image receives its own foreground processing; transparent PNGs are supported.
4. Defaults are geometry resolution 384 and 30 steps. Memory-saving mode stages the models. Optional Hi3D refinement and PBR follow; **the existing refinement and painting stages use the front reference**.
5. History retains every view for preview, original-image download and input reuse. Missing originals are reported and incomplete reuse is rejected.

Prepare this engine in Models & Environment when first selecting it. Its additional FP16 checkpoint is about **4.93 GB**, under `models/extensions/coarse_hunyuan_mv/` by default. Custom model roots also apply. The checkpoint includes its geometry, VAE and image-conditioning components; single-image 2.1 or mini weights cannot replace it.

Automatic UV processing preserves coordinates, normals and valid oriented triangles. A rejected native atlas may trigger one explicitly reported, independent GPU atlas attempt with the same degeneracy, duplicate and local-repair limits. A second failure stops with geometry retained, without silently restarting the whole mesh on CPU. Zero-area triangles are counted separately.

Additional views constrain geometry but do not guarantee hidden-surface accuracy or constitute calibrated photogrammetry. Inspect the sides, back and final materials.

Sources: [official model](https://huggingface.co/tencent/Hunyuan3D-2mv), [official three-view example](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/examples/shape_gen_multiview.py). The pinned revision and file list are in `hi3d_workbench/pipeline_models.py`; retain the upstream Tencent Hunyuan 3D 2.0 Community License when distributing weights.
