---
license: apache-2.0
tags:
- comfyui
- diffusion-single-file
base_model:
- speridlabs/idol
pipeline_tag: image-to-3d
---

# Idol

Repackaged model files for ComfyUI.

Original model repository: https://huggingface.co/speridlabs/idol

This model is a fine tuned version of TripoSplat and works with the existing TripoSplat workflows.

---

Place the files in the following folders:

```
📂 ComfyUI/
├── 📂 models/
│   ├── 📂 diffusion_models/
│   │   └── idol_fp16.safetensors
│   ├── 📂 clip_vision/
│   │   └── dino_v3_vit_h_bf16.safetensors
│   ├── 📂 vae/
│   │   ├── flux2-vae.safetensors
│   │   └── idol_decoder_fp16.safetensors
```

## Workflows

<table>
<thead>
<tr><th align="center" valign="middle" style="text-align:center;vertical-align:middle">Workflow</th><th colspan="2" align="center" valign="middle" style="text-align:center;vertical-align:middle">Thumb</th></tr>
</thead>
<tbody>
<tr><td align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/3d_idol_image_to_gaussian_splat.json">Idol: Image to Gaussian Splat</a></td><td colspan="2" align="center" valign="middle" style="text-align:center;vertical-align:middle"><a href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/3d_idol_image_to_gaussian_splat.json"><img src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/3d_idol_image_to_gaussian_splat-1.webp" width="200" height="200" alt="Idol: Image to Gaussian Splat"></a></td></tr>
</tbody>
</table>
