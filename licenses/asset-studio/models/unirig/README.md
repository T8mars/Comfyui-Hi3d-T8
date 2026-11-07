---
license: mit
tags:
- 3d
- unirig
- auto-rigging
- computer-graphics
- transformer
- tripo
datasets:
- Seed3D/Articulation-XL2.0
language:
- en
---
# UniRig: One Model to Rig Them All

**Paper:** [One Model to Rig Them All: Diverse Skeleton Rigging with UniRig](https://zjp-shadow.github.io/works/UniRig/static/supp/UniRig.pdf)  
**Code:** [UniRig](https://github.com/VAST-AI-Research/UniRig)  
**Project Page:**  https://zjp-shadow.github.io/works/UniRig/

🚨 **Note:** This model card currently contains **only the Skeleton&Skinning Prediction component** of the UniRig framework, trained specifically on the [**Articulation-XL2.0 dataset**](https://huggingface.co/datasets/Seed3D/Articulation-XL2.0). The skinning weight prediction model and models trained on the Rig-XL/VRoid datasets described in the paper will be released separately at a later date.

## Model Description

UniRig is a unified framework for automatic skeletal rigging of 3D models, developed by Tsinghua University and by [Tripo](https://www.tripo3d.ai) ([VAST AI Research](https://github.com/orgs/VAST-AI-Research)). It addresses the significant bottleneck of rigging in 3D animation pipelines by providing a powerful model capable of generating high-quality skeleton hierarchies and skinning weights for a diverse range of input meshes, including humans, animals, fictional characters, and even inorganic structures.

This release provides the **autoregressive skeleton prediction model** from the UniRig framework. Its purpose is to automatically generate a topologically valid skeleton hierarchy for a given 3D input mesh.

The model leverages:
1.  **A Shape Encoder:** Processes the input mesh (as a point cloud) to capture geometric features.
2.  **An OPT-based Transformer:** Autoregressively predicts a sequence of tokens representing the skeleton structure.
3.  **Skeleton Tree Tokenization:** A novel method (as described in the UniRig paper) to efficiently encode the skeleton's hierarchical structure and joint coordinates into a sequence suitable for the transformer.

This model serves as the first stage of the full UniRig pipeline. The predicted skeleton can be used as input for the forthcoming skinning weight prediction model or other downstream rigging tasks.

## Release Plan & What's Included

This Hugging Face model release includes:
*   ✅ **Skeleton Prediction Model Checkpoint:** Trained on the Articulation-XL2.0 dataset.
*   ✅ **Coming Soon:** Skinning Weight Prediction Model.
*   ✅ **Coming Soon:** Rig-XL and VRoid datasets.
*   ⏳ **Coming Soon:** Model checkpoints trained on Rig-XL and VRoid datasets (representing the full results reported in the paper).

Follow [VAST AI Research](https://github.com/orgs/VAST-AI-Research) updates for future releases.

## Requirements

- CUDA-capable GPU (>8GB VRAM)

## Usage

For detailed usage instructions, please visit our [GitHub repository](https://github.com/VAST-AI-Research/UniRig).