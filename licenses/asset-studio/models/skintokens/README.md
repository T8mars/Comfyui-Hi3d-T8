---
license: mit
language:
  - en
library_name: pytorch
tags:
  - rigging
  - skinning
  - skeleton
  - autoregressive
  - fsq
  - vae
  - 3d
  - animation
  - VAST
  - Tripo
---

# SkinTokens

Pretrained checkpoints for **SkinTokens: A Learned Compact Representation for Unified Autoregressive Rigging**.

[![Project Page](https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white)](https://zjp-shadow.github.io/works/SkinTokens/)
[![arXiv](https://img.shields.io/badge/arXiv-2602.04805-b31b1b.svg)](https://arxiv.org/abs/2602.04805)
[![GitHub](https://img.shields.io/badge/GitHub-Code-black?logo=github)](https://github.com/VAST-AI-Research/SkinTokens)
[![Tripo](https://img.shields.io/badge/Tripo-3D_Studio-ff7a00)](https://www.tripo3d.ai)

This repository stores the model checkpoints used by the [SkinTokens codebase](https://github.com/VAST-AI-Research/SkinTokens), including:

- the **FSQ-CVAE** that learns the *SkinTokens* discrete representation of skinning weights, and
- the **TokenRig** autoregressive Transformer (Qwen3-0.6B architecture, GRPO-refined) that jointly generates skeletons and SkinTokens from a 3D mesh.

SkinTokens is the successor to [UniRig](https://github.com/VAST-AI-Research/UniRig) (SIGGRAPH '25). While UniRig treats skeleton and skinning as decoupled stages, SkinTokens unifies both into a single autoregressive sequence via learned discrete skin tokens, yielding **98%–133%** improvement in skinning accuracy and **17%–22%** improvement in bone prediction over state-of-the-art baselines.

## What Is Included

The repository is organized exactly like the `experiments/` folder expected by the main SkinTokens codebase:

```text
experiments/
├── articulation_xl_quantization_256_token_4/
│   └── grpo_1400.ckpt        # TokenRig autoregressive rigging model (GRPO-refined)
└── skin_vae_2_10_32768/
    └── last.ckpt             # FSQ-CVAE for SkinTokens (skin-weight tokenizer)
```

Approximate total size: about **1.6 GB**.

> The training data (`ArticulationXL` splits and processed meshes) used to train these checkpoints will be released separately in a future update.

## Checkpoint Overview

### SkinTokens — FSQ-CVAE (skin-weight tokenizer)

**File:** `experiments/skin_vae_2_10_32768/last.ckpt`

Compresses sparse skinning weights into discrete *SkinTokens* using a Finite Scalar Quantized Conditional VAE with codebook levels `[8, 8, 8, 5, 5, 5]` (64,000 entries). Used both to tokenize ground-truth weights during training and to decode TokenRig's output tokens back into per-vertex skinning at inference.

### TokenRig — autoregressive rigging model

**File:** `experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt`

Qwen3-0.6B-based Transformer trained on a composite of **ArticulationXL 2.0 (70%)**, **VRoid Hub (20%)**, and **ModelsResource (10%)**, with quantization 256 and 4 skin tokens per bone, then refined with GRPO for 1,400 steps. **This is the recommended checkpoint** — it generates the skeleton and the SkinTokens in a single unified sequence.

> Both checkpoints are required for end-to-end inference: TokenRig generates the rig as a token sequence, and the FSQ-CVAE decoder turns SkinTokens back into dense per-vertex skinning weights.

## How To Use

The easiest way is to use the helper script in the main SkinTokens codebase, which downloads both checkpoints and the required Qwen3-0.6B config into the expected layout:

```bash
git clone https://github.com/VAST-AI-Research/SkinTokens.git
cd SkinTokens
python download.py --model
```

### Option 1 — Download with `hf` CLI

```bash
hf download VAST-AI/SkinTokens \
  --repo-type model \
  --local-dir .
```

### Option 2 — Download with `huggingface_hub` (Python)

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="VAST-AI/SkinTokens",
    repo_type="model",
    local_dir=".",
    local_dir_use_symlinks=False,
)
```

### Option 3 — Download individual files

```python
from huggingface_hub import hf_hub_download

tokenrig_ckpt = hf_hub_download(
    repo_id="VAST-AI/SkinTokens",
    filename="experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
)
skin_vae_ckpt = hf_hub_download(
    repo_id="VAST-AI/SkinTokens",
    filename="experiments/skin_vae_2_10_32768/last.ckpt",
)
```

### Option 4 — Web UI

Browse the [Files and versions](https://huggingface.co/VAST-AI/SkinTokens/tree/main) tab and download the folders manually, keeping the `experiments/...` layout intact.

After download, you should have:

```text
experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt
experiments/skin_vae_2_10_32768/last.ckpt
```

## Run TokenRig With These Weights

Once the `experiments/` folder is in place (and the environment is installed per the [GitHub README](https://github.com/VAST-AI-Research/SkinTokens#installation)), you can run:

```bash
python demo.py --input examples/giraffe.glb --output results/giraffe.glb --use_transfer
```

Or launch the Gradio demo:

```bash
python demo.py
```

Then open `http://127.0.0.1:1024` in your browser.

## Notes

- **Keep the directory names unchanged.** The SkinTokens code expects the exact `experiments/.../*.ckpt` layout shown above.
- **TokenRig requires both checkpoints.** `grpo_1400.ckpt` generates discrete tokens; the SkinTokens FSQ-CVAE (`last.ckpt`) is needed to decode them into per-vertex skinning weights.
- **Qwen3-0.6B architecture.** TokenRig adopts the Qwen3-0.6B architecture (GQA + RoPE) for its autoregressive backbone; the [Qwen3 config](https://huggingface.co/Qwen/Qwen3-0.6B) is fetched automatically by `download.py`.
- **Hardware.** An NVIDIA GPU with at least **14 GB** of memory is required for inference.
- **Training data.** The checkpoints were trained on a composite of ArticulationXL 2.0 (70%), VRoid Hub (20%), and ModelsResource (10%); the processed data splits will be released as a separate dataset repository later.

## Related Links

- Your 3D AI workspace — **Tripo**: <https://www.tripo3d.ai>
- Project page: <https://zjp-shadow.github.io/works/SkinTokens/>
- Paper (arXiv): <https://arxiv.org/abs/2602.04805>
- Main code repository: <https://github.com/VAST-AI-Research/SkinTokens>
- Predecessor: [UniRig (SIGGRAPH '25)](https://github.com/VAST-AI-Research/UniRig)
- More from VAST-AI Research: <https://huggingface.co/VAST-AI>

## Acknowledgements

- [UniRig](https://github.com/VAST-AI-Research/UniRig) — the predecessor to this work.
- [Qwen3](https://github.com/QwenLM/Qwen3) — the LLM architecture used by the TokenRig autoregressive backbone.
- [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet), [Michelangelo](https://github.com/NeuralCarver/Michelangelo) — the shape encoder backbone used by the FSQ-CVAE.
- [FSQ](https://arxiv.org/abs/2309.15505) — Finite Scalar Quantization, the discretization scheme behind SkinTokens.
- [GRPO](https://arxiv.org/abs/2402.03300) — the policy-optimization method used for RL refinement.

## Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@article{zhang2026skintokens,
  title   = {SkinTokens: A Learned Compact Representation for Unified Autoregressive Rigging},
  author  = {Zhang, Jia-Peng and Pu, Cheng-Feng and Guo, Meng-Hao and Cao, Yan-Pei and Hu, Shi-Min},
  journal = {arXiv preprint arXiv:2602.04805},
  year    = {2026}
}
```
