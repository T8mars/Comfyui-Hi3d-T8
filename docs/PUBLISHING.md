# Publishing

Repository: https://github.com/T8mars/Comfyui-Hi3d-T8

Comfy Registry publisher: `t8star`; immutable node ID: `hi3d-t8`.

Follow the [official publishing guide](https://docs.comfy.org/registry/publishing) and [metadata specification](https://docs.comfy.org/registry/specifications). The repository's `REGISTRY_ACCESS_TOKEN` Actions secret authenticates publishing; never commit it. `.comfyignore` keeps workbench code, development evidence and model files out of the Registry archive. Node dependencies do not install torch/CUDA; the backend is installed separately with the `workbench` extra.

Before release, verify the canvas example with ComfyUI, run the test suite and run `comfy node pack`. Inspect the archive for the node entry point, `comfy_nodes.py`, metadata and `examples/Hi3D-Refine.json`. The example must contain a `nodes` array and links, not an API class_type mapping. Workflows exported by the frontend are saved using its canvas serializer.

Publishing a GitHub release triggers `.github/workflows/publish-registry.yml`, which uses the official `Comfy-Org/publish-node-action`. The workflow can also be dispatched manually. A successful Action upload can still require Registry security review; verify the public version status and install endpoint before claiming that Manager can install it. Published versions are immutable: increment the version for code updates.

The [Hugging Face model repository](https://huggingface.co/t8star/Hi3d-Comfy) contains the original pinned models from the offline workbench. `tools/publish_models.py` verifies all model SHA256 hashes, uploads through the official Hub client and checks remote metadata/downloads after commit. Credentials are read from the existing Hugging Face credential cache and never written into repository files.
