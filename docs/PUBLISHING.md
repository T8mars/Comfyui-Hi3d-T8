# Publishing

Repository: https://github.com/T8mars/Comfyui-Hi3d-T8

Comfy Registry publisher: `t8star`; immutable node ID: `hi3d-t8`.

Follow the [official publishing guide](https://docs.comfy.org/registry/publishing) and [metadata specification](https://docs.comfy.org/registry/specifications). The repository's `REGISTRY_ACCESS_TOKEN` Actions secret authenticates publishing; never commit it. `.comfyignore` keeps workbench code, development evidence and model files out of the Registry archive. The node archive must include `hi3d_native/vendor/` (pinned official source and notices) and `web/hi3d_mesh_upload.js`. Nodes run inside ComfyUI using its existing Torch/CUDA; no workbench backend is required. Keep `/models/` anchored at the archive root in `.comfyignore` so the vendored Python `sparc3d/models/` package is retained.

Before release, verify the canvas example with ComfyUI, run the test suite and run `comfy node pack`. Inspect the archive for the node entry point, `comfy_nodes.py`, metadata and `examples/Hi3D-Refine.json`. The example must contain a `nodes` array and links, not an API class_type mapping. Workflows exported by the frontend are saved using its canvas serializer.

Push a matching version tag (`v` + `common.VERSION` / `pyproject.toml` version). `.github/workflows/release-code.yml` runs updater tests, fetches and verifies pinned upstream source, and publishes the source, node and code-update ZIPs plus checksum manifests. It then publishes the Registry version with the official `Comfy-Org/publish-node-action`. A manually created external GitHub release can also trigger `publish-registry.yml`. Avoid publishing the same version twice. A successful upload can still require Registry review; verify public version status before claiming Manager availability. Versions are immutable; increment them for code updates.

**Never attach weights, Python interpreters or runtime directories to GitHub Releases.** The complete package is distributed separately by T8star. A code update preserves models, runtimes, local configuration and history. Increase `updater.RUNTIME_API` if dependencies change; provide a matching complete package instead of silently installing new Python packages. See [update policy](UPDATES.md).

The [Hugging Face repository](https://huggingface.co/t8star/Hi3d-Comfy) hosts the shared core and all standalone extensions. Publish verified local models with the official Hub client:

```bash
python tools/publish_models.py --include-extensions --preupload
```

The script checks every file against the pinned source manifest, makes resumable per-component commits, then verifies remote sizes and LFS SHA256 / Git blob IDs. `--models-only` skips README publication while bulk transfer is in progress. Credentials come from the existing Hub cache, never repository files. Preserve all component licenses, including DINOv3 and the FLUX Non-Commercial base terms. `data/model-publication-v041/` contains local transfer receipts and is excluded from Git.
