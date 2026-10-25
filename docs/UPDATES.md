# Program updates and downloads

**English** | [简体中文](UPDATES.zh-CN.md)

GitHub Releases contain source, ComfyUI nodes and code updates only: **no model weights, Python interpreter or bundled runtime**. **T8star will distribute the complete package separately** and announce its download location. The source archive is not a ready-to-run bundle. The complete package's `bundle-manifest.json` identifies its included models.

## Windows console launcher

From v0.4.2, bundles include `Hi3D-Workbench.exe`, a visible console launcher that owns the server and its workers. Close with Ctrl+C, the console X, the webpage shutdown button or `Stop-Workbench.cmd`; closing the browser alone keeps the service running. The code updater continues to exclude executables for compatibility with v0.4.1. The updated start script builds a missing launcher using Windows' .NET Framework 4 compiler, or you can extract the separate launcher ZIP from the matching Release after applying its code update. Future launcher binary changes require that launcher asset; Python and models remain separate.

## Automatic updates for the standalone bundle

Starting with v0.4.1, the standalone bundle checks this project's stable GitHub Releases while idle, at most once every six hours. It downloads and verifies the code archive and every file using SHA256. Installation happens **on the next workbench launch**, after the previous server has closed; inference is never interrupted for an update.

Use **Models and environment → Program updates** to disable automatic downloads/installation or check manually. Network failures leave the installed program usable. Updates preserve `models/`, `runtime/`, `runtime-pipeline/`, `data/`, `config.local.json` and personal ComfyUI workflows.

The installer backs up affected code to `data/updates/backups/`. Failed writes are rolled back. An interrupted transaction is restored before a fresh process reload. Invalid backups stop startup with an error rather than claiming recovery succeeded. Backups are retained.

Alternatively, close the workbench and run `Update-Workbench.cmd`. Installation refuses to overwrite a running server in the same directory. Git checkouts disable automatic installation; update source checkouts with Git.

## First upgrade from v0.4.0

The old package has no updater. Close it and extract `Hi3D-Workbench-update-0.4.1.zip` over the existing workbench root, retaining its models and runtimes. Launch `Start-Workbench.cmd` to enable subsequent automatic updates. This ZIP cannot run by itself.

A release requiring different Python, CUDA or third-party dependencies will request a new complete package from the author. The updater never installs or replaces Python environments. Maintainers must increase `updater.RUNTIME_API` whenever the required runtime changes, and provide a matching complete bundle.

## Model downloads

From v0.4.5, **Models and environment → Model directory** shows the active absolute root and the saved path. Enter an absolute directory or a path relative to the workbench root; Chinese characters and spaces are supported. Save applies immediately while idle, with no restart. Running tasks, downloads, verification, diagnostics and unresolved worker recovery block a switch. The existing path is retained if a request is rejected.

The root contains `hi3d/`, `dino/`, `clip/` and optional `extensions/`. An absent directory can be saved and is created when a download needs it. Switching or restoring the default never moves or deletes existing models; **Restore default** selects the workbench's `models/`. The setting is stored as `models_dir` in `config.local.json` and survives code updates. Relative paths relocate with the workbench; an absolute external path must still be available after moving the bundle. Prepare and verify models at the selected location before inference.

Even if core files already exist at the new root, use **Reverify all core models** before expecting a ready status. Saving a directory does not certify its file contents.

[t8star/Hi3d-Comfy](https://huggingface.co/t8star/Hi3d-Comfy) hosts the shared Hi3D core and standalone extension models. ComfyUI needs only the core; standalone geometry/PBR dependencies depend on the selected engine. Code updates do not download all models or change pinned model revisions.

From the workbench root, download core files only:

```bash
hf download t8star/Hi3d-Comfy --local-dir models --include "manifest.json" "hi3d/*" "dino/*" "clip/*" "licenses/*"
```

Download every core and extension component (about 79.3GB, plus runtime and working space):

```bash
hf download t8star/Hi3d-Comfy --local-dir models
```

See the model repository's bilingual README for directories and licenses. UltraTex's FLUX.2 Klein 9B base uses a non-commercial license.

v0.4.5 optional accelerators can be installed separately without changing the runtime ABI: `runtime/python.exe tools/setup_pipeline_runtime.py --runtime runtime-pipeline --attention-only` for the pinned native Flash wheel, or `--uv-only` for the pinned Numba packer. Code updates retain SDPA/Torch fallbacks and never silently install these dependencies. GPU probes, numerical/model checks and first-use JIT timing are recorded separately from package import validation.
