# Program updates and downloads

**English** | [简体中文](UPDATES.zh-CN.md)

GitHub Releases contain source, ComfyUI nodes and code updates only: **no model weights, Python interpreter or bundled runtime**. **T8star will distribute the complete package separately** and announce its download location. The source archive is not a ready-to-run bundle. The complete package's `bundle-manifest.json` identifies its included models.

## Automatic updates for the standalone bundle

Starting with v0.4.1, the standalone bundle checks this project's stable GitHub Releases while idle, at most once every six hours. It downloads and verifies the code archive and every file using SHA256. Installation happens **on the next workbench launch**, after the previous server has closed; inference is never interrupted for an update.

Use **Models and environment → Program updates** to disable automatic downloads/installation or check manually. Network failures leave the installed program usable. Updates preserve `models/`, `runtime/`, `runtime-pipeline/`, `data/`, `config.local.json` and personal ComfyUI workflows.

The installer backs up affected code to `data/updates/backups/`. Failed writes are rolled back. An interrupted transaction is restored before a fresh process reload. Invalid backups stop startup with an error rather than claiming recovery succeeded. Backups are retained.

Alternatively, close the workbench and run `Update-Workbench.cmd`. Installation refuses to overwrite a running server in the same directory. Git checkouts disable automatic installation; update source checkouts with Git.

## First upgrade from v0.4.0

The old package has no updater. Close it and extract `Hi3D-Workbench-update-0.4.1.zip` over the existing workbench root, retaining its models and runtimes. Launch `Start-Workbench.cmd` to enable subsequent automatic updates. This ZIP cannot run by itself.

A release requiring different Python, CUDA or third-party dependencies will request a new complete package from the author. The updater never installs or replaces Python environments. Maintainers must increase `updater.RUNTIME_API` whenever the required runtime changes, and provide a matching complete bundle.

## Model downloads

[t8star/Hi3d-Comfy](https://huggingface.co/t8star/Hi3d-Comfy) hosts the shared Hi3D core and standalone extension models. ComfyUI needs only the core; standalone geometry/PBR dependencies depend on the selected engine. Code updates do not download all models or change pinned model revisions.

From the workbench root, download core files only:

```bash
hf download t8star/Hi3d-Comfy --local-dir models --include "manifest.json" "hi3d/*" "dino/*" "clip/*" "licenses/*"
```

Download every core and extension component (about 69.9GB, plus runtime and working space):

```bash
hf download t8star/Hi3d-Comfy --local-dir models
```

See the model repository's bilingual README for directories and licenses. UltraTex's FLUX.2 Klein 9B base uses a non-commercial license.
