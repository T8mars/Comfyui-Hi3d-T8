# Workbench performance and progress

**English** | [简体中文](PERFORMANCE.zh-CN.md)

Version 0.4.5 addresses UV preparation, progress reporting and optional TRELLIS.2 attention acceleration in the standalone workbench. ComfyUI nodes and canvas workflows remain Hi3D geometry refinement only; this release does not add the workbench's other engines to ComfyUI. GitHub source, node, update and launcher assets contain no Python interpreter or model weights. T8star distributes the complete bundle separately.

## Why a low GPU reading can accompany a slow task

The reported problem was not evidence of a CPU-only model renderer. Its geometry stage completed with CUDA in 158.684 seconds, producing 973,968 faces. The old UV path could finish a native unwrap, reject its vertex/normal correspondence and then repeat the full mesh on CPU xatlas. The interrupted task spent at least 1,381.888 seconds after its UV event; this is an interrupted wait, not the duration of a successful UV operation. See [the validation record](../VALIDATION.md).

Runtime imports, checkpoint reads, CPU parameterization/packing and export can leave the GPU mostly idle. Windows diagnosis also identified E-drive PageIn waits while loading DLLs and weights. These reads belong to storage and memory loading, not an attention kernel. Switching attention or using an INT8 model cannot by itself eliminate those waits. Preserve separate timings for loading, sampling, UV and export before comparing speed.

TRELLIS mesh encoding also starts with the official `o_voxel` CPU voxelization path. The workbench reports that as CPU preparation and changes to CUDA encoding only when the shape encoder's actual forward call starts. A low GPU reading during this preparation is expected; it is not reported as CUDA merely because the later encoder uses CUDA.

The storage snapshot mapped E: to disk/SCSI port 1, recorded a queue length of 28, and found `stornvme` event 129 resets for `RaidPort1` earlier that day. A separate two-second sample measured 14.808 ms per read across 1,782 E-drive reads. C:/D: had only one read in that sample (1.391 ms), so these figures are not a fair hardware comparison and do not establish a failed disk or a specific driver cause. Raw local evidence: `data/v045-storage-evidence.json` and `data/v045-read-latency.json`.

## Progress shown in the workbench and launcher

Both surfaces report actual stage/phase events, observed work counts, elapsed stage time, time since the last effective progress update and an execution device only when the stage explicitly reports it. TRELLIS samplers, checkpoint loading and UV preparation supply their own counters. The counter for checkpoints counts models loaded, not the percentage of weight bytes read.

Sampling accepts both `step/steps` and `step/num_inference_steps`. The displayed sampling call number identifies an actual sampler invocation; no total number of calls is invented. A phase with a known total has a local progress bar. A UV iteration or merge count without a known total shows the count and message without a percentage or ETA. A quiet loading phase can therefore show an increasing time since the last update without proving that the task is stuck.

Merged stderr progress displays can put `HI3D_EVENT` JSON in the middle of a carriage-return line. The shared parser recovers valid events with surrounding display text, rejects malformed/non-finite events and preserves original stage logs. The pipeline emits normalized events once, retaining the pipeline stage. Frequent callbacks are throttled for console output and disk writes while the API keeps the latest observed state.

## UV preparation and reuse

The default order is **native CuMesh → community GPU PEC → bounded CPU partitions**. The first route reuses CuMesh GPU clustering and xatlas parameterization, verifies source correspondence and repairs local UV-only failures. This avoids loading extra Comfy/LLVM modules on the common path. The pinned ComfyUI community route remains an alternative with original source indices and no automatic vertex welding. The safety path partitions source topology before CPU atlas work instead of sending the whole large mesh through a second unbounded unwrap. Structural UV failures remain visible; invalid UVs are not accepted just to finish a task.

Validation checks source-face coverage, winding, positions, corner normals, finite/unit-range UVs, collapsed triangles and exact duplicate/mirrored UV triangles. Local UV repair can separate affected UV corners without simplifying the source geometry. These checks do not prove the absence of every arbitrary partial overlap.

Verified UV data is cached under `data/uv-cache`. Its identity includes geometry, normals, winding, requested atlas size, padding, selected backend and algorithm/source hashes. Cache contents are verified again before reuse; a corrupt or incompatible entry is rebuilt. The cache contains UV/index data, not generated material pixels, so a new painting task still runs its texture model. A valid existing atlas can bypass unwrapping.

The original 973,968-face problem mesh has now passed both UV routes and verified cache reloads. Each retained all 973,926 non-degenerate source faces and excluded the same 42 exact zero-area faces; neither route reduced valid geometry.

| Explicit UV backend | First measured unwrap | Verified cache reload | Faces with local UV repair |
|---|---:|---:|---:|
| Native CuMesh | 65.155 s | 1.275 s | 108 |
| Community GPU PEC | 162.418 s | 1.437 s | 2,543 |

These are UV preparation timings, not texture-generation or complete-task timings. The community run included about 54 seconds of cold dependency loading and 18 seconds of first-use Numba JIT; this does not establish a general 2.5× algorithm advantage for CuMesh. The native run also needed fewer local repairs on this input, supporting its choice as the default rather than a universal quality ranking. Its 108 repaired faces included 107 duplicate-UV cases and 97 minimum-height adjustments, which are overlapping categories; the internal atlas side grew to 4,112. The community repairs addressed 2,543 coincident, opposite-winding source-face occurrences, with 14 minimum-height adjustments and no unrelated UV collisions in that repair report. Raw evidence: `data/uv-verified-v045/results.json` and each backend's events/repair diagnostics.

All four actual GLBs (each backend's generated and cached output) also passed re-import through `load_texture_input`: source positions were exact at float32 precision, normals passed `rtol=0, atol=2e-6`, and all 973,926 oriented source-face occurrences were retained. The exported UVs passed zero-area and exact-duplicate guards. This verifies the UV/geometry export round trip, not generated PBR appearance. Receipt: `data/uv-verified-v045/glb-roundtrip.json`.

For **new TRELLIS.2 geometry**, surface cleanup offers 100K, 300K and 1M face budgets, with 1M retained as the default. The raw generated geometry is saved separately for comparison. Lower budgets reduce downstream mesh work but can remove detail. **Existing-model texturing never applies this face budget or simplifies the uploaded mesh.**

## TRELLIS.2 attention choices

| Selection | Behavior |
|---|---|
| `auto` (default) | Probe native packed FlashAttention CUDA kernels; report and use fused SDPA if native support is unavailable |
| `flash_attn` | Require the native probe to pass; fail explicitly when the pinned extension is unavailable |
| `sdpa` | Use the explicit fused PyTorch SDPA compatibility path |

This choice controls TRELLIS.2 geometry/texturing. Hi3D's `attention` setting remains independent. TRELLIS dense attention remains SDPA; the native option accelerates compatible sparse packed calls. Unsupported native tensor types or head dimensions keep their precision and use the reported SDPA path. UltraTex's own inference/attention path is not switched by this control.

On Windows / RTX 5090 Laptop 24GB / 64GB RAM, native FlashAttention 2.8.3 passed packed-kernel comparisons with fused SDPA for FP16/BF16, head dimensions 64/128 and unequal sequence lengths. The BF16 microbenchmark below uses eight heads, head dimension 128, three warm-up calls and the mean of ten synchronized calls.

| Packed workload | Native | SDPA | Ratio for this kernel sample |
|---|---:|---:|---:|
| 256 sequences, each length 32 | 2.248 ms | 2.485 ms | 1.11× |
| 256 unequal sequences, lengths 1–255 | 6.720 ms | 13.945 ms | 2.08× |

These are operator timings, not whole-task speedups. In the same-image, seed-0, 512-resolution, 12-step geometry comparison with CPU offload and a 300K face budget, **SDPA took 359.211 seconds and native attention took 505.729 seconds**. The native run recorded 1,260 native packed calls and zero packed SDPA fallbacks, so the extension was used; it still did not make this measured task faster. Loading/PageIn and other work affected the total, and one run per mode does not establish a general ranking. Outputs also differed, so bitwise equivalence is not claimed. Local raw records: `data/uv-final-v045/attention.json` and `data/trellis-attention-v045/comparison.json`.

A repeat operator measurement in `data/uv-verified-v045/attention.json` produced different timings, particularly for ragged SDPA. This variability is another reason to retain the measured workload and raw records rather than advertise one universal speed multiplier.

## Optional community wheels

The runtime installer can add the pinned [kingbri1 FlashAttention Windows wheel](https://github.com/kingbri1/flash-attention/releases/tag/v2.8.3) or [Numba 0.61.2](https://pypi.org/project/numba/0.61.2/) / [llvmlite 0.44.0](https://pypi.org/project/llvmlite/0.44.0/) for the upstream optional CPU packer. Assets have fixed sizes and SHA256 hashes. Installation uses verified local wheels with `--no-deps`; it does not replace Torch, CUDA or NumPy. Native attention requires the recorded CPython 3.12 / Windows x64 / Torch 2.8.0+cu128 ABI; the Numba path separately requires NumPy 1.26.4.

For a compatible complete bundle, run from its root while the workbench is stopped:

```powershell
.\runtime\python.exe tools\setup_pipeline_runtime.py --runtime .\runtime-pipeline --attention-only
.\runtime\python.exe tools\setup_pipeline_runtime.py --runtime .\runtime-pipeline --uv-only
```

Source deployments use their application Python and `--runtime .pipelineenv` instead. `--skip-native-attention` and `--skip-uv-acceleration` retain the compatible alternatives during a full runtime setup. A code update preserves existing runtimes and does not silently install these wheels. The installer records `attention-acceleration.json` and `uv-acceleration.json` in the target runtime. An installation/import receipt is separate from a successful GPU probe, first-use JIT, correct UV output and measured performance; inspect each recorded status. If optional Numba is unavailable, the upstream Torch packer remains available with a reported reason.

The E-drive complete bundle has completed both optional installers and `pip check`. That confirms package installation/dependency consistency; its import receipts do not substitute for the separate GPU/UV acceptance below.

## Model location

**Models and environment → Model directory** shows the active root and saved configuration. A custom absolute path, or a path relative to the workbench root, can be saved while idle and applies immediately without a restart. Core folders `hi3d/`, `dino/`, `clip/` and optional `extensions/` belong directly under that root. This lets an existing model collection be used without copying it into the bundle.

Switching is blocked during tasks, downloads, verification, diagnostics and unresolved worker recovery. No weights are moved or deleted; a missing root is only created when a download needs it. **Restore default** selects the workbench's `models/`. Code updates preserve `config.local.json.models_dir`. Choosing a different directory does not itself resolve a storage bottleneck; compare real loading times at the chosen location.

After switching, existing core files still need **Reverify all core models** before the new root is reported ready. A directory containing weights is not automatically a verified installation.

## Complete complex-mesh PBR run

The actual E-drive EXE service completed existing-model TRELLIS.2 PBR on the original large mesh with `auto` UV/attention selection, 1024 sampling, 12 steps and a 4096 atlas. It used native CuMesh UV with 108 local UV repairs and retained all 973,926 valid faces; the geometry budget did not simplify this existing model.

| Measured scope | Result |
|---|---:|
| HTTP submission to observed successful state | 780.828 s |
| Pipeline supervisor report | 766.36 s |
| Isolated texture stage | 745.608 s |
| UV preparation within that stage | 69.358 s |
| 12-step sampling loop | 45.61 s |
| Native / SDPA sparse packed calls | 720 / 0 |
| Maximum sampled GPU utilization | 100% |
| PyTorch allocated / reserved peaks | 16,118.76 / 27,066 MiB |

These scopes overlap and must not be added. The reserved figure is a Windows WDDM/PyTorch allocator statistic, not physical VRAM residency or a minimum hardware requirement. The total includes loading and CPU work, so it is not a clean attention speed benchmark.

The actual GLB passed source comparison: 973,926 faces, position maximum error 0, normal/UV maximum errors about `2.98e-8`, and two embedded 4K base-color/metallic-roughness images. A material-preserving 80 mm GLB export passed. Three Blender studio-09 HDR views were rendered from the real GLB, with quarter/back appearance inspected. Local receipts: `data/v045-pbr-http/result.json`, `glb-validation.json`, `export-80mm.json` and `render-studio-09/render-manifest.json`.

This run exposed overly broad CUDA labels for encoding/baking. Its output acceptance does not validate those old labels. The follow-up observer reports CPU voxelization before the actual CUDA encoder forward, and the four official `cv2.inpaint` calls as CPU texture-edge filling. Rasterization is reported as CUDA only when observed; an unrecognized mixed baking method carries no invented device. Observation preserves the official function code/defaults/closure and uses a local forwarding proxy rather than changing module globals or texture math.

## Custom-path and corrected-progress acceptance

A second real task completed from a model root containing Chinese characters and a space. Its 1,280-face test mesh used 512 sampling, four steps and a 1024 atlas. HTTP submission-to-success took 192.625 seconds; the texture stage reported 182.962 seconds, 240 native packed calls and no packed SDPA fallback. PyTorch peaks were 3,078.08 MiB allocated / 3,272 MiB reserved. This small functional case is a different workload from the large run above and is neither a speed comparison nor a minimum-VRAM test.

Real HTTP polling observed CPU voxelization → CUDA shape encoding → material sampling steps 0–4/4 → CUDA rasterization → CPU edge filling. The stage log captured the four official inpaint calls' before/after counters (`0,1,1,2,2,3,3,4`); repeated boundaries do not mean eight textures. The resulting GLB contains embedded base color and metallic/roughness.

The live directory switch applied immediately, fresh core/extension workers read the chosen root, switching again during inference returned 409, and restoring the blank/default configuration after completion succeeded. The test root used NTFS junction aliases to existing weights: it did not move/copy 79 GB or benchmark another physical disk. Final health showed core ready and all 7/7 engines ready at the restored default. Evidence: `data/v045-custom-path-pbr/validation.json`, `result.json`, `worker-paths.json`, `custom-path.json`, `restored.json` and `polls.jsonl`.

## Validation scope and limits

The CPU regression suite passed 567 tests in 57.91 seconds, with one dependency deprecation warning. This includes mixed-output event recovery, real HTTP progress reads, worker lifecycle/cancellation, settings propagation, UV/cache fixtures and release checks. It does not count as complete GPU inference or fresh browser-render acceptance.

The complex-mesh UV runs, verified cache reloads, all four UV GLB checks, complete large PBR/GLB/export/render checks and custom-path/corrected-progress functional task passed. Earlier rejected UV attempts remain historical failures. The browser automation surface returned `Browser is not available: iab`, so fresh UI interaction has not been accepted; real HTTP/terminal progress tests do not replace it. See [validation](../VALIDATION.md).

For practical comparisons, use the same input mesh, reference, seed and output settings; separate cold loading from warm inference, and compare UV cache misses with misses or hits with hits. Keep the bundle on the chosen E-drive location with room for caches and outputs, and avoid competing large reads on that volume during a timing run. No result here establishes a minimum VRAM requirement or guaranteed speedup on another input or device.
