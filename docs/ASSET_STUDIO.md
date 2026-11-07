# Asset processing and single-character animation

**English** | [简体中文](ASSET_STUDIO.zh-CN.md)

**Development preview: final acceptance and complete-bundle delivery are unfinished.** See the [roadmap](../roadmap.md) for evidence and outstanding failures. These features belong to the standalone workbench; ComfyUI continues to provide independent Hi3D geometry refinement only.

| Operation | Backend | Result |
| --- | --- | --- |
| Material-preserving simplification | meshoptimizer | Target triangle count or ratio, retaining usable UVs, textures and attributes. Error protection can stop above the target. |
| Quad retopology | AutoRemesher / Blender QuadriFlow, CPU/GPU UV and CPU transfer | Textured, triangulated GLB with a new atlas; OBJ retains quads and other polygons. Their face counts use different units. |
| Automatic part segmentation | P3-SAM | Original-surface partitioning with selection, creation, merging, renaming and brush correction; assembly GLB and individual-part ZIP. |
| Part completion | Public light X-Part | Reconstructed hidden geometry, with the original partition retained. New interior materials are inferred from the corresponding original part. |
| Skeleton and skinning | SkinTokens / UniRig | Generate a rig, or reskin an existing skeleton; edit rest joints and inspect deformation. |
| Rigid part attachment | Explicit correction, without another AI model | Attach a separated tail, armour plate or accessory to one chosen bone; retain the original rigged file. |
| Motion import and retargeting | BVH, FBX and animated GLB | Select source armature and clip, then calibrate mapping, axes, scale and rest pose for a rigged character. |
| Text-to-motion | Kimodo-SOMA-RP-v1.2 | Duration, seed, body-pose and path constraints; animated GLB/FBX and motion-only BVH. |

## Workflow

1. Upload a GLB or select a saved result in Asset Studio. Existing meshes do not require another image-generation pass.
2. Simplify or retopologize as needed. Inspect actual counts, silhouette, details and materials. Retopology changes vertices and requires rebinding; the original rig and animation remain in the source file.
3. Inspect and correct part assignments. The spherical brush also selects rear-facing surfaces; radius zero selects individual triangles.
4. Rig a human or humanoid character. Check joint placement, shoulders, elbows, knees and clothing. Use the existing-skeleton option to regenerate weights while retaining the rig.
   If limbs pull clothing or the opposite leg into long strips, try limb-contact correction. It separates suspected contacts using existing neural weights while protecting normal shoulder/hip blends and retaining original triangles, UVs, materials and the skeleton. The raw rigged GLB remains downloadable. The resulting boundary can be open; this does not replace print repair or fix every skinning error.
   For a tail or accessory that should not bend with a leg, segment or brush-separate and save the part first. Choose rigid part attachment, select the part and bone, then add the assignment. The entire part follows one bone; this does not generate natural tail or cloth motion. Other parts retain their weights. Saved part names and identities remain editable after rigging and motion processing.
   Enable attachment ground avoidance if the part intersects the floor. It adds an independent attachment joint and pitches the part upward during subsequent motion, using the sole-defined ground while retaining the original body joints. It does not handle body, weapon or scene-object collisions, and reports cases it cannot resolve.
5. Upload motion or enter a text prompt. Check the source-to-target mapping before choosing speed, looping, root translation/in-place mode and foot-contact correction.
   Foot correction evaluates the actual skinned soles, resolves floor penetration even when source contact labels have a gap, and retains normal airtime. It does not provide full-body or scene collision handling.
6. Preview the complete clip and multiple views, then export. BVH contains motion, without the character mesh or textures; choose GLB/FBX for a textured character.

## Quality and resources

- Simplification and retopology solve different problems. A target quad count is not an exact-count promise or a guarantee of professional character edge loops.
- Retopology welds identical seam coordinates in its geometry copy and evaluates initial skin/morph deformation. Dense inputs use an error-limited proxy; the original high-resolution asset remains the material donor. Actual counts and algorithm-reported error are included in the report.
- QuadriFlow is an additional CPU option. Original surfaces remain the default. Solid-surface preparation can close openings, connect nearby structures and lose thin details; it only changes the inference copy. Generated small holes and degenerate faces are repaired when boundary/non-manifold counts do not increase, and actual topology is reported.
- Transfer offers nearest-surface or outward normal-cage projection, with nearest fallback for missed rays. Layered fur, clothing and armour can sample a different layer; compare against the source rather than treating the presence of textures as equal material quality.
- GLB uses Blender polygon tessellation instead of a simple triangle fan across concave polygons. The quad OBJ remains separate. New Blender stages participate in cancellation and restart recovery.
- Matching geometry, parameters and native-tool versions can reuse verified geometry and UVs. Materials are transferred from the current source again. Invalid caches are rebuilt.
- Retopology and part completion offer automatic, GPU and CPU UV selection. Automatic mode uses GPU for meshes with at least 50,000 faces when the private CUDA runtime is available; smaller meshes use CPU. GPU failures report their cause instead of silently restarting a long CPU unwrap. Cached layouts reject zero-area or duplicate UV triangles and preserve oriented source triangles, coordinates and normals. Inspect materials on newly reconstructed interior surfaces separately.
- Surface transfer handles base color, alpha, metallic, roughness, AO, normals and emission. It cannot create reliable textures for unseen interiors. Unsupported material extensions fail explicitly.
- Surface segmentation does not guarantee closed parts. Closed geometry does not establish printability, strength, connectors or assembly tolerances. Public light X-Part is not the full online model.
- Before baking, X-Part checks distances from the original part to the new surface. It rejects a result when more than 5% of original vertices deviate by over 3% of the part extent, retaining the original partition and candidate geometry. Passing this numeric check does not establish visual quality or reliable textures for new interiors.
- Newly generated parts merge only positions that coincide at glTF float32 precision and remove faces collapsed by that merge. Other valid oriented triangles remain, and loss of existing closure or winding consistency rejects the result. Uploaded originals and their original partitions are untouched.
- The original partition GLB and ZIP are published before completion starts. Downloads lists saved files even after failure or cancellation, without waiting for the 3D preview. Select `original-segmentation.glb` from model sources to edit it again; its hash-matched part metadata remains separate from the reconstructed output.
- Editing, rigging and motion processing retain reconstruction provenance and verified original-partition attachments. Reconstruction history does not guarantee that a newly edited cut is closed; inspect each part's current boundary report.
- X-Part defaults to shape guidance 7.5, with controls for guidance, steps and geometry resolution. Encoding, guided sampling and decoding use the GPU in separate FP32 stages; logs show actual encoder and sampling progress. Excessive guidance can add small fragments, so inspect shape and material quality.
- Animation covers one character. Automatic skinning can pull clothing incorrectly; generated motion can miss a prompt, slide or intersect geometry. Human motion is not directly compatible with animal skeletons. Current body constraints do not provide detailed finger-gesture control.
- Upstream SkinTokens requires at least 14GB VRAM; UniRig's model card specifies more than 8GB. Kimodo CPU text encoding reduces VRAM at the cost of host RAM and encoding time. Measurements on one GPU/input do not establish minimum requirements for other cards.
- GPU models run in stages and are released when their worker exits. CPU UV, surface transfer and native CPU stages may show no GPU activity. Progress counters describe their active stage, not an estimated percentage of the whole task.

## Installation and licenses

Asset backends use an isolated `runtime-assets`, with additional private dependency overlays for Kimodo and UniRig. Do not upgrade the original ComfyUI/Hi3D Python or Torch to install them. Pinned dependencies are recorded in [runtime_requirements](../runtime_requirements/). Asset models use the `studio` directory under the configured model root; follow the location shown by the workbench.

After extraction or copying, startup rehashes installed asset components whose file receipts are stale, showing actual file counts. It never downloads missing components automatically and reuses receipts that remain valid. This disk integrity check does not load every model onto the GPU.

GitHub code updates contain no models or Python. New backends require a separately built and validated complete bundle; copying node code does not install all features.

Licenses remain separate: SkinTokens/UniRig use MIT; P3-SAM/X-Part use the Tencent Hunyuan 3D-Part Community License; Kimodo code uses Apache-2.0 and the selected RP weights use the NVIDIA Open Model License. The Llama 3 text base and adapters retain Llama 3 terms, and base-model access requires an authorized account. Blender, native tools and dependencies retain their own licenses. Sources, pinned revisions and hashes are recorded in the [model catalog](../hi3d_workbench/studio/model-catalog.json).

Original licenses, notices and model cards are collected in the [notice manifest](../licenses/asset-studio/manifest.json). The four backend source revisions and Git blob hashes are pinned in the [source manifest](../runtime_requirements/studio-sources.json). Full asset-runtime packaging verifies these files and rejects missing, modified or extra source files. File integrity, visual quality and complete-bundle acceptance are recorded separately.

QuadriFlow runs through bundled Blender without extra AI weights. Distribution retains Blender's GPL and bundled third-party notices. QuadriFlow's actual license contains BSD-style terms; its README's MIT label must not replace the original notice.

## Developer validation

The [candidate evidence summary](../validation/asset-studio-candidate-0.6.0.json) separates backend/artifact checks, sampled renders and browser acceptance. Outstanding acceptance remains pending.

`tools/validate_studio_http.py` exercises a new isolated service and real workers: upload, simplification, motion import, cancellation during sampling and restart recovery. It keeps its evidence under the checkout's `data` directory and does not use production ports or data. Supply a rigged `--source`, a BVH/FBX/GLB `--motion` and a new `--output` directory; the required runtime features must already be ready.

`tools/validate_studio_motions.py` runs six categories sequentially, with `--seed` and `--text-device cpu|cuda`. Export success is recorded separately from visual acceptance. `tools/render_studio_motion_validation.py` runs inside Blender; optional `--follow-root` follows horizontal translation with fixed projection and a stationary world grid, without hiding vertical movement or reframing each pose.

`tools/validate_studio_deformation.py --source <rigged-GLB> --result <animated-GLB> --output <report-JSON> --floor <height>` independently reads saved animation, inverse bind matrices and weights, then measures every vertex and triangle at 30fps for floor penetration and severe stretching. Its diagnostic thresholds do not establish overall visual acceptance. Use the renderer's `--camera-from <old-render.json>` to compare with the same fixed framing and ground.

`tools/validate_studio_contacts.py --source <raw-rigged-GLB> --result <corrected-GLB> --stage <studio-contact-directory>` checks source/result/face-map hashes and rereads every original face, vertex attribute, UV, morph target, material, rig and animation. Optional `--bone-name <name> --angle 40` compares a local bend probe; it is separate from full-clip and browser acceptance.

For contact-separated outputs, run `tools/validate_studio_binding.py --source <pre-binding-GLB> --original-binding <raw-rigged-GLB> --result <corrected-GLB> --contact-stage <studio-contact-directory> --report <JSON-report>` to check the complete preservation chain. Add `--existing-skeleton` when reskinning an existing rig to verify its skeleton and every animation sample.

The experimental Qt-free AutoRemesher build uses `tools/build_studio_autoremesher.py --source <clean-upstream-checkout> --build <new-build-directory>`, Visual Studio 2022 and the Windows SDK. It requires upstream commit `4a49f82f720f647691c09a7d8a16fa63be208aad`, records source/binary hashes and retains dependency licenses. Building it does not install it, mark visual quality accepted or replace the current runtime.
