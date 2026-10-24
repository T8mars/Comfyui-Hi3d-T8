# Hi3D v0.3.1 · 20 轮联合复查

- 审查日期：2026-10-24；结束时间：12:02 UTC / 08:02 America/New_York。
- 基线：Git `7e58822`。启动时仅有原有未跟踪 `IMPLEMENTATION_STATUS.md`、`research/`；没有适用的 `E:/AGENTS.md` 或项目 `AGENTS.md`。
- 分工：本子 Agent 负责引擎、扩展模型管理和任务生命周期；根 Agent 负责 server/UI/打包/文档。只修改了 `hi3d_workbench/pipeline_models.py`，新增 `tests/test_pipeline_engine_audit.py`。
- 本次未启动 GPU 推理、未下载任何真实权重、未停止/重启 8378 服务、未改 ComfyUI 节点、未提交 Git 或上传。API 测试全部使用原测试框架的隔离临时 ROOT。

## 已复现与修复

1. **P1：文件校验与签名记录之间存在竞态。** `verify()` 原先先读哈希、后读 signature；同大小文件在两者之间变动，会把损坏文件标为 verified。下载完成路径有相同问题。现在在哈希前后取得完整签名，只在签名一致且大小/哈希均正确时保存成功记录。失败文件不能继续保持 ready。
2. **P2：扩展模型的小文件或配置损坏后，重新下载仍可能命中 Hub 旧缓存。** 原实现检出损坏即失败，但没有让 SDK 重新取固定 revision 的内容。现在仅在校验失败时执行一次 `force_download=True`，随后重新验证。仍错误则失败，绝不写入新的已验证清单。
3. **P2：扩展清单 repo/revision 不匹配时，验证结果没有错误条目。** readiness 原来会为 false，但 `verification_errors` 可为空。现在明确记录 `component/manifest:repo`、`component/manifest:revision`。
4. **P3：模型已就绪时默认提示“扩展模型尚未下载”。** 仅当实际 ready、下载状态 idle 且仍是这一默认文案时改为“扩展模型已就绪”。真实 failed/interrupted/complete 诊断不覆盖。

修前运行新增前三个回归实例，结果 **3 failed**：一个校验竞态、非 LFS 缓存恢复、LFS 缓存恢复。修后首次为 3 passed，扩展边界覆盖后最终为 **76 passed / 0 failed / 0 skipped**（24 个新增实例 + 52 个已有 asset-pipeline 实例）。唯一 warning 是已有 Starlette/AnyIO 别名弃用提示。

最终命令：

```text
.appenv/Scripts/python.exe -m pytest tests/test_pipeline_engine_audit.py tests/test_asset_pipeline.py -q --junitxml=data/qa/post-v03-agent-engine-tests.xml
76 passed, 1 warning in 10.54s
```

机器可读原始回执：`post-v03-agent-engine-tests.xml`，tests=76、failures=0、errors=0、skipped=0、time=10.525。

## 20 轮目标、执行证据与结论

| 轮 | 目标 | 实际执行证据 | 结论 |
|---|---|---|---|
| 01 | 明确改动和受保护范围 | `git status --short`、`git rev-parse --short HEAD`，查根与项目 AGENTS；读取现有验证回执。另 `git diff --numstat -- comfy_nodes.py __init__.py examples/Hi3D-Refine.json examples/README.md` 无输出 | 基线干净，仅原有未跟踪资料；本次不改变 ComfyUI 接口或画布工作流 |
| 02 | 三种入口仅要求它们真正使用的模型 | `test_modes_require_only_their_declared_model_components` 的 7 个参数实例，`test_existing_model_modes_reject_missing_mesh_before_model_process` 两实例、image-only 无 mesh 两实例 | 图片可独立生成，贴图不要求 Hi3D，已有粗模入口缺 mesh 会在模型进程前拒绝 |
| 03 | HTTP 参数不得控制本地命令或输出路径 | `test_untrusted_options_cannot_select_paths_commands_or_unsupported_modes` 8 实例与 `test_http_extra_fields_never_become_worker_commands_or_output_paths` | 非法 mode/路径/分辨率被拒；额外 worker_command/model_root/output_path 不进入请求文件 |
| 04 | 粗模→精修→贴图顺序与默认路径正确 | `test_supervisor_sequences_coarse_refine_then_textures_final_mesh`、`test_image_pipeline_omitting_refine_flag_textures_coarse_geometry_directly`、`test_supervisor_can_skip_hi3d_and_texture_for_geometry_only_image_mode`、`test_original_refine_mode_still_runs_hi3d_when_optional_refine_flag_is_omitted` | 贴图使用实际最终网格；默认不经过 Hi3D；旧精修模式兼容。这里是调度 fixture，不冒充模型质量测试 |
| 05 | 后续失败不得伪造成功，重用不得破坏原始输入 | `test_supervisor_does_not_claim_success_when_engine_returns_untextured_mesh`、两个 failed-job reuse 测试 | 灰模返回导致失败，保留 coarse/refined；图片失败可无粗模重用；贴图重用保留原始几何字节 |
| 06 | 前景 alpha 与保留背景路径不偷偷加载分割权重 | `test_foreground_uses_valid_alpha_and_keep_without_loading_onnx`，把 onnxruntime 导入显式置为不可用仍通过；已有 `validation/asset-pipeline-alpha.json` 实际 CPU U2Netp 回执另行阅读 | 有效 alpha 字节不变；keep 不触发 ONNX；之前 alpha=254 的真实 CPU 修复证据保留。本次没有重跑实际 ONNX |
| 07 | 空透明图片尽早拒绝 | `test_empty_transparent_foreground_fails_before_model_inference`，全透明 PNG、ONNX 导入不可用 | 返回“没有有效前景”，不运行模型 |
| 08 | 固定推理源码不能越界或悄悄被改 | `test_pinned_engine_sources_reject_mutation_and_escape[changed/traversal]`，真实临时源码与 source-manifest，检查 sys.path 未被添加 | 哈希变更与 `../outside.py` 都在 import 之前拒绝 |
| 09 | 无 CUDA 与离线模式边界 | `test_stage_no_cuda_fails_explicitly_and_forces_offline_before_model_import`，只 stub torch.cuda 的能力检查、不加载权重 | 在模型导入前强制 HF/Transformers OFFLINE=1；无 CUDA 明确失败，不做 CPU 假推理 |
| 10 | 阶段坏请求不能覆盖已有交付文件 | `test_stage_preflight_records_errors_without_loading_a_model` 7 个实例：非 JSON 对象、非 GLB、背景参数非法、缺图、缺网格、非法分辨率、非法贴图大小 | 每例写 failed 诊断；原结果字节不变；无 partial 文件；torch 导入被禁用，证明未进入模型阶段 |
| 11 | 子阶段 complete 不能变成整个流水线 complete | `test_child_progress_is_logged_and_stage_completion_is_not_pipeline_completion`，启动真实短 Python 子进程输出 complete/list/坏 JSON | 原始日志保留；下游事件变为 texture_complete；坏输出不杀死正常子阶段；process.json 身份完整 |
| 12 | 非零阶段退出传递真实失败原因 | `test_child_failure_uses_saved_stage_diagnostic`，真实子进程 exit 7 与临时 failure.json | 显示“texture 阶段失败：fixture CUDA OOM”，不显示成功 |
| 13 | 启动恢复只结束确证属于自己的工作进程 | 现有 `test_stage_recovery_reaps_only_verified_identity_and_blocks_unknown_survivor` 4 个实例，均启动真实 sleep 子进程 | PID+创建时间+请求路径匹配才终止；错时间、缺/坏身份记录只阻止新 GPU 任务，不误杀 |
| 14 | Supervisor 已退出时仍不能释放存活阶段的 GPU 入口 | `test_supervisor_exit_cleans_or_guards_surviving_stage_before_releasing_job` 5 个实例：匹配、错时间、缺记录、无关进程、AccessDenied | 匹配者清理；其余保留阻塞；独立 outsider 不被杀。全部子进程由测试 finally 收尾 |
| 15 | 扩展状态应拒绝坏清单和同大小文件篡改 | 现有 malformed schema 3 实例及 restored-mtime corruption 实例 | components/files 错型不崩溃；恢复 mtime 仍不能使损坏权重 ready |
| 16 | 哈希到 signature 的竞态不能给损坏文件背书 | 新增 `test_extension_file_changed_during_verification_is_never_marked_ready` 与 `test_extension_download_never_binds_hash_to_a_changed_file`；_hash 返回后真实改写同大小临时文件 | **发现并修复**。verify 写 verified=false；下载失败不改原 manifest；status 不 ready |
| 17 | 损坏 Hub 小文件缓存可恢复，但不无限重试 | `test_extension_download_repairs_corrupt_small_hub_cache[False/True]`，真实下载控制流、网络传输为本地 fixture | **发现并修复**。固定版本普通/force 两次调用，第二次正确则 ready；持续被改写时失败 |
| 18 | 断点分片不循环复用坏数据，也不读取忽略 Range 的完整响应 | `test_bad_download_ranges_are_quarantined_before_retry`、`test_range_download_rejects_a_server_ignoring_range` | 坏分片被保留到 invalid-* 诊断目录，下一次新分片成功；HTTP 200 在读取响应 body 前拒绝，无最终文件 |
| 19 | 跨进程锁、来源错误和默认状态须一致 | 新增真实 FileLock 子进程测试；repo/revision 两实例；ready 默认提示保留真实诊断状态的测试 | 下载锁存在时不改 manifest；**修复**错误来源漏报及 ready 文案矛盾；failed/interrupted/complete 信息不覆盖 |
| 20 | PBR 真实性、UV 接缝与缩放导出不退化 | 已有真实图片/UV GLB fixtures、80mm 导出、exact-copy、灰模拒绝、外部法线图片拒绝、UV seam 开/闭盒测试全部运行；阅读 `validation/asset-pipeline-portable-http.json`、`asset-pipeline-cancellation.json`、`asset-pipeline-portable-native-gpu.json` | 颜色/金属粗糙度图实际可解码且字节保留；UV 接缝不虚报洞，真实洞不隐藏；材质不默默导出 STL/PLY。原 GPU 全链路/取消证据存在，但本轮不将旧回执冒充新 GPU 执行 |

## 主 Agent 交叉检查与最终回归

- 失败上传会遗留 UUID 空目录，或在几何检查失败后遗留已转换的大 GLB。新增两个实例均在修前失败、修后通过；只清理本次新建目录，原有资产字节保持不变。
- 浏览器故障回归 9 组通过：健康请求失败/超时、历史缺失恢复、重连选项、尺寸导出参数锁定、异步蒙版保存、相机同步/重置、加载期间隐藏旧模型、缺失产物处理、版本缓存键。独立临时服务执行，零 JavaScript 错误。
- 三入口与模型依赖 6 组通过：图片无需粗模/Hi3D、纯粗模无需贴图组件、可选阶段只要求对应模型、单独贴图无需 Hi3D、原精修无需贴图环境、扩展校验期间暂停提交。使用明确 HTTP 状态夹具，没有启动 GPU。
- 最终完整套件：**180 passed，0 failed，0 skipped，36.62 秒**。唯一 warning 为已有 Starlette/AnyIO 别名弃用提示。全量回执见 `post-v03-all-tests.xml`；浏览器见 `post-v03-browser-audit.json` 和 `post-v03-modes.json`。
- 用修复后的校验逻辑重新校验本机真实扩展模型：四组件均 ready，来源/内容错误为空。不是网络 fixture。
- ComfyUI 节点实现和画布工作流相对 7e58822 字节未变。未修改权重、固定推理源码、采样数学或 CUDA 扩展。本轮沿用 v0.3.0 的真实生成/贴图/取消/导出产物验收，不将 CPU/HTTP 夹具称为新 GPU 推理。
- 发行目录和 ZIP 必须重新构建；最终归档结果以 `dist/RELEASE-VERIFIED-0.3.1.json` 和 `dist/SHA256.json` 为准，报告不为旧版 ZIP 背书。

## 验证范围

实际硬件仍仅 Windows / RTX 5090 Laptop 24GB / 64GB RAM。未验证 6/8/12GB 真卡、其他 GPU 或 Linux 扩展。传输恢复测试使用本地响应夹具，不能保证所有代理和远端限流行为。
