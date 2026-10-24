# Hi3D v0.3.2 · 第二次 20 轮联合审查

- 日期：2026-10-24（America/New_York）。基线 Git：`14c5aaa`，版本 0.3.1。
- 范围：扩展模型下载、阶段发布、Supervisor 与 JobManager 边界、工作进程身份恢复。读取了上一轮 `validation/post-v03-audit20.md`，优先新增未覆盖的故障回归。
- 没有发现适用的 `E:/AGENTS.md`、项目根或 `hi3d_workbench/`、`tests/` 下的 AGENTS。初始仅有原有未跟踪 `IMPLEMENTATION_STATUS.md`、`research/`，未修改这两处。
- 修改文件：`hi3d_workbench/engines/run_stage.py`、`jobs.py`、`pipeline_models.py`、`pipeline_worker.py`；新增 `tests/test_pipeline_second_audit.py`。
- 本子 Agent 没有启动 GPU 推理、下载真实模型、连接或重启 8378 用户服务、修改权重/上游固定源码/ComfyUI 节点、提交 Git、推送或打包。所有进程/文件/传输测试使用临时目录与明确 fixture；真实短 Python 子进程只运行脚本和退出码。

## 发现并修复的真实 BUG

1. **P1：阶段失败仍覆盖旧交付模型。** 原来在 CUDA 同步、显存统计和成功报告落盘之前就 `os.replace(temp, output)`。同步异常、统计异常、报告磁盘错误均会把阶段标失败，但旧 GLB 已丢失。现在先完成同步、结果元数据和成功报告落盘，再原子发布 GLB。调用方只有在进程退出 0 后才消费报告；发布失败仍覆盖报告为 failed，旧 GLB 保留。
2. **P2：Range 响应体没有长度上界。** 即便 HTTP 206 / Content-Range 正确，异常代理也可能不断发送超长 body；原实现会持续写磁盘，直到 EOF 才检查。现在每块写入前累计实际字节，超过请求长度立即拒绝，异常时删除该片 `.tmp`，最多重试 5 次。正确片和断点恢复逻辑保留。
3. **P2：退出 0 就接受无效/失败成功报告。** JobManager 原来对几何任务接受 JSON `null`、列表甚至 `status=failed` 的 report 并记为 succeeded。现在报告必须是对象且 `status=succeeded`，否则任务 failed，输入保留。
4. **P2：损坏诊断覆盖真实退出错误。** 子阶段或 JobManager 读取 `failure.json` 为 null/列表/坏 JSON/空 error 时，会抛解析/属性错误或显示 None，掩盖真实退出码。现在仅使用非空字符串 error；诊断不可读时回退到真实进程退出码与日志指引，原文件保持可诊断。

修前证据：`data/qa/v031-second-agent-before.xml`，**7 tests / 7 failures**，覆盖前 3 类问题的 7 个实例。修后同一批为 `v031-second-agent-first-after.xml`，**7 passed**。

追加诊断问题修前证据：`v031-second-agent-diagnostic-before.xml`，**18 tests / 4 failures / 14 passed**，4 个失败分别是子阶段诊断 null/list/坏 JSON/空 error；修复后另补真实 JobManager 短进程的 4 个同类实例。扩展新增文件最终 **22 passed**，见 `v031-second-agent-expanded-after.xml`。

## 20 轮执行证据

| 轮 | 检查目标 | 执行与结果 |
|---|---|---|
| 01 | 确认审查基线和协作边界 | `git status --short`、`git rev-parse --short HEAD`、查根与目标子目录 AGENTS，阅读上一轮报告；未触碰原有未跟踪资料与运行中的发布包 |
| 02 | 三入口只需要本入口模型 | 重新运行 `test_modes_require_only_their_declared_model_components`、图片无 mesh 和已有模型缺 mesh 参数实例；保持图片可独立进入、贴图不依赖 Hi3D |
| 03 | 默认跳过 Hi3D 与旧精修路径兼容 | 运行 Supervisor 顺序、默认图片粗模直贴图、纯几何跳过贴图、旧 refine 省略新 flag 等实例；最终贴图传入实际最终网格 |
| 04 | 后续阶段失败保留早期模型 | 运行灰模不能冒充贴图成功、失败图片/贴图重用实例；保留 coarse/refined、原始图片/网格字节，未把 CPU fixture 冒充模型推理 |
| 05 | 阶段错误预检尽早拒绝 | 运行 7 类请求预检与无 CUDA fixture；不加载模型，failed 诊断明确，已有结果字节不变 |
| 06 | alpha/keep/空透明前景边界 | 运行有效 alpha 和 keep 禁用 ONNX 导入、全透明拒绝实例；现有行为未退化，无真实 ONNX 或 GPU 运行 |
| 07 | 固定源码变更与越界导入 | 重新运行变更源码和 `../outside.py` source-manifest 实例；均在加入 sys.path 之前拒绝 |
| 08 | 成功阶段的发布顺序与旧失败清除 | 新 `test_stage_success_only_publishes_after_synchronization_and_clears_stale_failure`；同步 fixture 执行时旧结果仍原样且 partial 已存在，成功后发布并清除旧 failure |
| 09 | CUDA 同步失败不得覆盖旧 GLB | 新 `test_stage_failure_before_publication_preserves_previous_output[synchronize]`；修前失败、修后通过，旧结果字节与失败原因完整 |
| 10 | 显存统计异常不得覆盖旧 GLB | 同一参数测试 `[metrics]`；修前失败、修后通过，诊断保持原统计异常，没有 partial 残留 |
| 11 | 成功报告写盘失败不得覆盖旧 GLB | 同一参数测试 `[report]`；修前失败、修后通过，失败报告可记录而旧模型保持不变 |
| 12 | 最终发布受阻和短输出保持失败 | 新 `test_stage_publish_failure_keeps_old_deliverable_and_records_failure`、`test_short_engine_output_does_not_replace_previous_deliverable`；模拟结果被其他应用占用和短 GLB，旧文件保留，failed 报告正确，临时结果清理 |
| 13 | exit 0 不能给坏报告背书 | 新 `test_exit_zero_with_invalid_or_failed_report_is_not_job_success` 3 实例；真实短子进程写 null/list/failed 报告，修前记成功、修后正确失败且原输入保留 |
| 14 | 无法读取失败诊断时显示真实退出码 | 新 child 与 job 两组各 4 实例；真实短子进程 exit 7，null/list/坏 JSON/空 error 均回退 exit 7，不再显示属性异常或 None |
| 15 | 超长 206 响应必须有限写入与有限重试 | 新 `test_range_download_stops_before_writing_an_oversized_response`；修前继续读到 fixture 断言失败，修后首块超长立即拒绝且不继续消费，每次坏临时片删除，5 次有限尝试，无最终文件 |
| 16 | 中断和截断可恢复到正确固定字节 | 新 `test_range_download_retries_truncated_or_timed_out_response_then_recovers`；本地传输第一次 Timeout、第二次截断、第三次完整；退避 1/2 秒用收集器代替等待，最终 SHA 正确，无 .tmp/parts 残留 |
| 17 | 多片断点只补缺片并校验整文件 | 新 `test_range_resume_fetches_only_missing_ranges_and_verifies_assembled_bytes`；临时 4MiB 已验收尺寸片 + 最后短片，仅请求缺失 Range，组装字节与 SHA 正确，成功释放 parts |
| 18 | 元数据失败不破坏旧清单或持有锁 | 新 `test_metadata_download_failure_keeps_manifest_and_releases_crossprocess_lock` 的 auth/missing-file 两实例；fixture 403/缺目标文件失败后 manifest 原字节不变，状态 failed，FileLock 能重新获取；未向 Hub 发请求 |
| 19 | 跨进程校验/恢复不误杀未知工作进程 | 运行上一轮真实 FileLock 进程、stage recovery 4 身份实例、Supervisor 退出后 5 身份实例、JobManager 状态写盘/取消/启动恢复实例；只结束 PID+时间+请求匹配的进程，身份不明保持 GPU 入口阻塞，所有测试子进程 finally 收尾 |
| 20 | 全部目标模块回归与保护范围冻结 | 最终目标套件 **121 passed / 0 failed / 0 skipped，49.49s**；`git diff --check` 通过（仅平台 LF→CRLF 提示）；ComfyUI 4 个源/画布文件 diff 无输出。四个修复文件与新测试已冻结，等待根 Agent 全套/打包审计 |

最终命令：

```text
.appenv/Scripts/python.exe -m pytest tests/test_pipeline_second_audit.py tests/test_pipeline_engine_audit.py tests/test_asset_pipeline.py tests/test_job_lifecycle.py -q --junitxml=data/qa/v031-second-agent-final.xml
121 passed, 1 warning in 49.49s
```

唯一 warning 为现有 Starlette/AnyIO 别名弃用，未出现新 warning 或 skip。

## 未验证范围

- 本轮是错误处理与生命周期复查，没有重新推理、下载真实权重或证明新的画质/显存能力；原 v0.3.0/0.3.1 的真实 GPU 收据仍为原执行时间。
- CUDA 同步/显存错误由明确 CPU torch fixture 注入，不能当成真实硬件故障实验。传输测试是本地 response fixture，不保证所有代理/限流组合。
- 没有更改采样数学、权重、固定上游源码、CUDA 扩展或 ComfyUI 节点接口。
- 最终归档结果见外部 `dist/RELEASE-VERIFIED-0.3.2.json`；本报告不为旧压缩包背书。

## 主 Agent 交叉检查

- 新增失败导出清理回归：几何/真实内嵌 PBR 两种 GLB，分别在转换写入与 dimensions.json 落盘时注入失败。源模型与已有导出字节不变，本次半成品被删除。几何两实例修前实际失败，修后四实例通过。
- 启动真实临时 HTTP 子进程，指定随机空闲 `--port`。修前仍返回 8378；修后返回真实监听端口，伪造 Host 不影响端口报告，用户 config.local.json 字节不变。
- 完整套件 **207 passed / 0 failed / 0 skipped，58.17 秒**；唯一 warning 为已有 Starlette/AnyIO 别名弃用。见 `v032-all-tests.xml`。
- 浏览器故障 9 组、三入口/模型依赖 6 组均通过，零 JavaScript 错误。使用独立 8384 临时服务与明确 HTTP 夹具，不执行 GPU 推理。见 `v032-browser-faults.json`、`v032-browser-modes.json`。
- ComfyUI 四个源/画布文件相对 14c5aaa 字节未变。本轮修改失败处理与发布顺序，沿用原真实 GPU 产物验收，不把 CPU/HTTP 夹具算作新的建模结果。
- v0.3.2 完整目录与归档独立生成，保留旧目录。归档可复用已验证运行库的压缩数据；最终仍执行全部 CRC、全部自有文件、核心/扩展模型哈希检查，结果以外部归档审计为准。
