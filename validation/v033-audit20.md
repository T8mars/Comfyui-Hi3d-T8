# Hi3D v0.3.3 · 第三次 20 轮联合审查

- 日期：2026-10-24（America/New_York）；基线 Git `b732730`，版本 0.3.2。
- 开始时仅有原有未跟踪 `IMPLEMENTATION_STATUS.md`、`research/`，两处未修改。没有发现适用的 `E:/AGENTS.md`、项目根或目标目录 AGENTS。
- 已阅读 `validation/post-v03-audit20.md` 和 `validation/v032-audit20.md`，本轮优先补充任务建立事务、Supervisor 子阶段成功契约、诊断再次失败时的清理边界。
- 修改 `hi3d_workbench/jobs.py`、`pipeline_worker.py`、`engines/run_stage.py`；新增 `tests/test_pipeline_third_audit.py`。`tests/test_asset_pipeline.py` 只给原成功阶段 fixture 增加明确的 `status=succeeded`，原断言未改。
- 没有连接或重启 8378 用户服务，没有下载真实模型、启动 GPU 推理、修改权重/固定上游源码/采样算法/ComfyUI 接口，没有提交、推送或打包。所有新增测试为隔离临时目录和明确 CPU fixture；既有生命周期回归启动的短 Python 进程由其 finally 清理。

## 已确认并修复

1. **P2：建立任务失败会留下没有发布的 UUID 目录。** 图片/粗模复制、请求或初始状态落盘失败，都发生在工作线程启动之前，原代码仍留下已复制的大文件或半成品历史。现在只清理本次新建目录；递归删除前检查它不是符号链接且 resolve 后父目录仍为 jobs 根目录。旧任务、上传源文件、线程启动后已有失败历史均保持原行为。
2. **P2：Supervisor 可接受 failed 或缺失状态的子阶段报告。** 原代码对 extension 只要求报告是 dict，对 Hi3D 只要求 dict 和文件存在，子进程 exit 0 后可能把 failed/缺报告的阶段记为全链路成功。现在 coarse/texture/refine 都必须明确 `status=succeeded`，否则在向任务顶层复制结果前拒绝。贴图例使用真实可解码 UV/PBR 小 GLB，证明不是灰模校验偶然挡住错误。
3. **P2：失败诊断再次写盘失败会跳过 partial 清理。** CUDA/推理错误进入 except 后，如果成功报告路径或 failure.json 又遇磁盘错误，原先位于后面的 unlink 永远不会执行。现在诊断与失败事件放在 try 中，partial 删除在 finally；BrokenPipeError 同样不能跳过清理。旧交付文件字节不变，写盘错误仍暴露给进程日志。

修前有效回归：`v033-agent-before.xml`，15 tests，**14 failed / 1 passed**。一个原本已通过的实例是 Hi3D 完全缺少报告，既有代码已会拒绝；其余 14 个失败覆盖以上三类问题。最初一次测试收集因导入顺序触发既有隔离保护，修正测试导入顺序后才记录此有效修前结果，未将收集错误当成产品 BUG。

同一批修后：`v033-agent-first-after.xml`，**15 passed**。随后增加 9 个新边界实例，新增文件共 24 个实例；目标套件最终 **145 passed / 0 failed / 0 skipped，44.86 秒**，见 `v033-agent-final.xml`。唯一 warning 是既有 Starlette/AnyIO 别名弃用。

## 20 轮目标、执行证据和结论

| 轮 | 目标 | 实际执行与结论 |
|---|---|---|
| 01 | 确认基线、前两轮覆盖与受保护范围 | 读取两份旧审计、git 状态/HEAD、查 AGENTS。没有改原有未跟踪资料或用户正在运行的发行包 |
| 02 | 复制图片或粗模失败不得留下未发布任务 | 新 `test_failed_job_setup_discards_only_its_unpublished_folder[input.png/coarse.glb]`，真实复制后注入错误。修前两例失败，修后目录仅剩原有任务，源文件和旧输出字节不变，线程未启动 |
| 03 | 请求/初始状态写入失败的建立事务 | 同测试 `[request.json/state.json]`，真实 JSON 写入后注入错误。修前失败，修后无新 UUID 残留、active/process/memory 未占用 |
| 04 | 粗模退出 0 仍须有成功报告 | 新 Supervisor 测试 coarse 的缺文件、空 dict、failed 三例。修前全链路记成功，修后全部在向顶层复制 coarse 前拒绝 |
| 05 | 真正 PBR GLB 也不能替失败报告背书 | 同测试 texture 三例，输出使用真实 UV 与嵌入图片 PBR fixture。修前可完成，修后报告错误优先拒绝，顶层 textured.glb 不发布 |
| 06 | Hi3D 阶段也遵守明确成功契约 | 同测试 refine 三例。完全缺报告原已拒绝，空 dict/failed 修前错误接受；修后全拒绝，原始粗模/图片保留 |
| 07 | 阶段失败报告落盘错误不能跳过 partial 清理 | 新 `test_partial_result_removed_even_when_failure_diagnostic_cannot_be_saved[result.stage.json]`。同步错误后再注入诊断磁盘满，修前 partial 残留、修后删除，旧结果不变 |
| 08 | failure.json 落盘错误同样清理 | 同测试 `[failure.json]`。第一个失败报告成功、第二个写失败，仍移除 partial，保留真实诊断异常 |
| 09 | 失败事件管道断开不能跳过清理 | 新 `test_failure_emit_cannot_prevent_partial_cleanup`，同步错误后 `_failed` emit 注入 BrokenPipeError。finally 清理通过，原错误已写入 failure.json，旧 GLB 不变 |
| 10 | Windows 原生扩展硬件门槛在加载前拒绝 | 新 architecture fixture 的 CC 7.5 / 8.0 两例。foreground 被替换为禁止调用，仍获得明确 8.6 门槛诊断，未加载模型或触发 GPU |
| 11 | 损坏图片不应进入模型调用 | 新 `test_corrupt_reference_image_fails_before_calling_model`，真实 PIL 解码非图片字节失败，模型函数禁止调用，旧结果保留，failed 诊断与无 partial 验证通过 |
| 12 | 已有固定内容文件无需联网或读取密钥 | 新 `test_valid_pinned_download_destination_is_used_without_network`，真实哈希匹配，requests/get_token 均禁止调用，直接复用且字节相同 |
| 13 | 206 的总长度不符时不能消费响应体 | 新 `test_range_response_with_wrong_total_is_rejected_before_body_read`。5 次有限重试均在 iter_content 前拒绝，响应均关闭，无最终文件或 tmp |
| 14 | 大权重 SHA256 和小配置 Git blob 校验 | 新 `test_verification_checks_real_blob_size_hash_and_signature[sha256/git-sha1]`，独立 hashlib 构造期望，正确内容通过，错误大小和同大小错误内容拒绝 |
| 15 | 在线程启动前取消不能生成工作进程 | 新 `test_cancel_before_worker_launch_never_spawns_a_process`，Thread.start 延后、Popen 禁止调用；主动执行延后调度后任务 cancelled，GPU 槽和取消集合释放，输入完整 |
| 16 | 加强子阶段成功验证不破坏正常调度 | 运行原默认 coarse→texture、可选 refine、geometry-only、旧 refine 四类流程。成功 fixture 明确状态后通过；仍给 texture 实际最终几何，未改变模型算法 |
| 17 | 上轮报告/退出码修复仍成立 | 运行原 exit0+无效最终 report、真实短子进程 exit7+null/list/坏JSON/空error 诊断等实例。失败报告不能成为成功，真实退出码不被解析错误覆盖 |
| 18 | 哈希竞态、坏缓存与断点传输不退化 | 运行前两轮哈希前后同大小改写、force-download 损坏缓存恢复、Range超长/截断/超时/缺片恢复/坏片隔离测试，全部通过；这些传输是本地 fixture，未联网 |
| 19 | 恢复/取消/状态写失败不释放仍占用的 GPU 槽 | 运行所有既有 lifecycle 与 stage recovery 身份实例，真实短 Python 进程验证 PID+时间+request，只结束确证身份，未知身份保持阻塞；finally 清理属于测试的进程 |
| 20 | 最终目标回归和保护范围冻结 | 145 passed、0 failed、0 skipped；`git diff --check` 通过（仅平台 CRLF 提示）；ComfyUI 四个源/画布文件 diff 无输出。本子 Agent 修改已冻结 |

最终命令：

```text
.appenv/Scripts/python.exe -m pytest tests/test_pipeline_third_audit.py tests/test_pipeline_second_audit.py tests/test_pipeline_engine_audit.py tests/test_asset_pipeline.py tests/test_job_lifecycle.py -q --tb=line --junitxml=data/qa/v032-third-agent-final.xml
145 passed, 1 warning in 44.86s
```

## 验证界限

- 本轮没有新增真实 GPU 推理、真实模型下载或硬件显存实验，不能把 CPU/CUDA 能力 fixture 称作新的模型画质或兼容性证明。
- 未验证物理磁盘满且同时禁止删除、系统断电、所有代理和远端限流组合；若文件系统拒绝删除，新目录可能仍需用户修复权限后清理，原建立错误不会被忽略。
- 根 Agent 负责 server/UI/geometry/CLI/最终发行回归；本报告不为未重新审计的 ZIP 背书。当前20轮子 Agent 范围已结束并冻结，不再扩展。

## 主 Agent 交叉检查与第四类产品修复

- **P2：GLB float32 转换只检查整体坍缩，漏掉单轴尺寸丢失。** 一个边长为 1、平移至 X=1e8 的盒子原来能上传成功，但转为 GLB 后 X 尺寸为 0，产生 8 个退化面。现在逐轴比较转换前后的跨度，对原本非零却完全丢失的轴拒绝并提示居中/调整尺寸。真正平面和可以精确表示的偏移网格仍通过；不自动改变用户坐标。
- 修前原始 3 个实例为 **2 failed / 1 passed**，见 `v033-geometry-before.xml`。修后新增 GLB 场景变换上传路径，4 个实例均纳入完整套件：直接校验、NPZ HTTP 上传、GLB 场景 HTTP 上传、合法平面与可表示偏移正例。
- 修复两处测试环境依赖：导出回归自建真实 PBR 小模型，不再在服务导入后跨测试模块导入；浏览器模型状态夹具从自己的临时服务取基础响应并显式构造所需状态，不再要求用户 8378 已安装并加载模型。这两项是测试基础设施修复，不计入四类产品 BUG。
- 全量 **235 passed / 0 failed / 0 skipped，61.75 秒**；唯一 warning 为已有 Starlette/AnyIO 弃用。原始回执 `v033-all-tests.xml`。相对 v0.3.2 新增 28 个测试实例，所有旧断言保留。
- 独立临时 8386 服务完成浏览器故障 9 组和入口/模型依赖 6 组，均通过，零 JavaScript 错误。见 `v033-browser-faults.json`、`v033-browser-modes.json`。这些是明确 HTTP 夹具，不执行 GPU 模型推理。
- ComfyUI 的 `comfy_nodes.py`、`__init__.py`、`examples/Hi3D-Refine.json`、`examples/README.md` 与 b732730 内容一致（Git 工作区换行归一化后比较）。仍提供画布工作流，未增加图片建模或贴图节点。
- 本轮不新增 GPU 推理、画质或低显存兼容性结论。模型、上游源码、Windows 扩展和推理算法未变，真实生成证据仍来自原 v0.3.0 验收。
- 使用单独 v0.3.3 发行目录与归档，保留旧目录。复用原始不可变运行库和已压缩载荷后，仍对最终三个 ZIP 执行全量 CRC、自有源码、固定上游和模型哈希校验。最终可交付状态以外部 `dist/RELEASE-VERIFIED-0.3.3.json`、`dist/SHA256.json` 以及本机 `data/qa/v033-delivery.json` 为准；未审计的旧 ZIP 不代表新修复包。
