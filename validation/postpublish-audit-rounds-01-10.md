# 发布后联合审计：第 1–10 轮（后端 / ComfyUI）

日期：2026-10-23。检查对象是 0.2.1 发布后的工作区；修复由主 Agent 集成到 0.2.2。所有故障注入仅使用临时文件、模拟张量或本机临时 HTTP 服务，没有写入生产模型、历史任务或其他 ComfyUI 工程。

| 轮次 | 检查与真实证据 | 结论 |
|---|---|---|
| 1 | 源码及已发布 Registry 14 文件归档分别在 `python -I` 子进程中导入入口，明确阻断 numpy / PIL / torch / trimesh / FastAPI / Comfy 导入。均发现三个节点。 | 通过；节点发现不依赖推理环境。旧归档仅核对发现契约，不声称含本次修复。 |
| 2 | 读取实际 `examples/Hi3D-Refine.json`，核对 0.4 画布格式、5 节点、6 条类型与端口均一致的双向连线，以及 Refine 控件值。检查发布归档不含 API 工作流。 | 通过；仍是用户要求的画布工作流。localized_name 的 UTF-8 文本没有 U+FFFD，控制台乱码不是文件损坏。 |
| 3 | 捕获 Refine 实际准备上传的 PNG 字节：RGBA 原始 alpha `[[0,128],[255,255]]`，连接全零 MASK 后原实现变为全 255。修复后逐像素保持原 alpha，RGB MASK 的反向含义不变；相同软 alpha 不重复叠乘。 | **已修复**：图像 alpha 与 MASK 可见区域取交集，MASK 不会恢复透明背景。 |
| 4 | 在任何上传前注入非法通道、图像 NaN、蒙版 NaN 与四维蒙版，确认既不产生上传，也不触发模型卸载或任务提交。 | **已修复 / 加固**：明确拒绝非法图像 / MASK；不再把 NaN 隐式转换为黑色像素。 |
| 5 | 第一次上传完成时设置 Comfy 中断标记。原实现仍先卸载 Comfy 模型，再创建远端任务，随后才取消；修复后分别测试上传前、图像上传后、网格上传后三个窗口。 | **已修复**：提交前及时响应中断；没有远端任务就不创建 / 取消任务，也不卸载 Comfy 模型。 |
| 6 | 远端任务已创建后才触发中断，模拟取消端点失败，验证确实尝试取消该 job，最终仍向 Comfy 传播原始中断异常。成功路径捕获 PNG 与 NPZ，核对输出网格、报告与 job ID。 | 通过；中断异常不会被取消网络错误覆盖。 |
| 7 | 真实 `ThreadingHTTPServer` 接收 GET、上传 POST、取消 POST，并核对会话头；客户端实际使用超时分别为 15 / 600 / 10 秒。另让两个端点发送响应头后不发送响应体，将测试阈值压缩至 0.1 秒，实际发生 TimeoutError。 | **已修复**：轮询 / 取消不再共同等待 600 秒；保持上传长超时。该验证包含真实回环 HTTP，不只是 mock。 |
| 8 | 重新执行任务生命周期测试：真实临时子进程、PID / 创建时间 / request 路径身份校验、日志及磁盘写入失败、取消、连续重启、终态残留进程。 | 通过；旧修复未回退，未操作生产进程。 |
| 9 | 重新执行模型状态测试：同尺寸且恢复 mtime 的损坏、文件替换、校验中变更、清单 repo / revision 不符、损坏 metadata 及原字节备份。 | 通过；快照变化不能沿用旧 ready 结论，轮询不重新散列大模型。 |
| 10 | 重新执行模型下载边界：真实另一个进程持下载锁、损坏区间缓存清理、非 LFS Git blob 校验、强制重传修复与完成后的校验刷新。复审服务级 verify / download 检查窗口，并交由主 Agent 在 server.py 统一加锁。 | 下载 / 快照边界通过；服务层锁问题由主 Agent 修复和验证，本分支未擅自扩大跨 CLI 锁范围。 |

## 修复与验证

修改 `comfy_nodes.py`；新增 `tests/test_postpublish_backend.py` 的 15 项隔离测试。前两项实际复现原代码时为 **2 failed / 1 passed**：透明背景丢失，以及上传后中断仍提交任务。修复后新增测试全部通过。

组合回归命令：

```powershell
.appenv\Scripts\python.exe -m pytest tests\test_postpublish_backend.py tests\test_job_lifecycle.py tests\test_models_state.py tests\test_comfy_contract.py -q
```

结果：**64 passed**（其中新增 15 项）。最终整套测试和新版本发行校验由主 Agent 汇总。

发现与画布契约回执：[postpublish-backend-contracts.json](postpublish-backend-contracts.json)。本子审计没有新增 GPU 推理测量；主 Agent 负责修复集成后的画布与发行包验证。

## 追加交叉复审：诊断超时与父进程早退

主 Agent 将服务诊断修复交给本子 Agent 继续处理。只读复审发现新的可复现窗口：Windows 临时启动器创建继承 stdout 的子进程后立即退出，原 `run_probe()` 在第二次 `communicate(timeout=5)` 后抛出 `TimeoutExpired`，实测耗时 **5.25 秒**，子进程仍存活。此前按父进程 children 列表回收不能覆盖已脱离的子进程。

现已在 `server.py` 修复：每次探测通过 Popen 的独立环境副本传入随机 `HI3D_DIAGNOSTIC_ID`，不修改全局 `os.environ`；超时按标记精确匹配及创建时间识别继承标记的进程。两次回收与有界输出读取后仍未退出的已确认进程，以 PID＋创建时间保留在 GPU 阻塞状态中，由现有 health 字段显示，真实退出后自动解除。身份不符的复用 PID 不会保留为阻塞项。

`tests/test_postpublish_service.py` 的诊断测试使用独立临时脚本路径；finally 仅清理匹配精确路径、创建时间且再次核对身份的临时进程。实际覆盖父启动器存活、父启动器提前退出、注入 AccessDenied 后保持 GPU 阻塞及真实退出后恢复，以及诊断进程打印 passed 但退出码为 3 时仍报告失败。

该服务回归结果：**9 passed，15.76 秒**（包含主 Agent 原有下载互斥、几何边界测试；只有依赖自身的弃用警告）。所有临时子进程已回收，未运行 GPU 推理。源码与服务测试已冻结，交主 Agent 做整套集成验证。
