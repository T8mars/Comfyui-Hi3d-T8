# 发布后第 14–17 轮：独立前端复现与修复

检查日期：2026-10-23。基线为提交 `10f1c4187f6f2ac09b3e81d1ae80c23ae9b7144d` 的工作台前端。由独立子 Agent 在真实 Chrome 中先复现，再修改 `hi3d_workbench/web/app.js` 并回归。四轮均发现了可复现缺陷。

| 轮次 | 修复前证据 | 修复与验证 |
| --- | --- | --- |
| 14 | 成功记录有 `refined.glb` 但无报告时，精修预览 `src` 为空；`geometry={}` 时出现 `Cannot read properties of undefined (reading 'toLocaleString')`。 | 预览/导出以实际工件为准；缺失统计显示 `—`，几何报告不完整时明确提示，避免显示检查通过。无报告、空几何、完整报告三种情况均真实加载 GLB，导出按钮可用；完整报告仍显示 1,280 面。 |
| 15 | `worker.log` 请求 abort 导致整站离线并禁用导出；404/503 只弹错误却不更新日志区；已换为新任务后返回的旧日志 abort 仍令全站离线。 | 读取失败局限日志区，保留预览/导出，轮询自动重试；失败与成功均检查所选任务 epoch。abort/404/503 各自恢复后能读到日志；过期 abort 不覆盖新任务、不改连接状态。 |
| 16 | 两个历史列表请求倒序返回，旧 503 覆盖新成功列表并弹过期错误；旧请求结束会提前解锁仍待完成的新请求。 | 对 catch/finally 同样检查 `historyEpoch`。新列表保留、无旧错误提示，新请求未完成时刷新按钮保持锁定。 |
| 17 | 快速拖动恢复笔刷，仅恢复两个端点，中点 alpha 为 0（原始 96）；擦除笔刷单击无效，alpha 仍为 96。 | 恢复沿前后两点的连续胶囊区域重绘原始 RGBA；擦除单击显式画圆；丢失指针捕获时结束笔画。最终 alpha `[255,96,0,96]` 与原图逐点一致，原有透明像素保持 0；擦除单击为 0，恢复单击为 96，pointercancel 后悬停未继续绘制。 |

## 证据与复现

- `tools/browser_postpublish_ui.py`：本轮单组独立回归脚本；参数 `--baseline` 用于修复前记录失败结果，不将失败伪装为通过。
- `data/qa/postpublish-ui-baseline.json`：修改前的真实 Chrome 基线，四轮均失败。
- `data/qa/postpublish-ui-result.json`：最终四轮均通过，JavaScript 错误 0。
- `data/qa/audit20-browser-result.json`：同一隔离服务上对上一轮九组前端故障回归再次验证，九组全过，JavaScript 错误 0；这些是既有回归，不重复计入本轮 20 轮编号。
- `data/qa/postpublish-mask-baseline.png`、`data/qa/postpublish-mask-result.png`：实际蒙版画布截图。最终测试另加入原始透明像素断言，基线的三个采样点与最终的四个点数量因此不同。

运行方式（仓库根目录）：

```powershell
.\.appenv\Scripts\python.exe tools/serve_ui_audit.py --port 8379
.\.appenv\Scripts\python.exe tools/browser_postpublish_ui.py
```

测试使用 8379 上的临时数据目录及明确的 HTTP 故障夹具，蒙版上传与 canvas 操作真实执行。生产 8378 仅只读 `/api/health`，未上传、导出、启动推理或更改生产数据；夹具模型不作为效果或推理验收证据。

冻结前端 SHA256：`dd1d736f374380703155ba291235e811277a2de4b3a4679755fcb4c755c47c1b`。

本子任务只修改前端、上述回归脚本和本文；未改变 ComfyUI 画布工作流、模型权重、官方推理代码，也未发布或打包。
