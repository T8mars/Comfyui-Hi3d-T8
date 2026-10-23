# 随包验证证据

v0.2.2 新一轮20轮检查见 `postpublish-audit20.md`，后端与前端分别有独立记录。`postpublish-ui-baseline.json` 和 `postpublish-ui-result.json` 是四组新增浏览器基线/修复回归；`postpublish-existing-ui-result.json` 保留原九组回归。`postpublish-mask-result.png` 是实际蒙版画布，`postpublish-wheel-contract.json` 验证标准源码构建与独立导入，`postpublish-geometry-proof.json` 记录异常坐标原行为与明确拒绝。新真实推理及最终测试回执另列，不把夹具检查计作模型推理。

发布前新增 `comfy-canvas-result.json`：实际 ComfyUI 0.34.0 / 前端1.53.6，5节点6连线画布加载、保存、重新加载后执行真实512 / 25步推理，273.89秒，GLB18007536字节，零JavaScript错误。`examples/Hi3D-Refine.json` 使用前端实际保存的画布结构，发布包不提供API格式示例。`huggingface-publication.json` 记录上传提交及远端23文件校验，含16个与离线包相同的模型工件。

v0.2.1 的20轮检查见 `audit20-rounds.md` 与 `audit20-backend-rounds-01-10.md`。`audit20-browser-baseline.json` 是修复前隔离复现，`audit20-browser-result.json` 是修复后第11–19轮回归；`audit20-existing-browser-regression.json` 和 `audit20-browser-basic.json` 是新版运行时上的既有流程 / 蒙版测试。上述故障响应和临时worker均不代表模型推理。`audit20-model-provenance.json` 核对11个官方固定版本Git配置 / 通知文件。

新版独立运行时的实际512精修、导出与截图为 `v021-ui-inference.json`、`v021-inference-report.json`、`pink-v021-real-512.png`；模型自动校验为 `portable-model-verification-v0.2.1.json`。旧版实测证据保留并按版本区分。

这些是 2026-10-23 在 Windows / RTX 5090 Laptop 24GB / 64GB RAM 上生成的实际报告，不是模拟推理。环境与固定版本见各 JSON。

v0.2.0 新增 `browser-regression-result.json`、`pink-workbench-desktop.png`、`pink-workbench-390.png`：实际 Chrome 页面与交互回归。报告明确标注的故障/活动响应是 UI 测试夹具，不计作模型推理；历史预览、日志、条件图和输入复用使用实际完成任务。`test-isolation-result.json` 与 `model-restoration-2026-10-23.json` 记录隔离问题修复与全部文件恢复校验。

`v020-ui-inference.json`、`v020-inference-report.json`、`pink-v020-real-512.png`：本次更新后从真实界面提交的一次完整 512 精修与 80mm GLB 导出。`portable-model-verification-v0.2.0.json` 为新目录自动校验成功记录。`test-import-guard.json` 证明错误的提前导入会安全中止测试。

- `windows-comfy-512.json`：真实 ComfyUI 引擎中的 Hi3DRefine → SaveGLB。
- `windows-workbench-1024.json`：真实工作台双轮 Hi3D 推理。
- `offline-inference-512.json`：独立发行目录；移除宿主 Python/CUDA 路径，HF/Transformers 离线，代理指向不可达端口。
- `offline-inference-1536.json`：同一完整离线包，1536 / 三轮 / 每轮 25 步，最终输出 200 万面。
- `portable-model-verification.json`：独立包内全部必需权重的 SHA256 校验。
- `attention-numerics.json`：FP16 SDPA 与独立 FP32 参考的小规模算子误差。
- `portable-clean-probe.json`：独立运行时在全新 Triton 缓存中的 GPU 内核探针。
- `browser-result.json`、`workbench-1024.png`、`workbench-1536.png`：实际浏览器检查与双侧预览；本次流程没有外部 HTTP 请求。
- `final-runtime-diagnostics.json`、`final-api-export.json`：最终发行目录的 GPU 内核诊断与真实 GLB 米制尺寸导出，原粗模未改写。

显存数值为 PyTorch 张量分配峰值，不包含 CUDA 上下文、驱动与其他进程。每阶段 RSS 是该阶段结束时采样的进程内存，不是全程系统内存峰值。

输入是随包自制流程样例，不是视觉质量或几何精度基准。完整运行成功不代表生成结果封闭、可打印或比粗模更准确；报告保留边界边和非流形边等实际问题。Windows 使用明确标识的 SDPA 兼容路径，尚未在同一输入上对比官方 Linux FlashAttention 的完整输出。
