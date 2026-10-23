# 验证记录

日期：2026-10-23；设备为 RTX 5090 Laptop 约 24GB，系统内存约 64GB。

当前发行 **v0.2.1** 完成主 / 子 Agent 20轮交叉检查，过程见 [audit20-rounds.md](validation/audit20-rounds.md)。全套 **70项 pytest通过**；新增9组浏览器审计、原11组浏览器回归与实际蒙版绘制均通过，零JavaScript错误。浏览器测试还验证HTTP500、请求超时、残留进程阻塞、记录404/400恢复。worker测试仅使用临时假脚本，真实Hi3D推理另以 `v021-ui-inference.json` 和 `v021-inference-report.json` 记录，不混用两类证据。

第20轮发行审计已扩大到全部自有代码和节点包内容：CRC有效但jobs.py / ComfyUI节点陈旧也会拒绝。完整归档结果见发行目录 `RELEASE-VERIFIED-0.2.1.json`。固定revision的11个Git配置 / 通知文件只读核对见 `audit20-model-provenance.json`。此轮没有更改官方源码、权重与采样数学。

| 项目 | 当前证据 | 状态 |
|---|---|---|
| API/工件/任务/导出 | 当前70项pytest；含原21项以及异常worker/取消/重启/损坏记录、线程/磁盘异常、Git配置校验、模型JSON恢复、ComfyUI非法数据、上传事件循环和归档源文件检查 | 通过 |
| v0.2.0 UI 回归 | 新版独立 runtime 的真实 Chrome：粉白/署名、底部操作、跨两次轮询的提交锁、空步数、断线重连、实际历史日志/条件图/复用、失败任务导出、查看历史时停止活动任务、样例配对失败恢复、会话过期、同文件重选、800/390/375px布局 | 通过；失败与活动状态用明确的 UI 响应模拟，不计作模型推理 |
| v0.2.0 新版界面真实推理 | 包内 runtime，从新版底部按钮提交 512 / 25 步 / seed 0 / 卸载 / SDPA：247.54 秒、allocated 1722MiB；推理时 GPU 诊断 409；界面导出 GLB 80mm → 0.0799999982m，原结果 SHA 不变，零 JS 错误 | 通过 |
| v0.2.1 修复后真实推理 | 新版独立runtime与任务管理，实际界面512 / 25步 / seed 0 / 卸载 / SDPA：203.85秒、allocated 1722MiB；80mm GLB实际0.0799999982m，原结果未改变，零JS错误 | 通过；耗时差异不作为速度提升声明 |
| 最终包 API 与尺寸 | 独立运行时实际导出 GLB：100mm → 0.10000000149m；原粗模 SHA256 不变 | 通过 |
| 真实浏览器 | Chrome＋Playwright：GLB 加载、桌面/平板布局、透明蒙版实际绘制与保存；零 JS 错误、零外部 HTTP 请求 | 通过 |
| GPU 内核 | 原生 Triton 体素化、spconv、官方稀疏卷积 float32/float16、融合 SDPA | 通过 |
| Attention 数值 | 22 头、head_dim=96，GPU FP16 与独立 FP32 参考；两组最大绝对误差约 0.00050–0.00054；cross-attention 有限 | 通过小规模算子检查 |
| 独立 Windows runtime | 清除宿主 Python/CUDA 路径，禁用 HF 联网，使用全新 Triton 缓存运行 GPU 探针 | 通过 |
| 本机代理隔离 | 设置不可达代理、清空 NO_PROXY 后，Comfy 客户端直连本机后端成功 | 通过 |
| ComfyUI 网格桥接 | ComfyUI 0.34.0，Backend→LoadMesh→原生 SaveGLB；真实引擎成功，包围盒一致 | 通过 |
| 完整 Hi3D 512/25 步 | 真实 ComfyUI 引擎，234.47 秒；张量 allocated 峰值 1722MiB，reserved 1798MiB | 通过 |
| 完整 Hi3D 1024 | 真实工作台双轮推理，950.74 秒；allocated 8033MiB，reserved 8458MiB | 通过 |
| 完整 Hi3D ComfyUI 工作流 | Hi3DRefine → 原生 SaveGLB，512 / 25 步 / seed=0 / CPU 卸载 / SDPA | 通过 |
| 发布前画布保存与执行 | ComfyUI 0.34.0 / 前端 1.53.6：5节点6连线，加载画布、前端序列化保存、重新加载，再从该图执行真实512 / 25步；273.89秒，GLB 18007536字节，零JS错误 | 通过；发布示例仅含画布格式，见validation/comfy-canvas-result.json |
| 独立 Windows 模型包 | 包内解释器，HF 离线，移除宿主 Python/CUDA 路径，不可达代理；完整 512 推理 242.18 秒、allocated 1722MiB | 通过 |
| 独立 Windows 1536 | 三轮完整离线推理，2390.10 秒；allocated 15555MiB，reserved 17906MiB；2000000 面 | 通过 |

显存数值为 PyTorch 张量分配/保留峰值，不包含 CUDA 上下文、驱动和其他进程；不能据此认定 2GB/8GB GPU 可用。阶段 RSS 是阶段结束时采样，不是全程系统内存峰值。算子探针、模拟任务控制和网格桥接不计为完整模型推理。8/12/16GB 均没有真实设备验收；1536 只完成本次流程样例，不能推出其他输入或设备均可用。Windows SDPA 是显式兼容移植，不是官方测量环境。

512 输出 999999 面、9 个连通件、15 条边界边、11 条非流形边；1024 输出 1000000 面、34 个连通件、162 条边界边、42 条非流形边。1536 输出 2000000 面、73 个连通件、276 条边界边、83 条非流形边、2 个退化面。三档结果均不封闭，未检测自交、壁厚、强度或切片可打印性。输入为随包自制流程样例，不是美术或精度基准。运行通过不代表几何增益或可打印保证。

独立包的完整推理使用包内解释器和权重，HF/Transformers 离线、代理指向不可达端口，环境中移除宿主 Python/CUDA 路径。这验证当前机器上的独立运行性，不等于所有 Windows 机器兼容。Windows SDPA 未与官方 Linux FlashAttention 做同输入完整输出对照。Linux 安装路径尚未实机验收。

随包报告与截图保存在 validation/；开发机完整证据保存在 data/qa 与 data/jobs。

v0.2.0 测试修正过一个导入顺序造成的隔离问题：测试模块提前缓存路径，旧测试误写了开发目录模型文件。已从先前验收的完整包恢复清单与对应文件，并对全部 16 个文件真实 SHA256 逐一确认一致。新增测试改为 fixture 内延迟导入，正反收集顺序均为 21 passed，修正后的真实文件身份与清单 SHA 前后不变。进一步增加导入路径保护与损坏模型测试的独立 fixture；预先导入生产模块时，会在收集阶段明确报错并停止（test-import-guard.json），不会执行测试。仅测试创建的目录可恢复地移入 data/test-isolation-recovery；原完整包未受影响。恢复证据及隔离结果分别为 model-restoration-2026-10-23.json、test-isolation-result.json。发行包权重还会独立复核。
