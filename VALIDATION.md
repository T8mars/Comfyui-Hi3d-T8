# 验证记录

**v0.4.0**：独立工作台增加三种建模和三种贴图引擎选择，ComfyUI 节点及画布工作流保持 Hi3D-only。当前证据和待验收项见[质量进度](validation/quality-upgrade-progress.json)，20 项独立复核见[发行审查](validation/quality-release-audit20.json)。真实推理、自动化夹具、浏览器操作和整合包独立运行验证分别记录。“历史 v0.2.2”节起的数据属于历史版本，不表示新增引擎均通过相同验证。

## v0.4.0 当前验收范围

| 检查 | 结果与范围 | 证据 |
|---|---|---|
| 自动化回归 | 276 项通过；含 GLB 原始法线、非均匀缩放、镜像、UV 重排、下载和校验期间文件变化等回归。CPU 夹具不计作模型推理 | [回归回执](validation/v040/regression.json) |
| 原生 UV 与法线对应 | 实际宝箱 961688 面经原生退化面清理为 961664 面，42.281 秒；恢复后位置和源着色法线精确对应。完全分离三角面另走 CPU Atlas | [HTTP 与 UV 记录](validation/v040/source-http.json) |
| 多引擎推理 | 三种建模、三种贴图均有本机真实运行记录；法线导入修正后的狐狸 UltraTex 2048 与 Hunyuan Paint 512 已完成新的真实 HTTP 推理，另核对原始 GLB 法线、UV、面索引及 80mm 导出 | [引擎回执](validation/v040/runtime-validation.json)、[几何对照](validation/v040/geometry-benchmarks.json) |
| Windows 包独立运行 | 包内解释器、清除宿主 Python/CUDA 路径、HF 离线与不可达外网代理条件下，完成 Mini 建模、TRELLIS 建模／PBR、已有自定义法线网格贴图及 Hi3D GPU 内核探针；本轮未重跑完整 Hi3D 推理 | [独立包回执](validation/v040/portable.json) |
| ComfyUI 画布 | 节点、5 节点 6 连线的可视化 JSON 与 Hi3D worker 未改变；沿用下方历史真实画布保存、重载和执行证据，本轮未新增其他模型节点 | [回归与范围](validation/v040/regression.json)、[历史画布回执](validation/v022-comfy-canvas-result.json) |
| 新增界面交互 | 本轮 CUA 未暴露可操作浏览器，尚未完成新版控件的浏览器交互验收；HTTP／静态检查和旧版浏览器通过记录不能替代此项 | [回归范围](validation/v040/regression.json) |

本轮修复了两类可造成“贴图看起来差”的法线问题：原生 UV 展开后的顶点对应错误，以及 GLB 场景展开／TRELLIS 网格复制时丢失文件原始法线。导入后按节点逆转置变换法线，并处理镜像绕序。旧结果若曾以错误法线作为贴图条件，应重新推理；单纯修正导出法线不会重算材质。模型质量判断以新结果的正背面和多灯光检查为准，不能将全部问题归因于贴图模型。

实测仅覆盖 Windows／RTX 5090 Laptop 24GB／64GB RAM。更小显存、其他设备和新版 UI 的未验收项保持明确记录，不从测试数量或显存张量峰值推导兼容性。

## 历史 v0.2.2

**v0.2.2** 的20轮审计见 [postpublish-audit20.md](validation/postpublish-audit20.md)，后端/节点与浏览器基线分开记录。该轮修复未改变官方源码、模型权重或采样数学；整套 **102项pytest通过**（46.34秒，1条依赖弃用警告），回执见 validation/postpublish-pytest.txt。浏览器新增四轮及原九组回归通过，零JavaScript错误；对应发行包真实推理单独记录。

**v0.2.2 实际集成验收**：新发行目录内的独立解释器与模型，ComfyUI 画布加载→前端保存→重载→运行成功。5节点6连线，512 / 25步 / seed 0 / CPU卸载 / SDPA；画布全过程339.17秒，worker报告313.36秒，原生SaveGLB输出18,007,616字节，零JavaScript错误。当前prompt_id与后端job_id逐一绑定，见 [画布回执](validation/v022-comfy-canvas-result.json)、[截图](validation/v022-comfy-canvas.png) 和 [推理报告](validation/v022-inference-report.json)。本次张量allocated / reserved峰值1722 / 1798MiB，不代表整卡显存要求；输出1000000面、9连通件、12边界边、9非流形边，仍不封闭。

同次推理运行期间诊断请求实际返回409；任务结束后重新运行工作台浏览器冒烟，模型加载、蒙版操作及800px布局通过，零JavaScript错误、零外部HTTP请求。一次并行冒烟因运行中按设计禁用输入而超时，未计为通过；空闲重跑才作为验收。6条历史任务已迁入新目录，旧目录保留；个人数据不进入发行ZIP。Registry官方Action成功，0.2.2安装端点/CDN均200，节点源码与发布Git提交逐字节一致，平台仍为Pending，见 [发布回执](validation/registry-publication-v0.2.2.json)。最终归档CRC、当前自有源码、61个官方Python文件与16个模型工件以 `dist/RELEASE-VERIFIED-0.2.2.json` 为准。

日期：2026-10-23；设备为 RTX 5090 Laptop 约 24GB，系统内存约 64GB。

上轮发行 **v0.2.1** 完成主 / 子 Agent 20轮交叉检查，过程见 [audit20-rounds.md](validation/audit20-rounds.md)。全套 **70项 pytest通过**；新增9组浏览器审计、原11组浏览器回归与实际蒙版绘制均通过，零JavaScript错误。浏览器测试还验证HTTP500、请求超时、残留进程阻塞、记录404/400恢复。worker测试仅使用临时假脚本，真实Hi3D推理另以 `v021-ui-inference.json` 和 `v021-inference-report.json` 记录，不混用两类证据。

第20轮发行审计已扩大到全部自有代码和节点包内容：CRC有效但jobs.py / ComfyUI节点陈旧也会拒绝。完整归档结果见发行目录 `RELEASE-VERIFIED-0.2.1.json`。固定revision的11个Git配置 / 通知文件只读核对见 `audit20-model-provenance.json`。此轮没有更改官方源码、权重与采样数学。

| 项目 | 当前证据 | 状态 |
|---|---|---|
| API/工件/任务/导出 | 当前102项pytest；覆盖任务恢复、模型校验、节点透明与中断、诊断进程清理、几何边界和归档内容等 | 通过 |
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

历史验收的512输出999999面、9个连通件、15条边界边、11条非流形边；1024输出1000000面、34个连通件、162条边界边、42条非流形边。1536输出2000000面、73个连通件、276条边界边、83条非流形边、2个退化面。三档结果均不封闭，未检测自交、壁厚、强度或切片可打印性。输入为随包自制流程样例，不是美术或精度基准。运行通过不代表几何增益或可打印保证。

独立包的完整推理使用包内解释器和权重，HF/Transformers 离线、代理指向不可达端口，环境中移除宿主 Python/CUDA 路径。这验证当前机器上的独立运行性，不等于所有 Windows 机器兼容。Windows SDPA 未与官方 Linux FlashAttention 做同输入完整输出对照。Linux 安装路径尚未实机验收。

随包报告与截图保存在 validation/；开发机完整证据保存在 data/qa 与 data/jobs。

v0.2.0 测试修正过一个导入顺序造成的隔离问题：测试模块提前缓存路径，旧测试误写了开发目录模型文件。已从先前验收的完整包恢复清单与对应文件，并对全部 16 个文件真实 SHA256 逐一确认一致。新增测试改为 fixture 内延迟导入，正反收集顺序均为 21 passed，修正后的真实文件身份与清单 SHA 前后不变。进一步增加导入路径保护与损坏模型测试的独立 fixture；预先导入生产模块时，会在收集阶段明确报错并停止（test-import-guard.json），不会执行测试。仅测试创建的目录可恢复地移入 data/test-isolation-recovery；原完整包未受影响。恢复证据及隔离结果分别为 model-restoration-2026-10-23.json、test-isolation-result.json。发行包权重还会独立复核。
