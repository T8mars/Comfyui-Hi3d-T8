# 验证记录

> v0.4.6 补齐加速组件许可证与本验证文档；下列 v0.4.5 推理实测仍适用，推理代码未变。Packaging correction only: acceleration licenses and validation are included; inference is unchanged.

**v0.4.5**：独立工作台增加原索引 UV 展开／校验／缓存、真实网页和终端进度、TRELLIS.2 注意力选择、自定义模型目录与新生成几何的面数预算；UV 默认顺序为 CuMesh 原生→社区 GPU PEC→有界 CPU 分区。ComfyUI 节点及画布工作流仍只做 Hi3D 几何精修；GitHub 附件不含 Python 或模型。CPU 全套回归 567 项通过（57.91 秒，1 条 anyio 依赖弃用警告）。复杂网格 UV／缓存／GLB、注意力 GPU 探针、完整大网格 PBR／渲染／导出及自定义路径的第二次真实 PBR 已通过；新版浏览器交互未验收。详见[双语性能说明](docs/PERFORMANCE.zh-CN.md)与[本版说明](docs/releases/v0.4.5.md)。

## v0.4.5 当前验收范围

| 检查 | 结果与范围 | 证据 |
|---|---|---|
| 原慢任务定位 | 原几何 CUDA 阶段 158.684 秒、973,968 面；旧 UV 可在对应校验失败后转入整网 CPU 重做。取消前 UV 等待下限 1,381.888 秒，未计为成功展开耗时 | [问题任务记录](validation/v045-progress.json) |
| CPU 自动化回归 | 567 项通过，57.91 秒，1 条依赖弃用警告；覆盖真实进程混合 stderr 回车／stdout 事件、HTTP 与终端进度、坏 JSON、取消／锁、参数传递、UV／缓存及发行检查。夹具不计作模型推理 | 开发机 `data/v045-tests-final.xml`、`data/v045-tests-final.log` |
| 原生注意力 GPU 探针 | RTX 5090 Laptop 上 FlashAttention 2.8.3 的 FP16／BF16、head dimension 64／128、不等长 packed CUDA 对照通过；dense 保持 SDPA | 开发机 `data/uv-final-v045/attention.json` |
| 局部内核计时 | BF16／8 头／128 维，预热 3 次、同步计时 10 次均值：等长原生 2.248 ms／SDPA 2.485 ms；不等长原生 6.720 ms／SDPA 13.945 ms。仅是该算子样例，不能推导整任务提速 | 同上；脚本 `data/validate-uv-backends-v045.py` |
| 完整几何注意力对照 | 同图、seed 0、512／12 步、省显存、30 万面预算：SDPA 359.211 秒、原生 505.729 秒；后者记录 1,260 次原生调用、0 次 packed SDPA 回退。各一次且受加载／PageIn 干扰，不宣称整任务更快或输出逐字节一致 | 开发机 `data/trellis-attention-v045/comparison.json` |
| 存储与读页诊断 | E: 对应 SCSI 端口 1，瞬时队列 28，同日有 stornvme 129／RaidPort1 重置。另一次两秒采样 E 盘 1,782 次读取、平均 14.808 ms；C:/D: 仅一次读取，不构成公平硬件对比或硬盘故障结论 | 开发机 `data/v045-storage-evidence.json`、`data/v045-read-latency.json` |
| 完整包可选轮子 | E 盘独立运行时经正式安装器完成固定 FlashAttention、Numba／llvmlite 安装，`pip check` 通过。安装／import 回执与 GPU／JIT／速度验证分开 | 包内 `runtime-pipeline/attention-acceleration.json`、`runtime-pipeline/uv-acceleration.json` |
| 模型目录界面 | 两项新增 DOM／真实 JS 处理器测试通过：显示已配置／实际路径、保留编辑草稿和成功保存值、忙时不发送保存、409 保留配置、保存／恢复默认 API、中文空格路径。连同 CPU／CUDA 进度等共 26 项专项通过；已有核心文件需重新完整校验的提示已加入 | `tests/test_model_directory_ui.py`、`tests/test_job_progress.py`；新版浏览器交互未验收 |
| 模型目录后端 | 44 项目录用例并入全套回归，覆盖实际文件锁、token／origin 403、忙时 409、相对 ROOT／中文空格、worker 读取、恢复默认、不移动／不创建目录与根绑定校验。另有现场 EXE／新 worker／完整 PBR 验收，完成后已恢复默认根 | `tests/test_model_paths.py`；开发机 `data/v045-custom-path-pbr/validation.json` |
| 最终 UV 与缓存 | 原 973,968 面网格的两条路径及缓存回读通过内部逐角位置／法线／面覆盖检查。均保留 973,926 有效面，仅排除 42 精确零面积面。CuMesh 首次 65.155 秒／缓存 1.275 秒；社区首次 162.418 秒／缓存 1.437 秒。社区含约 54 秒依赖冷读、18 秒首次 JIT，不作公平算法倍速对比 | 开发机 `data/uv-verified-v045/results.json`、各后端 `events.jsonl` |
| UV 局部修复范围 | CuMesh 修复 108 面（107 重复 UV、97 最小高度调整，分类重叠），图集边长 4,112；社区修复 2,543 个源同位置反向面实例，含 14 最小高度调整，无无关 UV 冲突。保留有效源几何；早先拒绝结果仍记为失败 | 同上及 `community/diagnostics/community-repair.json`；历史失败 `data/uv-final-v045/results.json` |
| UV GLB 导出重载 | 两后端首次／缓存共四份 GLB，经 `load_texture_input` 实际重载审核：float32 源位置逐顶点完全一致；法线 `rtol=0, atol=2e-6`；全 973,926 个有向面多重集及零面积／完全重复 UV 检查通过。未计作 PBR 推理 | 开发机 `data/uv-verified-v045/glb-roundtrip.json` |
| 大网格完整 PBR | E 盘 EXE、已有百万面级模型、TRELLIS.2 1024／12 步／4K、UV／attention auto。HTTP 至成功 780.828 秒，管道报告 766.36 秒，贴图阶段 745.608 秒，其中 UV 69.358 秒、采样 45.61 秒；原生 packed 720 次／回退 0，GPU 采样最高 100%。PyTorch 分配／保留峰值 16,118.76／27,066 MiB，不作为物理显存／最低硬件要求 | 开发机 `data/v045-pbr-http/result.json`、`polls.jsonl` |
| PBR 输出与导出／渲染 | 973,926 面；源位置误差 0，法线／UV 最大误差约 2.98e-8；两张嵌入 4K BaseColor／MR。真实 GLB 三视角 studio-09 HDR 渲染、四分之三侧面／背面检查及 80 mm 保材质导出通过，未声称多 HDR 质量普测 | 同目录 `glb-validation.json`、`export-80mm.json`、`render-studio-09/render-manifest.json` |
| 自定义路径完整 PBR | 中文带空格的 junction 别名根，现场保存立即生效、fresh worker 核心／扩展路径一致、忙时切换 409、完成后恢复默认与 7/7 引擎就绪通过。1,280 面／512／4 步／1K，HTTP 192.625 秒，阶段 182.962 秒；原生 packed 240／回退 0，分配／保留 3,078.08／3,272 MiB。不同负载，仅功能回归，不与大网格比速或推最低显存；未搬移／复制 79GB | 开发机 `data/v045-custom-path-pbr/result.json`、`worker-paths.json`、`custom-path.json`、`restored.json`、`validation.json` |
| 实际 CPU／CUDA 子阶段 | 第二次 PBR 的 HTTP 实见 CPU 体素化→CUDA 编码→材质步数 0–4/4→CUDA 光栅化→CPU 补边；stage.log 记录四次 inpaint 前后计数 0,1,1,2,2,3,3,4。大网格第一次运行的旧整段 CUDA 标签不用于此项验收。观察器不改原采样／烘焙数学；无设备依据时不继承 CUDA | 同目录 `polls.jsonl`、正式任务 `stages/texture/stage.log`；编码 8 项／烘焙 6 项专项测试 |
| 浏览器交互限制 | 浏览器入口返回 `Browser is not available: iab`，新版交互未验收。真实 HTTP／终端集成与 DOM／JS 测试通过不能替代实际浏览器操作 | 保留实际工具不可用的范围限制 |

以上实测设备为 Windows／RTX 5090 Laptop 24GB／64GB RAM；不构成最低显存要求或其他设备兼容保证。代码更新保留 Python／模型，不会静默安装可选加速依赖。`data/` 为开发机原始记录，不作为 GitHub 附件分发；公开汇总随最终验收更新。

**v0.4.4**：补齐本地全模型包，Hi3D 核心及 14 个扩展组件就绪，覆盖 7 个引擎选项；新增 UltraTex INT8 ConvRot Base 9B。328 项回归通过。Windows 独立包实际完成宝箱贴图：114 个量化权重张量、1024 采样、颜色／材质各 25 步、4K 材质；嵌入 PBR 的 GLB 及 80mm 导出检查通过，并查看实际 GLB 的正面、侧面和背面渲染。推理阶段 511.034 秒，张量分配／保留峰值 11629.45／13790 MiB；不作为最低显存要求。Hugging Face 全部 95 个模型／配置／许可文件的远端哈希匹配。新控件未做浏览器交互验收。见 [v0.4.4 完整回执](validation/v044-int8.json)。

**v0.4.1**：新增独立工作台代码自动更新；模型与推理算法不变。本轮 313 项自动化回归通过，其中 37 项验证服务生命周期、更新包路径、校验、安装、任务锁、备份、失败回滚、中断恢复与缓存失效。已用原 v0.4.0 Windows 独立包实际安装代码更新，并验证其私有 Python 启动、更新 API、设置切换与会话令牌保护；逐文件对照原 ZIP 核对运行时与模型内容。新增浏览器交互未验证，本轮不重复计为模型 GPU 推理。完整回执见 [v0.4.1](validation/v041/)。

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
