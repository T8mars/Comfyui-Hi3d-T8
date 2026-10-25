# v0.4.4 · 全模型覆盖与 INT8 ConvRot

增加 UltraTex INT8 ConvRot Base 9B 独立选项及对应固定来源、字节数与 SHA256，保留 FP8；只有按原生量化张量加载成功才允许采样。模型页明确显示就绪数量，增加补齐全部组件入口，打包使用 `--all-engines` 并记录全引擎覆盖标记。大文件 HTTP 组件下载改为分块写入，校验长度与 SHA256，避免整文件进入内存。ComfyUI 接口不变。

# v0.4.3 · 首次升级启动兼容

2026-10-25。修复 Windows 旧版 `os.execv` 在替换解释器时提前结束原进程，导致 EXE 过早清理整个服务树的问题。稳定监督进程先完成代码安装，再启动全新服务解释器并等待其退出。v0.4.2 改为预发布；正式版使用 v0.4.3。

Fix Windows first-update startup by keeping a stable supervisor alive across code installation and launching the server in a fresh interpreter.

# v0.4.2 · 可见 EXE 启动与进程退出

2026-10-25。独立 Windows 工作台增加 EXE 控制台启动器、网页关闭按钮和停止脚本。关闭启动器清理其服务及推理子进程，Ctrl+C 支持正常退出；重复启动复用本目录服务，端口冲突拒绝操作其他目录服务，服务就绪后再打开浏览器。启动失败保留日志，侧栏与页尾增加七个社媒资源入口。ComfyUI 节点和模型算法不变。

Visible Windows console launcher, explicit shutdown, process-tree ownership, same-directory instance control and community links. GitHub provides a separate launcher ZIP without Python or weights; code updates remain compatible with v0.4.1.

# v0.4.1 · 代码自动更新与完整模型清单

2026-10-25。独立整合包增加正式 GitHub Release 自动检查、代码下载和下次启动安装，提供 SHA256 校验、代码备份、失败回滚与中断恢复。模型、Python 运行环境、配置和历史任务保留；依赖环境不兼容时要求新版完整版。源码 Git 目录不自动覆盖。

GitHub Release 仅提供源码、节点和代码更新包，不包含 Python 或模型。完整版由 T8star 另行提供。Hugging Face 模型仓库扩展至 Hi3D 核心及 13 个扩展组件，共 94 个权重、配置与许可文件；各组件原始许可分别保留。ComfyUI 节点与画布接口不变，继续只做 Hi3D 几何精修。

The standalone bundle gains stable-release code updates with staged installation, SHA256 verification, backups and rollback. GitHub distributes code only; T8star provides the complete Python/model bundle separately. The model repository covers all supported core and extension components. See [update instructions](UPDATES.md).

# v0.4.0 · 多引擎资产工作台

2026-10-24。ComfyUI 节点和可视化画布仍只复现 Hi3D；新增功能只进入独立工作台。当前验收见[质量记录](../validation/quality-upgrade-progress.json)与[20 项审查](../validation/quality-release-audit20.json)。

- 建模增加 Hunyuan3D 2.1 与 TRELLIS.2，贴图增加 Hunyuan3D-Paint 2.1 与 UltraTex，各自下载固定版本依赖并单独记录许可。
- 保留上传 GLB 的材质、可用 UV 和文件自带法线；正确处理非均匀缩放与镜像实例，修复 TRELLIS 复制网格时丢失法线的问题。
- 原生 UV 展开后，以精确坐标和有向邻面恢复重排前索引，保留重合硬边；无法证明对应关系或三角面完全分离时才使用 CPU Atlas。坏 UV 自动重展。
- 同视角历史模型对比、两种 HDR 灯光、曝光检查，以及原始／整理几何的分别导出。
- 导出可指定 X、Y、Z 尺寸轴；GLB 保留 PBR，STL／PLY 仅几何。
- 模型和下载元数据采用落盘同步；Windows 校验读取期间阻止文件写入／替换。整合包按所选引擎打包，并核对依赖清单、实际模型和运行时。

The standalone v0.4 workbench adds three geometry and three PBR engine choices, material-preserving GLB import/export, shared-light comparisons and three starting presets. ComfyUI nodes and the saved visual workflow remain Hi3D-only. The Windows package includes Hi3D, Mini Turbo and TRELLIS geometry/PBR; Hunyuan 2.1/Paint and UltraTex weights download separately. UltraTex's FLUX base is Non-Commercial. Validation distinguishes CPU regressions, real inference, portable-runtime checks and browser interaction; see [VALIDATION.md](../VALIDATION.md).

# v0.3.3 · 第三轮 20 轮联合复查

2026-10-24。逐轮证据见[复查记录](../validation/v033-audit20.md)，235 项测试与 15 组浏览器回归通过。模型权重、推理算法与 ComfyUI 节点和画布接口未变。

- 创建任务时复制输入或保存初始记录失败，清理尚未发布的任务目录，不留下无主半成品。
- 每个模型阶段都必须提供明确成功报告；退出码为 0 或存在 GLB 仍不能替代成功状态。
- 失败报告写盘或进度管道同时出错时，仍执行阶段临时 GLB 清理。
- 拒绝转换为 GLB 后单个轴向尺寸完全丢失的网格，提示居中和调整尺寸；保留合法平面输入。
- 浏览器回归与导出测试可独立运行，不依赖用户服务中已加载的模型或其他测试的导入顺序。

# v0.3.2 · 第二轮 20 轮联合复查

2026-10-24。逐轮证据见[复查记录](../validation/v032-audit20.md)，207 项测试与 15 组浏览器回归通过。模型、固定上游源码、采样算法和 ComfyUI 节点接口未变。

- 阶段同步、指标或成功报告写入失败时保留旧 GLB，完成检查后才发布新结果。
- 拒绝超出请求长度的分片响应，删除坏临时片并有限重试，防止异常代理持续写满磁盘。
- 退出码为 0 时仍要求有效成功报告；损坏或空失败诊断回退到真实退出码和日志。
- 几何或带材质导出失败时清理本次半成品，保留原模型与其他导出文件。
- 健康接口报告实际监听端口，正确反映 `--port` 覆盖且不改写用户配置。

# v0.3.1 · 20 轮联合复查与修复

2026-10-24。完整记录见[20 轮审计](../validation/post-v03-audit20.md)。ComfyUI 节点、画布工作流、模型权重与推理算法保持原有范围。

- 模型哈希校验前后比对文件身份，防止校验期间被替换的文件继承已验证状态；验证和下载收尾都应用此检查。
- 损坏的 Hub 配置和小权重缓存只强制重取一次，固定 revision 并重新校验；持续错误不会被标记就绪。
- 扩展清单来源不匹配时明确报告 repo/revision 错误；已就绪时不再默认显示“尚未下载”。
- 上传解析或检查失败时清理本次半成品，不遗留大文件，不删除原有资产。
- 更新浏览器审计工具以适配三入口工作台，并允许独立保存检查回执。180 项回归与 15 组浏览器检查通过；本轮未修改推理数学，沿用已有真实 GPU 产物验收。

# v0.3.0 · 图片建模与 PBR 工作台

2026-10-24。新增功能仅用于独立整合包，ComfyUI 三节点与原画布保持 Hi3D 几何精修。

- 新增 Hunyuan3D-2mini-Turbo 图片建模、CPU U2Netp 前景处理、TRELLIS.2 对最终网格的 UV/PBR 贴图；基础、精修与材质版本分别保留。
- 提供图片建模、已有模型贴图、模型精修三个入口。图片默认基础模型＋PBR，Hi3D 可选；卡通同图对照与机械、建筑实际产物另存验收记录。
- 扩展组件使用独立 Python 3.12／Torch 2.8 cu128，不修改原 Hi3D 或 ComfyUI 环境。固定模型、源码、Windows 扩展来源和哈希，保留各自许可。
- 材质 GLB 校验 UV 与真实内嵌贴图，缩放导出保留材质；STL/PLY 只导出几何。拓扑诊断识别相同坐标的 UV 接缝，不修改模型。
- 修复阶段监督进程退出后的子进程清理、跨阶段取消、坏模型记录与文件篡改校验。完整证据见[资产链验收](../validation/asset-pipeline-progress.md)。

# v0.2.2 · 发布后 20 轮联合检查

2026-10-23。新增检查过程与证据见 [发布后审计](../validation/postpublish-audit20.md)。官方源码、权重和采样数学保持不变。

- ComfyUI 保留 RGBA 与 MASK 透明区域，拒绝非法图像/蒙版；提交前及时响应中断，缩短轮询与取消超时。
- 已保存模型不因报告缺失而隐藏；日志失败局限当前日志区；历史旧响应不覆盖新结果；恢复笔刷连续绘制，擦除单击生效。
- 模型校验与下载启动互斥；诊断超时清理继承的子进程，未退出的已确认进程保持GPU阻塞。
- 拒绝复数、GLB溢出/整体坍缩坐标及缩放后溢出；归档排除私有文件和废弃API示例；修复验证工具对已删除API文件的依赖。

# v0.2.1 · 20 轮联合检查

2026-10-23。完成主 Agent / 子 Agent 交叉检查，逐轮证据见 `validation/audit20-rounds.md`。保留粉白界面与 By T8star 署名，不更改 Hi3D 模型权重、官方源码或采样数学。

- 异常 worker 输出不再遗留进程；取消、线程启动失败、短进程、磁盘写入失败均按实际进程状态处理。损坏历史不拖垮启动，身份无法确认的残留进程阻止重复 GPU 任务。
- 校验固定模型仓库 / revision；非 LFS 配置核对官方 Git blob ID，损坏缓存会重取。模型元数据损坏可读提示，覆盖前备份原件。
- ComfyUI 拒绝浮点面索引、非法 padding、越界索引及无效顶点，避免静默截断改变网格。
- 健康检查错误或超时暂停提交，失效历史任务恢复到新任务；重连保留 Attention 选择。
- 导出时锁定尺寸与格式；蒙版开始编码即锁输入；预览加载时隐藏旧模型；同步平移并正确重置视角；缺失工件不会继续预览或复用。
- 上传解析与几何检查移入线程池，保持状态请求响应。只信任回环 Host；页面 / API 不缓存，前端资源带版本键。
- 发行审计检查全部自有代码及节点包，避免 ZIP CRC 通过却打入旧版代码。

测试的假 worker 与浏览器故障响应只验证错误处理，不计作模型推理或质量测评。新目录发布，旧目录保留。

# v0.2.0 · By T8star

2026-10-23。工作台改为粉白配色，颜色参考本机 IndexTTS 启动后的 Web 工作台：粉色强调、白色圆角卡片与浅粉蓝渐变。未修改 IndexTTS 文件。

新手按「准备输入 → 开始精修 → 对比导出」操作。页头明确参考图与粗网格的输入要求，提供内置样例和输入准备说明；高级设置收纳步数、种子和 Attention 后端。主按钮在屏幕外时显示底部快捷操作，无需先找到页面底端。保留默认 512 / 25 步 / CPU 卸载，不改变 Hi3D 权重、推理数学或 ComfyUI 节点职责。

历史任务可以复用原输入和参数创建新任务，包含没有上传资产 ID 的旧离线任务。复用与尺寸导出不覆盖原件。正在运行的任务有独立提示栏，可在查看旧任务时找到并停止。

已修复：

- 提交前完整校验期间重复点击或诊断抢占 GPU：前端保持提交锁，后端互斥提交与 GPU 诊断。
- 输入上传、样例及蒙版并发导致错配：处理期间锁定输入，样例两份工件全部成功后才替换当前输入；同一文件可以再次选择。
- 切换任务或新输入残留旧指标、条件图、日志与尺寸链接；已完成的历史任务现在也会读取日志。
- 清空预览地址后查看器仍绘制上一份网格：无输入地址时明确隐藏旧视图；完成记录仅高亮「对比导出」。清空必填步数会展开设置并提示，不再提交为 0。
- 失败、取消或中断任务默认导出不存在的精修文件：按实际文件启用导出版本；只有粗模型时默认导出粗模。
- 断线继续显示就绪、会话过期报错：断线提示并暂停提交，自动重连与刷新本地会话。
- 模型文件仅存在就显示校验通过：校验记录绑定清单、文件身份与变更时间；失败或变化如实显示，首次启动自动完整校验。轮询只检查元数据。
- 重启后下载永远停留在下载中、损坏分片反复复用：按真实跨进程锁区分下载与中断，坏分片可重试。

完整包在新目录发布。保留旧目录的数据；可在关闭旧服务后，把旧版 `data` 复制到新版目录以继续查看记录。不要同时对同一数据目录启动多个服务。新版首次启动会校验当前目录的模型文件。

本次修改不引入其他 3D 模型。以当前版本测试与实际截图为准；先前 512 / 1024 / 1536 实测仅说明原推理路径已运行，不等于所有输入的效果保证。
