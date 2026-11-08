# Idol 原生能力与 Hi3D 整合包接入研究：20 轮

研究日期：2026-11-08。状态：**20/20 轮资料与源码研究完成；未开发、未安装、未下载权重、未运行推理、未做本机效果或浏览器验收。**

研究对象是用户提供的 `Comfy-Org/Idol`，原模型为 `speridlabs/idol`、推理仓库为 `speridlabs/idol-sdk`。它是 TripoSplat 路线的单图物体 3D Gaussian Splat 模型，**不是同名 CVPR 2025 人体 IDOL 项目**。

本报告把证据分为「源码事实」「作者公布」「本地只读核查」「研究建议／待验证」。本次网页与代码检索只能确认接口和实现，不能替代实际生成、显存测量、导出回读与页面操作。以下接入事项均是候选计划，不代表获得开发授权。

## 可用于决策的结论

Idol 值得作为独立的「图片生成 3DGS」候选功能研究，适合快速获得带外观的可旋转物体。原生输出不是现有工作台的 UV/PBR 三角网格。ComfyUI 官方确实已经提供 `SplatToMesh`，并在 Idol 模板中连接了 **SPZ、环绕视频、顶点色 GLB** 三个分支；不能说官方没有转网格路径。但转出的 GLB 使用 `KHR_materials_unlit`，不是原生生成的金属度／粗糙度 PBR，也没有自动骨架、蒙皮或动作。

建议未来分三期评估：**先保留原始 3DGS 并完成独立预览／导出，再验证官方转网格与顶点色 GLB，最后才评估 UV/PBR、拆件和绑定流程的衔接。** 既有 Hi3D ComfyUI 自定义节点仍只提供 Hi3D 精修；可选 Idol 工作流和本地工作台接入是独立交付项。双人动作范围不受影响。

本地已有固定 Comfy core 和可复用的 FLUX.2 VAE，接入并非必须重新下载整套模型或升级整个环境。现有 core 同时服务其他引擎，应优先比较「复用固定旧 core」与「隔离新版 core」；原作者 SDK 用作能力和参数对照。三条路线都仍需要真实验收。

## 研究基线与文件体积

| 对象 | 本轮读取版本 | 证据性质 |
| --- | --- | --- |
| Comfy-Org/Idol | `3d050f4bc705ca1202a0ae4d21d2d97987521307` | HF API 元数据，最后修改 2026-11-04 |
| speridlabs/idol | `2731c2291fc65a797c5cb627a5e5526fc1490902` | HF API 元数据，最后修改 2026-10-29 |
| speridlabs/idol-sdk | `4702a7f00850e34b6e83a2170cc123c3e9570744` | GitHub API 与固定版本源码 |
| ComfyUI 当前上游 | `0decb39ee5b5e17c200334589fc317dbbe98310e` | GitHub API 与固定版本源码；未安装 |
| Comfy 官方工作流模板 | `21d7cc4fb9afd983957f2d2d6b1d583860ccb40b` | 固定版本 JSON |
| Comfy-Org/BiRefNet | `884af709e9b795d89dafbf97f29c8d6a2386f5c6` | HF API 元数据 |
| 本地 Comfy core 来源固定值 | `7cebca52227c63b4180ceed448cbd7ad22903b6d` | `tools/setup_pipeline_runtime.py:27`；目录内版本声明 `0.42.0` |
| 本地 Hi3D 候选代码 | `f45a1f2128fb187aa82f0133f06c9d1949a4435b` | 当前项目现场 |

本地 `.pipelineenv/sources/comfy_core` 是解包源码目录。在其中执行 `git rev-parse` 会向上返回 Hi3D 仓库提交，不能据此声称 Comfy 上游提交；固定版本以来源清单和文件核对为准。

| Comfy 官方工作流文件 | 字节数 | HF 元数据公布的 SHA256 |
| --- | ---: | --- |
| `diffusion_models/idol_fp16.safetensors` | 741106994 | `9abf6a99fc6bdad701901b12de8fe1b256f02cae4948f7489d25d86b11b52a93` |
| `vae/idol_decoder_fp16.safetensors` | 576148442 | `6a146a1e6b2653f74422f7ea835eb8715c547c7de8fc4545051f15c247ae54a9` |
| `clip_vision/dino_v3_vit_h_bf16.safetensors` | 1681247696 | `a29ef35101a16966972a0d50732a6f3a608ff7cfffb2afa9bbe9007cb842cc53` |
| `vae/flux2-vae.safetensors` | 336211292 | `868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3` |
| `background_removal/birefnet.safetensors`（另一个仓库） | 444473596 | `9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154` |

前四个文件合计 **3334714424 字节**；包含 BiRefNet 为 **3779188020 字节，约 3.520 GiB**，不包括 Python、Torch、其他依赖、缓存和产物。这里是发布者元数据，不是新下载后的本地完整哈希验收。原 SDK 的四个权重文件合计 **3078979896 字节**，另外需要 HF 格式的 DINOv3／FLUX.2 VAE 与配置，不能把两个布局的体积或路径混用。

主 Agent 在本轮只读核查确认：本机 `models/extensions/ultratex_text_encoder/split_files/vae/flux2-vae.safetensors` 的文件 SHA256 与上表一致。未来适配器可在复核后引用这个现有文件；据此扣除后，Comfy 五文件方案的其余文件合计 **3442976728 字节**。这只是已确认的 VAE 复用空间，不推定其余文件已存在或需要全部重新下载。

## 20 轮研究记录

### 第01轮：模型身份与研究边界

**问题：** 是否属于此前同名的人体重建模型，能否直接补齐角色动画？

**证据：** [Comfy 模型卡][S1] 指向 `speridlabs/idol`，称其为 TripoSplat 微调；[原作者 SDK][S2] 将任务定义为单图物体到 3DGS；[TripoSplat 论文][S3] 研究的是基于密度控制的通用 Gaussian 表示。

**结论与影响：** 可研究角色、道具等外观生成，不能移植同名人体论文的骨架或可动画结论。路线图应建立独立 Idol 候选项。

**未知：** Idol 微调训练集构成、训练配方和各对象类别的覆盖率，在已读公开材料中不完整。

### 第02轮：原生输出究竟包含什么

**问题：** `.ply` 是否就是工作台已有的网格 PLY？

**证据：** [原生 Gaussian 导出代码][S4] 写入 vertex 元素，字段为位置、零法线占位、SH0 颜色、opacity、scale、rotation；没有 face 元素。生成对象内部使用 z-up，保存时进行坐标／四元数转换。

**结论与影响：** 这是带尺度、旋转与透明度的 Gaussian 集合。工作台现有 PLY 为三角网格几何，必须在上传识别、历史产物、下载说明与预览中区分，不能只按 `.ply` 后缀接入。

**未知：** 第三方查看器的朝向和颜色表现需用真实文件验证，不能根据扩展名推定互通。

### 第03轮：两个 VAE 与推理阶段

**问题：** 为什么要 DINOv3、FLUX2 VAE 和 Idol decoder，是否有隐藏图像生成流程？

**证据：** [原生 conditioning][S5] 用 DINOv3 提取图像特征，FLUX.2 VAE 编码输入图像；[pipeline][S6] 对形状 latent 与 camera token 采样，再由 octree 与 Gaussian decoder 生成点及属性；[Comfy 节点][S7] 对应地将两种图像条件送入采样器。

**结论与影响：** `flux2-vae` 是输入条件编码器，`idol_decoder` 是 3D 解码器。此链无需 FLUX.2-dev 32B 图像生成主模型或文本编码器。模型管理必须按各自用途登记。

**未知：** Comfy 重打包和 SDK 在精度、latent 编码与随机采样上是否逐值一致，未做数值对照。

### 第04轮：输入、抠图与裁切

**问题：** 透明图、照片和生成图怎样进入模型？多图是否表示多视图？

**证据：** [预处理源码][S8] 对有真实 alpha 的 RGBA 直接使用，否则调用 BiRefNet；腐蚀 alpha、裁切、缩放到 1024 并合成黑背景。原 SDK 的 synthetic 边距为 1.2，real 为 1.7。[Comfy 预处理][S7] 当前固定 1.2。SDK 可以遍历输入目录，但每张图独立处理。

**结论与影响：** 当前不是正侧背多视图条件模型，不能复用 Hunyuan3D-2mv 的多视图语义。应保存真正送入模型的预处理图，暴露裁切模式差异。

**未知：** 原工作台 U2Netp 替代 BiRefNet 是否影响效果，需要同图对照；不能静默替换后声称官方结果等价。

### 第05轮：密度、采样与随机性

**问题：** 一百万点是否比 262144 点增加真实细节？

**证据：** [Comfy 解码节点][S7] 支持 32768～1048576，按 32 个 Gaussian 一组；tooltip 明确 262144 对应 octree 点密度，更高数值是重采样已有点，增加资源开销而不带来新细节。SDK 默认 Euler／50 步；官方模板实际为 `dpmpp_2m`／20 步／CFG 3／simple，另有独立解码 seed。[SDK sampler][S9]、[模板][S10]。

**结论与影响：** 默认先研究 262144 点，把点数称作密度预算。记录生成 seed 与解码 seed；不能把两条不同采样流程当同参数基准。最终复核还发现引导值口径不同：SDK 的 `cfg_strength=w` 实现为 `(1+w)·v正 − w·v负`，Comfy 常规 CFG 为 `v负 + scale·(v正−v负)`，因此两个界面都填 3 也不代表相同引导强度；未来对照需连同采样器、预测量和时间表一起核对。[Comfy CFG 实现][S22]

**未知：** 哪组参数在本机最适合照片／卡通／细结构，需要实际多种子对照。

### 第06轮：Comfy 官方支持是否只是加载权重

**问题：** 官方实现已覆盖哪些流程？

**证据：** [固定工作流][S10] 包含模型加载、BiRefNet、前景 mask、`TripoSplatPreprocessImage`、`TripoSplatConditioning`、KSampler、`VAEDecodeTripoSplat`，并可打开逐步解码预览。模板引用的是 `comfy-core` 节点，不要求 Nynxz 等第三方 Idol 节点包。

**结论与影响：** 可以优先研究官方节点组合，避免先增加社区自定义节点依赖。模板内 `ver=0.22.0` 是保存元数据，不应直接当作最低可用版本保证。

**未知：** 本地固定 core 对整个模板中的分组、开关、预览及文件输出 API 是否完全兼容，尚未加载或运行模板。

### 第07轮：官方转网格能力与真实材质

**问题：** 能否导出 GLB，而不局限于 Gaussian？

**证据：** [官方 `SplatToMesh`][S11] 从密度／颜色体积提取表面，输出 vertices、faces、vertex colors，并设置 `unlit=True`；[SaveGLB 实现][S12] 将 unlit 写入 `KHR_materials_unlit`。官方模板已启用 GLB 分支，示例提取分辨率为 384。

**结论与影响：** **有官方转网格路径**，但这是后处理生成的顶点色网格，不是 Idol 原生输出的 UV 贴图或完整 PBR。可成为后续网格加工的研究入口，必须保留原始 Splat。

**未知：** 转出的细节损失、网格闭合性、实际面数、颜色保留与耗时未测；不能承诺可打印或直接用于动画。

### 第08轮：转网格的代价和几何风险

**问题：** 为什么转网格需要独立资源与质量门槛？

**证据：** [当前源码][S11] 的密度网格开销随分辨率三次方增长，存在 GPU／CPU 分支；新版默认 512、Taubin smooth=3、solid=True。solid/fill 会处理内部，节点说明提示低 fill 可能连杯口、衣领一起封闭；小连通件和低透明 Gaussian 可被过滤。

**结论与影响：** 应记录提取分辨率、阈值、过滤数量、闭合／连通件和颜色误差；独立输出转换诊断。将 384 改到 512 不是免费质量升级，理论网格体积比约 2.37，不能当作实测显存倍数。

**未知：** 本机 24GB 对最高分辨率、实心填充和复杂场景是否可用，须在将来实测。

### 第09轮：导出、查看和“动画”的区分

**问题：** 官方视频分支是否就是角色动画？

**证据：** [Gaussian 工具][S11] 提供 PLY、KSPLAT、SPZ 序列化、Splat 读取、变换、合并和渲染；文件节点仅保存 batch 的第一项并给出 warning。[模板][S10] 的视频为 75 帧、25fps 的相机环绕预览；原 SDK另存预测输入相机。[Comfy 解码][S7] 丢弃 camera stream，仅输出 SPLAT。

**结论与影响：** 环绕视频是静态物体的展示，不是骨骼动作。原生 API没有关节、权重、表情或动作输入；单人动画仍依赖现有绑定／动作工具且需先取得合格网格。

**未知：** 批量导出、坐标约定、相机保留以及不同格式回读需分别验证；不要让批量成功掩盖只保存第一项。

### 第10轮：Idol、基座与代码许可

**问题：** 模型卡 Apache-2.0 是否覆盖所有组合组件？

**证据：** Idol 原卡与 SDK 标注 Apache-2.0；[TripoSplat 官方仓库][S13] 将自身代码和权重标为 MIT；[BiRefNet 重打包卡][S14] 标为 MIT；ComfyUI 当前仓库为 GPL-3.0，官方工作流模板仓库为 MIT。本项目 [LICENSE](../LICENSE) 将本地原创集成代码与上游组件许可分开。

**结论与影响：** 打包时逐组件保留来源、许可与修改记录。不能把 Comfy 实现复制到原创 MIT 区域后省略其许可，也不能只凭 Idol 卡的单个标签覆盖编码器。

**未知：** 未来实际采用哪段 Comfy 实现以及如何分发仍未定；最终分发清单需按选择的具体文件复核。

### 第11轮：DINOv3 型号与再分发条件

**问题：** 能否直接复用 Hi3D 已有 DINOv3？

**证据：** Idol 要求 `dinov3-vith16plus-pretrain-lvd1689m`（ViT-H+），本项目既有记录为 ViT-L；名称和架构不同。[DINOv3 原始模型／许可证][S15] 使用独立 DINOv3 License，要求分发时保留该协议，不能改标 Apache-2.0；原仓库为 gated。

**结论与影响：** Comfy 的 H+ BF16文件可作为独立模型登记，不能把现有 ViT-L 当缺省替代。应将模型获取、协议文本、权重版本与哈希一起记录，已有另一型号访问资格不等于本型号完整就绪。

**未知：** 用户当前对 H+ 原仓库的访问状态未检查；本轮不执行账号登录、协议接受或下载。

### 第12轮：FLUX.2 VAE 的许可和可复用文件

**问题：** 使用 FLUX.2 VAE 是否意味着整个 Idol 功能只能非商业使用？

**证据：** [BFL 官方 FLUX.2 README][S16] 明确将 autoencoder 单独以 Apache-2.0 发布；其 `ae.safetensors` 的 HF 元数据哈希与 Comfy 的 `flux2-vae.safetensors` 一致。FLUX.2-dev 主生成模型的非商业许可不应直接套到这个单独发布的 VAE 上。主 Agent 本轮对现有 VAE 文件做了哈希复核，值见前表。

**结论与影响：** 当前证据支持将这个 VAE作为独立 Apache-2.0 组件处理，并在未来适配时校验复用已存文件。此结论不覆盖不需要的 FLUX 主模型、其他编码器或所有第三方重打包。

**未知：** 原 SDK使用 Diffusers子目录格式，现有 Comfy单文件不能仅靠改路径保证可直接加载；需格式适配或选 Comfy 路线。

### 第13轮：依赖冲突和 Windows 运行边界

**问题：** 能否把 SDK 原样装进现有推理环境？

**证据：** [SDK pyproject][S17] 要求 Python≥3.10、Torch≥2.5、Transformers≥4.56、Diffusers≥0.36、TensorDict≥0.8 等。本地 `requirements-pipeline-windows.txt` 固定 Torch2.8、Transformers4.57.1、**Diffusers0.35.2**；[attention实现][S18] 可用 PyTorch SDPA，不强制 flash-attn。

**结论与影响：** SDK有明确 Diffusers版本缺口，不能直接原地 `pip install` 改动现有环境。可研究私有 overlay 或隔离环境；`--no_optimize` 可作为初始对照，不能把关闭优化当已完成 Windows兼容。

**未知：** Windows Torch编译、cuDNN attention、BF16／FP16和真实峰值均未测试。Comfy全仓最新 requirements也不能直接覆写本地锁定依赖。

### 第14轮：脱网、模型路径与进程生命周期

**问题：** 完整包离线运行和任务结束释放能否直接满足？

**证据：** [conditioning][S5] 在 `HF_HUB_OFFLINE` 为真时跳过访问检查，网络异常也不会一律当作无授权；模型加载仍依赖预置本地权重、配置和缓存。[pipeline][S6] 默认联网下载并在实例中同时持有多个组件；原生未提供本项目式的逐阶段子进程恢复协议。本地 `studio_worker.py` 使用受控操作进程。

**结论与影响：** 不能声称 SDK完全不支持离线；未来须固定所有revision、使用可搬迁本地路径、禁用隐式下载，并验证干净缓存／无网络场景。进度、取消、OOM、重启和GPU释放应沿用本项目受控任务约定。

**未知：** 预置后的首启和迁移、compile缓存开销、失败恢复点尚无本机证据。

### 第15轮：显存和速度证据是否可用于承诺

**问题：** “7.2GB／不到2秒”能否作为整合包规格？

**证据：** [原作者模型卡][S19] 同时出现上述概述，以及50步／262144点／8图中位数表：RTX5090 32GB为14.1秒、RTX4090 24GB为19.1秒、RTX3080 10GB为46.2秒。未给出充分条件把“不到2秒”与该表对应。模型卡承认首次编译更慢。

**结论与影响：** 只作为作者公布数字列出，不据此承诺8GB全流程或本机5090 Laptop24GB速度。3DGS生成、网格提取、75帧渲染、首启编译与导出必须分别计时，不能合并宣传为同一2秒结果。

**未知：** 当前Windows整包各阶段Torch分配／保留、NVML整卡、CPU RAM、首次和稳态耗时全部待测。

### 第16轮：现有 Comfy core 能复用到什么程度

**问题：** 接入是否需要先升级所有 Comfy代码？

**证据：** 本地 `.pipelineenv/sources/comfy_core` 已存在 `nodes_triposplat.py`、`nodes_gaussian_splat.py`、`nodes_save_3d.py`。本地 `SplatToMesh` 默认384、smooth0且没有新版solid/fill；当前上游默认512、smooth3并加入这些参数。本地来源固定值见 `tools/setup_pipeline_runtime.py:27`。

**结论与影响：** 优先把本地固定实现列为第一条候选路线，先核对Idol权重识别和模板API；若需要新版网格修复，使用隔离源码／运行时做对照。直接覆盖原core会扩大UltraTex等既有流程的回归面。

**未知：** 文件存在不代表节点已运行、全模板已兼容或Idol效果通过。当前本轮只做源码比较。

### 第17轮：工作台预览与产物类型

**问题：** 能否简单加一个“Idol”引擎选项就交付？

**证据：** 本地 `engine_catalog.py` 只有coarse／texture类别，`geometry.py:9`要求真实Trimesh及非空faces；`web/index.html`使用model-viewer并将现有PLY说明为仅几何。[Spark官方仓库][S20]提供Three.js Gaussian渲染并支持PLY／SPZ／KSPLAT等；[GaussianSplats3D][S21]也是可研究候选。

**结论与影响：** 需新的明确产物类型、点数报告、独立Splat查看器与下载入口。复用当前GLB查看器只能服务后处理网格，不能替代原始Splat预览。查看器应随包提供固定版本资源，离线验收后再宣称支持。

**未知：** 尚未选定查看器、下载其依赖或验证本机WebGL2、排序性能、颜色和百万点内存。

### 第18轮：UV/PBR、拆件和绑定的衔接阻碍

**问题：** 官方GLB能否立即接入所有现有资产加工？

**证据：** 官方网格为顶点色unlit。现有 `studio/material_projection.py:138` 仅允许 `KHR_materials_emissive_strength`，遇到 `KHR_materials_unlit` 会明确拒绝；原生Splat没有UV、材质贴图或骨架。现有角色和动作验收以真实网格／蒙皮为基础。

**结论与影响：** 纯几何导入、页面浏览、材质转移和骨骼绑定要分开验收。未来可评估保留unlit顶点色、烘焙外观，或独立生成UV/PBR；不能静默删除unlit后宣称材质保真。Hi3D精修改变几何，也须保留原Splat／原顶点色网格并重新评估颜色路径。

**未知：** 高面数／薄结构的减面、P3-SAM拆件、SkinTokens／UniRig绑定质量均未测试，不应自动列为Idol现成功能。

### 第19轮：分期接入路线与取舍

**问题：** 哪条路线最符合整合包现有结构？

**证据：** 本地已有隔离runtime、模型路径和受控worker，官方core已有SPLAT类型及导出；SDK接口较小但依赖与权重布局不同，当前SDK没有官方网格后处理。

**研究建议：**

| 候选路线 | 可复用内容 | 新工作／主要风险 | 研究排序 |
| --- | --- | --- | --- |
| 复用本地固定Comfy core的私有worker | 既有模型加载、TripoSplat与旧SplatToMesh、现存VAE | 节点API抽离、server依赖、模板兼容、旧mesh质量 | 优先验证 |
| 隔离新版Comfy core | 新solid/fill与mesh改进、当前官方模板 | 新依赖／前端版本、回归与分发体积 | 旧版不足时对照 |
| 原作者Idol SDK隔离运行 | 原生预处理、预测相机、Python API | Diffusers升级隔离、HF格式缓存、另需mesh/预览、生命周期适配 | 原生基准对照 |
| 单独交付官方Comfy工作流 | 官方既有UI和三输出分支 | 用户仍需独立Comfy，不能冒充工作台已集成 | 可选交付，不改变原Hi3D节点范围 |

三期分别为：A原始3DGS预览／下载；B官方提取网格与顶点色GLB；CUV/PBR及资产加工兼容。每期单独验收，完整代码包仍排除模型与Python。本轮不执行任何一期。

**未知：** 无实测前无法给可靠开发工期；工作量主要在产物／预览、旧新core对照和材质兼容，不只在新增模型列表。

### 第20轮：交叉审查与未来验收门槛

**问题：** 研究是否遗漏足以改变接入决策的证据？

**复核结果：** 已纠正“同名人体IDOL”“官方无mesh路径”“FLUX VAE必然非商用”“旧core完全无节点”“SDK绝对不能离线”“百万点增加真实细节”“环绕视频等于角色动画”等误判。保留官方与SDK预处理、采样、相机、文件格式、旧新mesh参数的差异。

**建议的未来验收清单（均未执行）：**

- 固定模型／源码／许可／哈希；对现存VAE先复核后引用，单独识别DINO-H+，离线与移动目录不得触发下载。
- 原图覆盖透明背景、真实照片、卡通角色、机械道具、薄线／孔洞、半透明与高反光；同时保存预处理图、参数、版本和失败记录。
- 32768／262144／1048576点独立检查点数、有限数值、opacity／scale／quaternion、坐标和颜色；PLY／SPZ下载、回读、浏览器真实旋转并对照原图与正侧背视图。
- 原SDK50步与Comfy模板20步先作为两条不同基线；冷启动、热启动、各阶段耗时和GPU／CPU峰值单独报告，不拿作者显卡表替代本机值。
- 本地旧mesh与隔离新版mesh使用同一Splat对照；检查面数、连通件、边界／非流形、薄结构、封口、颜色和空网格，错误不得报成品成功。
- 顶点色GLB只按其真实材质类型验收；unlit材质转移、UV烘焙和PBR生成分别核对，骨架／动作只在合格网格上追加测试。
- 进度、取消、模型缺失、显存不足、重启恢复、原始产物保留、任务历史／复用和GPU释放必须覆盖；新增能力就绪不能仅等于文件存在。
- 保持原Hi3D工作台和Comfy精修流程可用；运行代码真的变更后再选必要回归，当前已通过的897项不因本次研究重复运行。

**决策：** 将Idol列入路线图的「研究完成、待开发决策／待验证」候选项。原始3DGS价值明确，官方mesh衔接可研究；本机兼容性、资源和最终质量尚无新增结论。

## 建议写入 roadmap 的摘要

- [x] 完成 Idol 原生能力、ComfyUI 官方节点／模板、依赖许可、现有整包衔接的20轮资料与源码研究；本次仅研究，没有安装、下载权重或推理验收。
- [ ] 候选方案：独立图片→3DGS入口，保存原始PLY／SPZ、独立预览和参数；再验证官方SplatToMesh→顶点色unlit GLB，后续UV/PBR／拆件／绑定另验收。
- [ ] 优先对照复用现有固定Comfy core与隔离新版core，原SDK作基准；已有FLUX.2 VAE可经哈希复核复用。DINO-H+、Idol主模型／decoder和BiRefNet分别管理，不直接升级现有pipeline或UltraTex运行时。
- [ ] 完成真实Windows生成、离线迁移、显存／耗时、浏览器旋转／下载、mesh质量、unlit材质兼容与失败恢复后，才决定默认纳入完整包。本研究不改变既有资产加工／单人动画验收状态，双人动作继续延期。

## 原始来源与本地证据

[S1]: https://huggingface.co/Comfy-Org/Idol/blob/3d050f4bc705ca1202a0ae4d21d2d97987521307/README.md
[S2]: https://github.com/speridlabs/idol-sdk/tree/4702a7f00850e34b6e83a2170cc123c3e9570744
[S3]: https://arxiv.org/abs/2605.16355
[S4]: https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/gaussian.py
[S5]: https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/conditioning.py
[S6]: https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/pipeline.py
[S7]: https://github.com/Comfy-Org/ComfyUI/blob/0decb39ee5b5e17c200334589fc317dbbe98310e/comfy_extras/nodes_triposplat.py
[S8]: https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/preprocess.py
[S9]: https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/sampler.py
[S10]: https://github.com/Comfy-Org/workflow_templates/blob/21d7cc4fb9afd983957f2d2d6b1d583860ccb40b/templates/3d_idol_image_to_gaussian_splat.json
[S11]: https://github.com/Comfy-Org/ComfyUI/blob/0decb39ee5b5e17c200334589fc317dbbe98310e/comfy_extras/nodes_gaussian_splat.py
[S12]: https://github.com/Comfy-Org/ComfyUI/blob/0decb39ee5b5e17c200334589fc317dbbe98310e/comfy_extras/nodes_save_3d.py
[S13]: https://github.com/VAST-AI-Research/TripoSplat
[S14]: https://huggingface.co/Comfy-Org/BiRefNet/blob/884af709e9b795d89dafbf97f29c8d6a2386f5c6/README.md
[S15]: https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m/blob/c807c9eeea853df70aec4069e6f56b28ddc82acc/LICENSE.md
[S16]: https://github.com/black-forest-labs/flux2/blob/50fe5162777813d869182b139e83b10743caef15/README.md#flux2-autoencoder
[S17]: https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/pyproject.toml
[S18]: https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/layers.py
[S19]: https://huggingface.co/speridlabs/idol/blob/2731c2291fc65a797c5cb627a5e5526fc1490902/README.md
[S20]: https://github.com/sparkjsdev/spark
[S21]: https://github.com/mkkellogg/GaussianSplats3D
[S22]: https://github.com/Comfy-Org/ComfyUI/blob/0decb39ee5b5e17c200334589fc317dbbe98310e/comfy/samplers.py

HF文件元数据读取端点：[Idol重打包](https://huggingface.co/api/models/Comfy-Org/Idol?blobs=true)、[原模型](https://huggingface.co/api/models/speridlabs/idol?blobs=true)、[BiRefNet](https://huggingface.co/api/models/Comfy-Org/BiRefNet?blobs=true)、[FLUX.2原文件](https://huggingface.co/api/models/black-forest-labs/FLUX.2-dev?blobs=true)。端点是动态的，本报告已保存本次revision与文件元数据。

本地只读核查包括：`roadmap.md`、`LICENSE`、`requirements-pipeline-windows.txt`、`tools/setup_pipeline_runtime.py`、`hi3d_workbench/engine_catalog.py`、`geometry.py`、`studio_worker.py`、`studio/material_projection.py`、`web/index.html`，以及已解包的Comfy节点源码。未改动原有 `IMPLEMENTATION_STATUS.md` 与 `research/`。
