# Hi3D T8

[English](README.md) | **简体中文**

**By T8star** — Hi3D V1.0 ComfyUI 几何精修节点与独立本地 3D 工作台。

**ComfyUI 节点**只复现官方 Hi3D：输入**同一物体的参考图＋粗网格**，输出精修几何；节点接口与画布工作流保持不变。**独立整合包 v0.4**另支持单图生成几何、多种建模／PBR 引擎选择，以及同灯光下的历史结果对比。已验证范围见 [roadmap.md](roadmap.md)。

[模型下载](https://huggingface.co/t8star/Hi3d-Comfy) · [画布工作流](examples/Hi3D-Refine.json) · [验证记录](VALIDATION.md) · [Hi3D 原项目](https://github.com/hi3d-official/Hi3Dv1.0)

![ComfyUI 中的 Hi3D 画布工作流](docs/images/comfyui-canvas.png)

## 下载与自动更新

**GitHub Release 只发布代码：源码包、节点包、程序更新包和小型 Windows 启动器，不包含模型权重、Python 或运行环境。含 Python 与模型的完整版由 T8star 另行提供，下载地址由作者公布。** 完整版包含的具体组件以其 `bundle-manifest.json` 为准。

[GitHub Release](https://github.com/T8mars/Comfyui-Hi3d-T8/releases/latest) · [完整更新说明](docs/UPDATES.zh-CN.md)

独立整合包 v0.4.1 起默认空闲时自动检查并下载程序更新，校验后在下次启动安装；模型、Python、配置与历史任务保留。可在「模型与运行环境 → 程序自动更新」关闭，或关闭工作台后运行 `Update-Workbench.cmd`。旧 v0.4.0 需先手动覆盖一次代码更新包。需要更换 Python 依赖时会提示获取新版完整版，不自动替换环境。源码 Git 目录使用 Git 更新。

### Windows 整合包启动与关闭

**v0.4.4 全模型整合包：**包含三个建模选项、四个贴图选项，其中 UltraTex 提供 **INT8 ConvRot Base 9B** 与 FP8 Base 9B 两种精度。完整覆盖需要 Hi3D 核心与 14 个扩展组件，模型及配置约 79.3 GB；只装默认组件的包不属于全模型版。「模型与环境 → 补齐全部引擎」会准备所有已注册依赖，并显示就绪数量。完整包清单应声明 `full_engine_coverage: true`。

制作全模型包时，`tools/build_release.py` 需同时使用 `--include-models --include-pipeline --all-engines`。INT8 底座采用 tsolful 发布并固定哈希的社区量化 **Base 9B**，不是蒸馏版或 KV 版；保留原始 FLUX 非商业许可，本项目不重新量化。不同 GPU 上 INT8 不保证比 FP8 更省显存或更快，应以实际报告为准。

双击 **Hi3D-Workbench.exe** 或 **Start-Workbench.cmd**，保留启动窗口查看日志。按 **Ctrl+C**、关闭启动窗口、点击网页的**关闭工作台**，或运行 **Stop-Workbench.cmd**，均可停止服务；**只关闭浏览器页面不会退出服务**。EXE 统一管理服务和推理子进程，关闭时释放其工作进程；启动失败时保留错误信息。重复双击会打开本目录已运行的工作台，不重复启动，也不会关闭其他目录的服务。

独立的 `Hi3D-Workbench-launcher-<版本>.zip` 仅包含小型 Windows 启动器和启停脚本，**不含 Python 或模型**。先安装对应代码更新，再把启动器解压到 `Start-Workbench.cmd` 同级目录。缺少 EXE 时，启动脚本可调用 `scripts/Build-Launcher.ps1`，使用 Windows 自带的 .NET Framework 4 编译器生成；源码随仓库提供，EXE 未做数字签名。新版完整版直接附带。工作台侧栏与页尾提供作者社媒、教程及资源入口。

## ComfyUI

1. Registry 版本可用后，在 ComfyUI-Manager 搜索 **Hi3D T8**（节点 ID：`hi3d-t8`，Publisher：`t8star`）安装；也可克隆本仓库到 `ComfyUI/custom_nodes/Comfyui-Hi3d-T8`，使用 ComfyUI 自身 Python 安装 `requirements.txt`。
2. 启动**独立的本地工作台后端**，等待 `http://127.0.0.1:8378` 显示模型校验通过，安装方法见[本地工作台](#本地工作台)。
3. 将 [Hi3D-Refine.json](examples/Hi3D-Refine.json) 拖入 ComfyUI **画布**。这是包含节点位置和连线的可视化工作流，不是 API 格式。在 **Load Image** 上传参考图，在 **Hi3D Load Mesh** 填写粗网格的本机绝对路径。
4. 建议从 **512 / 25 步 / CPU 卸载**开始。Windows 选择 `sdpa`，官方 Linux 环境选择 `flash`，点击运行。原生 **SaveGLB** 将结果保存到 ComfyUI 输出目录。

三个自定义节点为 **Hi3D Backend、Hi3D Load Mesh、Hi3D Refine**，使用 ComfyUI 原生 `MESH`。后端与 ComfyUI 运行在同一台电脑；节点包不安装或替换 ComfyUI 的 PyTorch/CUDA。详见[节点说明](docs/COMFYUI.md)。

## 模型与两个版本的区别

| 组件 | 用途 | 使用范围 |
|---|---|---|
| Hi3D transformer 512＋VAE | 512 精修 | ComfyUI 和工作台共用 |
| Hi3D transformer 1024＋同一 VAE | 1024、1536 精修；没有独立 1536 权重 | ComfyUI 和工作台共用 |
| DINOv2 Large＋CLIP ViT-L/14 | Hi3D 必需图像编码器，不是额外 3D 模型 | ComfyUI 和工作台共用 |
| Hunyuan3D-2mini-Turbo／Hunyuan3D 2.1／TRELLIS.2 | 可选的图片建模引擎 | 仅独立整合包 |
| TRELLIS.2 PBR／Hunyuan3D-Paint 2.1／UltraTex | 可选的已有模型 PBR 引擎 | 仅独立整合包 |
| DINOv3、DINOv2 Giant、FLUX.2 Klein 9B 和 Qwen3 | 按所选扩展引擎安装的依赖 | 仅独立整合包 |
| CPU U2Netp | 不透明图片的可选前景提取 | 仅独立整合包 |

[Hugging Face 模型仓库](https://huggingface.co/t8star/Hi3d-Comfy)提供两版共用的 **Hi3D 核心权重与配置**，以及独立工作台的**全部扩展组件**。没有专供 ComfyUI 的另一套 Hi3D 权重；扩展放在 `models/extensions`，各自保留固定版本、校验值与许可。UltraTex 的 fp8 底座沿用上游发布，未由本项目重新量化。Registry / 节点包只包含**代码和画布工作流**；单独下载权重不等于具备推理环境。

在**工作台根目录**执行，权重无需放进 `ComfyUI/models`：

```bash
hf download t8star/Hi3d-Comfy --local-dir models --include "manifest.json" "hi3d/*" "dino/*" "clip/*" "licenses/*"
```

以上只下载 Hi3D 核心。下载全部建模／贴图扩展请去掉 `--include`，全量约 79.3GB；目录和按需下载说明见[模型仓库](https://huggingface.co/t8star/Hi3d-Comfy)。目录应为 `models/manifest.json`、`models/hi3d/`、`models/dino/`、`models/clip/`，扩展位于 `models/extensions/`。清单记录原始仓库版本与 SHA256。后端也支持直接从官方原仓库下载。

<details>
<summary>实际贴图样例</summary>

![实际贴图样例](docs/images/quality-fox-v04.png)

独立工作台实测：TRELLIS.2 建模＋UltraTex PBR；本例未启用 Hi3D 精修。

</details>

## 本地工作台

Windows 版双击 **Start-Workbench.cmd**，打开 `http://127.0.0.1:8378`，可脱离 ComfyUI 使用。工作台提供三个入口：图片→基础模型→可选 Hi3D 精修→PBR；参考图＋已有模型→PBR；参考图＋粗模→Hi3D，可选继续贴图。图片模式默认**基础模型＋PBR**，Hi3D 按需勾选：同图卡通对照中，跳过精修的嘴部更干净。各阶段保留模型、日志和报告。建模与贴图引擎独立选择，可在“模型”页按需下载。上传的自包含 GLB 保留材质和可用的单材质 UV；历史结果可在同视角、HDR 灯光与曝光下左右对比。选择建议见[引擎与质量记录](docs/ENGINES.zh-CN.md)。带材质结果使用自包含 **GLB**；**STL/PLY 只包含几何**。

v0.4.4 全模型离线包包含两套隔离运行时、Hi3D 核心及全部 14 个扩展组件，Hunyuan 2.1、Hunyuan Paint、UltraTex FP8／INT8 ConvRot 均包含在内，准确清单见 `bundle-manifest.json`。旧基础包可在“模型”页补齐；所选引擎未就绪时会明确提示。节点包和源码包只含代码，源码部署需另行准备环境与权重。

提供三个起点：**轻量**为 Mini Turbo 256＋TRELLIS PBR 512／2K；**均衡**为 TRELLIS 几何 512＋PBR 1024／4K；**细节**为 TRELLIS 几何 1024＋UltraTex INT8 ConvRot 1024／4K。UltraTex 的 FLUX.2 Klein 9B 底座适用**非商业许可**。调大输出贴图尺寸不会补出生成中缺失的细节。详见[引擎选择](docs/ENGINES.zh-CN.md)和[工作台说明](docs/WORKBENCH.zh-CN.md)。

Windows 源码安装使用 Python 3.10 和独立环境：

```powershell
git clone https://github.com/T8mars/Comfyui-Hi3d-T8.git
cd Comfyui-Hi3d-T8
powershell -ExecutionPolicy Bypass -File scripts/Setup.ps1 -Python C:/Python310/python.exe -DownloadModels
```

独立工作台的图生／贴图阶段还需用 Python 3.12 执行 `C:/Python312/python.exe tools/setup_pipeline_runtime.py`，再执行 `.venv/Scripts/python.exe -m hi3d_workbench.pipeline_models download` 安装基础扩展模型。其他引擎在“模型”页单独下载，见[引擎说明](docs/ENGINES.zh-CN.md)。

Linux Hi3D 精修可执行 `bash scripts/setup-linux.sh`，再运行 `.venv/bin/python -m hi3d_workbench.model_cli download` 和 `.venv/bin/python -m hi3d_workbench --open`。Linux 脚本面向官方 Ampere/Hopper 环境，尚未在本机验收；新增图生／PBR 运行时目前只在 Windows 验证。更多操作见[工作台详细说明](docs/WORKBENCH.zh-CN.md)。

## 验证与限制

实测设备为 **Windows / RTX 5090 Laptop 24GB / 64GB RAM**。本轮多引擎推理、独立运行时检查及待验收项见 [VALIDATION.md](VALIDATION.md)；此前 ComfyUI 512 和独立工作台 512/1024/1536 精修实测保留为历史证据。自动化测试、HTTP 推理与浏览器操作分别记录。尚未验证更小显存设备；张量分配峰值不能作为最低显存要求。

Windows 使用明确标识的 **SDPA 兼容适配**，与官方 Linux FlashAttention 环境有区别。Hi3D 会归一化尺寸并重建拓扑；不保证结果封闭、可打印或比粗模更准确，导出前应对比检查。

随包 TRELLIS Windows 扩展要求 **CUDA 计算能力≥8.6**，因此该阶段不支持 RTX 20 系列或 A100（8.0）。通过架构检查不代表显存足够或该型号已经验收。

## 许可与署名

原创集成代码使用 [MIT](LICENSE)；UltraTex 在独立工作台运行时内另使用 GPL-3.0 的 Comfy 推理库与 **FLUX 非商业许可**底座，分别保留其许可；Hi3D 代码与权重使用 **Hi3D Open License v1.0** 并保留原始声明，编码器与运行时分别遵循各自许可。本项目为社区集成，详见 [NOTICE.md](NOTICE.md)。[Powered by Hi3D 1.0](https://www.hi3d.ai/)。

## T8star

[B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [免费画廊](https://www.openzhenzhen.com) · [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
