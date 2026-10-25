# 程序更新与下载

[English](UPDATES.md) | **简体中文**

GitHub Release 只提供源码、ComfyUI 节点和程序更新包，**不包含模型权重、Python 解释器或运行环境**。含 Python 与模型的完整版由 **T8star 另行提供**，下载地址由作者公布；不要将源码包称为开箱即用整合包。完整版实际包含哪些模型，以该包的 `bundle-manifest.json` 为准。

## Windows 可见启动器

v0.4.2 起附带 `Hi3D-Workbench.exe`，保留日志窗口并统一管理服务及推理子进程。支持 Ctrl+C、关闭窗口、网页关闭按钮与 `Stop-Workbench.cmd`；仅关闭浏览器页面不会停止服务。为兼容旧版 v0.4.1 校验器，代码更新包继续排除 EXE。新版启动脚本在缺少 EXE 时使用 Windows 自带的 .NET Framework 4 编译器生成，也可在安装代码更新后解压同版本 Release 的独立启动器 ZIP。今后启动器二进制有变更时需另行覆盖该附件；Python 和模型保持独立。

v0.4.7 修复 EXE 未接入 Python 标准输出与报错的问题。旧包需要同时更新代码和 **v0.4.7 启动器 ZIP**；仅更新代码无法修复旧 EXE。启动窗口会显示模型目录、访问地址、真实 HTTP 就绪状态、初始化等待及浏览器打开失败提示，任务进度也会传入同一个窗口。

## 独立整合包自动更新

v0.4.1 起，独立整合包默认检查本项目的正式 GitHub Release。空闲时后台检查并下载代码更新，最多每六小时检查一次；下载通过压缩包和逐文件 SHA256 校验后，**下次关闭并重新启动工作台时安装**。不会在推理过程中覆盖程序或强制重启。

「模型与运行环境 → 程序自动更新」可关闭自动下载和下次启动安装，或手动检查。网络失败不影响已安装程序。安装只覆盖清单中的程序文件，保留 `models/`、`runtime/`、`runtime-pipeline/`、`data/`、`config.local.json` 和 ComfyUI 个人工作流。

安装前备份到 `data/updates/backups/`；写入失败会恢复，断电留下的未完成事务在下次启动时先恢复并重载。若备份损坏，启动会停止并提示检查，不能视为已经恢复。备份不会自动删除。

也可先关闭工作台，再双击 `Update-Workbench.cmd` 检查、下载并安装。更新器拒绝覆盖正在运行的同目录服务。源码 Git 目录禁用自动安装，使用 `git pull` 更新。

## v0.4.0 首次接入

旧版尚无更新器：关闭工作台，将 Release 的 `Hi3D-Workbench-update-0.4.1.zip` 解压覆盖到原工作台根目录，保留原来的运行时和模型。之后双击 `Start-Workbench.cmd` 即可使用自动更新。该 ZIP 不能单独运行。

如果未来版本需要更换 Python、CUDA 或第三方依赖，更新器会明确要求获取作者提供的新版完整版，**不会在线自动安装或替换 Python 环境**。维护者改变运行环境时必须提升 `updater.RUNTIME_API`，新版完整版同步包含匹配环境。

## 模型下载

v0.4.5 起，「模型与运行环境 → 模型目录」显示当前绝对目录与已保存配置。可输入绝对路径，或相对工作台根目录的路径，支持中文和空格。空闲时保存立即生效，无需重启；任务、下载、校验、诊断或残留进程恢复未完成时禁止切换，请求被拒绝时保留原目录。

模型根目录内放 `hi3d/`、`dino/`、`clip/` 和可选的 `extensions/`。不存在的目录也可先保存，下载需要时才创建。切换和恢复默认都不会移动或删除已有模型；「恢复默认」选择工作台下的 `models/`。设置保存为 `config.local.json` 的 `models_dir`，代码更新保留。相对路径随工作台搬迁；绝对外部路径在搬包后仍需存在。推理前请在选定目录准备并校验模型。

新目录即使已有核心文件，也需点击「重新完整校验」，通过后才显示就绪；保存目录本身不代表文件已验证。

[t8star/Hi3d-Comfy](https://huggingface.co/t8star/Hi3d-Comfy) 提供共用 Hi3D 核心与独立工作台的扩展组件。ComfyUI 只需要核心；建模和贴图所需扩展按选择下载。软件更新不触发全量模型下载，也不改变固定权重版本。

从工作台根目录下载核心：

```bash
hf download t8star/Hi3d-Comfy --local-dir models --include "manifest.json" "hi3d/*" "dino/*" "clip/*" "licenses/*"
```

下载全部核心与扩展（约 79.3GB，另留运行环境和任务空间）：

```bash
hf download t8star/Hi3d-Comfy --local-dir models
```

完整目录与许可见模型仓库双语 README；UltraTex 的 FLUX.2 Klein 9B 底座适用非商业许可。

v0.4.5 的可选加速依赖可单独补装，不更换运行时 ABI：`runtime/python.exe tools/setup_pipeline_runtime.py --runtime runtime-pipeline --attention-only` 安装固定版本原生 Flash，改为 `--uv-only` 安装 Numba 排布加速。代码更新保留 SDPA／Torch 回退，不静默安装这些依赖。导入验证与实际 GPU、模型效果、首次 JIT 耗时分别记录。

v0.4.8 的按需加载与常驻内存属于代码更新，无需下载新模型或运行环境，v0.4.7 EXE 可继续使用。需要腾出内存时点击「释放常驻模型与运行时」；关闭启动窗口同样会结束常驻进程。

v0.5.0 代码更新保留现有运行环境和权重。默认仅复用解释器，各阶段释放权重；也可单独选择受限常驻模式。表面编码缓存按网格、模型与后端校验。可选 CUDA 体素化组件通过网页按钮或 `--voxel-only` 安装，固定哈希，不更换 Torch 或 `o_voxel`。缺失、过期或显卡不支持时，自动模式使用官方 CPU 路径。

ComfyUI 节点通过 Manager、Git 或完整节点 ZIP 单独更新。工作台代码覆盖包只更新工作台，保留旧更新器兼容性，不安装独立节点包。v0.5.0 将旧桥接画布替换为原生 Hi3D 模型加载与粗模上传，无需工作台服务。
