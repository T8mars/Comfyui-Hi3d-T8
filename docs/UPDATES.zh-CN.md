# 程序更新与下载

[English](UPDATES.md) | **简体中文**

GitHub Release 只提供源码、ComfyUI 节点和程序更新包，**不包含模型权重、Python 解释器或运行环境**。含 Python 与模型的完整版由 **T8star 另行提供**，下载地址由作者公布；不要将源码包称为开箱即用整合包。完整版实际包含哪些模型，以该包的 `bundle-manifest.json` 为准。

## 独立整合包自动更新

v0.4.1 起，独立整合包默认检查本项目的正式 GitHub Release。空闲时后台检查并下载代码更新，最多每六小时检查一次；下载通过压缩包和逐文件 SHA256 校验后，**下次关闭并重新启动工作台时安装**。不会在推理过程中覆盖程序或强制重启。

「模型与运行环境 → 程序自动更新」可关闭自动下载和下次启动安装，或手动检查。网络失败不影响已安装程序。安装只覆盖清单中的程序文件，保留 `models/`、`runtime/`、`runtime-pipeline/`、`data/`、`config.local.json` 和 ComfyUI 个人工作流。

安装前备份到 `data/updates/backups/`；写入失败会恢复，断电留下的未完成事务在下次启动时先恢复并重载。若备份损坏，启动会停止并提示检查，不能视为已经恢复。备份不会自动删除。

也可先关闭工作台，再双击 `Update-Workbench.cmd` 检查、下载并安装。更新器拒绝覆盖正在运行的同目录服务。源码 Git 目录禁用自动安装，使用 `git pull` 更新。

## v0.4.0 首次接入

旧版尚无更新器：关闭工作台，将 Release 的 `Hi3D-Workbench-update-0.4.1.zip` 解压覆盖到原工作台根目录，保留原来的运行时和模型。之后双击 `Start-Workbench.cmd` 即可使用自动更新。该 ZIP 不能单独运行。

如果未来版本需要更换 Python、CUDA 或第三方依赖，更新器会明确要求获取作者提供的新版完整版，**不会在线自动安装或替换 Python 环境**。维护者改变运行环境时必须提升 `updater.RUNTIME_API`，新版完整版同步包含匹配环境。

## 模型下载

[t8star/Hi3d-Comfy](https://huggingface.co/t8star/Hi3d-Comfy) 提供共用 Hi3D 核心与独立工作台的扩展组件。ComfyUI 只需要核心；建模和贴图所需扩展按选择下载。软件更新不触发全量模型下载，也不改变固定权重版本。

从工作台根目录下载核心：

```bash
hf download t8star/Hi3d-Comfy --local-dir models --include "manifest.json" "hi3d/*" "dino/*" "clip/*" "licenses/*"
```

下载全部核心与扩展（约 69.9GB，另留运行环境和任务空间）：

```bash
hf download t8star/Hi3d-Comfy --local-dir models
```

完整目录与许可见模型仓库双语 README；UltraTex 的 FLUX.2 Klein 9B 底座适用非商业许可。
