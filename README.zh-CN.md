# Hi3D T8

[English](README.md) | **简体中文**

**By T8star · v0.2.2** — 面向 ComfyUI 与独立本地工作台的 Hi3D V1.0 几何精修工具。

输入为**同一物体的参考图＋粗网格**，输出精修后的几何模型。仅使用官方 Hi3D 第二阶段模型，不生成初始粗模、纹理或 UV，不接入其他 3D 生成模型。

[模型下载](https://huggingface.co/t8star/Hi3d-Comfy) · [画布工作流](examples/Hi3D-Refine.json) · [验证记录](VALIDATION.md) · [Hi3D 原项目](https://github.com/hi3d-official/Hi3Dv1.0)

![ComfyUI 中的 Hi3D 画布工作流](docs/images/comfyui-canvas.png)

## ComfyUI

1. Registry 版本可用后，在 ComfyUI-Manager 搜索 **Hi3D T8**（节点 ID：`hi3d-t8`，Publisher：`t8star`）安装；也可克隆本仓库到 `ComfyUI/custom_nodes/Comfyui-Hi3d-T8`，使用 ComfyUI 自身 Python 安装 `requirements.txt`。
2. 启动**独立的本地工作台后端**，等待 `http://127.0.0.1:8378` 显示模型校验通过，安装方法见[本地工作台](#本地工作台)。
3. 将 [Hi3D-Refine.json](examples/Hi3D-Refine.json) 拖入 ComfyUI **画布**。这是包含节点位置和连线的可视化工作流，不是 API 格式。在 **Load Image** 上传参考图，在 **Hi3D Load Mesh** 填写粗网格的本机绝对路径。
4. 建议从 **512 / 25 步 / CPU 卸载**开始。Windows 选择 `sdpa`，官方 Linux 环境选择 `flash`，点击运行。原生 **SaveGLB** 将结果保存到 ComfyUI 输出目录。

三个自定义节点为 **Hi3D Backend、Hi3D Load Mesh、Hi3D Refine**，使用 ComfyUI 原生 `MESH`。后端与 ComfyUI 运行在同一台电脑；节点包不安装或替换 ComfyUI 的 PyTorch/CUDA。详见[节点说明](docs/COMFYUI.md)。

## 两个版本使用同一套模型

| 组件 | 用途 | 使用范围 |
|---|---|---|
| Hi3D transformer 512＋VAE | 512 精修 | ComfyUI 和工作台共用 |
| Hi3D transformer 1024＋同一 VAE | 1024、1536 精修；没有独立 1536 权重 | ComfyUI 和工作台共用 |
| DINOv2 Large＋CLIP ViT-L/14 | Hi3D 必需图像编码器，不是额外 3D 模型 | ComfyUI 和工作台共用 |

[Hugging Face 模型仓库](https://huggingface.co/t8star/Hi3d-Comfy)中的权重与配置与离线整合包**逐文件相同，未修改或量化**，没有专供 ComfyUI 的另一套模型。Registry / 节点包只包含**代码和画布工作流**；完整离线工作台另含独立 Python、依赖和模型。单独下载权重不等于具备推理环境。

在**工作台根目录**执行，权重无需放进 `ComfyUI/models`：

```bash
hf download t8star/Hi3d-Comfy --local-dir models
```

目录应为 `models/manifest.json`、`models/hi3d/`、`models/dino/`、`models/clip/`。清单记录原始仓库版本与 SHA256。后端也支持直接从官方原仓库下载。

## 本地工作台

Windows 完整离线版双击 **Start-Workbench.cmd**，打开 `http://127.0.0.1:8378`。提供输入预览、手工透明蒙版、CPU 卸载、同步三维对比、历史任务、环境诊断，以及可设置目标高度的 GLB/PLY/STL 导出；可以脱离 ComfyUI 独立使用。

Windows 源码安装使用 Python 3.10 和独立环境：

```powershell
git clone https://github.com/T8mars/Comfyui-Hi3d-T8.git
cd Comfyui-Hi3d-T8
powershell -ExecutionPolicy Bypass -File scripts/Setup.ps1 -Python C:/Python310/python.exe -DownloadModels
```

Linux 执行 `bash scripts/setup-linux.sh`，再运行 `.venv/bin/python -m hi3d_workbench.model_cli download` 和 `.venv/bin/python -m hi3d_workbench --open`。Linux 脚本面向官方 Ampere/Hopper 环境，尚未在本机验收。更多操作见[工作台详细说明](docs/WORKBENCH.zh-CN.md)。

## 验证与限制

实测设备为 **Windows / RTX 5090 Laptop 24GB / 64GB RAM**：完成真实 ComfyUI 512 精修及独立工作台 512/1024/1536 推理；20 轮审计后 **102 项测试**与浏览器回归通过。证据见 [VALIDATION.md](VALIDATION.md)。尚未验证更小显存设备；张量分配峰值不能作为最低显存要求。

Windows 使用明确标识的 **SDPA 兼容适配**，与官方 Linux FlashAttention 环境有区别。Hi3D 会归一化尺寸并重建拓扑；不保证结果封闭、可打印或比粗模更准确，导出前应对比检查。

## 许可与署名

原创集成代码使用 [MIT](LICENSE)；Hi3D 代码与权重使用 **Hi3D Open License v1.0** 并保留原始声明，编码器与运行时分别遵循各自许可。本项目为社区集成，详见 [NOTICE.md](NOTICE.md)。[Powered by Hi3D 1.0](https://www.hi3d.ai/)。

## T8star

[B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [免费画廊](https://www.openzhenzhen.com) · [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
