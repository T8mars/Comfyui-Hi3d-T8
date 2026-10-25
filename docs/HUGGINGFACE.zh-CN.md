# Hi3D T8 — 核心与工作台模型

[English](README.md) | **简体中文**

**By T8star。** 为 [Hi3D T8 本地工作台与 ComfyUI 节点](https://github.com/T8mars/Comfyui-Hi3d-T8)整理的完整模型组件。保留原始权重、配置、固定版本与各自许可；UltraTex 的 fp8 底座沿用上游发布，未由本项目重新量化。

**ComfyUI 只使用 Hi3D 核心，输入参考图＋粗模，输出精修几何。** 单图建模与 PBR 扩展仅用于独立工作台。本仓库不含应用或 Python；GitHub Release 只提供代码，含 Python 与模型的完整版由 T8star 另行提供。

## 包含哪些模型

| 目录 | 用途 | 许可 |
|---|---|---|
| `hi3d/` | Hi3D V1.0 精修；512／1024 权重，共用 VAE；1536 使用 1024 权重 | Hi3D Open License v1.0 |
| `dino/`、`clip/` | 核心必需的 DINOv2 Large、CLIP ViT-L/14 编码器 | Apache-2.0／MIT |
| `extensions/coarse/` | Hunyuan3D-2mini Turbo 图片建模 | 腾讯 Hunyuan 3D 2.0 社区许可 |
| `extensions/coarse_hunyuan21/` | Hunyuan3D 2.1 图片建模 | 腾讯 Hunyuan 3D 2.1 社区许可 |
| `extensions/coarse_trellis2/`、`extensions/trellis_structure/` | TRELLIS.2 建模与结构解码器 | MIT |
| `extensions/texture/`、`extensions/texture_encoder/` | TRELLIS.2 PBR 与 DINOv3 编码器 | MIT／DINOv3 License |
| `extensions/texture_hunyuan21/`、`extensions/paint_dino/`、`extensions/paint_upscaler/` | Hunyuan3D-Paint 2.1、DINOv2 Giant、Real-ESRGAN | 腾讯／Apache-2.0／BSD-3-Clause |
| `extensions/texture_ultratex/`、`extensions/ultratex_backbone/`、`extensions/ultratex_text_encoder/` | UltraTex 适配器与解码器、FLUX.2 Klein 9B fp8、Qwen3 与 VAE | MIT／FLUX 非商业／Apache-2.0 |
| `extensions/ultratex_backbone_int8/` | tsolful 的 Klein Base 9B INT8 ConvRot；固定 CivitAI 版本 3080042／文件 2959286 | 原始 FLUX 非商业许可 |
| `extensions/background/` | 可选 CPU U2Netp 抠图 | MIT／Apache-2.0 |

`manifest.json` 记录共用核心；`extensions/manifest.json` 记录 14 个扩展组件及来源版本；`model-catalogue.json` 列出全部 **95 个权重／配置／许可文件，合计约 79.3GB**，含文件大小与 SHA256。ComfyUI 与工作台共用同一 Hi3D 权重，没有专用的另一套模型。INT8 文件为社区发布的 **Base** 量化，不是蒸馏版或 KV 版，本项目未重新量化；FP8 版单独保留。

## 下载方式

在**工作台根目录**执行，模型无需放入 `ComfyUI/models`。

仅下载 Hi3D 核心，供 ComfyUI 精修使用：

```bash
hf download t8star/Hi3d-Comfy --local-dir models --include "manifest.json" "hi3d/*" "dino/*" "clip/*" "licenses/*"
```

下载全部核心与扩展（约 79.3GB，另留运行环境和任务空间）：

```bash
hf download t8star/Hi3d-Comfy --local-dir models
```

按需下载时，选择上表对应目录、`extensions/manifest.json` 及配套依赖。工作台的「模型」页也可从固定版本的上游或公开分发源只下载所需引擎。缺少可选模型不影响已安装的引擎；使用前在工作台中校验。

使用其他磁盘或共享目录时，在「模型与运行环境 → 模型目录」选择根目录，并将 `hf download --local-dir` 指向同一路径。所选目录内直接放置 `hi3d/`、`dino/`、`clip/`、`extensions/`。空闲切换立即生效，无需重启，不会移动原有文件；相对路径按工作台根目录解析，代码更新保留配置。

新位置即使已有核心文件，也需在工作台执行「重新完整校验」，通过后才显示就绪。

## 程序自动更新

独立整合包检查正式 GitHub Release，下载代码更新后在下次启动安装，保留模型、Python、配置和历史任务。程序更新不会下载整个模型仓库；需要更换运行环境的版本会提示获取作者提供的新版完整版。[更新说明](https://github.com/T8mars/Comfyui-Hi3d-T8/blob/main/docs/UPDATES.zh-CN.md)。

v0.4.5 增加 UV 修复／缓存、阶段与 CPU／GPU 进度，以及可选原生 FlashAttention／Numba 运行时轮子。**模型权重与这份95文件清单没有变化。** 加速轮子属于 Python 环境依赖，不是新增模型，不放在本模型仓库。程序更新保留旧环境及 SDPA／Torch 回退；新版完整版与补装命令见[性能说明](https://github.com/T8mars/Comfyui-Hi3d-T8/blob/main/docs/PERFORMANCE.md)。

## 验证与限制

实测 Windows／RTX 5090 Laptop 24GB／64GB RAM，未验证更低显存设备。Hi3D 精修可选，会重建拓扑；纹理由独立 PBR 阶段生成，带材质结果使用 GLB。需检查形状、背面、接缝和材质，不保证可打印或每次均改善。[引擎与质量记录](https://github.com/T8mars/Comfyui-Hi3d-T8/blob/main/docs/ENGINES.zh-CN.md)。

## 许可与署名

各组件保留独立许可，不把整个集合声明为 MIT。Hi3D 原 LICENSE／NOTICE／AUTHORS 保留；DINOv3 保留 Meta 许可；UltraTex 的 **FLUX.2 Klein 9B 底座适用非商业许可**。完整文本见组件目录与 `licenses/`，来源版本见清单。[Powered by Hi3D 1.0](https://www.hi3d.ai/)。

## T8star

[B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [免费画廊](https://www.openzhenzhen.com) · [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
