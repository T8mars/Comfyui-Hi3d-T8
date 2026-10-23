# Hi3D V1.0 — T8 模型包

[English](README.md) | **简体中文**

**By T8star。** 为 [ComfyUI-Hi3D T8 与本地工作台](https://github.com/T8mars/Comfyui-Hi3d-T8)整理的官方 Hi3D V1.0 权重和必需图像编码器。逐文件校验，未修改、微调或量化；属于社区分发，不是新模型。

Hi3D 以**同一物体的参考图＋粗网格**为输入进行几何精修，不生成初始粗模、UV 或纹理，不包含其他 3D 生成模型。

## 内容与来源

| 目录 | 原模型 | 固定版本 | 许可 |
|---|---|---|---|
| `hi3d/` | [hi3d-official/Hi3Dv1.0](https://huggingface.co/hi3d-official/Hi3Dv1.0) | `705fc0901c5d896cdcbf1dbf1feaa173e62bdd3a` | Hi3D Open License v1.0 |
| `dino/` | [facebook/dinov2-large](https://huggingface.co/facebook/dinov2-large) | `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c` | Apache-2.0 |
| `clip/` | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) | `32bd64288804d66eefd0ccbe215aa642df71cc41` | MIT |

`manifest.json` 记录每个模型工件的 SHA256 和字节数，权重与配置与离线工作台**完全一致**。DINOv2 和 CLIP 是 Hi3D 必需图像编码器。`transformer_512` 用于 512 精修，`transformer_1024` 同时用于 1024 和 1536，没有独立的 1536 权重。

## 下载与使用

在**工作台根目录**执行：

```bash
hf download t8star/Hi3d-Comfy --local-dir models
```

下载后应为 `models/hi3d/`、`models/dino/`、`models/clip/` 和 `models/manifest.json`。启动配置完成的工作台后端并等待 SHA256 校验；在 ComfyUI 中导入[画布工作流](https://github.com/T8mars/Comfyui-Hi3d-T8/blob/main/examples/Hi3D-Refine.json)。此集成不要求把权重放入 `ComfyUI/models`。

**ComfyUI 与独立工作台共用模型和推理后端。** Registry 节点包包含代码与画布工作流，完整离线工作台另含独立 Python、依赖与权重。本仓库仅提供模型和许可，不是可直接启动的软件。安装见 [GitHub 中文说明](https://github.com/T8mars/Comfyui-Hi3d-T8/blob/main/README.zh-CN.md)。

## 验证与限制

在 Windows / RTX 5090 Laptop 24GB / 64GB RAM 上完成真实 ComfyUI 512 和工作台 512/1024/1536 推理，尚未验证更小显存设备。建议从 512、25 步、CPU 卸载开始。Windows 使用明确标识的 SDPA 兼容适配，官方 Linux 路径使用 FlashAttention。详见[验证记录](https://github.com/T8mars/Comfyui-Hi3d-T8/blob/main/VALIDATION.md)。

输出只有几何。Hi3D 会归一化尺寸并重建拓扑，不保证封闭、可打印或必然优于粗模。

## 许可

Hi3D 文件保留 [Hi3D Open License v1.0](LICENSE) 的署名、可接受使用与规模条款，以及原始 `hi3d/LICENSE`、`hi3d/NOTICE`、`hi3d/AUTHORS`、`hi3d/THIRD_PARTY_LICENSES.md`。编码器许可保存在 `licenses/`；不将整个模型包标为 MIT。[Powered by Hi3D 1.0](https://www.hi3d.ai/)。

## T8star

[B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [免费画廊](https://www.openzhenzhen.com) · [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
