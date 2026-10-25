# Hi3D — Standalone ComfyUI nodes

[English README](../README.md#comfyui) | [中文 README](../README.zh-CN.md#comfyui)

**By T8star.** 在 ComfyUI 自身 Python 进程内执行官方 Hi3D 的「同物体参考图＋粗模→精修几何」。无需打开或安装独立工作台，不使用工作台 HTTP 接口。节点包含固定版本官方 Python 源码及许可，不包含模型和 Python 运行环境。

## 安装与模型

用 ComfyUI 自身的 Python 安装节点目录的 `requirements.txt`，重启 ComfyUI。保留现有 Torch/CUDA；需 NVIDIA 计算能力 8.0+、Triton、spconv 与图像编码器依赖。默认 `sdpa` 可使用 PyTorch 融合注意力；已有兼容 FlashAttention 时可选 `flash`。

默认模型根目录：`ComfyUI/models/Hi3D/`。其下放置 `hi3d/`、`dino/`、`clip/`，与工作台所用的原始 Hi3D 核心权重完全相同。`Hi3D Model Loader.model_root` 可指定其他根目录；留空时也支持 `extra_model_paths.yaml` 的 `hi3d` 路径。开启 `download_missing` 才会从固定版本模型仓库补齐核心，无默认后台下载。

```bash
hf download t8star/Hi3d-Comfy --local-dir ComfyUI/models/Hi3D --include "manifest.json" "hi3d/*" "dino/*" "clip/*" "licenses/*"
```

## 上传与工作流

导入 [Hi3D-Refine.json](../examples/Hi3D-Refine.json)，这是带位置和连线的画布 JSON。先在 `Load Image` 上传参考图，再点击 `Hi3D Load Mesh` 的**选择文件上传粗模**，或把 GLB / PLY / STL 拖到该节点；上传后自动选中，并保存在 `ComfyUI/input/hi3d/`。已有输入文件可直接下拉选择。

连接为：Model Loader → Refine；Load Image 的 IMAGE／可选 MASK → Refine；Load Mesh 的 MESH → Refine → 原生 SaveGLB。默认 512、25 步、`cpu_offload=true`、`sdpa`。原生 SaveGLB 保存到 ComfyUI 输出目录。`report_json` 提供分辨率、步数、种子及耗时。节点不写工作台历史。

只支持单张图片和单物体 batch，MASK 的 1 表示透明；已有 RGBA 透明度保留。Hi3D 输出归一化的几何，会重建拓扑，不保留原 UV 或材质，不负责初始粗模或贴图。

默认每次精修后释放模型；`keep_model_loaded` 可复用连续任务，但会占用模型内存。ComfyUI 中断标志会在采样更新及阶段边界检查，异常同样清理模型；控制台显示加载组件与实际采样步数。

旧版桥接画布需重新导入新版示例；不再使用 `Hi3D Backend`。许可证见节点的 `hi3d_native/vendor/` 与根目录 NOTICE。验证范围见 [VALIDATION.md](../VALIDATION.md)。
