# ComfyUI-Hi3D

By T8star · v0.2.2

只复现 Hi3D V1.0 的参考图＋粗网格几何精修。此节点小包不含推理运行时或权重；需要同机的 Hi3D 独立工作台后端。

1. 从完整离线整合包运行 `Start-Workbench.cmd`，确认 `http://127.0.0.1:8378` 显示模型就绪。
2. 将本文件所在的 `ComfyUI-Hi3D` 文件夹放入 ComfyUI 的 `custom_nodes`。
3. 用 ComfyUI 自身的 Python 安装此文件夹的 `requirements.txt`，然后重启 ComfyUI。它只要求轻量节点依赖，不安装或替换 torch/CUDA。
4. 拖入 `examples/Hi3D-Refine.json`。Load Image 上传参考图；Hi3D Load Mesh 填写同一物体的 GLB/PLY/STL 本机绝对路径。
5. 先执行 512 / 25 步 / CPU 卸载。Windows 后端选 `sdpa`，官方 Linux 后端选 `flash`。三个节点均在 Hi3D 分类下。

示例是 ComfyUI 画布保存格式，包含节点位置、连线与控件值；发布包不含 API 格式示例。已在 ComfyUI 0.34.0 / 前端 1.53.6 中完成加载、保存、重新加载和真实 512 / 25 步执行，原生 SaveGLB 输出通过，浏览器零 JavaScript 错误。更换占位图片名与网格路径后即可运行。

ComfyUI 与独立整合包共用同一套原始 Hi3D / DINOv2 / CLIP 权重，无专用量化版或第二套模型。权重与配置可从 https://huggingface.co/t8star/Hi3d-Comfy 下载，放在工作台的 models 目录，不是 ComfyUI/models。完整安装与中英双语说明：https://github.com/T8mars/Comfyui-Hi3d-T8 。

连接为 Backend → Load Mesh；IMAGE、可选 MASK、原生 MESH → Refine → 原生 SaveGLB。Hi3D 只支持单张参考图、单物体 batch；MASK 的 1 表示透明区域。推理前请求 ComfyUI 卸载模型，中断会请求后端停止任务。

Refine 输出精修 MESH、报告 JSON 和 job ID。结果、输入及日志同时保存在工作台历史中。工作台本身可独立使用，不要求打开 ComfyUI。seed 范围为 0–9007199254740991。

节点只连接本机回环 HTTP 地址，绕过系统代理。后端使用固定版本 Hi3D 与必需 DINOv2/CLIP；不接入其他 3D 模型，不生成初始粗模、UV 或纹理。Windows 是明确标识的 SDPA 兼容移植；实测范围和局限见完整工作台的 VALIDATION.md。

代码许可与署名见 LICENSE、NOTICE.md。Hi3D 本身受其自定义许可约束。社区集成；[Powered by Hi3D 1.0](http://hi3d.ai/)。
