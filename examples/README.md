# ComfyUI 工作流

1. 启动独立工作台后端，确认 `http://127.0.0.1:8378` 的模型已就绪。
2. 将 `Hi3D-Refine.json` 拖入 ComfyUI。
3. 在 Load Image 中上传参考图，将 Hi3D Load Mesh 的路径改为同一物体的本机 GLB/PLY/STL 文件；示例里的路径是占位值。
4. 默认 512、25 步、CPU 卸载。Windows 选择 `sdpa`；官方 Linux 环境选择 `flash`。
5. 执行后，原生 SaveGLB 把精修网格写入 ComfyUI 的输出目录，工作台同时保留输入、日志和检查报告。

本目录仅交付可拖入画布的 `Hi3D-Refine.json`，包含节点位置、连线和控件值。使用 ComfyUI 的「保存工作流」保存为画布 JSON，勿选择「导出 API」。示例中的图片名与粗模路径是占位值，需要替换为实际输入。

节点只调用 Hi3D；必需的 DINOv2/CLIP 由独立后端管理。节点不会替换 ComfyUI 的 torch/CUDA。工作流不生成初始粗模或纹理。
