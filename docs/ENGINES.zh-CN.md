# 工作台引擎与质量

[English](ENGINES.md) | **简体中文**

ComfyUI 节点仍只复现 Hi3D。下列选择只加入独立工作台。建模与贴图分阶段选择；贴图不能修正错误轮廓、缺失部件或变形的脸。

| 阶段 | 引擎 | 采样设置 | 适用方向 |
|---|---|---|---|
| 建模 | Hunyuan3D-2mini Turbo | 256／384／512，固定五步 | 较低开销的初稿 |
| 建模 | Hunyuan3D 2.1 | 256／384／512 | 可选的完整几何模型 |
| 建模 | TRELLIS.2 | 512／1024 级联 | 较细的图片条件几何，可整理生成表面 |
| 贴图 | TRELLIS.2 PBR | 512／1024 | 根据已有网格生成颜色、金属度和粗糙度 |
| 贴图 | Hunyuan3D-Paint 2.1 | 512／1024 | 六视角 PBR 生成与烘焙 |
| 贴图 | UltraTex，FP8 Base 9B | 颜色 1024／2048，MR 最高 1024 | 原先的八位浮点底座；六视角颜色与材质生成 |
| 贴图 | UltraTex，INT8 ConvRot Base 9B | 颜色 1024／2048，MR 最高 1024 | 固定 tsolful 发布的 Base 量化，原生 `int8_tensorwise`；不是 distilled/KV |

输出贴图尺寸可选 1024、2048、4096。贴图尺寸与生成采样分辨率分别控制不同阶段：将输出改为 4K 不会凭空补出预测中没有的细节。更大模型、更多步数也不一定更好。

工作台提供三个起点：轻量为 Mini Turbo 256＋TRELLIS PBR 512／2K；均衡为 TRELLIS 几何 512＋PBR 1024／4K；细节为 TRELLIS 几何 1024＋UltraTex 1024／4K。UltraTex 2048 颜色生成可手动选择，开销更高，MR 最高 1024；其底座适用非商业许可。Hunyuan 2.1 的建模和贴图保留为独立备选。这些是建议起点，不是通用质量排名；应使用同一几何、同一灯光比较后选择贴图器。

在“模型”页按引擎下载固定版本依赖，或选择“补齐全部引擎”。缺少可选模型不会禁用已安装的引擎。v0.4.0 基础包仅带基础组合；v0.4.4 完整配置包含 Hi3D 核心及全部 14 个扩展组件，覆盖三个建模和四个贴图选项，准确清单见 `bundle-manifest.json`。完整配置的模型文件约 79.3 GB，Python 与运行环境另计。UltraTex 的 FP8 底座与 INT8 ConvRot 底座分别来自原始 FP8 发布和 [tsolful 的 Base 9B 量化发布](https://civitai.com/models/2738890?modelVersionId=3080042)，均未由本项目重新量化。INT8 与 FP8 同为八位格式，不保证 INT8 更省显存或更快。

比较贴图时，在“已有模型贴图”中使用完全相同的 GLB 与参考图。可用的单材质 UV 会保留，多材质下有歧义的 UV 会重新展开。报告记录实际贴图输入网格哈希；历史对比会说明是同图同网格，还是只相同图片、不同几何。选择前应检查正面、背面、接缝和至少两种灯光。预览灯光与曝光只改变显示。

0.4.0 修复两类着色法线丢失问题。原生 UV 展开压缩／重排顶点后，通过精确坐标和有向邻面恢复原始索引，包括重合的硬边顶点。无法证明对应关系时退回原始索引 CPU Atlas；完全分离的三角面直接走此路径，避开原生空邻接内核。GLB 场景导入还会保留文件自带的 `NORMAL`，用逆转置变换法线，对镜像实例反转面绕序；TRELLIS 贴图阶段复制网格时同样保留这些数据。相关 CPU 回归已覆盖，导入修正后的最终推理重跑单独记录在[验证记录](../VALIDATION.md)。

贴图器需要单一、不重叠的单位图集；退化、平铺、重复三角 UV 和累计面积超出图集的明显重叠会重展，任意局部重叠尚未做完整检测。已有错误法线的 GLB 不会自动修复；若错误法线参与了贴图条件生成，应从原始几何重新贴图。`tools/restore_shading_normals.py` 可针对有对应原始几何的结果修正法线，且验证位置、面、UV 和材质像素不变；这属于导出修正，不是重新生成贴图。

TRELLIS 建模可同时保留原始生成的 `generated.glb` 和整理后的 `coarse.glb`。整理采用官方不透明物体导出的窄带表面重建方式，目标约 100 万面；细小开放结构可能改变，应对照原版。Hi3D 精修独立可选，会重建拓扑。

当前实测与待验收项见[质量记录](../validation/quality-upgrade-progress.json)。对比使用原创狐狸、机械猫头鹰、探险宝箱和天文台参考图，固定种子并保存输出哈希。Hunyuan 2.1 的首个狐狸结果有与主体连接的大平面，猫头鹰没有复现此问题，不能直接认定它全面优于轻量版。TRELLIS 1024 改善了狐狸脸部和铠甲，背部毛发与尾巴仍有伪影。这些是有限样本，不是通用模型排行榜。

实测设备为 Windows、RTX 5090 Laptop 24GB、64GB RAM。狐狸 1024 几何运行 378.718 秒，NVML 全设备显存峰值 13034.965MiB，包含驱动和桌面占用。约 100 万面的猫头鹰使用 1024 TRELLIS PBR 耗时 212.728 秒，PyTorch 分配／保留峰值分别为 11021.04／17892MiB。以上均不能作为最低显存要求；CPU 卸载也需要足够系统内存。

原项目：[TRELLIS.2](https://github.com/microsoft/TRELLIS.2)、[Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)、[UltraTex](https://github.com/yiboz2001/UltraTex)、[UltraTex Windows 适配](https://github.com/visualbruno/ComfyUI-UltraTex)。Hunyuan Paint 使用官方相机、条件与烘焙代码，Windows 光栅适配使用 nvdiffrast，UV 空洞使用 OpenCV 修补，未启用上游可选 C++ 顶点修补。UltraTex 调用固定版本 Comfy 推理库，不启动 Comfy 服务或新增节点。许可见[第三方声明](../NOTICE.md)。
