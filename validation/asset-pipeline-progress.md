# v0.3.0 资产链开发验收记录

范围：仅独立整合包增加图片建模与 PBR；ComfyUI 节点输入输出与原有画布工作流保持几何精修范围。真实引擎、工作台 HTTP 与随包运行时分别记录，模拟测试不计为真实推理。

## 当前证据

- `.appenv/Scripts/python.exe -m pytest tests -q`：154 项通过，61.06 秒，退出码 0。测试在临时根目录隔离，不改用户任务或模型。唯一 warning 为 Starlette 测试依赖的弃用提示。
- 12 组 Chrome 显式 HTTP 夹具流程通过，0 JS 错误；含模式所需组件、图片模式不传 mesh、复用缺粗模任务、真实 PBR 材料不被预览覆盖、失败阶段导出、默认跳过精修与显式完整链参数和 390px 布局。见 [asset-pipeline-ui.json](asset-pipeline-ui.json)。这些不是生成质量测试。
- Windows RTX 5090 Laptop / Torch 2.8 cu128 实际执行：packed SDPA、CuMesh UV、O-Voxel 网格体素输入、nvdiffrast 栅格、FlexGEMM 三维采样、Triton 稀疏卷积，共 6 项通过。见 [开发环境证据](asset-pipeline-native-gpu.json)。
- 复制后的独立 `runtime-pipeline/python.exe` 同样 6 项通过；清除 CUDA_PATH/HOME，强制包内 TCC 和新 Triton 缓存，确认无需借用开发 venv 或系统 CUDA SDK。独立 prefix、Python.h 与 python312.lib 查找通过。见 [随包运行时证据](asset-pipeline-portable-native-gpu.json)。这仍不等于完整模型推理验收。
- DINOv3 公开分发版本已实际下载，1212559808 字节，SHA256 `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`，与官方固定版本一致；保留原始许可、官方及下载源版本。

## 修复的边界

GLB 任意外部图片引用（包括法线）被拒绝；UV、内嵌颜色与金属度/粗糙度图片须真实可解码；无材质灰模不能报告贴图完成。80mm GLB 缩放保留材质和 UV，源模型不被覆盖，STL/PLY 不冒充材质格式。

新增坏模型清单、Windows 等长篡改恢复时间戳、阶段身份损坏、父监督进程已退出而子阶段仍存活、跨阶段取消竞态等回归。仅杀匹配请求和创建时间的已知进程；未确认身份的残留进程继续阻止 GPU 操作。

## 首个真实图片技术链

同一张原创兔子图片先由 Hunyuan 实际生成基础网格，再接 Hi3D、TRELLIS.2：分别 51.817 / 173.12 / 92.084 秒。最终 954901 面，内嵌 2048 颜色、金属度/粗糙度贴图与 UV；贴图前后最大顶点差 4.28e-8，80mm GLB 材质与尺寸复核通过。详见 [asset-pipeline-real-chain.json](asset-pipeline-real-chain.json)。

此处为开发环境各引擎的真实串接；随包工作台 HTTP 同链也已完成，见下一节。PyTorch 贴图峰分配 2842.29MiB；NVML 验证期整卡峰值 3976.199MiB，后者包含驱动、桌面与其他进程，只代表本次 24GB 真机观察。

独立真实六视图检查显示局部改善与恶化并存，见 [几何复核](asset-pipeline-geometry-review.md)。卡通嘴部精修过硬与衣服颜色漂移会保留说明，不用面数或单张截图代替效果判断。

同图同种子直接贴图对照也已完成：嘴部更干净，衣服颜色更接近参考，因此图片入口默认跳过 Hi3D，用户仍可勾选。已有机械网格的 1024／4K PBR 实跑 258.948 秒，保持 999999 面，最大坐标误差 4.47e-8；UV 与两张 4096²图片解码通过。PyTorch 分配峰值 6763.41MiB，NVML 整卡峰值 11560.707MiB。这是已有网格贴图案例，不能描述为机械图生模型实测。

## 随包工作台 HTTP 验收

`runtime/python.exe` 启动工作台，阶段使用包内 `runtime` / `runtime-pipeline` 与固定本地模型，无开发 venv：

- 建筑透明图片→基础模型→PBR：327.15秒；758788面、两张2048贴图，真实UI加载/版本切换/相机同步/80mm保材质导出通过。
- 兔子不透明图片→CPU前景→基础模型→Hi3D512/25步→PBR：594.00秒；阶段分别133.142／319.16／121.941秒。整卡NVML峰3952.574MiB（含桌面等），不是最低显存证明。此为worker编排耗时，不含提交前完整SHA校验。
- 已有网格贴图任务在真实采样阶段取消，2.281秒后无匹配残留进程；复用原输入建立新任务，实际贴图与80mm材质导出成功，原取消记录保留。
- UV接缝会拆分顶点，修正前报告曾把它们算成几何碎片。现只在诊断副本中合并完全相同坐标，GLB不变；闭合UV箱体、真实开口、HTTP上传和历史复用均有回归。建筑旧报告保留后重算；详见 [诊断修正](asset-pipeline-topology-correction.json)。
- v0.2.2的15个旧任务及相关工件共358文件无覆盖复制到新目录，旧目录保留；新服务无恢复错误。

图片默认跳过Hi3D。实际卡通、建筑、机械六视图质量局限见 [材质效果复核](asset-pipeline-texture-review.md)。全部扩展模型23文件（11866423657字节）离线哈希复核通过。

已有模型精修＋PBR也已通过：清除SDK变量/路径、使用包内TCC和全新Triton缓存，真实运行Hi3D512／25步与TRELLIS.2，材质检查和80mm导出通过。四个实际HTTP用例的完整回执见 [asset-pipeline-portable-http.json](asset-pipeline-portable-http.json)。最终7组真实UI检查通过，另有12组模式夹具和5组标注边界检查；均0JS错误。

发行归档的CRC、源码一致性、原模型与23个扩展文件完整性、SHA256目录由 `tools/audit_release.py` 检查，外部回执为 `dist/RELEASE-VERIFIED-0.3.0.json`，与归档及SHA256目录一起交付。
