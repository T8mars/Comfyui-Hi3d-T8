# 20 轮联合审计：子 Agent 第 1–10 轮

审计对象：E:/hi3d 源码、现有 Windows 0.2.0 包的只读模型元数据。此次修复计划交付 0.2.1，未修改官方 Hi3D 代码、模型权重或生产任务。所有进程测试均使用临时 ROOT 中的短小假脚本，不执行 CUDA 推理。

| 轮次 | 实际检查 | 证据与结论 |
| --- | --- | --- |
| 1 | 本地 API 的 Token、Origin、Host、路径边界 | 原 Token / 跨站拒绝测试通过。发现生产 TrustedHost 允许测试域名 testserver，已由主 Agent 移除；TestClient 改显式回环地址，新回归验证非回环 Host 返回 400。 |
| 2 | 推理分辨率、步数、seed 及未就绪模型提交 | 既有回归验证 256 分辨率返回 400，steps=0 / seed 超限返回 422，缺模型返回 409；未发现需要改变模型参数范围的问题。 |
| 3 | 图片 alpha、外部 GLB 资源、NPZ 非法数值、导出尺寸和原件保护 | 既有回归确认 alpha 保留，外部资源 / 非法顶点拒绝，80mm 导出 STL 为 80、GLB 为 0.08m，源粗网格字节不变；没有为审计更改几何算法。 |
| 4 | 同时提交、GPU 诊断与校验期间的任务槽 | 事件控制的隔离测试证明校验期间重复提交与 GPU 诊断均返回 409，首个请求只创建一次；活动推理也拒绝 GPU 诊断。 |
| 5 | worker 输出损坏或状态写入异常时的进程生命周期 | 真实隔离复现：坏 HI3D_EVENT 使旧代码先标 failed 并清空 active，但假 worker 仍存活。修复后坏 JSON / 非字典 / NaN 进度只写日志；其他异常先回收并等待进程退出，再释放槽。新增落盘异常回收测试。 |
| 6 | 取消、PID 重用、启动恢复、损坏任务记录 | 取消 / 恢复共用 PID、创建时间与完整 request argv 核对，错误身份不终止。损坏历史 JSON / schema 不再拖垮列表和启动；原坏文件保留并给出 recovery_errors。坏记录仍有匹配的工作进程时不误杀，recovery_blocked 阻止新 GPU 任务，真实退出后解除。交叉复审又补上 Popen 已成功而 PID 尚未落盘的崩溃窗口：合法 queued / running / cancelling 无 PID 记录也扫描 exact request argv。再次重启后的 interrupted 记录，以及终态已写而进程尚存的 failed / cancelled / succeeded 记录也保持阻塞，不误杀未确认的进程。每次恢复只抓一次进程快照，并实时排除已安全终止或 PID 已复用的旧快照；正常成功回收不会错误报告阻塞。三种无 PID 状态、连续重启、三种终态残留和恢复落盘失败均以隔离假 worker 回归；Windows venv 启动器及其 Python 子进程均纳入检查与安全清理。 |
| 7 | 极短进程、解释器不存在、thread.start 失败、持续磁盘错误 | 短进程退出早于 psutil 查询仍按真实退出码和工件判定；无法启动线程不遗留 active。取消时即使所有 state 写入持续失败，也终止 / 等待进程并清槽，内存状态仍显示 cancelled。活动 state 文件损坏不丢失真实进程归属。 |
| 8 | 模型文件校验绑定、固定 repo / revision 与模型元数据损坏 | 文件 stat / NTFS ChangeTime 绑定和内容变化检测通过。发现旧校验接受任意 revision；现拒绝非固定 repo / revision。manifest / verification / download JSON 或 schema 损坏不再崩溃，显示异常；损坏清单和校验记录不能使模型就绪。重新写入前保留坏文件备份。仅 download 状态损坏时，已完成的真实模型校验仍有效，但明确显示 interrupted 和异常。 |
| 9 | 下载断点与非 LFS 配置完整性 | 既有分片损坏清理、外部下载锁、重试测试通过。发现配置 JSON 的 Hub 缓存可能被损坏后重新写入本地 manifest；现用官方 Git blob_id 核对非 LFS 文件，失败强制重取，仍失败则拒绝。只读核对固定 commit 的 11 个 Git 配置 / 通知文件全部相符，详见 audit20-model-provenance.json。 |
| 10 | ComfyUI 网格桥接与启动边界 | 原 padding / 单物体约束通过。发现浮点面索引会被 int64 静默截断；现拒绝浮点索引、非法有效数量、越界面和非有限顶点。节点拒绝非回环 / 非 HTTP 地址的测试在网络前完成。只读核查启动器优先使用包内 runtime，源码安装固定 Hi3D commit；未在本轮重新安装 CUDA 依赖。 |

后端冻结及补充交叉修复后，全套独立新 Python 进程执行 `.appenv/Scripts/python.exe -m pytest -q`：**70 passed**，唯一 warning 来自 Starlette / AnyIO 的弃用别名。全套测试保留临时 ROOT 隔离保护。主 Agent 将继续第 11–20 轮浏览器 / 发行审计并交叉审阅上述修复。

本轮验证的是错误处理、并发与交付可靠性；不将假 worker 结果写成模型成功记录，也不把此次审计当作新增硬件性能或模型质量测评。
