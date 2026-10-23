# Hi3D v0.2.1 · 20 轮联合检查

日期：2026-10-23。主 Agent 与子 Agent 分工检查、交换发现并交叉复审。每轮是一个明确检查对象及其复现 / 验证闭环，不是把 20 次重复运行计作 20 个不同结论。没有为了凑轮数制造问题。

第 1–10 轮的逐项过程见 [后端记录](audit20-backend-rounds-01-10.md)，覆盖认证 / 路径、参数、输入 / 尺寸、GPU互斥、异常worker、取消 / 恢复、线程 / 磁盘、模型版本 / 元数据、下载缓存、ComfyUI桥接与启动。发现的问题均已修复；第2–4轮既有行为通过复核。

| 轮次 | 实际复现与改动 | 证据 |
|---|---|---|
| 11 | HTTP500 后仍显示模型就绪且可提交；修复为暂停操作并自动重连。请求有超时，另验证挂起请求与残留进程阻塞提示。 | baseline中run_disabled=false；result中true，超时与恢复断言通过 |
| 12 | 选中的运行记录消失后，按钮永久锁定；404 / 损坏记录400在没有相同活动任务时回到新任务，不删除原文件。 | baseline中new_task_enabled=false；result中true；额外400通过 |
| 13 | 手动重连把选择的flash改回sdpa；只在首次会话应用默认值。 | baseline flash→sdpa；result flash→flash |
| 14 | 导出80mm期间可把显示高度改成100mm，返回尺寸链接与界面错配；导出期间冻结格式 / 高度 / 来源。 | baseline参数未锁且显示100；result参数锁定且显示80 |
| 15 | toBlob编码期间尚未锁输入，关闭对话框后可换图，再被旧蒙版覆盖；编码开始即锁定并检查输入epoch。 | baseline input_locked=false；result true |
| 16 | 相机同步漏了平移，重置也保留偏移；按包围盒中心与尺度同步目标，重置回auto。 | baseline两侧目标不同且reset保留偏移；result平移收敛一致、reset=auto |
| 17 | 新网格尚在下载时显示上一份模型；加载 / 失败期间隐藏旧scene并显示占位提示。 | baseline旧scene可见且无占位；result旧scene隐藏且占位可见 |
| 18 | 历史任务工件缺失仍请求预览并允许复用；依实际available_files设置视图与复用资格。 | baseline两项均错误；result缺失预览隐藏、reuse禁用 |
| 19 | 升级资源没有版本缓存键；增加HTML/API no-store与CSS/JS版本键。这项为发布加固，不将缺少缓存键本身冒充用户已遇到的故障。 | result CSS/JS?v=0.2.1；API响应头测试 |
| 20 | 原归档审计只比对6个自有文件，不能发现旧jobs.py或旧ComfyUI节点；改为全量自有代码及节点内容比对。 | test_release_audit.py：CRC有效但源文件变旧仍被拒绝；最终归档记录在dist/RELEASE-VERIFIED-0.2.1.json |

`audit20-browser-baseline.json` 记录修复前真实 Chrome 中第11–18轮的八项失败；`audit20-browser-result.json` 记录修复后第11–19轮全部通过、零JavaScript错误。浏览器故障响应与临时假worker不执行Hi3D推理，也不向生产历史伪造成功记录。浏览器新增故障检查使用临时ROOT；模型完整文件不用于损坏测试。

后端冻结时全套70项pytest通过。第3轮另发现上传CPU检查占用事件循环，补充并发健康请求测试；可选诊断JSON损坏不再禁用整个health。固定官方revision的11个配置 / 通知文件Git blob核对通过，见audit20-model-provenance.json。

发布保留旧版本与原任务。最终完整包的CRC、自有代码、61个官方Python文件、16个模型工件SHA与排除个人数据检查，以发行目录中的最终审计JSON为准。当前硬件仍限先前实测设备，不把错误处理测试当作新显卡兼容或效果保证。
