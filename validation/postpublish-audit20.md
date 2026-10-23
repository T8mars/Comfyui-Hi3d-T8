# v0.2.2：发布后新一轮 20 轮联合检查

日期：2026-10-23。基线为 GitHub 发布后的 v0.2.1；本轮由主 Agent 与后端、前端两个子 Agent 分工复现和交叉复审。轮次表示有具体对象和证据的检查，不将同一项重复执行冒充新检查，也不要求每轮都有缺陷。

第 1–10 轮详见 [后端与节点记录](postpublish-audit-rounds-01-10.md)，涵盖发现契约、画布连接、RGBA / MASK、非法图像、中断窗口、取消失败、实际回环 HTTP 超时、进程恢复及模型文件校验。第 14–17 轮详见 [前端记录](postpublish-ui-rounds-14-17.md)。

| 轮次 | 主检查及实际结论 | 证据 |
|---|---|---|
| 11 | 使用标准隔离构建生成真实wheel，再在临时安装目录、独立 `python -I` 进程导入后端与静态资源。基础依赖只有 NumPy/Pillow/trimesh，服务依赖由workbench extra提供，不安装torch。无新缺陷。 | postpublish-wheel-contract.json |
| 12 | 给临时发行目录放入虚构.env、.secrets、根目录个人文件和已删除API示例，旧遍历逻辑会全部装包；现按发行根目录白名单并排除私有/缓存/废弃示例。最终审计也拒绝私有、逃逸及重复条目。 | test_postpublish_packaging.py，实际临时ZIP内容断言 |
| 13 | 旧真实推理脚本仍读取已删除的API文件、硬编码8388端口，512路径已损坏。改为调用可配置地址的真实画布加载/保存/执行检查；模型上传工具的离线比较路径跟随当前版本，保留显式路径覆盖。 | run_inference_validation.py --help；本轮真实512画布验收 |
| 14 | 成功任务缺少完整报告时隐藏现存精修模型或报TypeError。现依据工件预览导出，缺失指标为—，不伪造几何通过。 | postpublish-ui-baseline.json / postpublish-ui-result.json |
| 15 | 单条日志请求失败会误判全站离线；过期日志错误也影响新任务。现把失败局限于对应日志区，允许重试并核对epoch。 | 同上；abort、404、503及晚到错误均实际回归 |
| 16 | 旧历史请求失败覆盖新成功列表，旧finally提前解锁新请求。现成功/失败/收尾均绑定historyEpoch。 | 同上；逆序返回与刷新按钮锁定断言 |
| 17 | 快速恢复笔刷中段漏画，擦除单击无效。现使用连续胶囊区域和单击圆笔刷；原始透明/半透明仍保持，取消指针后不续画。 | postpublish-mask-result.png及像素alpha断言 |
| 18 | verify/download启动检查存在并发窗口；现共用服务锁覆盖启动。另复现诊断超时后继承stdout的子进程令清理继续阻塞；交叉复审继续发现父进程早退的残留窗口，已纳入修复与真实临时子进程回归。 | test_postpublish_service.py；故障进程仅在临时ROOT创建 |
| 19 | 复数NPZ顶点被静默截断；极大坐标溢出GLB float32，极小坐标整体坍缩，缩放导出也可能溢出。现输入与缩放后均明确拒绝，保留原mesh。 | postpublish-geometry-proof.json；缩放前后原顶点相等断言 |
| 20 | 收紧QA归属：只等待当前画布实际提交返回的prompt_id，不接受其他并发成功记录；样例图使用唯一文件名，避免覆盖现有input。最终核对新节点/源码/完整包、画布格式、官方源码与模型哈希，并按不可变版本发布v0.2.2。 | 本轮真实画布结果与dist/RELEASE-VERIFIED-0.2.2.json；Registry发布回执 |

故障注入均在临时ROOT或明确HTTP夹具中完成，不替换生产模型、输入或历史；界面夹具不计为真实模型推理。模型权重、官方Hi3D源码与采样数学未改变。冻结后整套 **102项测试通过**（46.34秒，唯一警告为Starlette/AnyIO依赖弃用），见postpublish-pytest.txt。真实推理及发行状态以单独验收回执为准。

集成验收已使用新 `dist/Hi3D-Workbench-Windows-0.2.2/runtime/python.exe` 完成：实际画布加载、保存、重载后执行512 / 25步 / seed 0 / CPU卸载 / SDPA；prompt `92959748-211c-485c-9b63-9a9f12f81842` 对应job `f815e82833d4474e9a8ededa733cef33`，画布全过程339.17秒，worker313.36秒，原生GLB18,007,616字节，零JS错误。详见 [回执](v022-comfy-canvas-result.json)、[截图](v022-comfy-canvas.png)、[报告](v022-inference-report.json)。本轮没有重测1024/1536；旧记录保留版本范围。

实际推理时GPU诊断409；空闲后工作台模型加载/蒙版/800px布局冒烟通过，零JS错误和外部HTTP请求。并行冒烟曾因运行时输入禁用而超时，该次不计通过。6条旧历史任务完整复制并保留旧目录。Registry新版安装与CDN返回200、14文件归档的节点源码与2abf900 Git对象完全相同，官方Action成功，平台状态仍Pending；详见registry-publication-v0.2.2.json。最终ZIP内容和SHA以dist/RELEASE-VERIFIED-0.2.2.json为准。
