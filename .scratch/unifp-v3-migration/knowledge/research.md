# UniFP → V3迁移：源码与实验证据索引

核查日期：2026-10-30。本文件只汇集本次已读取证据，设计和实施计划分别在`../design/`与`../tickets/`。

## 官方方法

- [官方源码事实审计](official_source_audit.md)：已完整阅读用户指定introduction四篇，再核对实际Actor、Critic、PPO、storage、runner、环境与配置；包含路径、行号、配置和SHA256。
- 本地官方Git HEAD为`68847a070f88d731058c3d8476929bc3b205f5bd`，工作树干净；这是本地源码版本，不自动认定远程训练归档的commit。
- [论文缓存](../../../../research/literature/projects/unifiedforce/assets/paper.html)：论文`arXiv:2505.20829v2`，Sec3.1/3.2的力位与网络定义、Sec4.1评估、Appendix C.4训练曲线。

## 已有复现实验

- [B2实验事实与统计口径](reproduction_source_audit.md)：无力、独立受力、奖励曲线与测试边界。
- [本轮float64原始数组复算](reproduction_validation.json)：三个已有NPZ的shape、有限性、指标和SHA256；没有新训练或推理。
- [已有官方曲线/分布对照](../../../../research/literature/projects/unifiedforce/reproduction/outputs/stage2_20261029/official_comparison/analysis_report.md)：原作者图像与本实验数据并列，附数据位置和未公开统计的边界。

## 当前V3迁移

- [V3源码与现存归档事实](v3_source_audit.md)：当前可动资产、两后端路径、动作/观测/特权/监督/奖励/课程、mean/sample结果、施力点、清理后路径与哈希问题。
- `/home/sorry/python_ws/interactive/unifp_reproduction/ranger_512_migration/SUMMARY.json`与`PROJECT_REPORT.md`：旧512/固定COM/16000冻结结果；关键原始轨迹已经清理的范围见源码审计。
- `/home/sorry/python_ws/interactive/unifp_reproduction/introduction/00-training-flow-zh-CN.md`、`01-actor-zh-CN.md`、`02-critic-zh-CN.md`、`03-ppo-zh-CN.md`：用户指定的官方完整流程说明。

## 本轮文件操作

`/home/sorry/python_ws/interactive/gaze_vr_isaaclab/AGENTS.md`已经逐字节复制为当前根目录`AGENTS.md`，`cmp`通过，SHA256为`025a1f3286bc797166960dc324379eabde79c70ca2a84c6a151cec180aa6f80c`。

本轮操作仅涉及规范复制、只读调研、已有数组重算与文档输出，没有修改训练业务代码、启动仿真/训练、远程命令或硬件操作。
