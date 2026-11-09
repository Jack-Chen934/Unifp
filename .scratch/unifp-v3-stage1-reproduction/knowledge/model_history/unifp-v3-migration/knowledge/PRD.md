# UniFP 到 Ranger Mini V3 + A1Arm：任务简介

日期：2026-10-30（Asia/Shanghai）。

## 开发目的与范围

用户要求把 `/home/sorry/python_ws/interactive/gaze_vr_isaaclab/AGENTS.md` 原样复制到当前研究根目录，先根据已完成的 B2+Z1 实验判断 UniFP 复现的有效范围，再结合当前 V3 迁移源码和官方完整流程，形成总体及分模块迁移计划，并分析主要差异、难点与可验证的创新方向。

本轮交付为文件复制、源码与实验证据调研、评估报告和待确认的迁移规划；不修改训练业务代码、不启动或延长训练、不操作机器人或关机。

## 当前代码与资料位置

- 当前研究根目录：`/home/sorry/python_ws/interactive/research`。
- 官方流程说明：`/home/sorry/python_ws/interactive/unifp_reproduction/introduction/00-training-flow-zh-CN.md`、`01-actor-zh-CN.md`、`02-critic-zh-CN.md`、`03-ppo-zh-CN.md`。
- 官方本地实现及当前迁移：`/home/sorry/python_ws/interactive/unifp_reproduction`，包括 `UniFP/`、`ranger_512_migration/` 及 Ranger 环境、资产和控制文件。
- B2+Z1 复现实验依据：`literature/projects/unifiedforce/reproduction/outputs/stage1_20261029/` 和 `stage2_20261029/`。
- 现有较早迁移说明：`literature/projects/ranger-mini-3-a1arm-unifp-migration-plan-zh-CN.md`；该文日期为 2026-10-19，需要与当前资产核对后才能使用其中结论。

## 关键约束与注意事项

- 区分实验事实、源码事实、推断和待确认设计。
- 复现有效不等于官方全部结论已复现；奖励、位置误差、力估计与实际接触力控制采用各自口径。
- 官方文档作为导航，实际源码和已保存数据作为实现事实；不能用过时资产结论覆盖当前版本。
- 当前复制的规范中的 `tavis` 环境说明来自源项目，本轮不据此切换研究或远程实验环境。
- 真实机器人参数、执行器带宽、传感器和力真值来源未核实的部分明确标记，不编造为已知值。
- 迁移方案须覆盖环境与动作、Actor、Critic、PPO/Storage/Runner、力模型与奖励、课程与随机化、评估、部署及模块依赖。

## 完成的基本含义

根目录规范与源文件字节一致；核心结论可追溯到实验结果或源码位置；给出明确的迁移进入条件、模块边界和分阶段检查项；设计与实施建议标为待确认，不声称已经完成实现或通过实验。
