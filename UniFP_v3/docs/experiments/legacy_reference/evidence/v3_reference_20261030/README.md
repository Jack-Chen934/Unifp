# Ranger Mini 3＋A1Arm 的 UniFP 复现

唯一进度、问题分析、实验比较与下一步计划见 [PROJECT_REPORT.md](/home/sorry/python_ws/interactive/unifp_reproduction/PROJECT_REPORT.md)。

UniFP 官方源码学习文档位于 `introduction/`，按顺序阅读：[完整训练流程与两个阶段](/home/sorry/python_ws/interactive/unifp_reproduction/introduction/00-training-flow-zh-CN.md)、[Actor](/home/sorry/python_ws/interactive/unifp_reproduction/introduction/01-actor-zh-CN.md)、[Critic](/home/sorry/python_ws/interactive/unifp_reproduction/introduction/02-critic-zh-CN.md)、[PPO](/home/sorry/python_ws/interactive/unifp_reproduction/introduction/03-ppo-zh-CN.md)。

正式 Lab 策略冻结于 10000 轮（±15 N）；当前诊断主线为 Gym、±60 N、名义 COM 的行为覆盖与残余估力误差定位。最新完成原报告 8.83 等量回放对照，尚未获得可替换正式策略的方案。

2026-10-25 已清理旧训练大数据并归并散落报告。清理/保留清单与历史原文位于 `archive/cleanup_20261025_v1/`；旧实验完整性记录只证明清理前状态。训练、诊断与资产源代码未改，本次未启动训练。
