# V3 训练碰撞模型优化任务简介

状态：设计与实施完成，静态／物理／性能及四阶段启动门控通过；本机开发配置已切换新资产，远程旧训练保留原身份，新正式训练未启动。

## 目的与范围

用户已确认碰撞模型优化方案并要求实施。当前 V3 A8000 的每轮主要耗时在物理仿真，本任务针对桌面基准派生资产的训练碰撞表示降低计算成本，保持机器人外观、运动学和已确认物性，并继续服务项目第一阶段的仿真复现。

开发根为 `/home/sorry/python_ws/interactive/unifp_reproduction`，开发项目为 `UniFP_v3/`。源资产为当前已验收的 `resources/robots/v3_a1_desktop/`，桌面 ZIP 是外观追溯依据。旧 `v3_a1/` 仅作历史参考，不作为优化来源。

## 现有调用链

`train_v3.main → resolve_v3_configs → V3PosForce → LeggedRobot._create_envs → gym.load_asset/create_actor → apply_desktop_assembly_filters → _initialize_creation_randomization → prepare_sim → Runner.learn`。

`build_run_identity` 对运行源码、实际资产目录和配置建立身份，训练入口拒绝身份不一致的严格 resume。

## 关键约束

- 外观和零位装配以桌面模型为准，优化不得改视觉文件、link/joint 坐标和四轮轴线。
- 保持 28 个 link、14 个驱动 DOF、固定夹爪、TCP 定义及关节顺序。
- 保留已确认质量、COM、惯量、驱动增益及限位；Gym 实际回读也要核实，不能只比较 XML。
- 保持轮地接触、末端接触和非相邻自碰撞，只沿用已有证据支持的局部装配豁免。
- 仿真验证与训练仅在远程专用环境运行；本机不使用 tavis 运行 Isaac Gym。
- 本任务不更改 Actor、Critic、PPO、奖励、随机化范围、策略周期、仿真步长或正式环境数。
- 保留官方副本、当前细分资产、归档及既有训练记录；本轮独立实施和验证，不停止现有训练或启动新正式实验。

## 完成的基本含义

交付独立可追溯的训练资产、必要的最小接入、关键物理与外观一致性证据，以及正式规模的实际耗时报告。资产质量与性能需分别说明；不能用形状数量下降代替物理正确性或完整复现验收。

更换碰撞几何属于新的资产身份，不能将旧模型的 checkpoint 静默视为新资产的严格续训。正式复现仍需按既有阶段和验收协议执行。
