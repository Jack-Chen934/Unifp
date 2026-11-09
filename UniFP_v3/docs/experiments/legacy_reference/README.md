# 旧 Ranger 迁移参考资料汇总

整理日期：2026-11-08。原目录为项目根目录下的 `legacy/`。本次按用户要求保留有用资料、核验重复副本后删除原目录；本目录是历史证据归档，不是当前开发版本或可直接运行的训练包。

## 当前还有什么用

- 旧模型构建脚本、URDF、厂家 CAD 和网格来源说明，支持追溯 Ranger 轮组划分、坐标重定位、质量与惯性估算。
- 旧控制、观测、奖励、环境桥与评估源码，支持查证早期实现差异；当前业务代码不应从这里导入模块。
- 512 环境固定 COM 训练的报告、配置、恢复验证和检查点，支持追溯早期训练及失败结论。
- 原文件清单和 SHA256，支持确认历史资料身份。历史路径通过本次整合清单查找保留位置。

当前 V3 运行资产是 `UniFP_v3/resources/robots/v3_a1_desktop_train_v6_coplanar_r2/`；Go2W/Piper 最终资产是 `UniFP_v3/resources/robots/go2w_piper_unifp_v3/`。本次未修改这两套资产、官方副本或训练配置。

## 历史训练结论

旧 Ranger Mini 3＋A1Arm 实验采用 512 个环境、每环境 24 步、seed `266281`，固定标称 COM；每轴受力课程上限为 ±60 N，自 global policy step `192001` 开始。累计完成 16000 轮，384000 个策略步。

16000 轮两验证 seed 的均值如下，均是原报告中的历史指标，本次没有重跑：

| 动作方式 | 受力偏移目标末端误差 | 估力 RMSE | 同条件恒零估计 RMSE |
|---|---:|---:|---:|
| mean | 50.348 cm | 27.702 N/轴 | 17.713 N/轴 |
| sample | 45.795 cm | 11.271 N/轴 | 17.846 N/轴 |

辅助估计器学到部分受力信息，但策略没有稳定地将其转化为底盘和末端控制。该终点属于失败诊断参考，不能称为整体恢复或当前项目验收通过。旧训练恢复会重建 PhysX、任务状态和历史缓冲，不能称为物理仿真完全连续。

## 资料入口

- [原训练报告](evidence/v3_reference_20261030/PROJECT_REPORT.md)
- [结构化实验摘要](evidence/v3_reference_20261030/ranger_512_migration/SUMMARY.json)
- [历史训练与评估源码](evidence/v3_reference_20261030/ranger_512_migration/logic/)
- [模型构建脚本](evidence/v3_reference_20261030/robot_description/a1arm_description/scripts/)
- [旧近似模型说明](evidence/v3_reference_20261030/robot_description/a1arm_description/README_approx_sim.md)
- [模型来源资料](evidence/v3_reference_20261030/robot_description/a1arm_description/)
- [逐文件保留、去重与删除清单](consolidation_manifest.json)
- [归档文件校验清单](SHA256SUMS.json)

`evidence/` 中的原件保持字节不变，可能包含当时的绝对路径、英文原文或已失效链接。原文中“保留”“可恢复”等表述仅描述历史整理时的状态；此次目录不是原布局的完整副本。旧路径的当前去向以 `consolidation_manifest.json` 的 `source`、`destination` 和 `action` 为准。已确认有现存副本的文件不在本目录重复存储。

## 保留与删除范围

整理前有 197 个普通文件，合计 619795625 字节（约 591.1 MiB），另有 36 个软链接。当前代码中没有发现以 `legacy` 为运行依赖的调用，也没有外部软链接指向该目录。

- 独立保留 115 个有用文件，共 244367921 字节（约 233.0 MiB），包括不可替代的历史源码、报告、原始几何、元数据以及 8000、9000、10000、16000 轮四份不同的训练检查点和一份 shape 属性快照。
- 52 个文件经 SHA256 逐项验证后去重，清单指向本目录或当前项目已有的保留副本；包括重复的 8000 轮检查点。
- 删除 30 个非活动 USD 导入产物。来源与构建资料仍可追溯，旧 USD 不属于当前仿真资产。
- 删除原目录中的 36 个内部软链接，仅保留其映射记录，不在新归档生成失效链接。
- 删除 `legacy/`，净减少约 358.0 MiB 的普通文件数据；最终释放空间以文件系统统计为准。

检查点继续仅本地留存，受现有 `*.pt`、`checkpoints/`、`logs/` Git 忽略规则保护。本次不启动旧训练，不加载旧权重续训，不把历史结果记作当前 V3 或 Go2W 的验收结果。
