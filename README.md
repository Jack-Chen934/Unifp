# UniFP 在 V3 轮式机器人上的仿真复现

基于官方 UniFP，将统一力位控制方法迁移至 Ranger Mini V3＋A1Arm。当前开展第一阶段仿真复现；完整验收并经用户确认之后，才进入轮式机器人特有创新。

**2026-11-09：首版受力复现冻结为`B1`分支。** V6 r2 seed1 A8000已通过；B已实现并持续训练，存在力信息学习与低谷后恢复，尚未通过完整受力／接触验收。B1绑定17800轮完整checkpoint，运行观察截至20:38累计17923；这不是实时进度。冻结不停止原参数训练。见[B1总结、参数与优化方向](docs/releases/B1/README.md)和[统一进度](docs/progress/2026-11-06_v3_status.md)。

## 项目结构

| 路径 | 用途 |
|---|---|
| `UniFP/` | 冻结官方代码快照，基线提交 `68847a070f88d731058c3d8476929bc3b205f5bd` |
| `UniFP_v3/` | 基于官方方法开发的V3环境、Actor／Critic、PPO、训练与评估入口、测试和机器人资产 |
| `UniFP_go2w/` | Go2-W＋Piper独立迁移目录；当前为既有记录、可行性方案与设计讨论入口，尚未实现训练代码 |
| `introduction/` | 官方完整流程、Actor、Critic和PPO的中文讲解 |
| `.scratch/unifp-v3-stage1-reproduction/` | 已确认的分模块知识库、设计、差异记录和实施票据 |
| `.scratch/unifp-v3-stage1-reproduction/knowledge/model_history/` | 五项已结束模型任务的创建记录、必要证据与旧资产来源清单 |
| `.scratch/unifp-v3-failure-visualization/` | 保留的策略视频、接触分类与可视化工具 |
| `docs/progress/` | 研究进度、训练快照与版本发布说明 |
| `UniFP_v3/docs/experiments/legacy_reference/` | 旧迁移有用资料汇总、来源与检查点索引；原 `legacy/` 已整合删除 |

## 复现路线与当前边界

1. **A：前8000个训练迭代的基础复现。** 零命令末端力／零人工末端外力；底盘运动及push按冻结协议处理。每迭代每环境24个策略步，不是总共8000个环境step。
2. **B：后32000个训练迭代的受力复现。** 承接A学习状态，累计40000迭代。A终点评估与B准备条件均通过才接续。
3. 三个独立种子完成全部六类仿真验收，并由用户确认后，才开展项目第二阶段创新。

A01—A11设计和A票据01—07已完成，V6 seed1正式A评估通过。B04—B11设计与首版实施已完成，正式B seed1继续原预算到累计40000；seed2／3尚未启动。单seed A通过不代表整个项目验收完成。

## 运行与资产

训练只在远程服务器运行。本机`tavis`为通用开发环境，不能视作项目Isaac Gym训练环境。远程使用Python3.8、PyTorch2.3.1+cu121与Isaac Gym Preview4；SDK需自行安装，配置和启动条件见[开发副本说明](UniFP_v3/README.md)与[启动记录](.scratch/unifp-v3-stage1-reproduction/knowledge/model_history/unifp-v3-collision-optimization/knowledge/训练启动记录.md)。克隆代码后需要另行准备对应环境和设备。

当前训练资产为`UniFP_v3/resources/robots/v3_a1_desktop_train_v6_coplanar_r2/`：28 links、14 DOF、约83.318kg。质量、惯性与执行器参数属于仿真基线，尚未实机标定。桌面基准、V6、r3／V8诊断资产保留，8组旧中间资产已按用户要求删除，创建记录和文件哈希保留于主线知识库。详见[实验汇总](UniFP_v3/docs/experiments/v3/实验汇总.md)。

## 文档入口

- [阶段知识库、设计与实施入口](.scratch/unifp-v3-stage1-reproduction/README.md)
- [A阶段实施票据总览](.scratch/unifp-v3-stage1-reproduction/tickets/part-a/overview.md)
- [A／B衔接计划](.scratch/unifp-v3-stage1-reproduction/development-plan.md)
- [论文和官方实现差异](.scratch/unifp-v3-stage1-reproduction/comparison/A04-论文与官方源码差异.md)
- [研究术语及阶段边界](CONTEXT.md)
- [开发规则](AGENTS.md)
- [Git收录范围、上游来源与历史备份](docs/progress/版本归档说明.md)
- [B1冻结身份、结果、可优化范围与验证入口](docs/releases/B1/README.md)

官方来源：[unified-force/UniFP](https://github.com/unified-force/UniFP)，论文：[arXiv:2505.20829](https://arxiv.org/abs/2505.20829)。原有许可证保留在各自目录。
