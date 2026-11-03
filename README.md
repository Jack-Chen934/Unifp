# UniFP 在 V3 轮式机器人上的仿真复现

基于官方 UniFP，将统一力位控制方法迁移至 Ranger Mini V3＋A1Arm。当前开展第一阶段仿真复现；完整验收并经用户确认之后，才进入轮式机器人特有创新。

**截至2026-11-03：A阶段实现与启动前验证完成，新碰撞资产 seed1 的8000轮正式训练正在远程运行，尚未完成效果验收。** 最新进度与证据见[研究进度记录](docs/progress/2026-11-03.md)，训练快照是定时读取结果，不是实时状态。

## 项目结构

| 路径 | 用途 |
|---|---|
| `UniFP/` | 冻结官方代码快照，基线提交 `68847a070f88d731058c3d8476929bc3b205f5bd` |
| `UniFP_v3/` | 基于官方方法开发的V3环境、Actor／Critic、PPO、训练与评估入口、测试和机器人资产 |
| `introduction/` | 官方完整流程、Actor、Critic和PPO的中文讲解 |
| `.scratch/unifp-v3-stage1-reproduction/` | 已确认的分模块知识库、设计、差异记录和实施票据 |
| `.scratch/v3-desktop-model-repair/` | 桌面基准资产修复的决策与验证记录 |
| `.scratch/unifp-v3-collision-optimization/` | 训练碰撞优化、旧训练归档摘要、新模型图片及运行证据 |
| `docs/progress/` | 研究进度、训练快照与版本发布说明 |
| `legacy/` | 旧迁移参考入口；完整旧代码和大数据仅在原本机归档保留 |

## 复现路线与当前边界

1. **A：前8000个训练迭代的基础复现。** 零命令末端力／零人工末端外力；底盘运动及push按冻结协议处理。每迭代每环境24个策略步，不是总共8000个环境step。
2. **B：后32000个训练迭代的受力复现。** 承接A学习状态，累计40000迭代。A终点评估与B准备条件均通过才接续。
3. 三个独立种子完成全部六类仿真验收，并由用户确认后，才开展项目第二阶段创新。

A01—A11设计和A实现票据01—07已完成；S0／S1／S2评估场景及候选阈值已冻结。当前只有新资产seed1在训练，B与seed2／3尚未启动。奖励上涨或短测通过不能替代正式评估。

## 运行与资产

训练只在远程服务器运行。本机`tavis`为通用开发环境，不能视作项目Isaac Gym训练环境。远程使用Python3.8、PyTorch2.3.1+cu121与Isaac Gym Preview4；SDK需自行安装，配置和启动条件见[开发副本说明](UniFP_v3/README.md)与[启动记录](.scratch/unifp-v3-collision-optimization/knowledge/训练启动记录.md)。克隆代码后需要另行准备对应环境和设备。

当前训练资产为`UniFP_v3/resources/robots/v3_a1_desktop_train_v1/`：28 links、14 DOF、约83.318kg，碰撞shape由927减少至80，保留桌面模型外观、运动链与已确认物性。质量、惯性与执行器参数属于仿真基线，尚未实机标定。`v3_a1/`为存在已知问题的历史资产；`v3_a1_desktop/`是修复后冻结基准。详见[碰撞优化实验](UniFP_v3/docs/experiments/2026-11-03_v3_collision_optimization.md)。

## 文档入口

- [阶段知识库、设计与实施入口](.scratch/unifp-v3-stage1-reproduction/README.md)
- [A阶段实施票据总览](.scratch/unifp-v3-stage1-reproduction/tickets/part-a/overview.md)
- [A／B衔接计划](.scratch/unifp-v3-stage1-reproduction/development-plan.md)
- [论文和官方实现差异](.scratch/unifp-v3-stage1-reproduction/comparison/A04-论文与官方源码差异.md)
- [研究术语及阶段边界](CONTEXT.md)
- [开发规则](AGENTS.md)
- [Git收录范围、上游来源与历史备份](docs/progress/版本归档说明.md)

官方来源：[unified-force/UniFP](https://github.com/unified-force/UniFP)，论文：[arXiv:2505.20829](https://arxiv.org/abs/2505.20829)。原有许可证保留在各自目录。
