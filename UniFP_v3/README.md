# UniFP V3 开发目录

基于官方提交`68847a070f88d731058c3d8476929bc3b205f5bd`开发。原官方快照位于`../UniFP/`，本目录随根项目仓库统一版本管理。

截至2026-11-03，A阶段票据01—07及设计A01—A11已完成对应实现与启动验证。新碰撞资产seed1 A8000已于17:35 CST在远程GPU1从零启动，尚未完成终点评估。最新进度见[研究进度](../docs/progress/2026-11-03.md)，设计和票据见[阶段入口](../.scratch/unifp-v3-stage1-reproduction/README.md)。

## 环境与入口

训练只在远程进行，环境为Python3.8／PyTorch2.3.1+cu121／Isaac Gym Preview4；启动需`PYTHONUTF8=1`。本机`tavis`是通用开发环境，不是本项目仿真训练环境。本机另有Preview4短测环境缓存。Isaac Gym SDK和权重不随Git仓库分发。

- 正式训练：`legged_gym/scripts/train_v3.py`。
- 启动门控：`legged_gym/scripts/preflight_v3.py`。
- 冻结评估：`legged_gym/scripts/evaluate_v3.py`。
- 参数解析：`legged_gym/scripts/v3_args.py`。
- A阶段冻结manifest：`configs/v3_a_initial_v1.json`。
- 当前已启动实验的精确环境、身份和命令：[启动记录](../.scratch/unifp-v3-collision-optimization/knowledge/训练启动记录.md)。训练存活时不要重复启动。

当前资产为`resources/robots/v3_a1_desktop_train_v1/`，从冻结桌面基准独立派生，碰撞927→80 shapes，保留视觉、运动链、28 links、14 DOF与实际actor物性。`v3_a1_desktop/`保留修复基准；`v3_a1/`存在已记录的历史问题。内部候选manifest保持验证时字节，最终验证状态记录在资产外部文档。

## 验证和实验记录

最终碰撞优化r02源码下48项本机和5项远程单测、物理回归、512×10＋跨进程续1轮、4096×100及四阶段启动门控通过。4096规模的V3接触对容量为`2**25`。短测通过不能证明最终追踪效果；A阶段仍需S0／S1／S2正式评估。

- [碰撞优化与性能验证](docs/experiments/2026-11-03_v3_collision_optimization.md)
- [桌面资产修复](docs/experiments/2026-10-31_v3_desktop_asset_repair.md)
- [奖励和随机化](docs/experiments/2026-10-31_v3_rewards_randomization.md)
- [PPO、恢复与检查点](docs/experiments/2026-10-31_v3_ppo_runner_checkpoint.md)
- [A阶段票据总览](../.scratch/unifp-v3-stage1-reproduction/tickets/part-a/overview.md)

旧正式训练已在1423轮停止，最近完整权重为1400轮，完整数据保留远程，不能计入新资产A8000验收。B受力复现与seed2／3未启动，第一阶段全部验收前不开展轮式创新。
