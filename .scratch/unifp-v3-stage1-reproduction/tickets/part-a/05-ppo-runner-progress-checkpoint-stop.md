# 05 · PPO、Runner训练时钟与完整检查点

状态：已完成（2026-10-31）。A07补充Q8已获用户确认并实施；本机24项回归、远程512×10及新进程恢复1轮均通过。恢复首步异常终止从512/512修复为0/512，I/S/C、双Adam200→220、物性与父哈希核验一致。未启动正式8000轮。

## 目标与范围

在官方PPO/Runner基础上落实A07双优化器、rollout、timeout处理、完整更新计数、主辅学习状态恢复、RNG恢复、间隔／终点保存以及A恰好8000轮停止；错误时不把不完整更新提交为完成轮次。

## 代码位置与接口

- 来源：`UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py`、`on_policy_runner.py`及storage。
- 新目标：对应`UniFP_v3/`路径。
- 接口：`set_training_progress`、`get_training_state`、`load_training_state`、完整checkpoint save/load，字段以A07/A09设计为准。

## 测试与验收

- 单测：小张量GAE代表例、timeout分支、提交计数、A8000停止、完整checkpoint round-trip与恢复后一次更新。
- 整体：512环境10迭代；新进程恢复再完成至少一轮；不要求仿真逐比特重现。
- 验收：I/S/N×S一致；8000后不采下一动作；身份不兼容明确失败。

## Blocked by

- 01-v3-repo-and-task-scaffold。
- 02-v3-environment-contract。
- 03-actor-critic-and-auxiliary-estimator。
- 04-reward-randomization-and-episode-metrics（训练状态字段对接）。

## 完成证据与后续边界

- 实现：V3专用Storage／PPO／Runner／identity及真实训练入口；创建级物性在prepare前应用，load只核验并恢复可变状态。
- 单测：19项pytest＋5项奖励／状态unittest；GAE、timeout、完整保存／恢复、缺失动量拒载、原子发布故障及8000精确停止边界通过。
- 整体：r06的512×10及独立r03恢复1轮退出0；I10→11、S240→264、C122880→135168，主辅Adam200→220，恢复首步物理终止0。
- 记录：`UniFP_v3/docs/experiments/2026-10-31_v3_ppo_runner_checkpoint.md`；摘要`knowledge/a07_training_smoke_20261031.json`，原始证据在`knowledge/a07_remote_20261031/`。
- 额外回归：远程8环境27步checker通过push、实际物性、可变状态和timeout合同；未把手动预置超时当自然轨迹。
- 下一票据06可执行；4096容量、A10阈值和候选动态稳定性仍未验收，普通短训策略的真实物理失败记录保留，不据此宣称复现有效果。
