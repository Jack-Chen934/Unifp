# 03 · Actor、Critic与辅助状态估计

状态：已完成（2026-10-31）。02环境／观测／action契约已完成；V3使用独立`V3ActorCritic`，官方`ActorCritic`保留供B2使用。A05/A06固定schema、网络shape、输出和梯度职责均已实现，并通过单测、Preview4配置装配及Python语法检查；尚未接入PPO训练，也未启动训练。

## 目标与范围

在官方模型结构上落实已确认A05/A06接口：58维单帧Actor观测、32帧历史与1856维输入，单标量Critic、98维帧与294维三帧历史，14维动作及已确认辅助预测标签；保留官方主／辅助两阶段更新关系和A阶段零力标签语义。

## 代码位置与接口

- 主要来源：`UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py`及V3任务模型装配。
- 新实现目标：`UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/`及V3 actor-critic配置。
- 接口shape、参数组、latent与辅助输出沿A05/A06设计记录，不增加额外head或损失。

## 测试与验收

- 单测：哨兵字段偏移、输出shape、辅助梯度路径及确定性动作均值。
- 验收：B2模型行为不变；V3前向／反向有限，核心shape与schema精确一致。

## Blocked by

- 01-v3-repo-and-task-scaffold。
- 02-v3-environment-contract（观测、标签与动作合同）。

## 实施记录与验证

- 新增`UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_actor_critic.py`，固定Actor输入1856、Critic输入294、辅助输出12、单帧58、动作14；实现64维共享latent、已确认Actor／Critic／encoder／decoder层宽、ELU、14维可训练对角高斯及确定性推理。
- V3配置选择`V3ActorCritic`；Runner导入该类以供配置名解析。B2继承配置仍选择原`ActorCritic`，未改官方模型实现。
- `get_student_latent`返回可微的12维辅助预测；辅助预测使用同一history encoder latent。策略输入仅拼接最新58维和latent。Critic单独接收294维特权输入。
- 验证通过：`PYTHONPATH=UniFP_v3 conda run -n tavis python -m pytest -q UniFP_v3/tests/test_v3_actor_critic.py`（3 passed）；三处Python文件`py_compile`通过；Isaac Gym Preview4/Python 3.8/Torch 2.0.1下按官方导入顺序核验V3/B2类解析及V3配置参数构造通过，参数量1,685,033。
- Preview4装配检查未创建环境或运行物理step；模型仍未由PPO rollout/update调用。票据05负责训练链集成。
