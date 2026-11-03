# 05 PPO、rollout缓冲区与Runner

状态：待确认设计草案。推荐结论：保留官方标量PPO核心；主要改schema、终止时序、辅助监督与完整训练状态，所有偏离有版本和小规模回归。

## 原算法需要保留什么

官方是全动作联合高斯log-prob、一个总reward、一套GAE、一个Critic，并非“底盘/臂分组PPO”。实际配置为clip0.2、gamma0.99、lambda0.95、value系数1、entropy0.01、5epoch×4minibatch、主LR1e-3、自适应KL目标0.01、主梯度norm1、辅助LR1e-5、24步rollout。[源码依据](../knowledge/official_source_audit.md#5-ppogae辅助监督及梯度)

首版保留优化顺序：每minibatch先PPO主更新，再用更新后的encoder前向辅助MSE更新；不detach共享latent。不为了轮式平台另写一个没有对照的算法。

## 具体修改

| 子模块 | 修改内容 | 检查方式 |
|---|---|---|
| policy初始化 | D_actor/D_value/D_label/A从同一schema获取 | 模型、环境、storage hash不一致拒绝加载 |
| storage | actor历史、value历史、label与valid mask、raw action/logp、value、reward、done | shape覆盖N=1和多env；时刻一致 |
| GAE | 保持标量；明确terminated与truncated、rollout截断 | 短序列手算，不把reset后obs做bootstrap |
| timeout | 新环境提供reset前final value obs；用`gamma*V(final_state)`补timeout后切断 | 与官方`V_old(s_t)`近似存在差异，版本化并独立回归 |
| auxiliary | mask与单位/loss系数显式；保留共享encoder双优化器状态 | 非零样本、零力样本分组；梯度归属测试 |
| 动作分布 | raw action正确保存；记录每组物理std、clip/饱和比例 | joint logp数值与sum逐维一致，mean/sample分别评估 |
| 学习率/熵 | 首版以官方值起步，按KL、std与物理探索幅度诊断 | 动作维数改变会改变entropy/KL总和，不能只看相同数字 |
| 日志 | 原单位物理指标、真实辅助MSE、loss与回报独立 | 原辅助返回日志额外除64，不能直接当真实MSE |
| runner | run manifest、课程时钟、所有计数和完整resume | 小规模save/load/update回归 |

## 样本与物理时间预算

| 环境数N | H_rollout | batch | 4 minibatch时每批 | 16000轮transition |
|---:|---:|---:|---:|---:|
| 512 | 24 | 12288 | 3072 | 196608000 |
| 4096 | 24 | 98304 | 24576 | 1572864000 |

相同轮数差8倍样本。增大N会同时改变batch、更新噪声、环境多样性与GPU负担，不能称仅速度变化。记录iteration、批量policy steps、总transitions、每环境仿真秒、wall time和优化器步数；消融优先等transition，再报告时间成本。

若策略dt改变，折扣的物理时域会变，可用`gamma_new=gamma_old**(dt_new/dt_old)`作为保持时间尺度的明确设计；lambda、rollout长度、历史长度、reward dt和课程时长也要同步复核。该公式是设计换算，不是当前配置已改。

## 完整checkpoint与恢复语义

```text
checkpoint = {
  model, ppo_optimizer, auxiliary_optimizer, learning_rates,
  iteration, optimizer_steps, policy_steps, transitions, simulation_time_s,
  curriculum_stage, force_amplitude, scheduler_clock,
  normalizer_state, python_numpy_torch_cuda_rng,
  config, schema_hash, asset_hash, source_hash, runtime_fingerprint
}
```

载入前核验schema/资产/后端；不匹配需显式新实验或迁移映射，禁止strict=False吞掉形状错误。环境状态、接触求解器和history若不完整序列化，清楚标“学习状态恢复、物理重启”，固定warm-up/重采样并排除相应统计；不承诺PhysX位级连续。

官方原runner未保存global_steps与辅助优化器，且循环中保存的iter元数据可能旧；新实现必须以真实当前轮次和独立课程时钟保存，移除`force_start_iteration*24`硬编码依赖。[原事实](../knowledge/official_source_audit.md#7-课程随机化和断点恢复)

## 最小调用框架与Q&A

```text
transition = sample_policy_and_value(obs_t, label_t)
result = env.step(transition.raw_action)
storage.add(transition, result, final_value_obs)
returns = scalar_gae(storage, final_rollout_value)
for minibatch in fixed_versioned_batching_rule:
    ppo_step(minibatch)
    auxiliary_step(minibatch.label, minibatch.valid_mask)
save_full_learning_state(actual_counters)
```

| 问题 | 推荐答案 | 用户确认 |
|---|---|---|
| 马上调低LR/熵能解决退化吗？ | 原因未证实；先记录std、KL、梯度和奖励分项，再做单因素试验 | 待确认 |
| 换大GPU就用同轮数吗？ | 同时报告等transition与wall time，先做容量/吞吐/动力学回归 | 待确认 |
| 修timeout算算法创新吗？ | 算明确记录的实现修正；与官方兼容路径分开验证，不包装成创新 | 待确认 |
