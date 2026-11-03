# 04 Critic与价值估计

状态：待确认设计草案。推荐结论：保留非对称训练和单标量V头，按V3重建特权字段，重点检查观测可用性与回报时序，不先引入双Critic。

## 当前链与保留内容

官方`149×3=447→512→256→128→1`，当前V3为`98×3=294→同MLP→1`。Critic读取特权历史，不共享actor encoder；不把当前候选action作为Q函数输入。标量V与一套标量GAE是官方基线的实际定义。[来源](../knowledge/official_source_audit.md#4-critic-的真实特权契约)

## V3特权schema

| 字段块 | 建议内容 | 单位/边界 |
|---|---|---|
| 真值状态 | base真实速度、姿态/角速度、A1与轮/转向状态、TCP位置/姿态 | m/s、rad/s、rad、m；frame明确 |
| 真实力与接触 | TCP净外力、单独注入力、接触反力、轮法向力/滑移真值、接触对象状态 | N/Nm，按实际物理可得；标签与reward来源一致 |
| 物理参数 | 质量/负载、COM/惯量、轮半径/转向零位、摩擦、执行器增益/时延 | 每项写实际值还是delta，不能混用 |
| 命令与执行状态 | 原任务命令、补偿目标、上一执行命令、模式、限幅/饱和与可达状态 | 合法训练特权，不泄漏给actor |
| 世界任务 | 对固定表面/物体的相对状态与法向 | 只有该任务开启时纳入；无效字段不补零假装已观测 |

前3帧作为首版保留，Dv由schema求和。当前98维不是目标长度；删除base-force无效字段、增加接触/执行器字段后应修改配置、storage与导出元信息。若某物理参数不随机但固定，记录常量的意义而不标成“已适应范围”。

## 接口、状态与失败

```text
value_obs_t = value_schema.pack(state_t, physics_parameters, task_state_t)
value_t: Float[N,1] = critic(flatten(last_3_value_frames))
rollout stores(value_obs_t, value_t, reward_t, terminated, truncated)
return_target = gae_with_explicit_timeout(final_value_obs)
```

reset清零对应环境的value历史；bootstrap使用真实的终止前下一状态。不能拿reset后的观测补timeout价值。特权中的force-offset目标不能同步送入Actor作为“命令”，否则泄漏外力真值。

## 诊断与检查

记录value loss、explained variance、returns/value均值和分位数、各阶段奖励尺度；按零力/非零力、可达/不可达、接触/离开接触分层看V误差。负载或力课程骤变时，V退化和Actor退化分别检查。合理数值并不单独证明策略成功。

测试seam：构造特定state验证字段位置与缩放；改变隐藏物性只改变critic分支；actor导出不包含特权输入；手算短rollout的V/return和timeout。

## Q&A

| 问题 | 推荐答案 | 用户确认 |
|---|---|---|
| 是否做底盘/机械臂双值头？ | 首版单值头；只有证据显示多任务回报冲突且稳定基线可比较时再做创新消融 | 待确认 |
| 轮滑移/接触真值可以输入Actor吗？ | 仅当有同定义的实机测量/估计接口；仿真真值默认仅训练和评估 | 待确认 |
| COM固定是否足够？ | 可以是诊断条件，但不能写成已验证负载/COM泛化；正式阶段按辨识范围随机化 | 待确认 |
