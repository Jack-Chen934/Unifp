# B-T03 · 非零力学习、Critic补偿与奖励统计

状态：已完成；45项相关测试通过。

## 目标与范围

让真实受力标签进入现有网络和主辅更新，完成补偿Critic槽、补偿reward及几何／补偿日志区分。复用A已有数学实现；已有正确部分只补非零输入验证，不重复重写。

## 开始时必读

B05、B06、B08-Q1／Q2、B11-U2／U4；B-T02的标签时序与实际测试结果。

## 代码位置与接口

相对于`UniFP_v3/`：

- `legged_gym/envs/v3/legged_robot_v3.py`：辅助[N,12]的人工力yaw标签×0.01；Critic单帧[N,98]末三维为补偿目标球坐标；`_compensated_target_world`、reward、`_physical_tracking_errors`及episode／policy trace。
- `legged_gym/b2_gym_learn/ppo_cse_pf/v3_actor_critic.py`、`v3_ppo.py`、`v3_rollout_storage.py`：主要核验现有路径，只有与设计不符才最小修改；`v3_episode_log.py`补必要B字段，不改A统计定义。

公式固定`p_comp_world=p_geo_world+(F_ext_world+R_yaw F_cmd_yaw)/200`，单位m；无额外裁剪。几何目标、补偿目标与误差分别命名，修正把补偿误差误标几何误差的日志别名。23项reward及权重沿用，无第二Critic或第二GAE。

Actor58×32=1856、Critic98×3=294、latent64、辅助12、动作14、命令15不变；接续所有网络参数与std。主Adam更新encoder／Actor／Critic／std、不含decoder；辅Adam只encoder／decoder，主后辅重新前向，四组MSE沿用[0.04,0.04,1,1]等价权重和标签缩放，不擅自加非零mask／重采样。接触力不回填人工力标签、Actor或reward。

## 测试与验收

扩展`tests/test_v3_actor_critic.py`、`test_v3_ppo.py`、`test_v3_rewards.py`及必要storage／日志测试，真实网络与Adam先红后绿。

- 非零标签产生有限有效梯度，参数更新边界及主后辅重新前向正确；A全零用例继续通过。
- 固定yaw下±20N抵消，单路20N→0.1m；Critic球坐标与世界补偿点对应，无未来真值进入Actor。
- reward使用刚执行命令／外力；几何／补偿曲线和物理单位标签清楚。相同输入不重复叠加force缩放。
- 单测通过不宣称外力估计已学会；真实11轮训练与容量放B-T08。

## Blocked by

- B-T02完成。

## 完成记录

2026-11-07：复用既有补偿reward、网络、storage和双Adam；新增共享world→task球坐标转换，修正Critic 95:98，零合力精确保留原命令；误差扩为6项独立统计，trace保留补偿世界目标并按当前转移生成监督标签。真实PPO分别用零力／非零力完成20次主与辅助更新，参数／Adam归属继续符合设计。45项相关测试通过，证据`knowledge/part_b_implementation/bt03/`。未据此宣称策略已学会受力；下一票B-T04。
