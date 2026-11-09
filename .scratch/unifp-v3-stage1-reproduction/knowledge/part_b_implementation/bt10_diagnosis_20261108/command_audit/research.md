# 实际源码依据

- `UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:30`：COMMAND_SCALE前9为速度／球坐标／姿态缩放，命令力9:15统一0.01。
- 同文件`:1649—1706`：Actor58维=roll/pitch2+angvel3+位置偏差10+速度14+last_actions14+commands15；命令力9:12位于Actor52:55，球坐标目标3:6位于46:49。真实人工外力6:9仅标签与Critic，补偿球目标仅Critic95:98。
- 同文件`:1010`：补偿world目标=几何world目标+R_yaw(external_yaw+command_yaw)/K；external_yaw由`:654`的applied_world按当前yaw逆旋转得到；K200。
- 同文件`:1626`：球坐标center取rootXY和名义Z；只逆yaw，输出radius(m)、elevation(rad)、azimuth(rad)。
- 同文件`:668—746`：冻结pending为applied→8子步施力→更新state／yaw标签→旧命令结算→推进命令和事件→terminal→局部reset→发布下一帧。
- 同文件`:1708`：32帧history左移，仅末帧追加当前缩放Actor，噪声不覆盖命令43:58。
- `v3_actor_critic.py:154`附近：历史1856→Encoder64，最后58维与latent拼接→Actor14；不把Decoder估计直接回接动作。`v3_ppo.py:88`附近保存当前Actor／Critic／labels再step，非直接真实力控制器。
- 官方`UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:375—423`：Actor见commands，补偿球坐标只入Critic；`:1042—1109`命令9:12随梯形包络更新；`:1890—1900`world外力与旋转后命令相加/K。
- `tests/test_v3_rewards.py:16`提供原方法的无Gym步骤tensor fixture，`test_v3_force_transitions.py:14`提供step／history mock fixture，用于确定性时序核验。

本轮实际tensor fixture复用`test_v3_rewards.make_reward_environment`，在独立审计脚本补齐history／特权物性／applied-pending缓冲；未导入依赖pytest的step fixture。原生实际step时序仍以源码与既有force_transition证据为准。
