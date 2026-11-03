# 当前 V3 与上一次训练的资产及接触机制对照（2026-10-31）

## 结论

当前项目使用的是基于此前 V3 `v3_original` 资产派生的 V3 固定夹爪模型，不是完全相同的 URDF 文件：只把两个夹爪滑动关节固定，其他 link/joint 几何和惯性等数据保留。上一次 8000→9000 轮 Gym 训练也加载 `a1arm_rgm3_original_movable.urdf`，所以资产来源与碰撞几何高度一致；该训练的 `create_actor(..., 1)` 将 actor 级自碰撞全部关闭。当前 Legged Gym 默认 `self_collisions=0` 并传入 `create_actor`，即开启自碰撞。故当前诊断中的大接触力／快速base-contact终止机制在历史训练中被配置屏蔽，不能从历史训练未报告同类故障推断资产没有该问题。

历史评估确实记录到少量非超时终止（不同条件中0或1次），但没有保存终止原因分类或接触力逐体数据。因此能确认“存在少量物理终止”，不能确认它们由机器人自碰撞导致；也没有证据证明曾出现当前约10.59kN的base净接触力。

## 溯源事实

- 当前资产清单记录源 URDF SHA-256 为 `49b2e1b1...498df`，派生资产仅将 `gripper1_axis` 和 `gripper2_axis` 从 prismatic 改为 fixed；清单说明其余惯性、碰撞、视觉、origin及其他关节数据保留：`UniFP_v3/resources/robots/v3_a1/asset_manifest.json:2-10`。
- 当前 V3 配置选择 `a1arm_rgm3_v3_fixed_gripper.urdf`：`UniFP_v3/legged_gym/envs/v3/v3_pos_force_config.py:22-30`。
- 当前基础配置 `self_collisions = 0` 的注释定义0为enable、1为disable；actor创建直接传此字段：`UniFP_v3/legged_gym/envs/base/legged_robot_config.py:82`、`UniFP_v3/legged_gym/envs/base/legged_robot.py:576`。
- 当前V3每个策略步对base_link净接触力范数大于1 N即终止：`UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:127-131`。
- 历史训练后端加载 `a1arm_rgm3_original_movable.urdf`（资产路径见`.../gym_ranger_fresh512_fixedcom_20261028_v1/backend_envelope.py:68-70`），并在`backend_envelope.py:109`调用 `gym.create_actor(env,asset,pose,'ranger',i,1)`。Gym注释／代码对应的 `1` 是关闭该actor全部自碰撞，不是只过滤相邻装配碰撞。
- 历史force-stage完成记录显示训练完成到9000轮、216000 global policy steps，并记载96000左右物理更新有外力／命令力：`.../gym_ranger_force_stage_20261028_v1/completion.json:2-22`。
- 历史独立评估 `evaluation_summary.json` 记录每条轨迹的`non_timeout_terminations`和`timeouts`，部分轨迹有1次非超时终止、其余为0次，timeout均为0；记录没有各刚体接触力、终止原因或具体接触对。该指标只能说明终止计数，无法据此归因。
- 当前接触诊断记录：`v3_contact_diagnostic_20261031.md`。自碰撞启用时静置一步出现约10.59 kN base净接触力及多个正碰撞重叠；全局关闭自碰撞只作对照，非最终修复。

## 判断边界

1. 对“现在使用的模型是不是之前的V3”：是同一V3资产家族/碰撞模型基线，但当前实际文件是14-DOF派生资产，两个夹爪关节已固定；历史训练直接加载原始16-DOF movable URDF，由task映射选择控制DOF。
2. 对“上次训练有没有这个问题”：历史运行的自碰撞开关与当前不同，当前已发现的内部link重叠响应在历史Gym训练中被 actor-level filter 关闭。不能说“历史训练已发生同一自碰撞问题”；可以说训练资料没有记录同类接触异常，同时少量非超时终止存在但原因未知。
3. 这不排除地面接触、base姿态失败等其他终止，也不证明当前所有正重叠都是非法装配重叠。当前唯一可直接判断的是碰撞响应与actor级自碰撞开关强相关；逐shape重叠审计仍未完成。
