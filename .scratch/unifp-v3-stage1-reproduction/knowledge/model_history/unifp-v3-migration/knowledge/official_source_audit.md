# UniFP 官方实现与完整流程：源码事实核查

核查日期：2026-10-30。本文仅记录已读取的源码、文档与真实接口，不包含迁移建议、实施顺序或验收标准。

## 1. 证据范围与版本

- 本地官方目录为 `/home/sorry/python_ws/interactive/unifp_reproduction/UniFP`。本次实际执行 `git rev-parse HEAD` 得到 `68847a070f88d731058c3d8476929bc3b205f5bd`，`git status --short` 无输出。本地目录有可用 Git HEAD 且工作树干净。这不是对远程服务器训练目录的版本认定；远程归档是否来自相同字节仍需另外核对。
- 已完整读取用户指定的四篇流程文档：`introduction/00-training-flow-zh-CN.md`、`01-actor-zh-CN.md`、`02-critic-zh-CN.md`、`03-ppo-zh-CN.md`；以下实现事实再追溯到 `UniFP/` 内实际源码。父目录的 Isaac Lab 移植脚本不视为官方实现。
- 实际注册的任务为 `b2z1_pos_force`，对应 `LeggedRobot_b2z1_pos_force`、`B2Z1PosForceRoughCfg` 与 `B2Z1PosForceRoughCfgPPO`。来源：`UniFP/legged_gym/envs/__init__.py:4-12`。
- 训练入口先读取配置，再创建环境和 runner，最后调用 `learn(..., init_at_random_ep_len=True)`。来源：`UniFP/legged_gym/scripts/train_b2z1posforce.py:13-19`。

以下路径均相对于 `/home/sorry/python_ws/interactive/unifp_reproduction/UniFP`：

| 简称 | 实际路径 |
|---|---|
| `ENV` | `legged_gym/envs/b2/legged_robot_b2z1_pos_force.py` |
| `CFG` | `legged_gym/envs/b2/b2z1_pos_force_config.py` |
| `BASECFG` | `legged_gym/envs/base/legged_robot_config.py` |
| `AC` | `legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py` |
| `PPO` | `legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py` |
| `STORAGE` | `legged_gym/b2_gym_learn/ppo_cse_pf/rollout_storage.py` |
| `RUNNER` | `legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py` |

## 2. 总体调用链与物理时间

实际链路：`train → task_registry.make_env → make_alg_runner → OnPolicyRunner.learn → PPO.act → ActorCritic.act / evaluate → env.step → process_env_step → compute_returns → PPO.update`。来源：训练入口 13-19、`RUNNER:66-88,103-161`、`PPO:77-104,106-212`。

`env.step` 接收动作 `[N,17]`，返回四项：`obs_dict`、`reward:[N]`、`done:[N]`、`infos`；观测字典包含 `obs:[N,2336]`、`privileged_obs:[N,447]`、`obs_pred:[N,12]`。`obs_pred` 在环境接口中是监督真值标签，不是网络预测。来源：`ENV:135-142,405-438`、`CFG:102-113`。

- 默认并行环境数 `N=4096`，seed 为 1。来源：`BASECFG:4-12,183-185`。
- 物理积分 `dt=0.005 s`，decimation 为 4，策略周期 `0.02 s`。来源：`BASECFG:164-166`、`CFG:227-235`、`ENV:1741-1743`。
- 每个 iteration 收集每环境 24 个策略步，即 98,304 条 transition、每环境 0.48 s 仿真时间；随后 5 epoch × 4 minibatch，每批 24,576 条。来源：`BASECFG:202-215`、`RUNNER:115-151`、`STORAGE:96-130`。
- 执行顺序为：裁剪动作 → 每子步计算 PD 与施力/仿真 → 刷新状态 → 更新目标 → 判终止 → 算奖励 → 重置结束环境 → 构造返回观测。来源：`ENV:100-180`。
- 历史缓存 reset 清零，返回观测可能已属于新回合；终止之前的实际物理状态与 reset 后返回观测不等同。来源：`ENV:175-180,297-303`。

## 3. Actor、历史编码与辅助状态估计

### 3.1 输入、历史和命令

Actor 单帧 73 维按实际拼接顺序为：roll/pitch 2；机身三轴角速度 3；17 受控关节相对固定默认角位置 17；关节角速度 17；最近已执行动作 17；步态 sin/cos 2；任务命令 15。不存在真实末端位置、真实基座线速度、真实外力、摩擦或质量输入。来源：`ENV:414-422`。

Actor 历史为 32 帧，从旧到新展平，形状 `[N,2336]`；最新帧是最后 73 个数。名义 32 个采样周期为 0.64 s，最旧帧到最新帧相差 31×0.02=0.62 s。噪声在入历史之前加入，使用高斯噪声；reset 将有关环境的全部历史清零。来源：`CFG:107-111`、`ENV:297-303,424-438,1464-1471`。

15 维命令布局：`[0:3]` 底盘 vx/vy/yaw-rate；`[3:6]` EE 球坐标 radius/pitch/yaw；`[6:9]` 姿态槽位；`[9:12]` EE 期望力；`[12:15]` 底盘期望力。默认姿态槽位没有正常训练路径中的采样写入，底盘显式力 scheduler 调用被注释。来源：`ENV:838-855,949-962,1037-1173,1440-1455`。

命令原始缩放为 `[2,2,0.25,0.5,1,1.3,0.5,0.5,0.5,0.01,0.01,0.01,0.01,0.01,0.01]`。关节位置比例 1，关节角速度比例 0.05，机身角速度比例 0.25。来源：`BASECFG:129-145`、`ENV:1440-1455`。

### 3.2 三条不同网络路径

| 路径 | 真实结构/接口 | 来源 |
|---|---|---|
| 历史编码 | `2336 → 512 → 256 → 128 → 64`，ELU，最后线性层 | `AC:9,46-64` |
| 动作 | `[最新73维,64维latent] → 512 → 256 → 128 → 17` | `AC:83-93,143-150`、`CFG:316-318` |
| 辅助解码 | `64 → 128 → 64 → 12` | `AC:11,67-79,177-180` |

**Actor 不把 12 维解码状态显式拼回动作输入。** `update_distribution` 只执行 encoder 和 actor，解码器在状态估计损失或推理日志中另行调用。Actor 与 decoder 共享的是 64 维 latent。`act_student` 的 `policy_info['latents']` 实际保存的是解码后的 12 维预测，不是内部 64 维 latent。来源：`AC:143-166,177-180`。

监督标签 12 维依次为：机身完整坐标系线速度 3；yaw 对齐目标坐标系实际 EE 球坐标 3；yaw 对齐 EE 注入力 3；yaw 对齐机身注入力 3。标签是人为注入的力缓冲 `self.forces`，不是全部接触反力的传感真值。来源：`ENV:349-366,405-412`。

### 3.3 动作分布与实际执行

- 训练为每动作维对角高斯，均值由 Actor 给出，std 为全环境共享的可学习 `[17]` 参数，初始 1；动作头最后无 tanh。所有 17 维 log density 求和，形成每个环境的一个 joint log-probability。来源：`AC:89,112-116,139-153`。
- 推理使用动作均值；该前向不调用 Critic，也没有 PPO 在线更新。来源：`AC:158-166`、`RUNNER:252-256`。
- 环境裁剪动作到 `[-100,100]`，而非 `[-1,1]`。动作残差为 `a × motor_strength × 0.25`，加到固定默认角后以关节 PD 算力矩，最后按 URDF 力矩上限裁剪。电机强度随机化乘在角度残差上，并非最终全部力矩上。来源：`BASECFG:144-145`、`CFG:82-84,227-235`、`ENV:107-108,877-885`。
- 17 策略动作对应 12 腿关节 + 5 臂关节，资产另外两个 DOF（wrist rotate、gripper）使用固定参考 PD，因此返回完整 19 DOF 力矩。不能把配置名称 `num_gripper_joints=2` 理解为两个夹爪手指。来源：`CFG:58-64,104-106`、`ENV:880-883`、`resources/robots/b2z1/b2z1.urdf:1747,1815`。
- 默认角来自名字字典，在初始化装配为不可学习常量；动态 `ref_dof_pos` 是步态参考，用于别的特征/奖励，不替代 PD 的固定默认角。来源：`CFG:39-65`、`ENV:877-879,1473-1480,1818-1848`。

## 4. Critic 的真实特权契约

单帧 149 维按拼接顺序为：真实基座线速度 3、EE 球坐标 3、EE 注入力 3、基座注入力 3、12 个腿关节相对步态参考误差、质量/COM 参数 22、摩擦 1、电机系数偏差 17、期望足支撑标记 4、真实足接触标记 4、机身投影重力 3、角速度 3、关节位置 17、关节速度 17、最近动作 17、步态 sin/cos 2、命令 15、受力偏移后 EE 目标球坐标 3。来源：`ENV:379-404`。

- 22 维 `mass_params_tensor` 是 `附加基座质量1 + COM偏移3 + 附加EE质量1 + 17链接质量增量`，不是 22 个绝对质量。来源：`ENV:781-819`。
- 足接触标记由足端世界 Z 接触力 >5 N 构成，与注入外力标签不同。来源：`ENV:344-345,362-366`。
- 3 帧 Critic 历史展平为 `[N,447]`，不经过 Actor 高斯噪声注入路径，不经过 encoder 或 decoder。来源：`CFG:108,112-113`、`ENV:424-438`、`AC:95-105,173-175`。
- 网络为 `447 → 512 → 256 → 128 → 1`，最后为无约束标量 V；不是 Q(s,a)，不单独接收当前候选动作。**仅一个值头，无底盘/臂分组值头。** 来源：`CFG:318`、`AC:95-105,173-175`。
- V 预测的是总奖励定义下折扣未来回报；环境先计算和求和即时奖励，Critic 不创造奖励。来源：`ENV:305-322`、`STORAGE:72-86`。

## 5. PPO、GAE、辅助监督及梯度

### 5.1 唯一标量回报与 GAE

Rollout 中 reward、value、return、advantage 形状均 `[24,N,1]`，全部是单通道。奖励在环境先加权求和；没有底盘/机械臂两组 GAE。来源：`STORAGE:34-47`、`ENV:305-315`。

采样动作、旧 V、旧 log-prob、旧均值/标准差均 detach，整个 rollout 包在 inference mode；三类输入与标签在 `env.step` 前缓存，时刻相配。来源：`PPO:77-88`、`RUNNER:119-130`。

GAE 按 `δ=r+γ(1-d)V_next−V`、`A=δ+γλ(1-d)A_next` 倒序递推，return 为 `A+V`，全批优势再标准化。return 不随优势一同标准化。来源：`STORAGE:72-86`。

超时处理实际将 `γ V_old(s_t)` 加到缓存奖励，随后仍以 done 切断 GAE；没有在该处额外评价真正的终止前下一状态。普通失败不补该项。来源：`PPO:90-100`。rollout 截断但未终止时使用 `last_values=V(last_critic_obs)`。来源：`PPO:102-104`。

### 5.2 默认优化参数

| 参数 | 值 | 来源 |
|---|---:|---|
| gamma / lambda | 0.99 / 0.95 | `BASECFG:206-207` |
| PPO clip / clipped value | 0.2 / 开启 | `BASECFG:198-200` |
| value loss coefficient | 1.0 | `BASECFG:198` |
| entropy coefficient | 0.01 | `CFG:314-315` |
| epochs / minibatches | 5 / 4 | `BASECFG:202-203` |
| 主 Adam 初始 LR | 1e-3 | `BASECFG:204` |
| adaptive desired KL | 0.01 | `BASECFG:205,208` |
| 主梯度裁剪 | norm=1 | `BASECFG:209`、`PPO:169` |
| 辅助 Adam LR / substeps | 1e-5 / 1 | `PPO:12-15` |
| 训练配置最大 iteration | 60000 | `BASECFG:215` |
| checkpoint 间隔 | 200 | `BASECFG:218` |

上述为 runner 实际继承/覆盖的配置，PPO 构造器形参 `gamma=0.998` 不是默认任务实际使用值。来源：`RUNNER:73-74`、`BASECFG:206`。

KL(old||new) 按全部动作维求和、batch 求均值，用于主学习率调节：>0.02 时除以 1.5、最低 1e-5；0<KL<0.005 时乘 1.5、最高 1e-2。不作为额外可微损失，也没有该阈值处的 epoch early stop。来源：`PPO:131-144`。

主损失为 PPO surrogate + value coefficient×clipped value MSE − entropy coefficient×entropy。随后用更新后的 encoder 重新前向，再做辅助监督；每 minibatch 顺序为主更新一次、辅助更新一次。来源：`PPO:147-200`。

### 5.3 辅助损失及实际更新范围

四组标签长度均 3，依次为 base velocity、EE position、EE force、base force；权重 `[0.2,0.2,1,1]`。源码先乘 pred 与 target 再算 MSE，因此实际平方误差系数是 `[0.04,0.04,1,1]`，不是 `[0.2,0.2,1,1]`。来源：`AC:13-15`、`PPO:187-196`。

| 模块 | PPO 主损失计算图 | 辅助监督计算图 |
|---|---|---|
| encoder | 策略损失可回传 | 可回传 |
| actor body / std | 可回传 | 不相连 |
| critic | value loss 可回传 | 不相连 |
| decoder | 不相连 | 可回传 |

来源：`AC:143-153,173-180`、`PPO:164-200`。这里没有在 actor 前向处 detach latent；encoder 被两套独立 Adam 状态交替更新。两个优化器虽都注册整个容器参数，实际梯度按各前向计算图产生。来源：`PPO:53-55,166-200`。

辅助日志还有这些真实行为：`adaptation_batch_size=64` 未将样本切成 64 条，只用于返回损失日志的额外除数；`num_train=...80%` 计算后未实际切片；辅助梯度没有主步骤的 norm clip。来源：`PPO:15,175-200,204-209`。minibatch 索引每 update 只生成一次，5 个 epoch 复用同一分组。来源：`STORAGE:96-118`。

## 6. 命令、坐标系与统一力/位奖励

EE 命令球心实际为 `[base_world_x,base_world_y,0] + R_yaw [0.2,0,0.8]`，不加真实 base Z，也不随 roll/pitch 倾斜，函数没有查询地形高度。同一个局部球坐标命令随底盘 XY/yaw 改变世界目标。来源：`CFG:18-21`、`ENV:949-978`。

末端端点范围 `r∈[0.35,0.95] m`、pitch `±2π/5`、yaw `±3π/5`；在球坐标做线性插值；移动 1–3 s、保持 0.5–2 s；初始每环境采样时长，后续重采样目标不每次重抽时长。路径进行离散碰撞/地下检查；源码没有整条路径完整 IK 可达保证。来源：`CFG:6-30`、`ENV:887-978,1388-1389`。

启用的 EE 奖励为：

`p_target_world = p_geom_world + (F_ext_world + R_yaw F_cmd_yaw) / K_ee`；`r_ee = exp(-2 × sum(abs(p_ee_world−p_target_world)) / tracking_ee_sigma)`，`tracking_ee_sigma=1`，奖励系数 2；每轴 `K_ee=200 N/m`。这是受力偏移目标的位置奖励；不是直接接触力误差奖励。末端姿态、球坐标跟踪等另外奖励系数为零。来源：`ENV:1891-1900`、`CFG:148-150,257-258,302-311`。

原始几何目标误差、受力补偿目标误差、注入力估计误差以及真实接触力控制误差是不同物理指标；上述公式只定义其中的目标与奖励，不把它们等同。变量来源：`ENV:362-377,405-412,1891-1900`。

基座速度统一奖励的参考是 `v_cmd + (F_base_ext_yaw+F_base_cmd_yaw)/D_base`，再经小命令清零，默认 `D_base=200`，只追踪平面线速度。来源：`ENV:2016-2029`、`CFG:167-168`。

所有非零奖励系数在注册时乘策略周期 dt；分项日志来自加权 episode 累积，而不是同单位的原始物理误差。来源：`ENV:1551-1570`、`RUNNER:168-180`。

## 7. 课程、随机化和断点恢复

- 力课程实际判断为 `global_steps > 8000 * 24`，阈值 192000 个批量策略步，判断在物理子步循环中，计数在批量 step 末只增加 1；阈值字面量 24 不随 runner rollout 配置自动改变。来源：`ENV:113-141`、`CFG:178`。
- 阈值前后不换 Actor、Critic 或 PPO，同一网络继续优化。实际改变为调用 `_push_gripper`，EE 命令力与实际外力分别独立采样/调度；每轴范围 ±60 N，概率各 0.8，上升/保持/下降/归零；不是向量范数最大 60 N，也不是所有环境一直有力。来源：`CFG:140-156`、`ENV:1037-1169`。
- `_push_robot_base` 虽有实现，但在 `step` 调用被注释，基座显式外力和力命令默认不启动；不能仅依据 3 维 base-force 输出宣称已学会非零 base force。来源：`ENV:125-130,1173,1544-1549`。
- 周期 `_push_robots` 直接改写 root 水平速度，是另一类扰动，不是 force buffer 注入力。来源：`ENV:1025-1035`；默认周期 8 s、水平速度幅值 0.8 m/s。来源：`CFG:97-99`。
- 域随机化配置包含摩擦 `[0.3,2]`、base 附加质量 `[0,15] kg`、COM 各轴 `±0.15 m`、motor `[0.85,1.15]`、EE 附加质量 `[0,0.2] kg`；腿质量随机化默认关闭。多项物性在创建时回调设置，不是每次 reset 全部重新采样。来源：`CFG:70-99`、`ENV:700-819,1713`。
- 官方 `RUNNER.save` 仅保存模型参数、主 optimizer 状态、`current_learning_iteration` 和 infos；不保存辅助 optimizer、global_steps、环境/课程历史或 RNG 状态。来源：`RUNNER:236-250`。
- `current_learning_iteration` 在整个 learn 循环结束才增加，循环中间按 `it` 命名的 checkpoint 仍可能含旧 `iter` 元数据。来源：`RUNNER:115-161,236-242`。
- `load` 只恢复模型、可选主 optimizer 和 iter，不恢复环境 `global_steps`；该计数新环境初始化为零。因此只调用官方 load 不等于恢复力课程状态。来源：`RUNNER:244-250`、`ENV:1549`。

## 8. 本次核查文件指纹

| 文件 | SHA256 |
|---|---|
| `AC` | `e44cdb63f7e05bc82a469559d6a24f54596e1e4dfa9e784c4b12cce39b53c971` |
| `PPO` | `a3366f5d951a19c819ed26c5a171df89c3ff682d4a97d9c8c77585146e778e15` |
| `STORAGE` | `ca7618b01150de14424201b2a61e50d42402f61f6771ab409d545f9cd90ae399` |
| `RUNNER` | `d44a1835b0ec3db083cc825249a97f5f535781c4a2cff8c91faede2ee470e9c4` |
| `CFG` | `a1213a398cb7928a90aaa7ca13ae0840ce970cf5cdfa81827057a9c5b3b751fc` |
| `ENV` | `220b05ed078bf7b12be3bd489610bb90c25b12a5b78538bc8af58484093a88c4` |

本次为本地一手静态核查，没有启动仿真、训练或新评估。
