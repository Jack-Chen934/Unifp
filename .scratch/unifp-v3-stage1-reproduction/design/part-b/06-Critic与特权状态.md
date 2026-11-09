# B06 · Critic与特权状态

状态：2026-11-07 B06-Q1／Q2／Q3均获用户确认，本章设计收束，当前进入B07-Q1。全部B设计确认后才拆正式票据并编码；以下确认不代表B业务代码已实现。

## 已确认结论总结

完整接续A8000的单价值Critic并继续训练，保持98×3＝294输入、294→512→256→128→1、单总reward／单return／单GAE；辅助MSE不进入环境reward。当前TCP、几何目标、补偿目标分别保留，补偿目标先在世界系按合力除K平移，再相对A任务中心逆yaw转球坐标并按原缩放填入末尾95:98；Actor仍只接收原几何目标与命令力。其余实际状态、创建级物性、执行系数和四轮接触字段沿用A，人工末端力槽填刚经历区间真值，底盘人工力槽为零，不把接触净力冒充人工力。Critic保留clean输入，同一帧接口构造普通与终止后继；reset前独立保存真实三帧快照，纯timeout补一次价值、物理失败不补，局部reset隔离。B不增价值头、输入维度、传感器或全局停训条件；K、奖励参数与正式验收由后续模块确认。

## 当前基线与目标

当前Critic为98维单帧、3帧294维、单标量V；外力真值槽已存在，但末尾compensated_target当前仍直接取commands[:,3:6]。目标是在固定shape内填入实际受力特权值与正确补偿目标。

继承V6 r2 A8000、14动作、58×32 Actor历史、98×3 Critic历史、12辅助标签、15命令槽和2.5ms×8控制；不切V8、不重跑A、不进入轮式创新。来源和父SHA见[总体计划](../../development-plan.md)；当前源码依据见[research](../../knowledge/research.md)。

## 按依赖讨论的决策

Q1：保持294→512→256→128→1与单reward／单GAE；不为受力单独建价值头。

Q2：明确当前TCP球坐标、几何命令目标与补偿目标的区别，记录世界系补偿后如何转换为既有球坐标槽及缩放。

Q3：非零力和真实接触特权值仅按已定义含义写入；不能把接触净力塞入人工外力槽。已确认的B04-Q4要求后继帧使用刚结束区间实际力及下一边界命令，补偿槽不得偷用待施加的下一外力。纯timeout继续用reset前真正终止后继Critic观测补价值一次。

推荐的最小接口／数据流框架（待具体签名确认，无业务实现）：

```text
critic_frame[N,98] = build_existing_fields(aligned_force_labels, geometric_target, compensated_target)
critic_history[N,294] = append_frame_once(critic_frame)
value[N,1] = critic(critic_history)
timeout_bootstrap(terminal_successor_value)
```

## 状态、失败与验证边界

本模块依赖前序B模块已确认接口，B04依赖A公共基线；测试最后由B11分别收束单元与整体部分。N为环境数，B为actor刚体数；力N、力矩N·m、位置m、时间s，yaw系指仅随底盘偏航旋转的坐标系。新增状态归属、更新时刻、reset子集隔离、持久化及异常语义在对应Q&A确定。

普通物理失败沿用官方局部reset并记录；NaN、非有限状态、执行异常及不兼容checkpoint在更新前退出。验证必须保留非零力样本，A的零标签结果不能替代B验收。方案不引入新策略头或专用力控器。

## 讨论记录

已确认：用户选择V6直接进入B，并保留V8诊断证据；B04／B05／B06全部确认，本章开头总结已补齐。当前进入B07，不能用本章确认代替后续模块的参数与恢复接口确认。

### Q1：是否沿用A的单价值Critic，用一个总reward与一套GAE评价联合任务？

状态：2026-11-07用户回复“同意”，确认以下单价值网络和价值学习组织；未修改业务代码。非零特权字段及补偿目标构造分别留Q2／Q3，reward具体项与权重留B08。

推荐：完整接续A8000的Critic参数并继续训练，不重置或冻结；保持3帧×98维输入294、ELU MLP `294→512→256→128→1`。`evaluate(critic_history[N,294]) -> value[N,1]`返回预期未来折扣累计总reward的标量估计，不输出力或位置预测，不额外增加力任务价值头、移动价值头或第二个Critic。物性、实际外力等已定义特权字段在训练中供Critic使用，Actor仍只读取其既有58维帧／32帧历史，不把Critic特权值拼给Actor。

定义联合环境reward为`r_total = sum_i(effective_weight_i * raw_reward_i)`，沿用A按策略dt一次加权的组织；具体奖励项和参数由B08确认。PPO保持每个env-step一份总reward、V、return与advantage；沿用单GAE、现有价值裁剪损失及主Adam对Critic的更新，不按奖励项分别归一化、分别算GAE或事后组合优势。辅助decoder的MSE仅进入已确认的辅助更新，不额外加进环境reward或Critic回报。

Critic学习的是整项任务的长期表现；即使奖励由底盘跟踪、TCP补偿跟踪、稳定性等分项组成，也可以通过单一加权总reward学习折中，不要求每一分项都有独立价值头。分项reward／误差仍单独记录用于分析，但不会自动形成多套优化目标。双Adam边界遵循B05，Critic不参与辅助Adam更新。

官方对比：官方同样用特权历史输入的单标量Critic与单reward／单GAE。V3的294输入来自A已确认98×3布局，B维度不改；实际人工力及补偿目标非零时在既有槽位填值，不以新建网络替代字段修正。真实timeout后继bootstrap等A已有修正继续沿用B04，不回退官方旧处理。

理由与风险：保留结构可接续A的价值表示和完整优化状态，避免同阶段同时改变策略和价值算法。进入B后reward／观测分布变化会使A的V暂时不准确，value loss可能升高；这不是必须重置Critic的证据。需结合回报量级、value loss、KL、TCP／底盘误差和失败率判断是否稳定，若出现持续异常先检查字段、标签时序与奖励尺度，不默默重建价值头或放宽失败规则。

最小调用框架（伪代码，无业务实现）：

```text
value_t[N,1] = model.evaluate(critic_history_t[N,294])
reward_t[N] = env.compute_existing_total_reward(executed_transition)
ppo.process_env_step(reward_t, done_t, existing_infos)
ppo.compute_returns(final_successor_critic_history)
ppo.update()  # 同一批联合优势更新策略，Critic拟合同一总return
```

验证接缝：复用模型294→1 shape、梯度参数范围、总reward与GAE／timeout测试；阶段迁移零更新时相同Critic输入输出一致，非零特权／补偿样本下前向与价值梯度有限。B06-Q2／Q3核验具体字段后，由B11组合必要验收；不为每个reward分项新增价值测试网络。

依据：`UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_actor_critic.py:12,18,106-109,325-350`、`v3_rollout_storage.py:60-63,125-155`、`v3_ppo.py:162-177,235-245`；官方`UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py`的critic_body／evaluate及`ppo.py`的单return／value_loss组织。

### Q2：是否保留当前TCP与几何命令，并将Critic末尾三维填为真正的补偿目标？

状态：2026-11-07用户回复“同意”，确认以下三种位置与补偿槽构造；未修改业务代码。不批准新的K数值、奖励权重或目标裁剪范围，这些留B08／B09。

源码事实：V3 `_build_frames`已提供当前TCP球坐标、15维命令和末尾三维compensated_target，但末尾值当前直接复制commands[:,3:6]；A两力为零时正确，B非零力时不能继续复制。官方compute_observations先将世界几何目标加上世界合力除K，再相对任务中心逆yaw旋转、转球坐标并缩放，放入Critic末尾三维。官方明确同时保留当前位置、原几何命令与补偿目标，不将三者混成同一个位置。

推荐固定槽位与语义（索引为单帧98维、Python左闭右开，三帧布局不变）：

| 量 | 单帧Critic位置 | 定义与单位／缩放 |
|---|---|---|
| 当前TCP | 3:6，位于12维状态标签内 | 当前root／DOF经FK得到的TCP，相对原任务中心的(r,elevation,azimuth)，m／rad／rad，逐轴×(0.5,1,1.3) |
| 几何目标 | 83:86，位于80:95的15维命令内 | commands[:,3:6]，目标本身不被力覆盖，同样缩放 |
| 补偿目标 | 95:98 | 几何世界目标＋(人工外力世界值＋yaw命令转世界)/K，再转回原任务球坐标，同样缩放 |

Actor的几何命令仍为原几何目标，不能把基于人工外力真值的补偿目标替换进Actor命令或历史，否则会改变已确认的可观测性。decoder位置监督仍是当前TCP，不预测目标或补偿目标。

任务中心保持A定义，不用实际工具COM或新的机械臂安装坐标系替换：

```text
center_world.x = root_world.x + ARM_MOUNT_X * cos(yaw)
center_world.y = root_world.y + ARM_MOUNT_X * sin(yaw)
center_world.z = env_origin.z + nominal_base_height + ARM_MOUNT_Z
```

该中心只随底盘XY和平面yaw变化，Z使用名义高度，不跟随实际底盘起伏；task frame只绕yaw旋转，保留A的几何任务定义。补偿按下列顺序构造，力用N、K用N/m、世界位置用m，N为批次环境数：

```text
p_geo_world[N,3] = existing_geometric_target(commands[:,3:6], root_state)
F_cmd_world[N,3] = R_yaw * command_force_yaw_n[N,3]
p_comp_world[N,3] = p_geo_world + (aligned_external_world_n + F_cmd_world) / K[N,3]
delta_yaw[N,3] = inverse(R_yaw) * (p_comp_world - center_world)
radius_m = norm(delta_yaw)
elevation_rad = asin(clamp(delta_yaw.z / max(radius_m, 1e-6), -1, 1))
azimuth_rad = atan2(delta_yaw.y, delta_yaw.x)
critic_compensated[N,3] = stack(radius_m, elevation_rad, azimuth_rad) * (0.5,1,1.3)
```

球坐标是非线性坐标，不能把F/K直接加到(r,elevation,azimuth)，也不能把N单位力除K后直接加进已缩放特征。实现可将现有世界点→任务球坐标运算提取为一个私有共享帮助函数，当前位置与补偿点共用，避免复制两套中心／yaw公式；保持当前函数外部调用不变，不新增通用坐标变换框架。角度使用现有atan2主值，近中心仅采用已有浮点安全处理，不因此抛新异常阻断正常训练。若合力精确为零，该环境末尾槽直接沿用几何命令三元组，保留A零力回归及避免不必要的往返浮点误差；非零力路径输出转换后的主值。

时序严格遵循B04-Q4：当前／后继Critic帧以该帧状态yaw、已发布命令和刚经历的实际人工外力构造；下一段待施加的随机外力不通过补偿目标间接泄漏。奖励仍使用对应执行区间的命令／外力快照，不拿推进后的Critic补偿目标回填旧转移reward。正常后继帧与终止后继帧共用构造，reset后人工力与命令力为零，补偿目标退化为新episode几何目标。

不额外将补偿点裁回几何目标采样范围、投影到可达空间或重新采样：这些操作会改变官方补偿关系。若非零力导致明显不可达，在B08／B09通过工作空间、K和力幅联合分析另议，不静默裁剪；已有最终观测统一clamp仍按A规则保留，必要时报告饱和。K值沿既有缓存定义，本项不将示例K=200N/m视为V3新的批准值。

官方对比：保留“世界点相加补偿→逆yaw→球坐标→缩放”的处理顺序及三种位置同时存在。V3必要适配为A已确认任务中心、TCP几何点和98维槽位；B真正新增的是将末尾三维的A零力简化值恢复为非零力补偿值，不改输入宽度。不能宣称该补偿点已通过实物接触验证，真实接触力定义留B08。

验证接缝：逐字段槽位／缩放核对；零力与正负力、yaw=0／90°及XYZ不同方向的解析对照；当前位置和几何命令不被补偿覆盖，Actor不读取人工真值；球坐标世界点往返与共享中心一致；后继／终止帧使用已施加外力而非下一事件。测试以固定输入独立计算期望结果，不只调用同一帮助函数验证自身，规模由B11收束。

依据：`UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:746-760,775-789,1374-1396,1413-1449`；官方`UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:350-403`。本项已确认，未修改代码或改变A运行。

### Q3：是否在既有98维内沿用实际物性与接触字段，填入已施加人工力，并复用同一后继帧接口？

状态：2026-11-07用户回复“同意”，确认以下整套字段契约；未修改业务代码。B04已确认的终止／reset规则直接沿用，本项确认Critic剩余字段与同一构造接口的对应，不新增传感器或训练限制。

推荐完整单帧布局如下（Python左闭右开索引，所有缩放沿A既有定义）：

| 槽位 | 字段及来源 | B处理 |
|---|---|---|
| 0:3 | 实际底盘系线速度，m/s×2 | 沿用A真值，不改为decoder预测 |
| 3:6 | 当前TCP球坐标，m／rad／rad，×(0.5,1,1.3) | 沿用Q2定义 |
| 6:9 | 刚经历区间的人工末端外力，当前yaw系N×0.01 | 填入实际已施加值 |
| 9:12 | 人工底盘外力，yaw系N×0.01 | 本次保持零 |
| 12:17 | base质量增量kg、base局部COM xyz增量m、工具质量增量kg | 沿用创建时已生效物性回读缓存，不重新随机化 |
| 17:18 | 实际摩擦系数 | 沿用创建时已生效回读缓存 |
| 18:32 | 14维执行系数相对1的偏差 | 沿用当前episode的有效样本与动作顺序 |
| 32:36 | 四轮接触标志 | 保留世界Z净接触力>5N及现有contact_valid掩码 |
| 36:39 | 底盘系投影重力方向 | 保留A定义，不作为外力标签 |
| 39:42 | 底盘系角速度，rad/s×0.25 | 沿用A |
| 42:52 | 十个位置关节的名义位置偏差，rad | 沿用A，驱动轮不加入绝对轮角 |
| 52:66 | 14维动作顺序关节速度，rad/s×0.05 | 沿用A |
| 66:80 | 上一执行动作 | 沿用A环境动作缓存与裁剪语义 |
| 80:95 | 15维命令及既有COMMAND_SCALE | 包含下一决策的几何／速度／力命令，按B04帧时序发布 |
| 95:98 | 补偿目标球坐标，×(0.5,1,1.3) | 按已确认Q2填入真正补偿值 |

本项不新增部件接触力三维向量、原始力传感器、接触对列表、事件峰值／计时器或下一段待施加外力。当前人工外力真值与物性用于Critic训练，Actor不直接读取它们；Critic不通过decoder估计替换真值。真实接触净力与人工注入力并非同一量：轮接触标志保留原含义，机械臂碰机身等接触不塞入6:12人工力槽，额外碰撞证据沿现有诊断／评估输出记录。

观测构造复用现有`_build_frames(env_ids)`与`_publish_observations`边界：先以同一批状态／命令／已施加外力构造clean Critic帧与标签，Actor按原规则加噪，Critic不额外加入Actor测量噪声；最终观测clamp仍保留A规则。每边界每个存活环境只追加一帧，时间最老到最新组成`[N,3,98]`并展平`[N,294]`；episode结束仅清对应行历史，不改网络或添加新的有效性维度。

沿用B04终止接口：`terminal_env_ids[K]`和独立clone的`terminal_privileged_obs[K,294]`在reset前构造，使用同一个clean帧方法；三帧历史为旧历史后两帧加真实后继帧。纯timeout的V只用此快照，普通失败不补价值；两者同时按失败。done行reset后新帧的人工力、命令力为零，创建级物性保持，执行系数按A局部reset语义重采，轮接触用contact_valid屏蔽尚未有效的新episode缓存。未done行的物性、力事件和历史不被清除。

这不是新增一套timeout算法、运行时穷举审计或每步CPU物性查询。保持现有4项step返回及infos字段，复用当前shape／有限值校验；必要字段核对集中在单测与短测。不因单次普通接触失败全局停训，也不在B中反复设置actor质量／COM，创建级物性恢复仍待B07沿既有GPU约束收束。

官方／A对比：保留官方非对称Actor／Critic、真实状态／外力／物性特权信息及clean特权输入的思路；98维布局、四轮接触、五维质量COM和14维执行系数是A已确认机器人适配，B继续沿用。B只启用既有末端人工力槽并补齐Q2补偿目标；没有新增“测得真实接触总外力”的能力。真实终止后继快照及物性创建时恢复是A已确认修正，B不回退、不另增第二种路径。

验证接缝：给各字段不同已知值核对98维顺序／缩放与3帧展平；只改变人工末端力时，Actor同状态输入不受其真值直接影响，Critic对应力槽及补偿槽改变；同一输入的普通后继／终止后继帧一致；reset仅对应行改变且快照不被覆盖；复用现有timeout单次补偿、创建物性恢复与有限性测试。无需对每一个字段再写一套原生仿真测试，B11整体验收补真实非零受力链路。

最小接口（伪代码，使用既有外部签名，不含业务实现）：

```text
actor_frame[K,58], critic_frame[K,98], labels[K,12] = env._build_frames(env_ids)
terminal_critic[K,294] = concat(old_critic_history[done_ids,1:], critic_frame_done).clone()
env.reset_idx(done_ids)
env._publish_observations(all_env_ids)
ppo.process_env_step(rewards[N], dones[N], existing_infos_with_terminal_snapshot)
```

来源：`UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:1410-1481,1040-1088`、`v3_ppo.py:116-157`、已确认B04-Q4与A06。当前未修改业务代码，Q3已确认并补齐B06整章结论总结。
