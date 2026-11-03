# UniFP 的 Critic：输入什么、预测什么、怎样学会预测

本文只讲官方 `b2z1_pos_force` 任务中的 Critic。先看 [完整训练流程](/home/sorry/python_ws/interactive/unifp_reproduction/introduction/00-training-flow-zh-CN.md)，动作与关节控制见 [Actor 专篇](/home/sorry/python_ws/interactive/unifp_reproduction/introduction/01-actor-zh-CN.md)，概率比率与整体更新见 [PPO 专篇](/home/sorry/python_ws/interactive/unifp_reproduction/introduction/03-ppo-zh-CN.md)。

核查对象：`/home/sorry/python_ws/interactive/unifp_reproduction/UniFP`，提交 `68847a070f88d731058c3d8476929bc3b205f5bd`。下文的数值例子是为解释公式而设置的算例，不是实际训练日志。

## 1. 先明确：Critic 输出的“分”究竟是什么

**Critic 接收当前状态的特权观测历史，输出一个实数，预测从这里继续按当前策略行动，未来能拿到多少折扣累计奖励。它不输出关节角度，也不直接给出某个动作的分数。**

在这个项目中，可用下面的记号理解：

\[
V_\phi(c_t)\approx\mathbb E_{\pi_\theta}\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k}\mid c_t\right].
\]

这里 `c_t` 是 Critic 的 447 维输入，不是“动作”。实际回合会终止，公式中的求和在终止处截断；没有走到终点的 rollout 会使用价值预测补上后续。`γ=0.99`，而且代码用 GAE 构造训练目标，不是每次都收集完整无限长回报。

代码证据：[Critic 的 evaluate 只有观测输入并返回 value](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:173)、[折扣系数与 GAE 系数](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/base/legged_robot_config.py:196)、[真实的 return 计算](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/rollout_storage.py:72)。

三个经常混淆的量：

| 量 | 谁产生 | 实际含义 |
|---|---|---|
| 即时奖励 `r_t` | 环境中人工编写的奖励函数 | 执行动作后，这一步的效果如何 |
| 价值 `V(c_t)` | Critic 神经网络 | 从当前状态继续行动，预计能累计得到多少奖励 |
| 优势 `A_t` | PPO 用奖励与价值预测计算 | 这次实际经历相对于原来的价值预期好多少 |

环境先调用奖励函数并加权求和得到 `rew_buf`；Critic 既不创造这些奖励规则，也不决定奖励项的权重。证据：[compute_reward](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:305)。

例如 `V=8.0` 不表示“这一步奖励为 8”“动作正确率 80%”“关节应转 8 rad”。它只是当前奖励定义下的累计回报预测值。奖励项在注册时还会乘策略时间步 `dt`，所以价值的数值大小也取决于奖励系数和时间尺度，不具有独立的角度或力单位。证据：[奖励系数乘 dt](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1551)。

也要区分 `V(s)` 和 `Q(s,a)`：这份 Critic 没有单独接收候选动作 `a_t`，因此不是“输入某个动作，评价这个动作”的 Q 网络。它确实会收到上一时刻已执行动作作为观测的一部分，但这并没有把其接口变成 `Q(s,a)`。证据：[149 维拼接中的 self.actions](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:379)、[evaluate](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:173)。

## 2. Critic 输入：447 个数到底装了什么

### 2.1 先看形状

默认配置：

```python
c_frame_stack = 3
single_num_privileged_obs = 149
num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
```

因此一个环境的 Critic 输入是 `3 × 149 = 447` 个浮点数。默认同时运行 4096 个环境时，完整批次形状是 `[4096, 447]`。证据：[任务维度配置](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:102)、[默认环境数](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/base/legged_robot_config.py:4)。

### 2.2 单帧 149 维：按代码拼接顺序逐项展开

下面的切片采用 Python 的 **从 0 开始、右端不含** 记法。例如 `[0:3]` 就是第 0、1、2 项。

| 单帧切片 | 维数 | 代码中的内容 | 送入网络前的缩放 / 解释 |
|---|---:|---|---|
| `[0:3]` | 3 | `base_lin_vel` | 机身坐标系真实线速度，乘 `2.0` |
| `[3:6]` | 3 | `ee_pos_sphe_arm` | 实际末端球坐标 `[r,pitch,yaw]`，分别乘 `[0.5,1.0,1.3]` |
| `[6:9]` | 3 | `forces_local[:, gripper_idx]` | 仿真注入的末端外力，转到仅跟随机身 yaw 的坐标系，乘 `0.01` |
| `[9:12]` | 3 | `forces_local[:, robot_base_idx]` | 仿真注入的机身外力，同上，乘 `0.01`；默认课程调用被注释，因此为零 |
| `[12:24]` | 12 | `diff = dof_pos[:,:12] - ref_dof_pos` | 12 个腿关节相对步态参考角度的误差，未额外缩放 |
| `[24:46]` | 22 | `mass_params_tensor` | 随机化参数增量，细分见下文；未额外缩放 |
| `[46:47]` | 1 | `friction_coeffs_tensor` | 地面接触摩擦系数，未额外缩放 |
| `[47:64]` | 17 | `motor_strength[:,:17] - 1` | 电机强度因子相对 1 的偏差 |
| `[64:68]` | 4 | `stance_mask` | 按步态相位生成的四足预期支撑标记，0 或 1 |
| `[68:72]` | 4 | `contact_mask` | 仿真足端世界竖直方向接触力是否大于 5 N，布尔量转为浮点 |
| `[72:75]` | 3 | `projected_gravity` | 单位重力方向转到机身坐标系；不是 `[0,0,-9.81]` 的加速度值 |
| `[75:78]` | 3 | `base_ang_vel` | 机身坐标系角速度，乘 `0.25` |
| `[78:95]` | 17 | `dof_pos - default_dof_pos` | 17 个受控关节相对固定默认角度的误差，乘 `1.0` |
| `[95:112]` | 17 | `dof_vel` | 17 个受控关节角速度，乘 `0.05` |
| `[112:129]` | 17 | `self.actions[:,:17]` | 最近一次已经施加的动作数值，没有再乘 `action_scale` |
| `[129:130]` | 1 | `sin(2π phase)` | 当前步态相位的正弦 |
| `[130:131]` | 1 | `cos(2π phase)` | 当前步态相位的余弦 |
| `[131:146]` | 15 | `commands * commands_scale` | 基座速度、末端位置/姿态槽位、末端力命令、机身力命令 |
| `[146:149]` | 3 | `ee_goal_offset_local_sphere` | **加入力偏移之后**的末端目标球坐标，乘 `[0.5,1.0,1.3]` |

这张表直接逐项对应 [compute_observations 的 torch.cat](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:379)。缩放值来自 [normalization.obs_scales](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/base/legged_robot_config.py:129)，命令的缩放顺序见 [commands_scale](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1441)。

几个必须结合代码理解的细节：

**第一，12 维 `diff` 与 17 维关节位置误差不是同一个东西。** `default_dof_pos` 是配置中的固定参考姿态；`ref_dof_pos` 则在这个默认姿态上，根据当前步态相位给部分腿关节增加变化。因此前者固定，后者可能随步态变化。这并不推翻“默认角度固定”的结论。证据：[diff 的定义](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:347)、[动态步态参考的计算](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1818)、[固定默认角度配置](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:41)。

**第二，22 维 `mass_params_tensor` 不是 22 个刚体的绝对质量。** 顺序实际是：`1 维机身附加质量 + 3 维 COM 偏移 + 1 维末端附加质量 + 17 维刚体质量增量`，也就是 `1+3+1+17=22`。其中代码通过 `props[1].com` 修改 COM；默认 `randomize_leg_mass=False`，所以最后 17 维增量为零。不要根据变量名把这 22 个数全部解释成质量。证据：[随机化参数的真实拼接](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:781)、[任务随机化开关](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:70)。

**第三，Critic 中的外力是仿真施加的 `self.forces`，不是 Actor 自己估计的力，也不是足端接触力总和。** `forces_local` 来自将这个施力张量旋转到 yaw 对齐坐标系；足端接触只在另外 4 维 `contact_mask` 中用到。证据：[施力与坐标转换](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:359)、[接触标记](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:344)。

**第四，特权信息没有经过 Actor 那条加噪路径。** 代码只给 `obs_buf` 的 Actor 单帧加高斯噪声，再把特权帧直接追加到 `critic_history`；最终环境返回前仍会把观测截到 `[-100,100]`。这不意味着 Critic 的世界没有随机性：物理参数、环境扰动依旧会改变这些真实观测。证据：[加噪与追加历史](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:424)、[返回前观测裁剪](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:135)。

### 2.3 坐标系：不是所有的“局部坐标”都相同

- 机身线速度、角速度、投影重力：用完整的 `base_quat` 逆旋转，跟随机身 roll、pitch、yaw。
- 实际末端球坐标：先减去 `get_ee_goal_spherical_center()`，再用仅 yaw 的四元数逆旋转；`r` 单位为米，pitch、yaw 单位为弧度。
- 末端/机身外力：只按机身 yaw 逆旋转，保留世界竖直方向。
- 力偏移目标的球坐标：也围绕相同目标球坐标中心，使用仅 yaw 的坐标系。

证据：[完整机身坐标系速度与重力](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:157)、[末端球坐标和外力转换](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:349)。

虽然变量叫 `ee_pos_sphe_arm`，这里的原点不是简单的“实际肩关节位置”。代码中的中心由机身世界 `x,y`、零 `z`，加上随 yaw 旋转的配置偏移构成；默认偏移为 `[0.2,0,0.8]`。不能根据局部注释把它理解成随机身 roll/pitch 全部转动的机械臂基座坐标。证据：[中心计算](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:975)、[中心偏移默认值](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:18)。

最后 3 维特权目标来自：

\[
x^{offset}_{world}=x^{command}_{world}
+\frac{F^{external}_{world}+R_{yaw}F^{command}}{K_p^{force}}.
\]

先算世界系偏移，再变换回上述局部球坐标。因此 Critic 既看到原始命令，也看到考虑当前注入外力后的目标；它更容易预测这种状态下接下来的奖励。这里的力刚度 `K_p^{force}` 是任务层面的力到位置偏移系数，不是关节 PD 控制里的关节刚度。证据：[偏移目标的计算](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:369)。

### 2.4 单帧缩放的实际数值例子

假设某一帧有以下物理量，缩放后 Critic 看到的就是右侧数字：

| 物理 / 控制量 | 原值 | 对应的网络输入 |
|---|---|---|
| 机身前进速度 | `0.30 m/s` | `0.30 × 2 = 0.60` |
| 末端球坐标 | `[0.60 m, 0.20 rad, -0.10 rad]` | `[0.30, 0.20, -0.13]` |
| yaw 对齐坐标系中的注入末端力 | `[20,0,-10] N` | `[0.20,0,-0.10]` |
| 某个关节相对默认姿态偏移 | `0.10 rad` | `0.10` |
| 某个关节角速度 | `2 rad/s` | `0.10` |
| 某个电机强度因子 | `1.10` | `1.10 - 1 = 0.10` |
| 最近动作 | `0.40` | `0.40`，此处没有转成 `0.10 rad` |

这不是另一套动作定义，而是状态输入的数值归一化。控制执行端仍按 Actor 专篇中的 `q_target=q_default+0.25×motor_strength×action` 计算。缩放与字段证据同 [输入拼接](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:379)、[缩放配置](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/base/legged_robot_config.py:129)；动作到关节目标的执行证据见 [扭矩计算](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:877)。

### 2.5 三帧怎样拼成 447 维

代码初始化长度为 3 的 `deque`，每个元素 `[4096,149]`，初始为零；每次计算观测把最新单帧追加到右边，最后按 `history[0]` 到 `history[2]` 顺序拼接：

```python
self.critic_history.append(self.privileged_obs_buf)
self.privileged_obs_buf = torch.cat(
    [self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)],
    dim=1,
)
```

正常稳定运行时：

```text
c_t = [ 单帧 p_(t-2) | 单帧 p_(t-1) | 单帧 p_t ]
         149 维          149 维        149 维
```

所以 `[0:149]` 是最旧帧，`[298:447]` 是最新帧。默认策略时间步是 `0.005×4=0.02 s`，相邻帧间隔 20 ms，最旧到最新相差 40 ms。回合重置时，对应环境的所有历史帧清零，随后追加新帧；不能把重置后头几帧理解成已有完整三帧真实历史。证据：[历史初始化](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1464)、[追加和展平](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:431)、[重置历史](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:300)、[策略 dt 计算](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1741)、[物理 dt](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/base/legged_robot_config.py:164)。

## 3. 网络内部：447 个输入怎样变成 1 个预测

任务配置把 Critic 隐藏层设为 `[512,256,128]`。实际网络是：

```text
[batch,447]
    ↓ Linear(447,512) + ELU
[batch,512]
    ↓ Linear(512,256) + ELU
[batch,256]
    ↓ Linear(256,128) + ELU
[batch,128]
    ↓ Linear(128,1)
[batch,1]
```

最后一层没有 sigmoid、softmax 或 tanh，所以 `value` 不是限定在 `[0,1]` 的概率，也没有预设正负范围。一次传入 4096 个环境，会各输出一个预测，形状为 `[4096,1]`，而不是把 4096 个环境合起来只打一个总分。证据：[隐藏层配置](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:313)、[Critic 层构造](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:95)、[默认激活及 ELU 实现](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:26)、[get_activation](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:182)。

真正的计算接口只有：

```python
def evaluate(self, critic_observations, **kwargs):
    value = self.critic_body(critic_observations)
    return value
```

来源：[actor_critic.py:173](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:173)。

**Critic 不经过 Actor 的历史编码器，也不先调用 12 维状态解码器。** `critic_body` 与 `actor_body` 是两个独立构造的 MLP；同在一个 `ActorCritic` 容器中不代表共享线性层。Critic 的输入直接来自仿真特权观测，Actor 的输入与潜变量路径见 Actor 专篇。证据：[Actor 构造与 Critic 构造](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:83)、[Actor 的编码路径](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:143)。

## 4. 一条真实训练数据经过 Critic 的完整路径

### 第一步：行动之前，先预测当前状态的价值

Runner 获取当前 Actor 观测、Critic 特权观测和监督标签。`PPO.act()` 分别计算动作与价值：

```python
self.transition.actions = self.actor_critic.act(obs).detach()
self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
self.transition.observations = obs
self.transition.critic_observations = critic_obs
```

这里发生的是“同一个时刻分别前向计算”，不是“Critic 先挑出一个动作，再交给 Actor 执行”。`detach()` 把这次采样得到的旧预测当作固定数据；Runner 的整段 rollout 还包在 `torch.inference_mode()` 中，不保留环境交互过程的梯度图。

证据：[PPO.act](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py:77)、[Runner rollout](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py:119)。

### 第二步：Actor 的动作进入环境，环境产生奖励

`env.step(actions)` 完成关节控制和物理仿真，计算 `r_t`、终止标记 `done_t`，再生成新观测。PPO 把刚才的旧价值与实际奖励对应起来存入 rollout：

```text
(c_t, a_t, r_t, done_t, V_old(c_t), ...)
```

`a_t` 来自 Actor，`r_t` 来自环境，`V_old(c_t)` 来自 Critic。它们是同一条 transition 的不同字段，不能混成一个“打分结果”。证据：[环境 step 的返回值](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:100)、[process_env_step](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py:90)、[add_transitions](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/rollout_storage.py:54)。

默认保存 24 步 × 4096 个环境：特权观测 `[24,4096,447]`，旧价值、奖励、return 和 advantage 均为 `[24,4096,1]`。这是张量存储顺序，不是把 24 帧再并进 Critic 的输入；Critic 每次依旧只吃 447 维。证据：[storage 张量创建](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/rollout_storage.py:24)、[24 步配置](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/base/legged_robot_config.py:211)。

### 第三步：rollout 结束，补上最后一个状态的价值

```python
last_values = self.actor_critic.evaluate(last_critic_obs).detach()
self.storage.compute_returns(last_values, self.gamma, self.lam)
```

如果第 24 步后回合还没结束，后续奖励尚未收集，就用 `V_old(c_24)` 近似未看到的未来。这就是 bootstrap：**用一个已有的价值估计，补足短轨迹后面的回报**。24 步 rollout 的结尾不等于回合结束。证据：[compute_returns 的入口](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py:102)、[最后状态的调用位置](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py:147)。

### 第四步：从后往前，用 GAE 构造监督目标

先忽略超时校正，记 `d_t` 为 `done`，则代码对应：

\[
\delta_t=r_t+\gamma(1-d_t)V_{old}(c_{t+1})-V_{old}(c_t),
\]

\[
A_t^{raw}=\delta_t+\gamma\lambda(1-d_t)A_{t+1}^{raw},
\]

\[
\hat R_t=A_t^{raw}+V_{old}(c_t).
\]

`γ=0.99`、`λ=0.95`。这份代码中 Critic 的拟合标签是 `returns_batch = R_hat`；Actor 使用的优势会再按整批均值、标准差归一化。因此“未经归一化的 A 为正”和“最终送给 PPO 的标准化 A 为正”不能无条件画等号。

证据：[GAE 循环与优势标准化](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/rollout_storage.py:72)、[γ、λ 配置](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/base/legged_robot_config.py:206)。

**Critic 的训练标签不是人工填写的“标准关节角度”。** 标签由执行得到的奖励、终止信息以及旧价值预测共同构造。它学的是未来奖励的预测能力，而不是监督学习的动作答案。

### 第五步：用当前网络重新计算价值，拟合刚才的目标

rollout 展平为 98,304 条样本，默认分 4 个 minibatch，每批 24,576 条，重复 5 个 epoch。每个 minibatch 重新执行：

```python
value_batch = self.actor_critic.evaluate(critic_obs_batch)
```

此时没有 `detach()`，也没有 rollout 外层的 `inference_mode()`，因此可以对 Critic 参数求梯度。证据：[batch 的展平与生成](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/rollout_storage.py:96)、[epoch 与 minibatch 配置](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/base/legged_robot_config.py:202)、[update 中重新 evaluate](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py:119)。

默认启用 clipped value loss：

\[
V^{clip}=V_{old}+\operatorname{clip}(V_\phi-V_{old},-0.2,+0.2),
\]

\[
L_V=\operatorname{mean}\left[\max\left((V_\phi-\hat R)^2,(V^{clip}-\hat R)^2\right)\right].
\]

注意代码里的 `target_values_batch` **是采样时保存的旧价值**，名字容易误导；真正拟合的 return 目标是 `returns_batch`。`target_values_batch` 来自 `storage.values`，不是人为设定的最终分数。

证据：[minibatch 字段的来源](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/rollout_storage.py:105)、[value loss 的实际实现](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py:154)。

数值例子：旧预测 `8.0`、拟合目标 `8.5`，更新过程中当前预测变为 `8.4`。

```text
裁剪后的预测 = 8.0 + clip(8.4 - 8.0, -0.2, 0.2) = 8.2
普通平方误差 = (8.4 - 8.5)^2 = 0.01
裁剪平方误差 = (8.2 - 8.5)^2 = 0.09
本样本 value loss = max(0.01, 0.09) = 0.09
```

这里是对**损失中参与比较的价值变化**做裁剪，并非强行限制 Critic 输出在某个固定区间，也不是最终更新必须精确停在 `8.2`。

### 第六步：反向传播真正修改 Critic 参数

```python
loss = surrogate_loss + value_loss_coef * value_loss \
       - entropy_coef * entropy_batch.mean()
self.optimizer.zero_grad()
loss.backward()
nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
self.optimizer.step()
```

默认 `value_loss_coef=1.0`。Critic 的参数沿 `value_loss → value_batch → critic_body` 的计算图更新；Actor 则沿策略目标的计算图更新。虽然损失合并相加、优化器也拿到整个 `ActorCritic` 参数集，但两者没有因此共享输出层，也没有让 Actor 直接接收 Critic 的那个标量。

证据：[损失与优化器更新](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py:164)、[Adam 注册的参数范围](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py:52)、[value loss 权重](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/base/legged_robot_config.py:196)。

## 5. 单条 transition 的完整算例

假设这条 transition 是 rollout 最后一条，回合仍在继续；Critic 旧预测 `V(c_23)=8.0`，执行 Actor 动作后环境给奖励 `r_23=0.12`，新状态的 bootstrap 预测 `V(c_24)=8.1`。

| 步骤 | 输入 | 输出 |
|---|---|---|
| 当前 Critic 前向 | `c_23`，447 维 | `V_old=8.0` |
| 环境执行 | Actor 的 17 维动作 | `r=0.12`、`done=0`、新观测 |
| rollout 末端预测 | `c_24`，447 维 | `V_next=8.1` |
| TD 残差 | `0.12 + 0.99×8.1 - 8.0` | `δ=0.139` |
| GAE | 此处为 rollout 最后一步，后续 GAE 初值为 0 | `A_raw=0.139` |
| return 标签 | `8.0 + 0.139` | `R_hat=8.139` |
| Critic 更新 | 重新前向的 `V_φ(c_23)` 与固定 `8.139` | 根据 value loss 更新网络参数 |
| Actor 更新 | 该样本的标准化优势、动作概率比率 | 按 PPO 更新动作分布，细节见 PPO 专篇 |

因此，Critic 原来以为“从这里出发未来约值 8.0”，当前短轨迹和后续预测提示它更接近 `8.139`，于是训练目标要求它修正预测。它并没有输出“把肩关节调到 8.139 rad”。

如果再往前一条样本是 `V(c_22)=7.9`、奖励 `0.10`、没有终止，则：

```text
δ_22 = 0.10 + 0.99×8.0 - 7.9 = 0.12
A_22 = 0.12 + 0.99×0.95×0.139 = 0.2507295
R_hat_22 = 7.9 + 0.2507295 = 8.1507295
```

这就显示出 GAE 的“后面发生的好坏继续向前传播”：前一步不仅参考下一状态的价值，还参考下一步的预测偏差。算例逐式对应 [compute_returns](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/rollout_storage.py:72)。

## 6. 跌倒终止、回合超时、rollout 截断：三者要分开

### 6.1 正常任务失败导致终止

环境会检查姿态角阈值等条件，并设置 `reset_buf`。例如机身 pitch 绝对值超过 `1.0 rad` 或 roll 超过 `0.8 rad` 会触发重置。此时 `done=1`，GAE 中的 `(1-done)=0`，不把重置后的新回合价值接到旧回合后面。证据：[check_termination](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:199)、[GAE 终止 mask](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/rollout_storage.py:79)。

沿用 `V_old=8.0`、`r=0.12` 的算例，若这是失败终止：

```text
δ = 0.12 + 0 - 8.0 = -7.88
R_hat = -7.88 + 8.0 = 0.12
```

### 6.2 仅因达到回合时间上限而超时

默认回合时间上限为 20 s。环境在 `episode_length_buf > max_episode_length` 时标记 `time_out_buf` 并设置 done，同时把 `time_outs` 交给 PPO。超时不一定说明策略失败，所以 PPO 在写 storage 前对奖励加一个 bootstrap 项。证据：[时间上限配置](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/base/legged_robot_config.py:4)、[超时判断](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:204)、[超时信息返回](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:292)。

**这里必须按这份代码的真实实现解释：**

```python
self.transition.rewards += self.gamma * torch.squeeze(
    self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
```

也就是说，它加的是本 transition 在行动前保存的 `V_old(c_t)`，**没有在这里重新计算一个专门的超时终止观测 `V(c_terminal)`**。后面的 GAE 仍因 `done=1` 截断，不读取重置后新回合的价值。

来源：[PPO.process_env_step:90](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py:90)、[行动前保存 value 的位置](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py:77)。

用同样算例：

```text
环境原始奖励 r = 0.12
写入 storage 的奖励 r' = 0.12 + 0.99×8.0 = 8.04
GAE 因 done=1 不再接下一回合
R_hat = 8.04
```

这是对当前仓库 bootstrap 实现的说明；不能把通用教材里“用真正的末状态价值 bootstrap”的表述原封不动写成它已经这样做了。

### 6.3 只是收满 24 步 rollout，回合并未结束

这种情况 `done=0`，不会因“收满 batch”触发环境重置；用第 24 步后的 `last_values` 补足后续即可，对应上一节的 `8.139` 算例。证据：[24 步采样和 compute_returns](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py:119)。

## 7. Critic 怎样帮助 Actor，它自己又被谁训练

可以按两条相连的数据路径理解：

```text
环境奖励 + 旧 Critic 价值
            ↓ GAE
      returns 与 advantages
       ↓                 ↓
Critic 的价值损失     Actor 的 PPO 策略损失
       ↓                 ↓
更新 critic_body      更新 actor_body、编码器、动作标准差
```

更精确地说：rollout 中的旧价值已 detach，return 与 advantage 是固定的 batch 数据。因此 Actor 的策略损失**不会沿着 advantage 反向传播回采样时的 Critic**；Critic 是通过单独的 value loss 学习的。两者通过训练数据与优势估计协作，而不是 Actor 把 Critic 的实时输出当控制输入。证据：[采样值 detach](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py:77)、[GAE 存储计算](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/rollout_storage.py:72)、[两项损失相加](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py:147)。

每个 minibatch 中还有一次状态估计辅助更新，但它走 `obs → adaptation_encoder → adaptation_decoder → MSE`。虽然辅助 Adam 也注册了容器的全部参数，这个辅助损失的计算图不包含 Critic，所以不会直接训练 `critic_body`。不能因为优化器参数列表相同，就说“辅助状态估计训练会更新 Critic”。证据：[两个优化器的初始化](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py:52)、[辅助更新真实前向与 backward](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py:178)、[get_student_latent](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:177)。

## 8. 前 8000 轮左右与后面的力课程：Critic 变了什么

两阶段的完整时间顺序见流程总览。本篇只强调 Critic 的变化：

| 项目 | 力课程触发前 | 力课程触发后 |
|---|---|---|
| 网络结构 | `447→512→256→128→1` | 相同，不更换网络 |
| 价值损失与 GAE | 已训练 | 继续按相同机制训练 |
| 末端注入力 / 力命令 | 默认无课程注入力、命令为零 | 课程会随机安排末端力命令与外力脉冲 |
| 特权输入 `[6:9]` | 零 | 随实际施加的末端外力变化 |
| 特权输入最后 3 维 | 无力偏移时就是原位置目标的对应球坐标 | 包含外力与力命令引起的位置偏移 |
| 机身外力课程 | 调用未启用 | 默认仍未启用，不能说这 3 维后来就一定非零 |

精确门槛是 `global_steps > force_start_step * 24`，而配置 `force_start_step=8000`。所以这里的“8000”对应约 8000 次 PPO iteration，不是仅 8000 次物理积分；严格条件是 global policy step 超过 192000。Critic 是在同一个训练过程中适应新分布，不是在门槛处重新创建或切换成另一个“力 Critic”。证据：[门槛与被注释的机身力调用](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:125)、[门槛配置](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:178)、[网络仅在 Runner 初始化时创建](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py:66)。

## 9. 推理时为什么可以不带 Critic

训练需要 Critic，是为了更好地估计优势和短 rollout 后面的回报；真正控制机器人只需要 Actor 根据可获得的观测输出动作。

项目的推理入口返回 `actor_critic.act_inference`，它调用 `act_student`，执行历史编码、Actor 动作均值计算，并可解码一些状态作为附带输出；这条执行路径没有调用 `evaluate()` 或 `critic_body`。因此在线控制不需要把摩擦系数、随机质量增量、真实注入力等 Critic 特权信息都测出来再运行 Critic。证据：[推理入口](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py:252)、[act_inference 与 act_student](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:158)。

**对照最初的问题：Actor 决定怎么动；环境奖励描述这一步实际发生的效果；Critic 预测从当前状态继续走下去的累计收益；PPO 用这些数据更新网络。** “Critic 打分”可以作为入门比喻，但具体代码中的“分”必须限定为价值预测，不能把即时奖励、动作评分和关节控制都混在一起。

## 10. 阅读与复核范围

- 文件定位、张量维度、坐标变换、GAE、超时 bootstrap、value loss 均按上述官方提交逐项静态核查。
- 没有修改官方源码，没有启动仿真或执行训练，也没有据此声称已验证实际训练收敛效果。
- 文章中的具体奖励 / 价值算例只用于手算解释，真实数值应以运行产生的日志和 rollout 张量为准。
