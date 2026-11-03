# UniFP Actor 详解：从观测到动作、目标角度和关节力矩

配套文档：[总体流程](/home/sorry/python_ws/interactive/unifp_reproduction/introduction/00-training-flow-zh-CN.md) · [Critic](/home/sorry/python_ws/interactive/unifp_reproduction/introduction/02-critic-zh-CN.md) · [PPO](/home/sorry/python_ws/interactive/unifp_reproduction/introduction/03-ppo-zh-CN.md)。

分析对象是官方 `b2z1_pos_force`，提交 `68847a070f88d731058c3d8476929bc3b205f5bd`。下文按实际执行顺序讲 Actor；数值例子均为解释代码而构造，并非实测轨迹。

## 1. 先回答：Actor 到底在做什么

在这个项目中，可以把 Actor 理解为“根据命令和观测，决定下一次关节目标如何偏离参考姿态”的神经网络策略。它输出腿与机械臂的 17 个动作数值；这些数值随后经过固定控制公式，变成目标角度和关节力矩。

```text
期望底盘速度、末端位置、力命令 + 传感观测历史
  ↓
历史编码器：推断与运动/扰动有关的 64 维特征
  ↓
Actor 网络：当前 73 维观测 + 64 维特征 → 17 维动作均值
  ↓
训练：从高斯分布采样动作；推理：使用均值
  ↓
动作 × 缩放与电机系数 + 固定默认角度
  ↓
目标关节角度
  ↓
PD 控制器根据实际角度/速度计算力矩
  ↓
物理仿真，机器人实际运动
```

这里把“历史编码器 + Actor 网络”合称为部署策略路径；代码中真正名为 `actor_body` 的模块只是其中的动作 MLP。出处：[Actor 构造](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:83)、[训练动作前向](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:143)、[动作转力矩](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:865)。

这里有两个层级的“命令”：输入的任务命令规定“希望机器人做到什么”，Actor 输出的关节动作决定“当前怎样驱动身体来完成任务”。训练时，前者由环境中的任务生成代码下达，后者才由神经网络学习产生。给定末端目标位置或力参考，并没有给出实现它的关节动作答案；策略仍需学习全身协调、平衡，以及面对不同状态和未知扰动时如何调整动作。PPO 负责更新策略参数，不是在线向 Actor 下达任务的上层控制器。

## 2. Actor 的输入：2336 个数分别是什么

### 2.1 一帧观测是 73 维

`compute_observations()` 实际拼接的是下面这些量。下标采用 Python 的左闭右开切片：

| 单帧位置 | 内容 | 维度 | 实际进入网络的表示 |
|---|---|---:|---|
| `[0:2]` | 机身 roll、pitch | 2 | `get_body_orientation()` |
| `[2:5]` | 机身三轴角速度 | 3 | `base_ang_vel × 0.25` |
| `[5:22]` | 17 个关节相对默认角度的偏差 | 17 | `(q - q_default) × 1.0` |
| `[22:39]` | 17 个关节角速度 | 17 | `q_dot × 0.05` |
| `[39:56]` | 上一个控制周期已执行的动作 | 17 | 环境 `self.actions` 的前 17 维 |
| `[56:57]` | 步态相位 sin | 1 | `sin(2π phase)` |
| `[57:58]` | 步态相位 cos | 1 | `cos(2π phase)` |
| `[58:73]` | 任务命令 | 15 | `commands × commands_scale` |
| 合计 | | **73** | |

源码摘录：

```python
obs_buf = torch.cat((self.get_body_orientation(),
    self.base_ang_vel * self.obs_scales.ang_vel,
    ((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos)
        [:, :-self.cfg.env.num_gripper_joints],
    (self.dof_vel * self.obs_scales.dof_vel)
        [:, :-self.cfg.env.num_gripper_joints],
    self.actions[:, :17],
    sin_pos,
    cos_pos,
    (self.commands * self.commands_scale)[:, :15],
), dim=-1)
```

出处：[观测实际拼接](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:414)、[缩放系数](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/base/legged_robot_config.py:129)。

这里的关节偏差是 `q - q_default`，不是 `q_target - q`。例如默认角度 `0.67 rad`、实际角度 `0.72 rad`，观测里该关节位置对应 `+0.05`；PD 所用的目标误差则还要看 Actor 这一步输出什么。

为什么称动作槽位为“上一周期动作”？环境先执行 `a_t`，再构造 `o_{t+1}`，所以 `o_{t+1}` 里包含 `a_t`。Actor 用 `o_{t+1}` 决定 `a_{t+1}`；它并没有提前看到自己的下一个输出。若 episode 结束，reset 会清零相应动作与历史。见 [step()](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:108)、[先执行后构造观测](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:169)、[动作清零](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:228)。

### 2.2 15 维命令也要拆开看

| 命令切片 | 含义 | 原始单位 | 进入网络的缩放 |
|---|---|---|---|
| `[0:3]` | 底盘 vx、vy、yaw rate | m/s、m/s、rad/s | 分别 ×2、×2、×0.25 |
| `[3:6]` | 末端目标 radius、pitch、yaw | m、rad、rad | 分别 ×0.5、×1、×1.3 |
| `[6:9]` | 末端姿态命令槽位 | rad | 各 ×0.5 |
| `[9:12]` | 期望末端力 xyz | N | 各 ×0.01 |
| `[12:15]` | 期望底盘力 xyz | N | 各 ×0.01 |

例如原始末端力命令 `30 N` 在这一输入槽位表示成 `0.30`。这是对输入特征缩放；不能据此把 Actor 输出的 `0.30` 也解释成 `30 N`。

当前默认代码中，姿态槽位没有启用对应的随机命令更新，底盘力调度的调用被注释。不能只看到 15 个槽位就认定所有命令都在有效变化。见 [命令缩放张量](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1440)、[末端目标命令更新](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:949)、[底盘力调用处](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:125)。

#### 这些命令具体由谁写入

`self.commands` 是环境维护的 15 维缓冲区，初始化为零。普通训练路径中的来源如下；它们不是 Actor 的输出，也不是辅助解码器的预测：

| 命令切片 | 生成与写入逻辑 | 默认执行情况 |
|---|---|---|
| `[0:3]` | `_resample_commands()` 按配置范围随机采样底盘速度 | 约每 5 秒及 reset 时重采样，另有零命令概率与小值清零 |
| `[3:6]` | `_resample_ee_goal()` 生成末端目标端点，`update_curr_ee_goal()` 插值后写入当前球坐标目标 | 每个控制周期更新当前插值点 |
| `[6:9]` | 预留末端姿态命令槽位 | 普通训练路径没有写入更新，保持初始化的零；另行生成的 `ee_goal_orn_quat` 不等于这三个槽位 |
| `[9:12]` | `_push_gripper()` 的 `cmd` 分支随机生成力参考，并按时间计划写入 | 力课程开启后可非零 |
| `[12:15]` | `_push_robot_base()` 的 `cmd` 分支包含写入逻辑 | 当前 `step()` 注释了该函数调用，故默认路径保持零 |

“上层任务给命令”是功能分工；此处无需另有一个上层神经网络，官方训练环境里的采样和调度函数就承担了这个角色。遥操作模式则跳过相应自动采样，由外部输入路径提供目标。出处：[速度命令采样](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:838)、[末端目标采样](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:887)、[末端力命令生成](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1037)、[底盘力命令生成](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1173)。

#### 官方末端球坐标相对于哪个坐标系

`[3:6]` 的球坐标相对于一个随底盘 XY 位置和 yaw 移动、保持水平的目标坐标系。设底盘世界坐标为 `(x_B,y_B,z_B)`、yaw 为 `ψ`，官方 B2+Z1 的球心为：

\[
c^W=[x_B,y_B,0]^T+R_z(\psi)[0.2,0,0.8]^T
\]

该坐标系 X 轴沿底盘的水平朝向，Y 轴向左，Z 轴沿世界竖直向上。它跟随底盘的平移 XY 和转向 yaw，但不跟随实际底盘高度 `z_B`、roll 或 pitch。配置注释写着高度偏移相对于地形，但当前球心函数的实际计算直接使用零 Z 加 `0.8 m`，没有在这里查询或加上局部地形高度；应以公式为准。

球坐标 `s=(r,θ,φ)` 中，`θ` 是相对水平面的仰角，`φ` 是从该坐标系 X 轴朝 Y 轴的方位角：

\[
p^C(s)=
\begin{bmatrix}
r\cos\theta\cos\phi\\
r\cos\theta\sin\phi\\
r\sin\theta
\end{bmatrix},\qquad
p^W_{\rm geom}=c^W+R_z(\psi)p^C(s)
\]

因此，同一组球坐标命令会随底盘 XY 和 yaw 改变其世界目标点，并非世界中的固定点。输入缩放 `[0.5r,θ,1.3φ]` 只改变数值尺度，不改变坐标系。监督标签中的真实末端球坐标也用同一球心和逆 yaw 旋转计算。出处：[官方球心配置](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:18)、[目标变换与球心函数](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:949)、[球坐标转换约定](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/utils/isaacgym_utils.py:59)、[真实末端坐标计算](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:352)。

#### 目标变化的幅度和时间

官方随机端点范围为 `r∈[0.35,0.95] m`、`θ∈[-2π/5,2π/5]`、`φ∈[-3π/5,3π/5]`；每次从范围内采样，没有固定“每次移动多少厘米”。路径有离散碰撞/地下检查，但这不构成完整的逆运动学可达性保证。当前目标是在起点 `s_A` 和终点 `s_B` 之间做球坐标线性插值：

\[
s^*(t)=s_A+\operatorname{clip}(t/T_{move},0,1)(s_B-s_A)
\]

控制周期约 `20 ms`，当前插值点随之更新；移动时长为 `1–3 s`，随后保持 `0.5–2 s`，一段名义周期为 `1.5–5 s`，边界按离散计时器触发。源码在环境初始化时为各环境采样这些时长，后续换端点并没有每次重抽时长。新段从上一个**期望端点**开始，计时器不会等机器人真正到达；reset 会中断当前段并重新设置初始化目标段。

Actor 直接收到的是当前插值目标和历史中的过去目标，没有独立输入最终端点、剩余时间或显式目标速度。球坐标线性插值也不等于笛卡尔直线或匀速运动。出处：[范围与时长配置](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:6)、[段起点与端点更新](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:916)、[时长初始化](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1388)。

### 2.3 从一帧变成 32 帧历史

代码将新观测追加到长度为 32 的 `deque`，按从旧到新的顺序堆叠，再展平：

```python
self.obs_history.append(obs_now)
obs_buf_all = torch.stack([self.obs_history[i]
                          for i in range(self.obs_history.maxlen)], dim=1)
self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)
```

```text
每个环境：32 × 73 = 2336
默认批量：[4096, 2336]
```

`observations[:, -73:]` 就是最新帧。开局和 reset 时历史被零填充，不是凭空已有 32 帧真实测量。若打开噪声，观测先加入按配置缩放的高斯噪声，再进入历史。见 [噪声与堆叠](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:424)、[历史初始化](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1464)、[reset 清零历史](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:300)。

### 2.4 哪些真值不直接输入 Actor

Actor 普通观测不含真实底盘线速度、真实末端位置、注入外力、质量、摩擦和电机系数。它能看到的是任务命令、本体运动测量及其历史。训练时，仿真真值另供 Critic 和辅助估计器使用。

尤其要区分“末端位置命令”和“真实末端位置”：Actor 直接接收前者，但后者不在这 73 维普通观测中。见 [普通观测](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:414)、[特权观测与监督标签](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:379)。

同样，“Actor 不知道实际外力”不等于“Actor 没有任何力相关输入”。`F_cmd` 是已知任务参考，进入 `commands[9:12]`；`F_ext` 是另一条独立采样并实际注入物理仿真的外力，不直接进入普通观测。环境知道 `F_ext`，因此可以算奖励、给 Critic 提供特权信息并构造监督标签；这些训练用途不等于把真实外力作为 Actor 前向输入。未知外力的影响只能通过运动历史间接体现。两者即使配置了相同的幅度范围，也不能视为同一股力。见 [`cmd` 与 `ext` 两条生成路径](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1037)、[外力实际施加](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:130)。

## 3. 历史编码器、Actor、解码器是什么关系

### 3.1 历史编码器

编码器结构由 `AC_Args` 和 `num_latent_dim` 决定：

```python
self.num_latent_dim = int(num_obs / num_single_obs) * 2
```

因此本任务是 `32 × 2 = 64` 维 latent：

```text
输入历史 [N,2336]
 → Linear(2336,512) → ELU
 → Linear(512,256)  → ELU
 → Linear(256,128)  → ELU
 → Linear(128,64)
输出 latent [N,64]
```

这是一次性处理展平历史的 MLP；这里没有使用 LSTM/GRU，也不是每个关节单独配一套编码器。见 [编码器定义](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:46)。

### 3.2 Actor 的实际输入是 137 维

```python
latent = self.adaptation_encoder_module(observations)
mean = self.actor_body(torch.cat(
    (observations[:, -self.num_obs_now:], latent), dim=-1))
```

```text
最新帧 [N,73] + 历史特征 [N,64]
 → 拼接 [N,137]
 → Linear(137,512) → ELU
 → Linear(512,256) → ELU
 → Linear(256,128) → ELU
 → Linear(128,17)
 → action_mean [N,17]
```

最后一层是线性输出，没有 tanh 把动作压到 `[-1,1]`。见 [网络构造](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:83)、[实际前向](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:143)、[本任务隐藏层](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:316)。

### 3.3 解码器是辅助监督分支，不是 Actor 的前置力传感器

它在代码里的准确名称是 **`adaptation_decoder_module`**，可称为“辅助状态估计解码器”；与 `adaptation_encoder_module` 一起构成适应/并行状态估计分支。它与动作头共享编码器，并非另一个控制关节的 Actor。`ActorCritic` 这个类同时容纳编码器、动作头、解码器和 Critic，不能仅按类名把内部所有模块都理解成动作输出网络。

```text
历史 → 编码器 → 64维 latent ──┬── + 当前73维 → Actor → 17维动作均值
                            └── 解码器 → 12维状态预测 → 监督损失
```

12 维预测是：

```text
[底盘线速度3，末端实际球坐标3，末端注入力3，底盘注入力3]
```

预测和标签都处在代码规定的缩放表示下。例如力乘 `0.01`，解读物理牛顿值时需要逆变换。真实力标签来源是代码向仿真注入的 `forces` 缓冲，并非任意接触反力的完整测量。

关键是：**Actor 直接使用 64 维 latent，没有先把 12 维预测值拼接到 Actor 输入。** 辅助监督使 latent 包含与这些物理量有关的信息，但“估计出外力后显式代公式算出动作”并不是这里的 Actor 前向路径。见 [解码器定义](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:67)、[状态预测接口](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:177)、[标签构造](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:405)。

推理函数还有一个容易误读的命名：`policy_info['latents']` 保存的是解码后的 **12 维预测**，并非内部 **64 维 latent**。见 [act_student()](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:161)。

可以把前向关系明确写成：

\[
z_t=E(H_t),\qquad \mu_t=A([o_t,z_t]),\qquad \hat y_t=D(z_t)
\]

其中 `H_t` 是观测历史，`D` 输出上述 12 维估计，外力估计只是其中一部分。这里不存在把 `\hat F_ext` 再送入动作头的连线；`act_student()` 先计算动作均值，再解码记录预测值。因此，若只省略解码器的推理计算、保留编码器权重和动作路径，相同输入的动作均值不会改变；原始接口仍会计算解码值用于记录。训练时去掉解码损失则是另一回事，会改变编码器接受的训练信号。

解码器的首要作用是**借助仿真真值监督共享表示**：预测误差反向更新解码器和编码器，促使 64 维特征保留与受力、速度、末端状态有关的信息；动作头随后使用这些特征。它不只是用于显示数值，也不是直接决定“应该施加多少力”的控制器。64 个特征没有被逐维指定物理含义；即使可以从中解码外力，也不证明 Actor 一定有效利用了这部分信息。具体梯度分工见第 8 节，估计与控制的评估区别见第 10 节。

## 4. Actor 输出的动作数值究竟是什么

### 4.1 训练时输出的是高斯分布的样本

```python
self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

# update_distribution()
self.distribution = Normal(mean, mean * 0. + self.std)

# act()
return self.distribution.sample()
```

对于一个环境，17 个关节各有一个高斯分布：

\[
a_{t,i}\sim\mathcal N(\mu_{t,i},\sigma_i^2),\quad i=1,\ldots,17
\]

- `mean:[N,17]` 随当前输入改变。
- `std:[17]` 是可学习参数，由所有环境共享，不是每次由输入网络再预测一套 std。
- 训练中执行的是采样动作，不一定正好等于均值。
- PPO 使用同一 17 维动作在新旧分布下的 log probability；各维 log probability 相加。

源码：[std 参数](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:112)、[分布/采样/log probability](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:143)。

例如一个关节均值为 `0.4`，当前标准差为 `0.1`，一次可能采到 `0.35`、下次可能采到 `0.47`。这里的数字是连续实数动作，既不是分类概率，也不是某个动作类别的编号。

### 4.2 推理时使用均值

`act_student()` 返回 `actions_mean`，不调用 `distribution.sample()`。因此固定权重、固定相同输入时，动作输出由网络前向确定。实际运动仍会受不同状态、扰动和输入噪声影响。

出处：[act_inference/act_student](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:158)。

### 4.3 动作本身不是角度，缩放后才是角度残差

环境先做：

```python
clip_actions = self.cfg.normalization.clip_actions
self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
```

默认 `clip_actions=100`；不是常见教程中假定的 `[-1,1]`。随后：

```python
actions_scaled = actions * self.motor_strength * self.cfg.control.action_scale
```

可用以下公式区分：

\[
\bar a_{t,i}=\operatorname{clip}(a_{t,i},-100,100)
\]

\[
\Delta q_{t,i}=0.25\,m_i\bar a_{t,i}
\]

\[
q^{target}_{t,i}=q^{default}_i+\Delta q_{t,i}
\]

- `a`：无量纲的策略动作数值。
- `Δq`：关节角度偏移，单位 rad。
- `q_target`：PD 希望关节达到的角度，单位 rad。
- `m_i`：`motor_strength`；本任务范围 `[0.85,1.15]`。

注意源码将这个电机系数乘在**动作残差**上，并非在算完 PD 后把整个力矩统一乘 `m_i`。复现时必须遵循实际公式。见 [动作裁剪](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:107)、[控制公式](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:877)、[action_scale](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:227)、[随机电机系数](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1713)。

## 5. 默认角度到底是什么，为什么不变

### 5.1 它是人写在配置里的参考姿态

配置明确注释为“动作为零时的目标角度”：

```python
default_joint_angles = {  # = target angles [rad] when action = 0.0
    'FL_hip_joint': 0.15,
    'FL_thigh_joint': 0.67,
    'FL_calf_joint': -1.32,
    # ...
    'z1_waist': 0.0,
    'z1_shoulder': 1.48,
    'z1_elbow': -1.5,
    'z1_wrist_angle': 0,
    'z1_forearm_roll': 0.0,
    'z1_wrist_rotate': 1.57,
    'z1_jointGripper': -0.785,
}
```

完整腿部参考值如下：

| 关节组 | hip | thigh | calf |
|---|---:|---:|---:|
| FL 左前腿 | 0.15 | 0.67 | -1.32 |
| FR 右前腿 | -0.15 | 0.67 | -1.32 |
| RL 左后腿 | 0.15 | 0.67 | -1.32 |
| RR 右后腿 | -0.15 | 0.67 | -1.32 |

全部单位为 rad。它们表示各个旋转关节沿自身 URDF 轴的关节坐标，不是机器人末端在世界坐标里的 x/y/z，也不是机身的 roll/pitch/yaw。见 [默认角度完整配置](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:39)、[FL 大腿旋转关节定义](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/resources/robots/b2z1/b2z1.urdf:374)。

### 5.2 初始化时读入常量，不由 PPO 学习

```python
self.default_dof_pos = torch.zeros(
    self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
for i in range(self.num_dofs):
    name = self.dof_names[i]
    angle = self.cfg.init_state.default_joint_angles[name]
    self.default_dof_pos[i] = angle
```

因此当前实现中默认角度是固定参考张量，按仿真资产的 `dof_names` 顺序从名字字典装配。PPO 优化的是网络参数，而不是把配置字典当作可学习参数。换配置并重新初始化可以改默认姿态，但这与训练自动改变默认角度是两回事。见 [默认角度初始化](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1473)。

reset 中随机化的是**实际关节位置**：腿部围绕默认值乘随机比例，机械臂前 5 个关节加随机偏移；并没有改写 `default_dof_pos`。见 [reset 的实际赋值](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:988)。

代码中另有 `ref_dof_pos`，它是在固定默认角度上叠加步态相位变化的**动态参考**，用于步态奖励和 Critic 特征。它与 `default_dof_pos` 是两个变量；动作转关节 PD 目标时使用的仍是固定的 `default_dof_pos`。因此看到 `ref_dof_pos` 变化，不表示默认角度被 PPO 改写。见 [动态参考计算](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1818)、[PD 使用固定默认角度](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:879)。

### 5.3 三种角度始终分开

| 名称 | 符号 | 来源 | 是否随控制时刻变化 |
|---|---|---|---|
| 默认角度 | `q_default` | 配置常量 | 不变 |
| 目标角度 | `q_target` | 默认角度加 Actor 给出的残差 | 会变 |
| 实际角度 | `q` | 仿真/编码器测得的实际关节状态 | 会变 |

固定默认角度并不把机器人锁在参考姿态；策略通过每个时刻变化的偏移持续产生不同目标。

## 6. 一次动作如何变成实际力矩：完整数值例子

考虑 `FL_thigh_joint`，假设此时 Actor 执行动作 `a=0.4`，`motor_strength=1`，实际角度 `q=0.72 rad`，实际角速度 `q_dot=0.20 rad/s`。

该关节配置：`q_default=0.67 rad`、`Kp=300 N·m/rad`、`Kd=7.5 N·m·s/rad`。

### 第一步：动作转角度偏移

```text
Δq = 0.25 × 1 × 0.4 = 0.10 rad
```

### 第二步：加到固定参考角度

```text
q_target = 0.67 + 0.10 = 0.77 rad
```

### 第三步：与实际状态比较

```text
位置误差 = 0.77 - 0.72 = 0.05 rad
```

### 第四步：PD 输出力矩

源码是：

```python
default_torques = self.p_gains * (
    actions_scaled + self.default_dof_pos_wo_gripper - self.dof_pos_wo_gripper
) - self.d_gains * self.dof_vel_wo_gripper
```

代入：

```text
位置项 = 300 × 0.05 = 15 N·m
阻尼项 = 7.5 × 0.20 = 1.5 N·m
τ = 15 - 1.5 = 13.5 N·m
```

再按该关节 URDF 力矩上限限幅，送给仿真。这个例子没有触发限幅。随后实际角度如何变化，由力矩、惯性、重力、接触和外力共同决定；不会在这一行赋值后直接跳到 `0.77 rad`。

出处：[PD 计算与限幅](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:877)、[默认角度](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:43)、[PD 增益](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:230)、[施加力矩与仿真](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:113)。

### 两个容易理解反的情况

**动作零不代表力矩零。** 若同一关节 `a=0`、`q=0.72`、`q_dot=0.20`：

```text
目标角度回到 0.67 rad
τ = 300 × (0.67 - 0.72) - 7.5 × 0.20 = -16.5 N·m
```

**下一步偏移不累加到上一步目标。** 若下一周期输出 `a=-0.2` 且 `m=1`：

```text
q_target_next = 0.67 + 0.25 × (-0.2) = 0.62 rad
```

不是从上一目标 `0.77` 减到 `0.72`。这是源码中 `actions_scaled + default_dof_pos` 的直接含义。

## 7. 为什么有 17 个动作，却产生 19 个力矩

当前资产有 19 个非固定关节 DOF；策略只控制其中前 17 个，即 12 个腿部关节和 5 个机械臂关节。准确的运行时对应顺序由 `self.dof_names` 确定，不应从配置字典的书写顺序猜测。

配置把末尾两个 DOF 统称为 `num_gripper_joints=2`，但对照资产，它们分别是 `z1_wrist_rotate` 与 `z1_jointGripper`，不能解释成“两个夹爪手指”。这两个关节没有独立 Actor 动作，而是用固定默认角度做 PD：

```python
last_torque = 64. * (0 + self.default_dof_pos[:, -2]
    - self.dof_pos[:, -2:-1]) - 1.5 * self.dof_vel[:, -2:-1]
gripper_torque = 64. * (0 + self.default_dof_pos[:, -1]
    - self.dof_pos[:, -1:]) - 1.5 * self.dof_vel[:, -1:]
torques = torch.cat([default_torques, last_torque, gripper_torque], dim=-1)
```

上面为便于排版，将源码的零张量项写成 `0`。整体形状是：

```text
策略动作              [N,17]
策略控制关节的PD力矩  [N,17]
另两个关节固定参考PD  [N, 2]
传给仿真的完整力矩    [N,19]
```

出处：[任务动作配置](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:102)、[DOF 名称与截取](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1643)、[末两关节控制](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:880)、[wrist_rotate 定义](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/resources/robots/b2z1/b2z1.urdf:1747)、[Gripper 定义](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/resources/robots/b2z1/b2z1.urdf:1815)。

## 8. Actor 是怎样被训练的

### 8.1 没有“正确关节角度”的逐样本监督标签

训练脚本没有给每个状态准备一个专家关节角度，让 Actor 对着它做角度 MSE。Actor 采样动作，仿真执行，环境依据任务完成程度计算奖励，PPO 再调整产生这些动作的分布。

```text
当前参数 → 动作采样 → 物理结果 → 奖励/未来回报估计
                                      ↓
                      PPO：调整旧动作在当前分布下的概率
                                      ↓
                          Actor 与编码器参数发生变化
```

奖励同时考虑速度、末端跟踪、平稳性、碰撞和力矩等，所以“正确”表示长期任务效果好，并不是某个关节唯一的标准答案。出处：[奖励项](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:271)、[采样与更新入口](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py:119)、[策略损失与参数更新](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py:147)。

#### 已经给了力参考，为什么还需要学习

力参考规定任务条件，没有规定各关节应该给多少角度残差或力矩。官方启用的 `tracking_ee_force_world` 奖励具体通过**力相关的位置偏移**定义末端目标；它不是简单要求机器人制造出一个数值等于 `F_cmd` 的接触反力。按代码统一到世界坐标后：

\[
p^W_{reward}=p^W_{geom}
+\frac{F^W_{ext}+R_z(\psi)F^{yaw}_{cmd}}{K}
\]

除法按轴进行，官方 `gripper_force_kps` 配置为各轴 `200 N/m`；这里的任务刚度 `K` 不同于关节 PD 的位置增益。奖励根据实际末端位置与 `p_reward` 的距离计算。Actor 直接看见几何球坐标目标和 `F_cmd`，看不见 `F_ext` 及完整的 `p_reward`；因此策略仍需利用运动历史中的受力线索，通过全身关节动作实现这种力与位移的关系。

例如仅看同一坐标系的一轴，`F_cmd=+20 N`、`F_ext=-10 N`、`K=200 N/m`，该轴奖励目标相对几何目标偏移 `+0.05 m`。Actor 得到的力命令是 `+20 N`，并未被直接告知 `-10 N` 外力或最终 `+0.05 m` 偏移。这个例子说明任务关系，不是对训练成功的保证。外力既改变机器人动力学，也改变奖励目标，因而不能只把它理解成必须完全抵消的扰动。出处：[奖励目标公式](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1891)、[任务刚度配置](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:148)、[偏移目标仅进入特权观测](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:371)。

### 8.2 哪些参数由哪些信号改变

| 模块或数值 | PPO 损失能更新 | 辅助估计损失能更新 | 原因 |
|---|---|---|---|
| `actor_body` 权重 | 是 | 否 | 动作概率经过它；状态解码不经过它 |
| `adaptation_encoder_module` | 是 | 是 | 同时连到动作分布和状态预测 |
| `adaptation_decoder_module` | 否 | 是 | 仅在状态预测监督分支使用 |
| `std` | 是 | 否 | 决定训练动作概率和熵 |
| `critic_body` | 是，由 value loss | 否 | 价值回归分支 |
| 默认角度、PD 增益 | 否 | 否 | 是环境配置/张量，不在可学习网络前向参数中 |

两个 Adam 在构造时都接收了 `actor_critic.parameters()`，但具体哪部分有梯度由当前计算图决定。不要仅根据 optimizer 的参数列表推断“每次所有模块都更新”。历史编码器收到的两种训练信号，以及加权 MSE 的真实写法，详见 [PPO 专篇](/home/sorry/python_ws/interactive/unifp_reproduction/introduction/03-ppo-zh-CN.md)。源码：[优化器定义](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py:53)、[两次反向传播](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py:166)。

每个小批次先执行 PPO 更新，再执行辅助状态预测的加权 MSE 更新。准确说法是“PPO 与辅助监督交替联合训练”，不是“PPO 奖励直接训练力解码器”。辅助分支虽然不更新动作头权重，却会更新共享编码器，从而影响后续动作；这正是它参与控制学习的间接途径。相应地，估力可以学得更好而任务控制仍不理想，两种损失的改善并不必然同步。

## 9. “联合策略”在这里具体指什么

可以从当前代码理解为：一个策略网络同时根据底盘与末端命令，输出全部受控腿部和机械臂关节的动作，使全身动作协同完成任务。

```text
同一个 Actor 的一次输出：[12个腿关节动作，5个机械臂关节动作]
```

它不是“PPO 发一个上层动作，Actor 再发下层动作”的两层在线控制结构，也不是腿和手臂各有一个互不相关的独立 Actor。PPO 是训练算法；Actor 是运行时出动作的策略；Critic 是训练时提供价值估计的网络。

这一解释针对公开代码的结构，不把“联合策略”这个中文词当成另一套代码里不存在的模块。依据：[单个 actor_body 的输出维度](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:83)、[训练调用路径](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py:119)。

## 10. 力课程前后，Actor 到底变了什么

| 项目 | 阈值前 | 阈值后 |
|---|---|---|
| 输入/输出结构 | 2336 维历史 → 17 维动作 | 相同 |
| 默认角度、动作尺度、PD 公式 | 固定配置 | 相同 |
| 末端力命令 | 0 | 可非零 |
| 注入末端外力 | 不启动调度 | 按时间计划注入 |
| 辅助解码的末端外力标签 | 0 | 随实际注入外力变化 |
| 解码出的外力是否直接输入动作头 | 否，动作头使用共享 latent | 相同 |
| 奖励使用的位置目标 | 原始位置目标 | 原始目标加力/刚度偏置 |
| Actor 参数 | 学习运动与位置跟踪 | 在原权重基础上继续学习力适应 |

所以力课程后，Actor 仍然控制关节位置残差，通过整机运动产生任务所需的行为；它没有换成一个直接输出牛顿值的力网络。阶段条件是 `global_steps > 8000 * 24`，具体计数单位见 [总体流程第 4 节](/home/sorry/python_ws/interactive/unifp_reproduction/introduction/00-training-flow-zh-CN.md)。源码：[课程开关](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:125)、[统一力/位置奖励](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1891)。

“8000”是训练轮数阈值，经每轮 24 个策略步换算，并不是只执行了 8000 个物理步。阈值前辅助解码器已经在训练，但末端外力标签恒为零；此时低估力误差可能只来自输出零，不能据此证明学会辨别非零外力。阈值后也不是所有环境立即持续受力，而是开始按各自时间计划采样和施加力脉冲。

### 10.1 解码器能衡量什么，不能替代什么

| 观察对象 | 可以回答的问题 | 不能单独推出的结论 |
|---|---|---|
| 解码外力与真实注入力的误差 | 在这些验证条件下，历史表示能否支持外力估计 | Actor 已利用估计改善控制 |
| 末端跟踪、底盘速度、稳定性与终止情况 | 闭环任务效果是否达到要求 | 改善一定由估力辅助任务造成 |
| 训练奖励与辅助损失 | 优化过程与各项训练信号如何变化 | 未见条件下控制或估力必然可靠 |

估力应与“始终预测零”的基线比较，并分别检查非零受力误差和无外力时的误报；控制则同时看几何目标误差、上述力偏移目标误差、运动跟踪与稳定性，不能只用估力 RMSE 或训练奖励代表整体效果。若估力好而控制差，只能说明“表示可解码出受力信息，但闭环控制尚未达到要求”，不能仅凭这一点确定动作头为何没有完成任务。要证明辅助估力确实带来控制收益，需要同条件下有、无辅助监督等对照实验；仅省略推理解码计算没有切断训练时已经形成的影响。

## 11. 最后沿一条样本串起来

以下只摘出一个环境、一个关节的值，其余维度省略：

```text
输入测量：q=0.72 rad、q_dot=0.20 rad/s
固定参考：q_default=0.67 rad
    ↓
单帧位置槽位：q-q_default=0.05
单帧速度槽位：0.20×0.05=0.01
加上机身状态、上一动作、命令、相位 → 73维
最近32帧 → 2336维
    ↓
编码器 → 64维特征
最新73维 + 64维 → Actor
    ↓
假设输出均值为0.35，本次训练采样动作0.40
    ↓
0.40×0.25×1=0.10 rad
目标角度=0.67+0.10=0.77 rad
    ↓
PD力矩=13.5 N·m
    ↓
仿真更新实际状态，环境计算奖励，构造下一观测
    ↓
PPO 在凑满rollout后，根据优势信号更新产生动作的网络
```

这条链分别区分了输入观测、网络特征、动作数值、角度目标、实际角度和力矩。上面的 `0.35/0.40` 是假设的策略输出，PD 参数与计算形式来自前文引用的官方代码。

---

核查范围：官方源码静态阅读；未运行训练、未测量真实动作分布或力控制效果。
