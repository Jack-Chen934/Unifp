# UniFP 官方项目：完整训练流程与两个训练阶段

本文只讲训练流程、每一步的输入输出和前后阶段变化。Actor、Critic、PPO 的原理与逐段代码分析分别放在自己的文件中。

## 文档导航

| 文档 | 阅读目的 |
|---|---|
| [01 · Actor](/home/sorry/python_ws/interactive/unifp_reproduction/introduction/01-actor-zh-CN.md) | 观测如何变成 17 个动作，动作如何变成目标角度和关节力矩，默认角度为什么固定 |
| [02 · Critic](/home/sorry/python_ws/interactive/unifp_reproduction/introduction/02-critic-zh-CN.md) | 447 维特权状态如何变成价值估计，它与即时奖励有什么区别 |
| [03 · PPO](/home/sorry/python_ws/interactive/unifp_reproduction/introduction/03-ppo-zh-CN.md) | rollout、GAE、概率比、裁剪损失和反向传播如何更新网络 |

建议顺序：先读本文的总流程，再读 Actor → Critic → PPO，最后回来看两个阶段的区别。

## 1. 分析对象与证据范围

- 官方仓库：`unified-force/UniFP`，本地目录为 `/home/sorry/python_ws/interactive/unifp_reproduction/UniFP`。
- 核查提交：`68847a070f88d731058c3d8476929bc3b205f5bd`。
- 实际注册任务：`b2z1_pos_force`，B2 四足底盘加 Z1 机械臂。
- 本文分析官方仓库的代码，不把父目录下的复现脚本当成官方实现。
- 本次仅静态核查源码并整理文档，没有启动训练或验证收敛结果。文中的数字演算是说明例子。

出处：[任务注册](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/__init__.py:12)、[训练入口](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/scripts/train_b2z1posforce.py:13)。

## 2. 先看完整执行顺序

```text
读取任务配置、创建并行环境
  ↓
创建 ActorCritic、PPO 和 rollout 缓冲区
  ↓
reset：初始化实际关节状态、底盘状态、命令、历史缓存
  ↓
获取 obs / privileged_obs / obs_pred
  ↓
┌──────────────── 每轮训练 iteration ────────────────┐
│                                                  │
│ 重复采样 24 个策略步：                              │
│   1. Actor 根据观测历史采样 17 维动作               │
│   2. Critic 估计当前状态的未来回报                  │
│   3. 保存旧动作概率、旧价值等训练信息               │
│   4. 动作 → 目标关节角度 → PD 力矩                  │
│   5. 执行 4 个物理子步；按阶段决定是否调度末端力     │
│   6. 计算奖励和 done；必要时重置结束的环境           │
│   7. 构造下一观测和估计器标签，写入 rollout         │
│                                                  │
│ 24 步结束 → 最后状态 bootstrap → GAE/return        │
│   ↓                                              │
│ 5 个 epoch × 4 个 minibatch：                      │
│   PPO 参数更新 → 状态估计辅助参数更新               │
│   ↓                                              │
│ 清空 rollout 有效游标，记录日志，按间隔保存模型       │
└────────────────── 进入下一轮 ─────────────────────┘
```

执行顺序可直接对照 [runner.learn()](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py:115)、[环境 step()](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:100)、[PPO.update()](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py:108)。

## 3. 各步具体输入输出

记 `N=4096`，表示默认并行环境数。这里的 `obs_pred` 是环境提供的监督真值；不是网络已经预测出来的结果。

| 步骤 | 输入 | 计算或行为 | 输出 |
|---|---|---|---|
| 1. 创建环境 | URDF、任务配置、仿真配置 | 创建 N 个机器人和地形，初始化随机物理参数 | 仿真状态、关节/刚体索引、固定默认角度、PD 增益 |
| 2. 创建学习对象 | 观测与动作尺寸、网络/优化配置 | 创建网络、两个优化器、rollout 缓冲区 | ActorCritic、PPO、storage |
| 3. reset | 需要重置的环境编号 | 随机初始化实际状态，重采样命令，清零历史等 | 初始仿真状态及观测字典 |
| 4. 策略前向 | `obs:[N,2336]` | 历史编码与动作分布采样 | `action:[N,17]`；另保存均值、标准差、log probability |
| 5. 价值前向 | `privileged_obs:[N,447]` | Critic 前向 | `value:[N,1]` |
| 6. 动作执行 | 动作、实际角度/角速度、默认角度、PD 增益 | 动作裁剪、角度残差缩放、PD、力矩限幅 | `torques:[N,19]`，包含额外两个固定参考控制的 DOF |
| 7. 仿真 | 力矩、当前物理状态、按阶段调度的外力 | 连续执行 4 个物理子步 | 更新的实际角度、速度、底盘/末端状态、接触等 |
| 8. 奖励与终止 | 新物理状态、命令、奖励配置 | 各奖励项加权；检查终止/超时 | `reward:[N]`、`done:[N]`、`infos` |
| 9. 下一观测 | 新状态、最后执行的动作、命令、历史缓存 | 按规定缩放并堆叠历史 | `obs:[N,2336]`、`privileged_obs:[N,447]`、标签 `obs_pred:[N,12]` |
| 10. 存储 transition | 步骤 4、5 的旧策略信息，以及步骤 8 的结果 | 写入同一时刻对应的训练样本 | 长度 24 的 rollout；每个时刻有 N 条样本 |
| 11. GAE/return | 奖励、done、旧价值、最后状态价值 | 反向递推回报目标和优势 | `returns:[24,N,1]`、`advantages:[24,N,1]` |
| 12. PPO 更新 | 旧 rollout、当前网络 | 重算同一旧动作的概率、价值，反向传播 | 更新 Actor、Critic、历史编码器和策略标准差 |
| 13. 辅助估计更新 | 相同历史观测、环境给出的 12 维标签 | 编码器→解码器，监督损失反向传播 | 更新历史编码器、状态解码器 |

注意步骤 8 之后会先 reset 已结束环境，再构造返回观测。因此 done=True 的样本返回的观测可能已属于新 episode；终止处理和 timeout 细节见 [Critic 专篇](/home/sorry/python_ws/interactive/unifp_reproduction/introduction/02-critic-zh-CN.md) 与 [PPO 专篇](/home/sorry/python_ws/interactive/unifp_reproduction/introduction/03-ppo-zh-CN.md)。

代码依据：[runner 初始化](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py:66)、[reset_idx](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:207)、[状态刷新与返回观测](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:144)、[观测构造](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:333)、[动作到力矩](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:865)、[GAE](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/rollout_storage.py:72)。

### 3.1 环境真正返回的接口

不是把六个张量都作为独立返回值。源码是：

```python
return {
    'obs': self.obs_buf,
    'privileged_obs': self.privileged_obs_buf,
    'obs_pred': self.obs_pred
}, self.rew_buf, self.reset_buf, self.extras
```

所以 runner 实际接收 `obs_dict, rewards, dones, infos` 四项，再拆开观测字典。见 [step() 返回值](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:135)。

### 3.2 reset 不等于重新随机化所有物理参数

reset 会重置实际关节和底盘状态、命令、力调度状态、历史观测等。摩擦、质量、质心、电机强度中的多项主要在环境创建时采样，不能把它们统称为“每次 reset 都重采样”。实际关节初始角度也会在默认角度周围扰动，但默认参考角度本身不变。

见 [reset_idx()](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:207)、[关节状态 reset](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:988)、[摩擦初始化回调](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:700)、[电机强度创建](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1713)。

## 4. 先把“8000 步”的单位说准确

配置写的是：

```python
force_start_step = 8000
```

真正启用课程的条件在 `step()` 的物理子步循环里：

```python
if self.global_steps > self.cfg.commands.force_start_step * 24:
    self._push_gripper(torch.arange(self.num_envs, device=self.device))
    # self._push_robot_base(torch.arange(self.num_envs, device=self.device))
```

且每次环境 `step()` 结束才执行 `self.global_steps += 1`。

因此应当分别讲：

- **阶段一：检查时 `global_steps <= 192000`，末端力调度尚未调用。**
- **阶段二：检查时 `global_steps > 192000`，调用末端力调度。**

默认每轮 rollout 为 24 个策略步，所以这大约对应 8000 轮 PPO iteration。它既不是 8000 个物理子步，也不是 8000 条 transition。环境 reset 自身也会调用一次零动作 step，判断又是严格大于号，因此具体切换时刻应以 `global_steps` 为准。

一个策略步包含 4 个物理子步：`0.005 s × 4 = 0.02 s`。这与 4096 个环境并行数无关：一次批量 step，`global_steps` 只加 1。阈值中的 `24` 在环境代码中是字面量；如果修改 runner 的 rollout 长度，阈值不会自动跟着改。

出处：[阈值配置](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:178)、[启动力条件](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:125)、[计数器递增](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:141)、[reset 内零动作 step](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/base/base_task.py:104)、[rollout 长度](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/base/legged_robot_config.py:214)、[物理 dt](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/base/legged_robot_config.py:165)、[decimation](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:235)。

## 5. 阶段一：先训练运动和末端位置跟踪

### 5.1 输入是什么

15 维原始命令按以下顺序组织，进入网络前还会缩放：

```text
0:3   底盘 vx、vy、yaw rate
3:6   末端目标球坐标：radius、pitch、yaw
6:9   末端姿态命令槽位
9:12  末端期望力
12:15 底盘期望力
```

阶段一采样底盘速度和末端位置轨迹，末端与底盘力命令保持 0。当前任务的姿态命令槽位没有启用相应的随机命令更新，姿态跟踪奖励权重也为 0，不能把它描述为完整六维末端位姿跟踪。

出处：[命令维度与速度范围](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:121)、[速度重采样](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:838)、[末端轨迹插值及命令](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:949)、[命令缩放顺序](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1440)、[姿态奖励配置](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:302)。

### 5.2 一次训练具体发生什么

1. Actor 看当前本体观测、命令和历史，输出 17 个腿/臂动作。
2. 动作变成相对固定默认角度的目标角度；PD 根据实际状态产生力矩。
3. 仿真更新实际运动和末端位置。
4. 奖励鼓励底盘速度跟踪、末端位置跟踪、步态及平滑安全行为。
5. Critic 预测未来回报，PPO 用采样结果更新策略。
6. 历史编码器与解码器同时学习速度、末端位置和力标签；此阶段注入力标签为零。

末端的统一奖励目标为：

\[
x_{target}^{world}=x_{cmd}^{world}+\frac{F_{ext}^{world}+R_{yaw}F_{cmd}^{local}}{K_{ee}}
\]

此时 `F_ext=0, F_cmd=0`，因此退化为 `x_target=x_cmd`。注意这里的 `K_ee` 是末端任务的虚拟刚度参数，不是关节 PD 增益。

出处：[统一末端奖励](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1891)、[监督标签](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:405)、[奖励项配置](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:271)。

### 5.3 “没有力课程”不等于环境没有扰动

地形、观测噪声和物理参数随机化仍然存在。周期性的 `_push_robots()` 也可直接扰动底盘水平速度。这与 `_push_robot_base()` 的显式力调度是不同机制，不能混为一谈。

出处：[底盘速度 push](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1025)、[周期性调用](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:829)、[随机化配置](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:70)。

## 6. 阶段二：同一策略加入末端力命令和外力扰动

### 6.1 没有重新建网或换算法

达到阈值后，同一个 runner 继续采样并更新；改变的是环境开始调用 `_push_gripper()`，不是重新创建 Actor、Critic，也不是切换另一套 PPO。

出处：[runner 持续循环](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py:115)、[环境阶段判断](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:125)。

### 6.2 两股力必须分别理解

| 力 | 怎么产生 | 到哪里去 | Actor 能否直接看到 |
|---|---|---|---|
| `F_cmd` 期望末端力 | 随机选择环境、方向大小与时间计划 | 写入 `commands[:,9:12]`，参与统一奖励目标 | 能，通过命令槽位；会乘 0.01 缩放 |
| `F_ext` 注入末端外力 | 独立的选择与随机时间计划 | 写入 `forces[:,gripper_idx,:3]`，由仿真接口施加 | 不能作为真值直接看到；由历史状态适应 |

每个轴的目标值在 `[-60,60] N` 范围采样，经过上升、保持、下降、归零。不是所有环境始终都承受满幅力。

出处：[力配置](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:140)、[command 调度](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1037)、[external 调度](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1113)、[施加到仿真](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:130)。

### 6.3 怎么训练出力适应

仍然是“Actor 出关节动作 → PD → 仿真 → 奖励 → PPO 更新”。变化在于位置目标经过力偏置，估计器也开始获得非零末端注入力标签。

例如在一个世界坐标方向上，假设 yaw 旋转可忽略、`x_cmd=0.60 m`、`F_ext=20 N`、`F_cmd=10 N`、`K_ee=200 N/m`：

```text
位置偏置 = (20 + 10) / 200 = 0.15 m
奖励使用的目标 = 0.60 + 0.15 = 0.75 m
```

这是解释源码公式的示例，不是训练测量值。Actor 仍然输出 17 个关节动作，没有改成直接输出 3 维末端力。当前默认奖励通过统一位置偏置目标组织学习，不能把它简化成“直接监督 Actor 输出正确的力值”。

出处：[末端刚度配置](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:148)、[统一奖励实际公式](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1891)、[动作维度](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:105)。

### 6.4 底盘力课程的公开代码边界

虽然配置里有底盘力参数，`step()` 中的 `_push_robot_base()` 调用却被注释。因此默认执行路径是：

```text
末端期望力课程：启用
末端注入外力课程：启用
底盘周期速度 push：启用
底盘显式力命令/外力课程：没有被调用
```

相应的底盘显式注入力标签默认保持零。不能因为估计器有 3 维底盘力输出，就宣称公开任务已经充分训练了这 3 维的真实非零力估计。

出处：[被注释的调用](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:129)、[力缓冲初始化](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1544)、[力标签来源](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:405)。

## 7. 两阶段对照

| 项目 | 力课程前 | 力课程后 |
|---|---|---|
| 网络与动作维度 | 同一套网络，17 维动作 | 继续使用并更新，仍为 17 维 |
| 底盘速度与末端位置命令 | 有 | 有 |
| 末端期望力命令 | 0 | 随机调度，可非零 |
| 末端注入外力 | 0 | 独立随机调度，可非零 |
| 统一末端目标 | 原位置目标 | 原位置目标加力/刚度偏置 |
| PPO 与辅助估计更新 | 已在进行 | 持续进行 |
| Critic 的用途 | 估计当前策略下未来回报 | 用途相同，但环境分布改变 |
| 默认关节角度 | 固定配置 | 仍固定配置 |
| 底盘显式力课程 | 未调用 | 仍未调用 |

这一对照来自上面第 4—6 节的配置与实际调用路径；三个学习模块的具体机制分别见文首的独立文件。
