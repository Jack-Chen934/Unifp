# 当前 V3 的两阶段训练：A位置复现与B受力复现

本文把 [04 · 官方两阶段详细流程](/home/sorry/python_ws/interactive/unifp_reproduction/introduction/04-two-stage-training-details-zh-CN.md) 对应到当前轮式机器人实现。核对日期：2026-11-08。这里说明代码和已确认训练口径，不作为实时训练进度或验收成绩报告。当前结果以run记录和冻结评估为准。

本项目的阶段关系为：**复现第一部分A（8000轮）→复现第二部分B（新增32000轮）→复现验收**。轮式机器人特有创新属于后续独立工作；不能把B受力复现叫作创新阶段。

## 1. A和B共同使用的训练骨架

两部分都沿用“历史编码器＋联合Actor＋特权Critic＋PPO主更新＋状态估计辅助更新”。A到B不改变网络shape、不新增一个力动作头，也不另起一个只控制机械臂的策略。

| 数据或模块 | 官方B2＋Z1 | 当前V3，A／B共用 |
|---|---|---|
| 机器人动作 | 17维；腿12＋臂前5 | 14维；转向4＋轮驱4＋臂6 |
| Actor单帧 | 73维 | 58维 |
| Actor历史 | 32帧、2336维 | 32帧、1856维 |
| latent | 64维 | 64维 |
| Actor动作头输入 | 73＋64＝137维 | 58＋64＝122维 |
| 辅助输出／标签 | 12维，四组各3维 | 12维，四组物理含义保留 |
| Critic输入 | 149×3＝447维 | 98×3＝294维 |
| 编码器／Actor／Critic隐藏层 | 512、256、128 | 保留 |
| 解码器隐藏层 | 128、64 | 保留 |
| 每轮rollout | 每环境24策略步 | 保留 |
| PPO更新 | 5 epoch×4 minibatch | 保留，20次主更新＋20次辅助更新 |
| 策略步长 | 0.02s | 保留 |
| 物理反馈步长 | 0.005s×4 | 0.0025s×8，反馈更细 |
| 策略std／熵系数 | 可学习std；有效配置0.01 | 正式基线保留；独立零熵诊断不等于已修改正式基线 |

正式训练按4096并行环境配置；V3配置类中默认32环境用于较小运行，不能据此认定正式训练只有32环境。训练身份中的配置快照才是该run的实际依据。

代码：[V3配置](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/envs/v3/v3_pos_force_config.py:10)、[网络](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_actor_critic.py:36)、[PPO有效配置及优化器](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_ppo.py:15)。

## 2. A阶段给什么任务，目标怎样变化

### 2.1 底盘和机械臂仍然同时学习

底盘命令保持 `vx∈[-0.6,0.6]m/s`、`vy∈[-0.4,0.4]m/s`、`yaw_rate∈[-0.6,0.6]rad/s`，每5秒重采样。V3的A阶段额外整组置零概率为0，不沿用官方30%的额外置零；当前采样函数也没有官方那段按小速度阈值将整行清零的处理。S0评估中的零速度是评估协议显式设置的，不能与随机训练命令分布混同。B阶段会另行设置30%的整组置零概率，见第5节。

末端目标仍是yaw任务坐标中的球坐标 `(半径,仰角,方位角)`，但根据V3工作空间改为：

| 参数 | V3范围 | 相比官方的变化 |
|---|---|---|
| 半径 | 0.45—0.70m | 官方0.35—0.95m，收缩工作空间 |
| 仰角 | 0—60° | 官方−72—72°，去掉下探区域 |
| 方位角 | −60—60° | 官方−108—108°，收缩侧向范围 |
| 移动时间 | 1—3s | 保留范围；V3每段重新采样 |
| 保持时间 | 0.5—2s | 保留范围；V3每段重新采样 |

reset时读取**实际初始TCP位置**作为首段起点，再采样目标端点；后续从上一目标端点开始新段。这样避免把固定球坐标初始点与V3实际初始姿态误认为同一点。方位角按最短角差插值。目标路径取10个点检查有限值、正半径及TCP离地至少0.05m；这不是全臂碰撞和全轨迹动态可达性保证。

目标当前值写到 `commands[:,3:6]`，姿态槽位 `6:9` 为零，力槽位 `9:15` 在A中强制为零。没有把整条未来轨迹、末端真实位置或末端位置误差直接加到Actor输入里。

代码：[目标与速度范围](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/envs/v3/v3_pos_force_config.py:97)、[reset衔接实际TCP](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:1275)、[速度采样](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:1346)、[目标段生成与检查](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:1374)、[逐步推进](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:1497)。

### 2.2 V3的世界目标参考中心

训练中，目标的水平中心随底盘平移与yaw转动；高度使用环境原点、名义底盘高度0.32m和机械臂安装高度0.09535m。安装点水平偏移为0.174m：

$$
\begin{aligned}
x_g&=b_x+0.174\cos\psi+r\cos p\cos(y+\psi),\\
y_g&=b_y+0.174\sin\psi+r\cos p\sin(y+\psi),\\
z_g&=z_{origin}+0.32+0.09535+r\sin p.
\end{aligned}
$$

因此目标高度不会因为底盘瞬间颠簸就同幅漂移。这个参考中心是V3几何适配项，不能直接使用官方的0.2m水平偏移和0.8m固定高度。

举例：底盘水平位置为零、yaw为0、`r=0.6m,p=30°,y=0`，当前目标约为 `(0.6936,0,0.7154)m`。目标从前一端点逐渐插值到这里，机器人须在底盘执行速度命令的同时跟踪它。

代码：[任务坐标常量](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:21)、[世界目标转换](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:978)。

## 3. A／B共用的观测和动作接口

### 3.1 58维观测为什么比官方73维少15维

| V3单帧切片 | 内容 | 维数 | 与官方相比 |
|---|---|---:|---|
| `0:2` | roll、pitch | 2 | 保留 |
| `2:5` | 底盘本体角速度 | 3 | 保留 |
| `5:15` | 转向4＋机械臂6的位置偏差 | 10 | 原17维，少7维 |
| `15:29` | 全部14个DOF角速度 | 14 | 原17维，少3维 |
| `29:43` | 最近执行的14维动作 | 14 | 原17维，少3维 |
| 无对应槽位 | 四足步态sin／cos | 0 | 删除2维 |
| `43:58` | 原结构的15维任务命令 | 15 | 保留槽位 |

合计少 `7＋3＋3＋2＝15` 维。轮驱是连续转动关节，训练需要其转速，但不把累计转角相对固定零位的偏差当成任务状态。因此V3角速度比位置偏差多4维，这不是遗漏传感器或维度拼错。

12维辅助标签仍为实际底盘速度3维、实际TCP球坐标3维、末端注入力3维、底盘注入力3维；A中后6维为零，B中末端注入力可非零，底盘显式注入力仍为零。

Critic单帧98维按代码顺序为：标签12＋质量／质心差量5＋摩擦1＋执行系数14＋轮接触4＋重力投影3＋角速度3＋位置偏差10＋角速度14＋上一动作14＋命令15＋补偿目标3。5维物性差量分别是底盘质量1、局部质心偏移3、工具质量1。只有Critic和监督分支获得这些特权真值，Actor依然只读本体历史和命令。

代码：[观测拼接、缩放和标签](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:1649)、[历史发布](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:1708)。

### 3.2 14维动作如何执行

| 动作切片 | 对象 | 控制目标 | 底层输出 |
|---|---|---|---|
| `0:4` | 四个转向关节 | 默认角度＋`0.25×缩放后动作`，rad | 位置PD力矩 |
| `4:8` | 四个驱动轮 | `5×缩放后动作`，rad/s，限制在±20rad/s | 速度误差反馈力矩 |
| `8:14` | 六个机械臂关节 | 默认角度＋`0.25×缩放后动作`，rad | 位置PD力矩 |

执行系数作用在动作残差或轮速参考上；位置目标还按URDF关节限位裁剪，最终力矩按effort限幅。动作输出没有直接等于末端速度或末端力，也没有插入解析逆运动学替Actor决定机械臂姿态。

同一动作执行8个2.5ms子步，每个子步重算反馈力矩。每轮24策略步对应192个子步；这与32帧历史是三个不同维度。

代码：[动作到位置／轮速与力矩](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:1542)、[子步执行](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:668)。

## 4. A8000怎样进入B：继承什么，重置什么

V3用显式 `reproduction_part=A/B` 区分两部分，不依赖某个仿真计数在进程重启后再次越过阈值。B必须从明确指定、累计完成8000轮的A边界检查点进入，记录父文件SHA256和血缘；同阶段恢复也使用明确检查点，不自动选择“最新文件”。

| 内容 | A→B处理 |
|---|---|
| Actor、Critic、编码器、解码器、std | 全部继承 |
| 主Adam、辅助Adam动量等状态 | 全部继承 |
| 实际主学习率及辅助优化器状态 | 恢复实际值，不把初始lr重新当成当前lr |
| 累计训练轮次／策略步／transition | 从A8000接着累计 |
| 已存档物性 | 按存档恢复；创建actor阶段施加需提前恢复的物性 |
| 随机数状态 | 恢复后开始新episode |
| 未结束episode、历史帧、PhysX接触缓存 | 不连续恢复；重新reset开始新episode |
| 力事件 | B分配并启用两路调度，reset时从零输出开始 |
| 底盘速度任务分布 | A额外置零概率0；B改为0.3，保留70%的非额外置零采样 |
| 输出目录 | B使用独立run，保留A及父检查点 |

这表示学习状态连续，不表示物理世界被逐子步无缝快照恢复。A／B的schema相同，所以没有因为进入受力部分就需要重建维度不同的网络。

预算与数据量按4096环境计算：

| 边界 | 累计训练轮数 | 累计每环境策略步 | 累计transition |
|---|---:|---:|---:|
| A结束 | 8000 | 192000 | 786432000 |
| B新增4000轮的观察点 | 12000 | 288000 | 1179648000 |
| B结束 | 40000 | 960000 | 3932160000 |

因此“B训练32000轮”指从累计8000到40000；“累计12000轮”只完成了B新增4000轮。累计计数不包含冻结评估的零更新采样。

代码：[A→B入口及边界校验](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/scripts/train_v3.py:41)、[A／B有效配置解析](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/utils/v3_config.py:34)、[完整学习状态恢复](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_on_policy_runner.py:374)、[创建时物性恢复接口](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:156)。

## 5. B每个策略步具体增加了什么

除力事件外，B还将底盘速度命令的额外整组置零概率设为0.3，用来混合静止与运动任务；这不是关闭底盘运动，也不是官方未生效的 `zero_vel_cmd_prob_after_force=0.8`。该值由 `resolve_v3_configs()` 覆盖，因此读取基础配置中的0并不能代表B运行值。

### 5.1 两路独立事件

命令力与注入外力分别采样，每轴峰值±60N、启用概率0.8、间隔3.5—9秒、上升时长候选1—3秒、保持1秒；上升时长受间隔限制，随后对称下降，剩余时间为零。参数范围沿用官方意图，V3明确在策略步边界推进一次事件。

命令峰值定义在yaw坐标，外力峰值定义在世界坐标。外力通过PhysX施加在TCP所属工具刚体的质心，每个子步维持本策略区间选定的外力，不把它当成任意TCP点的六维力／力矩加载。命令力只写命令，不直接加载到PhysX。

### 5.2 当前实际外力与下一步待施加外力必须分开

`pending_ee_force_world`是下一执行区间待用的外力，`applied_ee_force_world`是当前已执行区间实际采用的外力。它们分开是为了防止奖励或估计标签误用尚未施加的新一帧外力。

```text
动作前：Actor读当前历史和已发布命令
    ↓
冻结本区间外力：applied ← pending
    ↓
执行8个子步，施加同一外力并重算PD
    ↓
刷新实际状态，将applied按当前yaw转换成外力标签
    ↓
按刚结束区间的命令／实际注入力结算奖励和episode指标
    ↓
推进下一位置目标与下一力事件，得到新的pending
    ↓
局部reset结束的环境，发布下一观测、特权观测、标签
```

rollout保存的 `H_t/y_t` 来自动作前同一个观测包，奖励来自该动作执行后的区间结算。以上时序解释数据来源，不把标签错配成未来力，也不把接触真值加入人工注入力。

代码：[事件状态与梯形波](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:580)、[applied标签转换](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:654)、[完整step顺序](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:668)。

### 5.3 奖励保留官方位置—力关系

V3把实际世界外力转成当前yaw力标签，和yaw命令力相加，再统一转回世界坐标：

$$
x_*^W=x_g^W+\frac{R_z(\psi)(F_{ext}^Y+F_{cmd}^Y)}{200}.
$$

这与官方“世界外力＋旋转后的命令力”在坐标转换一致时是同一个关系。A中两路力严格为零，直接退化为原始几何目标；B中则允许力偏置。关键原始代码为：

```python
# V3 _compensated_target_world() 中的连续片段
force_yaw = self.external_ee_force_y + self.commands[:, 9:12]
yaw = self.rpy[:, 2]
force_world = torch.stack((
    torch.cos(yaw) * force_yaw[:, 0] - torch.sin(yaw) * force_yaw[:, 1],
    torch.sin(yaw) * force_yaw[:, 0] + torch.cos(yaw) * force_yaw[:, 1],
    force_yaw[:, 2],
), dim=1)
return self._geometric_target_world() + force_world / self.gripper_force_kps
```

出处：[V3补偿目标](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:1010)。末端奖励依然是补偿误差L1的指数；报告同时保存几何与补偿L1／L2，避免把合理柔顺位移误认为跟踪失效。

底盘显式力通道仍为零，底盘速度push另外保留。B不是把所有随机扰动都喂给力估计监督，更没有把底盘push速度换算成力标签。

## 6. 学习更新哪些部分保留，哪些实现细节不同

PPO公式、GAE折扣、5×4更新、辅助四组权重、可学习高斯std、KL自适应学习率等骨架保留。V3显式限定参数组：主Adam更新Actor、Critic、编码器、std；辅助Adam更新编码器、解码器。它把官方依靠前向梯度路径形成的有效分工明确写出来，并在两次反传间清理梯度。

V3的辅助预测在主更新后重新前向计算；日志按真实更新次数取均值，没有沿用官方辅助损失日志额外除64的归一化。因此比较官方和V3的辅助loss数值前，须统一缩放、权重和日志归一化，不能直接比较图上谁更小。

V3还显式区分物理终止和timeout，并保存终止时的特权观测供超时bootstrap使用。普通物理失败只reset对应环境；不因某个样本失败就停止全部训练。数值非有限或接口不一致是另外一类工程错误。

代码：[参数组与双Adam](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_ppo.py:53)、[主更新及重新前向辅助更新](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_ppo.py:193)、[终止与奖励结算](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:714)。

## 7. 为什么要分零力、受力跟踪和真实接触三类评估

这些是**本项目补充的冻结评估协议**，不是声称官方公开训练入口原样运行了这些同名评估。评估使用独立环境、固定协议、Actor均值动作、`policy_updates=0`，不拿评估样本继续训练。失败样本保留在分母里。

### 7.1 零力回归：检查A能力是否保留

| 场景 | 底盘命令 | 速度push | 两路末端力 | 主要用途 |
|---|---|---|---|---|
| S0 | 零速度 | 关闭 | 都为零 | 静态几何跟踪 |
| S1 | 运动命令 | 关闭 | 都为零 | 底盘运动下几何跟踪 |
| S2 | 运动命令 | 保留 | 都为零 | 运动＋速度扰动下几何跟踪及push恢复 |

默认每组32环境、每环境3个episode，最长20秒。零力回归不是只检查“力估计输出等于0”，还要看TCP、底盘、失败率和六维状态估计。

代码：[零力评估manifest](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/configs/v3_a_initial_v1.json)、[评估入口](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/scripts/evaluate_v3.py)。

### 7.2 受力跟踪：区分命令响应与未知外力响应

| 模式 | 人工外力 | 期望力命令，转到同一世界坐标后比较 | 理想补偿目标 | 主要检查 |
|---|---|---|---|---|
| M | `F` | `−F` | 几何目标 | 受匹配反作用力时能否维持位置 |
| E | `F` | 0 | 几何目标＋`F/K` | 未知外力下柔顺响应 |
| C | 0 | `F` | 几何目标＋`F/K` | 已知力命令能否改变动作 |
| R0／R2 | 两路独立随机 | 两路独立随机 | 按实时合成力计算 | 接近训练分布的综合表现 |

固定方向工况分六个世界方向、半幅／全幅及静态／运动；固定事件在1秒和9秒开始，每次2秒上升、1秒保持、2秒下降。R0为静态且无push，R2为底盘运动并有push，使用随机力事件。固定脚本和随机训练事件不能按同一条波形解释。

R2能综合检查补偿跟踪、底盘控制、估力与失败率，但不能代替所有M／E／C分项。当前诊断中只跑R2时，应明确它是子集结果。

代码：[受力评估manifest及观察节点](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/configs/v3_b_initial_v1.json)、[M／E／C／R事件实现](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/envs/v3/v3_force_evaluation.py:76)。

### 7.3 墙面接触与0N对照：检查真实接触能力

墙面评估建立真实碰撞墙，使用绝对世界位置轨迹；它不像随机训练目标那样随底盘平移。0—3秒接近参考点，5—7秒力命令上升，7—11秒保持，11—13秒下降，15—18秒平滑后退5cm。半幅／全幅期望法向力为10／20N，不能把它误写成训练的30／60N。

真实墙面力来自每个策略区间8个物理子步的接触测量，机器人受到的反作用力与作用在墙上的力方向相反。测量、接触有效性和混杂接触分类只用于诊断与评估，不作为Actor输入，也不加入人工注入力标签；评估没有暗中插入力传感反馈控制器。

独立0N对照使用零力命令，检查纯几何接近／离开是否也产生接触、残余接触或估力偏置。它与S0／S1／S2不同：前者有墙、固定世界轨迹；后者是零注入力条件下的常规任务回归。

默认接触组32环境×3episode，0N对照32环境×1episode。报告须区分有效测量、混杂接触、物理失败和成功率；归档生成成功不等于任务指标通过。

代码：[接触几何和阈值manifest](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/configs/v3_b_contact_v1.json)、[接近／施力／退出脚本](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/envs/v3/v3_contact_evaluation.py:270)、[子步接触采样](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/envs/v3/v3_contact_evaluation.py:342)、[接触指标汇总](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_evaluation.py:95)、[接触评估入口](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/scripts/evaluate_v3_contact.py)。

## 8. 看训练记录时应同时对应哪些数据

| 数据 | 正确理解 | 不能据此单独下的结论 |
|---|---|---|
| episode奖励／reward rate | 多个跟踪奖励和惩罚的加权结果；前者还受回合长度影响 | 奖励上升就表示真实接触力达标 |
| 几何TCP L2 | 实际TCP到原始目标的距离，m | 非零外力下必须始终为0 |
| 补偿TCP L2 | 实际TCP到力偏置目标的距离，m | 小误差就表示所有墙面任务成功 |
| force辅助loss | 缩放、分组加权后的MSE | 等于以N为单位的平均绝对误差 |
| 力估计轴向MAE | 反缩放后的估力误差，N | 策略一定使用了该信息产生正确动作 |
| std | 高斯探索幅度，动作单位 | 解码器误差或直接的物理关节噪声幅度 |
| 饱和占比 | 动作经控制转换后达到执行边界的频率 | 仅看平均奖励即可忽略执行限制 |
| 物理失败率 | 该协议下局部终止的比例 | 一次失败就必须停止整个训练 |
| 接触成功率／0N残余 | 真实墙面任务与无力对照表现 | 等价于人工注入外力估计精度 |

正式训练观察安排、各次结果与后续诊断放在 [统一进度记录](/home/sorry/python_ws/interactive/unifp_reproduction/docs/progress/2026-11-06_v3_status.md)；冻结协议在上述manifest中。两组200轮熵系数诊断属于独立开发实验，不能把其9000→9200计数混入原正式run累计轮次，也不能在结果出来前宣称正式训练已采用零熵。

实施与恢复入口：[train_v3.py](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/legged_gym/scripts/train_v3.py:26)；带观察记录的正式B入口：[train_v3_b_observed.py](/home/sorry/python_ws/interactive/unifp_reproduction/UniFP_v3/tools/train_v3_b_observed.py:74)。它们需要明确的检查点、输出目录与运行证据；阅读本文不自动触发训练、恢复或参数变更。
