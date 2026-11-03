# A05 · Actor与辅助状态估计

状态：A05-Q1—Q6已于2026-10-31全部确认，模块设计已收束；下一讨论为A06-Q1。业务代码尚未修改，A06—A11及A实施计划未完成前不编码。

已确认结论总结：Actor历史32×58=1856经1856→512→256→128→64的encoder形成latent，与最新58维拼成122维输入122→512→256→128→14的Actor；decoder为64→128→64→12，四组状态／人工力标签保持A04坐标／缩放。隐藏层ELU、输出纯Linear、从头采用nn.Linear默认初始化，A／B保持同一网络。辅助监督四组MSE的实际系数为[0.04,0.04,1,1]，A的两组零力标签仍训练；完整标签／预测流程见Q2。训练使用14维直接std对角高斯、初值1.0，评估用均值，非法std失败退出；可选state_estimates输出同设备detach预测张量。每minibatch先主Adam、再辅助Adam一步，辅助学习率1e-5，encoder共享，主优化器仅encoder／Actor／Critic／std，辅助仅encoder／decoder，阶段间全模型set_to_none=True；修正日志额外除64，不新增辅助梯度裁剪。两次更新均由PPO循环调度，选择官方机制不代表已证明优于合并损失。A06定义Critic，A07确认主PPO数值与完整恢复，A10—A11细化评估／测试；以上是已确认、待实现设计，不是物理验证或复现通过。

## 本部分职责与前置依赖

完成能实际训练和推理的统一Actor及状态估计；同时确定B续训需要的辅助输出和网络shape。

前置依赖：A04已确认的数据契约（Actor1856／标签12／动作14）；共同基线00—03。 同部分按04→11讨论，11最后收束测试；B依赖A，A不依赖B实现完成。

共同的A／B范围依据见[总体计划](../../development-plan.md)，源码与原提案见[公共05](../05-Actor与辅助状态估计.md)。当前官方调用链见[任务入口](../../README.md#当前官方链路)。本模块逐题确认记录见下文，后续模块不因本模块收束而自动获批。

## 按依赖排列的讨论项

| 问题 | 本次需要确定的内容 | 状态 |
|---|---|---|
| Q1 | 当前帧与历史输入字段、encoder／actor／decoder边界及逐层shape。 | 已确认；网络宽度亦已由Q4确认 |
| Q2 | 线速度、TCP状态和力相关监督标签的保留范围、物理可观测性与缩放。 | 已确认；完整流程已记录 |
| Q3 | A中全零力监督的处理与分组权重。 | 已确认；保留官方分组MSE |
| Q4 | 潜变量、网络宽度、激活、历史长度及初始化；B需要的结构在A确定。 | 已确认 |
| Q5 | 14维动作均值／std、采样／确定性推理和导出边界。 | 已确认；采样解释已补充 |
| Q6 | PPO与辅助损失对共享encoder的梯度路径、参数归属及诊断。 | 已确认；模块收束 |

具体讨论时逐题给出推荐答案，在本文件记录问题、推荐依据和用户确认结论；未展示的字段、数值或接口不自动获批。完成本议题后，在开头补齐覆盖全部结论的总结。

## Q1 · 历史编码、策略输入与估计器边界

**问题：历史、当前观测和辅助预测如何连接，避免把训练真值直接泄漏给策略？**

**推荐答复：**将A04的32帧Actor历史送入encoder；把encoder latent与最新58维观测拼接后送入14维Actor；decoder只从latent预测A04的12维辅助标签。decoder输出、`obs_pred`真值和Critic特权观测都不直接进入Actor。latent宽度`Z`及各隐藏层宽度留Q4决定。

**用户答复（2026-10-31）：**“同意”。

**已确认结论：**

```text
obs[N, 1856] → encoder → z[N, Z]
                              ├→ concat(obs[:, -58:], z) [N, 58+Z]
                              │   → actor → action_mean[N, 14]
                              └→ decoder → prediction[N, 12]
```

Actor仅接收历史编码及当前一帧Actor观测，不接收Critic特权输入、监督真值或decoder的物理预测值。encoder输出的latent同时作为Actor与decoder输入；其各维不预先赋予单独物理含义。随机策略的std参数和训练／推理差异留Q5决定。

| 对照对象 | 原有结构 | 本工作决定 | 差异说明 |
|---|---|---|---|
| 论文§3.2 | 文字说明历史经encoder产生latent，并送给state estimator和actor | 保留共享latent供策略与估计使用 | 论文该段未明确写Actor还接收最新原始帧 |
| 固定官方代码 | encoder吃展平历史；Actor输入是最新单帧与latent的拼接；decoder从latent预测标签 | 保留这一可执行代码边界 | V3维度按A04改为历史1856、当前帧58、动作14；预测标签维数12保持 |
| 信息权限 | 普通Actor使用观测与latent；Critic另用privileged输入；decoder预测不拼入Actor输入 | 保持训练真值与特权状态不进入Actor | 维持部署时Actor只需历史Actor观测的边界 |

本题讨论时未提前锁定latent宽度；后续Q4已确认V3取64维，并确定encoder／decoder／Actor隐藏宽度和激活函数。官方代码由`int(num_obs / num_single_obs) * 2`导出64维。依据见[官方ActorCritic](../../../../UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py#L43)及[A04论文／源码差异记录](../../comparison/A04-论文与官方源码差异.md)。

## Q2 · 辅助输出的标签、接收者与完整生命周期

**问题：12维辅助输出是什么、由谁监督、提供给谁，论文净力目标与当前实现之间的边界是什么？**

**推荐答复：**保留A04的四组三维预测、坐标／缩放与人工注入力标签。训练时预测送辅助监督损失，推理时可导出供诊断／评估或外部程序使用；预测和监督真值都不直接拼入Actor。论文用随机净力代理训练力估计、并在真实任务中使用估计力，不能据此断言其估计器不具备接触力用途；但保留同类监督通道也不证明V3已经具备真实接触反力估计能力，须在B真实接触场景中单独验证。

**用户答复（2026-10-31）：**“同意，但请记录完整流程”。

**已确认结论：**四组输出保持12维；监督只取已定义的仿真状态和人工注入力，接触反力不直接并入力标签。预测的训练、推理与下游用途按下述流程区分；任何改变标签来源或在线反馈路径的方案都须重新讨论，而不是沿用字段名静默替换。

### 1. 数据身份、物理含义与缩放

设N为并行环境数，T为每轮rollout步数（24），M为minibatch样本数，Z为Q4已确认的latent宽度64。以下浮点张量均为同设备float32，预测与真值必须采用同一字段顺序和缩放。

| 数据 | 来源／含义 | shape与接收者 |
|---|---|---|
| `obs` | A04发布的含噪Actor历史，旧到新，末58维为最新帧 | [N,1856]；encoder及Actor当前帧支路 |
| `z` | encoder从历史计算的学习表示，无逐维预设物理含义 | [N,Z]；Actor和decoder |
| `privileged_obs` | A06定义的训练侧特权历史 | [N,3c]；Critic，c待A06，不进入Actor／decoder输入 |
| 环境`obs_pred` | 边界时刻的监督真值；沿官方键名，并不是网络预测 | [N,12]；Runner、storage、辅助损失 |
| `prediction` | decoder从z计算的四组预测，使用此文档概念名区分真值，不提前锁定新增代码API | [N,12]；辅助损失、推理调用方及评估 |
| `action_mean`／采样动作 | Actor输出；采样和导出细节留Q5 | [N,14]；环境动作执行路径 |

| 输出／标签切片 | 物理量与坐标 | 原始单位 | 训练表示／还原 |
|---|---|---|---|
| [0:3] | 底盘线速度，完整B轴 | m/s | 真值×2；预测÷2还原 |
| [3:6] | TCP相对A04球心、Y轴下的r／仰角／方位角 | m／rad／rad | ×[0.5,1,1.3]；逐维除同一缩放还原 |
| [6:9] | 上一策略区间末次实际人工末端注入力，变换到当前边界Y轴 | N | ×0.01；预测×100还原 |
| [9:12] | 上一策略区间末次实际人工底盘注入力，变换到当前边界Y轴 | N | ×0.01；预测×100还原 |

Critic和标签不加Actor测量噪声；标签不做网络输入式clip，非有限值按A04失败处理。初始／reset帧人工力为零。A中两个人工力标签均零；B基线默认只启用末端人工力，底盘人工力仍零。力命令F_cmd可进入Actor观测，但它是任务参考，不能替代实际施力标签。正常轮地／物体接触不因人工力为零而关闭。

### 2. 训练采样：标签与旧观测配对入库

```text
策略边界t：环境先发布同包数据
  obs_t[N,1856]、privileged_obs_t[N,3c]、label_t[N,12]
  label_t的状态量来自边界t；力来自刚结束区间的末次实际注入力

Runner → PPO.act(obs_t, privileged_obs_t, label_t)
  obs_t → encoder → z_t
  concat(obs_t末58维,z_t) → Actor → action_t[N,14]
  privileged_obs_t → Critic → value_t[N,1]
  暂存obs_t、privileged_obs_t、label_t及动作／分布／价值
  label_t只交给转移存储，不参与Actor或Critic的前向输入

env.step(action_t)
  → 4个5ms物理子步 → reward_t、done_t、infos_t
  → 保存终止快照 → 局部reset → 发布边界t+1观测包

PPO.process_env_step → storage.add_transitions
  同行保存obs_t、label_t、action_t、reward_t、done_t等
  禁止用step后label_(t+1)覆盖旧label_t
  全轮存储obs[T,N,1856]、labels[T,N,12]等
```

A04保证已发布数据不被后续step／reset原地覆盖。局部reset的新初帧必须与新episode零力标签配对，不能把旧episode注入力留给新初帧。终止Critic快照单独走infos，不改变本题监督配对。returns／bootstrap归A07，不在本题更改。

### 3. 训练更新：预测送辅助损失，学习共享表示

```text
storage按同一批索引取样
  → obs_batch[M,1856]、label_batch[M,12]及PPO所需字段

PPO主路径：obs_batch → encoder → z → Actor → 策略损失
          privileged_obs_batch → Critic → 价值损失
辅助路径：obs_batch → encoder → z → decoder → prediction[M,12]
          prediction与无梯度label_batch逐组计算监督损失
          辅助反向传播可达encoder和decoder

后续动作使用更新后的encoder／Actor参数
  → 辅助任务通过改变共享encoder间接影响策略
  → prediction不作为动作输入拼回Actor
```

这里记录计算依赖和接收者，损失权重已由Q3确认，参数集合与更新顺序已由Q6确认；主PPO数值与完整恢复细节仍归A07。官方默认先主更新再辅助更新，两套Adam均注册模型参数；“注册了参数”不等于该参数一定在某条损失图上得到梯度，V3明确参数集合及验证方式见Q6。训练采样阶段无需为了生成动作额外消费decoder预测；官方Runner虽每步另算一次估计，局部变量未传给动作或奖励，这不构成新的反馈路径。

### 4. 推理与评估：动作、预测分别交给各自调用方

```text
传感／环境Actor历史obs[N,1856]
  → encoder → z
      ├→ concat(最新帧,z) → Actor → action_mean → 执行器
      └→ decoder → prediction[N,12]
                       → 推理调用方／日志／可视化
                       → 反缩放、坐标转换后与真值比较，计算物理误差
```

推理获得动作和预测都不依赖监督标签或Critic；训练后的decoder可作为估计器导出。官方`act_student`返回动作均值，将预测写入`policy_info['latents']`；其`get_student_latent`也实际返回12维预测而非64维latent。本工作文档明确区分这两个对象，Q5再确认实际导出接口、设备和字段名，不能只改名字而遗漏调用方。官方播放脚本在`VISUAL_PRED`分支读取末端／底盘力用于显示；其部分速度／位置可视化代码被注释，不能报告成默认已运行的评估功能。

评估记录分别报告速度m/s、TCP位置m和力N的误差，不能把混合缩放空间的训练MSE当作全部物理误差。TCP球坐标预测须按同一球心和Y轴换算后再比较位置；实际误差聚合、时序对齐及阈值由A10收束。A全零力输出只能检查零力偏差，不能证明非零外力或接触反力估计能力。

### 5. 论文中的外部用途与本任务范围

论文§3.2说明估计力可转为特定控制行为的命令；§3.3记录遥操作时的估计接触力，和机器人状态／图像一起供force-aware模仿学习使用。该上层策略预测位置／力命令，再交给底层统一策略，形成另一层调用链：

```text
底层估计器 → 估计力记录／上层行为或示教数据
  → （论文中的模仿学习）状态＋估计力＋图像 → 位置／力命令
  → 底层统一策略 → 关节动作
```

这不是将decoder预测默认拼回基础Actor，也不是本次启动上层模仿学习开发。当前范围保持第一阶段V3仿真复现，B的接触场景负责验证约定能力；不因本题同意新增高层控制器、模仿学习任务或轮式创新。

### 6. 原文／代码对照与边界

论文§3.2明确把反力纳入采样净环境力，训练时不显式建模接触反力。公开代码用人工施力缓存作为监督代理，与这一训练思路相容；因此不能断言“使用人工力训练必然无法估计真实接触力”。另一方面，缓存不是实际物体接触反力真值，V3仍须在真实接触场景以独立测量／仿真参考验证泛化，不能用注入力MSE代替该证据。

论文附录C.3的水平参考与官方yaw对齐球坐标并不天然冲突。V3保留同类表示，具体TCP、球心和工作空间按形态适配；不将笛卡尔与球坐标的表示选择夸大为论文目标本质不同。详细保留／修改见[A05差异记录](../../comparison/A05-论文与官方源码差异.md)，一手事实见[知识库第24节](../../knowledge/research.md#24-a05辅助预测的训练存储与推理接收者)。

## Q3 · A阶段的零力监督与分组权重（已确认）

**问题：A尚无人工外力时，是否保留两组力预测的零标签监督，如何保持与官方及B接续一致？**

**推荐答复：**保留四组监督及官方分组MSE，在A中也训练两组零力标签，不增加零力mask、删除力输出或冻结该输出层。采用官方乘预测／标签的权重[0.2,0.2,1,1]，即四个分组MSE的实际系数为[0.04,0.04,1,1]；四组分别对M×3个元素取均值后相加，不再除以四。A／B保持同一输出和损失定义，B仅随课程得到非零标签；是否改变课程分布由B09确认。

该方案保留公开代码监督目标及其缩放，不提前通过新mask引入算法差异。零力监督可能使A的力输出趋近零，这只是当前任务分布下的训练结果；B非零估力仍须学习与验证，不能据A的小MSE宣称完成受力估计。辅助学习率、更新频率、梯度处理及优化器保存属于Q6／A07；本题不一并批准。以上权重不是论文逐项明确规定的常数，出处为官方`AC_Args`与PPO逐组MSE实现。

**用户答复（2026-10-31）：**“同意”。

**已确认结论：**A保留四组监督，末端／底盘人工力标签均为零，仍参与分组MSE；不加mask、不冻结或删除力输出层。预测／标签乘子[0.2,0.2,1,1]对应损失系数[0.04,0.04,1,1]；每组三维跨minibatch求均值，四项相加。B沿用同一输出和损失定义，新增非零标签依课程处理。主辅优化器边界、辅助学习率1e-5与更新顺序已由Q6确认；主PPO数值和完整恢复归A07。

```text
L_aux = 0.04 × mean((v_hat_scaled - v_scaled)²)
      + 0.04 × mean((tcp_hat_scaled - tcp_scaled)²)
      + 1.00 × mean((force_ee_hat_scaled - force_ee_scaled)²)
      + 1.00 × mean((force_base_hat_scaled - force_base_scaled)²)
```

这里scaled表示Q2已确定的物理量缩放，损失乘子是另一层权重。例如力先乘0.01再计算损失；不把公式中的系数1误解释为直接对以N表示的误差赋权1。官方出处为`actor_critic.py:13–15`、`ppo.py:183–196`；论文仅说明MSE，不逐项规定这些乘子。验证时须用非零、非等值的四组误差独立计算预期标量，检查平方权重、分组均值与四项相加，不能只以A阶段全零样本证明实现正确。

## Q4 · 网络容量、激活与初始化（已确认）

**问题：迁移后是否保持官方可执行基线的网络容量、64维latent和初始化方式，避免在必要输入／输出适配之外引入架构变化？**

**推荐答复：**沿用官方训练任务实际生效的MLP隐藏层、ELU、32帧历史和64维latent，仅按A04修改encoder输入、Actor拼接宽度与动作输出。网络在A启动前建全，B接续不扩维、不增删头，不重新初始化既有网络。

| 模块 | 固定官方任务实际结构 | V3推荐结构 | 保留／修改 |
|---|---|---|---|
| Actor历史 | 32×73=2336 | 32×58=1856 | 历史长度保持；输入形态适配 |
| encoder | 2336→512→256→128→64 | 1856→512→256→128→64 | 仅第一层输入宽度改变 |
| latent | 64=32×2 | 64=32×2 | 保留容量，不按关节数同比缩小 |
| Actor输入 | 最新73＋latent64=137 | 最新58＋latent64=122 | A04形态适配 |
| Actor动作均值头 | 137→512→256→128→17 | 122→512→256→128→14 | 中间隐藏层保持，首末层shape适配 |
| decoder | 64→128→64→12 | 64→128→64→12 | 完整保留 |
| 非线性／输出层 | 隐藏层ELU；各支路末层纯Linear | 同左 | latent末层、预测末层和动作均值末层均不加ELU／tanh |

官方Actor构造函数的默认隐藏层是[256,256,256]，但本任务配置`b2z1_pos_force_config.py:313–318`通过Runner的`policy_cfg`覆盖为[512,256,128]。此处以实际任务调用链为准，不能误用未生效的构造默认。Critic字段、宽度与网络在A06讨论；本题不批准Critic参数。

从头训练使用`nn.Linear`默认的权重／bias初始化，由02统一seed控制并记录依赖版本。官方类虽定义`init_weights`正交初始化函数，但当前构造和Runner链未调用它；不额外启用正交初始化、不把输出层清零，也不加载官方B2或旧V3权重作为新训练初值。复用相同初始化方法不代表不同shape下的随机初值逐项相同；A→B恢复则加载已保存网络，完整学习状态归A07。

保留无RNN的展平历史MLP；不新增LSTM／Transformer、dropout、BatchNorm、LayerNorm或在线归一化。schema装配时校验1856=58×32与64=32×2，避免整数除法悄悄接受不整除的历史宽度。latent和隐藏层取值写入有效配置／模型shape快照，B必须兼容。std的参数化和初值已由Q5确认，主辅参数更新规则已由Q6确认，主PPO数值与完整恢复归A07。

验证接缝：检查线性层权重shape与隐藏层激活位置、输出[N,64]／[N,14]／[N,12]、最新58维拼接位置，以及相同seed／依赖／调用顺序下重新构建模型的初值一致性。独立改动decoder输出不应改变同次Actor前向动作；更改encoder参数可以同时影响两支路。真实学习能力仍须A11／训练评估，网络shape正确不代表复现通过。

**用户答复（2026-10-31）：**“同意”。

**已确认结论：**encoder为1856→512→256→128→64；Actor为122→512→256→128→14；decoder为64→128→64→12，Actor输入122=58+64。隐藏层ELU，各支路末层纯Linear，保留32帧展平MLP与nn.Linear默认初始化，不启用未被调用的正交初始化。A从头训练，B沿用同一网络并恢复参数；Critic由A06决定，std留Q5。

## Q5 · 动作分布、确定性推理与导出边界（已确认）

**问题：如何保留官方策略的采样机制，明确无量纲动作／物理执行边界，并让辅助预测有可追溯的推理输出？**

**推荐答复：**保留无tanh的对角高斯和14个与状态无关、按动作维独立可训练的直接std参数，初始均为1.0；训练采样，评估用动作均值。每次构造分布前要求std有限且严格大于零，非法则报告并停止运行，不偷偷clamp／取绝对值／改为log_std。直接std参数化保留官方行为，显式失败检查为工程差异；如果真实训练发现std失效，先记录证据并重新确认调整。

### 动作与概率语义

```text
obs[N,1856] → encoder → z[N,64]
concat(最新58维,z) → Actor → mu[N,14]
std[14]按环境广播，初值1.0
训练：a_raw = mu + std × epsilon，epsilon各维独立标准高斯
log_prob[N] = sum_14 Normal(mu,std).log_prob(a_raw)
entropy[N] = sum_14 entropy(Normal(mu,std))
评估：a_raw = mu，不采动作噪声
环境：a_exec = clip(a_raw, -03已确认限幅, +03已确认限幅)
      → 03转向／臂位置残差及轮速目标 → 4个物理子步
下一帧动作历史：a_exec，不是a_raw
```

以上a_raw、mu和std均为无量纲；std=1不是1rad或1rad/s，具体物理探索尺度还取决于03动作映射。PPO保存和求概率针对采样的原始a_raw；环境裁剪后的a_exec用于执行及历史，二者不得混为同一个随机变量。该选择继承官方顺序，裁剪会使实际执行分布不同于无界高斯；不新增tanh-squash或分布修正，存储／logp细节须在A07回填并验证。动作限幅值属于03候选及其冻结流程，不因Q5默认照搬官方clip=100或暗改为1。

std必须进入model state_dict，B恢复已学std而不是重置1.0；完整优化器状态由A07处理。std不因每个环境done而reset，MLP的局部历史清理沿A04。mu／预测／采样动作非有限、输入shape／设备／dtype非法时按已确认接口失败语义处理，不能将错误转换成普通episode done。

### 推理接口与接收者

保留`act(observations: torch.Tensor, **kwargs) -> torch.Tensor`的训练采样职责，以及`act_inference(ob: Dict[str, torch.Tensor], policy_info: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor`的动作均值返回协议。推理只读取ob['obs']，不要求Critic或监督真值；执行在eval及无梯度上下文，不更新模型、环境命令或历史。eval本身不消除观测随机性，环境噪声和域随机化是否关闭仍由A10确定。

当调用方传入字典时，清楚命名`policy_info['state_estimates']`，写入同设备float32[N,12]的已detach预测；不强制CPU／NumPy转换，日志调用方按需复制和反缩放。该键明确取代官方误称的`latents`，不保留双键别名；同步更新新V3调用方，B2官方调用方保持原样。不传policy_info时仅返回动作，不强制额外计算decoder。辅助训练仍通过现有`get_student_latent(observations: torch.Tensor) -> torch.Tensor`取得[M,12]预测，以减少无关改名；文档／docstring须明确其返回的是预测而非latent。

本题所称导出首先指从checkpoint构建可调用的仿真推理模型及其估计输出。动作路径仅依赖encoder＋Actor；估计输出还需decoder；Critic、优化器及监督真值不属于推理前向依赖。当前仿真阶段不新增C++／硬件部署、ONNX或独立JIT交付要求；官方PLAY:49–71可选的三个JIT文件记录为来源事实，若后续要求该产物则另行确认调用、schema和一致性验收。用于继续训练的checkpoint仍必须包含Critic／std／完整学习状态，不能以纯推理模型替代A07续训快照。

### 保留／修改与验证

保留：官方对角高斯、直接std、初始1.0、末层无tanh、概率／熵按动作维求和、训练采样／评估均值。修改：动作17→14；增加显式std／有限性失败检查；推理预测字段由误称latents改state_estimates并保留设备张量、移除可变默认字典。后两者是接口／运行工程差异，不能声称只是机器人尺寸适配；它们不改变有效std与相同输入下的高斯定义／动作均值。

验证接缝：固定mu／std／原始动作与独立计算的logp／熵比较；固定随机种子复核采样且不依赖大量统计样本；同一已发布obs重复推理动作一致且无梯度；将std设零／负／NaN时在分布采样前失败；传入／不传入输出字典的动作均值相同；预测[N,12]同设备、四组含义／缩放不变、每次字典独立；原始动作被环境裁剪后仍用原始动作计算PPO概率；checkpoint加载后mu／std／预测一致。实际测试容差归A11，优化器参数覆盖归Q6／A07。

**用户答复（2026-10-31）：**“同意”。用户先要求解释高斯采样，解释后确认Q5方案。

**已确认结论：**14维无tanh对角高斯，直接std每维独立可训练、初始1.0且与状态无关；训练采样／评估均值；有限性与正std检查失败时退出。原始动作交PPO保存并计算logp，限幅动作交环境执行及动作历史；std随模型保存、B恢复。预测以可选policy_info['state_estimates']同设备detach张量输出，移除误称latents与可变默认字典。当前仿真范围不新增部署格式交付。

### 高斯采样解释记录

Actor根据当前观测与latent给出mu，随后每环境、每策略边界为14维分别抽独立epsilon~N(0,1)，按a_raw=mu+std×epsilon生成完整动作。给定观测时各维探索噪声独立，这不妨碍Actor从共享状态输出协调均值。每20ms抽一次并保持4个5ms物理子步；不在各子步重新采动作噪声。

教学例mu=0.4、std=0.3时，epsilon为0／0.2／-0.5／1分别得到0.4／0.46／0.25／0.7；这只是公式示例，不是项目std或执行器参数。高斯无界，约68%的单维样本落在mu±std；原始动作和std都是无量纲，经03尺度／限幅后才有物理控制含义。

PPO用固定的已采样动作和优势更新其概率密度，正优势倾向于增加、负优势倾向于降低类似状态下该动作的密度；不是直接复制最优样本。14维联合密度相乘，对数密度相加，连续动作某一点的概率与概率密度须区分。采样本身不承担穿过环境的可微控制梯度；更新依赖新旧策略logp与优势，A07再确认公式和rollout处理。std由训练调整，不保证单调下降。

教学例原始动作1.3、假设限幅1时，PPO保存1.3，环境执行／历史写1.0；限幅1不是本项目确认数值。正式评估用均值只消除动作采样随机性，不自动关闭观测噪声、目标随机性和其他环境随机源，A10单独收束。

## Q6 · 梯度路径、优化器参数归属与诊断（已确认）

**问题：如何让策略学习与辅助估计共同训练encoder，同时防止无关参数被残留梯度更新，并明确相对官方代码的改动？**

**推荐答复：**保留两套Adam、每个PPO minibatch先主更新再辅助更新一次；共享encoder在两阶段都可学习，不detach encoder latent、不冻结encoder、不将两个损失改成一次合并更新。辅助学习率沿官方1e-5，且不随主PPO的KL自适应学习率自动改变。主PPO学习率、epoch／minibatch数量及GAE等由A07确认，Critic结构归A06。

### 为什么分两次更新，而不是只保留PPO或合并损失

用户追问“两次更新的区别、辅助更新属于哪一板块”以及“为什么不由PPO统领”。官方两次优化本来都位于`PPO.update()`，由同一训练循环调度；区别是优化目标和优化器状态，不是另建独立控制器或训练阶段。辅助网络归A05，优化调用／恢复由A07承接。

只保留PPO主损失时，策略可通过奖励学出有效控制，但没有直接要求decoder准确预测四组状态／力；按当前网络，decoder不在主损失前向路径上，其预测头也就不能靠该损失直接训练。辅助MSE给encoder／decoder提供已知仿真标签的直接监督，保留论文要求的显式状态／力估计输出。有效控制不自动等于预测数值准确，反之预测准确也不保证控制达标。

可以把主损失和lambda乘辅助损失相加，用一次backward和一套Adam更新；这种联合多任务训练是可行替代，不存在“PPO算法必须采用两次更新”的约束。分开更新使辅助学习率、频率和Adam动量可独立于主PPO设置。官方辅助学习率1e-5的保留属于代码基线选择，不证明其对V3最优，也不能仅凭名义学习率大小保证共享encoder的实际变化更小。

两种做法并不等价：官方先主步、再用主步后的encoder重算辅助预测，两套Adam对共享encoder各自维护历史；合并方案通常在同一参数状态求组合梯度，改变梯度缩放、动量和裁剪耦合关系。分别更新也不消除目标冲突：辅助更新encoder后仍会改变策略，且该变化不由PPO主步clip自动约束。当前没有本项目“两步优于合并”的对照证据，不能以稳定性或收敛性优势作已验证结论。本阶段推荐两步主要为了保持官方训练机制、减少未验证的算法变更；若未来需要改为合并更新，须另行确认设计和对照。

以上记录用户在Q6确认前关于原理的追问与答复；解释后用户已同意本题，结论见题末。

### 参数与损失依赖

| 参数模块 | PPO主损失路径 | 辅助MSE路径 | 推荐注册 |
|---|---|---|---|
| encoder | 策略surrogate经Actor回到encoder | 四组估计经decoder回到encoder | 两套Adam均注册，同参数各有独立Adam动量 |
| Actor均值头 | 策略surrogate | 无路径 | 仅主Adam |
| std[14] | 策略surrogate及熵项 | 无路径 | 仅主Adam |
| Critic | 价值损失 | 无路径 | 仅主Adam |
| decoder | 无路径 | 四组估计MSE | 仅辅助Adam |

熵在本方案为状态无关std的函数，不通过均值去训练encoder；独立Critic也不通过价值损失直接训练encoder。表中路径是可达性，不保证任一minibatch每个元素梯度都非零。主Adam的参数为encoder＋Actor＋Critic＋std；辅助Adam仅encoder＋decoder。

### 每个minibatch的更新顺序

```text
取obs_batch[M,1856]、标签[M,12]、Critic输入及PPO字段
  → 全模型zero_grad(set_to_none=True)
  → 重新计算当前策略分布／logp、价值和主损失
  → 主损失backward
  → 检查主阶段梯度有限，按主PPO max_grad_norm裁剪
  → 主Adam.step（decoder不更新）

  → 全模型zero_grad(set_to_none=True)
  → 用主更新后的encoder重新前向：历史→z→decoder→预测
  → 与detach的同刻标签计算Q3四组MSE
  → 辅助损失backward
  → 检查辅助梯度有限
  → 辅助Adam.step（只更新encoder＋decoder）
  → 清除残余梯度，进入下一个minibatch
```

沿官方不新增辅助阶段梯度裁剪、梯度投影或编码器目标网络。主阶段裁剪范数只覆盖主损失参与参数，不把上一阶段decoder残留梯度混进去；主max_grad_norm具体数值在A07确认。辅助更新后encoder已经变化，因此同一观测下Actor动作也可能变化，即使Actor头没有直接更新；“辅助只更新估计头，策略不会变化”的说法不成立。保持官方这种间接作用机制，但不承诺策略分布变化只来自PPO主步，亦不提前添加辅助步KL约束。

### 与官方代码相比的工程变化

官方PPO:53–55两套Adam都注册全模型参数，而PPO:167,198只调用各自optimizer.zero_grad且未显式传set_to_none。不同PyTorch版本的zero_grad默认行为会影响未参与当前图参数的grad是None还是残留零张量；Adam遇到零梯度也可能因历史动量更新参数。不能把“该头不在当前图里”直接当成“它必定不更新”，也不能无版本证据断言官方一定存在该问题。

推荐用明确参数集合＋全模型set_to_none=True清理固定目标更新范围。保留共享encoder的双优化器动量、两步顺序及辅助频率；参数注册／清梯度规则是明确的工程修改，不是论文原文要求，也不宣称与所有官方PyTorch版本逐位等价。两套优化器及实际学习率／状态须在A07完整保存恢复，不能只存主Adam；B不得重置辅助Adam或encoder动量。

### 诊断、失败与验证接缝

记录每组MSE、按Q3相加的总辅助损失，以及encoder／Actor／Critic／decoder各阶段的梯度范数、更新前后参数变化和std范围；分布式／逐参数全量trace不是本题要求，按minibatch汇总并在A10确定日志字段。全零力标签仍参与损失，所有目标无梯度。

官方PPO:207–209将已经取均值的辅助MSE日志又除以adaptation_batch_size=64，但实际监督仍吃整个PPO minibatch。推荐日志按实际辅助更新次数平均，不额外除64；这只修正报告数值，不改变反向传播标量。与官方日志对照需披露64倍显示差异，不能据此声称估计误差突然改善。

非有限损失／梯度按运行失败处理，在该optimizer.step前停止并保存诊断；一个minibatch的主步已完成而辅助失败时不尝试伪回滚，该运行不继续、不能作为完整更新checkpoint恢复边界，最后有效完整更新checkpoint按A07管理。训练数值错误不转成普通episode reset。

独立验证：主步后decoder逐项不变；辅助步后Actor头／Critic／std逐项不变；用可辨识合成输入确保encoder在两阶段均有有效梯度。先做主步建立Adam动量，再做辅助步，反向顺序也覆盖，以捕捉None／零梯度与动量问题。仅检查grad值不足以代替参数／optimizer状态变化检查。辅助预测必须在主更新后重新计算、同刻标签无梯度；两套Adam独立动量保存／恢复与一次后续更新一致性由A07／A11验证。诊断日志用独立构造的四组误差核验均值，避免依赖实现自证。

**用户答复（2026-10-31）：**“同意”。用户在确认前要求解释主／辅助更新区别及为何分开，相关取舍已完整记录。

**已确认结论：**两套Adam按参数表注册，encoder共享且保留独立动量；每PPO minibatch先主更新、再以新encoder重新前向辅助更新一次，辅助学习率1e-5。阶段间全模型set_to_none=True，主阶段保留裁剪、辅助不新增裁剪；目标无梯度、非有限损失／梯度失败退出。辅助MSE日志不额外除64，A07保存／恢复两套optimizer。此批准不冻结主PPO参数或Critic字段。

## 当前与目标调用关系及完整接口框架

当前源码位置为`UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py`及同目录`ppo.py`。实施位置为已规划、尚未建立的`UniFP_v3/`对应文件；新V3调用方使用Q5接口，共享B2路径如受影响必须在A11补回归，不能修改只读官方副本。沿用官方模块属性名`adaptation_encoder_module`、`adaptation_decoder_module`、`actor_body`、`critic_body`与`std`，不为本题新增通用网络／后端抽象。

构造参数继续由Runner传入`num_obs=1856`、`num_single_obs=58`、`num_obs_pred=12`、`num_actions=14`、A06已确认的`num_privileged_obs=294`及已确认网络／activation／init_noise_std配置。配置由02装配后注入；Q3权重和网络配置不与其他任务共享可变列表。Critic不因本模块收束自动照搬官方149维；A06-Q4d／Q5现已回填模型输入294及schema，终止数据边界见A06。

| 接口／调用方 | 完整类型与数据职责 | 状态／失败与验证接缝 |
|---|---|---|
| 模型分布更新，PPO调用 | `update_distribution(self, observations: torch.Tensor) -> None`；float32[M,1856]→当前Normal分布，mean[M,14]、std按[14]广播 | 更新模型分布缓存，不采动作、不改环境历史；校验输入／输出有限及正std |
| 训练动作，PPO.act调用 | `act(self, observations: torch.Tensor, **kwargs: Any) -> torch.Tensor`；观测[M,1856]→原始动作[M,14] | 采样消耗Torch RNG；分布供logp／熵读取，PPO保存原始动作 |
| 当前分布概率，PPO调用 | `get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor`；原始动作[M,14]→对数密度和[M] | 依赖同批obs刚生成的分布；不得用另一批缓存或限幅动作替代 |
| 分布只读属性，PPO调用 | `action_mean -> torch.Tensor[M,14]`；`action_std -> torch.Tensor[M,14]`；`entropy -> torch.Tensor[M]` | 只读当前分布值，不能借读取推进RNG／历史 |
| 估计预测，辅助损失调用 | `get_student_latent(self, observations: torch.Tensor) -> torch.Tensor`；[M,1856]→[M,12]预测 | 保持梯度可达encoder／decoder；不detach、不转CPU，不依赖标签／Critic；名称沿官方，返回不是latent |
| 确定性推理，Runner评估入口调用 | `act_inference(self, ob: Dict[str, torch.Tensor], policy_info: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor`；只读ob['obs'][N,1856]→mean[N,14] | eval及无梯度；可选写state_estimates[N,12]同设备detach张量；不请求则只需动作支路，无可变默认字典 |
| 推理内部学生入口（保留原调用关系） | `act_student(self, observations: torch.Tensor, policy_info: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor`；[N,1856]→mean[N,14] | 与act_inference相同输出语义，避免内部仍残留旧latents／CPU转换 |
| 价值接口，PPO调用（字段归A06） | `evaluate(self, critic_observations: torch.Tensor, **kwargs: Any) -> torch.Tensor`；[M,294]→V[M,1] | 与encoder分离的Critic；不向Actor开放privileged数据，字段及294→512→256→128→1网络已由A06确认 |
| 模型reset，PPO调用 | `reset(self, dones: Optional[torch.Tensor] = None) -> None`；可选long[N]0／1 | 当前为无循环网络，无需清隐藏状态；不重置参数／std，环境历史reset归A04 |
| PPO更新，Runner调用 | `update(self) -> Tuple[float, float, float, Dict[str, float]]` | 保留均值value loss、surrogate loss、aux loss及分组aux日志四项；shape／buffer生命周期细节归A07，阶段诊断汇总不改变该返回协议 |

下面为无业务实现的职责伪代码，完整学习参数、returns与checkpoint在A07收束：

```text
Runner装配：02有效配置＋A04 shape＋A06 Critic定义
  → ActorCritic（Q4网络、Q5 std）
  → PPO：主Adam[encoder,Actor,Critic,std]；辅助Adam[encoder,decoder]

采样：env同刻obs／privileged／label
  → PPO.act：act(obs)得到a_raw、evaluate(privileged)得到V
  → 保存a_raw对应logp／分布及旧obs／label
  → env.step(a_raw)：03裁剪执行、A04发布新包
  → storage同索引保存旧包转移，收集T=24

每minibatch：
  → 全模型清梯度(None)
  → 当前obs分布＋已存a_raw logp＋独立Critic → 主损失
  → 主backward／有限性检查／裁剪 → 主Adam.step
  → 全模型清梯度(None)
  → get_student_latent(obs)：新encoder→decoder预测
  → label.detach → Q3分组MSE
  → 辅助backward／有限性检查 → 辅助Adam.step
  → 清梯度、汇总诊断／实际MSE日志

评估：eval＋无梯度上下文
  → act_inference({'obs':历史}, 可选policy_info)
  → mean交环境执行，state_estimates交日志／评估
  → 不更新模型／优化器，环境噪声与评估协议归A10
```

测试接缝汇总：shape／切片与末层激活；固定输入的手算分组损失；采样logp／熵和均值推理；正std失败；人工力／接触力标签及旧新episode对时；主／辅助阶段有动量时的参数隔离；两套optimizer保存恢复；辅助日志去除额外64倍缩放。具体测试代码与容差由A11按单元／整体两类收束，真实Gym与学习效果不能由本轮静态文档检查替代。

## A／B衔接

A就定义未来力输出；A全零标签只支持零力偏差检查，不能当作非零外力估计已通过。

## 失败语义与验证范围

网络shape或标签定义不兼容时拒绝续训；不以strict=False掩盖缺失层或重新初始化。

验证范围：固定输入检查shape、确定性推理、梯度归属和标签缩放；B增加已知非零力trace的监督对应检查。

## 讨论记录

Q1—Q6已确认并完成模块总结、完整流程、接口框架及原文对照；当前进入A06-Q1。A05业务代码尚未实施，不生成正式实施票据或启动训练。00—03已确认结论继续有效（03新增数值仍按原流程批准／验证），A04-Q1—Q7已整组确认；本模块继承A04数据契约，并须逐项记录论文／固定官方代码的保留与修改及验证依据。按总体计划完成本部分全部设计与实施票据确认后才进入本部分编码。
