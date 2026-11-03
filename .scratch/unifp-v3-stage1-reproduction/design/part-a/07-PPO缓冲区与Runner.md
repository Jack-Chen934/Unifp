# A07 · PPO、缓冲区与Runner

状态：A07-Q1—Q7及GPU恢复时序补充Q8均已确认并实施；票据05完成（2026-10-31），本机24项回归、远程512×10＋跨进程1轮通过。恢复首步异常从512/512修复为0/512，I／S／C、双Adam、物性及来源哈希核验一致。A10阈值／4096规模／长期动态稳定性仍待06／07，未启动正式8000轮或B。

已确认结论总结：保持单reward／单V／单GAE及14维联合logp的PPO；T=24，Actor1856／Critic294／标签12／动作14的预分配缓冲区同刻配对、detach并copy_入库，dones为bool、完整采集且returns就绪后训练，整除失败且每update一次随机排列跨epoch复用。沿实际官方训练配置5epoch×4minibatch、主lr1e-3并按联合KL自适应、gamma0.99／lambda0.95、clip0.2、value1／clipped、entropy0.01、主grad1，主后辅lr1e-5且每批一次；参数集合／清梯度／辅助日志沿A05。纯timeout以reset前真实终止后继V替代旧V_t，只补偿一次并done遮断GAE；失败不补偿，24步存活切段保留后继V。Runner完整更新成功后提交迭代／策略步／转移数，日志和episode局部时钟不掌控学习进度。完整空边界检查点包含模型、双Adam、实际lr、RNG、学习计数、全局环境训练状态及配置／源码／资产／schema身份，严格恢复后新episode重建，不承诺PhysX逐比特连续。A按剩余预算累计到8000、每200轮及终点保存，同目录临时文件核验后原子发布、失败停止，A冻结／B独立代码和环境；长训前资源与跨进程恢复短测，A8000不自动进入B。逐项官方保留／修改见差异记录，跨模块回填项及测试见末节。

## 本部分职责与前置依赖

交付完整PPO训练、准确8000轮停止、完整检查点及恢复能力，不能推迟到B才实现恢复。

前置依赖：A04—A06的相关结论；共同基线00—03。 同部分按04→11讨论，11最后收束测试；B依赖A，A不依赖B实现完成。

共同的A／B范围依据见[总体计划](../../development-plan.md)，源码与原提案见[公共07](../07-PPO缓冲区与Runner.md)，逐题原文对照见[A07差异记录](../../comparison/A07-论文与官方源码差异.md)。当前官方调用链见[任务入口](../../README.md#当前官方链路)。本文件不替代尚未完成的逐项Q&A确认。

## 按依赖排列的讨论项

| 问题 | 本次需要确定的内容 | 状态 |
|---|---|---|
| Q1 | 单reward／单GAE／全部动作logp求和的保留及必要官方缺陷修复清单。 | 已确认 |
| Q2 | storage[T,N,D]字段、tensor所有权／detach、minibatch整除与采样语义。 | 已确认 |
| Q3 | done、timeout及rollout截断的bootstrap，明确reset前真实终止后继V而非旧V或reset后状态。 | 已确认；修改点已登记 |
| Q4 | 主PPO与辅助监督更新顺序、共享encoder、主／辅助优化器与数值参数。 | 已确认 |
| Q5 | 完整更新完成量、累计转移、策略步／课程计数；启动与reset额外步不得计成完整训练轮。 | 已确认 |
| Q6 | 完整checkpoint：模型、主／辅助优化器、已用调度器、RNG、计数、配置、资产／源码／schema身份；无半轮buffer恢复承诺。 | 已确认 |
| Q7 | 8000轮准确停止、写盘失败处理、恢复短测；A冻结运行检出、B隔离开发和资源检查。 | 已确认 |

具体讨论时逐题给出推荐答案，在本文件记录问题、推荐依据和用户确认结论；未展示的字段、数值或接口不自动获批。完成本议题后，在开头补齐覆盖全部结论的总结。

## Q1 · 单总奖励、单GAE与14维联合PPO（已确认）

**问题：是否保留官方单总奖励／单价值／单GAE和全动作联合概率的PPO主结构，仅作V3维度适配与已识别的必要实现修正？**

**推荐答复：**保持一个环境标量总reward、一个Critic输出V、一套returns／GAE及一个联合策略概率比；将官方17维动作改为V3的14维，四转向／四驱动／六臂共同接受同一任务优势信号。奖励分项及权重归A08，不能为轮式形态自行新增底盘／机械臂独立Critic、独立优势、独立PPO优化器或分组概率比。当前只确认算法组织，不提前锁定gamma／lambda／clip／学习率等尚未展示数值。

```text
环境奖励分项（A08定义） → 每环境一个reward_t[N]
特权历史[N,294] → Critic → V_t[N,1]
reward／V／终止资料 → 单套returns、GAE及rollout优势标准化
历史观测[N,1856] → Actor → 14维对角高斯
原始采样动作a_t[N,14] → 各维logp求和 → 联合logp_t[N]
新旧联合logp → 单一PPO ratio → 同一优势下的clipped surrogate
主PPO loss = surrogate + value_loss_coef * value_loss - entropy_coef * entropy
主Adam一步 → 按A05已确认规则重算辅助预测 → 辅助Adam一步
```

对角高斯指条件于观测时各动作维独立建模，不意味着各组独立优化。联合logp为14维logp相加，ratio为新旧联合logp之差取exp；不能把三个关节组各自ratio平均或分别clip后再称为相同PPO。Gaussian熵也按动作维相加，再按batch归约。PPO概率使用原始采样动作，环境限幅后动作只用于实际执行及历史，沿A05-Q5，不将执行动作重新代入旧logp。优势标准化保留官方对整个rollout样本归约的机制，不做逐关节组或逐环境独立标准化；采样与数值细节在Q2／Q4确定。

辅助监督沿A05-Q3／Q6，四组状态估计MSE不加入环境reward、returns或GAE；由同一PPO.update循环调度独立辅助Adam，不新增第三条更新算法。它可通过共享encoder影响后续策略，但不直接成为Actor额外输入或另一个任务价值头。A／B沿用同一联合结构，B改变已确认课程激励而不拆分优化对象。

### 与官方及已确认模块的对应

官方`PPO:77–104,119–170`单Critic价值、全动作联合ratio、clipped surrogate／value loss／entropy；`STORAGE:72–89`单GAE和全rollout优势标准化；`AC:134–153`熵／log_prob按动作维求和；`RUNNER:115–151`采集24步后算returns并更新。保持上述组织。输入shape由A04—A06适配为Actor1856、Critic294、标签12、动作14；reward分项具体公式及超参数均不因本题自动批准。

### 必要修正清单与后续归属

下表是现有证据与已确认约束的实施交接，不是把后续题目未展示方案整组批准。

| 位置／事实 | 已确认方向或问题 | 后续决策归属 |
|---|---|---|
| PPO:94–95以旧V_t补偿timeout | A04／A06已确认必须提供reset前终止后继输入，失败与纯timeout分开 | Q3确认具体补偿／GAE公式、终止子批映射和避免重复bootstrap |
| PPO:85–87暂存观测引用，STORAGE:54–67在step后copy | A04／A06要求发布包不可变、同边界对时 | Q2确认转移所有权／detach／存储字段及混合reset测试 |
| STORAGE:96–99用整除后的数量生成索引 | 不能静默丢弃尾部样本；02要求运行配置显式核验 | Q2决定整除校验及minibatch／epoch语义，不本题新增padding框架 |
| PPO:53–55两套Adam注册全模型，清梯度未显式set_to_none | A05-Q6已确认主／辅参数集、全模型清梯度及两步顺序 | Q4落实已确认边界，补主PPO数值；不重新开放合并loss或冻结encoder |
| PPO:207–209辅助日志额外除64 | A05已确认只修显示尺度，按实际辅助更新次数平均 | Q4落实，A10确定日志口径；训练loss定义不因此改变 |
| RUNNER:153–161保存命名／成员迭代更新位置 | 02要求8000为完整PPO训练迭代，不能以文件名猜完成量 | Q5明确计数及更新完成边界，Q7明确停止／写盘 |
| RUNNER:236–250仅模型／主优化器等有限状态 | A05要求两套Adam，A／B须兼容完整学习状态 | Q6逐项确认格式／身份／RNG／恢复承诺，Q7验证恢复／写盘失败 |

数值非有限／非法std沿A05失败，不额外引入静默clamp、换优化器或自动降级；如发现新事实，先回写所属设计再确认。精确恢复与PhysX逐步连续不是同一承诺，留Q6讨论。

验证接缝：独立构造14维分布核对联合logp／熵，原始动作与执行限幅动作可辨识；一个标量优势同时作用于全部动作组的策略梯度；辅助目标变化不直接改变本rollout的环境reward或GAE。Q2—Q4用可手算小轨迹验证截断及样本覆盖，Q6—Q7验证完整更新后保存／恢复；容差与整体训练验收在A11收束。本轮仅源码核对与设计记录，不执行学习实验。

**用户答复（2026-10-31）：**“同意”。

**已确认结论：**保留单总reward／单V／单GAE及14维动作联合logp、单ratio的PPO结构，统一rollout优势标准化，不拆底盘／机械臂PPO；辅助MSE不进入环境reward／GAE，主辅调度沿A05。Q1修正清单只确认后续承接范围，具体公式／参数／恢复在Q2—Q7逐项讨论。

## Q2 · 缓冲区所有权、字段与完整样本覆盖（已确认）

**问题：是否保留官方[T,N,…]缓冲区、全字段同索引及每rollout一次随机排列的采样方式，同时明确入库拷贝、无梯度和整除校验？**

**推荐答复：**保留简单预分配张量和官方Transition暂存对象，不新增回放池、循环网络采样器或通用batch类。T=24，N由02运行配置确定；整个rollout收集完成、returns／advantages计算完成后才能生成训练批。更新阶段不混入新采样转移，训练后清空有效长度，下一轮覆盖写入。

### 字段、shape与写入时刻

所有浮点字段float32，与仿真／学习显式设备一致；dones在缓冲区用bool存储，环境long[N]只接受0／1并在入库转换。这是相对官方uint8的类型整理，转换不改变终止语义。策略动作和logp均为原始采样动作，不是执行裁剪后动作。

| 官方缓冲区属性名 | 目标shape | 来源与写入时刻 |
|---|---|---|
| observations | [24,N,1856] | 动作前obs_t；每个索引内部已经包含32帧历史，不再按episode拆分重建历史 |
| privileged_observations | [24,N,294] | 同边界Critic三帧历史 |
| observation_preds | [24,N,12] | 同边界环境监督真值obs_pred_t，不是decoder预测 |
| actions | [24,N,14] | Actor在obs_t下的原始采样动作 |
| actions_log_prob | [24,N,1] | 该原始动作的旧14维联合logp |
| mu／sigma | 各[24,N,14] | 同次采样分布的旧均值／正std，供KL诊断／自适应使用 |
| values | [24,N,1] | 同边界旧Critic值V_t，不在学习中覆盖成新值 |
| rewards | [24,N,1] | step后的标量训练奖励；纯timeout如何补偿或增加辅助存储字段由Q3确定，环境原始奖励／统计不原地改写 |
| dones | [24,N,1]，bool | 该动作转移的episode结束标记；不是rollout结束标记 |
| returns／advantages | 各[24,N,1] | rollout满24步后由compute_returns填写，无梯度；具体GAE／标准化数值归Q3／Q4 |

bool dones不代替terminated／time_outs来源，混合终止及terminal_value的计算／存储协议仍归Q3。本题不预先增加全量[T,N,294]终止历史缓存；A06要求step返回的[K,294]资料必须有效，PPO如何消费和保存结果由Q3确认。

### 数据所有权与生命周期

1. PPO.act在无梯度模式完成采样，暂存动作前obs／privileged_obs／标签和动作／旧logp／mu／sigma／V。暂存字段全部detach；A04／A06已要求环境发布包在后续step／reset不可变，因此这些观测可保留只读引用直到本步入库，无需先clone整包再入库复制。
2. env.step完成后，将对应reward／done／已由Q3定义的终止处理结果与旧输入配对，add_transitions在no_grad下逐字段copy_到缓冲区自己的预分配存储；不能只保存环境视图列表。动作分布参数也必须复制，之后模型参数变化不得改写已存old_mu／old_sigma。
3. 每次入库只写step指向的一行，所有必需字段完成后才step+=1；单次失败停止运行，不发布半条转移为可训练样本。step>=24时报溢出；未收集完整24步时拒绝算returns／更新，不以未填的零槽补足。缺字段、shape／dtype／device或有限性不符时失败，不隐式广播／换设备。
4. 环境局部reset只改变环境／历史，不清rollout已有转移；done负责episode边界。update期间旧数据、old_values／logp／分布参数和targets固定无梯度；新策略／价值前向另算并带梯度，不能从old缓存反传。
5. clear只将有效长度step置0，并使上一轮returns失效，不每次把大缓冲区清零；下一次采集必须完整覆盖24行。returns是否已准备可用一个内部bool表示，add／clear时置False、compute_returns成功后置True，避免采满但沿用旧returns；不新增版本管理框架。checkpoint是否包含未完成rollout由Q6确认，沿当前范围不承诺半轮恢复。

### minibatch与epoch语义

设B=24×N，J为num_mini_batches，E为num_learning_epochs，M=B/J。J、E的具体数值在Q4确认，本题只要求正整数、J≤B且B%J=0，在采样前校验；不整除直接报配置错误，不丢样本、不填padding、不静默改J。

沿官方以时间优先展平：[T,N,D]→[B,D]，索引b=t*N+env_id。训练每个样本自带历史，因此无RNN的MLP可打乱这些索引，不需要把一个episode拼成整段再训练。所有字段使用同一batch_idx，避免Actor观测／标签／动作／价值错配。标量字段在batch中保留[M,1]，模型logp[M]与它对接时只显式移除末尾单维，不用无参数squeeze误删M=1的batch轴。

每次生成器调用仅产生一次同设备randperm(B)，切成J个等长minibatch；在本次update的E个epoch中复用该排列与分组，完整保留固定官方`STORAGE:96–131`的方式，不默默改为每epoch重排。下一次rollout update重新产生排列并推进同一Torch随机源。每个epoch恰好使用每个样本一次，总计E×J个主minibatch；每批的主Adam后辅助Adam仍按A05一次，不能把辅助子步混计成新的PPO epoch。相同seed和随机源状态可重现排列，但不据此保证不同硬件仿真逐步一致。

保留官方generator的11项顺序：`obs_batch, critic_obs_batch, obs_pred_batch, actions_batch, old_values_batch, advantages_batch, returns_batch, old_logp_batch, old_mu_batch, old_sigma_batch, None`。末项是无RNN的masks占位，不表示done mask被丢弃；done已用于Q3回报计算，推理网络无需循环隐藏状态。参数属性名target_values_batch沿官方可保留，但文档明确它是旧values，不是returns。具体类型签名及完整PPO接口在本模块收束时记录。

### 原文对照与验证接缝

官方`STORAGE:34–67`以相同属性及float32预分配，dones为byte，入库copy_；`PPO:85–87`暂存观测引用；`STORAGE:96–131`先展平，单次randperm跨epoch复用，所有字段同索引。本方案保留大部分组织，按V3适配宽度，明确只读引用到独立拷贝的生命周期、bool done、完整采集／returns状态、整除失败及显式标量轴。论文已核查段落不逐项规定这些内存／采样实现，不能把同名PPO直接等同于所有库的每epoch重排方式。

验证接缝：用时间／环境联合编码的样本及相互独立字段偏移检查b=t*N+env_id、同索引及标签配对；每epoch无重复／无遗漏，跨epoch顺序相同、下次update消费新随机排列；观测包跨step不变与入库后修改源／参数不改缓冲区分别验证。覆盖M=1、不可整除、缺一行／缺returns、溢出、局部reset保留旧转移、clear后旧returns不可用及detach无计算图。不可整除必须在真实采样前失败；实际E／J配置及容差留Q4／A11。本轮尚未编码或运行训练测试。

**用户答复（2026-10-31）：**在解释24为每环境采集的策略步数、N为并行环境数量后，用户答复“同意”。

**已确认结论：**保留T=24的预分配[T,N,…]缓冲区及Transition，按1856／294／12／14适配字段；无梯度暂存与同刻配对，入库copy_形成独立存储，旧数据不可被后续环境或参数变化改写。dones存bool；收满24步且本轮returns就绪才训练，clear只重置有效长度及就绪标志，局部reset不清已有转移。B=24N必须被J整除；全字段同索引、每update一次随机排列跨E个epoch复用，保留11项generator输出与标量末轴。E／J数值及GAE／timeout公式仍分别归Q4／Q3，N=4096示例不等于已通过容量预检。

## Q3 · 失败、纯超时与rollout切段的后继价值（已确认）

**问题：是否保留官方奖励补偿加done遮断的GAE结构，将纯timeout补偿改为真实终止后继价值，确保失败不bootstrap、episode之间不串接、24步切段不等于episode结束？**

**推荐答复：**继续使用Q2已有rewards／dones／values字段，不新增全量终止历史或逐转移后继价值数组。在PPO.process_env_step中即时消费A04／A06的终止快照，对纯timeout求reset前后继V并只补偿一次；普通失败不补偿。之后仍用done同时遮断常规后继项和GAE递推。这里bootstrap指以Critic估计未实际采集的后续折扣回报，不是发放额外任务奖励；gamma／lambda具体数值由Q4确认。

### 终止分类、取值来源与处理时刻

| 转移情况 | terminated／time_outs／done | 后继价值与GAE处理 |
|---|---|---|
| 普通存活、未到rollout末尾 | False／False／0 | 用values[t+1]，GAE可沿同episode递推 |
| 物理失败 | True／False／1 | 不做后继价值补偿；本步奖励及失败惩罚按A08保留，切断GAE |
| 纯episode超时 | False／True／1 | 用reset前旧episode的终止后继历史求V，训练奖励加gamma×该值，切断GAE |
| 失败且同时达到时限 | True／False／1，环境已执行失败优先 | 按失败处理，不因到时补偿；若infos互相矛盾则接口错误失败，不自行猜测 |
| 采满24步但episode存活 | False／False／0 | 用rollout末尾真实后继历史求V；不设done，不reset或清历史；仅在本轮末尾令后续GAE累计为0 |
| 第24步恰为失败或纯timeout | 按上述真实终止标记 | 失败不补偿，纯timeout单次补偿；不能再加reset后新初帧价值 |

PPO在env.step返回后、下一次采样及任何优化器更新前，无梯度消费终止资料，使用与本rollout旧values同一版本的Critic参数。快照虽在reset前构造，可在step返回后求值，因为A04／A06保证其内容独立且不被reset覆盖；不能重新读取模拟器reset后的body／命令来重建。

A06的terminal_env_ids为全部done的long[K]，terminal_privileged_obs为逐行对应的float32[K,294]，不是只含timeout。先核验ids范围／唯一性、与done集合相同、shape／device／dtype及有限性；terminated和time_outs为互斥bool[N]且dones与二者或一致。然后在这K行上计算row_mask=time_outs[terminal_env_ids]，按同一row_mask选出timeout的环境id与快照行，得到[L]和[L,294]。无需要求ids排序；不得把环境全局id直接当K行快照的行号。L=0跳过终止Critic前向；L>0时evaluate输出[L,1]，只写回对应环境的补偿位置。缺失／错配／非有限资料按已确认契约失败，不能改用旧V、零值或新episode观测。

### 训练奖励与GAE公式

令r_t为环境原始奖励、d_t为done的0／1浮点表示、u_t为纯timeout标记；V_t为动作前旧价值。用V_terminal表示reset前真实后继历史的价值，则缓冲区存储的训练奖励为：

```text
r_train[t] = r_env[t] + gamma * u[t] * V_terminal[t]
```

这里V_terminal仅需对纯timeout子批计算，其他行补偿为0；不能将未计算或非法值先乘0来掩盖。PPO先clone原始奖励再补偿，环境reward张量、episode奖励汇总及Runner原始奖励统计保持原值。缓冲区rewards明确表示补偿后训练奖励，日志不得将其混称为环境实际奖励。不为本题增加独立raw_rewards全rollout副本；原始统计按环境／Runner已有路径累积，具体日志键归A10。

采满T=24后，对最后一步仍存活的行评价真实后继privileged_obs；last_values为float32[N,1]，done行置0，存活子批写入旧Critic预测，空存活子批跳过前向。然后倒序计算：

```text
next_value[t] = values[t+1]          当t < T-1
next_value[T-1] = last_values
next_gae = 0                       每个环境本轮递推的初始边界

delta[t] = r_train[t] + gamma * (1-d[t]) * next_value[t] - values[t]
adv_raw[t] = delta[t] + gamma * lambda * (1-d[t]) * next_gae
returns[t] = adv_raw[t] + values[t]
next_gae = adv_raw[t]
```

纯timeout的后继价值已包含在r_train中，done=1使常规next_value和后续GAE为0；不能再把timeout当存活、再加一次V_terminal，或接入reset后新episode的优势。失败也是done=1，但没有奖励补偿，因此其return在该步仅为原始即时奖励。rollout末尾存活时done=0，delta保留last_values，即使next_gae从0起算也不把存活状态当真实终止。

returns使用未标准化adv_raw计算；随后再按Q1对全rollout优势统一标准化供策略损失使用，不能反过来用标准化优势构造Critic监督returns。标准差估计方式／epsilon及gamma／lambda具体数值在Q4落定。上述targets全部detach／no_grad，compute_returns仅在完整计算且有限性检查成功后将returns就绪标志置True；失败不更新或标记为就绪。

### 诊断算例与验证接缝

以下gamma=0.99只为解释算例，不批准训练超参数。设某步r=1、V_t=2，真实终止后继V=3，reset后新episode V=100：

| 情况 | 训练奖励 | 未标准化adv_raw（该步done时） | returns |
|---|---|---|---|
| 纯timeout | 1+0.99×3=3.97 | 3.97−2=1.97 | 3.97 |
| 物理失败（也包括失败且到时） | 1 | 1−2=−1 | 1 |

纯timeout不使用旧V_t=2，不能得到官方旧值补偿的return=2.98；也不使用reset值100。单步rollout末尾存活且真实后继V=3时，delta同为1.97、return为3.97，但其done仍为0，episode继续；不能因为数字相同就混淆边界语义。

后续测试用独立手算多步轨迹覆盖存活→timeout、存活→失败、timeout→新episode及rollout最后一行各终止类型；为下一episode设置巨大且可辨别的奖励／V，确认不会污染旧episode。打乱terminal_env_ids、混合K个done中仅L个timeout、K=0／L=0／所有环境done、M=1、非法标志／缺失资料和原始reward不变均覆盖。对奖励补偿公式与独立的“超时允许价值bootstrap但GAE在done断开”的双mask参考算例比较，确保补偿恰好一次；不把两种方案同时叠加到业务实现。

官方STORAGE:88–94的get_statistics通过done=self.dones后写done[-1]=1会改写实际存储。Q2已确认done表示真实episode边界，因此目标训练链不能让统计路径改写它；若保留调用该辅助方法，仅在局部clone上标记统计段尾。该约束不新增done语义，也不把rollout切段记作episode结束；用统计调用前后dones与returns不变验证。

### 与论文及官方的保留／修改

保留PPO:90–99的克隆奖励、超时奖励补偿、随后入库流程，以及STORAGE:72–86的done遮断／倒序GAE／先return后标准化优势结构。修改PPO:94–95的补偿来源，由旧transition.values改为真实终止后继V；按A04已确认的纯timeout／失败优先处理，而非到时即补偿。RUNNER:119–151的24步采集后算returns再update组织保持；末尾存活环境评价真实后继值，done行不评价reset值用于旧episode。论文已核查段落没有逐项规定这些边界写法，本修正属于明确记录的实现差异，不声称已有训练效果提升证据。本轮只记录方案，数值测试与物理训练均未执行。

**用户答复（2026-10-31）：**“同意，请记录这个修改点”。

**已确认结论：**保留超时奖励补偿及done遮断GAE的结构，纯timeout在入库前无梯度计算reset前真实终止后继V，并将gamma×该值一次性加入克隆后的训练奖励；不再以旧V_t补偿。物理失败与失败且到时均不补偿；纯timeout仍done=1，切断GAE，不串入新episode。24步采满但存活时，用真实后继V而不标done；returns先于优势标准化计算。终止ids按行映射、资料校验、原始奖励隔离及统计不能改写done均按本节执行。gamma／lambda等具体数值仍留Q4，本次确认不表示已修改业务实现或完成实验验证。

### 用户要求单独登记的修改点：纯超时bootstrap来源

- **官方原行为：**固定提交`68847a070f88d731058c3d8476929bc3b205f5bd`的`UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py:94–95`，将`gamma * transition.values * time_outs`加到训练奖励；transition.values是动作前旧V_t。
- **已确认目标行为：**新V3实现仍在`PPO.process_env_step`中克隆奖励、补偿后入库，但使用A04／A06提供的reset前终止后继历史，经同一旧Critic求V_terminal，只对纯timeout加`gamma * V_terminal`一次。
- **修改原因：**超时截断需要估计实际动作执行后的后续回报；动作前旧值及reset后新episode的值不对应这个后继状态。
- **保留内容：**单奖励、单V、单GAE、done遮断、先构造returns再标准化优势、rollout满24步后更新及既有缓冲区组织；补偿不改变环境原始奖励／episode统计。
- **影响边界：**纯timeout转移及同episode前序转移的returns／advantages可能改变，并影响后续训练；物理失败不补偿。该变化是明确的实现修正，不能将新训练称为与固定官方代码数值完全相同，也不据此认定性能已提高；对照报告须披露此差异。
- **待实施验证：**旧V／真实终止V／resetV可区分的手算样例、失败与超时同时发生、无序终止ids、跨episode隔离、rollout末尾三类边界、单次补偿及原始奖励不变；实施票据／A11承接，当前未执行。

完整保留／修改对照同步至[A07差异记录](../../comparison/A07-论文与官方源码差异.md#q3已确认结论与官方对照)。


## Q4 · 主／辅助更新与官方训练参数基线（已确认）

**问题：是否将固定官方训练配置的PPO数值整体作为V3第一阶段起点，并按A05已确认的主后辅、明确参数集合与清梯度规则实现，不因轮式形态预先调整学习超参数？**

**推荐答复：**沿用真实配置注入后的数值，而不是直接摘用PPO构造函数默认值。参数取自`CFG=UniFP/legged_gym/envs/base/legged_robot_config.py:196–209`；`B2CFG=UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:313–315`只在algorithm中重申entropy_coef=0.01，其余继承基类；`RUNNER:53–55,73–74`将algorithm配置作为kwargs传入PPO。下表推荐已由用户确认；此前已确认的A05参数／机制继续有效，不由本题重新开放。

### 参数表与每轮更新次数

| 参数 | 官方训练配置／源码 | V3推荐 | 业务含义 |
|---|---|---|---|
| num_learning_epochs | 5 | 5 | 一轮rollout数据重复学习5遍 |
| num_mini_batches | 4 | 4 | 每个epoch将全部24N样本分成4批；每批M=6N |
| 主learning_rate初值 | 1e-3 | 1e-3 | 主Adam起始学习率，随后按官方KL规则自适应 |
| schedule／desired_kl | adaptive／0.01 | adaptive／0.01 | 按14维联合分布的旧到新KL均值调节主学习率 |
| gamma | 0.99 | 0.99 | 每个20ms策略步的奖励折扣，也用于Q3超时补偿 |
| lam | 0.95 | 0.95 | GAE递推衰减，和gamma共同决定远期残差权重 |
| clip_param | 0.2 | 0.2 | 策略ratio使用[0.8,1.2]形成clipped surrogate；不是动作限幅 |
| use_clipped_value_loss | True | True | 使用相对旧V的价值变化裁剪损失 |
| value_loss_coef | 1.0 | 1.0 | 主损失中的价值损失系数 |
| entropy_coef | 0.01 | 0.01 | 主损失减去0.01×联合高斯熵均值 |
| max_grad_norm | 1.0 | 1.0 | 主阶段所有参与参数梯度的全局L2范数上限 |
| 辅助Adam学习率 | PPO:12–14为1e-5 | 1e-5（沿A05已确认） | 独立于主KL调度，保持固定 |
| 每minibatch辅助步数 | 1 | 1（沿A05已确认） | 每个主Adam步完成后重算辅助预测再更新一次 |
| 优化器 | PPO:53–55使用Adam，未传其他Adam选项 | 保留Adam默认其余选项 | 不改AdamW，不新增weight decay、warmup或学习率scheduler对象 |

E=5、J=4后，每rollout有20个主Adam步和20个辅助Adam步；这是一个完整训练迭代内部的更新，不是20个训练迭代。N=4096时B=98304、M=24576；N=512时M=3072，N=32时M=192。沿Q2，每update只生成一次排列，在5个epoch复用；每epoch每个样本一次，不新增epoch重排。显存／容量须真实预检，不能以本题确认替代预检或默默减小N／改变J。

PPO构造函数本身的默认值为epochs=1、minibatches=1、gamma=0.998、entropy=0、schedule=fixed（PPO:21–35）；这些并非官方训练配置注入后的生效基线。本工作以显式冻结的V3配置注入并保存配置快照，不依赖两份默认值偶然一致。上述配置是复现起点，无对V3最优或已经稳定收敛的证据；后续因实验调整须记录具体变更与依据。

### 主损失、价值裁剪与优势标准化

保持Q1／Q3组织，Q2所有旧targets无梯度，当前分布／价值重新前向。M为minibatch样本数，联合logp与entropy为[M]，V／old_V／returns为[M,1]；标量axis只squeeze(-1)，避免M=1失去batch轴。

```text
ratio = exp(new_joint_logp - old_joint_logp)
L_policy = mean(max(-adv * ratio, -adv * clamp(ratio, 0.8, 1.2)))
V_clip = old_V + clamp(V_new - old_V, -0.2, 0.2)
L_value = mean(max((V_new - returns)^2, (V_clip - returns)^2))
L_main = L_policy + 1.0 * L_value - 0.01 * mean(joint_entropy)
```

价值裁剪是以旧V为中心构造损失候选，不直接将网络输出／returns限幅，也不是按价值大小乘0.2。沿官方同一clip_param=0.2同时用于策略ratio与价值差值，不新增独立value_clip参数；value loss不额外乘0.5。动作维logp／熵先求和、样本再mean，不改为按14维平均；14维替换17维会改变联合熵／KL数值范围，但本阶段不主动按维度重缩放系数。PPO clip只约束目标函数的一部分，不保证全部更新后的实际KL都小于0.01，也不约束A05辅助步造成的全部策略变化。

全rollout先计算returns，再用`adv_raw=returns-old_values`构造优势；推荐显式采用样本标准差（B-1分母，等价常见torch.std默认的unbiased=True／correction=1），`adv=(adv_raw-mean)/(std+1e-8)`。归约范围为24N样本，不按minibatch重新标准化；实际B≥24，标准差可定义。常数优势转为零，不跳过整个更新，因为价值／熵／辅助项仍有学习信号。保留官方STORAGE:84–86的归约与epsilon含义，实际运行PyTorch按锁定版本使用等价API，不为了API参数名添加通用兼容框架。

### KL自适应主学习率

每个minibatch在主更新前，用当前策略（已包含此前minibatch主／辅助步的变化）与本rollout保存的旧分布比较。保留官方PPO:132–144的KL方向、动作求和、batch均值与调度分支：

```text
KL_sample = sum_i(log(sigma_new_i / sigma_old_i + 1e-5)
                  + (sigma_old_i^2 + (mu_old_i-mu_new_i)^2)
                    / (2*sigma_new_i^2) - 0.5)
kl_mean = mean(KL_sample)

if kl_mean > 0.02:
    lr = max(1e-5, lr / 1.5)
elif 0 < kl_mean < 0.005:
    lr = min(1e-2, lr * 1.5)
else:
    lr保持
```

它对应KL(old || new)的解析式加官方log内部1e-5扰动，因此不是无扰动解析KL，不能因同分布结果有微小非零就当实现失败。保留该1e-5位置、不改为log外epsilon、不按14维平均、不静默把负浮点近零值clamp为正数；测试使用与原表达式独立计算的参考值，并记录与精确解析KL的微小差别。边界0／0.005／0.02不触发相应严格不等式分支。公式中sigma须满足A05已确认的有限且正，出现非有限KL／loss／grad即停止本次运行。

调度只写主Adam各param_group的lr与self.learning_rate；不改辅助lr，不加KL惩罚项、提前停止epoch、额外scheduler、目标网络或辅助KL约束。学习率更新发生在本批主step之前；自适应是每minibatch而非每rollout一次。该调度能响应之前辅助步对共享encoder的影响，但不是对刚完成辅助步的独立限制。最后一批辅助更新完成后不额外为调度增加一次前向；完整保存主当前lr及两套Adam状态由Q6承接。

### 主／辅更新调度与已确认边界

沿A05-Q6，每个minibatch执行：

```text
固定旧数据／targets，模型全量zero_grad(set_to_none=True)
  → 重算当前策略分布及存储动作logp、当前V与熵
  → no_grad计算旧到新KL，更新主lr
  → 计算L_main，检查有限性，backward
  → 检查主梯度有限，主参数全局L2 clip_grad_norm_(max_norm=1.0)
  → 主Adam.step
  → 模型全量zero_grad(set_to_none=True)
  → 用主步后的encoder重新前向decoder，配对无梯度原标签
  → 四组三维MSE系数[0.04,0.04,1,1]相加，检查有限性，backward
  → 检查辅助梯度有限，不新增辅助grad clip
  → 辅助Adam.step，清理残余梯度
```

主参数encoder＋Actor头＋Critic＋std，辅助参数encoder＋decoder；共享encoder在两套Adam中有独立动量。两套Adam不合并，辅助MSE不加入L_main或GAE，不将decoder输出回灌Actor；旧mu／sigma／logp／V／returns在整个E×J循环中固定。只用主参数集合计算主全局范数；若裁剪返回非有限总范数也失败，不继续step。部分更新中途失败不可算一次完整迭代或保存成完整续训快照，计数／写盘由Q5—Q7落实。

为保持官方更新调用和随机源消费，保留PPO:124调用actor_critic.act(obs_batch)来刷新分布的路径，该调用会额外采样一个动作但此样本不用于PPO loss；loss始终使用storage保存的原始动作。该额外样本不送env.step，不计环境步，也不据此新增更新样本。若未来优化为仅update_distribution以省一次采样，会改变随机源推进，须明确记录；本题不顺手改变该路径。

主value／surrogate日志按实际主更新次数20平均；各组及总辅助MSE按实际辅助次数20平均，不再额外除64，沿A05修正。日志修正不改变辅助反传loss，不能把64倍显示差异报告为估计性能提升；具体指标键／采样频率归A10。

### 官方对照、验证接缝与确认范围

保留实际CFG超参数、PPO:132–170的KL调度／主损失／主clip和PPO:181–200的一主一辅顺序。明确参数集合／清梯度与日志修正是A05已确认变化，本题只落地对应数值；Q3超时后继修正继续有效。本题选择固定官方源码作为基线，不能称表中每一项都是论文逐项指定。

验证接缝：核对配置继承注入与PPO构造默认不同；N=32／512／4096的B／M／20个主辅步；独立手算正负优势下ratio clip、value clip取两候选最大值、无额外0.5、样本std与常数优势；KL旧到新方向、14维求和、epsilon位置、阈值等号／零／上下限与辅助lr不变。沿A05测试共享encoder及两套参数／动量边界，主clip全局范数1、辅助无新增clip；M=1及旧数据不反传；非有限输入／loss／梯度失败、原始动作而非更新时临时样本用于loss；辅助日志不再除64。以上仅为后续验证接缝，本轮不执行学习或仿真测试。

**用户答复（2026-10-31）：**“同意”。

**已确认结论：**沿官方实际注入配置使用5epoch／4minibatch、主lr初值1e-3、KL自适应目标0.01及官方阈值／倍率／上下限、gamma0.99／lambda0.95、clip0.2、价值裁剪开启／系数1、熵0.01、主全局梯度范数上限1；辅助lr1e-5、每主步后1次。保留主损失、样本std归一化及epsilon口径、KL方向／epsilon／每批调度和更新时随机采样调用；沿A05明确主／辅参数集合、全量清梯度、辅助重新前向、无新增辅助clip及辅助日志不再除64。每完整迭代20主步＋20辅助步，不改变Q3超时补偿修正；当前仅设计确认，未实施或完成运行验证。

## Q5 · 完整训练迭代、样本量与课程时钟（已确认）

**问题：是否由Runner在每个完整PPO更新成功后统一提交完成量，并将训练课程进度、episode局部时间和初始化／评估步明确分开，避免日志／reset／恢复改变8000轮的含义？**

**推荐答复：**Runner持有三个整数学习进度计数，使用明确单位；完成一次rollout24步、returns及全部20个主步和20个辅助步，且update成功完成、存储有效长度清零后，一次性提交。计数不由log、文件名、env.reset或任一Adam.step递增。持久化格式与恢复身份检查由Q6确认，准确停止／写盘由Q7确认。

### 计数及不变量

| 名称 | 单位／范围 | 初值与递增时刻 | 使用方 |
|---|---|---|---|
| completed_iterations | 完整PPO训练迭代，Python int≥0 | 新训练0；每次完整update成功后＋1 | 训练循环、保存／日志坐标、A8000门限、课程完整轮次 |
| training_policy_steps | 每环境训练策略步，Python int≥0；向量环境同步步次数 | 新训练0；同次提交＋T=24 | 课程累计训练时间及训练预算；不是N倍 |
| training_transitions | 全环境累计训练转移，Python int≥0 | 新训练0；同次提交＋N×T | 学习样本预算、对照曲线横轴 |
| rollout_step | 当前rollout有效长度0…24，沿storage.step | 入库成功＋1；update末尾clear置0 | 只作进行中位置，不是持久完成量 |
| episode_length_buf | 每个环境的真实episode策略步，long[N] | reset所选行置0，每个真实策略转移＋1 | 纯timeout／episode统计；不决定已完成训练轮 |

固定N、T=24的一条续训链，在完整更新边界应满足：

```text
training_policy_steps = completed_iterations * 24
training_transitions = training_policy_steps * N
主Adam已完成步数 = 辅助Adam已完成步数 = completed_iterations * 20
```

最后一式是从头正常运行且每批按Q4执行时的派生核验关系，不新增必须保存的两个全局Adam计数器，也不把内部逐参数Adam step无条件等同它（某些参数无梯度时内部步数可能不同）。三项持久进度虽可推导，保存后可相互核验；不能静默覆盖矛盾字段。是否允许改变N／T的继续训练及身份校验由Q6明确，当前不将换规模实验混报为同等样本预算的原训练。

完成第k次迭代后，模型、双Adam状态、当前主lr及三个完成量属于同一逻辑边界；日志横轴记k，不用零基循环索引k−1冒充已完成量。记录模型快照的迭代身份取同一计数，不能从文件名推断；完整续训快照的持久成功与训练内存里已完成是不同状态，保存失败不得声称已可恢复（Q7）。

### 进行中的rollout与失败语义

正在采集第j步之前（j=0…23），完成量仍为I，存储已有j行；本次动作的累计训练策略步索引为`training_policy_steps + j`。rollout内采集和主／辅更新不会提前将completed_iterations加1。已执行但未完成整轮的数据是本次运行的实际尝试，可记录诊断，不能计为已完成训练或恢复承诺。

```text
读取完整边界进度I／S／C
  → j=0…23：发布(I,S+j)训练时钟 → 采样／env.step／入库
  → 完整计算returns
  → PPO.update：20主步＋20辅助步，成功后clear
  → Runner统一提交(I+1,S+24,C+24N)
  → 同步环境可读进度（无物理／命令／观测副作用）
  → 读取新完成量写日志／判断保存和停止
```

若采集、returns或某个主／辅子步失败，立即停止，不把I改为I+1。发生失败前的环境与参数可能已经变化，不假装回滚本轮；仅可从此前完整快照恢复，不将部分更新模型配上旧I另存为续训快照。诊断可包含当时j和完成的子步位置，但不为半轮恢复新增存储机制。日志关闭、日志函数未调用或日志输出失败都不得让学习计数落后于已成功更新的模型。

### Runner向环境发布训练时钟

建议增加单一窄接口，避免Runner直接改环境计数属性，也避免环境自己用simulate次数推算学习进度：

```text
V3PosForceEnv.set_training_progress(
    self,
    completed_iterations: int,
    training_policy_steps: int,
) -> None
```

两个参数分别是已提交完整迭代数与本次动作之前的累计训练策略步索引；rollout中completed_iterations保持I，training_policy_steps可为S+j，完成边界则为(I+1,S+24)。输入为非负整数；正常同进程训练不倒退，恢复的初始赋值按Q6在新环境采样前进行。方法仅保存这两个只读进度源，不simulate、不加episode时长、不重采命令／物性、不推进历史或改写已发布观测。课程在A09／B09规定的采样事件读取它，不能为了同步进度重复执行课程事件。

不再让原环境global_steps因初始化／评估step累加后充当训练课程时钟。若仍保留物理调用计数用于诊断或局部事件，应明确它统计哪些真实调用，与学习完成量分开；普通目标段、命令保持、接触与局部reset时间继续使用各自episode／策略周期，不把全局时钟套到所有局部事件。force_start_step×24的硬编码严格大于比较不承担A／B切换，A／B身份沿总体计划明确配置。

评估在独立环境读取固定checkpoint进度，评估step只推进自身场景／episode计时，不递增训练完成量或训练课程时间；调试／初始化／reset中为刷新状态执行的额外simulate也不推进训练时钟。A04已要求reset刷新新状态，不把额外零动作env.step当第一条训练转移；reset后episode真实计时从0开始。官方脚本init_at_random_ep_len=True会直接随机写episode_length_buf；若后续A09希望错峰初始超时，须单独明确剩余时限／偏移与真实已执行长度，不能以随机数伪造真实样本数或全局课程进度。本题不提前批准是否采用初始错峰。

### 8000边界及A／B衔接

N=4096、T=24时，A完成量准确为：

```text
completed_iterations = 8000
training_policy_steps = 192000
training_transitions = 786432000
每环境累计训练仿真时间 = 192000 * 0.020s = 3840s
```

这不是墙钟耗时，也不是所有episode连成的单一不间断轨迹。A的第8000次更新完成后先停在该边界，不能因训练策略步达到192000就自动启用非零力。只有A结果及B准备／兼容性均通过既定门槛，才以同一学习完成量进入B；B第一轮开始时完整迭代仍为8000、累计训练策略步从192000开始。本题只规定计数接口，力课程范围／生命周期及B的完整轮内调度留A09／B09，保存恢复留Q6—Q7。

部分内完成轮次可由全局完整轮次和已确认部分起点推导；不新增会独立漂移的部分迭代计数。B后32000轮最终完成量40000，每环境策略步960000、N=4096时累计转移3932160000；不能把B32000误写成从0开始的训练或重置Adam／课程进度。

### 官方保留／修改及验证接缝

保留RUNNER:119–151的24步采样、returns、update顺序及N×T样本含义。修正RUNNER:160只在整个learn循环结束更新current_learning_iteration，以及RUNNER:164在log中增加tot_timesteps的计数位置；改为每个完整更新成功后统一提交并与日志解耦。RUNNER:156–157的零基文件名、:240的成员iter可错位，不能用于断定真实完成量；快照命名／恢复公式留Q6—Q7。官方ENV:125,141的global_steps及BASE_TASK:104–109的reset零动作step不再驱动训练课程。本题是计数与调用边界修正，不更改PPO优化公式。

验证接缝：无日志与有日志完成量一致；第一轮后I=1／S=24／C=24N；采满但未更新／辅助子步失败不提交；局部reset／多次get_observations／初始化／评估不改训练计数；20主辅子步不误计20轮；乱序或矛盾进度失败。检查7999→8000的更新提交顺序、课程传入的S+j及完整边界同步、reset不清全局时钟、B边界不自动启力；恢复计数和双Adam一致性归Q6—Q7，A11收束具体测试。不运行仿真或业务测试，当前仅设计记录。

**用户答复（2026-10-31）：**在进一步说明与官方的计数位置、快照身份、课程时钟和不完整更新处理差异后，用户答复“同意”。

**已确认结论：**Runner在每次24步采集、returns及20组主辅更新全部成功、存储清空后，统一提交completed_iterations＋1、training_policy_steps＋24、training_transitions＋24N；日志只读，episode局部reset／初始化／评估不推进训练完成量。采用set_training_progress发布(I,S+j)训练时钟，独立于局部episode长度；完整轮次、策略步及样本量按固定N／T互验，半轮失败不提交、不假装回滚或保存为完整快照。A完成8000轮后等待评估与B就绪，不自动启力；初始episode错峰仍留A09，完整保存／恢复与停止写盘留Q6—Q7。计数与课程调用边界修改已在差异记录登记，未修改业务代码。

## Q6 · 完整检查点与学习状态连续的恢复（已确认）

**问题：是否采用完整更新边界的训练检查点，保存并严格恢复模型、双Adam、随机源、学习计数及全局环境训练状态，允许新episode重建，并对不兼容或缺失状态明确失败？**

**推荐答复：**训练续接仅接受本工作明确版本的完整检查点；保存发生在Q5提交后的空rollout边界，同步获取同一版本状态，不与下一次采样／更新并发。恢复属于学习状态连续的分段续训，重建Gym后允许所有环境开启新episode；不承诺恢复PhysX内部接触／求解器缓存、旧episode历史或GPU逐比特轨迹。该承诺沿总体计划已确认边界，不把“恢复随机源”误称为“整个仿真从同一时刻无缝继续”。文件写盘协议、命名及失败处理细节归Q7。

### 检查点字段与所有权

推荐首版format_version=1、checkpoint_kind=training_boundary，使用单个torch.save字典及明确子字典，不建立通用序列化／迁移框架。字段职责如下，最终代码使用与文档一致的命名：

| 字段 | 保存内容／约束 | 必要作用 |
|---|---|---|
| model_state_dict | encoder、Actor、Critic、decoder及14维可训练std的完整state_dict | 恢复策略、估计器、价值和探索分布，不能只存Actor |
| main_optimizer_state_dict | 主Adam动量、步数及param_groups；另存主参数有序名称清单 | 恢复encoder／Actor／Critic／std的主学习历史，验证参数对应而非只看shape |
| auxiliary_optimizer_state_dict | 辅助Adam完整状态；另存有序参数名称清单 | encoder／decoder辅助动量独立保存，不能由主Adam复制或重建为空 |
| learning_rate_state | 主self.learning_rate、辅助实际lr，与两套Adam各param_group相互核验 | Q4自适应lr应从实际值恢复，不能回到主初值1e-3；辅助仍1e-5 |
| scheduler_states | 当前无scheduler对象，明确为空字典 | Q4调度在PPO内部，不为凑字段新增scheduler；将来新增时须另行设计 |
| progress | completed_iterations、training_policy_steps、training_transitions；保存时rollout_step=0／边界标记 | 按Q5核验I、24I、24IN，模型与计数同边界；边界标记不是失败后可伪造的完成证明 |
| rng_state | Python random、NumPy全局、Torch CPU、实际训练所用CUDA设备RNG状态 | 保存随机流位置，不能仅存初始seed；如A09新增独立Generator，必须明确名称与get_state／set_state并纳入，而不是假定全局seed覆盖它 |
| environment_training_state | 部分身份A／B、已确认的全局课程持久状态、创建期随机化实际物性等跨episode保持状态；接口按后文 | 不用仅恢复模型却从课程0开始，或静默重采本应创建期固定的摩擦／质量／执行参数 |
| identity | 任务名、seed、运行purpose、N／T／物理周期／pipeline、依赖与运行版本；固定官方提交及V3源码版本／内容摘要、资产清单SHA256、schema摘要 | 将同维不同意义、不同资产或不同执行模式的不兼容尽早暴露 |
| config_snapshot | 完整有效配置，包含派生值和Q4实际参数／Q3修正标识 | 核查配置，记录复现差异；不能只保存配置文件名或部分命令行 |
| lineage | run身份、当前部分、来源checkpoint的SHA256（从头运行明确为无父来源） | 分段恢复／A→B可追溯；路径是定位信息，SHA256才标识具体来源字节 |

主辅参数名称清单保存各param_group的有序成员，恢复装配须与清单相同后再加载state_dict；Adam按参数顺序关联状态，shape相同不保证模块对应。共享encoder在两套Adam中指向当前模型的同一参数对象，两份优化状态独立。模型strict=True加载，并核验tensor名称／shape／dtype／有限性、std有限正；优化器检查对应关系及已存在状态tensor的shape／有限性，合法的未初始化状态不可无依据当作损坏。缺失必需字段、未知format_version或非training_boundary文件在训练采样前失败，不静默补默认值、strict=False或重置辅助Adam。

schema不只包括1856／294／12／14维度，还包括单帧／历史长度、字段切片／顺序、归一化／坐标系／单位、关节顺序、动作解释、网络及std参数化、监督分组／权重。资产摘要覆盖实际加载的URDF及相关网格／经确认的生成产物；源码摘要来自实际冻结运行内容，单一Git提交号不能掩盖未提交的运行差异。具体摘要清单在实施／运行冻结时生成，本轮不伪造尚未创建的V3哈希。

### 保存／恢复的环境训练状态边界

不保存半轮rollout或正在进行episode的动作历史、episode计时、局部目标插值、当前接触／人工力缓存与sim root／DOF轨迹来承诺原episode继续。它们按A04在新episode重建，跨重启的未结束episode奖励统计不与新episode拼接；新的日志分段记录父检查点及物理reset事实。

需要保存的environment_training_state是跨episode学习状态：已确认的全局课程级别／阶段／时钟、非纯粹由progress和配置可推导的课程统计，以及按创建期生命周期保持的随机化实际属性／参数。A09尚未确定随机化范围与生命周期，因此本题先固定状态责任和接口，A09必须逐项回填字段、tensor shape／单位及应用回读方式，A设计收束前完成；不能写成任意万能env.__dict__转储，也不能把一个空字典称为已保存全部随机化状态。

```text
V3PosForceEnv.get_training_state(self) -> Dict[str, Any]
V3PosForceEnv.load_training_state(self, state: Dict[str, Any]) -> None
```

get_training_state只读并返回独立拷贝；load_training_state在新仿真初始化期间、首次恢复episode reset前使用，核验部分／shape／有效性，恢复跨episode状态，物性按A09施加并回读；不得保存采样目标却未验证实际生效值。创建期固定物性恢复原实际值，不能因重建环境静默重采；按episode重采的参数允许在新episode reset按已确认生命周期重新采样，不宣称恢复了旧episode物性。可由(I,S)纯推导的课程字段只核验，不另行保存矛盾的独立计数源。未定义的状态字段在设计收束前补齐，不在运行时静默忽略。

### 恢复顺序、随机源及设备

```text
读取指定checkpoint到CPU → 检查版本／kind／必需键和元数据
  → 比较有效配置、schema、资产、源码／运行身份及参数清单
  → 构造新环境／模型／双Adam（构造产生的临时随机消耗不算训练）
  → strict加载模型及双Adam，按目标训练设备放置模型／动量并核验
  → 恢复主实际lr、完整学习计数，核对三计数关系
  → 恢复跨episode环境训练状态（阶段、创建期物性、课程等）
  → 恢复checkpoint全部RNG状态；此前构造消耗不改变保存的流位置
  → env.set_training_progress(I,S)，新episode reset／发布首帧
  → 重核空rollout、参数／计数／物性与观测契约
  → 开始下一轮训练采样
```

load_training_state负责恢复持久状态及确定性物性应用，不在此偷偷随机采新episode。全部对象构造完、状态加载完再恢复RNG；新episode初始化与首帧噪声会从恢复的随机源位置消费新的随机数，之后不能再次覆盖RNG来掩盖这些消耗，也不在RNG恢复后重新set_seed。依赖Gym构造的随机物性即使在构造时生成过，也按生命周期被保存值恢复或由新episode的已定义规则重采，不能混用临时构造值。此顺序因此不等同于不中断训练的下一步：新episode reset本来就改变状态和随机数消费。

只恢复checkpoint实际记录且当前运行需要的CUDA RNG设备；设备类型、训练／仿真pipeline、实际使用的CUDA设备集合及依赖版本应按冻结身份核验。逻辑设备路径映射是显式装配步骤，不能只map_location后假定Adam动量与CUDA RNG已正确迁移。若缺少所需设备状态或CPU／CUDA模式改变，不以同一条严格续训接受；另建迁移／分叉实验需单独明确，不能静默降级。普通路径不新增多GPU训练或跨版本兼容层。

### 兼容性、A→B及权重加载的区别

同部分续训推荐严格保持N、T=24、控制／物理周期、schema／资产、优化器结构、训练超参数、seed和运行purpose，以及冻结的学习实现／物理运行身份；除输出路径、日志显示／保存频率等不改变学习语义的明确运行字段外，有差异就启动失败。学习率配置保存初始基线，当前lr由状态恢复；不能以配置初始值覆盖已恢复自适应lr。更改N／T的实验应新建分叉并另行确认预算与计数语义，不称为本阶段原链严格接续。

A→B是已经规划的显式接续类别，不能用同部分身份完全一致来阻止它，也不能随意放宽全部比较。只允许在A完整8000边界、A评估与B准备通过后，使用B07确认的配置／源码差异白名单；schema、资产、模型、双Adam、N／T／时钟及已累计学习计数继续兼容。B09定义的力调度状态初始化／转换须单独确认，不能机械复制A的零力开关，也不能重置全局课程到0。A07保存足以做此核验的源身份、完整状态和父检查点摘要；具体B转换接口／白名单归B07／B09，本轮不提前批准。

纯权重加载用于A10推理／评估时可以只消费模型及schema，不要求对推理恢复双Adam；它与训练resume分入口表达，不提供load_optimizer=False却仍标记“完整续训”的快捷路径。官方B2或旧V3文件缺少本契约必要字段，不能当本阶段完整续训快照；不从文件名猜真实迭代，也不伪造辅助动量／RNG。本题不新增旧文件自动升级器。

### 与官方的差异及验证接缝

官方RUNNER:236–250只保存model_state_dict、optimizer_state_dict、iter和infos，load_optimizer可关闭；没有同段辅助Adam、随机源、全局课程、完整配置／身份及已完成转移字段。新方案保留torch.save／state_dict的直接方式，补齐学习状态并将缺失／不兼容从静默默认改为明确失败；不宣称论文逐项要求此格式，不新增训练损失或优化器。

验证接缝：在完整更新后保存到内存／临时文件，重建同构模型与双Adam并加载；固定同一离线minibatch及随机源，比较模型输出、主／辅动量、当前lr、计数和下一次主／辅更新结果。该算法层往返验证不包含新的Gym reset，不混同物理轨迹连续。另测恢复RNG后Python／NumPy／TorchCPU／所用CUDA及已定义Generator的抽样序列、环境持久物性恢复回读、新episode初始化消费顺序、局部历史清零且全局计数保留；Gym整体仅要求恢复后合法观测并可再完成一次更新，不要求与未重启run逐比特相同。覆盖缺辅助Adam／RNG／身份、错误shape／参数顺序／std、矛盾计数、N或T变化、版本不符、同部分差异、纯权重文件误用、空／半轮buffer以及RNG恢复后被set_seed覆盖。Q7确认保存写盘失败／恢复短测流程，A09回填环境状态字段，A11冻结容差及整体验收；本轮尚未实施或运行测试。

**用户答复（2026-10-31）：**“同意”。

**已确认结论：**采用format_version=1的完整更新空边界检查点，保存完整模型、双Adam及有序参数清单、实际学习率、Q5三项进度、实际随机源、跨episode环境训练状态、有效配置、源码／资产／schema／运行身份与来源。训练严格恢复必需状态，不以纯权重加载、缺辅助默认值或strict=False冒充续训；同部分N／T等学习语义保持兼容，A→B的允许差异由B07明确。装配与状态恢复后恢复RNG，再开新episode；承诺学习状态连续的分段续训，不承诺PhysX／GPU轨迹逐比特连续。A09在A设计收束前补齐课程／随机化状态字段，Q7确定写盘与准确停止；本题无业务代码或运行验证。

## Q7 · 8000轮准确停止、可靠写盘与A／B隔离（已确认）

**问题：是否按已完成迭代准确控制A累计8000轮，保留每200轮检查点并以临时文件核验后原子发布，保存失败明确停止；A冻结运行、B独立代码及环境开发，启动前通过资源与恢复短测？**

**推荐答复：**保留官方每200轮保存频率，但改为Q5已完成量触发、以Q6完整学习状态同步保存。A长训练目标是累计8000，不是恢复后再追加8000。独立输出、显式来源、原子文件发布和冻结隔离属于运行可靠性修正；不新增后台保存服务、自动重启／自动降规模／自动清理数据或训练完成自动切B。

### 预算、停止与接口语义

保持议题01的learn入口语义：`learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None`表示本次额外请求的完整迭代数；初始episode错峰具体行为仍由A09确认，未确认前不能沿用官方脚本True作为默认已批准行为。入口根据已核验检查点计算A剩余预算，正式传`num_learning_iterations=8000-completed_iterations`，不直接将8000作为恢复后的追加量。

- 新A从I=0开始，本次完整A请求8000；从I=3000恢复，完整A剩余请求5000，最终I=8000。
- 开发／preflight可显式请求较短正整数预算，目标为`I+requested_iterations`，且不能超过该部分累计上限8000；超出直接报配置错误，不静默截断请求。正式用途按已冻结完整A预算执行，不能用短测覆盖正式结果。
- I=8000且请求0的A已完成检查只核验明确源checkpoint及其身份，返回不采样、不更新、不新开训练episode；宜由薄入口在创建Gym前检查。请求额外正数或I>8000时失败，不自动延长A或启B；负数／非整数预算也失败。
- 循环按已完成量执行至本次目标，完整更新后先Q5提交、同步环境进度，再判断保存／结束。最终轮提交后不再调用下一次env.step或再执行一轮PPO。
- 最后一轮step内部的局部done reset沿A04继续发生；“准确停止”指不新增训练转移／优化更新，不承诺阻止这一轮正常reset或文件核验所需操作。

A结束须有核验通过的I=8000完整检查点才能报告“训练完成且可恢复”，但这不等于A物理指标达标。A10／A11评估及B准备双门槛仍有效；A完成即停止等待，不因为B代码已准备就自行接续。

### 保存频率、命名与来源

沿官方CFG:218保存间隔200，触发条件改为`completed_iterations % 200 == 0`，并在本次learn目标边界额外保存。若最后目标也恰为200倍数，只执行一次保存，不重复覆写。开发短测结束即使不足200也保存本次完整目标边界。

检查点位于本次运行`checkpoints/`，文件名`model_{completed_iterations:06d}.pt`，例如model_000200.pt／model_008000.pt；文件内Q6的progress仍是身份真值，名字不代替校验。首次正常周期保存是完成200轮，不在完成第一轮后以model_0冒充0轮。A8000从头运行保留200…8000共40个检查点／seed；不自动只保留“最好”或删除旧检查点，A10另定报告／评估选点。

每次恢复创建新的输出运行目录，必须显式提供源文件路径并核验SHA256；不自动选latest、最后目录或根据文件名猜完成量。来源checkpoint只读保留，续训输出不写回来源运行目录。单个输出目录只允许一个训练写入进程；目标文件已存在则报冲突，不覆盖用户文件。父来源路径／摘要、当前完成量和输出目录写运行记录；不额外建立latest软链或数据库。

### 同步保存与文件发布

只在Q5提交完成量、rollout空且无更新进行中的合法边界调用`save(self, path: str) -> None`。训练线程在保存返回前不开始下一轮采样；state_dict不能在后台与优化器修改并发引用。推荐同目录临时文件方案：

```text
验证当前为合法完整边界、目标文件不存在、输出可写
  → 在同一checkpoints目录独占创建本次临时文件
  → 将Q6完整payload同步torch.save到文件
  → flush及fsync文件并关闭
  → 以CPU回读临时文件，核对必需字段、进度、身份及tensor结构／有限性
  → 计算完整文件SHA256
  → os.replace临时文件为目标.pt（同一文件系统）
  → fsync父目录，记录可恢复文件路径／摘要与完成量
  → save成功返回
```

回读核验不构造新训练模型、不恢复随机源、不调用set_seed，不改变训练参数／优化器／课程／RNG；回读只检验快照本身，不能以一次read成功替代Q6独立重建／再更新短测。原子replace只在文件系统支持的同目录发布语义下使用；显式单写入和目标存在冲突检查不扩展成并发保存框架。

写入、同步、回读、摘要或发布失败都抛出明确异常并停止本次训练，不继续采集。保留此前已发布检查点，临时文件不得当成可恢复产物；诊断记录失败路径、操作、errno及内存已完成量／最后确认保存量。仅能清理本次创建的未发布临时文件；不删除既有checkpoint或其他运行数据。

若replace已成功而目录fsync或随后输出记录失败，新.pt可能已经可读，但本次保存不能报告为持久成功；不通过删除／回滚已发布文件伪造“从未保存”。保留文件并标明失败位置，后续从显式路径重新校验完整内容与身份，再决定是否可用。原子可见不等于断电／网络存储永不丢失，不增加未实现的跨主机持久性承诺。

周期保存失败时，模型内存可能已完成例如第200轮，但最后确认保存仍为此前边界；不能将后者改记200。A8000最终保存失败也必须报告“已完成8000次更新但最终可恢复产物未确认”，并以失败退出，不将它混同“训练与保存均完成”。异常中不尝试把半轮模型应急写成完整快照；意外中断仅依已发布完整checkpoint恢复，当前不新增自动重启或半轮恢复机制。

### A冻结与B隔离的具体位置

沿01已确认的独立仓库、远程项目根及环境路径，推荐以下命名；路径在实施前做容量／存在性核验，当前不创建目录或分支：

| 用途 | 本地 | 远程 |
|---|---|---|
| A开发及冻结运行源码 | `P/UniFP_v3/`，分支v3-stage1-a，A实现／验证后固定提交 | `/mnt/data_8THDD/myl_unifp_v3_stage1/UniFP_v3/`，固定同一A提交 |
| B隔离开发源码 | `P/UniFP_v3_b/`，由A冻结提交建立Git worktree，分支v3-stage1-b | `/mnt/data_8THDD/myl_unifp_v3_stage1/UniFP_v3_b/`，需要远程开发／短测时独立同步 |
| A运行环境 | 本地tavis不因训练修改 | 已确认`/mnt/data_8THDD/conda_envs/unifp_v3_stage1`，绑定A源码并冻结依赖 |
| B开发运行环境 | 按实际本地用途核验 | 独立`/mnt/data_8THDD/conda_envs/unifp_v3_stage1_b`，绑定B源码；不能在A环境pip install -e重指B |
| 训练／评估输出 | 回传按run身份保存 | 沿既有runs根，每次运行新目录，目录身份含part／purpose／seed及唯一运行标识；下设checkpoints／日志，精确指标子目录由A10收束 |

P为项目根`/home/sorry/python_ws/interactive/unifp_reproduction`。A冻结运行后，不同步覆盖其源码／配置／资产，不切换A工作树分支，不更新A依赖或editable绑定；缓存／输出放01已确认位置，不让B工作区共享可修改的A运行文件。Git对象共享不意味着运行工作树共享，A／B资产内容变化也必须各自可追溯。B的新环境可参考A冻结依赖但须独立验证，不要求本轮装配。

A／B不得因同时开发就默认并发占用同一GPU；B的无GPU代码／单元开发可进行，真实Gym短测须重新核验当时资源，不干扰A。具体可用设备以实际运行时证据为准，不将历史A6000空闲情况当预留，不强行终止其他进程或清理服务器数据。

### 启动资源核验与恢复短测

长训练前沿01／A11核验所选GPU身份、显存、主机内存、输出及cache所在文件系统可用空间／配额／inode、目录写入与临时文件原子发布能力，以及解释器／Gym／legged_gym实际导入位置。用同一冻结配置、相同N的真实采样＋20组主辅更新测峰值；不因失败静默降低N或改minibatch数。A11确认预检长度、物理／学习容差；本题不伪称已完成容量验证。

磁盘估算用preflight产生的实测完整checkpoint体积S，至少覆盖剩余周期／终点检查点、一个额外临时文件、A10确认的日志／trace预算与cache需求，并记录明确余量。写盘前还需检查当时可写性；预检不能保证共享服务器之后永不满盘，运行中仍按实际保存错误失败。没有日志预算或实际检查点大小时不能以历史空闲TB数断言足够，不为本题增加守护进程或自动删除策略。

A启动长训前必须完成两层恢复验证：

1. 算法层：Q6独立重建模型／双Adam、固定离线batch／随机源，检查下一主辅更新一致性；具体容差在A11冻结。
2. 真实Gym层：preflight独立run先完成完整更新并保存，销毁该进程，再在新进程按显式checkpoint路径恢复并至少再完成一次完整更新、保存／回读成功；核对三计数、双Adam、当前lr、实际物性、schema、RNG恢复／新episode重建顺序和数据有限性。具体前后短测轮数与物理判据由A11确认。

preflight输出与正式／development长训输出分开，purpose不同不能以Q6“严格续训”把preflight权重偷渡为正式初值。计数／停止单元样例可把fake runner进度设到7999验证恰好一轮和8000 final保存，但不能伪造真实Gym checkpoint的学习完成量来证明长训已完成。A8000结果分析仍由A10／A11执行，本题通过只表示运行规则确认。

### 官方对照与验证接缝

保留CFG:218的200轮间隔、RUNNER:149–157的更新后保存组织及torch.save直接格式。修改零基命名／成员iter错位、额外追加预算混淆与直接写最终文件风险；新实现同目录临时文件发布、显式来源、独立输出、失败停止及冻结隔离。不将工程可靠性修正描述成论文新增算法或已有收敛性提升。

验证接缝：I=0／3000／7999／8000的请求和remaining关系，超预算／负数失败，200倍数与final去重，最终更新后不再step；序列化／flush／fsync／回读／摘要／replace／目录fsync各阶段失败保留前检查点，现有目标冲突不覆盖，临时文件不被选为resume，save不改RNG；关闭日志仍正确完成／保存。用临时目录和fake runner验证文件发布与预算，不先跑8000轮来测试停止分支；Gym跨进程恢复与目标N容量由A11收束。核对A／B导入路径、冻结源码指纹、独立环境与运行输出，资源不足不降级。当前仅设计方案，未创建V3代码／环境或tickets，未做远程修改或启动训练。

**用户答复（2026-10-31）：**“同意”。

**已确认结论：**沿额外请求语义计算A剩余量8000−I，超预算明确失败，准确停在已完成8000且不自动启B；保存间隔200及本次终点保存、完成量命名且去重。同步完整检查点通过同目录临时文件、flush／fsync、CPU回读、SHA256、原子发布及目录同步确认；保存失败停止、保留旧产物并区分内存完成量与已确认保存量。显式来源／独立输出、A冻结提交／资产／环境、B独立worktree／运行环境，以及目标N容量与跨进程恢复短测均按本节执行。路径为待实施位置，本次未创建目录／环境、未写业务代码或启动远程操作。

## 当前与目标调用关系及完整接口框架

当前基线固定为`UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/`中的PPO、RolloutStorage和OnPolicyRunner；实施位置在尚未创建的`UniFP_v3/`对应学习文件及`envs/v3/v3_pos_force.py`。只读官方／旧归档不修改。A07是学习与运行模块，不负责重定义A04—A06观测／动作／监督或A08奖励。

### 方法边界、数据契约与失败语义

所有浮点输入float32、同明确设备；无梯度阶段与带梯度阶段按Q1—Q4分开。T=24、N为并行数、M为minibatch大小、K为全部done数、L为纯timeout数。坐标／单位由A04—A06定义，学习器保存的是已缩放网络数据；reward／returns／优势是任务回报量，不混作米或牛顿。

| 接口／调用方 | 输入输出及职责 | 状态／失败／验证接缝 |
|---|---|---|
| Runner构造→PPO构造 | 读取独立有效algorithm配置，建立同构ActorCritic和双Adam；完整签名见下方 | 模型主辅有序参数集合按Q4，构造不将episode reset误计训练或提前恢复RNG |
| PPO.init_storage | N、T、obs_shape=[1856]、critic_obs_shape=[294]、obs_pred_shape=[12]、action_shape=[14]→None | 一次预分配，step=0、returns_ready=False；非法维度／设备失败 |
| PPO.act | obs[N,1856]、critic_obs[N,294]、obs_pred[N,12]→原始采样actions[N,14] | 无梯度旧状态／动作／联合logp／分布／V暂存；不消费decoder预测回灌Actor，std非法失败 |
| 环境step→PPO.process_env_step | rewards[N]、dones long[N]、infos终止契约→None | 同刻配对，按ids选终止快照求纯timeoutV、克隆奖励补偿一次后入库；缺失／错配／非有限失败 |
| RolloutStorage.add_transitions | Transition→None | 全字段copy_到[step]，成功后step＋1；缺字段／错shape／溢出失败，旧数据不可变 |
| PPO.compute_returns | last_critic_obs[N,294]→None | 根据storage最后done选存活行评价last_values[N,1]，交storage计算；无梯度，完整24行才允许 |
| RolloutStorage.compute_returns | last_values[N,1]、gamma、lam→None | Q3倒序GAE／returns，Q4全rollout样本std标准化，全部成功才returns_ready=True |
| RolloutStorage.mini_batch_generator | num_mini_batches=4、num_epochs=5→11项Tensor批迭代器 | 只对完整／ready存储使用，同索引、一次排列跨epoch；正整数／整除校验；末项None |
| PPO.update | 无外部新数据→四项损失均值tuple | 20次主后辅，旧targets无梯度，清理storage；失败不提交完整轮次，保留诊断 |
| Runner.learn | 本次额外完整迭代数、初始错峰开关→None | 完整更新后提交Q5计数，按Q7周期／终点同步save，达到目标不再采样；错峰尚待A09精化 |
| Runner.save | 明确新目标文件路径→None | 同步Q6完整payload、Q7文件发布；不改变RNG或覆盖既有文件，写盘失败停止 |
| Runner.load | 明确源checkpoint路径→None | 仅严格完整训练恢复，核对身份并恢复模型／双Adam／RNG／计数／全局环境状态，新episode重建；取消可选跳过优化器的训练恢复模式 |
| env.set_training_progress | completed_iterations:int、training_policy_steps:int→None | 无物理／采样／观测副作用，仅发布Q5时钟，rollout中为(I,S+j) |
| env.get_training_state／load_training_state | 独立Dict[str,Any]／state→None | 跨episode课程／创建期实际物性，字段按A09回填；不是sim全状态或env.__dict__序列化 |

get_statistics若保留使用，只在局部clone标记统计段尾，禁止改变真实dones。纯权重推理入口由A10承接，直接加载同构模型及schema，不调用Runner.load假装完整恢复。工厂恢复调用须避免构造期提前执行旧reset后又把它当训练首步；训练入口的零剩余已完成检查在Gym构造前处理。

### 完整签名与最小调用伪代码

以下为设计职责伪代码，含完整参数／返回类型，不含业务实现；不是新增可运行源码。Tensor为torch.Tensor，Transition为既有RolloutStorage.Transition，VecEnv为V3实际四返回协议；BatchTuple为下方11项别名，仅用于文档说明，不要求另建公共框架。

```text
BatchTuple = Tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    Tensor, Tensor, Tensor, Tensor, None,
]

OnPolicyRunner.__init__(
    self, env: VecEnv, train_cfg: Dict[str, Any],
    log_dir: Optional[str] = None, device: str = 'cpu',
) -> None
OnPolicyRunner.learn(
    self, num_learning_iterations: int,
    init_at_random_ep_len: bool = False,
) -> None
OnPolicyRunner.save(self, path: str) -> None
OnPolicyRunner.load(self, path: str) -> None

PPO.__init__(
    self, actor_critic: ActorCritic,
    num_learning_epochs: int = 1, num_mini_batches: int = 1,
    clip_param: float = 0.2, gamma: float = 0.998, lam: float = 0.95,
    value_loss_coef: float = 1.0, entropy_coef: float = 0.0,
    learning_rate: float = 1e-3, max_grad_norm: float = 1.0,
    use_clipped_value_loss: bool = True, schedule: str = 'fixed',
    desired_kl: Optional[float] = 0.01, device: str = 'cpu',
) -> None
PPO.init_storage(
    self, num_envs: int, num_transitions_per_env: int,
    actor_obs_shape: Sequence[int], critic_obs_shape: Sequence[int],
    obs_pred_shape: Sequence[int], action_shape: Sequence[int],
) -> None
PPO.act(self, obs: Tensor, critic_obs: Tensor, obs_pred: Tensor) -> Tensor
PPO.process_env_step(
    self, rewards: Tensor, dones: Tensor, infos: Dict[str, Any],
) -> None
PPO.compute_returns(self, last_critic_obs: Tensor) -> None
PPO.update(self) -> Tuple[float, float, float, Dict[str, float]]

RolloutStorage.__init__(
    self, num_envs: int, num_transitions_per_env: int,
    obs_shape: Sequence[int], privileged_obs_shape: Sequence[int],
    obs_pred_shape: Sequence[int], actions_shape: Sequence[int],
    device: str = 'cpu',
) -> None
RolloutStorage.Transition.__init__(self) -> None
RolloutStorage.Transition.clear(self) -> None
RolloutStorage.add_transitions(self, transition: Transition) -> None
RolloutStorage.compute_returns(
    self, last_values: Tensor, gamma: float, lam: float,
) -> None
RolloutStorage.mini_batch_generator(
    self, num_mini_batches: int, num_epochs: int = 8,
) -> Iterator[BatchTuple]
RolloutStorage.clear(self) -> None

V3PosForceEnv.set_training_progress(
    self, completed_iterations: int, training_policy_steps: int,
) -> None
V3PosForceEnv.get_training_state(self) -> Dict[str, Any]
V3PosForceEnv.load_training_state(self, state: Dict[str, Any]) -> None
```

保留官方构造默认值，实际训练由Q4已确认的有效配置kwargs注入；生成器显式传入(4,5)。构造默认不等于实际训练参数，不为整理接口额外改变默认值。PPO.update四项返回依次为mean_value_loss、mean_surrogate_loss、mean_auxiliary_loss及各监督组损失均值，保持官方tuple组织、不新增指标管理类。主lr／KL／grad等日志字段由A10确认。正式运行必须提供独立输出位置；log_dir可空仅支持算法层隔离测试，不代表允许无保存目录的长训。

```text
train_v3入口读取配置与显式源身份
  → 若源已完整A8000且额外预算0：仅核验后返回
  → 创建冻结版本的V3环境、ActorCritic、Runner／PPO／Storage
  → 新训练：设进度0，按A04 reset与发布首帧
    或续训：Runner.load按Q6装配／恢复状态／RNG，再新episode reset
  → get_observations只读得到obs／critic／label
  → Runner.learn(本次额外完整迭代数):
      读取已提交I／S／C
      24次策略循环：
        env.set_training_progress(I,S+storage.step)
        actions = PPO.act(动作前obs, critic, label)
        next_obs, rewards, dones, infos = env.step(actions)
        PPO.process_env_step(rewards,dones,infos)
          → 纯timeout终止子批evaluate → 奖励单次补偿 → copy_入库
        obs／critic／label更新为next_obs对应字段
      PPO.compute_returns(最后真实后继critic)
        → storage.compute_returns(Q3／Q4)
      PPO.update():
        storage.mini_batch_generator(4,5)
        → 每批主损失backward／clip／Adam
        → 重新辅助前向／MSE／Adam
        → 平均损失／clear
      Runner提交(I+1,S+24,C+24N)，env只读时钟同步
      → 只读日志，周期／终点save完整checkpoint，失败停止
      → 达到本次目标返回；A8000不再采样，不自动切B
```

## 收束依赖与验证责任

A07-Q1—Q7已确认，以上是已确认结论的接口整理；新增事实或与后续模块冲突须回写并确认，不能静默改变结论。模块级收束不等于运行验证通过或已可开始编码。

| 后续模块 | 必须回填的细节 | 与A07的衔接 |
|---|---|---|
| A08 | 奖励分项／系数、几何与零力补偿目标 | 环境输出单reward，PPO结构及Q3奖励副本语义不变 |
| A09 | 全局课程／随机化状态的逐字段名称、shape／单位／生命周期／施加回读、独立Generator（若有）、初始episode错峰 | 完成get／load_training_state的可实现字典契约，不为空泛状态；保持Q5训练时钟与Q6恢复顺序 |
| A10 | 指标键／频率、trace与日志容量、评估权重入口、源模型选取／报告对照 | 训练原reward与补偿后训练reward分清，辅助日志64倍修正披露，检查点身份按完成量 |
| A11 | 算法单元／文件故障／Gym跨进程恢复测试的具体容差及短测长度、同N容量与A8000验收 | 测试最后收束，Q1—Q7各验证接缝逐项覆盖；设计和正式tickets确认后才实施 |
| B07／B09 | A→B源码／配置差异白名单与课程状态转换 | A学习状态及schema兼容，来源只读，I8000→40000不重置优化器／学习时钟 |

## A／B衔接

A先通过保存→重建→恢复→再更新的短测；B不能默认官方save已包含辅助优化器／RNG／课程。

## 失败语义与验证范围

不完整状态、错误迭代或不兼容身份在采样前失败；写盘失败保留诊断，不能把未保存当作已完成可恢复。

验证范围：小型可解析轨迹验证GAE／timeout／数据不被覆盖；完整状态往返与后续更新；B增加8000边界和不兼容拒载测试。

## 讨论记录

A04—A06已确认并收束；A07-Q1—Q7全部确认并完成模块总结、完整接口与调用框架整理；Q3修改点单独保留，A08-Q1a已确认、Q1b三项目标域边界暂定用于验证，Q1c已确认、时长待动态验证，Q2已确认，Q3a—Q3c已确认，Q4a时间缩放已确认，Q4b主任务数值已确认为验证候选，Q4c执行器数值已确认为验证候选，当前进入A08-Q4d稳定／接触数值。A09／A10／A11跨模块回填见上表，仍未修改业务或启动训练。已确认的主／辅助更新边界继续有效，不将修正清单视为后续方案获批。00—03已确认结论继续有效（03新增数值仍按原流程批准／验证）；本模块逐项记录论文／固定源码对照，并在全部A设计及实施票据确认后才编码。


## 票据05实施回填（2026-10-31）

### 实际模块与调用链

V3使用独立`V3RolloutStorage`／`V3PPO`／`V3OnPolicyRunner`／`v3_run_identity`文件，保留本次不改的原B2学习链。前述伪代码中的PPO／Runner名称在实际V3入口对应以下类；模块拆分只是限定V3终端协议的作用范围，不新增算法分支。

```text
scripts.train_v3.main
  → resolve_v3_configs → build_run_identity → 显式resume身份预检
  → V3PosForce → V3OnPolicyRunner(env, train_cfg, new_output, identity, snapshot, device)
    → V3ActorCritic(1856,294,12,58,14) → V3PPO → V3RolloutStorage
  → [runner.load(explicit_path)] → runner.learn(additional_iterations)
    → 24 × {set_training_progress → alg.act → env.step → alg.process_env_step}
    → alg.compute_returns → alg.update(5×4组主后辅) → 提交I/S/C
    → flush_episode_records → iterations.jsonl/TensorBoard → runner.save
  → completion.json → destroy_sim
```

帧／辅助宽度读取`env.cfg.env.num_single_obs／num_pred_obs`，不假设BaseTask暴露这两个属性。运行身份中的Gym枚举使用字符串，实际引擎仍显式要求PhysX。运行命令设置`PYTHONUTF8=1`以读取中文资产来源清单。

### 实际恢复与失败语义

`V3OnPolicyRunner.validate_checkpoint(payload, algorithm, identity, num_envs)`同时服务于保存CPU回读、严格load及A8000零预算核验。format_version=1／training_boundary；主／辅有序参数清单分别25／14个参数张量，I>0时每个张量必须有完整Adam动量。空scheduler字典表示本实现没有独立scheduler对象；KL控制的主lr和辅助实际lr单独保存。

`save(path)`只接受本run的全新`.pt`路径；临时文件→文件fsync→回读／完整性核验→SHA256→原子发布→目录fsync→成功清单，全部成功才更新last_saved。`load(path)`要求新输出、显式父文件，恢复持久物性和双Adam／RNG后创建新episode；目录已有metadata则不能再次load。失败诊断保存部分rollout位置、已完成量、最后确认保存量和未落盘episode，不尝试从半轮续训。

### 已完成验证与剩余责任

本机19项pytest＋5项奖励／状态unittest通过；7999→8000为精确停止fixture，CPU离线恢复后下一更新参数逐位一致，不是实际8000轮成果，也不承诺PhysX逐位连续。远程最终r06训练10轮与独立r03恢复1轮通过；原先的首步物理异常已按补充Q8修复，恢复首步0终止。失败／诊断历史和最终结果以[实验记录](../../../../UniFP_v3/docs/experiments/2026-10-31_v3_ppo_runner_checkpoint.md)为准。完整4096负载、候选动态稳定性、A10物理阈值及正式启动仍由06／07承接；本模块完成不等于复现效果通过。


## 补充Q8 · 创建级物性须在prepare_sim之前恢复（已确认）

**已确认事实：**当前load在`prepare_sim`后重新写入创建级物性；相同512环境、零动作下仅重复该物性回写也能触发首步512个底盘接触失败。官方Preview4文档明确GPU pipeline不支持运行时改COM。将同一存档物性在actor创建阶段施加的进程内原型，两步均无终止。详见知识库第61节和实验记录；不是改变物理候选参数的依据。

**问题：是否将创建级物性的恢复提前到环境构造阶段，并让运行时load只核验创建物性，再恢复其余学习／可变环境状态？**

**推荐答复：**同意。保留原checkpoint字段和全部质量、COM、惯量、摩擦、随机化候选及PPO参数，调整调用时序。A07入口已有CPU预读payload，在身份检查后提取`mass_com_delta[N,5]`和`friction[N,1]`，经可选参数传给V3环境；不把这些张量混入JSON配置或另造一份物性来源。构造环境时收集实际名义body物性，再按存档质量／COM／摩擦施加并回读，时机在BaseTask的prepare_sim之前。首次训练不传该参数，维持既有创建随机化。

### 接口与模块边界

```python
V3PosForce.__init__(
    self, cfg: Any, sim_params: Any, physics_engine: Any,
    sim_device: str, headless: bool,
    creation_properties: Optional[Dict[str, Tensor]] = None,
) -> None
# creation_properties=None: 按原A09创建随机化。
# 非None: 仅含mass_com_delta[N,5]（kg,m,m,m,kg）和friction[N,1]（无量纲），CPU可读。
# 仅在actor创建阶段使用；不携带模型、Adam或RNG。

V3PosForce.load_training_state(self, state: Dict[str, Any]) -> None
# 在prepare之后只核验当前创建级实际回读缓存与存档一致；不再调用原生body/shape物性setter。
# 恢复执行系数、push、训练时钟及RNG，仍不reset或step。
# 物性不一致时明确失败，提示须在构造阶段传入；不静默重采样或接受错误物性。
```

原生物性回读已在创建阶段核对每个body与shape；入口到load之间不推进物理。由同一Actor构造的实际缓存完成load比对，模型／优化器仍由Runner负责。创建级物性的生命周期不改成episode级随机化。CPU与GPU采用同一构造／恢复协议，不增加互不一致的后端分支。

```text
resume_payload = CPU读取显式checkpoint → 检查源码／资产／配置／schema身份
creation_properties = payload.environment_training_state中的mass_com_delta与friction
env = V3PosForce(..., creation_properties=creation_properties)
  → create_actor → 获取名义物性 → 应用存档创建物性 → 全部原生回读校验
  → prepare_sim → 创建张量与观测缓存
runner = V3OnPolicyRunner(env, ...)
runner.load(checkpoint)
  → 完整checkpoint核验 → 模型／双Adam／实际lr
  → env.load_training_state：核验创建级物性，恢复其他环境状态
  → 恢复Runner RNG → 新episode reset → 正常24步采样／主辅更新
```

### 实施与验证（已完成）

1. 修改`legged_robot_v3.py`、`train_v3.py`上述接口与时序；保持源身份严格检查，新源码使用全新run，不改旧checkpoint身份绕过校验。
2. 在既有创建／load／reset公开接缝验证存档物性创建时实际施加、prepare后不再回写；原`check_v3_reward_runtime.py`中运行时篡改shape再load的CPU管线样例改为新进程／新仿真恢复对照，避免继续验证GPU不支持的语义。算法及文件23项回归保持有效，不增加无关边界穷举。
3. 最终源码重新执行远程512×10及新进程恢复1轮；I/S/C、主辅Adam200→220及父哈希不变，并新增恢复首步物理原因核查，确认不再全环境虚假触地。普通未训练策略引发的后续真实失败如实记录，长期稳定性仍由A11负责；不增加隐藏预热、放松终止或关闭随机化。

**用户答复（2026-10-31）：**“同意，修正恢复时序并重测（推荐）”。

**已确认结论：**采用上述可选creation_properties构造接口、创建前物性施加和load只核验职责；保持存档字段、数值、PPO及无隐藏物理步约束不变，按既有创建／load／reset接缝补回归，实施后重新完成512×10＋跨进程1轮验收。


### Q8最终实施证据

`legged_robot_v3.py:147–150,266–340`实现构造时施加存档创建物性及load只核验；`train_v3.py:84–90`显式传入两个存档创建字段。19项pytest＋5项奖励／状态unittest通过，远程r06的512×10及r03独立恢复1轮通过：I10→11／S240→264／C122880→135168，25个主Adam和14个辅助Adam参数张量的step均200→220；mass_com_delta／friction逐元素相同，父SHA和源码／资产身份一致，恢复首步物理终止0。原生物性setter仅在prepare之前执行，无隐藏预热step。补充的8环境27步checker也通过push与状态／timeout合同。普通短训练策略后续真实物理失败不被掩盖，仍交A11验证候选动态稳定性；不据此认定论文复现有效果或批准长训。
