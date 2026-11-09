# B07 · PPO、缓冲区与Runner

状态：2026-11-07 B07-Q1—Q4全部获用户确认，本模块设计收束；B04—B07已确认，下一项B08。全部B设计确认后才拆正式票据并编码。

## 模块收束摘要

已确认Q1—Q4：本次seed1从固定V6 r2完整A8000父状态显式进入B，保留网络／双Adam／实际lr、创建级物性及累计训练与push时钟；物性在actor创建时恢复，构造结束后恢复RNG，随后只开启一次新episode、末端力从0N起。24步rollout和主辅更新保持，B新增32000轮、累计40000轮停止，每200个完整更新及请求终点保存。B→B恢复完整学习和课程状态，但新episode不续演旧局部力事件或PhysX缓存。Q4将这些接口收束为独立远程B输出与来源记录、普通物理终止局部reset、数值／执行／保存异常退出并保留上个有效检查点；不新增物理失败全局停训或自动重启。以上整套规则已获用户确认；尚未修改B业务代码或启动训练。

## 当前基线与目标

当前Runner只接受A、最大8000，checkpoint校验也限定A边界；load在8000不再reset。目标新增显式A→B迁移和B→B恢复，完整继承学习状态并在新episode开启B。

继承V6 r2 A8000、14动作、58×32 Actor历史、98×3 Critic历史、12辅助标签、15命令槽和2.5ms×8控制；不切V8、不重跑A、不进入轮式创新。来源和父SHA见[总体计划](../../development-plan.md)；当前源码依据见[research](../../knowledge/research.md)。

## 按依赖讨论的决策

Q1：父模型固定V6 A8000路径和SHA；A→B允许差异逐字段列出，区分schema／资产不可变项与B源码、课程、输出目录等允许项。禁止strict=False或跳过全部identity校验。

Q2：恢复模型、双Adam动量、实际lr、RNG、物性及已完成计数；物性仍在actor创建时恢复，GPU prepare后不能重写COM。新B力事件如何初始化由B04/B09定义。

Q3：已确认增量32000、累计40000，24步rollout和主辅更新保持；完整更新边界保存学习状态、课程时钟与force_state快照，B→B重建新episode时清除旧局部力事件，不承诺恢复PhysX轨迹。

Q4：普通物理终止局部reset继续；NaN和执行异常退出。不再引入“首次物理失败全局停训”。A父输出只读，B全新目录，明确不恢复PhysX内部缓存。

推荐的最小接口／数据流框架（待具体签名确认，无业务实现）：

```text
parent = load_checkpoint(explicit_path)
validate_stage_transition(parent, target_identity, approved_differences)
env = V3PosForce(creation_properties=parent.physics_properties)
runner.restore_learning_state(parent)  # model + both Adam + lr + RNG + counters
runner.begin_part_b_new_episodes()
runner.learn(num_learning_iterations=40000 - runner.completed_iterations)
```

## 状态、失败与验证边界

本模块依赖前序B模块已确认接口，B04依赖A公共基线；测试最后由B11分别收束单元与整体部分。N为环境数，B为actor刚体数；力N、力矩N·m、位置m、时间s，yaw系指仅随底盘偏航旋转的坐标系。新增状态归属、更新时刻、reset子集隔离、持久化及异常语义在对应Q&A确定。

普通物理失败沿用官方局部reset并记录；不兼容checkpoint在启动核验时退出，NaN、非有限状态及执行异常在发现处退出，不把未完成主辅更新提交成完整轮。验证必须保留非零力样本，A的零标签结果不能替代B验收。方案不引入新策略头或专用力控器。

## 讨论记录

已确认：用户选择V6直接进入B，并保留V8诊断证据；B04—B06及本章Q1—Q4全部确认。用户先回复“同意，B07可以一起讨论”确认Q3并授权合并剩余内容，随后回复“同意”确认Q4整套方案。开头模块收束摘要覆盖本章全部结论；下一模块为B08，未开始业务实现。

### Q1：是否使用显式A→B迁移，固定本次父检查点并逐项记录允许差异？

状态：2026-11-07用户回复“同意”，确认以下父身份与显式阶段迁移边界；未修改业务代码。具体恢复顺序、物性／随机数与新episode初始化留Q2，预算／保存／B→B恢复留Q3，运行隔离与异常留Q4。

推荐：本次seed1主线固定以下已验收父文件，不自动搜索latest，不以1400轮／其他资产权重替代。该绑定仅针对当前seed1，未来其他seed须分别绑定自己的有效A8000父状态，不把同一父模型复制成多个独立训练seed结论。

```text
parent_checkpoint:
  /mnt/data_8THDD/myl_unifp_v3_stage1/runs/a8000_seed1_v3_v6r2_resume0767_20261105/checkpoints/model_008000.pt
parent_sha256:
  323a6688fe42559db1d0c545b778b456551ce0ac472f8169fec5ebce2d849dbc
parent_source:
  /mnt/data_8THDD/myl_unifp_v3_stage1/validation/a8000_repairs_20261104_r03/source/UniFP_v3
parent_source_git:
  6f93215118373445edd38c96b70aa5c88f17829d
```

父文件应为完整training_boundary，I=8000、S=192000、C=786432000（4096环境），rollout_step=0；模型、主辅Adam及其参数顺序／slot、实际lr、RNG、物性及环境持久状态按既有完整校验核验，不用只含model权重的导出文件替代。文件SHA校验确认具体父字节，来源记录确认其A阶段身份；运行时重新核验实际读取文件，不把历史路径存在当作实时验证。正式本次沿用4096环境，不在阶段迁移时顺便改采样规模或环境映射；小规模验证使用独立测试父状态，不冒充此正式父文件的完整物性恢复。

当前`train_v3.py`在构造环境前要求identity整体相同，Runner／checkpoint也限制A及8000；B业务源码和有效配置改变后不能直接按现有同阶段resume通过。建议保留同阶段恢复的完整校验，新增明确命名的A→B迁移模式，先确认父A，再核验目标B身份和允许差异；普通resume不自动猜测阶段，不用`strict=False`加载模型，不用移除全部identity比对来放行。

允许差异按含义归为以下范围，最终字段路径及源码文件清单在B08—B11确认和实施后冻结，不能把本项当作任意源码修改白名单：

| 必须保持的兼容项 | 本次阶段迁移允许的目标差异 |
|---|---|
| V6 r2有效资产字节及顺序、14动作与关节映射 | B已确认非零力事件、COM施力与reset扩展 |
| 网络参数名称／shape、58×32／98×3／12标签／15命令顺序及物理意义 | 既有力槽由零变非零，补偿目标按已确认接口填值 |
| 20ms策略步、2.5ms×8、24步rollout、原控制与碰撞语义 | reproduction_part从A变B，接续预算后32000／累计40000 |
| 主辅优化器参数边界、PPO／辅助损失基线及已完成计数 | 新B运行目录、阶段lineage、已确认B课程／reward／评估配置和相关源码 |
| 创建级物性与当前完整学习状态可恢复性 | 存档格式／字段为表达B事件与阶段状态所需的明确版本变化 |

CPU/GPU测试、物理设备编号或目录变化不靠伪造旧runtime身份解决：真实运行环境照实记录；正式采用现有远程Gym／Torch环境，具体允许的runtime差异在迁移校验实施中列清。若存在超出已确认清单的算法、资产、schema或依赖变化，应报告具体字段冲突，不静默按目标配置覆盖父状态。检查集中在加载／启动边界，训练step不重复扫描源码或全量hash。

迁移记录保存父路径与SHA、父／子source和配置摘要、明确的差异列表、恢复模式与累计起点计数。A父输出只读，目标B为独立身份；B新增计数可从0展示，但累计计数从8000继续，不能把接续训练标成从零独立训练。创建级物性恢复仍使用A已修正的创建时接口，不通过identity豁免绕过GPU COM约束。

官方对比：保留先无力后受力并延续学习参数／优化过程的核心目标。显式A/B目录、父SHA和允许差异清单来自本项目将官方课程拆为独立开发／训练阶段的需要，不是官方额外要求或新的学习算法。它用于准确接续已验收状态，不增加每轮门控或普通失败全局停训。

最小接口职责（伪代码，具体命令参数在拆票时映射现有入口，不含业务实现）：

```text
parent = read_explicit_checkpoint(path)
validate_complete_parent_A(parent, expected_sha, expected_progress)
transition_report = validate_A_to_B(parent.identity, target_identity, approved_transition_manifest)
# 同阶段B恢复使用独立的same-stage校验，不能复用A→B的差异豁免。
restore_complete_learning_state(parent)  # 实际顺序与新episode在Q2确认
write_parent_and_transition_lineage(transition_report)
```

验证接缝：有效完整A父可迁移；模型shape／动作顺序／有效资产改变、非8000父或缺Adam slot等真实不兼容给出明确原因；已批准的B课程／代码差异不被旧A整体identity相等要求误拦；同阶段恢复不意外继承A→B豁免。复用现有checkpoint校验，只增必要阶段差异处理，不建立通用迁移框架。B11确定最终单元与真实跨进程验收。

来源：`UniFP_v3/legged_gym/scripts/train_v3.py:38-70,96-103`、`b2_gym_learn/ppo_cse_pf/v3_on_policy_runner.py:43-58,364-408`、`v3_run_identity.py`及统一进度文件第2.1节。当前未访问远程父文件、未修改业务代码或启动B；父SHA为已归档实测记录。

### Q2：是否完整恢复学习／创建级状态，然后用恢复的随机流开启一次新B episode？

状态：2026-11-07用户回复“同意”，确认以下恢复顺序与新B episode语义；未修改业务代码。物理失败／timeout沿用B04，B事件幅值与时间参数留B09；本项不新增自动重置学习率或优化器的策略。

推荐区分三类状态：

| 类别 | 状态 | A→B处理 |
|---|---|---|
| 学习状态 | encoder／Actor／decoder／Critic／std、双Adam step／一阶和二阶动量、实际学习率 | 完整恢复，不重新初始化 |
| 持久状态 | 创建级质量／COM／摩擦、累计I／S／C、底盘push时钟及去重标记、Python／NumPy／Torch／所用CUDA随机流 | 沿用父状态，按既有职责恢复 |
| episode局部状态 | root／DOF运动状态、动作与观测历史、局部计数、执行系数、目标段、两路力事件及力缓存 | 按新B episode重新初始化；执行系数和任务样本从恢复的随机流重采，力从0N开始 |

两套Adam共享encoder但各自拥有独立的step与动量，必须分别恢复，不能将一套slot复制给另一套。学习率使用检查点实际值，同时恢复优化器param_groups中的lr与主算法learning_rate，不用构造函数默认值覆盖；辅助Adam亦恢复实际存档lr。主KL自适应调度继续基于接下来的真实更新，不在进入B时自动回升、warmup或重建scheduler。当前具体存档lr待实施时回读核验，不凭历史常量猜测。

恢复顺序建议如下：

1. 在Gym创建前读取明确父文件并完成Q1父身份／允许差异核验，提取创建级`mass_com_delta[N,5]`与`friction[N,1]`。
2. 用既有`creation_properties`构造环境，创建actor时施加父物性，早于prepare_sim；构造模型与学习器。构造期间正常消耗的随机数不作为新episode随机起点。
3. 严格恢复全部模型参数、两套Adam及实际lr，恢复累计I／S／C和持久push状态；按显式A→B模式映射阶段字段，不改写父文件，也不删除环境part校验。GPU prepare后只核验实际物性回读，不再写COM。
4. 在所有对象构造和状态加载完成后，最后将全局随机流恢复为父checkpoint的记录，再发布累计训练时钟；时钟setter不simulate、不抽样、不触发push。沿用Runner的所用CUDA流恢复职责，环境存档的随机状态不另生成第二个seed；设备对应关系须符合Q1运行身份。
5. 执行一次全环境新B episode reset，重采局部执行系数与目标／命令，初始化独立力事件但初始输出为0N；清旧动作、历史和人工力缓存，再发布初始观测。按A既有reset调用与发布规则生成一次，不因Runner.learn首次进入再reset第二次。
6. 首个rollout从累计I=8000／S=192000开始，首次完整更新后提交I=8001。新episode reset不算训练step或迭代，不执行隐藏预热，不提前消耗一个24步rollout。

当前A恢复在I=8000分支刻意不reset，因为A已完成；B显式迁移需在此边界开启新episode，不能直接沿用A的“不再采样”分支。初始化和恢复只在进入B时执行一次，不在每轮更新时重复。

两路末端力事件以新episode局部年龄0初始化，已施加／待施加力、yaw标签及施力张量初始为零；允许抽样事件参数，但按B04上升起点不直接跳到峰值。A checkpoint没有B事件记录，显式迁移时按B schema创建，不把缺字段当作任意B→B checkpoint也可缺失的兼容规则；B→B事件存档和新episode重启语义留Q3。

底盘速度push延续全局累计时钟，不因新episode把其进度清零。保存的last_push标记避免重复已执行事件；如果S=192000本来恰好是下一次应执行的push边界，则在首个真实env.step按原调度正常发生，不在reset或进度setter中额外补一次，也不人为跳过。末端力局部事件与全局速度push是两套不同时间语义。

完整学习状态续训不等于完整PhysX世界续演：本方案不恢复接触求解缓存、关节约束内部状态、未完成episode的物理轨迹或32帧运动历史。因此不承诺与“从不暂停、直接跑过8000”的训练逐步随机数消耗和轨迹完全一致；B新事件抽样也会增加随机数消费。保留学习器状态与物性，是避免丢弃A学习进度，不是保证新episode不改变采样分布。此差异在run metadata记录为fresh_episodes_after_restore。

官方／A对比：沿用官方无力→受力时继续学习的目标；对象重建、物性创建时恢复和随机流最后恢复沿用A已验证重启方法。B新增为A8000完成点开启非零力任务的新episode及显式阶段字段映射。正常新episode执行系数重采属于A既有随机化生命周期，不应误报为创建级物性丢失。

最小调用框架（伪代码，无业务实现）：

```text
parent = validate_explicit_parent_before_gym()
env = V3PosForce(..., creation_properties=extract_creation_properties(parent))
runner = build_existing_model_and_learner(env, target_B_identity)
restore_model_and_both_adam_and_actual_lr(parent)
restore_counters_push_and_verify_creation_readback(parent, explicit_transition_mode)
restore_saved_rng_after_construction(parent)
env.set_training_progress(8000, 192000)  # 不推进物理
env.reset()                           # 内含初始观测发布，不额外simulate
runner.episodes_ready = True
runner.learn(additional_iterations)   # 不重复reset，预算细节在Q3确认
```

验证接缝：小规模完整父fixture的模型／双Adam／lr／累计计数恢复逐值核对；创建物性回读一致且load无物性setter；恢复RNG后初始化结果在同配置／同父下可重复、调用顺序为构造→恢复→一次reset；reset不改累计计数、会重采局部执行系数、末端力初值零且输入shape不变；首轮主辅slot从父step继续、I8000→8001；push边界不漏／重；远程正式父只按4096原尺寸恢复。B11组织实际跨进程短测，不在此要求新旧整条物理轨迹位级一致。

依据：`UniFP_v3/legged_gym/scripts/train_v3.py:94-110`、`v3_on_policy_runner.py:195-224,364-408`、`legged_robot_v3.py:285-342,975-1015,1030-1088`，及已确认B04／B05／B06。当前未读取远程checkpoint的新值、未修改业务代码或创建新训练。

### Q3：是否按累计40000轮停止，沿用完整更新边界保存，并以新episode方式恢复B？

状态：2026-11-07用户回复“同意，B07可以一起讨论”，确认以下预算执行、保存周期和B→B恢复语义，并授权合并讨论剩余Q4。运行隔离与异常处置留Q4，力课程具体参数留B09，观察／评估频率留B10；未修改业务代码。

推荐保留现有24步rollout、PPO主更新后辅助更新的顺序和空buffer边界，不为B增加另一套训练循环。

| 项目 | 推荐执行规则 |
|---|---|
| 正式预算 | A起点I=8000，B新增32000，累计I=40000结束；中断后按40000减已完成I计算剩余，不再次追加32000 |
| 完整轮提交 | 收集24步、计算returns、全部主／辅更新和buffer清理成功后，才提交I+1、S+24、C+24N；半轮不能作为可续训checkpoint |
| 周期保存 | 沿用A每200个累计完整迭代保存；本次请求终点另存，正式I=40000必存；不另引入自动选best或自动删除旧checkpoint |
| 文件命名 | 使用累计编号，如model_008200.pt、model_012000.pt、model_040000.pt；B相对迭代I−8000仅作为展示，不替代恢复计数 |
| B→B恢复 | 同阶段身份校验；恢复模型、双Adam、实际lr、创建级物性、累计课程／push时钟与RNG，再按Q2顺序执行一次新episode reset |
| 已完成预算 | I=40000仅允许验证已完成的完整checkpoint，不创建Gym、不reset、不采样；不能继续越过40000 |

例如I=12000断开后，已完成B的4000轮，正式剩余28000轮；进度展示同时保留累计12000与B相对4000，不能将这次恢复再理解为“从12000加32000”。完整结束时S=960000，C=N×S；正式N=4096时C=3932160000。测试或分段请求仍可使用现有增量调用，但不得越过累计上限，不在本项决定新的评估频率。

保存内容延续A的模型、两个优化器及参数顺序、实际lr、I／S／C、物性、RNG、identity／配置／lineage；B增加阶段／力课程持久状态和两路事件的`force_state`边界快照。课程若完全由累计I／S和已记录配置计算，就直接复用这两个时钟，不增加第二份独立计数；若B09确认额外持久课程状态，再列明必要字段。底盘速度push时钟及去重状态照常接续。

必须区分存档快照与恢复后仍生效的状态：`force_state`记录保存边界的两路事件参数／阶段、待施加力与已施加缓存，供存档核验和诊断；具体字段、shape、单位按B04和后续B09冻结，不保存冗余完整trace。它不是新episode第一步的施力来源。B→B与A→B一样不恢复PhysX世界，因此reset时必须清旧局部事件、力缓存与历史，重采新事件并从0N开始；不能在初始机器人姿态上继续旧半段峰值外力。新采样从已恢复RNG接续，累计训练／课程时钟不清零。B→B缺失应有B字段应报告不兼容，不能使用A→B缺B字段的初始化豁免。

沿用现有临时文件序列化、flush／fsync、回读校验、SHA256、原子发布和索引记录；不重写通用存档框架。保存必须要求无待处理动作、无进行中更新、buffer为空且I／S／C一致，不在每个策略step扫描文件hash。主／辅更新中途失败时不把已变更的一半优化器包装成完整轮，恢复来源仍为上一个成功发布checkpoint；具体退出／诊断行为在Q4收束。零预算完成校验位于启动入口、Gym创建前，沿用A已有完成检查方向扩展到B。

官方／A对比：预算沿既定官方课程复现路线；24步采样和主辅更新不变。每200轮保存、完整学习器存档与原子发布沿用本项目A实现，不宣称为论文要求。B新增的是阶段／力课程字段及其明确的重启语义，不增加学习算法、每轮验收门控或对普通物理失败的停训要求。

最小调用框架（伪代码，无业务实现；保留现有增量learn接口）：

```text
checkpoint = read_and_validate_explicit_checkpoint_before_gym()
remaining = 40000 - checkpoint.completed_iterations
if remaining == 0:
    return verify_completed_B_checkpoint(checkpoint)
env, runner = construct_with_saved_creation_properties(checkpoint)
runner.restore_B_learning_persistent_state_and_rng(checkpoint)
env.reset()  # 新episode；不续旧force_state，不改累计课程时钟
runner.episodes_ready = True
runner.learn(num_learning_iterations=remaining)
    # 每轮24步→returns→主更新→辅助更新→clear→提交I/S/C
    # 每200个累计完整轮或请求终点调用原有save
```

验证接缝：I8000／12000的剩余预算分别为32000／28000；提交计数只发生在完整主辅更新后；周期和请求终点重合时只保存一次；保存包含必要B字段且缺字段不被A→B豁免；B→B恢复后模型／Adam／课程与push计数接续、初始力零、仅reset一次；I40000在构造Gym前结束且无新采样；临时发布失败不覆盖旧存档。B11分别组织单元和真实跨进程验证，不要求暂停前后物理轨迹位级相同。

依据：`UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_on_policy_runner.py:82-152,190-224,306-408`，及已确认Q1／Q2。上方B扩展已确认，当前A源码仍限定8000；本项未修改业务代码或启动B训练。

### Q4（合并收束）：是否沿用独立运行目录和既有异常机制，仅普通物理终止局部reset？

状态：2026-11-07用户回复“同意”，确认以下合并方案，B07全部收束。Q1—Q3不重新表决；本项不决定B08奖励、B09力课程数值、B10评估阈值或B11最终验收规模。

推荐在现有入口、Runner和日志接口上完成必要B扩展，不建立额外训练守护服务或通用恢复框架。

#### 运行身份、目录与数据边界

- 正式B仍只在远程`/mnt/data_8THDD/myl_unifp_v3_stage1/`运行。本机用于开发、单元测试和已授权短测，不能将短测计入正式B预算。
- 首次A→B与后续B→B均指定明确父checkpoint；每次新进程恢复使用独立`runs/<run_id>/`，保留已有`checkpoints/`、`iterations.jsonl`、episode记录、TensorBoard及完成／失败报告布局，不向A或旧B目录追加，不覆盖父文件。
- 正式B使用冻结的独立源码／配置快照，继续引用V6 r2已确认资产；A源码快照与输出保持只读。来源记录关联根A父身份、本次直接父文件路径／SHA、恢复模式、B配置及源码身份、累计起点和新episode语义。保留lineage链，不把后续B恢复误标为重新A→B。
- 评估／回放读取checkpoint，输出独立目录，不更新训练模型、双Adam或训练RNG；优先复用现有独立进程评估方式，具体调用和频率在B10确认。评估不消耗训练迭代预算，不向训练buffer写样本。
- 身份／资产／schema与存档完整性检查集中在启动、加载和保存边界，沿Q1列出的允许差异执行；不新增每策略step文件扫描，也不以静默换资产、改力范围或跳过字段校验来解决不兼容。

#### 训练遇到不同情况时的处理

| 情况 | 推荐行为与已有接口关系 |
|---|---|
| 普通物理终止，包括现有规则判定的机身／机械臂接触 | 保留失败样本和原因，只reset对应环境，其他环境继续；不因首次失败或失败计数较多自动全局停止 |
| 纯timeout | 按B04真实终止后继Critic做一次bootstrap，再局部reset；与物理终止同时出现按物理终止，不补bootstrap |
| 有限但较差的reward、TCP误差、估力误差或失败率 | 记录并按B10分析；本模块不加入自动停训阈值、不自动改奖励／力课程／模型资产 |
| NaN／Inf、非法策略分布或非有限loss／梯度 | 复用已有数值检查，在发现处抛出并退出；不通过填零、跳过更新或reset一行掩盖污染 |
| Gym／CUDA／执行异常，包括OOM | 停止当前进程；保留已有日志及最后有效存档，不自动换设备、缩N或重启训练 |
| checkpoint或训练记录写入失败 | 沿A既有错误上抛；尽可能记录失败操作，不覆盖既有存档；不在无法可靠保存的情况下继续消耗正式预算 |
| checkpoint身份、shape、阶段字段不兼容 | 在相应加载核验处报告具体原因，不自动退回只加载模型权重或使用strict=False |
| SSH断开 | 使用既有脱离会话的运行方式时，断开不等于训练退出；重新连接后先核对进程和进度，不重复启动第二份训练 |

物理终止的具体碰撞集合、阈值与reward沿既有已确认环境合同，不在本项新增、删除或放宽终止条件。普通失败局部reset与官方常规采样方式一致；数值检查、明确父SHA、独立目录及保存失败退出是本项目工程约束，不称为论文规定。用户此前取消的“首次普通物理失败即全局停训”不恢复。

退出时复用Runner现有异常捕获，尽力记录异常类型／消息、已提交I／S／C、最后成功存档、当前rollout位置／是否正在更新及现有episode诊断，再原样抛出。若更新中途失败，不对半更新模型调用save；若日志在完整更新之后失败，也不承诺该轮已持久化，恢复以最后成功发布并核验的checkpoint为准。保存文件已发布但索引写入失败时保留文件及错误信息，后续对明确文件重新校验，不自动删除或选择latest。

诊断写入本身失败只报告次级错误，不替换原始异常；原生崩溃、kill或断电可能来不及生成failure.json，不能承诺都有Python报告。沿用正常退出的资源释放，不新加自定义信号处理或后台自动重试。计划内暂停优先采用Q3已有的分段请求终点保存；突然中断可能丢失上个checkpoint后的进度，每200轮保存意味着正常周期内最多接近200轮未持久化，已有数据不伪造补计。

最小调用关系（伪代码，映射现有main／learn／save；不新增通用异常框架）：

```text
main:
    validate_parent_and_target_identity_before_gym()
    build_environment_and_runner_for_new_output_directory()
    try:
        restore_complete_state_and_begin_new_episodes_once()
        runner.learn(remaining_budget)
        write_completion_for_committed_and_saved_progress()
    finally:
        destroy_sim_if_created()

runner.learn:
    try:
        collect_24_steps()  # 普通done由env局部reset，不转成Runner异常
        compute_returns_and_run_main_then_auxiliary_updates()
        clear_storage_and_commit_progress()
        write_iteration_records_and_save_if_due()
    except execution_or_recording_error:
        best_effort_write_existing_failure_diagnostics()
        raise  # 不保存半更新状态，不自动重启
```

验证接缝：普通终止不结束Runner且非done行不被重置；纯timeout与物理终止优先级沿B04回归；注入一次主／辅更新异常时不提交该轮、不发布半更新checkpoint；保存／日志故障保留原异常和旧有效文件；已有输出目录和父目录拒绝覆盖；连续A→B→B记录正确来源及累计进度；独立评估不改变训练学习器状态。B11汇总这些单元／整体检查，不要求用真实OOM、破坏远程磁盘或反复崩溃来验收。

源码依据：`UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_on_policy_runner.py:24-61,91-164,306-408`、`v3_ppo.py:182-269`、`UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:470,533-690,743-744,956`、`UniFP_v3/legged_gym/scripts/train_v3.py:96-123`。当前已确认本模块设计，未创建远程B进程或修改业务代码。
