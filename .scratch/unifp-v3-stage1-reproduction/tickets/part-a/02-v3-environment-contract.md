# 02 · V3环境、观测动作与目标时序

状态：完成（2026-10-31）。桌面基准新资产、局部碰撞shape过滤、EFFORT/stiffness导入修正及V3四环境资产/reset验收已完成，详见[资产实验记录](../../../../UniFP_v3/docs/experiments/v3/实验汇总.md)。旧`v3_a1`资产已标注为历史问题资产，不再是当前V3配置。A04公开shape、14维映射、观测历史、显式PD、目标段采样/插值、底盘速度命令采样、纯timeout快照、训练时钟发布和局部reset均已接入，并通过32环境定向Gym短测。路径筛查采用官方方法：每段取10个均匀球坐标样本并按V3的5cm TCP离地候选筛查，不复制B2禁入盒、不运行在线IK，也不以旧离线IK分支跳变阻塞训练。旧离线IK的0.407mm TCP点偏差继续披露为诊断局限。A09物性随机化／底盘push及其跨episode状态归04；PPO rollout与checkpoint归05。全臂连续路径可达性、碰撞、速度雅可比、自由底盘动态与跟踪稳定性仍未验证，几何余量仍未冻结；这些属于后续实验验证，不阻止本票交付。23项奖励仍为零占位，当前环境仍不可用于学习。证据见[FK/接触核查](../../knowledge/a08_active_asset_fk_contact_audit_20261031.md)及[分支原因与官方对照](../../knowledge/a08_ik_branch_diagnosis_20261031.md)。

## 目标与范围

后续状态回填（2026-10-31）：上述“23项奖励仍为零占位”是02交付时的状态，现已由04实现；A09随机化／push及状态接口也已完成。当前实现与证据以[04票据](04-reward-randomization-and-episode-metrics.md)为准，PPO／Runner仍由05接入。

基于官方环境接口建立V3真实Gym环境：加载确认的V3资产，落实14维动作、A04观测／历史／Critic字段、命令时序、局部reset、终端快照、Q5c单步顺序及Runner训练时钟发布。A阶段末端命令力和人工注入力恒为零，底盘任务命令槽与末端力槽分离。A09实际物性随机化、底盘push与跨episode随机化状态由04实施；PPO rollout和checkpoint由05实施。

## 代码位置与接口

- V3目标：`UniFP_v3/legged_gym/envs/v3/`环境与配置；V3资产单独位于确认的resources目录。
- 接口：`reset_idx(env_ids)`、`step(actions[N,14])`、观测生成、`set_training_progress(completed_iterations, training_policy_steps)`；shape、单位、坐标按A04/A06设计文档。随机化状态快照接口由04按A09/A07契约接入。
- 只在A09候选参数经短测后配置化；不将候选默认值误作冻结长训参数。

## 测试与验收

- 单测：坐标／TCP转换、动作顺序、局部reset隔离、零末端力不变量、旧目标与终端快照对齐。
- 整体：32环境Gym加载、名义reset、接触／测量、24步环境rollout、目标插值／段切换、timeout终端快照及局部reset。底盘push前后行为由04验收。
- 验收：无shape／单位错位、跨env污染或关键非有限值；真实状态与A04合同对齐；训练时钟只由Runner发布且reset不推进全局计数。

## Blocked by

- 01-v3-repo-and-task-scaffold。

## 实施记录

- 已从归档中的`v3_original`源复制原始URDF及47个URDF网格引用到`UniFP_v3/resources/robots/v3_a1/`；归档存在35个被引用的网格别名符号链接，开发资源中复制为普通文件并保留URDF要求的相对路径。来源、派生及网格SHA-256记录于`asset_manifest.json`。
- 派生URDF仅将`gripper1_axis`、`gripper2_axis`从prismatic改为fixed，保留origin和parent/child，移除不再适用的axis、limit、mimic字段；解析所得可动关节数为14。原始资产文件未修改。
- 静态来源哈希、全部网格路径／哈希、14-DOF拓扑检查通过。此前明确拒绝构造的骨架已替换为环境实现初稿；现已在Preview4导入该环境模块，并独立创建GPU PhysX仿真和加载V3资产，确认14 DOF／28刚体。完整环境刚体属性回读、接触／动态稳定性及step/reset仍未验证。
- 在`UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py`将V3环境改为官方`LeggedRobot`子类，增加按名称建立的14维策略／Gym DOF映射、58／98维单帧字段、32／3帧历史、12维标签、候选TCP变换、动作位置／轮速反馈、四轮接触字段和step／reset返回形状。配置在`v3_pos_force_config.py`中显式给出14关节名义姿态、03短测控制候选及平地配置。
- `commands[3:6]`现从reset后Gym刚体状态读取的TCP球坐标起步；按A08验证候选域独立采样下一端点，按每段移动／保持时间候选及20ms策略周期向上取整，并在每个策略边界做球坐标插值。`commands[:3]`按配置候选范围reset采样并每250策略步重采；末端姿态／力与底盘力命令槽保持零。候选段筛查在段初始化时取10个均匀球坐标样本，核对TCP有限值、半径非零和5cm离地候选；不代表非相邻机械臂碰撞、关节路径可达或连续物理安全通过。
- reset会在root／DOF索引写入后刷新Gym张量，从`arm_seg6`与已选TCP局部偏移读取球坐标并设为当前目标；增加root／DOF/TCP有限值及TCP候选地面高度核验。四环境验证确认命令起点与同次Gym读回一致；后续独立URDF FK已确认名义零速度姿态对齐，但移动base姿态和非零关节速度运动学尚未验证。
- reset当前目标取root／DOF索引写入和Gym刷新后读到的真实刚体TCP；独立URDF FK只完成名义静态姿态交叉核验。速度Jacobian、移动base及非零速度运动学仍未完成，不能将静态FK结果扩展为动态运动学验收。
- Critic现按98维已确认切片中的名义质量／COM变化量零、名义摩擦1、名义执行系数偏差零构造字段；补偿后目标取A阶段几何命令目标。质量／COM与执行系数实际应用及回读仍待A09，摩擦必须在Gym加载后回读验证；当前名义槽位不是已验证资产物性。
- Actor动作裁剪改读官方`cfg.normalization.clip_actions`。reset首帧发布时明确将所选环境历史清零后写入最新帧，避免直接调用全量reset时历史遗留。
- reset现把真实TCP球坐标作为首段起点；按A08验证候选采样半径／仰角／方位目标及独立移动／保持时长，并以20ms策略周期向上取整。每策略step推进球坐标目标，方位走最短角差；底盘机身轴`vx/vy/yaw_rate`在reset采样、每250策略步重采，命令力、末端力与末端姿态槽为零。当前候选筛查限于端点域及10个路径样本的TCP有限值／非零半径和候选离地高度，未覆盖整臂碰撞／关节逆解路径，不可据此放行训练。
- Q5c单步继续在当前物理状态刷新后推进下一边界命令，再生成reset前terminal Critic history，最后局部reset。新增root、DOF、刚体、接触及力矩核心状态的非有限值停止检查；episode计数按已确认`20s / 20ms = 1000`步取最近整数，避免基类因`ceil(20 / 0.01999999955)`产生1001步。
- 定向回归位于`UniFP_v3/tools/check_v3_desktop_runtime.py`：真实4环境Gym实例覆盖reset TCP起点、零末端力命令、目标插值／最短方位／段切换、250步底盘命令边界、1000步timeout终止快照新命令以及局部reset隔离。2026-10-31使用Preview4 Python 3.8/Torch 2.3.1实际通过；静态名义姿态FK和离散接触抽查已完成，速度Jacobian、连续整臂路径碰撞／动态可达性、32环境整体回归及Runner/PPO联调仍未完成。
- 2026-10-31在专用Preview4 Python 3.8/Torch 2.3.1运行时完成32环境定向整体短测：目标从reset TCP启动、A末端力槽为零、目标策略边界插值／最短方位路径、移动保持切换、250步底盘命令重采、1000步纯timeout终止Critic快照与reset前命令对齐均通过；timeout环境随后局部reset，其他31个环境未受污染。报告位于`.scratch/unifp-v3-stage1-reproduction/knowledge/model_history/v3-desktop-model-repair/knowledge/v3_runtime_check.json`。该测未覆盖完整A04物理可达／碰撞、Actor/Critic Runner联调或训练。
- 旧资产碰撞问题和最终候选的局部过滤、Gym验收见[资产实验记录](../../../../UniFP_v3/docs/experiments/v3/实验汇总.md)。旧诊断中的actor级filter=1仅是历史因果对照，不是当前实现；当前活动桌面资产保留非相邻自碰撞。目标策略时序已实现，但筛查只覆盖当前候选合同中的TCP点高度／有限性；完整碰撞和运行动态可达性没有运行时证明。奖励、随机化、训练级错误汇总和Runner终端价值对接仍未完成；当前不可训练。
- 2026-10-31补充历史训练对照：当前实际资产是原V3 movable URDF仅固定夹爪关节的派生版；历史Gym训练加载原始movable URDF但actor filter=1关闭全部自碰撞，当前默认filter=0启用自碰撞。旧评估仅记终止计数，少量非超时终止原因未知，没有逐体接触力记录，故不能据此确认旧训练复现过当前内部自碰撞高力异常。详见`knowledge/v3_previous_training_contact_comparison_20261031.md`。
- 2026-10-31独立FK核查：新增`UniFP_v3/tools/compare_desktop_fk.py`及探针姿态四元数输出；名义reset姿态的Gym刚体与活动URDF FK最大位置差`2.99e-7 m`、最大姿态差`5.86e-7 rad`。活动资产的六个arm关节及arm安装关节与A08旧审计URDF XML一致；305组既有位置解在活动URDF上的TCP目标残差最大`7.11e-10 m`。这确认静态几何解一致，不验证执行器/动态行为。
- 2026-10-31活动资产接触抽查：新增`UniFP_v3/tools/probe_a08_gym_contacts.py`。活动Gym在1mm contact offset、局部批准装配过滤下对305个保存姿态逐个复位检查，0/305报告接触候选；已知`[0,3,-1,0,0,0]rad`正对照检出arm/gripper与base接触，base单步净力峰值6.44N。名义姿态在10mm接触偏移下的656个候选降至1mm时为0，说明需区分接触壳候选与真实穿透/受力。
- 2026-10-31路径连续性与原因：12条首段的逐点位置IK都出现2.38—5.17rad单关节跳变；位置-only目标仅3维约束6个关节，数值平移Jacobian秩为3，且V3运行时TCP纯z偏置使第6轴位置雅可比为零。原离线解先试名义姿态、成功阈值只看TCP位置残差，没有连续分支排序，因此跳变反映审计求解器未约束选解，不能作为活动策略已经跳变或工作空间必然不可达的证据。官方UniFP直接由策略动作经PD控制末端位置跟踪，没有调用其IK helper；它也没有全臂连续IK／碰撞保证。详见[分支跳变原因及官方对照](../../knowledge/a08_ik_branch_diagnosis_20261031.md)。
- 305组旧解的`<0.1mm`残差使用离线候选TCP偏置；V3运行时`_tcp_position`采用`[0,0,0.07961206]m`，相对候选偏置横向相差约`0.407mm`。按运行时偏置重算后305/305残差约`0.407mm`，因此不满足旧审计的`<0.1mm`阈值；需先统一TCP定义，再判断求解误差与通过标准。A08目标边界、5°限位余量、5cm离地间隙及2cm非豁免间隙均未冻结。
- 2026-10-31新增`set_training_progress(completed_iterations, training_policy_steps)`，按A07/A09只发布Runner提供的训练时钟，并派生`training_transitions = num_envs * training_policy_steps`。接口校验非负Python整数、T=24中间rollout边界和同进程单调性；测试确认不推进`common_step_counter`、episode长度、命令或仿真状态，环境reset不清除该全局进度。
- 2026-10-31使用本机Preview4 Python 3.8/Torch 2.0.1完成32环境定向短测；在24个连续策略step内验证命令插值，并覆盖官方10点TCP高度筛查、地面拒绝、训练时钟发布与非法计数拒绝、reset隔离及纯timeout终端快照。报告位于`.scratch/unifp-v3-stage1-reproduction/knowledge/model_history/v3-desktop-model-repair/knowledge/v3_runtime_check.json`。未覆盖PPO更新、A09物性随机化／push恢复或checkpoint往返；这些由后续票据负责。
