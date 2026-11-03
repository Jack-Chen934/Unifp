# A08活动资产FK与Gym离散接触核查

日期：2026-10-31。仅用于活动桌面URDF的独立运动学和有限离散姿态接触核查；不冻结A08目标余量，不是动态轨迹、速度/力矩或稳定性认证，未运行训练。

## 资产与方法

- 活动资产：`UniFP_v3/resources/robots/v3_a1_desktop/urdf/a1arm_rgm3_desktop_fixed_gripper.urdf`，SHA-256 `1dc2f47cf62baf707d9f1e7595b9df6b4455193feabedbd0b46e940ddffb9c0d`。
- A08原离线IK审计采用`legacy/v3_reference_20261030/robot_description/a1arm_description/urdf/a1arm_rgm3_original_movable.urdf`。本次按XML逐项核对`arm_joint1`—`arm_joint6`及`connect_joint`，活动URDF与旧审计来源中的这7个关节元素相同；TCP偏置仍取同一`v3_parameter_derivation_20261031.json`候选点。
- 新增`UniFP_v3/tools/compare_desktop_fk.py`，按URDF parent/child链、origin的xyz/rpy和joint axis独立计算刚体齐次变换，再与Gym回读姿态比较；另用活动URDF重算305个保存关节解的TCP目标残差，并检查12条首段路径的相邻关节解变化。
- 新增`UniFP_v3/tools/probe_a08_gym_contacts.py`，对每组已保存的6臂关节角独立设置状态，以活动URDF创建单机器人GPU PhysX仿真，固定base、关闭重力、每个姿态推进一个0.005 s物理步；使用现有局部装配shape过滤和`contact_offset=0.001 m`，碰撞形状`AssetOptions.thickness`固定为`0.01 m`，读取`get_env_rigid_contacts`。每个姿态相互独立，不将关节解串接成轨迹。

## 结果

| 检查 | 结果 |
|---|---|
| 名义姿态Gym刚体FK对照 | 最大刚体位置差`2.9861e-7 m`；最大姿态差`5.8619e-7 rad` |
| 名义TCP点（world） | `[0.41560481, -0.00013966, 1.19719541] m`；包含base初始`z=0.34 m` |
| 305组保存IK解在活动URDF上的离线候选TCP目标复算 | 305/305残差小于0.1 mm；最大`7.1080e-10 m`，P95 `3.7363e-10 m` |
| 同305组按V3运行时TCP偏置`[0,0,0.07961206]m`重算 | 残差最大`0.0004068953 m`，P95 `0.0004068952 m`；305/305小于0.5mm，0/305小于0.1mm |
| 305组保存关节姿态的低偏移Gym接触查询 | 0/305姿态报告接触候选 |
| 12条首段采样路径相邻解 | 12/12至少有一个相邻关节解大跳变；路径间最大单关节跳变量介于`2.3825`与`5.1650 rad`，多数为腕部关节分支变化 |
| 非相邻碰撞正对照`[0,3,-1,0,0,0] rad` | 1mm接触偏移下报告36个接触点，涉及`arm_seg6`/`gripper1`/`gripper2`与`base_link`；单步base净接触力峰值`6.44 N` |
| 名义姿态接触偏移对照 | `contact_offset=0.01 m`报告656个候选；降至`0.001 m`后为0。故高偏移下接触列表包含接触壳候选，不能直接当作实体碰撞证据 |

原始报告：`fk_comparison_nominal_20261031.json`、`a08_gym_contacts_all_saved_20261031.json`、`gym_active_nominal_contact_probe_20261031.json`、`gym_active_nominal_contact_offset001_20261031.json`、`gym_active_nonadjacent_low_offset_20261031.json`。工具已通过Python 3.8语法编译；上述探针使用Isaac Gym Preview4/Torch 2.3.1在RTX 5060 GPU PhysX运行。

## 解释与限制

活动模型的机械臂运动链与原A08位置审计模型相同，离线审计TCP点定义下305组已保存解在活动URDF上仍满足位置方程；Gym名义姿态回读也与独立FK一致。当前V3运行时`_tcp_position`使用`[0,0,0.07961206]m`，旧离线候选还带`x/y=[-0.00014369,0.00038068]m`，因此同一批关节解按运行时点定义的残差固定约`0.407mm`，未达到旧审计0.1mm阈值。有限接触抽查未发现这些离散姿态的碰撞候选，正对照表明低接触偏移查询能够报告明确碰撞姿态。

但12条已存首段路径使用逐点独立优化结果，关节角在相邻采样点出现最高约`5.17 rad`跳变。这些解不是可直接插值或跟踪的连续关节轨迹；离散姿态逐点无碰撞不能证明关节空间或笛卡尔路径中间段无碰撞。还未验证完整候选域以连续分支追踪的IK路径、速度雅可比、速度/力矩边界、自由底盘随动和动态稳定性。候选目标域仍可作为后续验证输入，但5°限位余量、5 cm离地间隙及2 cm非豁免部件间隙均未冻结。

跳变的结构性原因和官方UniFP对照另见[分支跳变诊断](a08_ik_branch_diagnosis_20261031.md)。关键补充是：6轴位置灵敏度随TCP定义而变。离线审计候选点对6轴灵敏度约`4.07e-4 m/rad`；当前V3运行时代码使用纯z轴TCP偏置，对6轴平移位置的灵敏度在数值精度内为零。旧解不能作为当前实际TCP的0.1mm精度保证。

## 后续准入事项

1. 为首段及不同常规端点建立有明确分支连续性约束的IK续接解；记录失败点、每步关节增量和可用限位余量。
2. 对连续关节路径上的插值样本使用活动资产和生产碰撞参数检查非相邻碰撞及TCP/机身/地面几何；接触壳候选与实际非零碰撞力分开记录。
3. 在现有32环境V3任务内核验运动状态、速度雅可比或有限差分、自由底盘耦合和目标跟踪稳定性；未通过前不启动训练。
