# V3桌面模型修复：源码与资产事实

## 1. 桌面压缩包来源

- 文件：`/home/sorry/桌面/a1arm_description.zip`，SHA-256=`9638c2794cfa6874b57a8def527899b0352a29997e203b3a3b51dbc0a38346d8`。
- `robot_description/a1arm_description/urdf/a1arm_rgm3.urdf`在压缩包解出的归档目录中有完全相同字节，URDF SHA-256=`a3eacf4b991ba7bfa4be5e57e229adeb918c71a741f17be53d4f4ba799471f2c`。
- 桌面URDF包含27个link、26个joint、11个可动joint：6个臂旋转、2个夹爪移动、root x/y移动和root yaw旋转；四个轮link通过`bl_joint`等固定在`base_link`。轮/根关节定义位于压缩包URDF第559—585、718—757行。
- `base_link`使用`ranger_mini3_decimate.collision.glb`和`.visual.glb`，几何原点旋转约π绕x，见URDF第587—606行。四轮视觉和碰撞引用`ranger_mini_v3_wheel*.glb`，其视觉/碰撞网格局部偏移分别记录在第607—685行。
- `aluminum_profiles`到`base_link`的固定偏移为(0,0,-0.33)m，见第580—585行；`arm_base`通过`connect_joint`连接型材，偏移(0.174,0,0.42535)m，见第300行附近。
- 压缩包中实际包含的URDF只有`a1arm_gripper.urdf`与`a1arm_rgm3.urdf`，不包含approx_sim_v3或original_movable URDF。
- 桌面URDF内`base_link`质量为0.16272kg，四个轮link各2.27860kg；包含所有具有质量的link时总质量为17.595035kg。根dummy link为0质量，`end_effector_link`没有惯性。以上来自XML字面值，未表示厂家实测值。

## 2. 当前UniFP_v3运行资产

- `UniFP_v3/resources/robots/v3_a1/asset_manifest.json:2-10`记录运行URDF派生自归档`a1arm_rgm3_original_movable.urdf`，源SHA=`49b2e1b1acf728bdf1ec403126a219f7f62d7e5bbd0912c6b81af745fda498df`；只将两个gripper轴从prismatic改fixed。
- 当前运行URDF为28 link、27 joint、14 movable DOF。它有8个wheel steering/drive DOF和6个arm DOF；free root通过Gym actor root处理，没有桌面URDF的三个root dummy links。资产质量总和83.317915kg。
- 当前运行URDF的`base_link`质量65.8456kg，且包含`ranger_mini3_decimate.visual.glb`视觉网格以及四个轮固定部分视觉件；四个wheel drive link各2.2786kg，steering link各0.01kg。当前碰撞体包括arm、相机、型材和轮等多个approx_sim碰撞网格。
- `UniFP_v3/legged_gym/envs/v3/v3_pos_force_config.py:22-30`选择上述派生URDF，初始root高度0.32m；基础资产配置`self_collisions=0`开启自碰撞。
- 当前接触诊断`.scratch/unifp-v3-stage1-reproduction/knowledge/v3_contact_diagnostic_20261031.md`仅针对当前派生URDF。它记录actor自碰撞开启时约10.59kN base净接触力及部件正重叠；actor filter=1时因全局关闭自碰撞而base力消失。该结果不能直接证明桌面URDF同一形态下也会产生同样异常。

## 3. 项目先前V3物理估算记载

- `legacy/v3_reference_20261030/robot_description/a1arm_description/README_approx_sim.md:25-36`记载轮距0.370m、轴距0.494m、厂家CAD约0.10m轮半径；Mini 3底盘质量按手册75kg约束，因缺少分件惯性而集中到`base_link`并以盒体近似；四轮各沿用2.2786kg；A1、扩展架、相机约8.318kg；组合约83.318kg。文件明确要求装配实称替换估算，且不将旧近似称作高保真模型。
- 同README第52—54行记录旧V3的Gym/Lab迁移测试曾观察到16活动DOF（8轮系＋6臂＋2夹爪）且轮系关节可响应小目标；这属于另一URDF的历史测试事实。
- `legacy/v3_reference_20261030/robot_description/a1arm_description/meshes/original_movable/provenance.json`记载`original_movable` URDF由approx_sim_v3物理模型派生，四轮视觉三角面被分配到固定、滚动、转向三部分，原视觉三角面全部保留；轮胎仍是近似碰撞，底盘依旧无显式悬架。

## 4. 运行时接触实现与旧训练差异

- `UniFP_v3/legged_gym/envs/base/legged_robot.py:576`将`cfg.asset.self_collisions`传入`gym.create_actor`；当前默认0使actor内部形状可互碰。
- Isaac Gym Preview4官方本地文档`~/.cache/unifp/isaacgym_preview4/pkg/isaacgym/docs/_sources/programming/assets.rst.txt:107-109`说明actor collision_filter=0启用所有actor内部shape碰撞，任何正值关闭全部自碰撞。
- 历史训练使用原始movable URDF并传actor filter=1，见旧记录`gym_ranger_fresh512_fixedcom_20261028_v1/backend_envelope.py:76-82,109`。因此当前启用自碰撞造成的碰撞表现与旧训练设置不等价。

## 5. 本轮重新核验的原始记录

- 机器可读记录：`desktop_asset_audit_20261031.json`。记录直接读取桌面ZIP和当前运行URDF，包含压缩包/URDF哈希、逐link质量、全部活动关节、46处mesh引用及逐引用哈希、桌面URDF行号锚点。46处是引用次数，不代表46个不同网格文件。
- 桌面URDF第560—579行：四个轮系连接均为`fixed`，安装点为前后x=±0.250m、左右y=±0.190m、z=-0.100m；左轮连接包含约-π的绕x旋转。当前运行URDF的对应转向安装点为x=±0.247m、y=±0.185m、z=-0.100m，滚动关节另有z=-0.120m偏移。不同link坐标系不能直接用joint origin差值当作轮胎顶点的偏移量。
- 归档`meshes/original_movable/provenance.json`的`wheel_partition`记录四个轮视觉组曾分别应用配准平移。fl约(-0.0010,+0.0176,-0.0004)m，fr约(-0.0012,-0.0158,+0.0054)m，bl约(-0.0020,+0.0211,+0.0086)m，br约(-0.0022,-0.0188,+0.0054)m；各组scale=1。其“全部原始三角面保留”的记载不等于原始世界空间装配位置保留。
- 本轮XML求和：桌面整机17.595035372850802kg，当前运行资产83.3179153728508kg；桌面`base_link`为0.16272kg，当前为65.8456kg。数字来自文件声明，不是测量。
- 只读故障检查已运行：从ZIP及当前URDF解析轮系活动关节，桌面输出0，当前输出8；以目标8个轮系DOF为断言时桌面版本退出码1，输出“桌面模型的轮系活动关节为0，目标为8”。这是固定轮缺陷的结构复现，不能代替真实动力学测试。
- 数值比对同名link inertial后，桌面和当前V3除`base_link`与四个wheel link之外，其余同名刚体mass、COM及惯性张量数值相同；`base_link`两者物性不同，四轮质量均为2.27860kg但COM/惯量不同。桌面轮COM相对wheel link约z=0.04364m；桌面轮关节为fixed，因此源文件没有声明滚动关节轴，不能仅据COM偏移认定相对实际轮轴偏心。当前V3将wheel link COM置零，局部y轴为滚动轴（`Iyy=0.011393 kg·m²`，`Ixx=Izz=0.006911753 kg·m²`）。JSON记录逐link质量，COM和张量须查对应URDF。
- 当前V3的`base_link`分配为65.8456kg、COM=(0,0,-0.095)m，碰撞盒尺寸0.70×0.46×0.24m，近似惯量对角为(1.687842,3.160589,4.216313)kg·m²，见运行URDF第494—505行。历史生成脚本第172—201行记录集中质量估算的方法，但该段采用的旧COM和惯量并非当前运行文件原值，不能以该段证明当前所有数值的生成过程；当前数值可与0.72×0.50×0.256m均匀盒体公式交叉核对。

## 6. 厂家手册重新核对

- 资料：AgileX Robotics Team，《RANGER MINI 3.0 User Manual》，V1.0.0，2024-06-25；本轮核对日期2026-10-31。厂家署名原始手册PDF托管地址：https://www.agilex-italia.it/wp-content/uploads/2025/03/RANGER-MINI-3.0-User-Manual.pdf 。
- 第5/41页、1.2 Specifications：整车底盘总重75kg；轴距494mm，前/后轮距364mm；尺寸720×500×345mm；四轮转向，独立悬架。75kg是该底盘产品总重，不是base_link分件质量，也不包括本项目额外机械臂等上装。
- 前文第3节引用的370mm是旧项目README及当前关节安装点表达的值；它与本轮手册的364mm不同，不能写成厂家已确认370mm。桌面fixed安装frame、轮胎实际几何轮心与手册尺寸的定义也不能直接等同。
- 手册此规格表没有逐link质量、质心和惯性张量，不能据此将现有V3分配称作厂家分件标定值。

## 7. V3实际导入后的物性设置链

- `UniFP_v3/legged_gym/envs/base/legged_robot.py:510`开始`_create_envs()`，先以asset options加载URDF，再为每个env创建actor。
- 同文件第578—581行：回读actor刚体属性 → `_process_rigid_body_props()` → `set_actor_rigid_body_properties(..., recomputeInertia=True)`。第262—273行的回调只在启用质量随机化时修改`props[0].mass`，并非逐link恢复URDF惯量的回调。
- 当前`V3PosForce`继承该创建链，见`UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:30`及构造函数中的`super().__init__`。仅修改URDF不能证明最终actor惯量与URDF一致；本轮尚未做这条链的数值回读对比。
- 历史轮视觉构建脚本`legacy/v3_reference_20261030/robot_description/a1arm_description/scripts/build_ranger_visual_usd.py:165-218`先将源轮网格变换到base坐标，以连通分量定位轮胎，按面分成rolling/steering/fixed，随后对所有组施加`drive-center`配准平移。这是当前V3网格曾经重定位的直接源码依据。

## 8. 连续关节导入刚度与实际轮可动性

- `gym_wheel_drivemode.json`记录连续驱动关节hasLimits=False、driveMode=3（EFFORT），但stiffness=3.402823466e38；每轮施加约1N·m时，0.5s后速度仍接近0。不能用14DOF数量证明轮可动。
- `gym_wheel_stiffness_zero.json`在相同候选和输入下，仅清零导入stiffness，三轮速度恢复约0.49858rad/s；fr轮运动后出现局部接触，速度约-0.1306rad/s。这同时确认异常导入刚度和运动时碰撞归属仍需处理。
- 原URDF没有这个异常stiffness字面值，V3当前父类`_process_dof_props`只读限位，不清理该导入驱动属性；V3实际采用显式PD力矩输出。

## 9. 轮总成运动归属检查

- 桌面fl视觉轮总成的轮胎分量径向最大尺寸约0.10001m，另有2972面上方壳体分量径向最大约0.10145m；初始0.105m分组阈值把该壳体误归为rolling，运动后出现轮体—底盘接触。历史原视觉分组使用0.101m，可将该壳体留在steering。
- 原collision网格已独立简化和外扩，分量与visual不一一对应，不能直接用同一个径向阈值决定运动归属。修复限定collision：按与源visual的最近表面关联确定归属，滚动轮胎接触使用从桌面轮胎几何恢复的圆柱包络；保留全部visual面、零位位置与已批准质量/惯量。
