# 当前源码与诊断结果依据

日期：2026-11-04。本文件只汇编实际源码／日志／实验事实；修改建议见同任务的设计及计划文档。

## 2026-11-05官方终止与观察层停止的源码依据

- `UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:176—180,199—205`：接触、姿态越界、超时设置reset_buf，然后仅reset_idx选中的环境，并继续观测。
- `UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py:124—151`：dones传入PPO，结束episode更新统计，继续rollout及优化；没有单次物理失败终止整次训练的分支。
- 本目录`train_observed.py`修改前，append_record在pending_failures非零时保存后raise；save_and_observe也在评估失败数非零时raise。停止由项目观察层触发，非官方环境规则。
- `UniFP_v3/legged_gym/scripts/train_v3.py:112—118`：正式恢复预算为8000减去runner.completed_iterations。
- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_on_policy_runner.py:365—407`：load核验完整身份，恢复双优化器、学习率、进度、环境持久状态与RNG；记录父checkpoint路径及摘要，reset开始新episode，拒绝复用父run输出。

## reset到TCP与标签调用链

- `UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:1038—1086`：reset写选中root／DOF，调用refresh，从`_tcp_spherical_position()`初始化目标与commands。1077—1078行注释认为refresh已给出新TCP。
- 同文件`1366—1396`：TCP来自`rigid_state`中arm_seg6位姿及局部偏移`(0,0,0.07961206)m`，再按yaw对齐中心转换为`[m,rad,rad]`。
- 同文件`1398—1449`：辅助标签含底盘速度、TCP球坐标和力真值，Critic拼接这些标签；Actor单帧58维、Critic98维、辅助标签12维。
- `UniFP_v3/tools/compare_desktop_fk.py:57—93`：根据URDF关节图、root变换和DOF求所有link FK，不依赖Gym缓存。
- 本目录`reset_baseline.json`的`constructor_reset`：缓存／FK差1.454375m；重复refresh五次不变；物理推进后差5.42e−7m但目标仍差1.448894m。局部reset差0.717458m，推进后缓存恢复而目标仍差0.717382m。其他环境root行差0。真实脚本为`probe_reset.py`。

## 动作到轮系物理调用链

- `legged_robot_v3.py:1300—1339`：转向／机械臂位置反馈；驱动为`d_gain*(target_velocity-actual_velocity)`，最后按URDF effort裁剪。驱动目标由5×action×执行系数生成，限制±20rad/s。
- `v3_pos_force_config.py:56—94`：驱动增益0.227385，转向位置增益24.9218；`100—103`命令范围vx±0.6m/s、vy±0.4m/s、yaw±0.6rad/s。
- 本目录`wheels_no_self.json`中的`dof_properties`：全部driveMode=3、stiffness=0；驱动导入damping=0、effort=80Nm；转向导入damping=0.2、effort=35Nm。
- 六份`wheels_*.json`及trace：4环境解析轮命令，先静置1秒，再执行8秒，统计后4秒；未运行policy模式、无优化器／任务reset。原样前进0.000782m/s；原样关闭自碰撞0.000803m/s；相同关闭条件下仅改光滑胎面后0.291506m/s。侧移对应0.002352→0.192935m/s。
- `wheels_gain16.json`：前后速度0.228428／−0.224671m/s；基座净接触峰值412.84／399.27N。侧移和旋转仍近0。
- `wheels_gain1.json`：侧移／旋转基座净接触峰值348.27／108.29N；`wheels_no_self.json`相应值均0。没有逐接触对日志。
- `wheels_round_gain4.json`：光滑代理且关闭自碰撞，增益×4时偏航仅0.047522rad/s，命令0.3rad/s。

## 活动碰撞资产和既有检查

- `UniFP_v3/tests/test_training_collision_asset.py:117—141`：现有训练候选保留源胎面collision和mesh字节，检查包络极值及内缩轮毂。
- `UniFP_v3/tools/probe_training_collision_runtime.py:189—215`：名义小动作阶段转向动作0.4、驱动动作0.1，执行250策略步；记录终止次数、DOF均值，没有目标底盘速度及其误差判据。215行也注明不是训练后能力验收。
- `UniFP_v3/legged_gym/utils/v3_asset_collision.py:8—50`：按manifest核验导入shape数，仅对指定装配件设置局部filter bit，保留其他部件自碰撞。
- 本目录`rolling_support.json`：基于活动URDF的4轮碰撞顶点按Y轮轴采样1441角度，支撑半径变化2.91—3.04mm；假设单轮400N时几何抬升力矩量级9.18—9.61Nm。仅为几何计算，不是物理引擎阻力矩测量。脚本与网格摘要同存。
- `diagnostic_round_contact.urdf`只将四个胎面改为半径0.1016m球碰撞，视觉和物性未改；运行时关闭自碰撞。该证据不含有效训练manifest，临时资产目录已删除，生成器保留。

## 训练／评估与来源身份

- 训练固定源码：`/mnt/data_8THDD/myl_unifp_v3_stage1/validation/collision_optimization_20261103_r02/source/UniFP_v3`；解释器：`/mnt/data_8THDD/conda_envs/unifp_v3_stage1/bin/python`。
- 诊断开始时GPU1利用率0%，GPU2约98%，所有本轮仿真在GPU1。实际诊断目录为`/mnt/data_8THDD/myl_unifp_v3_stage1/diagnostics/a8000_20261104`。
- 前序`.scratch/unifp-v3-stage1-reproduction/knowledge/model_history/unifp-v3-collision-optimization/knowledge/a8000_analysis_20261104/evaluation_r02/S1/summary.json`：平面速度误差0.361549m/s，超过冻结0.15m/s。`motion_diagnostics.json`首步TCP L2均值1.187235m；排除首秒的S1命令／实际速度模长均值0.3868／0.0514m/s。
- 同目录`paper_aligned_metrics.json`反算S1四轮目标±20rad/s比例55.07%／64.85%／67.99%／21.01%，不是实际轮速或力矩饱和率。
- 本目录`identity_verification.json`与S1正式archive manifest比较45项source、113项asset，无差异；权重SHA256未变。`evidence_integrity.json`保存20份下载证据的逐项摘要一致结果。

## 清理事实

`cleanup_manifest.json`保存149个已删除文件路径，合计424,746,597字节。其中422,844,154字节是r01失败归档内与r02摘要一致的三份重复大文件；其他为本项目可再生缓存和临时代理目录。完整r02、正式训练数据、其他Go2W／Piper工作内容与torch_extensions缓存保留。

## 2026-11-04实施后的源码与实测事实

- `UniFP_v3/legged_gym/utils/v3_kinematics.py:58—83`：从当前root／DOF张量计算世界TCP，输入root的四元数为xyzw，输出[N,3]米；不依赖刚体缓存。`legged_robot_v3.py:1368—1374`已接入此路径。
- `artifacts/t01/t01_reset_gpu.json`：构造／局部reset新路径TCP误差约8.28e−8m、目标误差约1.08e−7m，辅助与Critic标签误差为零，其他行与训练计数不变；旧路径报告`artifacts/t01/t01_reset_old_red.json`失败。
- `UniFP_v3/tools/build_rolling_collision_asset.py:25—53`当前从源collision采样9个轴向胎肩截面，名义半径0.1016m、16分片，仍是未准入候选生成器；正式配置未切换。
- `artifacts/t02/wheels_cadprofile_gain8.json`：相同新解析输入下，train_v4前后均速0.2692／−0.2665m/s，侧移与yaw仍近零，底盘接触峰值1177.6／434.5N；驱动增益只在诊断实例乘8。
- `artifacts/t02/desktop_visual_clearance.json`：原visual四轮最大半径100.01／100.12mm，原collision最大径向104.22／104.42mm；后底盘组件71／61的动态接触点距visual2.95／3.21mm。转向±1、±1.57rad且驱动零位时，visual顶点穿入数与双向边—三角面穿越数均为零；最小采样间隙为后右0.545mm，同姿态原collision凸体相交。该报告仅证明所列有限姿态。
- `artifacts/local_regression_t01_t02_final.log`：最终本机静态／纯代码回归50项通过；真实Gym奖励／恢复5项见T01远程日志。尚无最终新资产的512／4096整体通过结果。

## 2026-11-04 零位轮胎支撑诊断补充

- V3默认根高度为`0.34 m`，见`UniFP_v3/legged_gym/envs/v3/v3_pos_force_config.py:36-39`。
- train_v5四轮半径和轮心记录位于`UniFP_v3/resources/robots/v3_a1_desktop_train_v5/asset_manifest.json:2929-3040`；四个驱动轴均为`0 1 0`，其各角关节origin不同，见`UniFP_v3/resources/robots/v3_a1_desktop_train_v5/urdf/a1arm_rgm3_desktop_fixed_gripper.urdf:1438-1575`。
- 隔离GPU PhysX探针新增稳定阶段trace。z=0.34m从空中下落，FR／BL先承重，1秒后BR离地约8.3mm且BR驱动轮保留约-1.08rad/s。结果见`.scratch/unifp-v3-stage1-reproduction/knowledge/model_history/unifp-v3-a8000-diagnosis/artifacts/t02/local_yaw_settle_contacts.json`和`local_yaw_settle_pose.json`。
- 同一探针仅将初始根高度覆盖到0.32m和0.33m，1秒后均未形成稳定四轮支撑；0.32m有穿地冲击。结果见同目录`local_yaw_settle_height_032.json`和`local_yaw_settle_height_033.json`。
- 按URDF FK和manifest视觉半径算得四胎最低点z为-0.319610、-0.325419、-0.328610、-0.325419m；最佳平面残差峰峰约4.527mm。该事实证明支撑点高度不一致并使探针初态不对称，但不足以断定它是长训偏航失败的全部原因。

## 2026-11-04 共面候选实际结果

- `UniFP_v3/tools/build_rolling_collision_asset.py:266`的`build_coplanar_rolling_asset`复制源资产，从实际滚动分片计算支撑平面，并将逐轮径向补偿写入输出目录。第320行以输出URDF为基准解析mesh路径。首版误用源目录的问题已修正；源恢复完整性见`artifacts/t02/v6_r2_integrity.json`：188个源mesh与原manifest哈希一致，候选只64个rolling_tire mesh不同，URDF字节相同。
- `UniFP_v3/tests/test_rolling_collision_asset.py:157`在独立临时输出内检查源manifest／URDF／网格未变、≤2.3mm补偿、非胎面网格字节不变，并独立回读导出网格的平面支撑。`artifacts/t02/v6_r2_geometry_readback.json`记录实际支撑峰峰残差9.38900549e−8m；manifest公式预估值不作为唯一几何证据。
- 有效候选`v3_a1_desktop_train_v6_coplanar_r2`的FL／FR／BL／BR补偿约+2.22463／−2.22468／−2.26365／+2.26370mm。源零位碰撞残差约4.527mm；早先“源碰撞已共面”来自生成器误写期间的失效状态，不是原始资产事实。
- `artifacts/t02/v6_r2_forward.json`与`v6_r2_reverse.json`：CPU张量管线＋GPU PhysX、4环境、seed1、默认根高0.34m、先静置1秒再动作8秒。后4秒正／反偏航均速+0.243802／−0.242730rad/s，逐步绝对误差0.056648／0.059718rad/s；前后／双侧移主速度误差均小于0.03m/s。运动全程base_link净接触力峰值为0N，正／反偏航真实子步轮驱动力矩峰值约10.72／8.66Nm。
- 同两份报告的`settle_trace`：环境0最后0.5秒四轮采样z接触力大于1N的比例为1.00／0.88／0.88／0.92，1秒末BR轮速约+0.232rad/s；这些是采样事实，不等同于连续接触或完全静止。
- `artifacts/t02/v6_r2_gain120_forward.json`仅在探针实例把驱动反馈提高20%至5.45724Nm*s/rad，正偏航均速0.245216rad/s，逐步误差0.065996rad/s；正式配置保持4.5477Nm*s/rad。各证据／配置哈希见`v6_r2_comparison_summary.json`。
- `artifacts/t02/v6_r2_regression.log`为14项本机静态测试通过；`v6_r2_control_regression.log`为专用Gym环境1项配置测试通过。事故副本失效metadata与清理清单保存在`artifacts/t02/invalid_v6_coplanar_metadata/`，重复日志清理清单为`v6_r2_duplicate_log_cleanup.json`。

## 远程复核前的探针事实（2026-11-04，GPU分支修复）

`knowledge/probe_wheels.py`此前在静置和运动循环中仅`cpu_contacts=True`时调用物理推进；GPU分支返回None，未调用已选定的推进函数，且汇总力矩无法由None计算。因此该版本不能用于GPU物理结论；既有本机`--cpu-contacts`结果不受此缺陷影响。修复后`advance_interval`统一推进原控制decimation，CPU管线才阻塞fetch，双方均采集真实子步力矩峰值。调用点均无条件推进。`knowledge/test_probe_wheels.py`复现了缺失调用与返回值错误，修复前3项失败，修复后3项通过。

本机与远程r01的`legged_robot_v3.py`及`v3_kinematics.py`哈希相同；配置差异为默认资产路径（远程v1、本机v5）和一行说明，驱动阻尼均4.5477。远程复核使用独立r02源码副本及显式V6 r2资产路径，避免修改冻结的旧训练源码。

### 悬空轮与时序对照的直接证据

`artifacts/t02/v6_r2_gpu/airborne_gain120_factor1/2.json`及同名trace：两组均使用V6 r2、4环境seed1、根高1000m（全程无地面接触）、驱动增益5.45724；前者物理dt=0.005s、decimation=4，后者dt=0.0025s、decimation=8，策略dt均约0.02s。原生actor回读四轮Iyy均0.01139300037kg*m²。5ms组前进工况稳态轮速目标误差约16.8—17.1rad/s、实际施加轮力矩达到80Nm；2.5ms组稳态四轮轮速目标误差均为输出精度下的0，峰值16.749Nm。两组wheel/base净接触力全程0N。故该对照中的大轮速偏差无需地面摩擦即可出现；只细化更新即可消失。

`ground_factor2_gain100_sign1.json`恢复默认根高0.34m且保持正式驱动增益4.5477，仅细化物理/PD至2.5ms：正偏航MAE仍0.069165rad/s，平移通过。说明更新稳定性风险与带载稳态欠速不能混为一个已解决问题；细化时序本身不足以通过偏航门槛。
