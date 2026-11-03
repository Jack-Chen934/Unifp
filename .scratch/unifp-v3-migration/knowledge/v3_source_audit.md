# Ranger Mini V3＋A1Arm 现有迁移：源码事实审计

核查日期：2026-10-30。范围为本地当前工作树、冻结摘要、保存的评估摘要；本轮未启动训练、未运行仿真、未连接实机。源码路径以下以 `/home/sorry/python_ws/interactive/unifp_reproduction/` 为根。本文只记录事实与可直接复算的结果，建议另见 `../v3_recommendations.md`。

## 1. 当前存在两条运行路径

- 根目录 `ranger_unifp_vec_env.py:11–24` 直接依赖 Isaac Lab；`RangerUniFPVecEnv` 继承 `UniFPVecEnv`，资产默认 `v3_original`，见 `ranger_unifp_vec_env.py:27–36,227–247`。
- 已冻结的 512 环境训练通过 `ranger_512_migration/logic/backend_envelope.py:1–8,36–51,61–81,272–320` 在 Isaac Gym Preview4 中提取并执行这些任务方法；物理 `dt=0.005 s`、decimation 4、策略周期 `0.02 s`，GPU PhysX、CPU tensor/learner，solver 32/1，平面地形，自碰撞关闭。它不是已完成的 Isaac Lab 正式训练管线。
- `ranger_512_migration/logic/ranger_512_fixedcom_baseline.py:95–105` 直接导入本地官方 `ActorCritic` 和 `PPO`，不是另外重写的一套 PPO。
- `PROJECT_REPORT.md:38–40` 记录旧日志、轨迹、评估原始数据和项目内 `IsaacLab/` 已删除。现在保存的 128 条评估完整性结论是归档摘要结论，本轮不能对所有原始轨迹再次完整复算。

## 2. 实际资产与动作语义

- 当前 Gym 资产为 `robot_description/a1arm_description/urdf/a1arm_rgm3_original_movable.urdf`，见 `backend_envelope.py:80–81`。默认 Lab 资产为 `assets/v3_original/a1arm_rgm3_original_movable.usd`，见 `ranger_unifp_vec_env.py:27–32`。
- XML 直接求和得到当前可动资产标称总质量 `83.3179153728508 kg`，共 4 个 revolute 转向关节、4 个 continuous 轮驱动关节、6 个 arm revolute 关节、2 个夹爪 prismatic 关节。旧 `a1arm_rgm3.urdf` 则是 `17.595035372850802 kg`，轮固定，另外含 3 个平面虚拟根关节。因而“当前只有固定轮、总质量 17.6 kg”的旧结论不适用于当前训练资产。
- 当前可动资产 `base_link` 质量 `65.845600 kg`，惯性对角 `1.6878422133 / 3.1605888 / 4.2163132533 kg·m²`，COM 为 `(0,0,-0.095) m`，车身碰撞盒 `0.70×0.46×0.24 m`；见可动 URDF `499–509`。四轮分别为 `2.27860 kg`，碰撞圆柱半径 `0.1 m`、长度 `0.08 m`；见 `564–589,612–637,660–685,708–733`。
- 可动资产无平面虚拟根关节。根为物理自由基座，依靠轮地接触和轮驱动运动。`backend_envelope.py:109,230–235` 中设置 root state 是初始化/reset/push 运输接口；常规动作由关节 PD 目标执行，并不是常规策略动作直接设置 root pose。
- 资产名字包含 `original_movable`，但这不构成质量、轮胎和执行器实测标定证明。保留的近似生成器明确声明 link inertia/轮位置/执行器限值为估计，75 kg 是底盘标称质量，再叠加臂与安装件：`robot_description/a1arm_description/scripts/build_ranger_mini3_approx_sim_urdf.py:15–26,110–114,172–210`。当前可动 URDF 本身的具体数字见上述条目，本轮未发现整机称重、惯性辨识或驱动器响应标定记录。
- `ACTION_NAMES` 的严格次序是 `[fl,fr,bl,br]_steering` 4 维、对应 wheel_drive 4 维、`arm_joint1..6` 6 维，共 14 维。位置动作 10 维＝4 转向＋6 臂，轮驱动不使用轮转角作为 position error，见 `ranger_unifp_vec_env.py:34–37,76–108`。
- 动作先乘每维 motor strength；转向与臂目标是 `q_default + 0.25*a`，四轮目标速度是 `5*a rad/s`，见 `ranger_unifp_vec_env.py:100–108`。这里的 0.25 是默认姿态偏移幅度，不是增量积分；轮 `5 rad/s` 在 `0.1 m` 接触半径下对应 `0.5 m/s`。
- 转向内环 `kp=300,kd=20,effort=35 Nm`，轮内环速度 `kd=20,effort=80 Nm`，臂内环 `kp=80,kd=4`、力矩限值 `[40,40,27,7,7,7] Nm`。位置/速度目标由 PhysX implicit drive 实现，奖励用的 PD 力矩估计不等于独立测量的电机力矩，见 `ranger_unifp_vec_env.py:89–115`、`backend_envelope.py:97–104,312–315`。
- 夹爪两个自由度不属于 14 维策略动作。`end_effector_link` 是 `arm_seg6` 的零变换固定子链接，因此当前把 `arm_seg6` 作为末端测量/施力 body 与这个零偏置 frame 一致；它不等于已标定真实工具接触点，见可动 URDF `258–267`，`ranger_unifp_vec_env.py:121,254`。

## 3. Actor／辅助估计器的实际接口

- Ranger actor 单帧为 58 维：roll/pitch 2＋base angular velocity 3＋position error 10＋joint velocity 14＋previous/current-action slot 14＋commands 15。无 gait sin/cos，32 帧 FIFO 共 `1856` 维。见 `ranger_unifp_vec_env.py:76–85,227–230`、`unifp_observations.py:42–63,78–101`。
- 15 维命令依次为底盘速度 3、球坐标末端目标 3、姿态偏置 3、末端命令力 3、基座命令力 3，缩放依次见 `unifp_observations.py:18–25`。这些槽存在不代表每个槽都有有效任务激励或直接控制目标。
- 当前 Actor 不直接输入仿真真值外力、真实 base linear velocity 或真实 EE position；估计器以历史本体观测和命令共同推断这些量。Actor 也未输入轮电机电流、驱动器估算力矩、odom 或接触传感器读数；详见上述拼接源码。
- 官方共享网络将 `1856` 维历史送入 encoder `[512,256,128]→64 latent`，decoder `64→[128,64]→12`，Actor 用当前 58 维＋64 latent 输出 14 维 mean。Actor 输入的是隐变量，不是 decoder 预测的 12 维物理量直接拼接，见 `UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:43–93,143–166,177–180`。
- 12 维监督目标是 base velocity 3、EE spherical position 3、EE force 3、base force 3。前两组权重 0.2、力两组 1.0；实际代码在 MSE 的两侧都乘 weight，等效系数平方。见 `actor_critic.py:13–15`，`ppo.py:183–196`。
- 监督标签由 `RangerPrivilegedInputs.observe()` 同一次状态观测产生：base velocity body frame ×2；EE position 在 yaw frame/goal center 下转 spherical 后缩放；外力由 world 转 yaw frame再 ×0.01。见 `ranger_unifp_vec_env.py:131–149`。base force 在当前任务一直为零，相关字段仍保留。

- 本轮对用户本地冻结checkpoint只读加载，`model_state_dict['std']`在8000→16000变化为：转向范围`0.6401..0.6519→1.3579..1.3939`；轮速范围`0.2411..0.2496→0.8158..0.8349`；机械臂范围`0.3867..0.4851→1.0021..1.0875`。经动作比例映射，16000的随机sample单步1σ约为转向目标`0.3395..0.3485 rad`、轮目标速度`4.079..4.174 rad/s`、臂目标`0.2505..0.2719 rad`（未额外乘motor随机幅值）。这说明sample含显著动作噪声，单凭结果差异不能认定该噪声就是退化原因。

## 4. Critic／PPO／梯度链

- Critic 单帧98维、3帧共294维。字段为监督目标12、mass/COM delta5、friction1、motor-strength delta14、wheel contact4、gravity3、angular velocity3、joint position10、joint velocity14、action14、commands15、force-offset goal3，见 `ranger_unifp_vec_env.py:38–46,118–155`。
- wheel_contact 是轮 net force 的 world Z 分量 `>5 N`，不等于滑移估计，也不是无条件可部署 Actor 信号，见 `ranger_unifp_vec_env.py:144`。
- Critic 是 `[512,256,128]→1`，learner用5 epoch×4 minibatch、初始LR1e-3/自适应KL0.01、gamma0.99、lambda0.95、entropy0.01，辅助LR1e-5，24步rollout。见 `ranger_512_fixedcom_baseline.py:95–104`，`ppo.py:12–15,131–144`。
- 原生 PPO actor 分布是独立14维 Normal，log probability 与 entropy 在14维上求和；PPO 内没有 body/arm 的 `8/6` 特殊分组或分别优化。见 `actor_critic.py:113–153`，`ppo.py:119–170`。8/6 分组只在执行器索引、奖励切片及 morphology adapter 中出现。
- PPO 的 Adam 与辅助 Adam 都以整模型参数构造；PPO 梯度通过 actor latent 同时更新 encoder，辅助 loss 梯度经decoder更新encoder/decoder。辅助步不会因为优化器包含actor参数就自动产生actor head梯度。不能把现状写为“估计器与策略完全解耦”，见 `ppo.py:53–55,164–200`、`actor_critic.py:143–145,177–180`。
- 正常 step 仅对策略动作数值 clamp 到 `[-100,100]`，不是按机器人真实关节/轮速/加速度的部署限幅，见 `unifp_vec_env.py:304–309`。本轮未发现 Ranger ROS2/CAN、运动模式切换、watchdog 和真机 supervisor 的完整控制链实现。

## 5. 课程、目标、奖励与随机化

- 最终512训练不是根模块默认 `±15 N`；`native_backend(...max_force_n=60,fixed_com=True,official_shared=True)` 开启官方共享桥，并在192001 global policy step后施加每轴±60N的独立命令/外扰流。见 `ranger_512_fixedcom_baseline.py:80–84`，`force_stage_runner.py:78–80`，`audit_snapshot.py:1018–1077`。
- shared bridge 用官方 `_push_gripper`、force reset block、速度命令重采样、目标轨迹/碰撞拒绝、适用奖励、终止与 push 方法。它不是单纯根目录 `UniFPForceCurriculum` 的默认实现，见 `audit_snapshot.py:1030–1089,1090–1141,1142–1255`。根模块与冻结训练路径必须区别。
- shared bridge保留motor strength跨reset固定，根模块原默认却在reset重采样；见 `audit_snapshot.py:1079–1084` 和 `unifp_domain_randomization.py:96–101`。当前固定COM训练仍保留质量0..15kg、末端负载0..0.2kg、摩擦0.3..2.0、motor幅值0.85..1.15、观测噪声、8s一次底盘速度push，见 `baseline_plan.json` 和 `unifp_domain_randomization.py:33–67`；COM固定为有意实验偏离，不是完整域随机化已验证。
- A1目标球域 `radius .35..75m,elevation .15..1.4rad,azimuth -1.2..1.2rad`；goal center offset为`(.174,0,ROOT_HEIGHT+.09535)`，见 `ranger_unifp_vec_env.py:258–266`。可动URDF中安装架offset -0.33m、臂安装高 .42535m，组合为 .09535m，见 URDF `264–267,493–497`。这一中心不是未扣除安装架的 .42535m。
- 当前保留17项适用奖励，删除四足gait/airtime项；base部分切片8、arm部分切片6，见 `ranger_unifp_vec_env.py:179–213` 和 `audit_snapshot.py:1142–1186`。列表有末端位置跟踪，无独立姿态追踪奖励；有wheel/arm joint速度、加速度、动作差、力矩罚；没有显式轮地滑移、转向一致性、接触法向力任务奖励。
- 末端受力偏移目标为 `p_goal + (F_external_world + R_yaw*F_command_yaw)/Kp`，当前Kp=200N/m。每轴命令、外扰都可±60N，因此同向叠加每轴偏移最高可到0.6m；这是代数上界，不代表每个样本均达到，也没有证明偏移后目标一定可达。见 `ranger_unifp_vec_env.py:140`，`audit_snapshot.py:1051–1052`，`evaluate_stage_verified.py:143–146`。
- 当前目标拒绝是原球域、包围盒/插值采样的迁移，并非完整 IK 可达性、自碰撞、力矩能力证明。冻结baseline计划的limitations也明确写未做全域IK证书；当前自碰撞关闭、平地限定。

## 6. 旧迁移验证的准确口径与结果

- `mean` 指推理时直接执行高斯策略的确定性均值，`sample` 指在同样的均值上叠加 `model.std*Normal(0,1)`；不是不同力分布，也不是“多次实验均值”开关，见 `evaluate_stage_verified.py:171–175`。
- 每个checkpoint组合2个验证seed `266391/266392`、32环境×1000策略步（20s），分别跑zero/force与mean/sample；模型只读、独立仿真进程，不更新optimizer。见 `evaluate_stage_verified.py:44–47,112–126,155–200`。
- EE误差统计是每样本欧氏距离的均值（cm）；`ee_force_target_error` 是相对受力偏移目标，`ee_error` 是原几何目标。force RMSE per-axis 是所有合格样本与XYZ元素的平均平方再开方；只算history≥32的样本。见 `evaluate_stage_verified.py:59–100,138–149,165–183`。它与官方B2新实验XYZ MAE或向量L2均值不可直接混用。
- 保存的 `protected_log_sources/.../gym_ranger_force_stage_20261028_v1/evaluation_summary.json`含8000/9000各8条验证，原始trace已清理。直接从其中2 seed算术平均可得到：

|检查点/模式/条件|原目标EE均距cm|偏移目标EE均距cm|XY RMSE m/s|yaw RMSE rad/s|估力RMSE N/轴|恒零基线N/轴|
|---|---:|---:|---:|---:|---:|---:|
|8000 mean zero|4.2472|4.2472|0.07846|0.10994|0.06607|0|
|8000 sample zero|3.0136|3.0136|0.08544|0.13359|0.03945|0|
|8000 mean force|5.1695|18.3511|0.08028|0.11169|17.75794|17.75817|
|9000 mean zero|17.3207|17.3207|0.10328|0.15440|23.02449|0|
|9000 mean force|20.9471|20.2170|0.11561|0.16809|22.04961|17.75817|
|9000 sample force|16.6018|15.1404|0.12050|0.18502|9.74726|17.75817|

- `PROJECT_REPORT.md:7–11`及`SUMMARY.json`保存训练终点16000=8000无力＋8000受力、单训练seed266281、512环境。`PROJECT_REPORT.md:21–34`的终点两验证seed算术平均如下：

|动作模式|偏移目标EE均距cm|XY RMSE m/s|yaw RMSE rad/s|估力RMSE N/轴|恒零基线N/轴|
|---|---:|---:|---:|---:|---:|
|mean|50.348|0.6393|0.5566|27.702|17.713|
|sample|45.795|0.5607|0.6227|11.271|17.846|

- 报告记录sample受力位置最佳验证checkpoint10000约13.956cm，但仅验证选点，没有独立未见测试。保存的总结认为估计器学到部分信息，但未转成稳定受力控制；这与较差终点指标一致。报告未将退化唯一归因于resume；resume恢复模型、两个Adam、LR、计数器、Python/NumPy/Torch RNG，但物理、task、history和脉冲重建。

## 7. 清理后可执行性与数据存续

- 依现存位置求值，`ranger_512_migration/logic/ranger_512_fixedcom_baseline.py:27–31` 的 `ROOT=AUDIT.parents[2]` 实为 `/home/sorry/python_ws`，而不是项目根目录；同文件 `plan.json`读取点在58/234行等，但目录当前只有`baseline_plan.json`和`final_plan.json`，无`plan.json`。
- `logic/evaluate_stage_verified.py:17–20,37–40`指向`ranger_512_migration/gym_ranger_fresh512_fixedcom_20261028_v1/run.py`，此路径不存在。
- `logic/backend_envelope.py:36–38`和对应`audit_snapshot.py`仍从项目内 `IsaacLab/source/.../math.py`提取函数；当前目录不存在。`backend_envelope.py:53–60`仍需要旧`gym_ranger_20261024/lab_transport_defaults/result.json/shape_properties.pt`和本机IsaacSim schema绝对路径；前述旧目录不存在。保存的`logic/shape_properties_result.json/shape_properties.pt`有类似原始证据，但没有自动重新定位接口。
- `logic/ranger_force_continue_13000_to_16000.py:17–25`在导入时读取`HERE/plan.json`，现路径不存在，并引用旧实验目录。保护hash表中旧路径也包含已删除logs。
- 父任务只读重算checkpoint哈希：最终16000与summary记录匹配；8000实际SHA256为`3ed0adc13c3dc84b87f82606160c3eb0df6e52727fb34ea089f96c686b40110e`，`SUMMARY.json`的baseline条目漏掉末尾`e`，只有63字符。保存的8000/9000评估摘要保有正确64字符hash。这是清单录入错误的证据，不能据此判定权重损坏。
- 因此，文件/权重/摘要可作为审计材料，但当前清理后的迁移包不能据此认定已经具备原位重跑能力。这一结论来自路径静态求值和存在性检查，本轮没有运行训练入口。

## 8. 施力点与力矩定义

- Gym运输层把给定body force和torque写入全刚体buffer，再调用`gym.apply_rigid_body_force_tensors(...,GLOBAL_SPACE)`，没有施力位置参数；见`backend_envelope.py:222–226,264–267`。本地安装包官方API文档`/home/sorry/.cache/unifp/isaacgym_preview4/pkg/isaacgym/docs/programming/tensors.html:391–406`明确该API作用于刚体COM；`apply_rigid_body_force_at_pos_tensors`才支持给定施力点并计算偏置力矩。因此当前受力训练是在`arm_seg6`刚体COM施加纯外力、显式附加moment为零，不是已经在TCP接触点施力的训练。
- 当前`end_effector_link`与`arm_seg6`原点零变换，但`arm_seg6`原点与自身COM并不重合：可动URDF `170–174`给COM约`(0,0,-0.038796)m`。夹爪/工具真实接触点又需另外定义。任务位置误差点、力标签点、施力点与力矩参考点是不同接口。

## 9. 审计界限

本轮没有硬件I/O实测、没有验证固件能接受独立4转向＋4轮速控制，也没有读取到可证明轮胎参数/全身惯性已经实测标定的资料。现有summary的历史完整性声明不等同于当前仍保有全部原始数据；现有README还保留旧Lab10000/±15N说明，而最终冻结报告是Gym512/±60N/固定COM16000，两者不能合并为一个实验。
