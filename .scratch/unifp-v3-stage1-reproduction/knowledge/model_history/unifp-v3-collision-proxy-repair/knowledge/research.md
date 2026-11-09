# 源码与实验依据

- 活动资产路径由`UniFP_v3/legged_gym/envs/v3/v3_pos_force_config.py:25—37`指定为`v3_a1_desktop_train_v6_coplanar_r2`；它是本次只读基线，不覆盖。
- 终止规则在`UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:495—510`：绝对roll>0.8rad、绝对pitch>1.0rad或base_link净接触力>1N时终止。
- 训练shape从桌面CAD的143个base collision源组件按语义归组为凸代理，生成规则见`UniFP_v3/tools/build_training_collision_asset.py:100—168, 201—238`。
- 当前V6的`asset_manifest.json`中`base_link_proxy_1`映射到19个桌面CAD source_components；当前shape总数158。`base_link_proxy_13`是碰撞XML名称，V6当前local_shape索引16、r3索引26，均映射到`[14,22,37,43,44,66]`。V6索引13／r3索引23才是保留组件source5；名称后缀、local_shape和source_component是三个不同索引域，不能相互替代。
- `UniFP_v3/tools/build_rolling_collision_asset.py:80—175`展示了从原始source component mapping拆分低位轮区代理、同步重映射shape index和manifest的方法。
- 最新原生Gym终止观测保存在`.scratch/unifp-v3-failure-visualization/artifacts/terminal_pose_probe/failures.json`；Gym原生DOF顺序在`static_flip_probe/summary.json`中记录。
- `.scratch/unifp-v3-failure-visualization/artifacts/native_terminal_geometry_audit.json`用实际终止刚体pose与FCL比对桌面视觉／碰撞和训练碰撞：step454视觉间距6.921mm，训练底盘proxy相交，桌面base碰撞与桌面腕部碰撞间距2.232mm；把当前训练base proxy换上桌面腕部碰撞仍相交，而桌面base与训练腕部碰撞间距2.380mm。
- step158视觉间距1.687mm，当前训练代理间距0.0445mm；桌面基准collision也相交，说明contact margin和源collision表达需保留诊断，不能预设全由代理造成。
- actor实际shape `contact_offset=0.01m`、`rest_offset=0m`。PhysX接触力读取位于V3 step完成后；本次记录是20ms策略边界，不能重建每个2.5ms子步。
- 活动URDF视觉坐标修复已在独立视频回放使用`flip_visual_attachments=False`验证；本修复不改视觉附件显示标志或正式V3配置。

## V3 仿真创建顺序与候选原生崩溃

- `UniFP_v3/legged_gym/envs/base/base_task.py:55—58`在`create_sim()`返回后立即调用`prepare_sim()`。
- `UniFP_v3/legged_gym/envs/base/legged_robot.py:21—47`先调用`BaseTask.__init__`，之后才创建PyTorch物理状态buffer；`legged_robot.py:193—200`的`create_sim()`先创建地面，再创建actor。
- `UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:82—100`在基类创建actor后安装装配shape filter和creation randomization；当前没有在`prepare_sim()`前设置DOF状态。
- V3配置默认臂姿态为`arm_joint2=1.0 rad`、`arm_joint3=-1.5 rad`，其余12个转向、驱动和臂关节为`0 rad`，来源为`UniFP_v3/legged_gym/envs/v3/v3_pos_force_config.py:39—54`。
- `legged_robot_v3.py:344—378`调用基类构造；在基类已经完成`prepare_sim()`及buffer初始化后，V3才于`legged_robot_v3.py:449—451`首次调用`reset_idx()`。
- 上一轮总结记载“prepare前写入默认姿态可通过”，但缺少覆盖销毁阶段的原始日志，不能作为已证实根因。本轮同一候选完整销毁测试仍失败，见下方独立CPU/GPU对照。远程实际显卡是RTX A6000；此前将A8000训练阶段名称误写成GPU型号，已纠正。
- 本机RTX 5060 Laptop GPU的计算能力`sm_120`不受缓存环境PyTorch 2.0.1+cu117的CUDA kernel支持；该限制针对PyTorch GPU计算，不意味着CPU tensor配合GPU PhysX不可用。此前V6轮系诊断正是CPU tensor＋GPU PhysX运行，证据见`.scratch/unifp-v3-stage1-reproduction/knowledge/model_history/unifp-v3-a8000-diagnosis/knowledge/research.md:73`及其`artifacts/t02/v6_r2_forward.json`。

## 不导入PyTorch的CPU/GPU PhysX对照

- 探针为本任务`knowledge/probe_native_backend.py`：只导入Gym API、NumPy和现有装配filter函数，不导入PyTorch；CPU状态数组、零重力、2.5ms、2步、14 DOF、完整原生销毁。每次运行用独立进程，阶段日志与退出码保存在`artifacts/local_backend_20261106/`。
- CPU PhysX：V6默认姿态、V7默认姿态、V7全零姿态均通过创建、filter、prepare、simulate和销毁；最终DOF数值有限。
- GPU PhysX＋CPU状态数组：V6默认姿态完整通过；V7默认姿态即使不安装filter也在销毁阶段出现`free(): invalid size`，启用filter结果相同。故在该复现中PyTorch CUDA、filter和默认姿态前置均不是堆损坏的必要条件。
- 远程RTX A6000同一探针对V7仍发生native heap corruption，原始日志为`/mnt/data_8THDD/myl_unifp_v3_stage1/diagnostics/failure_visualization_20261106/native_backend/a6000_v7_before.log`，退出码134。远程解释器与SDK均实际存在，路径见根目录`AGENTS.md`，此前“远程缺运行时”的判断无效。
- V6的`base_link`有50个shape，V7有68个；总shape分别158和176。临时诊断变体移除部分新增shape，只用于定位：保留前9个新增shape或后10个新增shape分别通过；64、65、62、63个base shape变体失败，60个通过。原始日志和命令均保存在上述artifact目录；这些变体缺少部分碰撞，禁止训练或当作修复候选。
- 后续计数对照：保留V7的68个base shape但删除9个铝型材shape，总量167时通过；保留59个base shape但复制9个铝型材shape，总量176时失败；61个base shape、总量169时失败。因此不能解释成单个base刚体64-shape边界，当前观测与actor总shape数量相关，尚未证明通用硬上限或越界代码位置。
- `v7_gpu_zero_step_backtrace.log`记录未执行物理步（steps=0）也在销毁时出错，回溯为`physx::PxgAABBManager::~PxgAABBManager()` -> `Sc::Scene::release()` -> `NpScene::~NpScene()` -> `GymPhysX::~GymPhysX()`。这定位的是发现堆损坏的位置，不是首次写越界的指令。
- 完整GPU aggregate创建诊断亦失败，见`gpu_v7_aggregate.log`；没有据此修改业务创建流程。

## 实际V3 CPU tensor管线

- `knowledge/probe_cpu_tensor_v3.py`调用当前真实V3创建、`step()`和`reset_idx()`；仅禁用该诊断实例的随机化、push和噪声，`use_gpu_pipeline=False`。控制、状态检查、接触判据与正式V3保持一致，无PPO。
- V6 CPU tensor＋GPU PhysX、4环境、静置25步＋轮动作25步（每段0.5秒）通过；无物理失败，核心状态有限，局部reset的其他三环境root/DOF/command/history/episode计数变化均为0；模拟器完整销毁。
- V7 CPU tensor＋CPU PhysX、同样规模通过；全部176个actor shape均保留。该结果不能替代GPU训练运行时的准入。

## 保留全部部件的临时11组诊断

- 诊断夹具保存在`artifacts/grouped11_diagnostic/`，未注册到正式资源目录或配置。原19个目标源组件完整保留，按局部邻接合并为`[8] [10] [23] [25] [57] [99] [92,118] [26,72] [9,94] [42,65,40,93,114] [18,91]`共11组；各组为源V7局部凸包顶点并集的凸包，最大31顶点，总actor shape数168。
- `groups.json`保存从19到11的8次合并与新增凸包材料体积。分组按合并前后凸体并集的体积差选择；凸包合并会填补局部凹隙，不能只以体积小认定几何安全。
- 本机真实V3、CPU tensor＋GPU PhysX、4环境、静置250步＋轮动作250步（每段5秒）通过；无物理失败、base净力峰值0N、状态有限，局部reset不改变其他三环境的已检查行，完整销毁通过。证据`local_backend_20261106/v3_cpu_tensor_gpu_grouped11.json`。
- 9个历史失败姿态与2个原生终止姿态的FCL对照保存在`grouped11_all_nine_geometry.json`和`grouped11_native_geometry.json`。原生step454候选collision表面间距2.38290mm，桌面基准为2.23242mm，原错误凸包相交未恢复；step158候选间距1.73560mm。step1268和2066的已知collision相交仍保留，不宣称全部失败机制已修复。
- 夹具依赖本项目V7 r2非目标网格，只用于探索性诊断，不是正式独立候选身份；尚不能据上述短测宣称可训练。
- 远程完整GPU tensor管线4环境名义250步静置、250步轮动作、250步臂动作已通过，三项物理失败均0、局部reset的9项其他环境行差异均0、进程退出0。报告`remote_grouped11_gpu_pipeline_4env_r1.json`读回总质量83.3179115kg；与既有最终V6名义报告`unifp-v3-a8000-diagnosis/artifacts/t04/final_nominal32.json`逐刚体比较mass/COM/inertia，所有差异<=1e-8。

## 临时11组128环境策略采样

- 相同1400轮checkpoint、seed1001、128环境、sample动作、关闭push（S1）运行20秒（1000个policy step），使用保存物性前128行、冻结源码身份核验和零策略更新；自碰撞和1N判据保持，GPU进程正常退出。候选3次base接触终止，旧V6相同关闭push设置的前20秒为2次（step158/454）。新轨迹不是历史训练重放，不能由该小样本声称总体失败率改善或恶化。
- 新事件为step423/env41、480/env51、766/env96；base净力为76.3052、14.2981、12.3451N，均与腕部相机净力逐分量反向，无roll/pitch越界。
- `grouped11_new_failures_geometry.json`的独立FCL审计显示三者最近base表面均是`base_grouped_0`，对应未与其他部件合并的原source component 8。step423/480训练collision与桌面基准collision均相交；step766训练collision间距0.01160mm，桌面基准collision仍相交。视觉表面间距分别4.432、4.564、2.151mm。不是这8次凸包合并造成的已证实新填隙；桌面源collision与视觉包络间的剩余差异仍存在。

## 正式11组r3生成与静态合同

- `knowledge/build_local_base_proxy_candidate.py:28—30`保存已确认固定11组；`32—80`从桌面源collision逐部件生成紧凑凸包，按float32顶点边界再组合，组内最多31顶点。`119—150`核对目标19个源组件唯一完整覆盖，仅替换V6 base local shape1。
- 同文件`154—181`同步当前shape索引；`193—199`保存V6上游来源、设置实际V6直接来源manifest哈希，注明继承历史refinement记录的索引属于当时版本。`202—213`写168总shape、分组与网格哈希。
- `knowledge/test_local_base_proxy_candidate.py`5项合同已通过，证据`artifacts/grouped_r3_green_tests.log`。自包含r3资产为`UniFP_v3/resources/robots/v3_a1_desktop_train_v7_local_base_proxy_r3/`；201个文件传输远程后逐项SHA256相同，见`artifacts/grouped_r3_acceptance/asset_files_sha256.json`。
- 最终r3的11个保存姿态FCL结果在该目录`nine_geometry.json`与`native_geometry.json`。原生step454表面间距为2.38289775mm，桌面collision为2.23241885mm；step1268仍由source99对应单部件组相交，step2066仍由非目标`base_link_proxy_13`与`arm_seg4_proxy_1`相交。
- 本机4环境CPU tensor＋GPU PhysX、250步静置＋250步轮动作均0失败、局部reset其他行差异0、状态有限、完整销毁退出0；远程4环境完整GPU管线静置／轮／臂各250步同样0失败，局部reset9项差异均0、退出0。最终r3所有28个body的mass/COM/inertia与V6名义报告最大差值均为0，总质量83.3179115088416kg。证据为`local_cpu_tensor_gpu.json`、`remote_nominal_4env.json`、`actor_properties_comparison.json`。
- 最终r3双向运动报告`wheels_sign_1.json`、`wheels_sign_-1.json`保持2.5ms×8配置，8项主轴误差均通过既有`control_candidate_gate.json`门槛；base净力峰值均0。主轴平移MAE0.00739—0.01479m/s、偏航MAE0.04614／0.03840rad/s，汇总`movement_gate.json`。
- 最终r3两个1400轮策略128×20秒sample均退出0，各3次base_contact和125次timeout，无roll/pitch触发，策略更新0、冻结源码身份核验通过。S1事件423/480/766，保留push事件423/481/765；V6既有相同配置前20秒各2次，逐窗口提取依据在`v6_first20s_baseline.json`。本次事件数没有显示总体失败改善。
- 最终S1三次事件的FCL为`new_failures_geometry.json`：均最近source8；视觉间距4.431665／4.564363／2.150734mm，前两次候选collision相交，第三次间距0.011605mm，三次桌面源collision都相交。`acceptance_summary.json`分别记录静态／目标几何／运行时通过及`physical_failure_reduction_demonstrated=false`。

## P0局部包络及逐侧替换事实（2026-11-06）

- 只读诊断入口为`knowledge/audit_local_envelopes.py:188`，调用既有`unifp-v3-failure-visualization/knowledge/audit_failure_geometry.py:24,60`的FCL加载／配对接口；`audit_local_envelopes.py:46`读取同一URDF origin与mesh scale生成截面网格，`:81`按碰撞名称查当前local_shape和源组件，`:109`输出±30mm法向射线交点，`:127`输出局部截面PNG/PDF。所有长度计算在base_link米制坐标下进行，射线和图片展示单位为mm。
- r3 URDF的`:577`、`:607`、`:727`分别为source8对应`base_link_local_1_group_0`、source99对应`base_link_local_1_group_5`及`base_link_proxy_13`。当前索引分别1、6、26（从0计）；原proxy13的源组件为`[14,22,37,43,44,66]`。`base_link_solid_5`在r3索引23，与proxy13不同。
- 桌面URDF的`:4411`、`:4957`分别是source8/source99；base视觉与这两个源collision均为`rpy=3.14000 0 0`且scale为1。r3对应凸包则已将该变换烘焙进顶点、XML origin为0。本次未发现需要以重复应用origin或单位scale解释的加载差异；没有追溯原CAD的collision生成算法。
- 7个输入case固定为423、480、766、1268、2066、158、454，共63项逐侧替换；visual取完整link，desktop取目标manifest映射源组件，r3取目标shape。逐姿态的visual/r3间距及相交判定均与既有完整link审计一致，数值容差1e-8m。输入姿态来源及全矩阵见`artifacts/envelope_audit_20261106/envelope_audit.json`；1268/2066为URDF FK，其余为保存的原生终止刚体pose。
- 以底盘视觉最近点为0、指向对侧视觉最近点为法向，同一射线上source8在423/480/766的源碰撞接触侧壳面位置分别+4.266218/+4.713215/+3.598473mm，r3为+4.264403/+4.712175/+0.819824mm。source99在1268的源/r3位置为+3.902949/+3.902610mm。以上是单射线壳面交点，不是全局膨胀误差、穿透深度或最小间距。
- 423中r3底盘＋相机视觉分离0.167257mm，而r3＋r3相交；480和1268中仅替换底盘为源或r3碰撞、保持对侧视觉就相交。766中源底盘＋视觉相交，r3底盘＋视觉也在别处相交，但r3＋r3分离0.011605mm。这些差异不能用单侧或单条射线统一概括。
- 2066的桌面源相交对为`base_link_solid_43`＋`arm_seg4_solid_50`；source43＋机械臂视觉亦相交。r3底盘＋机械臂视觉分离0.190040mm，r3双方相交。`arm_seg4_proxy_1`当前局部索引1，映射`[0,5,7,13,28,33,44,45,50]`。同一视觉法线上源底盘/r3壳面位置为+3.700223/+2.842799mm。
- 158中r3间距1.735598mm；454中r3间距2.382898mm、桌面源2.232419mm，已修复假相交未恢复。FCL不重建10mm contact_offset或此前2.5ms子步，故正间距不能独立解释／排除原生接触力，也不证明实体不包含。
- 输出包含7组PNG/PDF、68份输入文件SHA256、依赖版本、每个姿态来源和局部坐标基。原始输入运行后重算哈希未变；`verification.json`记录全部201个r3文件与已有资产哈希清单一致、63项矩阵和14张图存在、5项既有资产合同全部通过。本次没有修改资产、Gym业务实现、参数或训练状态。

## P1观测接缝的当前源码事实

- `UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:477—487`每个策略间隔循环8次：计算`_compute_v3_torques`→提交力矩→simulate→刷新DOF；CPU管线阻塞fetch，GPU管线不在该处显式fetch。`:489—491`在循环后刷新root、净接触力及刚体状态，`:505`在reset前进入`_accumulate_episode_metrics`。
- `legged_robot_v3.py:1302—1341`的显式力矩只使用动作、执行缩放、DOF位置/速度、增益与限值；不读取刚体pose或净接触力。故在下一次力矩计算入口观测已完成simulate的刚体pose和接触力，与控制反馈使用的DOF刷新接缝不同。
- `v3_pos_force_config.py:178`设置PhysX内部substeps为1，正式dt为2.5ms、策略decimation为8；`legged_robot_config.py:181`的contact_collection=2指一次simulate内部的全部子步，不能直接解读为8次策略decimation的合计。
- 既有独立回放`unifp-v3-failure-visualization/knowledge/replay_v3.py:43—88`在reset前保存终止证据；`:140—177`核验冻结源码/检查点身份并构造实例；其`_accumulate_episode_metrics`接缝与真实环境一致。既有回放没有每次simulate后的观测序列。

## P1实际子步证据

- 新增观测器`knowledge/observe_contact_substeps.py`只继承既有独立回放，在下一次`_compute_v3_torques`之前读取上一次simulate结果，并在`_accumulate_episode_metrics`读取最后一次结果；原力矩与episode结算委托父类。环形缓存最多16帧，只有失败行落盘，不调用状态setter或学习器。实际配置通过断言核对dt=0.0025s、decimation=8、PhysX内部substeps=1。
- 原始证据在`artifacts/envelope_audit_20261106/contact_substeps/`：远程4环境2步冒烟通过，V6/r3各128环境×20秒，2/3次终止，零更新、身份核验通过、完整销毁退出0。每事件16帧，共80帧。对应5次终止的原生刚体pose全部分量和base净力向量与原先相同事件记录差异均0；这是这5个边界的比较，不是全轨迹逐状态同一性证明。
- `knowledge/analyze_contact_substeps.py`将每帧世界系xyz+xyzw变换到base_link，再调用既有FCL接口。80个采样点的visual均无表面相交；V6158/454和r3 423/480/766各自visual最小间距为1.615813／6.798598／4.255094／4.564363／1.999638mm；collision相交帧数为3／11／6／1／7。
- V6 step158在相对终止边界-10/-7.5/-5ms表面相交，峰值力129.263961N；-2.5ms间距0.006763mm、0ms间距0.044509mm且力5.230741N。r3 step766在-35/-32.5/-17.5/-15/-7.5/-5/-2.5ms相交，峰值128.110461N；0ms间距0.011605mm且力12.345142N。终止边界静态分离不代表此前子步未相交。
- r3 step423在-15ms的FCL间距0.085793mm、力21.289727N；随后帧相交。每帧是simulate结束后的几何与求解器净力，不包含求解前姿态或内部接触流形；未唯一判定margin／求解时序／cooking／缓存各自贡献，也未把contact_collection=2解释为8次decimation力累加。
- `remote_tools_sha256.txt`核对观测器、启动包装和原回放与本机一致；`remote_r3_asset_sha256.json`的201个文件与此前完整清单逐项相等。没有覆盖冻结源码、资产、检查点或原输出，也没有修改正式配置。

## 后续局部视觉对应的输入事实

- `artifacts/envelope_audit_20261106/visual_correspondence.json`保存桌面base视觉输入哈希与结构。该GLB只有1个scene geometry、34932个三角面；合并重合顶点后15691个顶点、343个连通表面。该数目不是CAD零件数，场景未提供source_component到独立visual geometry的一对一对应。
- 桌面source8／source99／source43 collision均为凸体，原始网格分别70／53／123顶点；各自bounds和SHA256已记录于上述JSON。source8覆盖约0.46m长的局部区域，而视觉中相邻连通表面可能更大，不能根据完整连通表面名称推断对应源部件。

## P2接口所依赖的现有源码

- `knowledge/build_local_base_proxy_candidate.py:23—30`固定输入为V6／桌面、输出为r3、支持近似误差3mm；`:83—108`的`build_candidate(output_root)`拒绝已有目录，再复制来源资产。该生成器交付的是已有11组r3身份，不包含视觉裁切逻辑。
- `UniFP_v3/tools/build_training_collision_asset.py:32—49`的`collision_points`加载mesh、应用scale和origin，返回link-local米制`[V,3]`点集；`:52—85`的`compact_hull`从源凸包顶点中取子集，至多64顶点，返回未被该近似包住的源点到支持平面的最大正距离。该算法本身不是视觉覆盖保证，3mm默认值也不是0.5mm视觉外扩精度。
- `knowledge/build_local_base_proxy_candidate.py:191—213`设置独立候选状态、直接来源及上游哈希、历史shape映射和生成器哈希。旧`local_base_shape_map`的语义为V6到r3，非通用“任意直接来源到当前版本”。
- `UniFP_v3/legged_gym/utils/v3_asset_collision.py:26—58`使用manifest的`collision_solid_counts`、`source_groups`、`visual_clearance_records`及`rolling_collision_records`构造装配filter，不读取`proxy_records.support_error_m`作为运行时判据。
- `.scratch/unifp-v3-failure-visualization/knowledge/replay_v3.py:141—147`的`--asset_override`读取候选manifest，要求`asset_status=independent_local_base_proxy_repair_candidate`，再覆盖该诊断实例的URDF路径；正式配置文件未被该参数写入。

## V8已实施接口与离线结果

- `knowledge/build_visual_envelope_candidate.py:38`裁切完整三角面；`:80`以覆盖半空间重建凸体，最多34面／64顶点；`:166`固定r3与桌面输入并生成独立V8。支持面外移0.5mm，proxy13最终合并另外五源组件时margin=0，最终GLB回读核对覆盖。来源及直接／历史映射分别记录在候选manifest，最后写manifest。
- `knowledge/test_visual_envelope_candidate.py`的6项测试已通过；独立参考使用Trimesh三角面裁切及SciPy视觉变换。Shapely／FCL仅在隔离临时路径`/tmp/unifp-v3-geometry-fcl`，未修改tavis或Gym依赖。正式候选201文件身份见`artifacts/visual_envelope_v8_20261106/asset_files_sha256.json`。
- `knowledge/audit_visual_envelope_regression.py`使用已有BVH表面接口与FCL Convex实体对照，读取14个保存姿态及80个native子步，不从关节位置重建替代原生pose。全部V8表面／实体均无相交；三处真实视觉正对照分别沿最近点方向平移对侧刚体，先核验裁切真实视觉面相交，再核验对应目标collision相交。两项集成测试通过，原始数值／位姿／输入哈希见`geometry_regression.json`。
- `knowledge/probe_cpu_tensor_v3.py`当前诊断按静置、轮系、小幅臂动作三段执行，读取实际actor物性、contact／rest offset与filter。V8本机CPU tensor＋GPU PhysX的4环境各5秒均0次物理终止，局部reset其他行最大差异0，168shape，完整销毁退出0。该本机工作树与远程冻结源码身份分别记录，不混称同源码。

## V8原生与首窗口实际证据

- `artifacts/visual_envelope_v8_20261106/runtime_acceptance.json`核对两端实际actor mass／COM／inertia与r3名义基线，全部最大绝对差为0；本机／远程三段各5秒0终止，reset隔离，8项运动全部通过。名义测试与保存随机物性的冻结回放属于不同实例。
- `v8_seed1001_20s/summary.json`记录3次物理终止／125次超时、0学习更新与正式10mm／0offset。`v8_initial_failure_analysis.json`记录终止原生pose与实际表面：296为两个gripper合计与base净力抵消，479／838为wrist_camera单对抵消；不能使用旧单对工具的next假设解释296。视觉间距分别约0.48mm、0.18mm和0.0115mm。
- `knowledge/summarize_visual_envelope_comparison.py:21`以最多5个已记录最大净力中非base的最少组合寻找1e-4N内抵消组，保留未能抵消事件为未解；不等同读取PhysX成对接触流形。`test_visual_envelope_comparison.py`使用这3个真实事件检验双夹爪与单相机，先红后绿。

## T11冻结策略对照与残余proxy13来源事实（2026-11-07）

- 九个隔离零更新实例按相同1400轮checkpoint、S1/sample、自碰撞、offset与128×60秒完成。`artifacts/visual_envelope_v8_20261106/comparison_summary.json`保存每seed的失败次数、完整episodes、净力组合分类、终止原生pose、实际URDF几何表面距离与所有输入身份。总数V6/r3/V8分别25/27/19，seed序列分别9/9/7、8/6/13、5/6/8。该计数是有限诊断，不是统计显著性证据或训练准入。
- 回放失败接触力存于`failures.json`的`largest_contacts`，最多五个刚体的净力，不是PhysX成对接触点。`identify_counterparts`搜索最少刚体子集，检验与base净力向量和是否在1e-4N内；V8 seed1001 step296双夹爪组合通过独立单测，camera单体对也通过。未满足则标作未确定，不猜测。
- `residual_proxy13_side_matrix.json`将V8 seed1003四个arm_seg4事件在相同原生pose分别核验source43重建凸体、proxy13完整组、五个保留source shape各自、arm视觉／代理。step97与1217完整proxy13碰撞，但source43-only与arm视觉间距10.581／9.104mm；保留source44／22分别与arm_seg4代理间距0.810／0.386mm。step67完整proxy13也近接触，source44与arm视觉约1.856mm。该矩阵说明完整组合shape存在部件间凸化填隙来源；不能用source43的0.5mm包络参数解释所有残余接触。当前没有生成拆分或新增shape资产。

- 同一逐侧矩阵进一步区分两类扩张：step67／1064的未紧凑化完整组合凸包已与arm视觉相交，而所有五保留组件逐个及source43-only均分离，证明组合填隙；step97未紧凑化完整组合凸包与arm代理间距0.810214mm、最终64顶点shape相交，step1217对应间距0.351585mm与0.004606mm，证明顶点预算下外包近似也贡献额外扩张。两类贡献需分别表述。
