# Ranger Mini 3 + A1Arm 近似仿真模型

**2026-10-22：UniFP 环境默认已改为厂家几何修正 v3。当前资产为 `urdf/a1arm_rgm3_approx_sim_v3.urdf` 和 `assets/v3/a1arm_rgm3_approx_sim_v3.usd`，来源、尺寸差异、生成命令和截图统一见 [项目报告第 5.4 节](/home/sorry/python_ws/interactive/unifp_reproduction/PROJECT_REPORT.md#section-5)。2026-10-22 重启训练后，旧独立估力/诊断脚本已从根目录移入 `archive/v3_restart_cleanup_20261022/root_sources_before_cleanup.zip`，下文相关脚本名表示历史位置。下文 v2 参数与实验保留作历史技术记录；不可套用到 v3。`RANGER_ASSET_VERSION=v2` 可恢复旧联合策略环境。**

## 文件

- 原始组合描述：`urdf/a1arm_rgm3.urdf`（本次未修改）
- 仿真模型生成器：`scripts/build_ranger_mini3_approx_sim_urdf.py`
- 历史 v2 可动模型：`urdf/a1arm_rgm3_approx_sim_v2.urdf`
- 历史 v2 Isaac Sim 转换结果：`assets/a1arm_rgm3_approx_sim_v2.usd`
- v1 的无版本后缀 URDF/USD 保留用于追溯；当前短测、驱动和外力实验默认加载 v2。
- Isaac Lab articulation 短测：项目根目录 `smoke_ranger_mini3_approx.py`
- 轮地驱动验证：项目根目录 `test_ranger_approx_drive.py`
- 2026-10-19 排障记录和运动数据：项目根目录 `artifacts/ranger_approx_drive/`
- v2 首轮驱动回归：项目根目录 `artifacts/ranger_approx_force/v2_drive_results.json`；恢复分关节限值后的回归：`artifacts/ranger_force_coverage_v1/drive_results.json`
- 配对外力测试与分析：项目根目录 `test_ranger_approx_force.py`、`analyze_ranger_approx_force.py`、`analyze_ranger_force_coverage.py`
- [外力测试完整报告](/home/sorry/python_ws/interactive/unifp_reproduction/artifacts/ranger_approx_force/REPORT.md)
- [多姿态/增益/限值覆盖测试报告](/home/sorry/python_ws/interactive/unifp_reproduction/artifacts/ranger_force_coverage_v1/REPORT.md)
- [外力估计训练基线：已覆盖姿态有效，未见姿态泛化失败](/home/sorry/python_ws/interactive/unifp_reproduction/artifacts/ranger_force_estimator_v1/REPORT.md)
- [第二版估计：扩展姿态、物理反算与学习残差](/home/sorry/python_ws/interactive/unifp_reproduction/artifacts/ranger_force_estimator_v2/REPORT.md)
- [硬件接口后续未完成任务](/home/sorry/python_ws/interactive/unifp_reproduction/introduction/RANGER_HARDWARE_INTERFACE_PENDING.md)

## 历史 v2 近似

- 四轮各拆为转向关节（绕 Z）和驱动滚动关节（绕局部 Y），历史 v2 轮心按轴距 0.494 m、轮距 0.364 m 布置（本次手册核对轮距为 0.370 m）。底盘是自由浮动根，不再使用规划用的 X/Y/偏航虚拟关节。
- 轮胎用半径 0.153 m、宽度 0.14 m 的圆柱作为碰撞与显示几何。该半径错误地参考了总成包围盒，不能视为轮胎测量值；v3 已改为厂家 CAD 约 0.10 m 半径，仍待实测。
- Ranger Mini 3 底盘链接总质量约束为手册标称的 75 kg。因没有公开分部件惯性，缺失底盘质量被集中估算到 `base_link`，并按 0.72 × 0.50 × 0.256 m 均匀盒体近似惯量；四个轮子仍沿用原 URDF 的 2.2786 kg/link。
- v2 修正轮胎质心与惯量：圆柱几何以驱动关节为中心，因此将旧 CAD 的偏移质心归零，并按均匀圆柱重算惯量（局部 Y 为轮轴）。v1 留用旧质心造成偏心轮，首轮无力基线出现非预期漂移；该轮实验仅保存在 `artifacts/ranger_approx_force/initial_invalid_diagnostic/`，不用于外力结论。
- URDF 中的 A1Arm、扩展架和相机约 8.318 kg 作为底盘额外上装，因此组合模型总质量约 83.318 kg。应以装配后实称值替换。
- 转向 effort 35 Nm、轮驱动 effort 80 Nm、最大关节速度和关节阻尼是数值稳定/首轮测试用的估计值，不是 AgileX 标定值；需要通过实机 command-to-motion 响应辨识。
- **历史设置：** `artifacts/ranger_approx_force/` 的首轮外力数据和驱动回归，使用六关节统一 40 Nm、Kp=80、Kd=4。旧数据不可称为分关节限值试验，详见旧报告及其 `torque_limit_audit.json`。
- **当前设置：** 脚本从 v2 URDF 读取并分别设置 `[40,40,27,7,7,7]` Nm；新增覆盖实验核验了 PhysX 限制，并以4姿态×3组PD×2种轮驱完成432个受力条件。力矩限值和增益仍需实机标定，详见新报告。
- 官方 Mini 3 有独立悬架，但本模型没有显式悬架关节/弹簧阻尼；当前是平整硬地上的简化四轮接触模型。
- 网格引用通过 `meshes/approx_sim/` 下的安全文件名符号链接，规避 Isaac URDF importer 对 `.visual.glb` 文件名的 USD prim 路径处理问题。

这版模型适合验证整机拓扑、关节控制接入和初步的平地轮式行为；**不应据此宣称已完成高保真动力学或力估计 sim-to-real 模型**。质量集中、无悬架、简化轮胎接触和未经测量的执行器限值都会影响外力响应。

## 重建与验证

```bash
cd /home/sorry/python_ws/interactive/unifp_reproduction
python robot_description/a1arm_description/scripts/build_ranger_mini3_approx_sim_urdf.py
OMNI_KIT_ACCEPT_EULA=Y /var/local/sorry/conda/envs/isaaclab-unifp/bin/python \
  IsaacLab/scripts/tools/convert_urdf.py \
  robot_description/a1arm_description/urdf/a1arm_rgm3_approx_sim_v2.urdf \
  robot_description/a1arm_description/assets/a1arm_rgm3_approx_sim_v2.usd \
  --joint-stiffness 0 --joint-damping 1 --joint-target-type none --merge-joints --headless
OMNI_KIT_ACCEPT_EULA=Y /var/local/sorry/conda/envs/isaaclab-unifp/bin/python \
  smoke_ranger_mini3_approx.py --headless
```

短测门槛：Isaac Lab 读取 16 个活动关节、17 个刚体；包括 4 个 steering 与 4 个 wheel-drive joints；初始基座高度设为约 0.26 m，并向全部 8 个轮系关节发送小幅目标。此前通过 80 个 PhysX 步，8 个关节状态均有可测变化。转换/加载仍有空 steering link visuals 的 unresolved-reference 警告，但未阻止 articulation 创建或物理步进。

GUI 已在 v1 上启动并检查过基本外观，截图在 `artifacts/ranger_approx_drive/settled_gui.png`；v2 本轮完成的是无窗口物理回归和外力测试，尚未重新检查 GUI。v2 每段 2.5 s 的驱动结果：直行约 0.738 m，侧移约 0.736 m，原地旋转约 1.201 rad，三项脚本门槛均通过。这验证基本运动方向与执行链路，不代表轨迹跟踪误差已经标定；旋转目标 0.6 rad/s × 2.5 s = 1.5 rad，实测仍有差距。

```bash
OMNI_KIT_ACCEPT_EULA=Y /var/local/sorry/conda/envs/isaaclab-unifp/bin/python \
  test_ranger_approx_drive.py --headless --output artifacts/ranger_force_new_run/drive_results.json
OMNI_KIT_ACCEPT_EULA=Y /var/local/sorry/conda/envs/isaaclab-unifp/bin/python \
  test_ranger_approx_force.py --headless --suite coverage --output-dir artifacts/ranger_force_new_run
/var/local/sorry/conda/envs/isaaclab-unifp/bin/python analyze_ranger_force_coverage.py \
  --output-dir artifacts/ranger_force_new_run
```

旧固定姿态外力数据：两种轮驱模式共36个受力条件和2个无力对照，均超过信号响应门槛；旧数据不可因当前脚本修订而重新归类。新覆盖实验为432个受力条件、24个独立无力对照，均超过仿真响应门槛，且没有记录到关节力矩请求接近配置上限；但低位伸展姿态在20 N负Z力下距关节限位仅约2°。两轮均未训练力估计器或 UniFP 策略。

当前外力脚本默认输出至 `artifacts/ranger_approx_force_limits/`，发现已有 `traces.npz` 时拒绝覆盖；使用 `--suite coverage` 时务必配合 `analyze_ranger_force_coverage.py`，并为新实验指定新目录。

2026-10-20新增 `--suite estimator`：独立随机姿态/三维梯形外力及全程零力试验，配合 `train_ranger_force_estimator.py` 做完整试验划分和未见姿态测试。已训练三套本体观测MLP；噪声/延迟为实验假设，未见姿态的零力误报严重，尚不可部署。硬件接口核对按用户要求列入后续任务。

同日第二版新增 `--suite estimator_broad` 和 `--physics --models arm physics_residual`：768条试验覆盖更大目标姿态范围，并留出二维目标区域；加入URDF重力/雅可比反算及学习残差。新目标区域名义误差约1.214 N，压力噪声条件下误报仍明显；详见第二版报告。第一版与第二版数据/权重分别保留。

同日第三版新增 `--suite estimator_robust`、`train_ranger_force_robust.py`：新采集768条试验，实际改变PhysX机械臂质量±10%、局部质心各轴±5 mm、有效PD增益±8%，并回读核验；输入仅使用名义参数。训练加入逐试验噪声和0/20/40 ms延迟随机化。新数据保留目标区域、0.10°/40 ms条件下，物理残差模型全程零力误报15.97%→1.18%，但2–5 N漏检28.61%→40.07%，尚未同时改善两者，保留第二版对照且不接入动作控制。详见[第三版报告](/home/sorry/python_ws/interactive/unifp_reproduction/artifacts/ranger_force_estimator_v3/REPORT.md)。所有版本数据和权重分别保存；本轮为无窗口仿真。

同日完成无接触启动校准实验：`--calibration-prefix-s 1.0`将施力延后1 s并延长记录；新采集768条完整试验，0.1–1.1 s拟合局部自重/名义伺服残差，之后冻结。新增`ranger_force_calibration.py`中的显式校准状态机，以及`train_ranger_force_calibrated.py`。新数据压力条件保留目标区域中，第三版原模型/第三版加输出校正/新校准残差模型分别为0.64%/0.44%/1.69%零力误报，2–5 N漏检51.45%/51.56%/46.82%；未同时解决两者，同域测试小力也未改善，因此不替换第三版或接入控制。详见[校准报告](/home/sorry/python_ws/interactive/unifp_reproduction/artifacts/ranger_force_calibration_v1/REPORT.md)。流式与离线三种观测条件最大差小于0.000005 N，持续受力不更新冻结基线。实机无接触声明与SDK仍待核实，本轮没有GUI。

同日完成因果降噪对照：新增`ranger_causal_filter.py`与`train_ranger_force_denoise.py`，比较移动平均、指数平滑和5点速度拟合；验证集选出的非恒等候选为`ols_velocity5`，并与原差分采用同初始化、噪声样本和训练流程配对重训。新种子20261024采集768条，仅用其中128条进行最终独立测试。0.10°白噪声下速度标准差0.06169→0.02764 rad/s；保留姿态压力测试RMSE1.974→1.923 N，小力漏检44.00%→43.39%，零力误报0.81%→0.89%，三帧确认整链路延迟中位数均140ms，尚未稳定同时改善检测指标。检查点内保存`arm_filter`，流式类自动读取。详见[因果降噪报告](/home/sorry/python_ws/interactive/unifp_reproduction/artifacts/ranger_force_denoise_matched_v1/REPORT.md)。本轮继续无窗口运行，未接入动作控制。

同日完成恒定小力诊断：新增`--suite estimator_hold`与`analyze_ranger_force_hold.py`，新种子20261025采集744条9s试验（720条1–5 N受力、24条零力），使用上一轮冻结的原差分/5点速度模型分解物理先验、网络修正和最终结果。无噪声恒力后期平均先验增益0.915，最终仅0.617/0.594，证实稳定小力被网络进一步压缩；压力条件下网络同时大幅抑制零力误报，不能直接删除。1 N引起的最大关节变化中位数0.199°、最弱0.041°，后者低于假设0.10°单帧噪声；不等于实机可测。保持数秒未见估计持续衰减至零，冻结基线未更新；正则化仅解释小部分先验低估。详见[恒定小力诊断报告](/home/sorry/python_ws/interactive/unifp_reproduction/artifacts/ranger_force_hold_v1/REPORT.md)。本轮未训练新模型，未改阈值或接入控制，仍为无窗口仿真。下一步应在另外划分的新数据上比较小力/长恒力训练和幅值约束，同时保留零力误报及短脉冲延迟回归门槛。

2026-10-20进一步采集独立混合力数据：480条训练/验证试验和新种子128条独立测试，覆盖零力、短双脉冲和1–5N长恒力。两套同结构、同初始化/归一化/噪声与批次的模型只比较普通MSE和额外小力幅值损失。预留姿态、0.10°/40ms条件下，普通MSE的恒力增益0.646、3–5N检出87.1%、平衡MSE5.232 N²；幅值约束依次为0.622、82.2%、5.855 N²，未达预期。旧模型继续保留，不替换检查点或接入控制。详情见[混合力独立测试报告](/home/sorry/python_ws/interactive/unifp_reproduction/artifacts/ranger_force_mixed_training_v1/REPORT.md)。本轮无窗口运行，实机接口仍待后续核实。

2026-10-20进一步实施小力可观测性诊断：仅复用种子20261026的训练/验证轨迹，在名义和压力噪声下分别进行12次噪声重采样。压力条件下普通MSE的验证集1–2N输出范数帧级AUC为0.806，但固定2N阈值且方向一致时小力帧命中率仅12.0%；物理先验在完整零力轨迹上的误报率为58.8%，普通MSE压至1.43%，同时存在幅值低估。数秒长恒力的关节平均位移有区分信号，不能推出实机或短脉冲同样可测。[可观测性报告](/home/sorry/python_ws/interactive/unifp_reproduction/artifacts/ranger_force_observability_v2/REPORT.md)。随后将训练观测噪声实现从3组增至12组，保持架构、损失、优化步数和验证步间隔不变。候选在验证集仅有约0.001 N²的综合分数改善，却增加零力误报且落后旧冻结基线，因此未进入新独立测试。[匹配训练报告](/home/sorry/python_ws/interactive/unifp_reproduction/artifacts/ranger_force_noise_diversity_v2/REPORT.md)。所有工作仍为headless仿真/离线分析，未接入控制或硬件。

2026-10-20进一步扩充独立物理轨迹：新增种子20261105采集480条训练/验证试验，安全筛查后保留383训练和96验证；与原种子20261026的384训练、96验证合并，并与仅原始轨迹的模型在相同网络、普通MSE、三组观测噪声和8130次更新预算下比较。测试前验证平衡MSE由匹配对照4.997降到4.035 N²；全新种子20261029的压力观测独立测试中，同域/预留姿态由5.174/4.515降到4.132/3.727 N²。同域稳定小力漏检29.2%→16.1%，但预留姿态28.7%仍高于旧冻结模型27.0%，不能认为跨姿态小力识别已解决。流式/离线输出差最大2.63×10⁻⁶ N，冻结校准和未来输入隔离测试通过。详见[物理扩充报告](/home/sorry/python_ws/interactive/unifp_reproduction/artifacts/ranger_force_physical_expansion_v1/REPORT.md)。当前仍为无窗口仿真，未接入控制或实机。

## 下一步校准

2026-10-20完成[固定容量稀疏长历史对照](/home/sorry/python_ws/interactive/unifp_reproduction/artifacts/ranger_force_history_v1/REPORT.md)：新采集1,440条且全部通过安全筛查（1,152训练、288验证），比较100/300/500ms的六个历史采样点×普通MSE或幅值＋恢复损失，共18个模型。各组159输入、37,379参数，启动校准和最新物理先验相同。压力新预留验证姿态的普通MSE中，100→300→500ms恢复MSE0.9716→0.9368→0.9277 N²，短脉冲漏检13.31%→15.81%→17.92%；500ms在6/6配对增加短脉冲漏检，较长历史候选均未通过全部门槛，未采集或使用20261222/20261223预留测试。36次流式核验最大差3.10×10⁻⁶ N，旧连续六帧检查点兼容。此为固定六点的稀疏历史，不能据此否定稠密长历史；下一轮应保留较密近处历史并比较时序编码。仍为无窗口仿真，未接入控制或设备。

2026-10-20完成[动态阶段单独消融](/home/sorry/python_ws/interactive/unifp_reproduction/artifacts/ranger_force_phase_v1/REPORT.md)：新物理种子20270110/20270111共1,440条，整对安全筛查后保留1,150训练、288验证；新预留验证区域q1∈[0.15,0.35]、q2∈[1.10,1.50]。六种方案各三模型种子，共18个模型，网络/六帧历史/预算不变。压力预留姿态中，相对仅幅值组，下降沿组和恢复段组均在6/6配对降低恢复MSE，同时在6/6配对增加稳定小力漏检；恢复段组为稳定漏检37.61%→38.75%、恢复MSE0.8700→0.8173 N²。四个动态候选仍未通过全部预设门槛，保留20261222/20261223未用测试。18个检查点、540次验证记录和36次流式核验通过，最大差约3.82×10⁻⁶ N。阶段时长不同，此比较固定单帧系数而非总权重；不能将组合损失的退化简单归因于单一阶段。下一轮更适合比较更长因果历史或显式动态状态，而不继续在本验证集搜索权重。当前仍为无窗口仿真，未接入设备或动作控制。

2026-10-20完成新一轮[成对零力与动态约束实验](/home/sorry/python_ws/interactive/unifp_reproduction/artifacts/ranger_force_dynamic_v1/REPORT.md)：新采集1,440条物理轨迹，整对安全筛查后保留1,150训练、288验证；普通MSE、幅值0.05/0.10/0.25、幅值0.10＋动态约束各训练三个种子，共15个模型。压力观测、预留验证姿态下，幅值0.10使稳定小力漏检36.30%→33.37%、短脉冲漏检23.63%→19.97%，但恢复MSE0.7205→0.7757 N²。动态项降低恢复误差但损失部分小力收益；四个候选均未通过事前固定的全部验证要求，因此未采集或使用新测试种子20261222/20261223，不替换现有模型。15个检查点、450次验证记录与30次流式/离线核验通过，最大输出差见报告。所有指标为本轮验证结果，姿态/数据分布与旧轮不同，绝对数值不可直接横比；仍为无窗口仿真，未接入控制或硬件。

定向多种子实验后完成[短脉冲诊断](/home/sorry/python_ws/interactive/unifp_reproduction/artifacts/ranger_force_targeted_v1/short_pulse_diagnostic_v1/REPORT.md)：原108组评估复现一致；压力观测预留姿态的候选22/216次事件评估漏检，全部未形成幅值连续三帧，方向条件没有单独阻断；其中只有1次能在事件后240 ms补充确认。单独移除40 ms延迟未减少漏检，单独移除角度噪声反而揭示更多幅值低估，需同时检查幅值裕量、短脉冲连续性和零力误报。本轮只读诊断，模型/阈值/原测试记录不变，无GUI或实机控制。

2026-10-20完成定向小力与多种子实验：旧测试只读诊断定位到预留姿态2–3 N幅值低估；新增960条训练/验证和256条独立测试，重新划分姿态区域并筛除旧训练的重叠目标。三方案各3个训练种子，在新测试2物理种子下完整报告。压力观测、新预留姿态中，稳定幅值约束候选相对旧数据重训对照的MSE3.025→2.849 N²，稳定2–5 N漏检36.8%→29.2%、2–3 N漏检49.3%→39.5%；但短脉冲事件漏检8.8%→10.1%，且一组配对小力退化。该新划分与上轮不同，绝对指标不可直接比较。流式/离线6次核验最大差2.87×10⁻⁶ N。详见[定向多种子报告](/home/sorry/python_ws/interactive/unifp_reproduction/artifacts/ranger_force_targeted_v1/REPORT.md)。仍为无窗口仿真，未接入控制或硬件。

2026-10-20完成[近期密集＋长期独立编码对照](/home/sorry/python_ws/interactive/unifp_reproduction/artifacts/ranger_force_dual_history_v1/REPORT.md)：新种子20270310/20270311采集1,440条配对条件，全部通过安全筛查（1,152训练、288验证）；比较约37.4k参数的100ms密集基线和保留100ms密集帧、另编码120–500ms历史的双分支网络，各3个模型种子、普通MSE和8130次更新。压力噪声预留姿态下，综合MSE由2.7455降至2.6250 N²、恢复MSE由0.8939降至0.7386 N²、稳定小力漏检41.61%降至39.17%，但短脉冲漏检18.38%升至19.23%；压力同域短脉冲漏检24.00%升至28.21%。候选未通过预设全部门槛，未采集或读取保留测试种子20261222/20261223。6个模型、180次验证记录和12次流式核验通过，最大离线/流式差2.74×10⁻⁶ N；旧连续历史检查点兼容。仍为headless仿真，未接入动作控制或实机。

1. 量测四个轮心、轮胎外径/宽度、底盘离地高度和安装架位置，校准当前轮组坐标。
2. 获取 AgileX CAD/BOM，或实测装配后的总质量、重心和惯量；将 `base_link` 集中质量换成合理的底盘、悬架、轮毂等质量分布。
3. 对比平地直行、侧移、旋转和制动的命令/反馈曲线，辨识轮驱动/转向速率、延迟、死区、限幅、轮地摩擦和悬架行为。
4. 仅在轮式响应和末端受力响应都达到预先设定误差门槛后，把模型用于外力估计策略训练。
