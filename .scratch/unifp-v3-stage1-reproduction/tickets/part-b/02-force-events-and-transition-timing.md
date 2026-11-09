# B-T02 · 真实力事件、子步施力与转移时序

状态：已完成；23项相关单元／回归测试通过，原生施力验收按计划在B-T07收束。

## 目标与范围

交付两路独立梯形事件→实际COM施力→旧转移结算→下一命令→终止快照／局部reset→下一观测的完整环境链。沿用B04全部结论与B09采样规则；不改网络、资产、执行器和PPO算法。

## 开始时必读

B04全部、B09全部、B11-U1／U2／U3与G1；B-T01完成记录。源码事实见主线`knowledge/research.md`。

## 代码位置与接口

相对于`UniFP_v3/`：`legged_gym/envs/v3/legged_robot_v3.py`的构造、`step`、`reset_idx`、`_publish_observations`、帧构造、`_sample_base_velocity_commands`与命令推进；配置在`v3_pos_force_config.py`。优先保留事件状态在同一环境，不建通用事件框架。

- 两路事件各存世界外力／yaw命令峰值、阶段时钟与起点，输出[N,3]、单位N；`pending`和`applied`分开，不借同一张量覆盖上一转移标签。提供B-T05评估子类所需的窄事件推进覆盖点。
- 每策略边界推进一次；事件间隔tick175—449，rise为1—3秒且≤(I−1秒)/2，hold1秒，两路各0.8启用，起点间隔不是事件结束后再等I。reset可以采参数，初始输出精确零。
- 各轴正式目标±60N；B-T07扫描前不能称已验证，最终F_max按已确认流程冻结。无训练迭代级幅值课程。
- 人工力固定世界方向，在8个2.5ms子步复用，目标`arm_seg6`刚体COM，`apply_rigid_body_force_tensors(..., None, GLOBAL_SPACE)`；不物理提交F_cmd、不加TCP偏心矩，底盘N力槽为零。
- 旧命令／力先结算reward与日志，再推进下一命令。H_t配刚经历的F_(t−1)当前yaw表达×0.01；done前clone真实294维后继，纯timeout一次bootstrap，物理失败优先。
- 局部reset仅清对应行事件、施力缓存和历史；新episode其他物理标签真实，不能整体置零。B采用0.3整行零底盘命令概率、5秒重采；A的p=0保持原RNG调用路径。push和父创建物性沿用。

## 测试与验收

新增`tests/test_v3_force_events.py`、`test_v3_force_transitions.py`，扩展真实V3奖励／状态测试；先红后绿。七类单测合同详见B11，测试侧仅替代物理和抽样边界。

- 手算梯形边界、yaw0／90°和roll／pitch姿态、8子步同一力、局部reset与零命令概率；计数／精确零精确比较，张量代数atol／rtol=1e-5。
- 构造已施加20N、待施加−10N、reset0N，验证标签、终止快照和奖励没有跨帧混用；同步回归现有timeout／GAE测试。
- 可用本机已授权CPU＋GPU做4环境确定性施力早期短测，证据按B11-G1存档；完整两端、单自由刚体及最终身份核验由B-T07汇总，不为此反复造脚本。
- 普通失败局部reset；真实没有施力、非有限值、方向错误或reset串行须修复。

## Blocked by

- B-T01完成。

## 完成记录

2026-11-07：已交付两路独立事件、pending/applied、8子步COM施力、当前yaw标签、终止快照／局部reset、B零命令混合及旧A身份兼容。新增事件3项、真实step链2项测试，结合reward／identity／runner／PPO／storage共23项通过。测试只替代Gym调用，不复制step算法；已审查命令力不提交、torque=None、后继标签和reset隔离。证据：`knowledge/part_b_implementation/bt02/verification.json`。本票未运行原生仿真，不宣称施力已获原生验收；B-T07统一验证。下一项B-T03直接实施。
