# Go2-W UniFP 迁移：源码与资料事实

核查日期：2026-11-03（Asia/Shanghai）。本文只记录当前工作树和官方来源能直接确认的事实，不记录未确认方案。

## 1. Go2-W 官方模型

- 本地模型包位于 `UniFP_v3/resources/robots/go2w_official/robots/go2w_description/`，来源为宇树官方 `unitreerobotics/unitree_ros` 的固定提交 `3ae893d905d11795ddf1075825ee5010364a1109`；来源和 BSD 3-Clause 许可记录于 `UniFP_v3/resources/robots/go2w_official/README.md:1-13`。
- 模型入口是 `.../urdf/go2w_description.urdf`。URDF 定义了 `base`、四条腿、轮组、IMU 和雷达等链接；`FL_hip_joint`、`FL_thigh_joint`、`FL_calf_joint` 位于第79、105、134行，`FL_foot_joint` 连续轮关节位于第213行，其他三条腿的对应关节位于第241-701行。
- 通过 XML 解析确认：33 个 link、32 个 joint、16 个非固定 joint；活动关节为 12 个 revolute 和 4 个 continuous。21 个 mesh 引用均能在官方包内解析到。
- 轮视觉/碰撞网格使用 `left_wheel.dae` 和 `right_wheel.dae`，引用位置见 URDF 第200-209、362-371、525-534、687-696行。模型包含轮关节，但不提供机械臂、夹爪或工具链接。
- 关节限位和执行器字段是 URDF 声明值，例如 FL 髋、腿、踝和轮关节分别在第79-88、105-114、134-143、213-219行。产品资料没有证明这些字段等于实机持续扭矩、闭环带宽或安全限值。

## 2. UniFP 原任务与当前迁移参考

- 当前项目第一阶段边界是 Ranger Mini V3 + A1Arm 的仿真 UniFP 复现，见根目录 `README.md:1-5` 和 `CONTEXT.md:7-13`；轮式特有创新在该阶段验收后进行。
- 项目术语把 TCP 定义为统一末端位置、速度、目标和施力作用位置的参考点，见 `CONTEXT.md:57-59`。Go2-W 官方模型没有 TCP，因此原任务接口缺少必要对象。
- 项目术语还规定策略直接给轮组与机械臂执行器提供控制目标，不用底盘速度控制器替代轮组动作，见 `CONTEXT.md:61-63`。这条约束不能未经硬件接口核验就套到 Go2-W 的固件控制模式上。
- Ranger 历史适配动作由 4 个 steering、4 个 wheel drive 和 6 个 arm joint 组成，见 `legacy/v3_reference_20261030/ranger_unifp_vec_env.py:34-36`；其 critic 还包含 `wheel_contact` 等轮式特权字段，见同文件 `:38-44`。Go2-W 的 12 腿关节 + 4 轮关节不能复用这组动作顺序或字段含义。

## 3. Ranger 迁移流程中可复用的事实

- 历史票据将迁移拆成：证据冻结、硬件/机器人契约、资产/执行器/TCP 物理、无学习环境与 schema、官方风格学习器、无力技能、受力课程、真实接触、泛化统计、部署导出，见 `.scratch/unifp-v3-migration/tickets/overview.md:7-41`。
- 其中“机器人契约→资产与执行器→无学习环境→学习器→无力基线→受力课程→接触与泛化→部署”的依赖关系适合 Go2-W；但 Go2-W 的 TCP/机械臂缺失会阻断原力位任务，应在机器人契约阶段先分流。
- 历史迁移建议明确要求首版先保持官方单 Critic、单 GAE、PPO 和辅助估计结构，只重写机器人/任务 schema，见 `.scratch/unifp-v3-migration/official_recommendations.md:1-16`。这只能在新任务仍有可比观测、动作和目标时成立。
- 历史轮式研究候选包括轮地滑移/执行器饱和可辨识估计，以及转向约束下的底盘—机械臂耦合顺应，见 `.scratch/unifp-v3-migration/design/08-轮式创新与消融.md:9-48`。Go2-W 没有转向关节，不能照搬 Ranger 的转向约束假设；其轮腿接触和腿姿态约束需要另建问题定义。

## 4. 已知资料缺口

- 官方 SDK2 README 没有确认 Go2-W 专用控制 API、通信周期、控制模式或安全停机语义；开发者中心本轮访问也未能核对型号手册，详见 `.scratch/go2w-robot-research/knowledge/research.md:44-50`。
- Go2-W 产品页给出总电机数、速度、载荷和扭矩等产品规格，但没有逐关节名称、减速比、持续/峰值扭矩、传感器坐标系、标定和实机数据，详见 `.scratch/go2w-robot-research/knowledge/research.md:15-34,58-72`。
- 因此，在没有用户提供目标任务与执行器资料前，不能冻结 Go2-W 的 UniFP 动作维度、TCP、力标签、控制频率或 sim-to-real 验收阈值。
