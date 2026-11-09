# 当前源码与性能事实

核查日期：2026-11-03。路径以开发根为基准。本文只记录已读源码与已有实测。

## 资产生成与实际规模

- `UniFP_v3/tools/build_desktop_asset.py:262` 的 `separate_collision_parts` 遍历每个 collision，按 CAD 面邻接连通组件生成独立 mesh；跳过零体积组件，输出 per-link 数量和 `source_groups`。
- 同文件 `:325` 的 `build_asset` 从桌面 ZIP 生成派生资产；`:413` 默认调用上述拆分，`:422` 输出来源哈希、轮系几何、14 DOF 和碰撞清单。
- `UniFP_v3/resources/robots/v3_a1_desktop/asset_manifest.json` 记录当前碰撞数：arm_base 242、base_link 143、6 个臂段 266、两指 46、4 个轮体 24、4 个转向 link 80、8 个相机／连接件 101、型材 25，合计 927。
- XML 核查为 28 link、927 mesh collision；官方 `UniFP/resources/robots/b2z1/b2z1.urdf` 为 47 link、32 collision，类型为 21 box、7 cylinder、4 sphere。
- 修复记录 `UniFP_v3/docs/experiments/v3/实验汇总.md` 保存桌面来源、当前独立归档和短测结果。记录说明整体包围凸包及全身关闭自碰撞曾涉及错误接触或违背确认边界。

## 配置、导入与过滤

- `UniFP_v3/legged_gym/envs/v3/v3_pos_force_config.py:25` 当前指向独立训练资产 `v3_a1_desktop_train_v1`，设置 `collapse_fixed_joints=False` 和 `replace_cylinder_with_capsule=False`；`:175` 为 5ms、substeps=1、接触容量 2**25。
- `UniFP_v3/legged_gym/envs/base/legged_robot.py:519` 读取配置路径；`:538` 加载资产；`:576` 创建 actor；`:581` 提交刚体属性时传入 `recomputeInertia=True`。
- `UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:81` 在父类创建 actor 后，从实际 URDF 的上两级定位 `asset_manifest.json`，逐 actor 应用过滤，再初始化创建期随机化。
- `UniFP_v3/legged_gym/utils/v3_asset_collision.py:28` 读取 manifest；`:33` 按 link 核对实际 shape span 数；`:36` 通过 base 的四个 `*_fixed_collision` 组索引过滤其与对应轮体的局部安装接触；`:42` 过滤固定闭合两指。其他形状保留原过滤位。
- 该函数使用实际 link 名与 span，未硬编码 927 或某个碰撞编号，但依赖四组固定安装件的语义分离。

## 物性与关节

- `legged_robot_v3.py:112` 在 actor 创建后采集 base_link 与 arm_seg6 的名义物性，随后创建期随机化；`:187` 提交随机化时使用 `recomputeInertia=False`，`:192` 回读检查惯量。
- 因此物性链路包含导入、父类属性提交、V3 名义值采集、随机化等阶段，不只是 URDF 数值。
- `UniFP_v3/tests/test_desktop_asset.py:64` 检查 14 DOF 与总质量 83.3179153728508 kg；`:79` 检查非轮视觉和上装物性；`:103` 检查轮组视觉零位双向顶点偏差；`:133` 检查轮质量、惯量与正定性；`:157` 检查凸碰撞几何可形成三维体。

## 身份与恢复

- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_run_identity.py:51` 定位实际资产根；`:54` 汇编运行源码；`:55` 对资产根下全部文件建立清单；`:81` 保存源码与资产哈希。
- `UniFP_v3/legged_gym/scripts/train_v3.py:68` 对 resume 身份不一致报错。碰撞资源和配置路径属于该身份比较范围。
- 正式训练远程使用独立 r03 源码目录，当前正式资产未被本任务改动。

## 现有耗时证据

证据目录：`.scratch/unifp-v3-stage1-reproduction/knowledge/a8000_seed1_20261103_performance/`。

- `official_comparison.json`：同服务器官方历史 8000 轮日志累计 25664.8 秒；末 100 轮采样平均 2.06807 秒、更新 0.62288 秒。历史运行并非与当前同时对照。
- 相邻进度快照截至 870 轮：最近 100 轮采样平均 42.63636 秒、更新 0.65050 秒。正式 metadata 为 GPU PhysX、GPU pipeline、RTX A6000。
- `profile_probe.json`：GPU1、512 环境、实际每 actor 927 shapes，12 个零动作完整策略步平均 0.22256 秒；48 个纯物理子步含 PD／提交／simulate／刷新／CUDA 结清共 2.31788 秒。两段状态序列不同，不能视为精确分项占比。
- 该短测无学习器、无 PPO 更新、无训练日志写盘。碰撞优化的新测量见下文；原历史数据保留其原始口径。

## 独立训练碰撞资产与回归依据

- `UniFP_v3/tools/build_training_collision_asset.py:88` 的 `component_groups` 在每个link局部坐标分组；`:170` 的生成入口只替换collision、拒绝覆盖输出；`:221` 保留轮拱下缘单组件和四个原轮胎碰撞，`:234` 对其他分组生成最多64顶点凸代理，`:263` 写来源与误差清单。
- 最终 `v3_a1_desktop_train_v1/asset_manifest.json` 为80 shapes：底盘32、型材9、arm_base1、六臂段10、两指4、四轮8、四转向8、相机与连接件8。保留原碰撞20项，新增代理60项，实际最大51顶点。URDF仍为28link／14DOF。
- `UniFP_v3/tests/test_training_collision_asset.py:62` 比较排除collision后的完整XML及视觉字节；`:77` 检查预算、支持面误差、来源完整覆盖和过滤索引；`:117` 检查原轮胎字节、轮组极值0.5mm与内缩轮毂；`:143` 检查型材空隙；`:158` 捕捉两个真实轮拱回归点。凸包支持面误差只约束漏出源凸包的方向，不等同于凹陷填充误差。
- `UniFP_v3/tools/probe_training_collision_runtime.py:36` 沿真实GPU创建链回读物性、运行步骤；`:122` 的physics-only分支单独计量PD／提交／simulate／DOF刷新；`:169` 起核对局部reset；`:193` 起按策略顺序（四转向、四驱动、六臂）施加动作。
- `UniFP_v3/tools/benchmark_collision_training.py:13` 包装原训练入口的环境创建、learn、iteration日志与save时间，不修改学习实现。iteration区间含上一轮尾部IO，learn墙钟含末轮保存。
- `knowledge/runtime/r02/collision_candidate_nominal32.json` 记录最终源码250步零动作1次终止、base峰值53.5326N，250步轮系动作与250步臂动作均0次终止；9个被核对张量的非目标环境reset差值全部为0。
- 上述32环境候选与 `knowledge/runtime/collision_baseline_nominal32_full.json` 的28个link实际mass／COM／inertia JSON完全相同；总质量83.3179115088416kg。原资产同条件零动作终止4次、base峰值98.3668N。
- 首批独立远程副本 `validation/collision_optimization_20261103/source/UniFP_v3` 来源较旧，缺3个文件、另5个文件与本机不同，包括缺少V3专用2**25容量配置。其100轮361.91秒是旧副本的测量，不能作为最终源码验收证据。`validation/collision_optimization_20261103_r02` 在执行前核对完整源码清单与本机48项单测来源完全一致。
- 最终源码 `knowledge/runtime/r02/timing4096.json`：100轮learn墙钟403.0105446秒、创建26.4169201秒、最后checkpoint写入0.840659秒、完整平均4.030105秒／轮、区间中位数4.039909秒。原始计时wrapper包含正常日志与最后保存；开发100轮只保存一次。
- 远程 `validation/collision_optimization_20261103_r02/gpu_during.csv` 和 `gpu_processes_during.csv` 记录GPU1总显存13564MiB、本次训练进程12000MiB；这是运行中快照，不是全程峰值。checkpoint完成记录的Torch allocator峰值1920056320 bytes，未覆盖PhysX全部分配。
- `knowledge/asset_release.json` 记录独立归档113个文件与成功资产逐字节一致。资产内部README／manifest保留生成时的候选状态，验证证据保存在资产目录之外；修改状态文件同样会改变严格identity。

## 本地 Isaac Gym 一手说明

- `/home/sorry/.cache/unifp/isaacgym_preview4/pkg/isaacgym/docs/_sources/programming/assets.rst.txt:50` 说明可由碰撞几何覆盖COM与惯量，`:55` 说明导入选项默认不覆盖；`:62` 说明PhysX使用凸碰撞，未启用凸分解时每个mesh以单凸包近似。这描述的是导入选项，不能代替本项目父类创建后属性提交的实际回读。
- `/home/sorry/.cache/unifp/isaacgym_preview4/pkg/isaacgym/docs/_sources/programming/assets.rst.txt:72` 的凸分解示例设置 `vhacd_params.max_num_vertices_per_ch = 64`。该处是配置示例，不足以独立证明所有碰撞路径的GPU硬限制；凸包顶点数是文档明确暴露的复杂度参数。

## 仓库工作流文件状态

开发根 AGENTS.md 中引用的 `docs/agents/issue-tracker.md`、`docs/agents/domain.md` 在本次检查路径下不存在；当前复现任务实际使用 `.scratch/<feature>/knowledge、design、tickets` 布局，根 `CONTEXT.md` 记录项目阶段。

## 最终身份与同GPU对照结果

下列`knowledge/`证据路径均位于`.scratch/unifp-v3-stage1-reproduction/knowledge/model_history/unifp-v3-collision-optimization/`，远程一手输出对应`validation/collision_optimization_20261103_r02/`。

- `knowledge/runtime/r02/verified_summary.json`记录五次真实运行的checkpoint计数、双Adam和身份核对全部通过；恢复物性及父检查点SHA一致，恢复首步无物理终止。
- 最终源码512环境独立进程100步对照：完整step原0.2330613秒、新0.0484330秒（4.812倍）；独立物理段原0.1889053秒、新0.0264837秒（7.133倍）。原／新零动作随机化终止事件302／122，不能当作训练后能力失败率。
- `knowledge/runtime/r02/old_resume_rejection.json`及console记录真实训练入口拒绝旧资产检查点，新输出目录和Gym仿真均未创建。
- `knowledge/runtime/r02/start_gate.json`为`ready=true`、无blockers，四阶段文件哈希与当前源码／资产／正式seed1配置匹配。正式学习配置SHA为`2bffdf4d7e1173b3c6f157090d561c683da78e1da7a5abab7df5a0cc03ebeefd`。
