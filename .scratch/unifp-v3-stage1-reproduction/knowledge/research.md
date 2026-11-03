# 第一阶段源码与实验依据

核查日期：2026-10-30至2026-10-31，各节以其标注时点为准。本文件仅汇编源码、现存数据及本轮文件整理事实，不包含迁移方案、执行计划或验收阈值。

## 1. 证据位置与版本身份

开发根 `P=/home/sorry/python_ws/interactive/unifp_reproduction`；官方根 `U=P/UniFP`；旧版根 `L=P/legacy/v3_reference_20261030`；研究根 `R=/home/sorry/python_ws/interactive/research`。下文路径以这四个前缀表示，行号均对应实际文本文件。

- 实际读取 `git -C U rev-parse HEAD` 为 `68847a070f88d731058c3d8476929bc3b205f5bd`，`status --porcelain` 为空；origin 为 `https://github.com/unified-force/UniFP.git`。这是本地官方身份，不是对历史远程目录版本的认定。
- 官方 README 第 15–17、40–44 行写 Isaac Gym Preview4、Python3.8、CUDA11.2+；第32–36行将 ROS2 sim2real、MuJoCo sim2sim 和模仿学习数据管线列为尚未发布。其 overview 中概括的部署能力不能代替实际公开接口。
- `P/introduction/00-training-flow-zh-CN.md:17–25,30–94` 介绍固定提交的官方链路；同目录 `01-actor-zh-CN.md`、`02-critic-zh-CN.md`、`03-ppo-zh-CN.md` 分析对应组件。它们是讲解资料，一手接口以 U 下源码为准。
- 归档清单 `L/manifest.json`、核验结果 `L/verification.json` 和官方指纹 `L/official_baseline.json` 为机器可读证据。192 个普通文件共619709491字节逐项哈希保持；36个绝对内部软链接重新定位后有效；23个旧顶层条目整体归档，官方目录93个普通文件及4份 introduction 文档字节保持。
- 归档整理时本地可读 `/proc` 进程的 cwd/fd 未发现目标旧目录占用；该次整理未连接远程服务器、未停止进程、未运行新训练。后续本轮只读远程核查见第8节。

## 2. 官方总体调用关系与时序

`U/legged_gym/scripts/train_b2z1posforce.py:13–19` 取任务配置、创建环境、创建 runner 并 learn。任务在 `U/legged_gym/envs/__init__.py:4–12` 注册。runner 在 `U/legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py:103–161` 收集 rollout、计算 return、更新网络并保存。

本轮补核：训练入口第8行直接导入isaacgym；`U/legged_gym/utils/task_registry.py:65–103` 的 `make_env(name, args=None, env_cfg=None)` 合并参数、设置种子并直接构造注册任务；第105–156行的 `make_alg_runner(env, name=None, args=None, train_cfg=None, log_root="default")` 将环境及配置传给OnPolicyRunner，并在resume开启时load。未通过旧Lab任务的AST提取层。本地 `/home/sorry/.cache/unifp/isaacgym_preview4/pkg/isaacgym` 目录存在，此次没有运行import或物理仿真，不能据目录存在认定环境兼容。

目录与包补核：`U/legged_gym/__init__.py:3–4` 从自身文件位置推导项目根，`U/setup.py:4–17` 声明安装包与依赖，其中包括 `numpy==1.23`、`mujoco==3.2.3`；这是文件内容，未验证这些声明可在目标Python版本共同安装。`U/legged_gym/envs/__init__.py:4–12` 导入并注册B2任务；`ENV:1614–1634` 将配置中的资产路径拆为目录和文件名后交给Gym加载，`CFG:242` 当前资产路径是相对路径 `resources/robots/b2z1/b2z1.urdf`。实际启动目录与模块解析路径会影响资源定位。本轮 `git worktree list` 只列出现有官方main工作树，开发根尚无 `UniFP_v3/`。

后续源码简称：

| 简称 | U 下实际路径 |
|---|---|
| ENV | `legged_gym/envs/b2/legged_robot_b2z1_pos_force.py` |
| CFG | `legged_gym/envs/b2/b2z1_pos_force_config.py` |
| BASE | `legged_gym/envs/base/legged_robot_config.py` |
| AC | `legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py` |
| PPO | `legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py` |
| STORAGE | `legged_gym/b2_gym_learn/ppo_cse_pf/rollout_storage.py` |
| RUNNER | `legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py` |

补核任务与模型装配：`U/legged_gym/utils/task_registry.py:44–63` 按任务名保存任务类及环境／训练配置；`RUNNER:66–79` 从环境读取观测、特权观测、预测标签、单帧观测和动作维度，再构建ActorCritic与缓冲区。`AC:21–25,43–46,83–105` 接受维度参数并构建Actor／Critic；当前任务注册文件只有B2任务，本轮未新增V3注册或运行其装配。

默认物理 dt=.005s、decimation=4，策略周期=.02s，见 BASE:164–166、CFG:227–235、ENV:1741–1743。默认4096环境、每轮24策略步，见 BASE:4–12,214。ENV:100–180 的顺序是动作裁剪、逐子步施力和积分、刷新状态、更新目标、判终止、奖励、重置、下一观测。返回为四项：观测字典、reward[N]、done[N]、infos；字典是 obs[N,2336]、privileged_obs[N,447]、obs_pred[N,12]。obs_pred 是环境标签而非网络预测，见 ENV:135–142,405–438。

## 3. 官方 Actor 与辅助估计

ENV:414–422 的单帧73维依次为 roll/pitch2、角速度3、关节位置17、速度17、动作17、步态相位2、命令15；CFG:107–113、ENV:424–438 将32帧展平到2336。ENV:297–303 清零 reset 对应历史。Actor输入没有真实基座线速度、真实末端位置或真实外力。

AC:46–93 的 encoder 为2336→512→256→128→64，actor 将最新73维与64维latent拼接后输出17维动作；decoder为64→128→64→12。AC:143–166,177–180 表明 decoder 物理预测不显式回灌 actor，二者共享 latent。AC:113–153 用对角高斯，动作头无tanh，所有动作logp求和；AC:161–166推理返回均值。

ENV:405–412 的12维标签为基座body frame线速度3、yaw对齐球心坐标下末端球坐标3、yaw frame末端注入力3、yaw frame基座注入力3。力标签是注入缓冲的值，不是全部接触反力。ENV:362–366,1440–1455 给出变换与缩放，力缩放 .01。AC:13–15和PPO:187–196将 .2/.2/1/1 同时乘预测与标签后求MSE，等效平方系数 .04/.04/1/1。

## 4. 官方 Critic 与 PPO

ENV:379–404的特权单帧149维含标签、腿步态误差、质量/COM增量、摩擦、电机系数、足接触、重力、本体状态、命令及受力偏移目标。3帧历史447维，见CFG:108–113、ENV:431–438。AC:95–105,173–175 输出一个标量V，不是分组价值头。

STORAGE:34–47,72–86 中 reward/value/return/advantage 均单通道，单GAE；PPO:77–88先缓存当前输入与标签，ENV.step返回后在PPO:90–100保存转移。timeout用旧V加到奖励再截断，未单独评价真正终止前下一状态。BASE:198–209配置gamma=.99、lambda=.95、clip=.2、5epoch×4minibatch、LR1e-3、KL目标.01、gradnorm1；PPO:12–15辅助LR1e-5。

PPO:53–55 两套Adam都注册模型参数；PPO:164–200先主更新再辅助更新。主策略梯度可达encoder，辅助梯度可达encoder/decoder；actor head不会仅因被注册就获得辅助图梯度。RUNNER:236–250只存模型、主optimizer、iter和infos，未保存辅助optimizer、力课程计数、RNG和物理状态。RUNNER:115–161中循环内iter元数据可能滞后于文件名。

## 5. 官方动作、目标、施力与奖励

CFG:58–64,104–106、ENV:877–885 表明17动作对应12腿＋5臂，完整19DOF中另外两维固定参考PD。动作裁剪±100，残差 .25×action×motor_strength 加默认角后PD并按URDF力矩限幅，见BASE:144–145、CFG:227–235。

ENV:949–978 的末端球心随base XY/yaw变化，不随真实base Z/rollpitch；CFG:18–21 偏置[.2,0,.8]m。CFG:6–30与ENV:887–978给目标范围和球坐标插值、离散碰撞拒绝；没有整段轨迹完整IK可达证明。

ENV:1891–1900 的世界坐标奖励目标是 `p_geom + (F_ext_world + R_yaw F_cmd_yaw)/K`，K=200N/m（CFG:148–150）；对三轴绝对位置误差之和取指数。它是补偿位置奖励，不是直接接触力误差奖励。ENV:130调用`apply_rigid_body_force_tensors`，没有显式TCP位置参数。

ENV:125–130的力课程是global_steps > force_start_step×24，CFG:178为8000；步数每次批量策略step加1。EE命令力和外力独立调度，每轴±60N，见CFG:140–156、ENV:1037–1169。base力调度调用被注释；底盘速度push是另一路，见ENV:1025–1035。质量、COM、摩擦、电机等域随机化见CFG:70–99、ENV:700–819；多项在创建时设置，并非reset全重采样。

## 6. 已归档 V3 的实际情况

`L/ranger_unifp_vec_env.py:27–46,76–108,227–266`记录旧14动作（4转向＋4轮速＋6臂）、Actor58×32、Critic98×3和12标签；转向/臂位置残差、轮目标速度分别用不同单位。它没有成为本轮新建实现。

`L/ranger_512_migration/logic/backend_envelope.py:36–81,97–104,222–226,272–320`用Gym桥和AST提取旧Lab任务方法；不是完整Lab训练。旧根资产和实际冻结512训练路径不同，不能仅由根README推断运行协议。URDF `L/robot_description/a1arm_description/urdf/a1arm_rgm3_original_movable.urdf:499–509,564–589`含base质量及四轮碰撞定义；此前XML求和83.317915kg，仅为文件标称值，不是称重证明。

`L/PROJECT_REPORT.md:7–15`记录512环境、24步/轮、单训练seed、16000轮；第21–34行报告终点确定性mean受力偏移目标误差50.348cm、估力RMSE27.702N/轴，sample误差45.795cm、估力11.271N/轴。mean是确定性动作均值，sample是加策略噪声，并非力分布命名。第38–40行说明旧原始轨迹和日志已清理；归档并未恢复缺失数据。

旧路径失效、COM施力与TCP偏置等更完整的归档前源码记录保存在 `P/.scratch/unifp-v3-migration/knowledge/v3_source_audit.md:79–95`。该历史记录中的原根路径现对应L；归档文件普通字节已通过manifest核验。历史记录不表示本轮重新执行了旧评估。

## 7. 已有 B2 复现与论文证据边界

本轮直接读取论文缓存 `R/literature/projects/unifiedforce/assets/paper.txt:2` 中的第4.1节：零力位置跟踪采用6000步随机末端轨迹，并评估末端位置估计；受力仿真以命令力匹配施加外力时的位置跟踪作为间接评估，对应Fig.3(c)；直接力控制评估使用实机测力计，对应Fig.3(d)。该文本未给出可直接等同于源码变量的力符号约定。

`R/literature/projects/unifiedforce/reproduction/outputs/stage2_20261029/official_comparison/analysis_report.md:5–15`报告8环境×6000步固定零力评估：8000轮平均L2约5.05cm，16000轮约6.63cm；第25–45行说明16k与官方图目测37–38k预算不同，Fig3b是逐轴分布而非时间曲线，缺作者原始MAE。

此前float64复核记录在 `P/.scratch/unifp-v3-migration/knowledge/reproduction_source_audit.md:7–40`及同目录`reproduction_validation.json`：8000轮L2=5.0476cm、16000轮6.6317cm；最终随机受力补偿误差11.2453cm、估力XYZ MAE6.0864/4.8267/5.4815N。此次仅复用本地已有核验记录，没有重新推理。

`.../official_comparison/paper_check.md:13–39,56–76`记录Fig3b逐轴误差分布、作者精确分箱/seed/MAE未公开；Fig3c匹配力、Fig3d真实接触力与随机独立施力协议不同。上述证据不能给出“全论文实验已完成”的事实，也不能替代新V3数据。

## 8. 本轮本地与远程环境只读快照

本轮执行的输出汇编在同目录 `runtime_inventory_20261030.json:1–122`。本地通过 `conda env list --json`、`nvidia-smi` 和Python标准库 `importlib.metadata` 读取安装信息；远程使用已有SSH别名 `gpu-server`，以BatchMode、主机密钥严格检查及单次连接执行只读命令，没有安装依赖、改动远程文件或启动训练。

- 本地GPU为RTX 5060 Laptop，驱动580.178.04，总显存8151MiB；tavis元数据为Torch2.7.0+cu128、NumPy1.26.4。Gym缓存py38环境元数据为Torch2.0.1、NumPy1.23.5，另有torch231目录记录Torch2.3.1+cu121。
- 本轮独立SSH已成功执行 `hostname` 并返回 `gpuf3`。远程4张RTX A6000，每张49140MiB，总显存与当前占用见快照；不能据此判断设备预留情况。
- 远程根分区98%、可用37G；`/mnt/data`为93%、可用272G；`/mnt/data_8THDD`为85%、可用1.1T。以上均为命令显示的即时值。
- 远程Conda环境注册文件 `/home/myl/.conda/environments.txt` 包含 `/mnt/data_8THDD/conda_envs/unifp`；实际包目录为其 `lib/python3.8/site-packages`，元数据为Torch2.3.1和NumPy1.23.0。`isaacgym.egg-link:1` 指向 `/mnt/data_8THDD/myl_unifp/isaacgym/python`，`UniFP.egg-link:1` 指向 `/mnt/data_8THDD/myl_unifp/UniFP`。本轮未读取该远程仓库提交或启动任务，不能将其认定为本地固定基线或已确认的V3开发目录。
- 远程系统Python为3.12.3，与上述Conda目录不是同一解释器。本次仅以标准库读取元数据，未导入Torch、Gym或验证GPU与物理仿真兼容性。

## 9. 官方训练与演示入口

- `U/legged_gym/scripts/train_b2z1posforce.py:13–24` 调用get_cfgs、make_env、make_alg_runner与learn，主入口将headless设为True。当前scripts目录仅有该训练脚本与 `play_b2z1posforce.py`，没有本轮V3训练／评估脚本。
- `U/legged_gym/scripts/play_b2z1posforce.py:20–46` 将环境数配置限制为1、关闭多项随机化与噪声、设置test=True，经注册器构造环境，再设置resume并获取推理策略。环境数仍会经过make_env的参数覆盖逻辑，不能仅凭第23行断言最终运行环境数恒为1。`:131–142` 设置play并循环推理和step，开启FIX_COMMAND时修改命令。`:189–194` 默认开启模型导出及预测可视化。
- `U/legged_gym/utils/helpers.py:73–95` 在load_run或checkpoint为-1时选择排序后的最后运行或模型；`:124–151` 定义任务、恢复、设备、环境数、种子和训练轮数等命令行参数。
- `RUNNER:244–250` 的load接口默认加载主优化器，可传load_optimizer=False跳过其加载；`:252–256` 将模型切换eval并返回act_inference。`AC:161–166` 的act_inference编码观测并返回actor输出的动作均值，而非Normal.sample。

## 10. Gym包声明与远程依赖版本补核

- 本地缓存 `/home/sorry/.cache/unifp/isaacgym_preview4/pkg/isaacgym/python/setup.py:31–51` 声明Gym包版本为1.0.preview4、Python范围为 `>=3.6,<3.9`，依赖包括Torch>=1.8.0、torchvision>=0.9.0、NumPy>=1.16.4、SciPy、PyYAML、Pillow、imageio和ninja。这是安装声明，未据其验证本机或远程运行。
- `U/setup.py:9–17` 声明isaacgym、matplotlib、numpy==1.23、tensorboard、mujoco==3.2.3、pyyaml、wandb、params_proto、pydelatin。本轮未运行安装解析器，不据声明认定这些包在目标解释器下可同时安装。
- 通过SSH以系统Python标准库读取已有远程环境的Conda元数据，Python记录版本为3.8.20、build为h4a871b0_2_cpython；输出补记于 `runtime_inventory_20261030.json` 的“依赖版本只读补核”。
- `/mnt/data_8THDD/conda_envs/unifp/lib/python3.8/site-packages/torch/version.py:4` 实际文本为 `__version__ = '2.3.1+cu121'`，第6行为 `cuda: Optional[str] = '12.1'`。包元数据的2.3.1与版本文件中的CUDA后缀共同描述该已有安装；没有执行CUDA内核或Gym仿真。

## 11. 日志装配与运行工具现状

- `U/legged_gym/utils/task_registry.py:139–148` 在默认日志根、None和显式log_root之间选择，随后将计算出的log_dir传给OnPolicyRunner；默认日志位于仓库根的logs。`RUNNER:92–99` 在learn内调用wandb.init并创建SummaryWriter。源码中这些初始化不证明当前安装的日志依赖已经通过离线运行验证。
- 本轮通过SSH和Python标准库核对，远程tmux、git、rsync分别可解析至 `/usr/bin/tmux`、`/usr/bin/git`、`/usr/bin/rsync`。`/mnt/data_8THDD` 和 `/mnt/data_8THDD/conda_envs` 均存在且os.access(W_OK)为True；`/mnt/data_8THDD/myl_unifp_v3_stage1` 与 `/mnt/data_8THDD/conda_envs/unifp_v3_stage1` 均不存在。输出汇编见 `runtime_inventory_20261030.json` 的“运行工具与目录只读补核”，没有创建目录或运行tmux任务。

## 12. 配置对象与种子实际装配顺序

- `U/legged_gym/envs/base/base_config.py:3–25` 以递归方式将成员配置类实例化；`utils/task_registry.py:50–63` 在register中保存配置实例，get_cfgs返回保存的实例引用，并将训练配置seed写到环境配置，没有在该函数中复制配置。
- `U/legged_gym/utils/helpers.py:97–120` 对env_cfg应用环境数和步态观测开关；seed只在cfg_train非None且args.seed非None时写到训练配置。`helpers.py:37–47` 在seed=-1时随机选取0到9999，然后设置Python、NumPy及Torch种子。
- `U/legged_gym/scripts/train_b2z1posforce.py:14–18` 先get_cfgs，再make_env，最后make_alg_runner；`utils/task_registry.py:90–93` 在make_env里仅以env_cfg更新环境参数后调用set_seed(env_cfg.seed)，`:136–148` 在后面的make_alg_runner中才应用训练命令行覆盖并创建模型。因此args.seed不同于注册配置时，环境初始化可以先使用旧seed，后续训练配置记录另一个值；这是当前调用顺序事实，本轮未运行种子实验或修改实现。
- `U/legged_gym/envs/b2/b2z1_pos_force_config.py:102–113` 定义B2动作数、历史长度和观测维度的类级表达式；`:313–321` 定义B2训练配置。类属性中的计算表达式在类定义时求值，不会因后续只改输入字段而自动重新计算所有关联字段。

## 13. 物理循环、策略动作保持与训练计数

- `BASE:164–166` 的Gym步长为0.005s、substeps=1，`CFG:235` decimation=4；`ENV:1741–1743` 派生策略dt=0.02s。`ENV:107–115` 在子循环外裁剪并保存actions，在四次物理循环中以相同actions和更新后的关节状态重新计算torques；`:130–133` 每次施力、simulate并刷新DOF状态后，在子循环结束统一post_physics_step。动作保持与固定力矩保持不是相同实现。
- `ENV:141` 的global_steps每次批量env.step结束加1，不乘num_envs；`:125` 的力课程门限为force_start_step×24且使用严格大于比较，常数24并非在此读取runner配置。`RUNNER:115–151` 每轮收集num_steps_per_env次批量step，再执行compute_returns和update；`BASE:214` 默认num_steps_per_env为24。这些是当前计数事实，未在本轮更改课程或预算。
- `ENV:1752–1765` 以策略dt对episode、push间隔和持续时间等执行ceil换算；`ENV:116–120` 在test模式下按sim_time和elapsed_time差值sleep，这是墙钟节流，不改变sim.dt。

## 14. rollout存储与样本日志

- `RUNNER:75–79` 读取num_steps_per_env并传入init_storage；`:115–151` 每轮执行T次env.step、compute_returns与一次PPO.update。`:163–164` 在log中增加T×num_envs到tot_timesteps；该计数在日志函数中更新，不是环境的global_steps。
- `STORAGE:34–47` 以[T,N,...]保存观测、标签、动作、奖励、终止、价值和分布参数；`:96–99` 令batch_size=N×T，mini_batch_size=batch_size//num_mini_batches，索引数量为num_mini_batches×mini_batch_size。若N×T不能整除minibatch数，当前生成器构造的索引数量少于总转移数；本轮未修改生成器。
- `BASE:5,214` 默认N=4096、T=24，因此每轮98304条转移；已读官方策略周期0.020s下，每轮每环境24步对应0.48s，含episode重置时并非一段无中断轨迹。以上为源码参数直接算术推导。

## 9. 开发目录归属整理

2026-10-30 按用户指定，将 `R/.scratch` 整体迁入 `P/.scratch`，包含当前第一阶段任务和历史迁移任务。迁移时50个普通文件逐项SHA-256一致，无软链接；原位置已不存在。随后修正开发入口、知识库及外部研究报告的相对链接，并同步已确认的正式训练预算。75处本地文件链接核验有效，官方仓库仍为固定提交且工作树干净。完整迁移前指纹及迁移后文档修订记录见 `P/.scratch/unifp-v3-stage1-reproduction/knowledge/scratch_move_verification_20261030.json`。历史 `unifp-v3-migration/deliverable_manifest.json` 保留原始字节，其中路径和哈希表示原研究根下的历史快照，不是当前文件校验清单。

## 16. 议题03资产来源与配置隔离补核

核查日期：2026-10-30；只读解析归档XML、来源记录和厂家固定提交，未修改资产、创建新V3实现或运行仿真。机器可读结果位于同目录 `asset_source_audit_20261030.json`。

- `L/robot_description/a1arm_description/vendor/agilex_ranger_mini_v3/provenance.json:2–4` 记录来源仓库 `https://github.com/agilexrobotics/ugv_gazebo_sim`、提交 `27633a956c845903ee630538afeb17fe70afdd84` 及包许可声明。已按记录逐个复算本地5个原始文件SHA-256，均与记录一致；这只证明与历史来源清单相符。厂家GitHub固定提交文件树可读，包含对应xacro、package.xml和三份网格。raw.githubusercontent.com的5次文件读取均超时；上游Git树对象核验结果以JSON中实际保存的比较项为准，不把超时记成文件不一致或已完成下载比对。
- 后续通过厂家固定提交的Git tree核验：xacro、package.xml、steering_wheel.dae和wheel_v3.dae的本地Git blob SHA-1均与上游一致。ranger_base.dae在Git树中是133字节LFS指针，不能将其blob SHA-1与展开后的网格直接比较；读取指针后，oid为SHA-256 `cf8d73c2d048b781f7dcd6d63122a79114836961ee347a9c5be716f5eef5501e`、size为38454037，均与本地网格一致。因此5项原始材料均完成对应来源核验；没有将LFS指针与实体内容的表示差异误判为资产被改写。该核验不包含A1来源、尺寸装配或动力学验证。
- 厂家缓存 `.../vendor/agilex_ranger_mini_v3/package.xml:11` 声明BSD；同目录 `ranger_mini.xacro:6–13` 的base_link质量为10，惯性字段亦为文件给定值，没有在文件中提供整车称重依据。旧生成器 `.../scripts/build_ranger_visual_usd.py:1–6,98–109` 明确继承v2臂／控制拓扑，再替换厂家几何及接触参数；因此来源记录与后续组合模型是不同层次的文件。
- 本轮XML直接求和：`.../urdf/a1arm_rgm3.urdf` 27个link、质量17.595035372850802kg，关节类型为fixed×15、prismatic×4、revolute×7；`original_movable`和`approx_sim_v3`两份URDF均为28个link、质量83.3179153728508kg，关节类型为continuous×4、fixed×11、prismatic×2、revolute×10。完整文件名、哈希及计数见JSON。关节类型计数包括虚拟关节或夹爪，不等于策略动作数。
- `L/robot_description/a1arm_description/README_approx_sim.md:25–36` 将质量集中、轮胎近似、无显式悬架、未经标定的执行器限制及适用性限制明确写出；其中v2参数为历史记录，不能整体套到v3。该说明不是实测报告。本轮未找到可据以确认A1机械臂独立厂家版本的来源链，也未将原有模型中的惯性和限值标成厂家核实值。
- 基于实际 `U/legged_gym/envs/base/base_config.py:3–25` 的最小内存验证：构造含类属性列表的成员配置，经BaseConfig递归实例化后成员对象的 `__dict__` 为空；对配置实例copy.deepcopy，原对象和副本仍读取同一类属性列表。验证未导入Gym或创建仿真，事实输出同步到workspace_verification.json。普通实例深拷贝与可变配置默认值隔离不是等价事实。

## 17. 议题03完整讨论所需的关节与驱动事实

本轮未运行Gym，以下为当前源码与XML行为。完整关节名称、parent／child、origin、axis、limit及dynamics保存在 `asset_actuator_inventory_20261030.json`，不是新配置。

- 厂家固定提交的 `L/robot_description/a1arm_description/vendor/agilex_ranger_mini_v3/ranger_mini.xacro` 中四个转向关节的axis均为(0,0,-1)，轮驱动axis为(0,1,0)。前三个转向关节origin为(0.25,-0.19,-0.1)、(0.25,0.19,-0.1)、(-0.25,0.19,-0.1)m，右后为(-0.23,-0.206,-0.1)m；相对位置并非完整对称矩形。不能仅据厂家文件已对应来源推断尺寸、符号及对称性已验证。
- 旧可动URDF含6个arm_joint、4个转向、4个轮驱动、2个夹爪可动关节。旧转向限位约±1.5708rad、effort=35Nm、velocity=2rad/s，轮驱动effort=80Nm、velocity=20rad/s；6臂关节effort为[40,40,27,7,7,7]Nm。这里只记录文件值，未验证其硬件依据。原有夹爪两指为0—0.03m的prismatic关节，不能将其effort单位记成Nm。
- `L/ranger_unifp_vec_env.py:34–38,57–70` 的旧动作顺序为fl／fr／bl／br转向、四轮驱动、六臂；ARM_HOME为(0,1,-1.5,0,0,0)rad，轮与转向初始目标为零。`:100–115` 的目标为默认角＋0.25×动作、轮速5×动作rad/s，默认implicit_pd路径给PhysX发送目标；另行返回的裁剪PD估计并非该implicit drive实际请求张量。
- 同文件`:109–111` 注释说明旧实现选择隐式驱动与低惯量关节的5ms显式反馈振荡有关；本轮没有复现实验，不能据注释判定所有显式力矩实现都不可用。旧kp／kd不构成新显式执行器已稳定的证据。
- `U/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:113–132,877–885` 则在每物理步显式计算PD力矩、裁剪后调用set_dof_actuation_force_tensor，再simulate与refresh_dof_state_tensor；该调用关系与旧implicit drive不同。
- `U/.../legged_robot_b2z1_pos_force.py:1617–1637` 将drive mode、固定关节合并、圆柱替换、fix_base_link、阻尼、armature与碰撞厚度等传给Gym；`:1682,1687` 用配置创建actor并设置刚体属性，recomputeInertia=True是当前调用参数，不能仅由配置URDF惯量推断最终运行惯量未被改写。
- 本地Gym一手文档 `/home/sorry/.cache/unifp/isaacgym_preview4/pkg/isaacgym/docs/programming/tensors.html:391–406` 区分COM施力API与给定位置施力API；后者自动产生偏置力矩。单位、坐标枚举和作用点是不同参数，不能把GLOBAL_SPACE当作“施力点已是TCP”的证据。

## 18. 现有V3默认资产定位与静态检查

核查日期：2026-10-31；结果见同目录 `current_v3_asset_check_20261031.json`，未修改资产、未运行Gym。

- `L/ranger_unifp_vec_env.py:27–29` 默认资产键为`v3_original`，对应`assets/v3_original/a1arm_rgm3_original_movable.usd`；`L/ranger_512_migration/logic/backend_envelope.py:80–81` 的Gym加载文件为`urdf/a1arm_rgm3_original_movable.urdf`。另一个键`v3`指向`assets/v3/a1arm_rgm3_approx_sim_v3.usd`，两者是不同变体。
- 直接解析`original_movable`及`approx_sim_v3`两份URDF，均为28个link、27个joint、16个可动关节、27个含inertial的link，唯一根link为`base_link`。可动关节含两指夹爪，14维策略动作与16个资产可动DOF不是相同计数。
- 本次检查涵盖基本parent／child引用、名称、正质量、惯性正定和网格路径可读性；两份文件在这些检查项中均未报告问题。`original_movable`引用47个不同网格路径，`approx_sim_v3`为37个，均能解析为现存文件。
- 默认URDF的SHA-256为`49b2e1b1acf728bdf1ec403126a219f7f62d7e5bbd0912c6b81af745fda498df`；变体URDF为`a5b019bdd78caa2bcb77fb3a41a1a303ddd8fdd944ce4a1ae34510392eb75a28`。本次未验证网格格式的Gym兼容性、全部惯量物理约束、几何穿透、TCP含义、四轮接地或动态稳定性。

## 19. 现有V3参数表的字段对应

2026-10-31从当前归档URDF及控制文件导出[参数事实表](current_v3_parameter_table_20261031.md)，未导入旧运行模块或执行仿真。

- `L/.../urdf/a1arm_rgm3_original_movable.urdf:556–726`给出四转向安装点的x=±0.247m、y=±0.185m、z=-0.1m，轮轴再偏移z=-0.12m，碰撞圆柱半径0.1m、轴向宽度0.08m。由此直接算得轴距0.494m、轮距0.370m，零转向水平接地时base原点理论高度0.32m；这不是动态接地测量。
- 同文件`:493–508`给出安装架相对base的z=-0.33m、base质量65.845600kg、局部COM=(0,0,-0.095)m及0.70×0.46×0.24m碰撞盒；`:264–267`的臂座相对安装架平移为(0.174,0,0.42535)m，两个安装关节rpy均为零，组合臂座相对base平移为(0.174,0,0.09535)m。
- `L/ranger_unifp_vec_env.py:33–37,57–70,100–115`分别给出旧初始高度／默认臂角、隐式驱动参数及动作尺度。参数事实表保留原数值和来源，区分URDF夹爪effort=200N与旧驱动effort=10N；没有将旧隐式参数验证为新显式执行器配置。

## 20. 进入04前的几何计算与环境接口核查

核查日期：2026-10-31；只读解析当前URDF／碰撞GLB和官方源码，未运行Gym、未修改资产或业务代码。

- 本地NumPy、SciPy、trimesh解析了当前URDF、两指碰撞GLB及其场景变换。两指位移为0m时，三角顶点变换到arm_seg6坐标的最大z均约0.079612m；两指顶点最近距离约8.548微米。顶点距离不是三角面穿透检查；约8cm距离也不是从官方Z1的grasp_offset字段移植的实测值。
- `gripper1_collision.glb`的三角形152重心在arm_seg6坐标为(-0.000143687734,0.000380680222,0.079612056714)m，端面法向近似+z。该计算保留当前网格哈希，不代表PhysX凸分解之后的表面仍逐三角形对应。几何选择、模型质量矩阵和指定带宽输入的代数计算见同目录`v3_parameter_derivation_20261031.json`；这些计算不是仿真稳定性或跟踪结果。
- 在臂角[0,1,-1.5,0,0,0]rad、轮组零角和两指固定0m的计算假设下，按逐体质量／惯性与运动链雅可比构成自由基座6维＋14旋转关节的20×20质量矩阵，最小特征值约7.97535×10⁻⁵；从逆矩阵对角元素得到名义等效惯量。代数计算不含轮地约束，不验证全姿态耦合或5ms的PhysX稳定性；所输入的带宽只属于设计计算条件。
- `U/legged_gym/envs/base/base_task.py:104–109`调用`reset_idx`后推进一次零动作step，将step前两项命名为`obs, privileged_obs`再返回。实际`U/.../legged_robot_b2z1_pos_force.py:142`的前两项为观测字典与reward向量；故reset第二项的变量名与其实际内容不一致。
- `U/legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py:88`仅解包丢弃reset两项；`:100–105,125–135`在训练开始及每次step后从观测字典取`obs`／`privileged_obs`／`obs_pred`。`U/.../env/vec_env.py:22–35`抽象step标注的五项返回与实际B2的四项返回也不一致；本轮未修改公共抽象类。
- `U/.../ppo_cse_pf/ppo.py:77–88`在env.step前保存三类输入，部分字段直接保存张量引用；`:90–104`在step后缓存转移，超时时以该转移旧V补偿奖励。`U/.../legged_robot_b2z1_pos_force.py:174–187,207–303`先求奖励／终止、再reset、再构建下一观测；构建历史时新分配obs张量，`get_observations:440–441`只返回缓存字典。


## 21. 训练迭代、官方受力门限与检查点实存字段

2026-10-31只读核查当前官方工作树；本节记录源码行为与算术，不表示新V3功能已经实现。

- `U/legged_gym/envs/b2/b2z1_pos_force_config.py:178`设`force_start_step=8000`。`U/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:125–132`在物理子步以`global_steps > force_start_step * 24`判断是否调用`_push_gripper`，随后应用刚体力并simulate；`:141`在一次批量策略step结束时才将global_steps加1。比较是严格大于，常数24直接写在环境里；该变量不是按N累计的转移数，也不是直接记录已完成PPO更新的迭代数。
- `U/legged_gym/envs/base/legged_robot_config.py:214`定义每轮每环境24策略步；`U/legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py:115–151`循环收集这些step后计算return并更新。N=4096时，8000轮为786432000条转移，32000轮为3145728000条，总40000轮为3932160000条。按20ms策略周期，每环境对应3840s、15360s、19200s仿真进展；这些数值不是墙钟耗时。
- `U/legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py:53–55`创建主optimizer和adaptation_module_optimizer；`:167–170,198–200`分别执行主更新与辅助更新。`U/.../on_policy_runner.py:236–250`的save／load仅显式处理模型、主optimizer、iter和infos；没有显式保存辅助optimizer、RNG、环境课程／global_steps或PhysX状态。infos为调用方可选参数，不能假定它自动包含这些状态。
- `U/.../on_policy_runner.py:156–161`在循环内以局部it命名并保存模型，`current_learning_iteration`只在整个learn循环结束后增加num_learning_iterations；save写入的iter来自该成员而非局部it。因此循环内检查点文件名对应的迭代位置与所存iter不一定一致；循环结束后的最终save发生在成员更新之后。


## 22. A04完整议题的官方接口与观测事实

2026-10-31只读核查；下列为当前官方源码与静态几何事实，不是V3设计结论。

- `ENV:27–38,1440–1455`定义15维命令布局：底盘速度3、TCP球坐标3、末端姿态3、末端力命令3、底盘力命令3。`ENV:949–962`将当前插值TCP球坐标写入[3:6]并另算姿态四元数，未将该四元数写成commands[6:9]；`:125–129`未调用已被注释的底盘力分支。`BASE:129–155`给出固定输入缩放、clip=100及默认噪声参数。
- `ENV:414–422`Actor单帧由roll／pitch2、base角速度3、关节位置17、关节速度17、动作17、步态sin／cos2和命令15构成；真实base线速度、TCP位置和人工外力不在普通观测里。`ENV:405–412`的12维监督标签依次是base线速度B、TCP球坐标、Y轴末端注入力、Y轴base注入力，每组3维；标签不是任意物体接触反力。`ENV:379–404`另行形成训练Critic特权输入。
- `CFG:107–113`历史长度为Actor32／Critic3。`ENV:424–438`用`randn_like`加入高斯噪声，再按旧到新堆叠展平；`:1464–1471`初始化历史为零，`:300–303`清零所选reset行。`ENV:1328–1336`构造noise_vec时已乘noise_level，`:427`在噪声应用时又乘一次；默认level=1时不改变该倍数，但非1时系数平方生效。函数注释中的uniform描述与实际randn不一致。
- `ENV:824–830`在post_physics回调中根据episode步数重采底盘命令，并可调用push；该回调发生在`:169`，早于`:176–177`终止／奖励计算。`:172–173,949–973`同样在奖励计算前更新当前TCP目标和目标计时。`CFG:125`底盘命令重采间隔5s，`CFG:9–10`目标移动／保持时长分别为[1,3]s／[0.5,2]s；`ENV:1388–1389`在buffer初始化中采样时长，`:939–940`重采目标时清计时但未在此重采时长。
- `ENV:975–978`的球心使用root世界XY、零基准Z和yaw旋转后的固定offset；`CFG:18–21`球心offset为[0.2,0,0.8]m。没有在球心函数中加入实际base高度／roll／pitch或查询地形高度。球坐标转换为`U/legged_gym/utils/isaacgym_utils.py:59–74`的r／仰角／方位角约定。
- `ENV:932–947`对候选球坐标插值点做包围盒与地下筛查，最多循环10次；源码在循环次数耗尽后仍继续使用现存目标，没有在这里明确抛出无有效目标异常。该检查不是完整关节IK可达性证明。
- `ENV:199–205`终止条件含配置body的净接触力范数>1N、abs(pitch)>1.0rad或abs(roll)>0.8rad，并以episode_length_buf>max_episode_length判断到时；`CFG:248`默认terminate_after_contacts_on为空。`BASE:11`episode时长20s；不能只看到1N代码就认定官方默认确实启用了base接触终止。
- 以当前URDF、名义臂角[0,1,-1.5,0,0,0]rad和已有TCP表面候选复合静态FK，TCP相对base为[0.415604810,-0.000139661,0.857195411]m。扣除现有名义臂座平移[0.174,0,0.09535]m后，球坐标为[0.799237970m,1.263697986rad,-0.000578057rad]。完整输入、原资产哈希、公式与局限见[a04静态运动学记录](a04_static_kinematics_20261031.json)。未运行Gym，未修改URDF，不以此认定名义姿态无碰撞或任务范围已验证。

## 23. A04论文与公开代码的对应事实

核查日期：2026-10-31。论文为《Learning a Unified Policy for Position and Force Control in Legged Loco-Manipulation》，arXiv:2505.20829v2，2025-10-04。材料为[本地文本](../../../../research/literature/projects/unifiedforce/assets/paper.txt)、[PDF](../../../../research/literature/projects/unifiedforce/assets/paper.pdf)和同目录`paper.html`。文本是网页提取结果，含导航与重复公式，不是作者原始实验数据；UTF-8文件SHA-256为`e63007a2903274fb9e0a67d3837e303d4c19439a848718860a7fec93b2a412d5`，共59291个Unicode字符。以下区间均为对该文本执行`read_text()`后的0基、左闭右开字符索引；源码简称沿用第22节，官方提交为`68847a070f88d731058c3d8476929bc3b205f5bd`。

| 论文定位 | 原文直接表述 | 当前官方代码对应事实 |
|---|---|---|
| §3.2，字符[16357,19050) | 式(5)列出base姿态、角速度、关节位置／速度、上一动作、命令与feet clock；四足命令分为base速度、末端位置、末端力、base力四类；动作为默认关节姿态残差，通过位置PD执行 | `ENV:414–422`给出实际单帧切片，`CFG:102–113`设17动作、73维单帧；`ENV:27–38,1440–1455`定义15维命令，额外包含3维姿态槽。73维是该平台实现宽度，原文该段没有逐项规定73维切片 |
| §3.2，同一区间 | 历史H=32，经encoder生成latent，分别送入state estimator和actor；估计外力F=F_ext+F_react、末端位置和base速度 | `CFG:107–108`配置Actor32／Critic3帧；`ENV:405–412`监督四组三维：base线速度、TCP球坐标、末端力、base力；`AC:143–180`提供历史编码、策略与估计接口。原文的数学历史下标与代码deque实际长度分别可查 |
| §3.2，字符[19050,21162) | 反力不显式建模，而是纳入采样的净环境力；位置采样略超仅机械臂工作空间但在全身运动下可操作；力线性升高、保持、回零；先reaching／locomotion再随机力，估计器采用MSE监督 | `ENV:1037–1169`人工力函数与`:405–412`的forces缓存对应；`:125–129`在门限后调用末端力，base施力调用被注释。缓存表示注入Gym的训练力，不是从所有真实物体接触中直接测得的反力合计 |
| 附录C／C.1，字符[48200,50410) | PPO、MLP状态估计器、Isaac Gym与4096并行环境；末端半径[0.35,0.85]m；vx±0.8、vy±0.6m/s、wz±0.8rad/s；末端／底盘力命令和两处环境净力范围均为各轴±60N | `CFG:28`的半径上限为0.95m；`:128–130`为vx±0.6、vy±0.4m/s、wz±0.6rad/s。论文表述与公开代码默认范围并非完全相同；源码base力函数存在也不表示默认调用已开启 |
| 附录C.3，字符[52480,54000)中的C.3段落 | 为实现全身协调，在world-aligned base frame中估计末端位置，参考相对base投影保持固定高度和方向 | `ENV:352,975–978`使用base世界XY、yaw和固定offset构造参考；该球心函数未随实际roll／pitch和base高度起伏，也未查询地形高度。C.1的body frame概称需要结合该解释与实际公式读取 |
| 附录C.4，字符[52480,54000) | Fig.A.7及文字说明两阶段课程；进入受力阶段奖励先下降，随后locomotion恢复、reaching稳定，并解释任务多样性带来的差异 | `CFG:178`设force_start_step=8000；`ENV:125`实际判据为global_steps严格大于8000×24。该论文文本未出现`8000`或`8,000`，不能由文字给出8000＋32000的精确迭代预算 |

`ENV:138`同时裁剪监督标签；`:1333–1336,427`在构建噪声和应用噪声时分别乘noise_level。`TASK:104–109`公开reset推进零动作step；`ENV:169,173,176–179`显示命令／目标更新早于奖励，随后才局部reset。`ENV:202–204`包含接触、倾覆与严格大于的timeout比较，`CFG:248`默认终止接触body列表为空。这些都是固定公开实现行为；论文上述段落未逐项指定这些接口与边界细节。

## 24. A05辅助预测的训练存储与推理接收者

本节为2026-10-31只读核查的实际调用事实。路径均相对固定官方仓库`U/`；未运行训练或Gym。

- `legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:43–47,143–150`以历史展平宽度除单帧宽度再乘2定义latent维数，并从历史算latent，将最新一帧与latent拼接后交Actor。`:161–166`的学生推理从同一latent另外求decoder预测，动作均值作为返回值，预测经detach／CPU／NumPy转出，写入`policy_info['latents']`。`:177–180`名为`get_student_latent`，实际返回decoder预测。该预测与encoder中间latent不是相同张量。
- 同文件`:13–15`的监督顺序为base_velocity、gripper_pos、force_ee、force_base，长度均3，乘子为[0.2,0.2,1,1]。`legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:352–357,362–366,405–412`对应yaw对齐TCP球坐标、两处yaw轴施力缓存以及完整base轴线速度；没有在标签构造中累加物体接触力张量。
- `legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py:103–105,121–130`动作前传入观测、特权观测和监督标签，step后再取得新包；`:123`另求的预测局部变量`latent`在该收集循环未用于修改动作、命令或奖励。`:252–256`返回`act_inference`作为推理函数。
- 同目录`ppo.py:77–88`只把Actor观测传给`actor_critic.act`，将Critic观测传给evaluate，并暂存动作前标签；`:90–100`在step后写转移。`rollout_storage.py:54–67`用copy_分别存观测、标签、动作等；`:96–131`在展平后用同一batch_idx选各字段。因此辅助监督与动作前Actor历史配对，不是和随后的新历史配对。
- `ppo.py:119–128,164–200`每个minibatch先主损失更新，再由`get_student_latent(obs_batch)`重新算预测，以无梯度标签做分组MSE。每组预测和标签均先乘weight，实际相当于weight²乘该组MSE；四组相加，没有在此除以组数。`:53–55`两套Adam均注册整模型参数，实际图上的梯度可达性另由前向依赖决定。辅助图经过encoder和decoder，Actor头、Critic头和std不在该前向计算图中。
- `ppo.py:14–15,181–200,207–209`声明辅助子步为1，`adaptation_batch_size=64`用于日志分母；实际辅助预测仍用当前obs_batch，未在该段按64再次拆批。不能把该常数直接说成实际辅助监督minibatch大小。

- `legged_gym/scripts/play_b2z1posforce.py:132–146,155–159`把同一policy_info字典传给推理，在VISUAL_PRED分支读[6:9]末端力和[9:12]底盘力，并与施力缓存显示比较。`:167–179`的部分速度／位置显示代码被注释。脚本中的预测先于该次env.step取得，比较缓存在step后读取；代码存在该顺序，不能据此认为曲线已经按同一物理时刻对齐。
- 论文文本§3.2字符[18500,21162)说明latent供Actor与state estimator使用、以MSE监督状态与力，并明确采样净力包含反力且不显式建模反力。§3.3字符[21250,21900)说明估计力与状态／图像进入force-aware模仿学习、再预测位置和力命令。以上是一手论文用途，公开播放脚本不是这一整套上层流程的实现证据。

## 25. A05网络结构与初始化的有效配置链

2026-10-31只读核查固定官方提交；本节不代表V3方案已确认或运行验证通过。

- `U/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:9–11,21–29`定义encoder隐藏层[512,256,128]、decoder隐藏层[128,64]，构造函数的Actor／Critic隐藏层默认均为[256,256,256]、activation默认elu、init_noise_std默认1.0。`U/legged_gym/envs/b2/b2z1_pos_force_config.py:313–318`的具体训练任务policy配置则显式指定Actor／Critic隐藏层均为[512,256,128]。
- `U/legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py:53–55,66–72`把train_cfg['policy']作为关键字实参传给ActorCritic，所以该任务Actor实际使用[512,256,128]，而不是未覆盖的构造默认值。该具体policy类没有另写activation字段，因此使用构造默认elu。
- `actor_critic.py:46,51–93,182–184`将latent宽度定义为历史长度乘2，按默认32帧为64；encoder为2336→512→256→128→64，Actor为137→512→256→128→17，decoder为64→128→64→12。ELU放在隐藏层后，三条支路的末层Linear后未增加激活；没有在这些网络中创建循环层、dropout或归一化层。
- `actor_critic.py:53–105`直接构造nn.Linear；`:118–122`定义正交初始化辅助函数，但该文件的构造和同目录Runner中未找到调用。因而当前构造路径沿用nn.Linear本身默认初始化，不能因函数定义存在就把正交初始化记成实际行为。`:113`的动作std是独立可训练参数，其初始化和约束属于另一参数路径。

## 26. A05动作分布、裁剪与可选导出的源码事实

2026-10-31只读核查；路径均相对固定官方仓库`U/`，不表示V3方案已批准。

- `legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:29,113,143–153`用初始1.0构造num_actions个独立可训练std参数，按环境广播为Normal的scale；没有在该路径使用log_std／softplus／clamp。均值由Actor线性末层产生，无tanh；act调用distribution.sample，log_prob按最后动作维求和。`:140–141`的熵也按动作维求和。`:116`设置Normal类的默认参数校验为关闭，这与显式验证scale有限且为正不是同一行为。
- 同目录`ppo.py:77–87`存actor采样返回的动作和对应logp；`legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:107–108`在环境step内才裁剪并保存执行动作。`legged_gym/envs/base/legged_robot_config.py:145`默认clip_actions=100。`ppo.py:124–135,148–152`在当前分布上计算所存动作logp、分布KL和重要性比率，没有在这里计算tanh变换的Jacobian或将裁剪后动作替代存储动作。
- `actor_critic.py:158–166`推理返回动作均值，辅助预测经detach／CPU／NumPy写入传入字典；默认参数为可变字典。`on_policy_runner.py:252–256`仅切换eval并返回该方法，单独调用eval不等于进入无梯度模式；当前网络本身无dropout。`play_b2z1posforce.py:27–33`另外明确关闭环境噪声和若干随机化，故环境是否随机与是否使用动作均值是不同开关。
- `play_b2z1posforce.py:49–71`在EXPORT_POLICY分支分别深拷贝encoder、decoder和Actor body到CPU，以torch.jit.script保存三个独立文件。该段未把输入历史构造、最新帧拼接、观测缩放或动作物理映射封装成一个完整执行程序；三个文件的存在不等于部署调用链已经完备。

## 27. A05主更新与辅助更新的参数、顺序和日志事实

2026-10-31只读核查；路径均相对固定官方仓库`U/`，未执行学习更新。

- `legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py:12–15`定义辅助Adam学习率1e-5、辅助子步1；`:53–55`两套Adam均以actor_critic.parameters()注册所有模型参数。主学习率来自构造参数learning_rate，`:132–144`的KL自适应只写主optimizer.param_groups，没有在该段同步辅助Adam学习率。
- `ppo.py:164–170`主损失包含策略surrogate、价值项和熵项，调用主optimizer.zero_grad、backward、全模型clip_grad_norm_、step。`:181–200`随后对同一obs_batch重新前向估计器，对无梯度标签计算四组MSE，再调用辅助optimizer.zero_grad、backward、step；这里没有辅助梯度裁剪。两次zero_grad均未显式传set_to_none，本段代码不能单独确定不同PyTorch版本下的grad清理表示。
- `actor_critic.py:143–146,173–180`表明策略均值依赖encoder与Actor头，value依赖独立Critic头，估计预测依赖encoder和decoder；std为直接参数且不依赖均值网络，因此当前熵项只依赖std。辅助更新encoder参数可以改变同一输入的后续动作均值，即使Actor头参数没有由辅助图直接求梯度。
- `ppo.py:192–196,202,207–209`的每组MSE已经默认mean归约，累计日志在循环结束后又除以num_updates×辅助子步×adaptation_batch_size。实际预测使用完整obs_batch，没有在此按64拆批；因此该额外常数会缩小报告的辅助MSE，不改变之前已经backward的辅助损失标量。
- `on_policy_runner.py:236–250`保存／加载模型和主optimizer，没有同段辅助optimizer字段；将两套Adam动量都视为已完整恢复不符合此处实际内容。Adam更新时None梯度和零梯度的具体处理须结合实际运行PyTorch版本核验；本轮未据未执行的数值实验认定某个官方无关参数已经被误更新。


## 28. A06力字段与物理接触报告的官方来源

2026-10-31只读核查，路径相对固定官方仓库`U/`；下列为源码事实，不是V3方案或验证结论。

- `legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:125–130`在计数门限后调用末端外力更新，base施力调用被注释，随后将forces送入世界坐标刚体施力接口；`:362–366,380–385,405–412`将施力缓存变换到yaw轴，分别进入Critic和辅助标签。
- 同文件`:345,391`以足部net contact force的世界z分量大于5N形成四足接触字段；`:150,1349–1354`获取并刷新net contact force张量。`legged_gym/envs/base/legged_robot_config.py:181`设置contact_collection=2，代码注释为all sub-steps；这不是对四个控制decimation物理步显式求平均的实现。
- 同文件`:370–377,1891–1900`分别为Critic补偿目标与奖励补偿目标，将世界人工力和yaw命令力旋转后的值相加，再除刚度并加几何目标。代码中的足接触字段与上述人工力缓存是不同来源，未在此把足部接触净力加进辅助力标签。


## 29. A06质量／COM字段与应用缓存的官方事实

2026-10-31只读核查；路径相对固定官方仓库`U/`，未加载Gym或变动物性。

- `legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:781–817`构造mass_params顺序为base增重1、COM增量3、gripper增重1、逐链接增重17，共22。`:785`修改props[0].mass，`:812`修改props[1].com；两者在本函数使用不同索引。`:792–798`逐链接分支作用于前17个body，关闭时返回17个零。`legged_gym/envs/b2/b2z1_pos_force_config.py:74–81,89–90`开启base质量／COM和gripper质量随机化，关闭逐链接质量随机化。
- 同一ENV文件`:1669`分配[N,22]缓存，`:1685–1690`取得body属性、生成并设置修改后的属性，recomputeInertia=True，再将mass_params转换张量保存。该段缓存保存采样得到的变化量，没有在该段设置后重新回读形成质量／COM差值。
- 当前V3 TCP绑定body为arm_seg6，来自已确认03的几何与接口记录及本地静态资产推导；具体TCP候选保存在`P/.scratch/unifp-v3-stage1-reproduction/knowledge/v3_parameter_derivation_20261031.json:207–224`。这不是本轮Gym实际body映射或物性回读结果。


## 30. A06摩擦与motor_strength字段的官方事实

2026-10-31只读核查；路径相对固定官方仓库`U/`，未调用Gym或执行控制。

- `legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:713–727`在创建阶段采样256个摩擦桶并按环境分配，将同一环境的所有robot shape摩擦设为同一值；关闭随机化时该函数构造全1的friction_coeffs缓存，但该分支未逐shape重新赋值。`:1711`将缓存转换为friction_coeffs_tensor，`:388`直接输入Critic，没有减1。
- 同文件`:1713–1719`将12个leg和5个arm随机执行系数拼成motor_strength，关闭时取全1；`:389`向Critic输入前17项减1。`:877–879`以actions × motor_strength × action_scale构造位置残差，再加默认角形成PD位置目标；最终输出力矩裁剪在`:885`，motor_strength没有在该段乘最终力矩或kp／kd。
- `legged_gym/envs/b2/b2z1_pos_force_config.py:71–84`默认启用摩擦和电机随机化，范围分别[0.3,2.0]与[0.85,1.15]。这些是固定官方配置数值，不是V3已经确认的随机范围。


## 31. A06四轮刚体的资产映射事实

2026-10-31只读解析XML，未加载Gym。`L/robot_description/a1arm_description/urdf/a1arm_rgm3_approx_sim_v3.urdf:532–543,580–591,628–639,676–687`分别定义bl／br／fl／fr转向与驱动关节，各continuous驱动关节的child为bl_Link／br_Link／fl_Link／fr_Link，每个轮body有一个碰撞体。四转向安装点为x=±0.247m、y=±0.185m，fl／fr的x为正，bl／br的x为负；l为y正、r为y负。XML枚举轮驱动的顺序为bl、br、fl、fr，不等于此前参数推导记录采用的fl、fr、bl、br顺序。XML质量求和83.3179153728508kg。这些事实不证明Gym加载后的body枚举顺序或接触有效性。


## 32. A07联合PPO、GAE与计数的官方事实

2026-10-31只读核查，路径相对固定官方仓库`U/`，未运行优化更新。

- `legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:134–153`熵和动作log_prob均沿最后动作维求和。`ppo.py:119–170`对当前分布计算整条动作logp，联合概率比为新旧logp差取exp，再计算单surrogate和单value loss，加权合并熵项。
- `rollout_storage.py:72–89`按单reward／单value倒序计算GAE和returns，done遮断后继项；随后用整个rollout的优势mean／std标准化。`:96–99`用整数除法得到minibatch大小，并按num_mini_batches乘该大小生成索引。此处没有单独底盘／机械臂优势或价值头。
- `on_policy_runner.py:115–161`每轮先采集num_steps_per_env步再计算returns并更新；循环中按索引it保存，current_learning_iteration只在循环结束后增加请求轮数。`:236–250`保存／加载模型、主优化器等有限字段，未在同段保存辅助Adam、环境课程或随机源的完整状态。代码位置表示当前事实，不是V3已确认的恢复接口。


## 33. A07缓冲区所有权与采样的官方事实

2026-10-31只读核查；路径相对固定官方仓库`U/`，未运行采样或训练。

- `legged_gym/b2_gym_learn/ppo_cse_pf/rollout_storage.py:8–22,24–52`定义Transition暂存对象及[T,N,…]预分配存储；dones在`:39`转为byte，其余分配使用torch.zeros默认浮点dtype。`:54–67`将暂存字段逐项copy_后递增step，`:55–56`检查溢出；`:69–70`的clear只重置step，未清零存储。
- 同文件`:72–86`按完整num_transitions_per_env范围倒序计算回报及全rollout优势标准化，函数内未检查已采集step是否达到该长度。`:96–131`生成minibatch同样未检查step或returns是否已准备。
- 同文件`:97–99`计算B=T×N、M=B//J，再调用randperm(J×M)，没有整除检查。`:99`位于`:113`的epoch循环之前，因此同一次生成器调用的所有epoch复用同一随机排列。`:101–111`沿前两轴展平，`:118–131`用同一索引读取各字段并按11项顺序yield，最后一项为None。
- `legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py:77–87`对动作、价值、logp及分布参数detach，观测／特权观测／标签直接暂存引用。`:90–99`在env.step后克隆奖励、进行timeout处理并入库，随后清Transition。此处暂存引用、detach和入库copy_是不同操作；detach本身不分配独立存储。


## 34. A07超时补偿与统计副作用的官方事实

2026-10-31只读核查，路径相对固定官方仓库`U/`；以下仅为源码事实，未执行GAE或训练测试。

- `legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py:90–99`先clone rewards，再按infos中的time_outs乘以已保存的transition.values与gamma加到训练奖励中；该values在`:79–80`由动作前critic_obs取得。此路径没有对reset前终止后继特权历史另行evaluate。
- `legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:176–180`按check_termination、compute_reward、reset_idx、compute_observations顺序执行。`:202–205`先判碰撞／倾角失败，独立计算episode到时标记再与reset_buf取或；这段time_out_buf未排除同时物理失败。`:292–294`按send_timeouts配置将该标记交给算法。
- `legged_gym/b2_gym_learn/ppo_cse_pf/rollout_storage.py:72–86`的delta和GAE递推都乘以1-done，returns先由未标准化advantage与values相加，随后全rollout标准化advantages。`on_policy_runner.py:119–151`在24步采样循环后，以最后返回的privileged_obs调用compute_returns，然后update。
- `rollout_storage.py:88–94`的get_statistics用done=self.dones取得同一张量引用，随后done[-1]=1；这会原地改写实际缓冲区的末行dones，函数内没有在该写入前clone。


## 35. A07训练配置注入与更新数值的官方事实

2026-10-31只读核查，路径相对固定官方仓库`U/`；未装配运行环境或执行学习更新。

- `legged_gym/envs/base/legged_robot_config.py:196–209`配置value_loss_coef=1、use_clipped_value_loss=True、clip_param=0.2、entropy_coef=0.01、epochs=5、minibatches=4、lr=1e-3、schedule=adaptive、gamma=0.99、lam=0.95、desired_kl=0.01、max_grad_norm=1。`legged_gym/envs/b2/b2z1_pos_force_config.py:313–315`的algorithm继承该基类，仅重申entropy_coef=0.01。`legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py:53–55,73–74`读取algorithm配置并以kwargs传入PPO。
- 同学习目录`ppo.py:21–35`构造默认epochs／minibatches=1、gamma=0.998、entropy=0、schedule=fixed，与上述配置不同。`:12–15,53–55`辅助Adam lr=1e-5、子步1；两套Adam均只显式传lr，其余Adam选项没有在该调用中重写。
- `ppo.py:132–144`在每个minibatch主更新前计算log(sigma_new/sigma_old+1e-5)加其余高斯KL项，沿动作维sum、batch mean；KL大于0.02时主lr除1.5且下限1e-5，小于0.005且大于0时乘1.5且上限1e-2。该段仅更新主optimizer参数组。
- `ppo.py:148–164`策略ratio使用联合logp差exp；价值裁剪以旧values加新旧差clamp为候选，与未裁剪平方误差逐元素max再mean，未额外乘0.5。`:169`主阶段clip_grad_norm_，`:181–200`主步后辅助重新前向且该段无辅助梯度裁剪。`rollout_storage.py:84–86`returns后全rollout优势以默认torch.std并加1e-8标准化。
- `ppo.py:124`调用ActorCritic.act刷新分布；`actor_critic.py:143–150`该调用在构造Normal后执行sample。`ppo.py:125`取storage actions计算logp，未将这次新采样动作发送环境或用来替换存储动作。


## 36. A07完成量、日志计数与初始化的官方事实

2026-10-31只读核查，路径相对固定官方仓库`U/`；未启动训练或仿真。

- `legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py:84–88`初始化tot_timesteps和current_learning_iteration为0并调用env.reset。`:115–116`按成员迭代和请求轮数构建循环，`:151`执行update，`:154–157`按日志目录调用log、按循环索引it条件保存；`:160–161`到整次循环结束才将current_learning_iteration增加请求轮数。`:236–249`保存成员iter并在load写回。
- 同文件`:163–165`在log函数中增加tot_timesteps=N×T和tot_time；这些字段的递增位于日志函数内部。循环中以it命名文件，与save读取的成员current_learning_iteration来源不同。
- `legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:125`在decimation子步内比较global_steps>force_start_step×24，`:141`每个环境step返回前使global_steps＋1；`:153–154`每个post_physics_step增加episode_length_buf和common_step_counter。`legged_gym/envs/base/base_task.py:104–109`的reset调用reset_idx后执行一次零动作step。
- `legged_gym/scripts/train_b2z1posforce.py:19`调用learn时传init_at_random_ep_len=True；`on_policy_runner.py:100–101`据此以torch.randint_like直接覆盖episode_length_buf。该赋值本身没有执行对应数量的物理或策略转移。


## 37. A07检查点加载与随机源初始化的官方事实

2026-10-31只读核查，路径相对固定官方仓库`U/`；未读取用户训练权重或运行恢复测试。

- `legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py:236–250`的torch.save字典仅含model_state_dict、optimizer_state_dict、iter、infos；load读取文件、load_state_dict模型，可选恢复主优化器，再写回成员迭代。该段未恢复辅助Adam、随机源、课程或全量运行配置／身份。
- `legged_gym/utils/task_registry.py:147–155`先按有效配置构造OnPolicyRunner，再在resume分支取得路径调用runner.load。`on_policy_runner.py:84–88`构造时先初始化计数并调用env.reset，早于该load；构造与reset可发生随机消耗，不能仅据初始seed断定保存时随机流已被恢复。
- `legged_gym/utils/helpers.py:37–47`的set_seed依次设Python random、NumPy、Torch CPU及CUDA随机种子；`task_registry.py:93`在环境创建路径调用。设置种子与保存／加载各随机生成器的当前state是不同操作；上述load没有get_rng_state／set_rng_state调用。


## 38. A07保存间隔与直接目标文件写入的官方事实

2026-10-31只读核查；路径相对固定官方仓库`U/`，未执行保存或文件系统故障测试。

- `legged_gym/envs/base/legged_robot_config.py:214–218`定义每环境24步、max_iterations=60000、save_interval=200。`legged_gym/envs/b2/b2z1_pos_force_config.py:319–321`的runner只覆盖运行名／实验名；`legged_gym/scripts/train_b2z1posforce.py:19`将train_cfg.runner.max_iterations传入learn。
- `legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py:149–161`在update后日志与周期保存，循环索引it%save_interval==0时以model_it命名，最后增加成员迭代并再次保存。`:236–242`直接torch.save到path，此函数没有同目录临时文件、flush／fsync、回读或原子发布步骤；此源码事实不等于已发生实际文件损坏。


## 39. A08球坐标候选、初始化与路径的官方事实

2026-10-31只读核查，路径相对固定官方仓库`U/`，提交同第23节；未执行采样、Gym或训练。

- `legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:887–890`分别对半径、仰角、方位角调用torch_rand_float。`:921–928`初始化分支从配置读取start／end；`:935`普通段start取上一goal，`:936–940`最多10次采样并筛去通过collision_check的行，循环耗尽后该段未显式抛出失败。
- 同文件`:942–947`对球坐标插值离散点转换为笛卡尔坐标，检查禁入盒与地下边界；该段没有关节逆运动学或全机器人连续碰撞检查。`:949–973`按goal_timer／traj_timesteps作截断插值，再转换到世界坐标，增加timer并调用重采样。`:1388–1389`在缓冲区初始化时采样traj／hold时长，普通段重采函数没有重新采这些时长。
- `legged_gym/envs/b2/b2z1_pos_force_config.py:9–14,24–30`分别给出时长、离散检查点数及B2初始起止点／随机范围。半径[0.35,0.95]m与论文附录[0.35,0.85]m的差异见第23节。
- 本地静态证据`P/.scratch/unifp-v3-stage1-reproduction/knowledge/a04_static_kinematics_20261031.json:4–5,8–15,38–44`记录URDF来源及SHA、名义臂角和TCP候选球坐标，其中仰角1.2636979856rad约72.4°；该记录的limitations明确未验证碰撞、PhysX、地面支撑或动态可达性。


## 40. A08当前V3限位、几何与离线位置求解事实

2026-10-31依据当前工作树只读资产计算；没有运行Gym或控制器，没有改动旧代码／资产。

- `L/robot_description/a1arm_description/urdf/a1arm_rgm3_original_movable.urdf:47–53,76–82,105–111,134–140,163–169,192–198`定义六个臂关节的origin、局部z轴与角度边界；`:264–269,493–497`定义base_link至arm_base的固定安装链。`:499–509`的base质量／碰撞盒为65.8456kg、0.70×0.46×0.24m，局部中心z=−0.095m。四轮半径／安装值见第31节与当前参数事实表。
- 本轮分析记录见[A08离线位置核查](a08_workspace_position_audit_20261031.md)及[原始JSON](a08_workspace_position_audit_20261031.json)。输入使用当前资产SHA、TCP候选和名义臂角；r0.45—0.70m、仰角0—60°、方位±60°，每个臂角限位内收5°。45个网格端点、128个随机端点、12条首段路径的132个点均求得位置残差<0.1mm的解。模型名义TCP与既有静态记录一致。
- 本次求解损失只有位置，不含姿态、碰撞、速度／力矩、分支连续性或动力学；132个路径点含重复初始点。采样、版本、算法参数、关节解及每点残差在上述JSON逐项保存。上述数字是本次计算输入和结果，不代表配置已获批准或物理验证通过。


## 41. A08补偿公式、刚度缓存与论文反力项的对应事实

2026-10-31只读核查，路径相对固定官方U，提交同第23节；未执行Gym、补偿计算测试或训练。

- 论文文本身份见第23节。对同一文本read_text后的字符[10500,13800)，§3.1由完整阻抗式(1)简化为F=K(x−x_des)，式(2)为x_target=x_cmd+(F_ext+F_cmd−F_react)/K。附录A式(A.6)将无外部扰动／主动施力命令的情况写为x_target=x_cmd；这些是数学描述，不是当前工具接触测量代码接口。
- `legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:370—377,1891—1900`分别在特权目标与位置奖励中读取`forces[:,gripper_idx,:3]`，将current_Fxyz_gripper_cmd按base_yaw_quat转至世界坐标，相加后逐元素除gripper_force_kps并加几何目标。该两处没有读取真实工具物体接触反力，也没有使用decoder预测。
- `legged_gym/envs/b2/b2z1_pos_force_config.py:148—150`将randomize_gripper_force_gains设为True、gripper_force_kp_range设为[200,200]。ENV:758—768调用torch.rand为每个环境生成一个标量再广播至3轴；范围相等时得到固定200，仍发生随机数调用。ENV:1520—1522分配命令力及[N,3]刚度缓存。
- `legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:125—130`将人工力缓存传入Gym世界坐标刚体施力接口；`:362—366`将两处人工力缓存转至yaw轴。刚度配置与CFG:230关节stiffness是不同字段；此处补偿计算没有将关节PD值作为分母。


## 42. A08默认奖励配置和四足函数的实际行为

2026-10-31静态读取固定U源码与AST；未导入Gym或运行奖励／训练。

- `legged_gym/envs/b2/b2z1_pos_force_config.py:252—311`定义rewards和独立scales类；scales没有基类，实际列出27个非零和9个零权重，不能因为base配置中存在某项就认定本任务启用了它。三个主项tracking_lin_vel_force_world=2、tracking_ang_vel=1、tracking_ee_force_world=2。arm_orientation／tracking_ee_orn／tracking_ee_orn_ry等为0。此处为乘dt前的配置值。
- `legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1886—1900`几何和补偿位置误差均为世界三轴绝对值求和，指数为exp(−2*error/tracking_ee_sigma)。`:2011—2029`分别有普通base速度指数函数与base力补偿／non_stop_sign门控版本；`:2057—2060`转向项比较commands[2]与base_ang_vel[2]，平方误差取指数。
- 同文件`:2121—2126,2182—2192,2206—2212,2222—2230,2254—2262`分别涉及默认腿姿态、参考腿角、髋角／足接触力、alive常数1与接触力加权body线速度拖动、步态相位接触匹配。足拖动读取刚体世界线速度，不是轮胎接触点滑移；足接触力阈值来自CFG:263的200N。
- `ENV:303—322`遍历函数并将分项加到reward与episode_sums，only_positive_rewards分支可裁总reward，再处理termination；`:1551—1574`剔除零权重，将非零权重乘策略dt后注册函数与episode_sums。CFG:256关闭only_positive_rewards；当前独立scales未列termination，函数存在不等于该项被默认启用。
- V3当前URDF质量求和83.317915372851kg见参数事实表；取9.81m/s²并四轮均分，算术静载约204.3N/轮。它不是实际动态轮载或可直接采用的阈值。


## 43. A08执行器正则、状态采样与软边界的源码事实

2026-10-31只读核查固定U及当前V3默认URDF；没有导入Gym或执行奖励测试。

- `U/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:114,865—885`每子步覆盖torques并按URDF effort裁剪后提交；`:1931—1945`力矩／速度按腿0:12与臂12:17平方和，`:1962—1976`加速度用last_dof_vel差除策略dt、动作变化用last_actions与actions差。固定CFG默认torques非零而torques_arm未列；dof_vel／dof_vel_arm、dof_acc／dof_acc_arm和action_rate／action_rate_arm均非零。
- 同ENV`:1902—1912,1926—1929`分别是机身轴垂向速度平方、机身轴xy角速度平方、projected_gravity的xy平方和，以及root world z减配置高度的平方。独立scales未列orientation；函数存在不等于默认启用。
- 同ENV`:744—755`由URDF上下限中点及宽度乘soft_dof_pos_limit构造内部位置软界；`:1987—2000`分别为超软角界线性和、超软速度界线性和（逐关节最大1）、超软effort界线性和。默认scales启用dof_pos_limits和torque_limits，未列dof_vel_limits。CFG`:259—262`分别为0.8、1、0.9、0.50m。
- 同ENV`:1978—1984`的collision按选定body净接触力范数大于0.1N计数，termination按reset且非time_out返回；默认CFG未列termination。ENV`:230—234`在reset清零动作与速度历史；`:184—189`在post_physics_step后段更新历史，具体调用先后见第23节及已确认A04／A07记录。
- `L/robot_description/a1arm_description/urdf/a1arm_rgm3_original_movable.urdf:3—257,269—492,499—542,572—590,620—638,668—686,716—734,741—762`静态XML统计：28个link中23个具有collision；四个轮为fl_Link／fr_Link／bl_Link／br_Link，另19个为base_link、arm_base、arm_seg1—arm_seg6、gripper1／gripper2、aluminum_profiles及四组相机／连接件。该统计不代表Gym实际刚体映射或接触过滤已经验证；各link碰撞定义见该文件对应link的collision元素。


## 44. A08主跟踪权重和sigma的固定配置事实

2026-10-31只读核查固定U提交同第23节，未导入Gym或运行训练。

- `U/legged_gym/envs/b2/b2z1_pos_force_config.py:257—258`分别设置tracking_sigma=0.25、tracking_ee_sigma=1.0；`:275—276,281,304`分别设置tracking_lin_vel_force_world=2.0、tracking_ang_vel=1.0、alive=1.5、tracking_ee_force_world=2.0。
- `U/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:1891—1900`以世界三轴绝对位置误差和计算exp(-2*error/tracking_ee_sigma)；`:2011—2014,2057—2060`分别将机身xy速度平方误差和、机身z角速度平方误差除以同一配置tracking_sigma后取负指数。源码是同一数值字段复用，不是两份独立速度尺度。
- 上述实际配置定义与函数行为为代码事实；它们本身不提供V3动态跟踪精度、最优权重或实验验收阈值的证据。


## 45. A08既有V3位置解的软关节区间静态对照事实

2026-10-31复用既有位置解，没有重新运行IK、Gym或训练；没有修改既有位置核查记录。

- `L/robot_description/a1arm_description/urdf/a1arm_rgm3_original_movable.urdf:52,81,110,139,168,197`给六臂关节角度和effort，[40,40,27,7,7,7]N·m；`:561,609,657,705`给四转向角范围±1.5708rad、35N·m，`:569,617,665,713`给四驱动80N·m。
- [软角界静态对照JSON](a08_soft_limit_saved_solution_audit_20261031.json)记录来源文件SHA、公式与结果，输入为第40节已保存305个六臂位置解。按URDF中点和原宽度的0.80／0.90／0.95／0.97构造软界，越界解个数为84／46／29／0；最大超界角总和分别约0.964607／0.388647／0.100667／0rad。以权重−10、周期0.020s作算术对照，对应每步最大罚绝对值约0.192921／0.077729／0.020133／0。
- 这305项包含路径重复初始点。统计的是既有解分支，未寻找同一目标其他解，不证明目标必须越界或不可达；不含碰撞、速度／力矩、动态或学习证据。上述不同软比例是本轮计算输入，不是已获批准的运行配置。


## 46. A08底盘稳定及碰撞数值的源码事实

2026-10-31只读核查固定U，未运行Gym或接触实验。

- `U/legged_gym/envs/b2/b2z1_pos_force_config.py:262,282,285,290,298`分别配置base_height_target=0.50m、lin_vel_z=−1.5、ang_vel_xy=−0.02、collision=−5、base_height=−2。ENV路径同第43节，`:1902—1908`读取机身轴线／角速度平方，`:1926—1929`读取root world z与目标高度平方差，`:1978—1980`按选定body净接触力范数>0.1N计数；此0.1为函数内常数，不是现有配置字段。
- ENV`:200—205`物理结束接触采用选定body净力范数>1N，并判pitch绝对值>1.0rad或roll绝对值>0.8rad；固定CFG`:248`的terminate_after_contacts_on为空列表。A04对V3的base_link映射、失败优先及阈值候选是已确认设计，不能写成该默认B2配置已经采用V3映射。
- 第42、43节已记录选定body计数、净力与接触对的区别以及默认配置开关。以上源码不提供V3正常接触背景力、0.1N误触发率或动态稳定性实验结果。


## 47. A08官方episode奖励统计与Runner归约事实

2026-10-31只读核查固定U环境reset、reward注册及Runner记录路径，未运行Gym或训练。

- `U/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:305—322`每个策略转移清零总reward，遍历有效分项乘已缩放weight并累加到`rew_buf`及逐环境`episode_sums[name]`；如果启用正值裁剪只裁总reward，分项累计保留未裁前各项。默认配置关闭裁剪，termination另行处理。
- 同ENV`:285—289`在reset的一组`env_ids`上，对每个分项episode_sums先做组内均值，再除以固定`max_episode_length_s`，写入一个`extras["episode"]`标量并清零这组环境的累积器。因此提前失败episode没有按本身真实episode时长归一；同一reset组中各环境的episode贡献先合并，未保留独立条目。
- 同ENV`:1551—1574`仅非零weight注册函数并乘策略dt一次；对每个注册项分配每环境的float32累加向量。
- `U/legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py:132—142`每个step收到`infos['episode']`时把整个映射追加到`ep_infos`，另独立累计总环境reward与episode长度缓冲区，并按done环境抽取总回报／长度后清零。`:163—181`在迭代日志阶段，把多个reset批次的同键值拼接后求均值。环境输出单标量时，每个reset批次标量被同权处理，数值结果并不等价于按全部完成环境数作加权均值。
- 这些是固定官方实现事实，不意味着V3必须沿用固定最大episode时长分母；也不提供V3奖励字段、真实物理误差统计或日志频率设计结论。

## 48. V3 URDF网格引用与归档软链接事实

2026-10-31开始实施票据02时对固定资产做只读路径核查；新开发仓库内资产复制结果不属于本节源码事实。

- `legacy/v3_reference_20261030/robot_description/a1arm_description/urdf/a1arm_rgm3_original_movable.urdf:12—18`等网格元素使用相对引用`../meshes/approx_sim/...`；引用集合共47个不同路径。
- 归档中的部分`meshes/approx_sim/*.glb`引用是符号链接，例如`arm_base_collision.glb`指向同目录上一级的`arm_base.collision.glb`。文件系统按该链接读取时返回目标文件字节；URDF引用名与解析后的目标文件名存在差异。
- 归档文件系统的符号链接目标位于归档内资产目录，不是机器人资源资产的一部分。此事实只涉及文件路径解析，不代表Gym已接受网格格式或物理几何。


## 49. 当前V3资产、运行时与环境调用事实

2026-10-31按当前工作树核对桌面基准资产、V3环境实现和Gym回归。静态文件事实与运行记录分开列出。

- `UniFP_v3/legged_gym/envs/v3/v3_pos_force_config.py:25—34`将活动V3配置指向桌面派生URDF，关闭fixed-joint collapse及cylinder替换并以`base_link`作为接触终止名称；`:36—54`设置名义root高度0.34m及14个关节默认值；`:96—116`定义15维命令、底盘速度候选、A08目标/时长验证候选和名义base高度0.32m。旧`resources/robots/v3_a1/asset_manifest.json`标记为历史问题资产，其派生URDF SHA256为`03db4e846dd77b51f1d7bb62a373a5fa4e3cd16535a21ae6a1068c7e24dd0239`。桌面候选归档与记录为`UniFP_v3/docs/experiments/archives/v3_a1_desktop_accepted_20261031.tar.gz`及`UniFP_v3/docs/experiments/2026-10-31_v3_desktop_asset_repair.md`。
- `UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:32—68`仅在V3路径显式设Gym DOF `EFFORT`并清零导入stiffness，创建V3 actor后按活动资产manifest应用shape级局部装配过滤；`:87—138`按关节／body名称构建索引、分配观测历史／标签／命令／目标状态并执行初始reset。此实现没有修改`UniFP/`官方副本。
- 同文件`:140—218`实现物理step、核心非有限状态停止、base失败／timeout区分、命令推进、terminal Critic帧和done局部reset；`:220—297`实现局部root／DOF reset、Gym张量刷新、TCP读回和候选地面状态核验；`:299—490`实现底盘命令采样、球坐标目标段采样／插值／候选筛查和命令更新；`:492—521`按14维动作计算10个位置关节与4个轮速关节显式PD力矩；`:547—578`由`arm_seg6`和TCP局部偏移取世界TCP，并按yaw及单一名义高度来源转换至球坐标。
- 同文件`:580—675`构造58维Actor帧、98维Critic帧、12维监督标签和32／3帧历史；当前step仍将奖励置零，尚未接入A08奖励或PPO Runner。目标筛查仅覆盖采样域、有限性、非零半径和TCP候选离地条件，不证明整臂碰撞或动态可达性。
- `UniFP_v3/tools/check_v3_desktop_runtime.py:17—154`在32个真实V3 Gym actor上验证reset TCP目标起点、A零力命令、策略边界目标插值／段切换、250步底盘命令重采、1000步timeout终止Critic新命令对齐及局部reset隔离，并回读质量／惯量和轮系状态。2026-10-31报告位于`.scratch/v3-desktop-model-repair/knowledge/v3_runtime_check.json`；该短测仍不是Actor/Critic训练闭环或完整物理安全认证。

## 50. A04目标命令时序及timeout定向回归事实

2026-10-31基于当前V3工作树实现后，在专用Isaac Gym Preview4 Python 3.8/Torch 2.3.1运行时执行4环境GPU PhysX定向短测；报告为`.scratch/v3-desktop-model-repair/knowledge/v3_runtime_check.json`。

- `UniFP_v3/legged_gym/envs/v3/v3_pos_force_config.py:96—116`设置15维命令及A08验证候选；仰角、方位角用弧度保存，几何余量仍未冻结。配置的`base_height_target=0.32m`与TCP球心z安装偏移`0.09535m`共同决定名义球心高度`0.41535m`。
- `UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:87—90`将20s episode按20ms策略周期取最近整数为1000步，避免基类对浮点比值执行ceil得到1001步。`:172—218`完成单步核心状态有限性、base失败／timeout判定、命令推进和reset前终止Critic构造；`:252—297`在root／DOF索引写入并刷新Gym张量后读取TCP，核验核心有限值和候选离地高度；`:319—490`实现目标段采样、策略边界插值、候选筛查和250策略步底盘速度命令重采。筛查不计算IK、全臂碰撞或连续轨迹间隙，不构成冻结安全域认证。
- `UniFP_v3/tools/check_v3_desktop_runtime.py:47—139`在32环境真实Gym实例核对reset目标与TCP读回、A零力命令、目标插值与段交接、250步底盘命令重采、1000步纯timeout终止Critic新命令槽及done环境局部reset隔离。报告记录32环境、21个实际策略步并包含一次预置timeout边界；base净接触峰值0N，14 DOF／28 bodies，actor总质量83.3179115kg。该检查仍不是奖励／网络／Runner训练闭环，也不是独立FK、碰撞或动态可达性证明。

## 51. 活动桌面资产独立FK与离散接触探针接口

2026-10-31新增两个只读诊断入口；它们不属于V3训练调用链。

- `UniFP_v3/tools/probe_desktop_asset.py:44—62`构造单Actor GPU PhysX诊断仿真；`:156—168`读取刚体接触候选和net contact force；`:182—188`输出Gym刚体位置、xyzw姿态四元数及DOF状态。
- `UniFP_v3/tools/compare_desktop_fk.py:16—54`将URDF origin和revolute/continuous/prismatic joint转换为齐次运动变换；`:57—95`按parent/child关系遍历链接；`:97—143`将独立FK结果与Gym刚体位置／姿态逐body比较并计算TCP位置；`:146—203`使用保存的球坐标目标和6维关节解重算静态TCP误差，并计算每条11点路径的相邻关节角变化。
- `UniFP_v3/tools/probe_a08_gym_contacts.py:18—41`定义审计输入、活动资产、局部过滤、样本范围和接触偏移；`:49—83`创建固定base、零重力、EFFORT模式的单Actor仿真并应用manifest定义的局部过滤；`:85—118`逐样本写入6臂关节状态、推进一个物理step，并按body name聚合`get_env_rigid_contacts`返回的接触候选；`:138—162`提供命令行入口及JSON产物。
- 两个入口的输入shape均为每姿态6个臂关节角（rad），TCP局部偏置是`arm_seg6`坐标系下三维位置（m）；接触入口单姿态step为0.005s、base固定且重力为零。该接口不计算控制器、连续轨迹、速度雅可比或自由底盘动力学。

## 52. 官方B2-Z1末端目标、策略动作与IK helper事实

2026-10-31按固定官方工作树的V3迁移参照实现核对控制调用链。

- `UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:102—106`将策略动作宽度配置为17；`:227—235`设置P控制、动作尺度0.25、decimation=4及各关节组PD增益。`:58—64`给出B2＋Z1名义关节角。
- `UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:865—885`将策略动作缩放后并入位置PD目标，追加末端夹爪关节控制并限幅输出力矩；其环境训练step从基类接收策略动作并送入该计算器。`:1886—1900`分别按末端世界位置误差及受力补偿后的末端位置误差计算指数奖励。
- 同文件`:942—947`把球坐标起止目标线性插值成10个笛卡尔末端点，只检测禁入盒与地下高度。`:949—963`每策略步发布插值目标并更新相应末端目标姿态表示；该检测没有读取关节解或检查全臂碰撞。
- 同文件`:857—863`定义阻尼最小二乘`control_ik`辅助函数；当前固定官方树搜索结果仅有定义处，无调用点。因此该辅助函数不是这里确认的策略→关节控制执行链。
- 同文件`:2241—2252`的`_reward_action_smoothness`实际切片仅覆盖动作前12维；不能用它断言官方手臂输出有该项显式平滑奖励。
- Z1官方URDF的六个命名关节及其局部轴／硬限位见`UniFP/resources/robots/b2z1/b2z1.urdf:1569—1753`；V3桌面资产另有自身`arm_joint1..6`链和限位，不能一一复用Z1的关节解。
- `UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:548—556`以`arm_seg6`位姿加局部`[0,0,0.07961206]m`构造V3运行时TCP；旧A08参数记录的候选点为`[-0.00014369,0.00038068,0.07961206]m`，两者横向偏置距离约`0.00040690m`。按两种点定义重算保存IK解会得到不同目标位置残差，不能混称同一TCP定义。

## 53. V3目标路径筛查的官方采样对照

2026-10-31按官方10点筛查要求调整V3实现后，工作树事实如下：

- `UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:11—14`将禁入盒上下界、地下阈值和`num_collision_check_samples=10`作为B2-Z1配置。`UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py:942—947`在球坐标段上插值后转成笛卡尔TCP点，按任意样本进入该禁入盒或低于地下阈值拒绝候选。
- `UniFP_v3/legged_gym/envs/v3/v3_pos_force_config.py:105—114`包含V3目标球坐标候选域、1—3s移动、0.5—2s保持、`num_path_check_samples=10`、5cm候选最小TCP高度及最多10次候选重采。V3没有与B2球心／机器人布局对应的笛卡尔禁入盒配置。
- `UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:319—355`对未通过筛查的候选重采，所有候选耗尽时抛出包含环境索引和地面拒绝计数的`RuntimeError`。`:396—442`将每条段的起止球坐标按10个含起止点的均匀进度采样，对方位使用最短角差，按`r*sin(elevation)`加V3名义球心高度判断TCP离地，并拒绝非有限路径或非正半径。
- 此次调整前，`:396—440`按移动段策略step采样，最大样本约为`ceil(3s/0.02s)=150`。调整后路径筛查样本数固定为10，与官方样本数相同；没有复制B2禁入盒数值，也没有在该检查中求解机械臂关节或检查全臂碰撞。
- 路径端点/每段数据shape为`[K,3]`，元素顺序为半径m、仰角rad、方位角rad；环境索引为`[K]` long tensor。筛查返回两个`[K]` bool mask：整段有效性与地面拒绝。路径筛查只在目标段选择时运行，不是在物理step逐帧调用。

## 54. 环境训练进度与物理step计数的现状

2026-10-31核对A07/A09训练时钟设计与V3环境实现：

- `UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:70—137,208—245`完成基类初始化后将`completed_iterations`、`training_policy_steps`、`training_transitions`初始化为0；`set_training_progress`接收完整迭代数与累计策略步，检查Python整数／非负值、24步rollout边界和同进程单调性，再设置`training_transitions=num_envs*training_policy_steps`。
- `UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:140—219`的环境`step`每次只推进一次真实策略转移及episode计数；`:442—480`处理任务目标与命令时序。reset不应更新全局学习进度。
- `UniFP_v3/legged_gym/envs/base/legged_robot.py:82—92`的`post_physics_step`每次调用均执行`common_step_counter += 1`；其`:275—288`把该物理调用计数用于父类按环境时钟触发的push。A07/A09已确认训练课程时钟应由Runner发布，`common_step_counter`不等价于完整PPO迭代数或训练课程计数。
- `.scratch/unifp-v3-stage1-reproduction/design/part-a/07-PPO缓冲区与Runner.md:324—369`定义三计数及中间rollout语义：`T=24`，完整边界为`training_policy_steps=24*completed_iterations`，全环境转移数为`N*training_policy_steps`；采集中可发布`(I,S+j)`，方法只保存进度且无仿真副作用。`.scratch/unifp-v3-stage1-reproduction/design/part-a/09-课程与域随机化.md:94—109`确认课程只读该发布进度、初始化／reset不推进全局时钟。
- 因此基类`common_step_counter`继续承担官方环境的物理调用／push时序用途，V3发布的完成迭代数与累计策略步数属于独立学习进度状态；二者不互相推算。进度setter没有Gym、张量状态、episode长度或命令副作用。

## 55. 票据03实施前的V3 Actor/Critic源码与装配链

2026-10-31核对A05/A06模型确认记录与本地实现：

- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py:7—16`的`AC_Args`沿用官方可变类列表和通用的`gripper_pos_loss`名称；`:21—47`按构造输入维数计算latent，`:51—105`构造共享历史encoder／decoder、策略MLP和独立Critic MLP，`:113—116`建立直接std并关闭Normal全局参数校验；`:143—180`提供训练采样、推理预测和Critic value路径。
- 该本地`ActorCritic`是V3仓库内随官方复制的共享B2类，不应为V3语义／推理接口修改；B2配置默认类名是`ActorCritic`，见`UniFP_v3/legged_gym/envs/base/legged_robot_config.py:211—214`。
- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py:40—42,64—79`仅导入通用`ActorCritic`，并从runner配置的`policy_class_name`解析模型类。`:79`把环境提供的`num_obs`、`num_privileged_obs`、`num_pred_obs`、`num_single_obs`和`num_actions`传入构造函数。
- `UniFP_v3/legged_gym/envs/__init__.py:4—12`将V3任务注册到本地TaskRegistry；`UniFP_v3/legged_gym/envs/v3/v3_pos_force_config.py:157—165`当前仅覆盖V3 rollout步数、迭代上限和实验名，V3仍继承base PPO policy class默认值。
- 活动V3环境`UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:625—661`生成监督标签：`[0:3]`base线速度×2，`[3:6]`TCP球坐标×`[0.5,1,1.3]`，`[6:12]`两种人工力标签且A阶段为零；`:682—704`发布`obs[N,1856]`、`privileged_obs[N,294]`、`obs_pred[N,12]`。环境张量顺序对应A05/A06已确认接口。
- 新V3模型装配应由V3 policy config显式选择，保留base/B2的`ActorCritic`默认类；构造维数必须与上述环境接口一致。A05确认网络为1856→512→256→128→64 encoder、122→512→256→128→14 Actor、64→128→64→12 decoder；A06确认Critic为294→512→256→128→1。PPO更新实现仍以A07/A05调用关系为准，不在模型类中自行更新参数。

## 56. V3专用ActorCritic实现与当前装配状态

2026-10-31按A05/A06已确认设计核对实现，并使用本机Isaac Gym Preview4 Python 3.8/Torch 2.0.1执行模型装配检查；未创建环境、未执行物理step或训练。

- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_actor_critic.py:11–28`固定V3各shape、64维latent、Actor/Critic/encoder/decoder层宽及四组辅助标签元信息；`:36–101`检查固定输入schema与网络配置，并构建encoder `1856→512→256→128→64`、Actor `122→512→256→128→14`、decoder `64→128→64→12`、Critic `294→512→256→128→1`和14维直接可训练std。
- 同文件`:127–172,216–272`提供带shape／dtype／设备／有限性检查的history encoder、对角正态动作分布、采样动作、按动作维求和的log probability及12维辅助预测；`:274–323`提供动作均值推理和可选同设备`state_estimates[N,12]`输出；`:325–350`单独计算Critic value，不使用Actor history或latent；`:352–371`为无递归状态的reset与未使用forward接口。模型本身不执行PPO更新。
- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py:40–43,67–79`导入V3及原ActorCritic供`policy_class_name`解析，并仍从环境取得五个schema维数构造策略。`:67`以配置类名执行`eval`，因此V3类需在Runner全局作用域导入。
- `UniFP_v3/legged_gym/envs/v3/v3_pos_force_config.py:193–203`将V3 Runner配置显式设为`V3ActorCritic`并设置已确认策略层宽（行号已按票据04后配置更新）；`UniFP_v3/legged_gym/envs/base/legged_robot_config.py:211–214`仍以官方`ActorCritic`为基础/B2默认类名。按官方入口先导入`isaacgym`后，Preview4检查确认两类名分别解析到正确类，且V3配置参数成功构造模型；该模型总参数1,685,033。
- `UniFP_v3/tests/test_v3_actor_critic.py:10–124`验证固定线性层宽、采样／推理／价值／辅助输出shape、动作推理无梯度、策略／Critic／辅助损失梯度范围，以及错误schema和非正std失败。`PYTHONPATH=UniFP_v3 conda run -n tavis python -m pytest -q UniFP_v3/tests/test_v3_actor_critic.py`结果为3 passed；模型、Runner和测试文件通过Python语法编译检查。
- 当前V3模型实现使用新独立类；同目录原`actor_critic.py`与官方`UniFP/`文件未修改。真实PPO rollout与更新、checkpoint及训练评估仍未运行，需由后续训练票据接通。

## 57. V3奖励、随机化与episode统计实施前源码事实

2026-10-31核对04票据实施前的V3活动工作树与所继承官方基类；本节为修改前快照，行号及“当前”均对应当时状态，实施后的事实见第58节。

- `UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:145–207`定义V3策略步：动作经4个5ms子步执行，刷新root/contact/rigid状态后计算失败与timeout；当前明确将`rew_buf`清零，然后推进命令、构造terminal Critic帧、局部reset并发布下一观测。该覆盖方法没有调用基类`compute_reward()`，所以V3当前reward仍恒为零，episode统计也未按A08更新。
- 同文件`:135–143`预留末端／底盘人工力、五维质量／COM变化、摩擦、14维执行系数和轮接触有效标志等张量；`:274–316`reset清动作历史、标签和两类人工力，当前不采样episode执行系数／随机化样本，也未记录完成episode。
- 同文件`:623–673`构造Actor/Critic/标签帧。当前Critic拼接宽度为98；`:658–672`已放入上述物性、四轮接触、状态、动作和命令槽位。`_build_frames`中的力标签目前直接补零，随机化相关特权缓存仍是占位初值，奖励尚未读取对应A08物理量。
- `UniFP_v3/legged_gym/envs/base/legged_robot.py:45–47`的基类初始化会调用`_prepare_reward_function()`，但该函数只建立基类通用奖励表；`:158–175`的`compute_reward()`按非零项乘权重求和并更新基类`episode_sums`，`:475–498`会原地删除零权重并将其余scale原地乘`self.dt`。V3显式step并未调用该计算函数；A08决定使用20ms策略周期、保留原始配置权重并派生有效权重，故不能直接复用其原地scale副作用。
- 基类配置`UniFP_v3/legged_gym/envs/base/legged_robot_config.py:94–99`默认摩擦随机化开启、范围`[0.5,1.25]`，base增重关闭，push开启；`:623`按`push_interval_s/dt`转换基类物理step push间隔。基类`:209–232`在actor创建期按64个桶为所有shape分配摩擦，`:262–273`的mass随机化会按`props[0]`增加质量，`:367–372`以`max_push_vel_xy`整体改写base XY速度。V3活动配置覆写目标、控制及reward scales，但截至当前未单独覆写A09生命周期和数值；票据04需按V3 name mapping和已确认数据契约显式接管这些行为。
- 该修改前快照的`:191–200`推进目标／命令时，reward仍为清零占位，尚无真实奖励／物理误差结算链；不能描述成已实现奖励误用了新命令。A08-Q5c要求按本次旧命令与积分后新状态先结算，再推进命令、构造终端快照、保存记录和局部reset。票据04后的具体执行顺序见第58节。

## 58. 票据04后的奖励、随机化与episode数据链

2026-10-31核对当前工作树及本机Preview4短测；下列路径相对开发根`P/`。此节为实施后事实，替代第57节对应的修改前状态。

- `UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:33–53,536–581`固定23个奖励名、19个非轮接触body名及4类物理误差；保留原始配置权重、派生单次dt加权，调用各raw函数一次后求和。负总reward不截断，raw或总值非有限时抛错。`:57–79`恢复URDF硬限位，独立构造软位置界。`:628–810`按转向／驱动／机械臂实现各奖励，驱动轮无绝对转速惩罚。
- 同文件`:387–456`先检查A力通道为零并按发布时钟push，执行4个物理子步，刷新状态并判物理终止／纯timeout；按执行期间旧命令结算reward与物理误差、暂存episode记录，再更新动作历史／命令、捕获终端294维Critic输入、局部reset及发布下一观测。`infos`包含终止类型、终端环境索引与输入、episode记录及push标记。`:583–626`以世界TCP位置计算几何误差，补偿采用yaw旋转后的人工末端力与命令力之和除`[N,3]`虚拟刚度；A两力为零，补偿目标退化为几何目标。
- 同文件`:112–216`创建时采样按名称定位的base／臂端质量与base局部COM；质量变化同比缩放名义中央惯量，COM平移不改变中央惯量。以`recomputeInertia=False`提交，回读质量、COM、惯量全部矩阵元素及各shape摩擦；`mass_com_delta[N,5]`按base质量kg／COM xyz m／臂端质量kg排序，`friction[N,1]`存回读值。摩擦使用继承基类的创建时64桶采样。
- 同文件`:218–238,812–848`按Runner发布的累计训练策略步触发底盘速度push，不依赖物理调用次数；400策略步一个事件，各环境独立采世界XY速度，同一时钟不重复注入。进度setter不调用物理step或push。`:877–925,1139–1178`在局部reset时采14维执行系数，系数作用于动作目标残差／轮速目标，之后再硬裁剪，不缩放PD增益或力矩。
- 同文件`:240–298`保存／恢复实际物性、执行系数、push、三计数和Python／NumPy／Torch／CUDA RNG；张量快照独立置于CPU。恢复方法回设物性及RNG，不调用reset／step，不保存PhysX全状态。`:1237–1288`的辅助力标签读取实际人工力缓存乘0.01，特权帧使用实际物性缓存，不以literal zeros替代力缓存。
- 同文件`:458–534`为每环境分配raw累积、reward return、真实采样步数及`[N,1000,4]`误差样本；done时取真实样本mean/P95，按真实采样步数乘dt计算时长，补偿／几何TCP误差在A共用样本。完成记录先置于`pending_episode_records`，局部reset仅清对应计数。环境输出`timeout_bootstrap_bonus=None`，表示尚无PPO终端V求值。
- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_episode_log.py:9–51`按批次序列化／append／flush／fsync JSONL，正常落盘拒绝未解析timeout bonus，各episode等权汇总，空batch仅返回count0。`:54–75`以exclusive模式创建metadata并fsync，已有metadata不覆盖。当前Runner尚未调用这两个接口。
- `UniFP_v3/legged_gym/envs/v3/v3_pos_force_config.py:116–149,159–173`实际配置23项权重、软界／sigma及摩擦、base／臂端质量比例、COM、执行系数、push候选；这些候选来源和确认状态见A08/A09设计，源码注释仍明确为验证候选。
- `UniFP_v3/tests/test_v3_rewards.py:12–141`的4个单测验证单次dt加权、负总reward、物理误差mean/P95、局部清理、软／硬位置界和执行系数作用位置；`UniFP_v3/tests/test_v3_episode_log.py:11–53`的2个单测验证等episode归约、空batch、metadata不覆盖及未解析bonus拒绝。2026-10-31本机回归：模型／日志／桌面资产10项pytest通过，奖励4项unittest通过。
- `UniFP_v3/tools/check_v3_reward_runtime.py:16–128`运行8环境27步，检查实际物性回读、惯量比例、局部执行系数、Torch RNG恢复、push与timeout／episode记录。报告`.scratch/unifp-v3-stage1-reproduction/knowledge/a08_a09_runtime_20261031.json`记录全部接口断言通过、奖励范围约`[-0.0294281,0.1223764]`，并保留第23步env3／第24步env2的接触物理终止记录；不含训练网络或optimizer。timeout判定计数手动预置，但record时长仍由真实采样数得到0.54s。
- `UniFP_v3/tools/check_v3_desktop_runtime.py:17–205`显式关闭随机化和push以核验名义合同；本机32环境25步短测非timeout终止为0，结果存于`.scratch/v3-desktop-model-repair/knowledge/v3_runtime_check.json`。两次Gym短测均使用GPU PhysX／CPU张量pipeline；没有目标N的GPU张量容量或训练性能结果。详细命令和实验观察保存在`UniFP_v3/docs/experiments/2026-10-31_v3_rewards_randomization.md`。核查时官方`UniFP/`的`git status --porcelain`为空。

## 59. 票据05实施前的学习链与远程装配事实

2026-10-31读取活动源码及SSH只读核查，以下均为实施前状态。

- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py:53–55,77–104,119–210`仍为官方实现：两套Adam注册全模型，timeout补旧V，辅助日志额外除64；`rollout_storage.py:34–131`使用byte dones、入库copy，但没有完整／returns就绪门控，minibatch按整除后的长度生成索引。源码问题及已确认边界见第35—38节和A07。
- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/on_policy_runner.py:84–161,236–250`构造时reset，采用日志内部分计数、零基保存名及有限字段checkpoint。`UniFP_v3/legged_gym/scripts/train_v3.py:1–44`仍只解析配置和打印身份，不创建仿真或训练。
- `UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:240–298,387–456,866–925`已提供独立训练状态、终端Critic输入及episode记录；reset不step、不推进训练时钟。`v3_episode_log.py:9–75`已有JSONL／metadata窄接口，当前没有Runner调用。
- SSH别名`gpu-server`本轮返回主机`gpuf3`，4张A6000；`/mnt/data_8THDD`可用约901GiB。已有`/mnt/data_8THDD/conda_envs/unifp`，独立`unifp_v3_stage1`及项目根尚不存在；未停止既有进程，未运行训练。GPU空闲量是瞬时观测，不代表设备预留。

## 60. 票据05实际学习链与入口装配事实

2026-10-31，当前工作树中V3训练链已接通，以下为源码和已发生运行的事实，不包含后续验收结论。

- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_rollout_storage.py:10–202`实现预分配、配对copy、bool done、完整rollout门控、全量GAE及一次排列跨epoch采样。`v3_ppo.py:12–272`定义双Adam参数边界、真实终端Critic子批bootstrap、主后辅的20组更新和真实辅助日志均值。
- `v3_on_policy_runner.py:25–165`构造时不reset，以24步采集和全部主辅更新成功为完整迭代提交点；完成后同步I／S／N×S、逐episode落盘和checkpoint。`:226–404`为静态完整状态核验、原子保存和严格恢复；CPU回读核验后才发布检查点，新进程恢复持久物性、双Adam、RNG后重建episode。
- `v3_run_identity.py:10–84`对活动源码与资产字节生成manifest，记录schema和有效学习配置摘要及运行依赖／设备信息；追加预算与保存频率不属于学习语义。`scripts/train_v3.py:30–101`直接构造V3环境与学习器，不走原B2 Runner；零预算完成核验发生在创建Gym前。
- `envs/base/base_task.py:32–36`仅直接暴露N、总Actor／Critic宽度及动作宽度；`envs/v3/v3_pos_force_config.py:18–19`的58／12仍存于`env.cfg.env`。远程r02构造Runner时因此出现AttributeError；修正fixture为真实配置位置后本地可稳定复现，再修正Runner读取位置后19项pytest通过。
- Isaac Gym `gymapi.SIM_PHYSX`类型为`SimType`，Python3.8原生`json.dumps`无法直接序列化；远程r03创建模型后写metadata时出现TypeError。本机Preview4最小命令复现同错，转为`str(gymapi.SIM_PHYSX)`后得到可JSON编码的`SimType.SIM_PHYSX`字符串。入口身份现使用此字符串；PhysX选择不变。
- 远程独立环境`/mnt/data_8THDD/conda_envs/unifp_v3_stage1`已创建，editable仅绑定新V3项目，原unifp绑定仍指官方副本；Python3.8.20／Torch2.3.1+cu121／A6000，使用物理GPU1。1035个源码／活动资产逐文件SHA256与本地一致，manifest见同目录`a07_runtime_source_manifest_20261031.json`。
- r01中文清单读取失败、r02维度装配失败和r03元数据失败均发生于首个完整更新之前，console保留；r03另有I=0的failure.json。`PYTHONUTF8=1`为后续真实运行的显式环境变量。当前r04已发布I=1、2、3的完整更新日志及逐episode记录，尚不据此认定整体短测验收。

- 完整性自审中，`tests/test_v3_runner.py:183–188`删除已训练checkpoint的一个辅助Adam参数slot，旧validator未抛异常；修正`v3_on_policy_runner.py:279`，I>0时空slot也须具有step／exp_avg／exp_avg_sq。相同回归随19项pytest通过，I=0仍允许优化器未初始化；不改变有效训练更新。

## 61. GPU恢复首步异常的官方约束与远程因果对照

2026-10-31，以下均为已发生的源码读取与真实Gym观测。

- 本地Preview4原生文档`/home/sorry/.cache/unifp/isaacgym_preview4/pkg/isaacgym/docs/api/python/gym_py.html:4901`的`Gym.set_actor_rigid_body_properties`明确说明GPU pipeline不支持改变COM，质量和惯量可改变。不能以set/get值一致推定GPU动力学状态已正确应用。
- `UniFP_v3/legged_gym/envs/base/base_task.py:53–55`在创建环境后调用`prepare_sim`；`envs/v3/legged_robot_v3.py:86–109,111–149`的创建随机化先于该调用；同文件`:292`的`load_training_state`又调用`_apply_creation_properties`，发生于仿真已经prepare之后。恢复路径因此重复回写COM及其他物性。
- 最终源码r05主run有129条物理终止记录、涉及99个环境，最早第12个策略步；恢复r02则在第一个策略步S=240发生512条终止，全部环境受影响。完整checkpoint的25个主Adam／14个辅助Adam参数张量的step均从200接到220，I10→11、S240→264及父哈希均正确；物性缓存完全相等。状态字典核验不能排除真实物理异常。
- 远程512环境、GPU张量pipeline、相同配置／seed、零动作的两步对照中：`restore`完整恢复和`properties`仅重复回写物性两组，首步均512个base_contact失败、无roll／pitch失败；base高度从reset的0.34m跌至约0.0027–0.0048m，接触力约26322–32658N。两组第二步局部reset后均无终止。`fresh`不重复回写组两步均0终止，首步base高度约0.3375–0.3400m、base净接触力0。
- `staged`进程内原型把同一存档的创建物性在actor创建阶段施加，prepare之后的load只核验已施加物性；完整恢复同一RNG后，两个零动作策略步均0终止，首步高度约0.3375–0.3400m、base净接触力0。该原型未改变磁盘业务代码，未隐藏预热step，不是正式训练或最终恢复验收。
- 四组原始console及提取JSON位于`knowledge/a07_remote_20261031/恢复首步诊断/`与`knowledge/a07_restore_physics_diagnostic_20261031.json`。不存在对奖励、动作、终止阈值或随机化候选的调整。

## 62. 用户确认Q8后的创建物性恢复实现

2026-10-31，Q8已获用户确认，当前源码事实如下。

- `UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:306–340`的构造方法增加可选`creation_properties`，先核验仅含mass_com_delta[N,5]／friction[N,1]且shape／有限值匹配，然后才进入原生基类创建流程。`:147–150`在原创建随机化收集名义物性后选取存档值，调用已有原生施加／回读路径，早于BaseTask.prepare_sim；新训练传None仍走原流程。
- 同文件`:266–304`的load核验当前创建级实际回读缓存与存档在1e-5绝对／相对容差内一致，仅恢复执行系数、push、计数和RNG，无物性setter、reset或simulate。`UniFP_v3/legged_gym/scripts/train_v3.py:84–90`从已预读显式checkpoint提取两个创建字段传入环境，不把张量写进JSON有效配置。
- `UniFP_v3/tests/test_v3_rewards.py:61–84`新增一个公开load接缝回归，匹配物性恢复可变状态不需native actor，不匹配则拒绝；旧代码明确在此因访问envs触发AttributeError，修正后5项unittest全部通过。纯Torch／文件pytest19项也通过，总计24项。
- `UniFP_v3/tools/check_v3_reward_runtime.py:73–89`不再运行时篡改shape再要求load回写；保留当前创建物性实际回读与执行系数／RNG恢复核对。跨进程创建物性恢复对应实际训练入口调用链。


## 63. 票据05最终远程训练／恢复与回归观测

2026-10-31，最终修复版r06训练10轮、独立r03恢复1轮均退出0，以下为文件和实际运行事实。

- 父文件`ticket05_a_seed0_512x10_20261031_r06/checkpoints/model_000010.pt`记录I=10／S=240／C=122880；子文件`ticket05_a_seed0_512_resume_20261031_r03/checkpoints/model_000011.pt`记录I=11／S=264／C=135168，两次rollout_step=0、last_saved为各自终点，没有failure.json。
- 主Adam25个参数张量、辅助Adam14个参数张量，所有slot完整有限，每个step从200递增至220；主实际lr从3.375e-5到1e-5，辅助保持1e-5。子模型发生真实更新、CUDA RNG前进、新episode执行系数重采；创建级mass_com_delta与friction逐元素完全一致。静态完整validator再次通过。
- 父文件SHA256为de617e00942b9e5e94a5fffa7301b04ec821654ba2ab8d932c377ca5b0a56921，子文件为e9ae516221ac22690c2f0d7b71be17455ca56e61af0d1aa3185d8cc0062bf5ad；发布清单与子lineage父哈希／路径完全一致。1035个实际源码／资产文件与本地manifest一致，运行间源码未变化。
- 恢复首步S=240的物理终止数为0；旧实现为512。r06共129条提前物理失败记录，涉及99环境；r03有74条、涉及74环境，普通短训策略的真实失败保留。短测时长短于20s自然超时，完成episode样本存在截尾偏差，没有由这些样本推断最终成功率。
- r06的10轮value loss从0.0631916到0.0430750，辅助loss从0.0375349到0.0199444；采集82.9339s、更新4.2575s（未计创建／写盘）。Torch峰值283805696B，不含PhysX等全进程显存；没有4096规模结果。
- 修正后的`tools/check_v3_reward_runtime.py`另在远程运行8环境27步，全部合同断言通过，奖励范围[-0.0294281,0.1223764]；push单次触发、末端注入力为零、实际物性回读、可变状态恢复和手动预置的纯timeout记录通过。该回归为CPU张量pipeline／GPU PhysX，不替代512规模GPU张量训练。证据为`a07_runtime_regression_20261031.json`及console。
- 最终摘要在`knowledge/a07_training_smoke_20261031.json`，原始console／metadata／iterations／episodes／发布清单位于`knowledge/a07_remote_20261031/`的r06和r03目录。旧异常摘要及四组因果对照均独立保留。24项本机回归通过；官方`UniFP/`的git status仍为空。
