# 票据07源码事实

核查对象为当前`UniFP_v3/`工作树。以下路径均相对开发根`/home/sorry/python_ws/interactive/unifp_reproduction/`；行号对应本记录生成时的源码。

## 训练入口与配置

- `UniFP_v3/legged_gym/scripts/train_v3.py:39`解析显式运行身份、复制配置并验证V3 A任务及仿真／学习设备一致；`:51`收集运行环境，`:63`生成源码／资产／配置身份，`:65`在恢复前比较身份；`:69`对零迭代请求执行A8000 checkpoint核验后直接返回，不创建Gym；`:82`在formal用途执行启动门控；`:99`构造环境，`:102`创建Runner并执行明确预算。
- `UniFP_v3/legged_gym/scripts/v3_args.py:25`定义任务部分、运行用途、seed、环境数、最大迭代、显式checkpoint、manifest、启动证据、输出目录和设备等CLI字段。
- `UniFP_v3/legged_gym/utils/v3_config.py:55`规定运行用途与A/B身份，`:63`校验seed，`:67`深拷贝task配置，`:73`覆盖环境数，`:79`处理迭代预算。
- `UniFP_v3/legged_gym/envs/v3/v3_pos_force_config.py:13`给出默认32环境、14动作、32×58 actor历史、3×98特权观测和20秒episode；`:105`给出末端候选目标域，`:159`给出候选随机化／push，`:179`开启观测噪声。

## PPO计数与checkpoint

- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_on_policy_runner.py:64-90`验证请求预算不会超A8000，并规定零预算仅接受已保存完成量8000的边界。
- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_on_policy_runner.py:107-160`每轮采集24策略步、完成一次更新后提交迭代／策略步／转移计数，并按间隔或本次目标保存。
- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_on_policy_runner.py:226`校验checkpoint格式、身份、完整更新计数、环境状态、模型、双Adam及RNG结构。
- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_on_policy_runner.py:299`将完整训练边界状态写入新临时文件、回读校验、哈希并原子发布。
- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_on_policy_runner.py:365`显式加载checkpoint，恢复模型、双优化器、进度、环境可变状态及随机源，并启动新episode。

## 测试与评估

- `UniFP_v3/tests/test_v3_runner.py:130`覆盖双Adam／RNG恢复后继续更新；`:191`为7999到8000精确停止的边界fixture。
- `UniFP_v3/tests/test_v3_runner.py:217`覆盖部分rollout失败不提交计数；`:237`覆盖checkpoint发布失败保留先前文件。
- `UniFP_v3/tests/test_v3_evaluation.py:28-189`覆盖A8000边界、评估指标重算、缺失或不完整阈值、归档哈希及只读权限。
- `UniFP_v3/legged_gym/scripts/evaluate_v3.py:43-67`要求显式checkpoint和manifest、检查用途、seed及环境数，并核对A8000身份；`:72-95`依序采集、归档三组场景及汇总；`:98-217`的`evaluate_scenario`配置单场景、创建与销毁仿真、使用Actor均值采集数据，不创建optimizer。
- `.scratch/unifp-v3-stage1-reproduction/design/part-a/11-单元测试整体测试与验收.md:37`记录已确认的关键单测类别、32环境冒烟、512环境10轮与新进程恢复、正式规模预检、长训阻断项及A8000失败处理。
- `.scratch/unifp-v3-stage1-reproduction/tickets/part-a/06-evaluation-and-immutable-archive.md:14-22`记录push后50策略步（1秒）完整窗口、失败分母和归档接口；Q9阈值已保存到`configs/v3_a_initial_v1.json`；当前没有A8000检查点。

## 早期远程证据边界（2026-11-02 r03前）

- `UniFP_v3/docs/experiments/v3/实验汇总.md:99-130`记录512×10及跨进程续1轮已通过、32环境旧合同测试与8环境随机化短测结果；同一记录说明恢复首步物理问题已修正，但普通短训仍有提前接触终止，512测未触发400策略步push，不能外推4096容量或长训效果。
- `UniFP_v3/docs/experiments/v3/实验汇总.md:17-20`记录纯Torch测试通过，但本机`tavis`无Isaac Gym；没有A8000 checkpoint、A11量化阈值和真实A评估结果。
- `UniFP_v3/tests/`当前包含资产、Actor/Critic、episode日志、评估、PPO、奖励、rollout storage、运行身份和Runner测试；独立真实Gym工具位于`UniFP_v3/tools/check_v3_desktop_runtime.py`及`check_v3_reward_runtime.py`。

## 当前票据07门控实现

- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_preflight.py:15`的`validate_stage_artifacts`读取四阶段绝对路径JSON报告，核验SHA-256并比较报告`result`与对应证据字段；`:61`的`assess_a_start_gate`接收总证据、当前身份、评估manifest和正式环境数，返回`ready`／阻断原因。纯Torch单测身份不比较runtime，真实Gym／恢复／容量则比较。两接口都不执行仿真。
- `UniFP_v3/legged_gym/scripts/preflight_v3.py:48`在输出前合并报告文件校验阻断项；`:52`以排他创建模式写新报告。runtime文件是调用方记录，离线CLI不会探测远程GPU，也不证明报告由可信执行器产生。
- `UniFP_v3/legged_gym/scripts/train_v3.py:82-93`在formal用途检查报告文件并按当次实际身份运行启动门控，失败时在Gym创建前终止；development／preflight用途保留原有短测预算流程。
- `UniFP_v3/tests/test_v3_preflight.py`覆盖完整证据、缺项及阈值、单项报告过期、正式规模不匹配、非零人工末端外力、报告哈希与内容绑定，以及CLI退出／不覆盖旧报告。通过fixture是合成输入，不是实际远程测试报告。
- 本机缓存`/home/sorry/.cache/unifp/isaacgym_preview4/py38/`可导入Gym并执行`tests/test_v3_rewards.py`的5项CPU张量测试；该测试使用`V3PosForce.__new__`构建fixture，不创建sim或推进仿真。

## 2026-11-02远程原始证据（r01历史）

- `knowledge/ticket07/remote_20261102_r01/verified_summary.json`保存32环境名义合同、8环境随机化与push、32环境GPU更新、512×10及新进程续1轮、4096候选容量的独立结果。检查点源码／活动资产指纹与当前本机文件一致，阶段报告及原始日志哈希一致，见同目录`local_integrity.json`。
- 同目录`reward_unit.console.log`记录标准库unittest执行5项奖励张量测试通过；`unit_cpu.console.log`记录本机31项pytest测试通过。
- 同目录`capacity4096.console.log:22`包含PhysX的`foundLostAggregatePairsCapacity`警告，要求容量22993765，并明确说明会遗漏交互；训练入口仍返回0且生成检查点。
- `UniFP_v3/legged_gym/envs/base/legged_robot_config.py:179-180`与原官方配置一致，分别为`max_gpu_contact_pairs=2**23`及`default_buffer_size_multiplier=5`；V3配置继承这两个字段。
- `UniFP_v3/legged_gym/envs/base/legged_robot.py:576`创建actor时使用环境索引`i`作为collision group；`UniFP_v3/legged_gym/utils/helpers.py:60`从CLI的`subscenes`设置PhysX子场景数。
- 独立容量诊断进程中把倍率改为32后，有效配置确认为32，但同一PhysX警告仍出现；业务源码未改。对应原始日志为远程同一验收目录中的`evidence/capacity4096_buffer32.console.log`。
- 第二个对照保持倍率5、仅将`max_gpu_contact_pairs`改为33554432（2**25）后，4096环境完成24策略步及1次更新，原遗漏交互警告消失；模型、计数及双Adam均通过回读核验。有效物理配置、两个对照的逐项检查和运行时间保存在本知识库`remote_20261102_r01/capacity_probes_summary.json`，三个实验源码／活动资产指纹相同。

## 2026-11-02容量配置落盘后的源码事实

- `UniFP_v3/legged_gym/envs/v3/v3_pos_force_config.py:175-181`在V3自己的`sim.physx`子类中将`max_gpu_contact_pairs`覆盖为33554432；该类继续继承`LeggedRobotCfg.sim.physx`其他字段。
- `UniFP_v3/legged_gym/envs/base/legged_robot_config.py:179-180`及官方副本对应行仍为`2**23`和倍率5。相对r01检查点源码清单，内容变更仅为上述V3配置文件，活动资产摘要全部不变。
- `knowledge/ticket07/remote_20261102_r02/unit_cpu.console.log`记录本机最终源码31项单测通过。最终版本远程快照位于`/mnt/data_8THDD/myl_unifp_v3_stage1/validation/ticket07_20261102_r02/UniFP_v3/`。

## 2026-11-02 Q8／Q9实施后的r03当前事实

- `UniFP_v3/configs/v3_a_initial_v1.json:1`是schema2单文件协议，包含S0／S1／S2阈值、每组32环境×3 episode、评估seed1001和push seed2001。当前接口依据见`knowledge/ticket06/research.md`的Q8／Q9实现后调用关系。
- `knowledge/ticket07/remote_20261102_r03/evidence/unit_cpu.console.log:3`记录42项本机pytest通过；同目录`reward_unit.console.log`记录5项远程奖励张量unittest通过。
- `knowledge/ticket07/remote_20261102_r03/evidence/verified_summary.json:5482`记录512恢复首策略步物理失败为0；`:5484-5485`记录短测与启动证据完整，`:10172`记录三组开发合同检查通过。该文件由r03实际日志和checkpoint回读产生。
- `knowledge/ticket07/remote_20261102_r03/start_gate_cli.json:2`为独立CLI返回的`ready=true`，`:3`为无阻断项；输入formal配置摘要为`933693b9bf7c3ea9727d1c1eeeea90aee02f6161cc009b60e47ac71106f2d2b3`。
- `knowledge/ticket07/remote_20261102_r03/evaluation_development/verified_summary.json:4`记录A8000强校验拒绝真实一轮checkpoint；同目录S0／S1／S2的`summary.json`分别记录96个episode，物理失败39／38／73个，TCP L2均值约0.54975／0.53256／0.53299m，三组均为`ungraded_or_incomplete`。S2保留91次push和58个完整50策略步窗口。
- `knowledge/ticket07/remote_20261102_r03/local_integrity.json:2`记录59项回传产物与身份检查通过；原始约608MB trace字节保留在远程，`evaluation_inventory.json`保存远程逐文件摘要和大小，未在本机回读trace字节。
