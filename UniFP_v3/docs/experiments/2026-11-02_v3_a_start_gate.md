# V3 A阶段长训启动门控记录

日期：2026-11-02

## 目的

实现票据07的离线启动证据检查，并明确离线逻辑测试与远程仿真验收的边界。本记录不启动A8000训练。

## 实施

- 新增`legged_gym/b2_gym_learn/ppo_cse_pf/v3_preflight.py`，比较活动源码／资产／schema／运行身份、正式配置摘要、32环境Gym冒烟、512环境10轮与续训1轮、以及正式环境数容量报告；检查A10评估manifest和阈值是否存在。
- 新增`legged_gym/scripts/preflight_v3.py`，根据当前工作区和显式配置重算身份，输出采用排他创建模式，避免覆盖既有门控报告。
- `train_v3.py`的formal用途在创建Gym前再次按当次运行身份检查证据；缺少证据、身份过期或阈值未确认时停止。开发／预检用途保持原有流程。
- 四阶段报告现在都需带绝对路径和SHA-256；离线CLI与formal入口读取报告文件，校验摘要，并逐字段比较报告结果与证据对象。缺少报告、报告被改写或证据字段与报告内容不一致都会阻断。
- 新增`tests/test_v3_preflight.py`，覆盖完整合成证据、缺项、过期阶段身份、正式环境数不匹配、非零人工末端外力以及CLI不覆盖既有报告等情况。
- 新增报告绑定测试，验证缺少报告被阻断，且修改阶段字段后会因报告内容不匹配被阻断。

## 本机验证

在`tavis`环境、`UniFP_v3/`目录运行：

```text
python -m pytest tests -q --ignore=tests/test_v3_rewards.py
```

结果：`python -m pytest tests -q --ignore=tests/test_v3_rewards.py`为31 passed；`compileall`通过。`tests/test_v3_rewards.py`因当前Isaac Gym Python环境没有pytest、`tavis`又不含Isaac Gym，未在本轮运行。

## 尚未验收

- 最终源码身份的31项本机单测、5项远程奖励单测、32环境Gym／GPU更新、512×10＋新进程续1轮及4096×1容量预检均已完成；4096检查点回读contact pairs为2**25、倍率为5，原遗漏交互警告消失。报告、原始日志、源码／资产身份均核验通过。
- A10-Q9绝对阈值已在`configs/v3_a_initial_v1.json`冻结；当前没有A8000 checkpoint，因此没有A阶段效果结论。
- 2026-11-02晚远程连接恢复，复用主连接PID 78331存活。本轮源码已同步到独立`validation/ticket07_20261102_r01/UniFP_v3/`并通过内容摘要核验；历史开发目录未覆盖。原始输出、阶段机器报告和参数对照见[远程验收记录](2026-11-02_v3_ticket07_remote_validation.md)。
- 通过测试中的证据对象是合成fixture，只证明门控逻辑及文件绑定行为。SHA-256只能证明读取的文件与引用摘要一致，不能证明报告由可信执行器生成；真实验收仍须保留并核对实际命令及原始输出，不可把人工填写的`passed: true`当作测试事实。

结论：票据07门控及其短测链路已经完成，包括4096环境容量预检。用户确认的V3专用容量覆盖已落盘，最终源码身份证据通过。4096一轮产生545个已结束episode且全为physical_failure；这不能用来判断策略学习效果，但应在A长训效果评估中保留与分析。该句为r03前状态；r03身份和启动门控已通过，但当前仍无A8000效果结论。

## 远程目录与历史训练核验

- `/mnt/data_8THDD/myl_unifp/UniFP/`是官方复现目录。其力控制追加训练目录已有`model_16000.pt`，`status.json`记为`complete`，完成时间为2026-10-30 07:35（北京时间）。这不是V3阶段A训练结果。
- `/mnt/data_8THDD/myl_unifp_v3_stage1/`是独立V3目录；当前保存的是512环境10轮及新进程续1轮的票据05记录。本次核查没有发现该目录下的A8000运行记录。
- 本地与远程活动桌面URDF的SHA-256均为`1dc2f47cf62baf707d9f1e7595b9df6b4455193feabedbd0b46e940ddffb9c0d`。
- 远程V3环境解释器为`/mnt/data_8THDD/conda_envs/unifp_v3_stage1/bin/python`，Torch为`2.3.1+cu121`，可识别四张GPU。本轮只读检查没有启动或停止训练。

## 最终身份容量修正复测（r02）

- 用户确认后，仅在`V3PosForceCfg.sim.physx`将`max_gpu_contact_pairs`覆盖为`2**25`；官方配置与`default_buffer_size_multiplier=5`保持不变。
- 本机31项单测，远程奖励5项单测、32环境Gym／GPU更新、512环境10轮和新进程续1轮、4096环境1轮均通过；双优化器、训练计数、checkpoint谱系及恢复物性核验通过。
- 4096有效容量为33554432，迭代完成24策略步／98304转移，无PhysX遗漏交互警告。采样54.57秒、更新0.77秒；PyTorch峰值约1.92GB不包含PhysX。GPU0运行前后各有5786MiB占用；其他GPU没有用于本次验收。
- 4096轮记录的545个终止episode全部为physical_failure，涉及543个环境。该统计是短时未训练策略数据，不能视为整体失败率或效果结论，但说明当前模型的动态表现仍须严格保留在A阶段评估中。容量检查通过只消除了已观察到的接触配对容量风险。
- 本机证据在`.scratch/unifp-v3-stage1-reproduction/knowledge/ticket07/remote_20261102_r02/`，远程隔离快照在`/mnt/data_8THDD/myl_unifp_v3_stage1/validation/ticket07_20261102_r02/`。r02时正式身份及Q9阈值尚未冻结；此为历史证据，不能代替r03。


## r03后续收束

2026-11-02在Q8/Q9确认后，r03完成最终身份重验：四份票据07阶段报告和实际原始日志摘要通过；capacity绑定正式seed1配置摘要；本机源码、活动资产及完整formal身份与远程相等。三组一轮模型开发评估均完成96个episode，但仅为ungraded合同检查：S0／S1／S2物理失败分别为39／96、38／96、73／96，TCP L2分别0.54975／0.53256／0.53299m。S2记录91次push和58个完整1秒窗口。独立preflight CLI `ready=true`、无阻断项；本机59项回传产物完整性检查通过。

4096容量更新完成且无PhysX遗漏交互警告；采样52.67秒、更新0.76秒、PyTorch峰值约1.92GB（不含PhysX）。该轮结束742个episode全为physical_failure，只作短测诊断。512恢复的身份、计数、Adam和物性核验通过，首个续训step物理失败0，但整轮仍有74个物理失败episode。门控就绪不等于策略效果达标。seed1启动材料在`.scratch/unifp-v3-stage1-reproduction/knowledge/ticket07/`；A8000尚未启动。
