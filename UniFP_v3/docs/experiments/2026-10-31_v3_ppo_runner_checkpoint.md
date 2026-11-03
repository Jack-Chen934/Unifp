# V3 票据05：PPO、训练时钟与完整检查点

日期：2026-10-31。依据已确认A05／A07／A09设计实施，只覆盖复现A。官方`UniFP/`保持只读，V3仓库内原B2学习器也保留；新增`v3_rollout_storage.py`、`v3_ppo.py`、`v3_on_policy_runner.py`和`v3_run_identity.py`实现V3专用接口，避免把已确认的V3终止协议强加给B2。

最终状态：票据05已完成。本机24项回归、修复版远程512×10与新进程恢复1轮验收通过；恢复首步异常接触已消除。下文保留实施中失败与确认过程，以末节最终结果为准。

## 实施与官方对照

| 模块 | 保留 | 修改与作用 |
|---|---|---|
| 缓冲区 | 预分配`[T,N,…]`、时间优先展平、全字段同索引、一次排列跨epoch复用、11项minibatch接口 | bool done；detach／copy形成独立存储；采满且returns就绪才更新；不整除明确失败；统计在clone上标段尾 |
| 回报／PPO | 单reward／单V／单GAE、全rollout样本std归一化、14维联合logp与熵、策略／价值裁剪 | 纯timeout按无序terminal IDs映射真实终端后继V，只补一次；失败不补；不评价reset后的V用于旧episode，环境reward不变 |
| 学习参数 | 实际官方配置5epoch×4batch、gamma0.99／lambda0.95、clip0.2、主lr1e−3、KL目标0.01、主clip1、价值1／熵0.01 | 显式使用生效值；KL公式及log内1e−5、每批调度、额外高斯采样调用均保留 |
| 辅助学习 | 每主步后重算encoder／decoder并用四组三维MSE，辅助lr1e−5 | 主参数为encoder／Actor／Critic／std，辅助为encoder／decoder；共享encoder保留两份独立Adam动量；全模型清梯度；辅助不新增clip；日志去掉额外÷64 |
| Runner时钟 | 24步采集→returns→主辅更新 | 全部20组主辅更新成功且清空buffer后统一提交I／S／N×S；reset、日志与内部Adam步不控制完成轮次；发布动作前时钟`(I,S+j)` |
| 检查点 | `torch.save`及state_dict | 完整模型、双Adam／有序参数清单、实际lr、RNG、环境持久物性／push／计数、配置、源码／资产／schema／运行身份和父文件SHA；严格恢复后新episode |
| 写盘／停止 | 更新后保存，间隔200 | 完成量命名、同目录临时文件、文件fsync、CPU回读核验、SHA256、原子发布、目录fsync；终点去重；失败停止；A累计8000后不采下一步、不自动B |

主／辅助loss没有合并；decoder预测不成为Actor额外输入；未新增轮式创新、优化器或奖励。物性／随机化和控制候选参数没有因短测而调整。

## 输出与恢复责任

每个显式新输出目录包含`episode_metadata.json`、`episodes.jsonl`、`iterations.jsonl`、`checkpoints.jsonl`、`tensorboard/`及`checkpoints/model_XXXXXX.pt`。空episode批次只记count0；PPO填好真实timeout bonus后再正常落盘。故障记录保存已完成episode、未消费暂存、完成量、最后确认保存量及保存失败操作，不把半轮参数存为完整续训快照。

同部分严格续训核对实际源码和活动资产字节摘要、schema、N／T／seed／purpose、学习配置及依赖／设备／pipeline。追加预算、输出目录、显示／保存频率不属于学习语义；新进程从显式父文件恢复到新目录，不自动选latest，不覆盖父产物。新episode在物性和双Adam／RNG恢复后重建；不承诺PhysX旧episode或逐比特轨迹连续。A8000的零预算核验分支在创建Gym前校验完整文件；B尚未实现，不能借此入口开启。

## 本地验证

本机不运行仿真训练，`tavis`仅执行纯Torch／文件测试。19项pytest和4项Gym环境下的奖励unittest通过，共23项；其中票据05新增9项，覆盖：

- 手算GAE／returns与done隔离、raw returns先于优势标准化、入库所有权及一次排列完整覆盖。
- 旧V=2、terminal V=3、reset V=100可区分的混合终止／无序ID算例；纯timeout return=3.97，环境reward仍为1。
- 真实网络完成20次主Adam和20次辅助Adam，参数分组／独立动量、无残留梯度、辅助日志无÷64。
- 独立对象恢复双Adam、RNG、新episode后的下一次真实更新；CPU该算例参数逐位相同，不推广为GPU／PhysX承诺。
- 人工fixture的7999→8000恰好24次step、终点只保存一次；该样例不是实际7999轮训练证据。
- 半轮采集失败不提交或保存；文件发布故障保留上一checkpoint，内存完成量与最后持久成功量分别报告；保存不消费RNG。
- 追加预算不改变学习身份，但学习超参数、实际源码与活动资产变化可被检出。

实际命令：

```sh
PYTHONPATH=. conda run -n tavis python -m pytest -q tests/test_v3_runner.py tests/test_v3_ppo.py tests/test_v3_rollout_storage.py tests/test_v3_run_identity.py tests/test_v3_actor_critic.py tests/test_v3_episode_log.py tests/test_desktop_asset.py

LD_LIBRARY_PATH=/home/sorry/.cache/unifp/isaacgym_preview4/py38/lib PYTHONPATH=. conda run -p /home/sorry/.cache/unifp/isaacgym_preview4/py38 python -m unittest discover -s tests -p test_v3_rewards.py -q
```

新增／修改Python文件语法、函数类型标注和docstring存在性核查通过；训练入口在Preview4解释器下`--help`检查通过，没有创建Gym。

## 远程装配与整体测试

远程主机`gpuf3`；新环境`/mnt/data_8THDD/conda_envs/unifp_v3_stage1`从已有依赖克隆，并独立editable绑定`/mnt/data_8THDD/myl_unifp_v3_stage1/UniFP_v3`。原`unifp`的绑定仍为`/mnt/data_8THDD/myl_unifp/UniFP`，未重指或修改其依赖。Isaac Gym使用已有Preview4原生包，新编译缓存独立放在新项目`cache/torch_extensions`。本轮1035个源码／活动资产文件逐个SHA256核对一致，清单位于开发根`.scratch/unifp-v3-stage1-reproduction/knowledge/a07_runtime_source_manifest_20261031.json`。

使用物理GPU1映射逻辑`cuda:0`，A6000，Python3.8.20／Torch2.3.1+cu121；GPU PhysX与GPU张量pipeline，A阶段seed0、512环境、T=24、默认随机化／push及噪声开启。短测通过tmux运行，不占用本机训练、不停止既有服务器任务，不用短测权重作为正式种子初值。

首次运行`ticket05_a_seed0_512x10_20261031_r01`在创建环境时退出1：SSH会话下Python默认使用ASCII，`v3_asset_collision.py:28`读取包含中文来源信息的资产清单时抛`UnicodeDecodeError`，尚未创建Runner或执行训练更新。原console日志保留；后续命令显式设置`PYTHONUTF8=1`，不更改物理／学习配置、不丢弃中文信息。重试名`ticket05_a_seed0_512x10_20261031_r02`，目标10次完整更新；结束状态和跨进程恢复证据完成后回填，当前不宣称整体短测通过或05完成。

### r02接口装配错误

`r02`在环境成功创建后、Runner构造期间退出1：`V3PosForce`没有直接暴露`num_pred_obs`与`num_single_obs`。这两个字段实际位于`env.cfg.env`，而`BaseTask`仅直接暴露总观测宽度。原单测fixture误加了这两个直接属性，掩盖了差异。将fixture改成真实配置位置后，`test_v3_runner.py -x`稳定出现相同`AttributeError`。这属于已确认schema的读取位置错误，修正Runner从既有配置取12／58，不修改环境协议、网络、奖励或训练参数；源码身份改变后使用全新run重做整体短测。

### r03元数据序列化错误

`r03`已创建环境、模型及PPO，首次reset后写metadata时退出1：运行身份直接保存Gym的`SimType`枚举，JSON不支持该类型；尚未完成任何更新，failure.json保留I=0。仅将运行身份中的物理引擎枚举显式转为其字符串表示，保持严格比较与可追溯性；不改变PhysX选择或算法配置。原始CLI启动为该入口集成错误的复现接缝，新run须重新执行10＋1验收。

### 首次10＋1通过与完整性自审

r04完成512环境10轮并保存完整检查点，独立进程`ticket05_a_seed0_512_resume_20261031_r01`严格恢复后完成第11轮，两次退出0。自审发现validator对已经训练的文件仍容许单个Adam参数缺失动量，PyTorch加载后会在下次更新静默重新初始化该参数的动量，不符合已确认完整恢复承诺。在现有checkpoint公开load接缝删除一份辅助Adam slot，现有往返测试可复现未拒载。仅要求I>0时所有注册参数具有step／exp_avg／exp_avg_sq，I=0仍允许尚未初始化Adam。此修正不改变训练数学或有效checkpoint，但改变源码身份，因此最终版本重新运行全新10＋1，不用改写旧身份绕过验证。

### 恢复首步全环境物理终止（待定位）

最终源码r05主run完成10轮，r02恢复run计数／双Adam与父哈希均正确，但恢复第一个策略步S=240时全部512环境物理终止；该run共591条物理失败episode，512条时长仅1步。新训练r05的129条失败来自99个环境，最早第12步，并没有首步全灭。不能把可更新与状态字典一致当成恢复物理行为验收通过，票据05保持实施中。诊断先以同检查点／零动作区分prepare_sim后物性回写、RNG／reset和策略输出，不修改物理参数、关闭随机化或放松终止阈值。

### 根因与修复方向已实证，接口调整待确认

官方Preview4本地API文档`gym_py.html:4901`明确GPU pipeline不支持运行时改变COM。四组512环境零动作对照：

| 组别 | 唯一关键操作 | 首步终止 | 第二步终止 |
|---|---|---:|---:|
| fresh | 正常创建，不重复回写物性 | 0/512 | 0/512 |
| properties | prepare后只重复回写物性 | 512/512 | 0/512 |
| restore | 当前完整load | 512/512 | 0/512 |
| staged | 同一存档物性在创建阶段施加，load只核验 | 0/512 | 0/512 |

异常组均为底盘接触判定，roll／pitch均未超界；首步底盘高度从0.34m落到约0.003m，接触力约26–33kN。移除策略采样和RNG差异后仍能复现，根因在GPU原生物性恢复时机。staged为进程内诊断原型，未改业务源码、未增加隐藏物理步，不把两步诊断当成长期稳定或完整10＋1验收。

修复提案已回写A07补充Q8和A09：显式在环境构造时传入存档创建级物性，创建阶段施加并回读；运行时load仅核验这些物性，再恢复可变执行采样／push／RNG／时钟。因新增环境构造参数并调整load职责，按AGENTS流程等待用户确认后实施。票据05仍未完成；06尚不可执行，正式训练未启动。

### Q8确认与实际修复

用户于2026-10-31确认“同意，修正恢复时序并重测（推荐）”。业务改动仅为：`V3PosForce.__init__`新增可选`creation_properties`，入口从预读的显式checkpoint提取mass_com_delta／friction传入；创建actor阶段应用与逐body／shape回读，发生在prepare之前。`load_training_state`核验已施加的创建级物性，恢复可变执行采样、push、计数、RNG，不再调用原生物性setter。新训练的随机化采样流程不变，没有隐藏预热步。

新增一项load公开接缝回归：不提供任何原生actor时，匹配的创建物性仍能恢复可变状态；不匹配明确拒绝。该测试先在旧代码中因尝试调用物性setter失败，再在修正后通过。本机现为19项pytest＋5项奖励／状态unittest，共24项。既有runtime checker删除运行时篡改物性再load的CPU管线样例，保留物性实际回读及可变状态恢复，跨进程创建物性恢复由本票最终GPU整体测试覆盖。

最终修复版启动`ticket05_a_seed0_512x10_20261031_r06`，随后独立进程恢复run采用`ticket05_a_seed0_512_resume_20261031_r03`。1035个源码／资产哈希再次一致；保留旧异常run及所有因果对照日志，不将旧版本的可更新误写为完整恢复验收。最终状态完成后回填。


## 最终验收结果（2026-10-31）

最终版本的两个run均正常退出0，未出现failure.json。原始console、metadata、逐轮／逐episode和发布记录已回传到开发根`knowledge/a07_remote_20261031/`（完整路径为`.scratch/unifp-v3-stage1-reproduction/knowledge/a07_remote_20261031/`）；完整核验摘要为`knowledge/a07_training_smoke_20261031.json`。模型保留在远程独立runs目录，未覆盖任何父产物。

| 项目 | r06：新训练10轮 | r03：新进程恢复1轮 |
|---|---:|---:|
| 完整迭代I | 10 | 11 |
| 每环境累计策略步S | 240 | 264 |
| 累计转移C | 122880 | 135168 |
| 最后持久成功量 | 10 | 11 |
| 每个主Adam参数的step（25个参数张量） | 200 | 220 |
| 每个辅助Adam参数的step（14个参数张量） | 200 | 220 |
| 实际主lr | 3.375e-5 | 1e-5 |
| 实际辅助lr | 1e-5 | 1e-5 |
| 恢复首步异常终止 | 不适用 | 0/512（旧实现512/512） |

父检查点：
`/mnt/data_8THDD/myl_unifp_v3_stage1/runs/ticket05_a_seed0_512x10_20261031_r06/checkpoints/model_000010.pt`

子检查点：
`/mnt/data_8THDD/myl_unifp_v3_stage1/runs/ticket05_a_seed0_512_resume_20261031_r03/checkpoints/model_000011.pt`

父SHA256：`de617e00942b9e5e94a5fffa7301b04ec821654ba2ab8d932c377ca5b0a56921`；子SHA256：`e9ae516221ac22690c2f0d7b71be17455ca56e61af0d1aa3185d8cc0062bf5ad`。发布清单、文件实际SHA、子lineage中的父SHA／路径均一致。两个文件均约29.5MB，静态完整validator再次通过，双Adam全部slot有限；模型在恢复后实际更新，CUDA RNG继续前进，新episode执行系数重新采样，mass_com_delta／friction与父文件逐元素完全一致。最终源码／资产1035个文件与本地逐文件一致。

10轮中价值loss由0.06319到0.04307，辅助loss由0.03753到0.01994，全部有限；这只表明学习链能更新，不能作为效果复现结论。采集耗时约82.93s、更新约4.26s，未计创建／写盘，设备为共享服务器上的单张A6000；不据此外推4096容量或长训耗时。Torch峰值分配约271MiB，不含PhysX等全进程显存，不能当作总GPU占用。

24项本机回归为19项纯Torch／文件pytest＋5项奖励／状态unittest。额外在远程执行更新后的8环境27步原有奖励／随机化checker，所有合同断言通过，奖励范围约[-0.0294281,0.1223764]；push单次触发、末端力为零、物性实际回读、执行系数／RNG恢复及纯timeout记录均通过，证据为`knowledge/a07_runtime_regression_20261031.json`。其中timeout年龄为手动预置，不能当作自然20s轨迹。官方`UniFP/`仍为干净参考，未改官方资产、PPO或其工作树。

### 保留的验证边界

- r06短测有129条提前物理终止记录，涉及99个环境；r03恢复短测有74条，涉及74个环境，但首步终止为0。普通短训练策略的物理失败如实保留，没有调整奖励、执行器、候选随机化或终止阈值来消除这些记录。
- 每环境4.8s与恢复后的0.48s均短于20s超时上限，完成episode数据存在截尾偏差，不能直接据此计算最终成功率或可靠跟踪曲线；恢复后的新episode也不承诺PhysX轨迹连续。
- 512×10＋1尚未达到400策略步push时钟；实际push功能由上述8环境checker单独验证，不虚称本次PPO短测已遇到push。
- 8000精确停止为边界fixture验证，不是已经训练8000轮。4096容量／性能、候选长期动态稳定性、A10物理阈值及正式启动门控仍由06／07承接。
- 票据05完成只说明训练、保存、恢复和计数链路已验收；未启动正式8000轮、受力B或项目创新阶段。
