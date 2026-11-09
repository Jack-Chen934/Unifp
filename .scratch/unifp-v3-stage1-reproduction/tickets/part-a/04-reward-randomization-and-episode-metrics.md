# 04 · 奖励、随机化与episode统计

状态：已完成（2026-10-31）。A08的23项奖励、A09创建／reset随机化、独立底盘push、训练状态保存／恢复及逐episode统计接口已实现。14项单测及2个真实Gym短测的数据合同断言通过；随机化零动作短测有2个环境触发物理终止，已保留真实失败记录，不据此认定动态稳定性或长训配置冻结。未启动训练，PPO／Runner集成由05承接。

## 目标与范围

落实A08的23项奖励、单次加权、零力目标退化、物理误差及episode记录；落实A09确认的A/B身份、末端零力／底盘push区分、训练时钟和随机化候选生命周期。参数候选支持在正式配置冻结前依据训练证据版本化调整。

## 代码位置与接口

- V3奖励与环境统计目标：`UniFP_v3/legged_gym/envs/v3/`。
- 训练状态对接A07：`get_training_state`／`load_training_state`包含随机化样本、push／课程状态和RNG；字段与A07一致。
- 日志按Q5b3逐episode记录并有界flush；文件格式由本任务实施时按依赖和运行负担选定。

## 测试与验收

- 单测：23项各加一次、环境reward与PPO timeout补偿隔离、真实时长rate、代表性mean/P95、记录无漏重。
- 验收：误差来自物理状态而非reward反推；reset只清对应环境；无样本不补零；训练期不开全物理步trace。

## Blocked by

- 02-v3-environment-contract。

## 实施记录与验证

- 环境：`legged_robot_v3.py`注册固定23项并保持原始权重与单次dt加权；旧命令／新状态先结算奖励和物理误差，再推进命令、捕获终端Critic及局部reset。保留负总reward，控制硬限位与奖励软限位分开。
- 随机化：创建时实际回读摩擦、base／臂端质量和base COM；质量变化同比缩放中央惯量；14维执行系数局部reset采样、在动作目标硬裁剪前使用。push按Runner发布时钟同步触发，每环境独立采速度，用途标签不隐改物理，A力通道保持零。
- episode：真实样本mean/P95与时长，逐episode记录、等权汇总；只清done行；先保存完成记录再执行可能失败的目标采样／reset。`v3_episode_log.py`提供批量JSONL和不可覆盖metadata，空batch不补零。
- 状态：`get_training_state`／`load_training_state`覆盖实际物性、执行系数、push、I/S/N×S和各RNG，不是PhysX全状态快照；恢复后由05启动新episode。
- 验证：模型／日志／资产10项，奖励4项，8环境27步随机化短测，32环境25步名义合同短测均通过相应断言；类型标注／docstring、语法和diff检查通过。详细命令、官方对照及失败证据见[实验记录](../../../../UniFP_v3/docs/experiments/v3/实验汇总.md)。

## 05衔接与剩余验收

- 05在每次动作前发布`set_training_progress(I,S+j)`，完成主／辅更新后提交完整迭代；episode记录时钟当前是动作前进度。
- `infos`提供`terminated`、`time_outs`、`terminal_env_ids`、`terminal_privileged_obs`、`episode_records`及`base_push_applied`。纯timeout只补一次真实终端V，补偿只影响PPO奖励副本，不污染环境reward和物理指标。
- 环境record的`timeout_bootstrap_bonus=None`由05求值后填入；正常JSONL拒绝未解析值，故障诊断可保留null。05每迭代flush并管理`pending_episode_records`消费，故障时保留已完成记录后传播原错误。
- metadata的源码／资产／配置身份、raw/effective权重、dt及单位由05传入；05串接双优化器checkpoint、RNG恢复及A8000精确停止。
- 2个随机化零动作环境的接触物理终止、候选参数稳定性和目标规模GPU容量仍须A11验证；本票完成不等于长训启动门控通过。
