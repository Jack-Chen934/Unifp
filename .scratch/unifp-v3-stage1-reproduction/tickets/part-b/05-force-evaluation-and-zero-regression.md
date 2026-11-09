# B-T05 · 人工力评估、估力诊断与零力回归

状态：已完成；62项相关测试通过，单环境原生评估／视频链已走通。

## 目标与范围

交付可对A父和B中期／终点checkpoint运行的零更新评估：原A零力S0／S1／S2、14组受力、估力／状态估计曲线和完整失败统计。固定墙评估在B-T06实现。

## 开始时必读

B10-Q1／Q2／Q4、B05-Q3、B11-U6／G2／G5；B04已执行力与待施加力的对齐合同。

## 代码位置与接口

相对于`UniFP_v3/`：

- 新`legged_gym/envs/v3/v3_force_evaluation.py::V3ForceEvaluation(V3PosForce)`仅覆盖事件源／脚本命令与评估episode选择，共用step／reward／reset；训练仍用V3PosForce。
- `legged_gym/scripts/evaluate_v3.py`、`v3_args.py`增加显式`--protocol force_tracking`及必要评估来源／场景配置；默认A路径保留。明确支持原A8000父和合法B完整边界，不移除原A精确8000校验。
- `legged_gym/b2_gym_learn/ppo_cse_pf/v3_evaluation.py`扩纯汇总、绘图与封存；`configs/v3_b_initial_v1.json`记录人工力协议、B10阈值及观察安排。F_max冻结前标明未冻结，不伪造60N验证。`v3_a_initial_v1.json`只读。

14组=静止／移动×半幅／满幅×M／E／C十二组＋R0／R2，各32×3共1344episode；六世界方向均分，确定性事件1／9秒起、rise2／hold1／fall2秒。随机组委托训练事件，不复制。世界命令每策略边界逆yaw表达，报告20ms内yaw残差。

逐样本区分几何／补偿、零／非零／过渡和命令力／人工力；动作前预测与已发生人工力配对。失败占全部配额；方向未定义、延迟截断和无完整push窗口单列，不填0。六类阈值逐项使用B10原表，不另设相对A退化门槛；同时报告相对A差值和同样本恒零预测。source checkpoint与eval scene身份分开。

## 测试与验收

扩展`tests/test_v3_evaluation.py`及受力事件测试，先用手算trace验证分母、阶段、零参照、方向／延迟有效性和各阈值，再实现最小collector／统计。

- 同均值策略、无optimizer更新，不改训练文件／计数。输出summary、原始trace、必要曲线与哈希来源；不存全1856历史或每步模型参数。
- 32×3／20秒／task1001／push2001的原零力合同与门槛不改变，B checkpoint可按同协议回归；A父用新源码结果和旧r03分别保留。
- 小配额开发运行明确标为smoke，不冒充正式全配额；完整A基准与成本由B-T08汇总，零力全回归在B-T07先做。
- 复用现有视频／回放工具生成固定代表和失败样本，不另写一套控制循环；作者原始数据缺失时只比较趋势／量级。

## Blocked by

- B-T04完成。

## 完成记录

2026-11-07：已交付force_tracking与zero_force_regression显式入口、14组manifest、独立脚本子类、力估计与恒零参考／方向／延迟／push统计、固定代表曲线和reset前视频、A8000同协议差值入口。原A默认精确8000校验保留，B不能据smoke或未冻结幅值声明正式通过。62项相关测试通过；本机CPU Torch＋GPU PhysX以明确合成fixture完成1环境×20秒／1000步，498个非零区间、30N峰值、0物理终止，视频及8张图归档。证据`knowledge/part_b_implementation/bt05/`。完整两端与矩阵验收仍由B-T07／08负责；该视频不是正式策略效果。下一项B-T06。
