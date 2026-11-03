# 06 · A阶段评估、阈值与只读归档

状态：A10-Q8/Q9已确认；schema2三组评估和冻结manifest实现完成。本机42项测试通过；r03远程短模型每场景96／96 episode、场景合同和数据汇总完成，均为ungraded，不代表A效果通过。正式seed1配置身份与独立启动门控通过；A8000终点评估尚未执行。

## 目标与范围

建立独立评估入口，使用精确A8000模型、冻结manifest、Actor动作均值及独立场景seed；评估不更新optimizer或训练计数。保存episode策略边界trace、A08物理误差、A05估计误差、失败分母、push恢复结果与可复算图表；归档源码／资产／schema／配置／模型／脚本指纹。

## 代码位置与接口

- 来源：官方play入口与`UniFP_v3/`任务注册／环境；新评估入口与数据工具单独配置。
- 接口：`legged_gym/scripts/evaluate_v3.py --checkpoint ... --manifest ... --output_dir ...`；manifest含A8000边界、环境数、task/reset seed、独立push seed、每环境episode数、动作均值模式、场景协议及冻结阈值版本。
- 源码身份从checkpoint有效config重算；仅完整双Adam/RNG checkpoint可用；A8000迭代、24步/轮、rollout边界和源码／资产／schema／配置身份任一不符即拒绝评估。
- 评估使用独立场景时钟驱动A8000之后的周期push，不改环境内训练迭代/策略step/转移计数；输出episode provenance仍标为A8000 checkpoint边界。
- trace记录输入状态估计标签与step后物理状态，并保留目标、命令、力通道、push、done原因、reward、A08误差和实际随机化样本；episode汇总含回报与回报率。训练期默认关闭逐步trace的CPU拷贝。
- push恢复以事件后完整50个策略step（1秒）误差均值按S2冻结门槛判定，同时保留完整trace；截断窗口作为诊断保留且不通过恢复项。
- 归档复制精确checkpoint、原始trace、episode、有效评估配置、训练日志、summary和图表，记录文件SHA256及git状态，再设目录与文件只读。
- A10-Q9已确认绝对阈值并写入版本化schema2 manifest，覆盖TCP／底盘误差、失败率、六个状态估计MAE维度及S2恢复表现。缺少任一场景／适用指标则拒绝启动或不能输出通过结论。

## 测试与验收

- 单测：已知trace复算mean/P95、状态估计MAE、push恢复窗口、失败分母；检查A8000边界和阈值完整性；检查归档哈希、图表与只读权限。
- 首版历史验证（Q8／Q9前）：本机`tavis`排除需要Isaac Gym的`tests/test_v3_rewards.py`后，24项回归通过；新增评估器单测5项通过；相关Python文件`py_compile`和Git diff空白检查通过。全套pytest因`tavis`未安装Isaac Gym，在收集`tests/test_v3_rewards.py`时报`ModuleNotFoundError`。
- 远程整体验收：当前r03使用真实一轮模型检查三组场景数据合同；没有A8000 checkpoint。正式训练结束后按已冻结manifest运行，任何缺trace／身份／episode均不出通过结论。
- 图表包含训练reward rate和TCP L2在迭代／累计转移双横轴、评估episode误差分布、逐维状态估计绝对误差分布；不重建论文未公开数据点，Fig.3(c)/(d)不属于A。

## Blocked by

- 02-v3-environment-contract。
- 03-actor-critic-and-auxiliary-estimator。
- 04-reward-randomization-and-episode-metrics。
- 05-ppo-runner-progress-checkpoint-stop。

## 2026-11-02补充：场景覆盖缺口

启动前核查发现原入口只有随机运动与push混合协议，不能分别报告A10要求的底盘零速度／运动／扰动恢复结果。Q8／Q9确认后，本补充票据新增三个配置协议和逐组评估；训练任务、step、PPO及A8000严格checkpoint门槛不变。既有r02容量／训练链路证据保留为历史，不因它自动更新新源码身份。

## 2026-11-02 Q8／Q9补充实施切片

状态：实现与r03远程开发验证完成；用户已确认全部设计；Blocked by：01—05与原06已完成、Q8／Q9已确认。

范围：仅修改`v3_evaluation.py`、`evaluate_v3.py`及相关单测，新增`configs/v3_a_initial_v1.json`。schema2校验三组11／11／13项阈值；新增场景配置装配、`summarize_evaluation(..., *, scenario_name)`及三组总体判定；入口顺序构造／销毁环境，分别归档。既有preflight复用同一校验器，不改训练算法、环境step或A8000身份门槛。

单元验收：缺组／缺适用阈值不能通过；S0／S1不依赖push指标，S2无完整50步窗口不能通过；5%／10%失败分母及三组任一不通过均正确。整体验收：使用真实短训checkpoint验证三组配置／trace，明确标注开发检查，正式A8000入口不放宽。最终源码身份改变后既有r02仅保留为历史，需重做对应证据。

执行顺序：manifest校验测试→实现；场景／分组统计测试→实现；三组入口／归档；本机42项全测、评估／门控23项专测、py_compile与diff检查均通过。远程r03三组开发运行及源码身份报告已完成，均为未评级短模型合同验证。不自动启动A8000或B。
