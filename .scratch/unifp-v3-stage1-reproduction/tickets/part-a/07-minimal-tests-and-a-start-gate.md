# 07 · 最小充分测试与A长训启动门控

状态：r03当前源码／资产及formal seed1配置下，本机42项回归、远程奖励5项、32环境合同、512×10＋续1、4096×1容量及三组场景开发合同均通过；报告、原始输出和身份摘要已核验。独立`preflight_v3`返回`ready=true`、无阻断项。该门控仅表示启动证据完整，不是A效果验收；A8000尚未启动。

## 目标与范围

整合A11关键单测、32环境Gym冒烟、512环境短训及跨进程恢复、正式目标规模容量预检、A启动阻断清单，并在A8000结束后按A10冻结评估协议判断A阶段结果。遵守用户简洁性约束，不穷举边界、不将非关键诊断阻塞训练。

## 代码位置与接口

- 新测试目标：`UniFP_v3/tests/`及独立scripts/preflight入口。
- 串联任务02—06，不引入训练期昂贵同步assert或通用测试框架。
- 启动门控读取源码、资产、配置、设备、A10阈值和checkpoint身份清单。
- 四个阶段都必须引用绝对路径的JSON机器报告及SHA-256；门控校验文件摘要，并要求报告中的阶段结果与证据字段完全一致。哈希绑定用于发现缺件、替换和字段不一致；它本身不证明报告由可信执行器生成，仍须保留实际命令与原始输出供验收核对。

## 测试与验收

- 单测与真实Gym整体测试分开报告。
- 阻断项：关键物理接口、A末端零力、非有限核心状态、reset隔离、PPO计数／停止／恢复、A10阈值和目标规模容量。
- 非关键性能波动／可视化提示记录后处理；A未达标不自动延长、不启B、不改阈值。
- 完成启动前全链路检查，但不在本票据自动启动8000轮训练。
- 已完成的离线单测只验证门控逻辑，不替代真实Gym／远程训练结果。缺少可读、哈希匹配且与字段一致的阶段报告，或实际远程运行结果和目标规模容量报告缺失时必须维持阻断。
- 本轮远程原始输出与计算出的四阶段机器报告保存在`knowledge/ticket07/remote_20261102_r01/`；容量报告的`passed`为false。源码、活动资产、报告及原始日志哈希核验通过。
- 完整记录见`UniFP_v3/docs/experiments/2026-11-02_v3_ticket07_remote_validation.md`。倍率5→32无效；保持倍率5、contact pairs容量2**23→2**25通过最终源码身份重测。原参数失败报告与两个诊断对照均保留；r02四阶段报告、检查点配置和日志摘要可复核。容量配置已按A11-Q6确认落入V3，不改变官方副本。

## Blocked by

- 02-v3-environment-contract。
- 03-actor-critic-and-auxiliary-estimator。
- 04-reward-randomization-and-episode-metrics。
- 05-ppo-runner-progress-checkpoint-stop。
- 06-evaluation-and-immutable-archive。

## 2026-11-02最终身份重测

- 结果：本机31项、远程奖励5项、32环境合同与PPO更新、512环境10轮及跨进程续1轮、4096环境1轮全部通过；A11各阻断性短测通过。
- 4096检查点回读`max_gpu_contact_pairs=33554432 (2**25)`、`default_buffer_size_multiplier=5`；24策略步、98304次环境转移、主辅Adam各20步。原始日志没有`simulation will miss interactions`。
- 4096该轮采样54.57秒、更新0.77秒；PyTorch峰值约1.92GB，不包含PhysX分配，GPU0测试前后均为5786MiB已用。其他GPU不作为本测试资源。
- 单轮共记录545个已结束episode，均标记为physical_failure，涉及543个环境。这是未经长训策略的短预检样本，不能充当学习效果判断；须保留在正式启动评估中处理，容量通过也不代表V3动态稳定或追踪能力通过。
- 结果与文件摘要见`.scratch/unifp-v3-stage1-reproduction/knowledge/ticket07/remote_20261102_r02/`。此轮未启动A8000或B。


## 2026-11-02 r03最终身份结果

- 正式配置SHA-256：`933693b9bf7c3ea9727d1c1eeeea90aee02f6161cc009b60e47ac71106f2d2b3`；源码／活动资产清单分别45／993项，本机重算完整formal身份与远程记录相等。
- 四阶段机器报告、阶段原始日志SHA绑定正确；本机59项产物完整性检查通过。独立formal `preflight_v3`返回`ready=true`、`blockers=[]`。
- 4096容量检查完成1个更新、24策略步／98304转移、主辅Adam各20步，未出现PhysX遗漏交互警告；GPU0验后利用率0%。该轮742个终止episode全部物理失败，只作短测诊断。
- 512恢复迭代10→11，主辅Adam 200→220步，身份／父checkpoint摘要／恢复物性通过；首个恢复策略步物理失败0，但整轮74个episode物理失败。
- 三组真实1轮模型开发数据：S0物理失败39/96、TCP L2 0.54975m；S1 38/96、0.53256m；S2 73/96、0.53299m。S2 91次push、58个完整50步窗口。结果均`ungraded_or_incomplete`，短模型数据不是A8000效果结论。
- 远程证据：`/mnt/data_8THDD/myl_unifp_v3_stage1/validation/ticket07_20261102_r03/`；本机核验件在`knowledge/ticket07/remote_20261102_r03/`。约608MB原始trace保留远程并有逐文件摘要清单。
- seed1正式启动材料已准备，输出目录为`/mnt/data_8THDD/myl_unifp_v3_stage1/runs/a8000_seed1_v3_a_initial_v1`，本轮未启动A8000。
