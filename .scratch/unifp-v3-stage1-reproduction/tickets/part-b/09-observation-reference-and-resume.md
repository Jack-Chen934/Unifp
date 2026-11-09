# B-T09 · 评估参考配置修复与12000完整续训

状态：已完成；用户授权继续，实际从12000恢复并更新至12003。

## 范围

只修观察工具的受力评估manifest选择，评估用同协议A8000封存manifest，当前manifest仅提供观察日程。除顶层observation_schedule外任一内容变化必须拒绝。训练核心／identity／资产／PPO／奖励／力幅值／阈值不变。新运行目录从原完整12000恢复至累计40000，保留既有数据与失败报告；不重跑A或自动重试12000评估。

## 源码事实与接口

knowledge/research.md和bt08/check_12000_20261108/检查报告.md已记录失败调用链。tools/train_v3_b_observed.py:observation_commands构造CLI；evaluate_v3.py:127按整manifest hash比对，当前新增观察安排导致冲突。复用现有force_references路径，不加CLI或共享层。该tools文件不进入训练source_manifest，单独固定工具SHA。

## 验证与完成定义

先用真实observer→evaluate.main参考检查的集成测试复现原ValueError，再最小修复；确认仅观察差异放行，阈值变化不放行。观察、门控相关回归通过；新冻结副本与原启动版除工具和测试外业务字节一致，原身份准入仍通过。远程GPU1从12000完整恢复并实际到12001以上，记录实际父SHA、恢复模式、进程／目录与观察安排，再交付。普通奖励下降／物理终止不全局停训；工程错误原样传播。

## Blocked by

- B-T08已完成，12000完整检查点已保存核验。
- 用户本轮继续训练授权。


## 完成记录

窄修工具及真实参考guard集成回归通过，13项相关回归全通过；原核心及资产身份不变，真实preflight通过。GPU1正式同阶段恢复，PID1981990，首轮12001，启动验收到12003、数值有限。父SHA、run路径及机器记录见knowledge/part_b_implementation/bt09/verification.json。下一效果检查16000；正常奖励下降只记录，预算40000。
