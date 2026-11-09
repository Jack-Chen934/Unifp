# B-T01 · A8000到B的完整学习状态迁移

状态：已完成；2026-11-07实施、CPU张量验证与代码审查通过，后继B-T02可执行。真实4096迁移按计划归B-T08，未启动B训练。

## 目标与范围

交付显式A→B迁移的最小可验证纵向切片：读取明确父文件，验证阶段允许差异，创建时恢复物性，恢复模型／双Adam／实际学习率／计数／push／RNG，开启一次新B episode。先用真实学习器与张量环境验证；真实4096迁移由B-T08统一验收。本票不宣称非零力任务已完成。

## 开始时必读

- `knowledge/PRD.md`、`knowledge/research.md`末尾B源码核对；B07-Q1／Q2及B11-U5。
- `development-plan.md`第七节的父路径、SHA与V6身份；不重新选择资产或训练父。

## 代码位置与接口

以下代码路径均相对于`UniFP_v3/`：

- `legged_gym/scripts/train_v3.py`、`v3_args.py`：保留明确`--reproduction_part B`、`--resume`父文件及独立输出目录；A入口行为保留。A→B与B→B恢复模式明确区分，不支持B无父文件从零正式启动。
- `legged_gym/utils/v3_config.py`、`envs/v3/v3_pos_force_config.py`：按A／B解析隔离配置及预算；已有A分支不被B参数覆盖。B课程数值在B-T02落地。
- `legged_gym/b2_gym_learn/ppo_cse_pf/v3_run_identity.py`：在已有identity基础上校验明确阶段迁移差异，生成父SHA、来源／配置差异与lineage；不关闭全部identity比较，不引入通用迁移框架。
- 同目录`v3_on_policy_runner.py`：`load(path, *, transition_from_a=False)`表达显式迁移，默认仍为同阶段恢复；拆分必要的校验与恢复职责，支持累计8000起点，真实保存恢复细节交B-T04。
- `envs/v3/legged_robot_v3.py`：沿用`creation_properties`，迁移时明确映射A持久状态到B；`load_training_state`不写prepare后的质量／COM。B力事件初始状态由B-T02补齐。

创建物性`mass_com_delta[N,5]`、`friction[N,1]`在prepare之前施加；模型1856／294／12／14与64 latent不变。构造和加载完成后最后恢复RNG，再一次reset；不调用simulate，不因learn入口再reset。正式N4096；父I8000／S192000／C786432000保持，不切片父状态。

## 测试与验收

先扩展`tests/test_v3_run_identity.py`、`test_v3_runner.py`和既有物性load测试，确认当前A-only限制／迁移缺失导致预期失败，再最小实现。

- 用真实模型／双Adam的完整小规模张量父fixture核对参数、slot、实际lr、创建调用顺序、RNG与一次reset；fixture计数明确是测试，不修改正式检查点。
- 已批准的A→B差异可通过；资产／schema／动作映射改变、缺Adam slot、非A8000父应明确拒绝。B→B不得继承A→B缺字段豁免。
- A同阶段恢复、A8000完成态不采样仍通过既有回归。父文件字节不变，失败不覆盖既有输出。
- 保留实际测试输出与本票代码／设计一致性审查记录；未经B-T08不声称真实迁移验收完成。

## Blocked by

- 用户确认本组正式实施计划。
- A8000父证据已具备；实现前核对归档事实，真实访问时重新核验SHA。

## 完成记录

- 实现：`v3_run_identity.py::validate_a_to_b_transition`逐字段核对配置、来源及runtime差异；`V3OnPolicyRunner.load(..., transition_from_a=True)`完整恢复学习／持久状态，阶段字段显式映射后一次reset。既有创建物性／环境load路径无需修改，仍在prepare前施加、load只核验。
- 入口：`train_v3.py`＋`v3_args.py`新增显式`--transition_from_a --parent_sha256`，摘要错误在反序列化／创建Gym之前失败；B需要明确父文件，正式B入口保留待B-T08完成的限制。`v3_config.py`为B独立标记实验名称，A配置不变。
- 验证：真实模型／双Adam的张量父fixture、I8000边界、实际lr、三类RNG reset样本、物性／push透传、父字节不变、错误边界／缺Adam／N不一致拒绝；原A8000完成态零reset零采样。
- 总计50项相关pytest回归通过；Gym仅导入、未create_sim。完整日志与源码哈希见`knowledge/part_b_implementation/bt01/unit.log`及`verification.json`。
- 离线实际配置：归档t04 A身份／配置与当前真实B配置核对通过，报告`archived_identity_check.json`；runtime是归档比较输入，不冒称本机／远程原生验证。
- TDD记录：先观察身份函数缺失、Runner A-only及CLI缺迁移参数的需求失败，再最小实现通过。首次CLI检查还遇到Ninja不在PATH，使用既有py38/bin解决；没有把环境失败计为需求失败。
- 审查：未改资产、schema、网络／PPO算法或官方代码；同阶段identity相等仍保留；没有新增逐step验证、只载权重初始化或通用迁移框架。本票无未解决关键问题。
- 后续边界：B-T02补非零力生命周期并逐字段扩展已确认课程白名单；B-T04补B完整预算／保存恢复；B-T08核对远程实际父SHA、4096创建物性与跨进程10＋1轮。本票不宣称这些已经完成。
