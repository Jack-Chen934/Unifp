# B-T04 · B完整保存恢复与40000轮精确停止

状态：已完成；56项相关单元／回归通过。

## 目标与范围

完成B→B跨进程恢复合同与精确预算，使真实主辅更新只在完整轮边界提交和保存。使用真实Runner／PPO的张量环境先验证；不新增512原生初始化路径，不伪造正式父状态。

## 开始时必读

B07-Q2／Q3／Q4、B04时序、B11-U5／G4；B-T01—03完成记录。

## 代码位置与接口

相对于`UniFP_v3/`：`legged_gym/b2_gym_learn/ppo_cse_pf/v3_on_policy_runner.py`的`learn`／`validate_checkpoint`／`save`／`load`及RNG；`envs/v3/legged_robot_v3.py`的`get_training_state`／`load_training_state`／`set_training_progress`；`scripts/train_v3.py`的正式剩余预算与开发请求预算。

- B累计I范围8000—40000，S=24I、C=N×S；正式剩余预算40000−I，开发显式增量不超终点。A仍以8000停止，B不使A入口接受B状态。
- 持久化模型、双Adam／实际lr、创建物性、RNG、push时钟和去重、I／S／C、lineage及B事件快照。A→B缺B字段允许仅在明确迁移模式；B→B必需字段缺失不套该豁免。
- 恢复后创建样本不变、RNG最后恢复，仅一次新episode；力从0重新开始，局部执行系数重采。事件快照用于来源诊断，不续演旧半事件、历史或PhysX缓存。
- 每200完整轮、请求终点及40000原子保存；半更新不发布为完成轮。错误退出留诊断，不自动重试／换设备／缩N／启seed2或关机。

## 测试与验收

扩展`tests/test_v3_runner.py`、`test_v3_run_identity.py`和相关环境load测试，真实网络／PPO／双Adam先红后绿。

- 完整小规模测试父I8000→8001，B存档恢复后再更新；slot继续、实际lr／RNG／push不丢，load不写COM。
- I39999→40000恰好24步；完成40000不多采，零预算完成核验不创建Gym。
- 保存异常不覆盖旧文件，不把半轮计数写成完成；A既有恢复／停止回归通过。
- 恢复对比采用相同新episode语义，不要求GPU物理轨迹位级相同。真实4096跨进程证据在B-T08取得。

## Blocked by

- B-T03完成。

## 完成记录

2026-11-07：B累计8000—40000、A上限8000、实际24步完整轮提交已实现；扩展B快照、严格缺字段核验及显式A迁移豁免，保存恢复保留双Adam／实际lr／push／RNG，事件仅诊断并以一次新episode清零。原子保存、半轮异常不发布等原路径沿用。真实CPU学习器A→B→B往返和40000边界通过，结合原A门控／评估共56项通过。CLI完成校验位于Gym构造前已作源码审查。证据`knowledge/part_b_implementation/bt04/`；原生4096跨进程仍由B-T08完成。下一项B-T05。
