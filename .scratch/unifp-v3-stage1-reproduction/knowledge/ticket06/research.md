# 票据06源码事实

核查对象为当前`UniFP_v3/`工作树；行号对应本票据实施后的源码。仿真入口依赖远程V3环境中的Isaac Gym，本机`tavis`未安装Isaac Gym。

## 训练模型身份与边界

- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_on_policy_runner.py:226-265`校验完整训练boundary checkpoint；`:320-358`保存模型、双optimizer、progress、环境状态、RNG、config snapshot和identity，并回读校验。
- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_run_identity.py:27-85`生成运行身份，包含活动Python源码、资产、观测schema、学习配置、运行环境和官方提交指纹。
- `UniFP_v3/legged_gym/scripts/train_v3.py:82-98`要求恢复身份一致；零budget完成核验要求A8000且不创建Gym；`:106-118`创建环境和训练runner。

## V3评估调用链和数据

- `UniFP_v3/legged_gym/scripts/v3_args.py:17-31`定义V3 CLI参数；本票增加显式`--checkpoint`和`--manifest`。
- `UniFP_v3/legged_gym/utils/v3_config.py:23-92`复制V3配置并设置seed、环境数与`run_purpose`。A任务默认配置位于`UniFP_v3/legged_gym/envs/v3/v3_pos_force_config.py:87-177`，包括15维命令、移动目标、零末端力配置及底盘push候选。
- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_actor_critic.py:270-294`的`act_inference`返回动作均值并可输出同一策略调用的辅助估计。
- `UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:410-483`执行策略动作、物理step、奖励、终止、episode统计及reset；step在reset之前已持有terminal physics state。
- `UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:495-543`保存终止episode的步数、时长、类型、回报、奖励分项和误差mean/P95；失败发生在reset前。
- `UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:548-600`的评估trace在reset前复制当前step物理量、任务命令、实际随机化样本、push及终止原因；关闭时不做CPU拷贝。
- `UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:622-634`计算TCP几何L1/L2、底盘平面速度误差和yaw rate误差；`:1228-1254`定义候选TCP世界位置及球坐标转换。
- `UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:1295-1337`构造Actor输入、Critic输入和12维辅助监督标签；base速度乘2、TCP球坐标乘`(0.5,1,1.3)`，力标签乘0.01。

## 本票新增实现

- `UniFP_v3/legged_gym/scripts/evaluate_v3.py`加载显式manifest和checkpoint，按A8000边界校验源/资产/schema/config身份，只实例化Actor，不创建PPO或optimizer；使用Actor均值逐step记录trace，并把task/reset seed与push seed分离。
- `UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py`新增按需policy trace；开关关闭时返回空列表，训练路径不执行trace张量CPU拷贝。
- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_evaluation.py`验证manifest与A8000 boundary、汇总指标/失败分母/push恢复、画训练迭代与累计转移曲线和状态估计逐维误差分布，并复制/哈希/只读封存结果。
- 当前没有A8000 checkpoint；Q9确认的绝对阈值已保存在`UniFP_v3/configs/v3_a_initial_v1.json`。本机不能做Isaac Gym整体评估，短训模型不得代替A8000评估。

## 2026-11-02启动前评估协议核查（Q8／Q9实施前的历史事实）

- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_evaluation.py:38-52`的manifest严格限制字段，且只接受`configured_randomized_moving_targets_with_base_push`一种场景协议；`:53-68`的阈值必须覆盖全部既定指标，每项仅有数值`max`。
- `UniFP_v3/legged_gym/scripts/evaluate_v3.py:63-77`按默认V3配置构造评估环境，没有按静止／运动／push拆分配置的分支。`:100-139`按每环境完成episode数量控制归档；每个episode最长1000策略步，已完成配额的环境不继续纳入trace。
- `UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:1082,1112-1129,1287-1290`在reset及命令重采时调用均匀速度采样。活动配置`v3_pos_force_config.py:95-103`给出x、y、yaw非退化区间，没有保证一组episode三项底盘速度命令都为零的协议。
- `UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:231-249`在独立评估时钟上按配置执行周期push；`:493-496`以姿态／底盘接触判物理失败，以1000步边界判纯timeout。设置零速度命令本身不会固定底盘。
- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_evaluation.py:225-238`阈值比较使用指标的`mean`字段；P95虽已报告，但当前不是同一阈值字段的独立判定对象。
- `.scratch/unifp-v3-stage1-reproduction/design/02-总体参数与配置边界.md:17`已确认正式目标4096环境、训练seed1／2／3、24策略步rollout及A8000预算；并非新增待定的环境数或种子方案。
- `.scratch/unifp-v3-stage1-reproduction/design/part-a/10-评估对照与数据保全.md`已确认评估覆盖底盘静止／运动、移动目标及push前后恢复，但上述当前入口未单独实现静止场景。

## 2026-11-02 Q8／Q9实现后调用关系

- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_evaluation.py:18-27`定义S0/S1/S2及每场景适用阈值集合；`:30-78`只接受schema2协议和三组精确阈值表，S0/S1为11项、S2为13项；`:82-103`只把S0速度命令范围置零并按S2开关push，不固定底盘或改动其他随机化配置。
- 同文件`:142-145`的`summarize_evaluation`按场景单独应用阈值；`:278-303`按每个环境逐一核验episodes_per_env完整性，失败率分母为所有实际完成episode，缺少恢复指标或配额不全不能通过；`:308-320`只有三份场景报告全通过才汇总为通过。
- `UniFP_v3/configs/v3_a_initial_v1.json:1-125`保存用户确认的完整阈值表、32环境×3配额、seed1001／push seed2001及`v3-a-initial-v1`版本。
- `UniFP_v3/legged_gym/scripts/evaluate_v3.py:40-95`先调用既有严格A8000边界及运行身份验证，再依序创建、销毁场景仿真，将每组写入独立只读档案并汇总；`:98-217`创建场景有效配置和Actor，仅执行动作均值推理及环境step，不创建PPO或修改训练计数。
- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_preflight.py:8`继续调用同一schema验证器，故正式启动门控检查的是相同三组冻结manifest。

## 2026-11-02 r03三组开发数据

- `knowledge/ticket07/remote_20261102_r03/evaluation_development/verified_summary.json:2-4`记录开发合同检查通过、开发通过不等于效果通过，以及A8000校验拒绝真实一轮模型。
- 同目录`S0/summary.json`、`S1/summary.json`、`S2/summary.json`均记录96个episode、真实checkpoint迭代1及策略更新0；物理失败分别39／38／73次，TCP L2均值约0.54975／0.53256／0.53299m。S0零底盘命令且无push，S1运动且无push，S2有91次push及58个完整50步窗口；三组结果均为未评级开发数据。
- `knowledge/ticket07/remote_20261102_r03/evaluation_inventory.json`引用三组原始trace远程文件和摘要；本机同步了episode、summary及有效配置，原始约608MB trace仍保存在远程r03隔离目录。
