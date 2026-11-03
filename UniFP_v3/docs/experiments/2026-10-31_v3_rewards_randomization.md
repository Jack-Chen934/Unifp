# V3 奖励、随机化与 episode 统计实施记录

日期：2026-10-31。对应 A 阶段票据04、已确认 A08/A09 设计。本次完成环境奖励、物性／动作目标随机化及日志接口；未启动训练，未修改官方 `UniFP/` 副本或桌面名义资产。

## 交付内容与官方对照

| 模块 | 保留的官方机制 | V3 适配与明确边界 |
|---|---|---|
| 主奖励 | TCP 指数跟踪、底盘平面速度与 yaw 跟踪、alive；原始权重乘策略周期 | 固定23项，按转向／驱动／机械臂分组；删除四足特有项；驱动轮没有绝对转速惩罚；保留负总奖励 |
| 目标补偿 | 几何目标加世界坐标人工外力／命令力之和除虚拟刚度 | `K=[200,200,200] N/m`；A 的两种末端力严格为零，补偿目标与几何目标相同；接触净力和底盘 push 不写入人工力标签 |
| 控制边界 | 显式 PD 和 URDF effort 硬限幅 | 恢复 URDF 硬位置限位供控制使用，另存0.95软位置界供奖励使用，避免基类将软界误用成控制硬界 |
| 摩擦 | 创建时64桶采样、每环境一个系数 | 覆盖机器人全部接触 shape 并回读；特权观测保存实际值 |
| 质量／COM | 创建时固定采样、episode 内保持 | 按 `base_link`／`arm_seg6` 名称定位；质量按名义值比例扰动，中央惯量同比缩放；局部 COM 平移不改变中央惯量；禁用惯量重算并回读核对 |
| 执行系数 | reset 时按环境／执行器独立采样 | 适配14维；缩放动作目标残差／轮速目标，再进行硬裁剪；不是力矩或电机增益缩放 |
| 底盘 push | 随机改写世界 XY 速度 | 从 Runner 发布的训练策略步钟读取；每400策略步同步触发，每环境独立采速度，同一时钟最多一次；用途标签不隐式改变物理 |
| episode 统计 | 基于本次真实转移统计 | 先按旧命令和新物理状态结算，再推进命令、截取终端 Critic、局部 reset；逐 episode 独立记录并等权汇总 |

奖励配置保留原始权重，另派生一次 `weight × dt`，重复注册不会重复乘周期。每项只计算一次，分项和总值出现非有限值时明确失败。TCP 误差直接来自物理位置，速度误差直接来自状态与对应命令，均不从 reward 反推。

## 首版候选参数

下列均来自已确认设计，仍为验证候选，不是长训冻结配置。

- 主项：TCP／底盘平移／yaw／alive 权重分别为 `2 / 2 / 1 / 1.5`；TCP sigma 为1，速度 sigma 为0.25。
- 力矩平方：三组均 `−5e−6`；动作变化：转向／驱动 `−0.02`、臂 `−0.045`。
- 速度平方：转向 `−8e−4`、臂 `−2e−4`；加速度平方：转向／驱动 `−2.5e−7`、臂 `−4.5e−7`。
- 软 effort 越界：三组均 `−0.005`，阈值为 URDF effort 的0.90；软位置越界 `−10`，软区间为硬区间宽度的0.95。
- 稳定／接触：base z 速度 `−1.5`、xy 角速度 `−0.02`、高度误差 `−2`、非轮 body 接触计数 `−5`；接触计数阈值0.1N，19个非轮 body，四轮豁免。
- 摩擦 `[0.3,2.0]`；base／臂端质量比例 `[0.9,1.1]`；base 局部 COM 各轴 `±0.02m`；执行系数 `[0.85,1.15]`。
- push 间隔8s，世界 XY 各速度分量 `±0.8m/s`；策略周期20ms，每次动作执行4个5ms物理子步。

## 统计、状态与票据05接缝

环境为每个完成 episode 输出真实步数／时长、环境回报／每秒回报、23项 raw 均值及物理误差 mean/P95。误差样本为 `[N,1000,4] float32`，4096环境约62.5MiB；A 的几何／补偿 TCP 指标复用样本，不保存全物理子步 trace。局部 reset 只清对应环境计数，终止记录在可能失败的命令采样／reset 之前保存到 `pending_episode_records`。

`get_training_state`／`load_training_state`覆盖实际物性、执行系数、push、三种训练计数及 Python／NumPy／Torch／CUDA RNG。快照独立存于 CPU；恢复不执行物理 step 或 reset。恢复后由票据05按已确认流程开启新 episode，不承诺恢复 PhysX 全状态。

`v3_episode_log.py`提供每迭代一次批量 append／flush／fsync 的 JSONL 接口和独占创建的 metadata 文件。正常记录须由 PPO 填好真实 `timeout_bootstrap_bonus`；环境层保留 `None`，不伪造0。空批次仅返回 episode 数0，不补旧均值或零误差；各完成 episode 等权汇总，平均 episode P95 不等同于全部样本合并后的 P95。

票据05仍需接入双优化器、24步 rollout、纯 timeout 的真实终端 V 补偿、JSONL／TensorBoard、完整 checkpoint 与8000次完整更新后的精确停止。环境中的 episode 时钟为动作前 Runner 已发布进度；日志中须清楚区分这一采集时钟与完整更新提交边界。`pending_episode_records`正常消费后应清空，故障时先保存诊断再传播原错误。

## 已执行验证

以下命令工作目录均为 `UniFP_v3/`：

```sh
PYTHONPATH=. conda run -n tavis python -m pytest -q tests/test_v3_actor_critic.py tests/test_v3_episode_log.py tests/test_desktop_asset.py

LD_LIBRARY_PATH=/home/sorry/.cache/unifp/isaacgym_preview4/py38/lib PYTHONPATH=. conda run -p /home/sorry/.cache/unifp/isaacgym_preview4/py38 python -m unittest discover -s tests -p test_v3_rewards.py -q

LD_LIBRARY_PATH=/home/sorry/.cache/unifp/isaacgym_preview4/py38/lib PYTHONPATH=. conda run -p /home/sorry/.cache/unifp/isaacgym_preview4/py38 python tools/check_v3_reward_runtime.py

LD_LIBRARY_PATH=/home/sorry/.cache/unifp/isaacgym_preview4/py38/lib PYTHONPATH=. conda run -p /home/sorry/.cache/unifp/isaacgym_preview4/py38 python tools/check_v3_desktop_runtime.py
```

- 第一组10项通过：模型3项、episode日志2项、桌面资产5项；第二组奖励4项通过。共14项，不等于14项均为本票新增。
- 奖励单测覆盖：理想转移0.13、两个计数接触后−0.07且不截负；两样本误差 mean=0.05m、P95=0.095m、实际时长0.04s；匀速驱动轮豁免、20ms加速度、软／硬位置界分离；系数1.1时0.25rad→0.275rad、5rad/s→5.5rad/s及硬裁剪；重复注册不重复乘 dt。
- 8环境27步随机化短测通过数据与接口断言：物性实际回读、质量／惯量同比缩放、局部执行系数重采、物性和 Torch RNG 恢复、时钟 setter 无物理副作用、同一时钟 push 只触发一次、末端力标签仍为零、timeout 和局部 episode 清理。原始报告见开发根 `.scratch/unifp-v3-stage1-reproduction/knowledge/a08_a09_runtime_20261031.json`。
- 32环境25步名义资产／环境合同短测通过，非 timeout 终止为0；该工具显式关闭新增物性／执行系数随机化和 push，以保持名义测试含义。
- Python 语法、函数类型标注／docstring 静态核查及 `git diff --check`通过；官方 `UniFP/` 工作树保持干净。

## 观察与证据边界

8环境未训练零动作短测在第23步的 env3、第24步的 env2 产生 base 接触物理终止；真实失败记录已保存，未删除、跳过或改写为成功。短测中所有 reward 有限，观测到的 reward 范围约 `[-0.0294281,0.1223764]`。本次通过的是奖励／随机化／统计数据链断言，不代表候选参数的动态稳定性或跟踪效果通过验收；接触来源及候选适用性留 A11 整体验证继续核查。

排查中发现短测工具曾设置 `contact_offset=0.001m`，最终改为读取活动配置的 `0.01m` 后重跑。没有修改活动配置、接触终止阈值或奖励来规避失败。

timeout 测试通过预置终止判定计数触发边界，记录仍使用真实27次采样的0.54s，不声称完成真实20s episode。上述 Gym 短测使用 GPU PhysX 与 CPU 张量 pipeline；本机旧 Torch 对该 GPU 的 CUDA kernel 支持有限，因此本结果不覆盖目标环境数下 GPU tensor pipeline 的显存／吞吐验收。`tavis`只运行纯 Torch／文件测试，不是项目仿真训练环境。

票据04交付完成；长训配置、A阶段复现效果、恢复后学习和远程容量尚未验收，未启动任何训练。
