# V3 A8000评估与证据归档实现记录

日期：2026-11-01。实现票据06，未运行Isaac Gym仿真、未启动训练、未评价策略效果。

## 交付内容

- `legged_gym/scripts/evaluate_v3.py`将配置回显入口改为A阶段确定性评估入口。必须显式提供A8000 checkpoint、JSON manifest与新归档目录；校验完整双Adam/RNG boundary、A8000计数和重算的源/资产/schema/学习配置身份。只加载Actor权重，不创建PPO或optimizer，推理使用Actor动作均值。
- manifest独立指定评估任务seed及push seed、并行环境数和每环境episode数。评估任务随机性使用seed初始化；push使用独立Torch Generator。独立评估时钟从A8000边界继续周期push，不改动训练迭代/step/transition计数；不存在Runner和优化器更新。
- `V3PosForce`新增默认关闭的逐step trace开关；评估开启后，在终态reset之前记录世界坐标TCP目标/实际位置、底盘状态、Actor估计输出和对应输入标签、目标段、力命令/外力、动作、push事件、done原因、reward与奖励分项、A08误差。训练调用保留空trace，不做step级CPU复制。
- `v3_evaluation.py`汇总TCP L1/L2、底盘平面速度和yaw率误差mean/P95、估计器六个速度/TCP维度MAE/RMSE、失败/timeout分母及push后50步恢复窗口。缺A11阈值时结论为`ungraded_or_incomplete`；有阈值时强制覆盖A10要求的跟踪、失败、估计器与push恢复类别并指定版本。
- 用户确认push恢复候选窗口为事件后50个策略step，即1秒（20ms/step）；归档仍保留完整step trace，A11再确认量化门槛。
- 归档精确checkpoint、manifest、policy trace、episode记录、训练迭代日志、有效评估配置、summary及图表；清单记录逐文件SHA256、源身份、Git commit/status和diff哈希，归档文件/目录设置只读权限。曲线提供reward rate和TCP L2按迭代与累计转移双横轴，另有episode误差和估计器逐维误差分布。
- 论文比较边界明确标注：V3估计误差可按Fig.3(b)逐维分布和单位作有限对照，不补造作者未公开原始点；Fig.3(c)/(d)不混入零力A阶段结果。

## 验证与限制

- `conda run -n tavis python -m pytest -q tests --ignore=tests/test_v3_rewards.py`：24项通过，其中新增评估器单测5项。
- 修改文件通过`python -m py_compile`语法检查，Git diff空白检查通过。
- `tests/test_v3_rewards.py`需Isaac Gym；当前`tavis`无`isaacgym`模块，因此本机未完成全套测试，也未通过远程Gym整体验收。
- 当前没有A8000 checkpoint；A11短测和量化阈值均未确认，故无真实A8000评估数据、论文效果结论或A阶段通过结论。远程启动训练前仍须票据07门控与A11阈值确认。
