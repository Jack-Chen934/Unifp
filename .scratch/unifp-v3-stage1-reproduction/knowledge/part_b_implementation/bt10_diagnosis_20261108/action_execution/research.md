# 当前调用链依据

- `UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:668`：step先将pending外力冻结为applied，执行底盘push，按clip_actions裁剪；8个子步依次计算原力矩、提交、推进仿真、刷新关节。
- 同文件`:735`附近：更新root／yaw和applied标签后，按当前区间命令结算奖励并调用_accumulate_episode_metrics；随后才更新命令／力事件、局部reset和发布下一观测。因此预reset钩子可记录真实终止步，且目标与本步奖励口径一致。
- 同文件`:979`／`:1010`：几何目标XY随root平移、yaw旋转，Z名义高度；补偿目标加上yaw命令力与world实际外力之和除以K200。下一步动作前的预目标应使用pending力，不能误用上一周期applied标签。
- 同文件`:1540`附近：动作乘每episode执行系数，位置残差乘0.25并受URDF关节边界限制，轮速目标乘5并限速；位置PD及轮速误差力矩最后受effort限制。该函数只读取状态并返回力矩。
- 同文件`:618`：command／external是两个独立梯形包络事件，以episode local tick推进；command为yaw系任务力，external为world系实际注入力，二者不可混称物理外力。
- 上轮`sequential_checks/check_9000/summary.json`与`check_12000/summary.json`保存独立采样的误差／全子步限幅，TCP 0.1286→0.3284m，joint2限幅0.0983→0.4473，原检查点SHA已核对。
