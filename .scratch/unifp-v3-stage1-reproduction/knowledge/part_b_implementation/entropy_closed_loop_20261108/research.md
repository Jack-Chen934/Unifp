# 当前源码依据

- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_on_policy_runner.py:374`严格按传入身份恢复模型／双Adam／实际lr／环境／RNG，再reset一次；`:99`写元数据，`:137`写迭代日志，`:155`按save_interval或终点保存；`:334`存档保存identity、完整配置和lineage。
- `v3_run_identity.py:120`指纹覆盖legged_gym全部Python与活动资产，学习配置排除预算、存档频率、TensorBoard等输出参数；runtime包含CUDA_VISIBLE_DEVICES。独立外部启动脚本不在源码manifest内，必须另记SHA。
- `v3_ppo.py:193`原更新在每个batch前向后计算KL并调整lr；主Adam后重新前向辅助，std保持可学习。`v3_rollout_storage.py:157`保留11字段batch接口。
- `legged_robot_v3.py:1550`附近原控制器返回已限幅力矩；位置／轮速目标在前面裁剪。读取返回力矩不改变执行。`get_training_state`不保存新增诊断计数器。
- `scripts/evaluate_v3.py:100`以checkpoint自己的配置与runtime重建身份；`v3_evaluation.py:541`接受任意完整B边界，不强制12000或正式目的。配置中熵变化不改变评估场景；manifest阈值不依赖训练熵。
- `tools/train_v3_b_observed.py:20`构造冻结观察命令，R2应使用A参考归档内的原manifest；现有源manifest只有观察日程与之不同。
- 实时服务器gpuf3，物理GPU3为RTX A6000、占1780MiB且利用率0%；正式GPU1 PID1981990到12635轮。9000父SHA为d0c64674c852300776662b389e14047e998f8444486f74db5fabea684f089b1f。远程专用Python有Torch和Gym、无pytest。
- 首次原生启动在资产清单读取时因ASCII locale触发UnicodeDecodeError，尚未创建Runner／采集训练；现有资产清单含中文且原读取使用默认编码。独立启动需传入C.UTF-8／PYTHONUTF8=1，业务源码和资产无需变动；首份console日志保留。
- 2026-11-08 21:15父接触评估在归档发布阶段退出：`evaluate_v3_contact.py:198`的shutil.move跨越/tmp与数据盘，rename先报EXDEV，fallback复制后删除源只读plots触发PermissionError。该次contact子场景已产生summary与归档，zero_control未执行，顶层summary缺失；两组200轮尚未启动。将诊断进程TMPDIR置于同一数据盘可使目录rename无需跨盘复制／删除；不改变物理／训练参数或业务源码。
