# 当前源码依据

- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_actor_critic.py:100`：std是14维直接可学习参数；`:164`附近广播为对角Normal尺度；`:203`起熵对14维求和。Decoder不参与分布尺度计算。
- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_ppo.py:54`：主Adam包含std，辅助Adam仅包含Encoder／Decoder；`:217`起用保存的原始动作计算logp；`:225`附近联合KL自适应主学习率；`:242`主损失为surrogate＋value_coef×value_loss－entropy_coef×entropy；`:246`全主参数一起裁剪，再调用原Adam；`:251`后辅助重新前向、更新。
- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_rollout_storage.py:157`起：24×N扁平化；`:177`一次随机排列在5个epoch中重复，4个mini-batch，共20次主更新。
- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_on_policy_runner.py:374`：same_stage恢复严格校验身份和环境数，恢复模型、双Adam、实际lr、训练进度、环境训练状态与RNG，再重置新episode；不恢复未完episode／PhysX缓存。修改有效配置后不能冒充原same_stage身份。
- 已保存`../sequential_checks/check_{9000,10000,12000}/rollout.pt`，包含24×128条真实观测、旧分布、原始动作、GAE优势与returns。此前采样脚本`../sequential_checks/collect_checks.py:193`附近恢复双Adam；`:271`附近保存rollout，随后只在内存副本执行原更新。
- B05-Q1确认全模块／std继续学习、不重置；B07确认完整学习状态恢复。现有设计没有批准新的std上界或冻结规则。
- 官方有效配置`UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py:315`明确指定`entropy_coef=0.01`。不能拿官方PPO构造函数的默认0.0冒充官方实际运行配置；V3三个存档的有效值同为0.01。
