# B2+Z1 复现实验证据

核查日期：2026-10-30。本文件只记录本地实验事实、统计口径及现有材料的边界；迁移建议另见设计与总体报告。

## 1. 数据与统计复算

本次读取三个已有 NPZ，用 float64 重新计算逐轴绝对误差、欧氏距离均值和三轴同时小于 0.1 m 的比例；没有启动新训练或推理。结果、文件路径和 SHA256 见同目录 `reproduction_validation.json`。

| 模型累计轮次 | 无力位置平均 L2 | X/Y/Z MAE | 三轴同时 <10 cm |
|---|---:|---|---:|
| 8000 | 5.0476 cm | 2.6603 / 2.6204 / 2.3069 cm | 95.1646% |
| 12000 | 6.8657 cm | 3.4615 / 3.4475 / 3.4135 cm | 89.8979% |
| 16000 | 6.6317 cm | 3.3532 / 3.3344 / 3.2612 cm | 92.4917% |

三组误差数组均为 `[6000,8,3]`，元素有限。12000 与 16000 的实测外力、命令力数组最大绝对值均为 0。float64 复算与历史 float32 汇总存在远小于报告精度的舍入差异。

## 2. 评估接口与误差定义

源码：`literature/projects/unifiedforce/reproduction/evaluate_stage2.py`。

- 第 25–39 行：8 环境、seed1、5×5 地形；禁用观测噪声以及列出的质量、摩擦、电机、COM 随机化和 push；`no_force` 将力课程阈值设为很大的数，`sampled_force` 设为 0。
- 第 48–57 行：在 reward 回调、reset 之前采样；补偿目标为 `p_goal_world + (F_external_world + R_yaw F_command_yaw)/K`。
- 第 62–65 行：估计误差来自解码器预测与环境监督标签之差，再除以力缩放系数；记录的是力估计误差，不是接触物体的闭环力误差。
- 第 74–97 行：输出 L2、逐轴 MAE、P95、实际力/命令力峰值、非超时 reset；保存每环境每步原始数据。

`total8000/no_force.json` 和 `sampled_force.json` 都明确说明：受力评估中的命令力与外力独立采样，不是论文的匹配力评估。

## 3. 受力阶段的实际结果

来源：`literature/projects/unifiedforce/reproduction/outputs/stage2_20261029/followup4000/analysis4000/summary.json`、`total8000/no_force.json`、`total8000/sampled_force.json`、`official_comparison/summary.json`。

| 指标 | 受力训练 4000 轮 | 受力训练 8000 轮 |
|---|---:|---:|
| 随机受力补偿目标平均 L2 | 11.1726 cm | 11.2453 cm |
| 随机受力补偿目标 P95 | 27.2361 cm | 27.3946 cm |
| 随机受力三轴同时 <10 cm | 67.6042% | 66.8938% |
| 随机受力估力 XYZ MAE | 6.6930 / 5.8394 / 6.5089 N | 6.0864 / 4.8267 / 5.4815 N |
| 无力估力 XYZ MAE | 3.6072 / 3.8875 / 3.4913 N | 2.1281 / 1.8674 / 1.6036 N |

最终受力补偿目标 L2 为 11.2453 cm，未补偿原始位置命令 L2 为 19.4052 cm；二者不是同一个目标。最终无力与受力评估的非超时 reset 均为 0，仅限这次评估设置。

## 4. 奖励曲线与课程恢复

来源：`official_comparison/data/all_training_rewards_16000.csv`、`official_comparison/summary.json`。

| 同名 TensorBoard 标签 | Stage1 末 1000 轮 | 累计 16000 末 1000 轮 |
|---|---:|---:|
| `Train/mean_reward` | 138.2473 | 132.6201 |
| `Episode/rew_tracking_lin_vel_force_world` | 1.8481 | 1.8365 |
| `Episode/rew_tracking_ee_force_world` | 1.6184 | 1.4404 |

本实验共无力 8000 + 受力 8000 = 16000 轮；受力由三个续训进程完成，9000/12000 附近会重建仿真和回报缓存。

源码 `literature/projects/unifiedforce/reproduction/train_stage2.py:153–201` 在检查点中补存辅助优化器、学习率、课程计数等，并显式恢复 `env.global_steps`；此脚本检查官方 force gate 与 24 步 rollout 的对应。它不是未改动的官方入口。

## 5. 官方对照的已知边界

来源：`official_comparison/paper_check.md:13–76`；论文一手缓存 `literature/projects/unifiedforce/assets/paper.html` / `paper.txt`，论文标识 `arXiv:2505.20829v2`。

- Fig. A.7 是奖励训练曲线，图末约 37000–38000 轮是图像目测，不是作者原始标量；当前 16000 轮的训练预算不同。
- Fig. 3(b) 是无力逐轴位置跟踪误差分布；正文描述多数落在 0.1 m 内，没有公开可精确对齐的逐轴 MAE、分箱、trial/seed/环境数。
- 当前最终模型各轴 <10 cm 的比例为 96.55% / 97.05% / 97.67%；这与“三轴同时”92.49% 是两种统计。
- Fig. 3(a) 是位置估计误差，Fig. 3(c) 涉及匹配力条件，Fig. 3(d) 是实机接触力试验；当前零力估力 MAE 不等于 Fig. 3(d) 的结果。
- 本次核查的一手公开材料没有作者的原始标量或误差数组，不能计算与作者间的显著性差异。

## 6. 样本预算与资产版本

B2+Z1 此次训练为 4096 环境 × 24 步/轮；V3 冻结记录为 512 环境 × 24 步/轮。若二者均训 16000 轮，分别为 1572864000 与 196608000 条环境 transition，相差 8 倍。相同轮数并不代表相同样本预算，也不能由样本数直接推导相同收敛程度。

当前本地官方目录 `unifp_reproduction/UniFP` 的 Git HEAD 可读取，值为 `68847a070f88d731058c3d8476929bc3b205f5bd`，本次读取的 `git status --short` 为空。远程训练归档是否存在完整 Git 元数据是另一件事，不能由本地 HEAD 自动推定。
