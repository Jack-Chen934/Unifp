# Ranger 512 固定 COM 受力续训迁移摘要

本文件是清理后的唯一实验报告。可迁移源码、检查点和哈希清单位于 [`ranger_512_migration`](/home/sorry/python_ws/interactive/unifp_reproduction/ranger_512_migration)。详细结构化摘要是 [`SUMMARY.json`](/home/sorry/python_ws/interactive/unifp_reproduction/ranger_512_migration/SUMMARY.json)。

## 训练配置

- 512 个并行环境，每环境 24 步，8 个线程，训练 seed `266281`。
- 固定全身标称 COM，critic COM 偏移为 0；Ranger 14 维动作、原生 PPO 与辅助联合更新保持不变。
- 每轴受力课程范围 ±60 N，受力从 global policy step `192001` 开始。
- 16000 轮完整结束，global policy steps `384000`，PPO 与辅助 Adam 均为 `320000` 步。
- 16000 检查点 SHA256：`498f039122bf33075d4144ebcd7a05adcd91fde9214ac761aed9def79ebabb25`。

## 验证覆盖

受力验证使用 seed `266391`、`266392`，分别覆盖 `mean`/`sample` 动作、全零力/受力四种组合，每个边界为 32 环境 × 1000 步。9000 轮之后的 14 个边界为 112 条验证，加冻结的 8000/9000 共 128 条力协议验证；原始 result、trace、initial-state SHA 和 float64 指标复算全部通过。

## 16000 轮终点

两验证 seed 的算术平均如下：

|模式|受力偏移目标末端误差|受力 XY 误差|受力 yaw 误差|估力 RMSE|自身恒零基线|相对恒零改善|
|---|---:|---:|---:|---:|---:|---:|
|mean|50.348 cm|0.6393 m/s|0.5566 rad/s|27.702 N/轴|17.713 N/轴|-9.989 N/轴|
|sample|45.795 cm|0.5607 m/s|0.6227 rad/s|11.271 N/轴|17.846 N/轴|+6.575 N/轴|

终点前 100 轮训练窗口的每步奖励为 `0.009725`，完整 episode return 为 `9.310`，几何末端误差为 `46.766 cm`，XY 误差为 `0.5520 m/s`，yaw 误差为 `0.6376 rad/s`。奖励局部回落不能替代独立验证控制指标。

## 结论

辅助估计器学到了一部分受力信息，尤其是 `sample` 动作下的受力估计；但策略没有稳定地把该信息转换为受力条件下的底盘和末端控制。16000 轮终点不能称为整体恢复。

按明确目标进行验证选点时：`sample` 的受力偏移目标最佳检查点为 10000 轮（两 seed 均值约 `13.956 cm`），受力估力和非零向量误差最佳检查点为 10500 轮；`mean` 的受力偏移目标训练检查点最佳为 9000 轮，受力估力训练检查点最佳为 14000 轮。它们都是验证选择，不是独立测试，也不替代16000轮终点结论。

恢复训练时模型、两 Adam、学习率、计数器和 Torch/NumPy/Python RNG 均实值恢复；PhysX、任务状态、history、episode 统计和力脉冲重新构建，因此不能称物理仿真完全连续，也不能把退化单独归因于暂停。

## 迁移与清理

保留 `UniFP/`、`robot_description/`、根目录 Ranger/UniFP 源码、固定 COM/受力配置和 Isaac Gym Preview4 运行缓存。`ranger_512_migration/logic/` 保存 512 环境训练入口、受力阶段入口、评估器、边界复核器及最终/基线检查点对应计划；`source_manifest.json` 保存 168 项受保护源码的 SHA 清单。

已删除实验轨迹、旧训练日志、评估原始数据、诊断 artifacts、旧 archive 和项目内 `IsaacLab/`。IsaacLab 不属于本固定 COM 512 训练路径；迁移到高显存服务器时仍需在目标机重新安装匹配的 Isaac Gym/Python/CUDA 环境。
