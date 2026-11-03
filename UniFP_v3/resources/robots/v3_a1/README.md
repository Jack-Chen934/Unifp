# V3/A1 机器人资产

本目录是 A 阶段 V3 资产的运行时路径。`urdf/a1arm_rgm3_original_movable.urdf` 是独立保存的原始URDF；`urdf/a1arm_rgm3_v3_fixed_gripper.urdf` 仅将两指关节固定在零位，所需网格已按原相对路径复制。`asset_manifest.json` 记录来源与SHA-256。本项目运行时不得从 `legacy/` 读取资产。

静态文件与拓扑检查已执行；Isaac Gym导入、属性回读、接触及动态稳定性尚未验证。
