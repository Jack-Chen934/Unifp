# 排查与视频交付

本任务的历史审计、隔离诊断、视频和显示定位已交付。当前结论、可复制命令、资产边界及后续计划统一见[总报告](../../../docs/progress/2026-11-06_v3_status.md)，不在此维护旧的“显示根因未知”状态。

| 任务 | 范围 | 依赖 | 状态 |
|---|---|---|---|
| T01 | 历史episode失败分布与来源审计 | 无 | completed |
| T02 | 冻结检查点隔离诊断及失败前遥测 | T01 | completed |
| T03 | MP4、可运行脚本与中文说明 | T02 | completed |
| T04 | 显示单变量对照、9姿态几何与2次原生pose核验 | T02、T03 | completed |

显示错位源于flip_visual_attachments=True；独立回放`--unflip_visuals`已修正并重录，正式显示配置尚未显式修正。物理接触与显示错位分开记录；原始视频、静态对照、失败pose与JSON全部保留。

[纠正显示的视频](../artifacts/video_s1_unflipped/policy.mp4)属于V6的8000轮mean/S1演示，不能当作r3验收。局部几何修复后续转入[碰撞任务](../../unifp-v3-stage1-reproduction/knowledge/model_history/unifp-v3-collision-proxy-repair/创建与决策记录.md)，该任务的独立r3也尚未证明总体失败改善。
