# V3 接触终止诊断（2026-10-31）

## 运行条件

- Isaac Gym Preview4、Python 3.8、Torch 2.3.1。
- RTX 5060 上GPU PhysX、CPU状态张量、关闭GPU pipeline。
- 当前开发副本V3 URDF、名义关节姿态、base root高度0.32m。

## 观测事实

- `base_link`按actor刚体名称查得索引0；Gym返回28个刚体，净接触张量reshape为`[1,28,3]`，shape数量一致，未发现该索引错位证据。
- 自碰撞启用时（actor collision filter=0），一次无控制仿真后`base_link`净接触力约10.59kN，姿态仍接近水平；同一帧`aluminum_profiles`、arm段及轮体也出现数千牛净接触力。
- `get_env_rigid_contacts()`报告69个接触候选，其中有正`initialOverlap`的部件对包括：`wrist_camera_connector`—`gripper2`（约16.1mm）、`wrist_camera_connector`—`gripper1`（约9.3mm）、`rear_top_camera`—`aluminum_profiles`（约7.3mm）、`wrist_camera_connector`—`arm_seg5`（约7.7mm）、`arm_seg5`—`arm_seg3`（约14.1mm）、`arm_seg2`—`arm_base`（约15.8mm）、轮体之间（约4.9mm）。API报告的接触列表也含负overlap和零接触冲量候选；不能把列表中每一项都解释成有效碰撞或从body净力反推单一力对。
- Preview4官方文档说明`create_actor(... collision_filter=0)`允许同actor所有shape自碰撞，任何大于0的值会关闭该actor全部自碰撞。将actor filter设为1作诊断时，base接触力为0，轮地仍有约200N净支撑力；四环境20个连续小动作策略步均未done，episode length递增至20。
- 故此，终止触发与actor级自碰撞响应高度相关；当前数据无法证明唯一的致力碰撞对。全局关闭自碰撞只能作为因果诊断，不能作为最终修复，因为已确认A03要求保留非相邻部件自碰撞并只排除有证据的装配重叠对。

## 未解决项

- Preview4 URDF导入后的shape-index范围与每个link碰撞shape对应关系尚未审计。
- 上述正重叠对尚未逐一与URDF装配关系／网格坐标核验；特别是arm_seg3—arm_seg5属于非相邻臂段，不可未经验证直接豁免。
- 需设计按shape filter位掩码选择性排除合法装配重叠，并保持其它非相邻自碰撞；若现有API／资产不能表达确认规则，则先回写设计并请用户确认替代策略。
- 最终修复后要重跑静置、非零动作、终止条件及活动环境局部reset；临时global-filter=1的20步结果仅为诊断对照。
