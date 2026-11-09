# 终止姿态交互可视化

完整分类见[classification.json](classification.json)。HTML包含完整三角面，默认显示视觉，点击图例叠加碰撞代理。PNG为接触附近的局部截面。合力配对仅是候选，几何不是PhysX原生接触对或因果结论。

| 探针／事件 | 候选部件 | 几何类别 | 视觉表面距离 | 查看 |
|---|---|---|---:|---|
| probe_12000_mean／0 | gripper1 | 仅碰撞代理表面相交 | 12.440 mm | [截面](probe_12000_mean_event0.png)／[交互三维](probe_12000_mean_event0.html) |
| probe_12000_mean／2 | wrist_camera | 仅碰撞代理表面相交 | 14.515 mm | [截面](probe_12000_mean_event2.png)／[交互三维](probe_12000_mean_event2.html) |
| probe_12000_mean／5 | arm_seg4 | 仅碰撞代理表面相交 | 2.397 mm | [截面](probe_12000_mean_event5.png)／[交互三维](probe_12000_mean_event5.html) |
| probe_12000_mean／8 | wrist_camera | 终止边界表面分离，子步或余量待定 | 3.831 mm | [截面](probe_12000_mean_event8.png)／[交互三维](probe_12000_mean_event8.html) |
| probe_12000_sample／5 | arm_seg4 | 终止边界表面分离，子步或余量待定 | 0.711 mm | [截面](probe_12000_sample_event5.png)／[交互三维](probe_12000_sample_event5.html) |
| probe_12000_sample／6 | arm_seg6 | 终止边界表面分离，子步或余量待定 | 2.895 mm | [截面](probe_12000_sample_event6.png)／[交互三维](probe_12000_sample_event6.html) |
| probe_12000_sample／8 | arm_seg4 | 机身与部件真实视觉表面相交 | 0.000 mm | [截面](probe_12000_sample_event8.png)／[交互三维](probe_12000_sample_event8.html) |
| probe_12000_sample／12 | wrist_camera_connector | 仅碰撞代理表面相交 | 2.631 mm | [截面](probe_12000_sample_event12.png)／[交互三维](probe_12000_sample_event12.html) |
| probe_12000_sample／14 | gripper2 | 仅碰撞代理表面相交 | 19.143 mm | [截面](probe_12000_sample_event14.png)／[交互三维](probe_12000_sample_event14.html) |
| probe_12000_sample／20 | wrist_camera_connector | 终止边界表面分离，子步或余量待定 | 7.090 mm | [截面](probe_12000_sample_event20.png)／[交互三维](probe_12000_sample_event20.html) |
