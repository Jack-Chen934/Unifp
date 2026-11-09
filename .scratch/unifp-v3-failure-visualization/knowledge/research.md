# 源码与历史事实

- `UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py:495—510`：|roll|>0.8rad、|pitch|>1rad或base_link净接触力>1N即terminated，episode结算后局部reset。
- 同文件`:637—683`：episode记录只保留physical_failure合并类型和汇总误差／原始奖励，无终止前姿态与接触力；不能从历史摘要恢复准确分项。
- 同文件`:975—1020`：训练进度为runner维护的24步边界；独立评估可使用evaluation_policy_step时钟。
- `UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_actor_critic.py:156—228`：PPO动作从Normal采样，均值推理为独立接口。
- `UniFP_v3/legged_gym/scripts/evaluate_v3.py:129—165`：冻结权重推理创建新环境，Actor均值，无优化器；正式评估32环境。
- `UniFP_v3/legged_gym/envs/base/base_task.py:31—36`：headless默认将graphics_device_id设为-1，摄像头录制需在独立子类create_sim阶段启用图形设备。
- 历史只读审计实际命令为SSH内python流式读取两个episodes.jsonl并断言无physical_failure；输出AssertionError：1648次，涉及1367环境。单环境最多4次；episode时长中位9.39秒，6次少于1秒。该审计复现的是已记录症状，不是原PhysX轨迹。
- 最终8000轮正式三场景均通过，各96episode零失败；最后训练失败为第3127轮。Gym远程GPU1目前空闲，存在ffmpeg及Xvfb。

## 首个真实GPU复现结果

`early_sample`：1400轮冻结策略，128环境、seed1001、60秒／环境，训练场景、sample动作；结束384个episode中8个失败、376个timeout。观察回调记录数与原infos中的physical_failure数一致，8次均只有base_contact判据为真，没有roll/pitch越界。

失败姿态roll约−0.013至−0.002rad、pitch约0.0077至0.0102rad，root高度约0.3247—0.3261m。base_link净接触力约2.16—227.77N；8次均存在wrist_camera或wrist_camera_connector的同大小净接触力。原遥测只存norm，方向及实际关节位置由后续版本补记，不能仅凭同模长将历史1648次一概归类。

脚本追加观测字段dof_position_rad、接触vector_n及500策略步进度提示；新增--disable_self_collision仅作用于隔离actor的创建标记。原业务create_actor直接读取cfg.asset.self_collisions。该对照不作为正式模型修复，不改变训练快照。

视频已经实际生成并下载：H.264，960×720，50fps，1000帧，20秒，SHA256为b489b6f277ce62e9c6f20fb52f36eaaa3bc0d735768ede2d6fd13cdfe1fd067a；远程完整解码通过，抽取8秒帧。Gym原生画面中部分上装部件看起来分离，此显示现象尚未单独完成几何归因，不能拿视频外观代替物理接触证据。

## 完整对照新增事实

`artifacts/early_mean_summary.json`：1400轮mean、training、128环境、60秒，384次timeout、0失败。`artifacts/early_sample_no_self_collision_summary.json`：关闭actor自碰撞20秒、128环境、sample，128次timeout、0失败；对应基线前1000策略步有2次失败。

`artifacts/early_sample_no_push_summary.json`：1400轮sample、S1关闭push、128环境、60秒，9失败、376次timeout；9次均仅base_contact成立。`early_sample_no_push_failures.json`中8次相机／支架和1次arm_seg4的净接触向量与base_link逐分量反向，相加残差均为0N。roll范围−0.004971至−0.002395rad，pitch范围0.007735至0.009759rad；base接触力5.23至304.12N。

所有对照使用保存物性前128行，新建reset／RNG／PhysX状态。最终模型额外sample运行未执行；已有原训练3501—8000轮442368个结束episode零失败、最终正式mean三场景288episode零失败，不能将其写成新增sample对照。

## 显示与几何后续事实

基类`envs/base/legged_robot_config.py:84`的flip_visual_attachments=True由`base/legged_robot.py:527`传入Gym，V3 asset未覆盖。`artifacts/static_flip_probe/`实际原生摄像头单变量对照中，True发生错位而False恢复装配；两组刚体相对URDF FK位置最大误差0.299／1.165微米。与桌面zip直接比较19个原始视觉mesh引用摘要一致。

`terminal_geometry_audit.json`对9姿态的独立FCL三角表面检查：视觉间距0.660—15.740mm，7个训练代理相交、5个桌面基准collision相交。`native_terminal_geometry_audit.json`使用追加10秒短测终止前实际刚体pose：step158／454视觉间距1.687／6.921mm；训练proxy前者间距0.0445mm、后者相交。实际shape contact_offset=0.01m，rest_offset=0；未做降低接触余量对照。只检测终止边界表面，不证明实体包含或此前子步情况。

`video_s1_unflipped/`为同8000轮权重、S1、mean、20秒新回放，显示False、自碰撞开启，0失败，完整解码与抽帧核验通过；正式配置仍为True，未实施业务修复。

真实step454姿态的混合几何对照：当前底盘proxy＋桌面腕部collision相交；桌面底盘collision＋当前腕部proxy间距2.380mm。说明这个具体案例的形状差异可定位到底盘代理；不是仅由腕部proxy替换造成。此为离线几何对照，不是新动力学因果实验。

## 2026-11-07三资产冻结姿态分类输入

- `unifp-v3-collision-proxy-repair/knowledge/summarize_visual_envelope_comparison.py`的`summarize_run`逐个读取原生终止前`body_pose_world_m_xyzw`，按base逆变换组合到base_link坐标，调用`audit_failure_geometry.py:24,60`的全网格表面检测。`comparison_summary.json`的`runs[].events[].counterpart_geometry`保存同姿态visual／collision的布尔相交、最小间距、具体形状对；目前没有视觉体积IoU或PhysX接触流形。
- 同文件`identify_counterparts`用最多五个最大刚体净力的最小抵消组识别对侧，支持双夹爪组合；不是精确接触点反演。已有71事件全部有对侧几何。按终止边界布尔相交分类，V6为visual相交0／仅proxy相交20／二者均无相交5；r3为1／22／4；V8为2／11／6。此分类描述采样边界，不证明整个物理子步历史。
- 历史子步`contact_substeps/substep_analysis.json`已证明V6 158与r3 766此前发生过代理相交，虽然终止边界分离，因此不得把第三类写成“无接触”或“纯求解器错误”。
- `docs/progress/2026-11-06_v3_status.md:99`记录正式训练1648次失败、最后一次在3127轮；3501—8000轮442368个结束episode零失败。正式8000轮S0/S1/S2的Actor mean共288episode零失败。当前三seed对照的checkpoint为1400轮；二者不能混用。
