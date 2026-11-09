# B-T06 · 固定墙接触评估与全局actor映射

状态：已完成；73项V3测试与本机完整脚本短测通过，见文末完成记录。

## 目标与范围

实现B08固定墙零更新场景及B10接触协议，真实测反力，正确处理额外actor索引与局部reset；训练环境不创建墙，不加入专用力控器或测力反馈。

## 开始时必读

B08-Q3a／Q3b／Q3c全部、B10-Q3、B11-U7／G3；B-T05评估归档和无更新接口。

## 代码位置与接口

相对于`UniFP_v3/`：新`legged_gym/envs/v3/v3_contact_evaluation.py::V3ContactEvaluation(V3PosForce)`、`scripts/evaluate_v3_contact.py`和`configs/v3_b_contact_v1.json`。纯统计／归档复用`v3_evaluation.py`。

仅从`legged_robot_v3.py`提取已确认五个窄接缝：`_bind_simulation_tensors()`、`_bind_robot_rigid_state()`、`_sync_robot_state_tensors()`、`_commit_robot_reset(env_ids)`、`_after_physics_substep(physics_index)`；默认无墙路径保持既有逻辑或no-op。具体位置按当前源码映射，不重构共享B2／base环境。

- 全局root[2N,13]、rigid[N(B_robot+1),13]，DOF仍14N；DOMAIN_SIM查询机器人／墙索引，不假设actor_id=env_id。逻辑缓存高级索引后必须显式scatter到完整raw root，setter用实际robot actor ids，wall不随局部reset变化。
- 8个2.5ms物理子步采[N,8,3]墙／指力；策略区间均值用于命令跟踪，末子步反力配下次动作前decoder，reset首次valid=False。不额外simulate、不在子步写文件。
- 世界+X平面box(0.05,0.40,0.40)m，参考q=(0,1.4,−1,0,0,0)，按B08已确认指尖外缘定位；参考root0.32m、真实reset0.34m，不能按checkpoint挪墙。自由底盘零速度命令、名义物性／执行系数、人工力／push关闭，仅两指允许接触。
- 墙反力取−F_wall，与两指和／接触部位核对；不回填Actor／Critic／监督／reward，接触decoder仅泛化诊断。
- 20秒接近／升力／保持／降力／退离脚本沿B10；Ac=min(20N,F_max)，两档各48共96episode，另0N32episode。9—11秒稳态，19—20秒释放；每档成功≥80%、物理失败≤5%，其余误差／有效接触条件原样引用B10，不删除缺测／污染／失败样本。

## 测试与验收

新增`tests/test_v3_contact_evaluation.py`并扩评估统计，先红后绿；用交错全局索引、假原生setter和手算8点数据调用真实窄接口。

- robot映射、完整raw写回、局部reset隔离、wall姿态、采样均值／末次与动作配对正确；固定世界目标在base平移／yaw后表达正确。
- 手算污染／缺测／提前结束样本仍占成功率分母；可测失败轨迹也报误差。
- 同源码无墙A路径回归，不能因接触支持改变原14N布局或控制输入。
- 本机／远程各4环境测量正对照由B-T07汇总；全配额A父基准在B-T08。A策略没有学会接触不判接口失败。

## Blocked by

- B-T05完成。

## 完成记录

待填写：索引／测量／统计验证与审查。后续可执行B-T07。


## 部分实施记录与新增约束

已提取V3内部五个接缝，未改共享base／B2；接触类已实现全局raw与逻辑robot缓存隔离、真实actor索引reset和8点采样。64项相关回归通过，本机无墙4环境×5秒零物理失败，局部reset隔离正常；日志、源码哈希及实际范围见`knowledge/part_b_implementation/bt06/partial_verification.json`。

SDK不允许先创建所有环境再补墙，原草图不能直接执行。B08-Q3d已给出最小修订：接触子类预建wall资产，在现有逐机器人物性回调内创建当前env的墙，最后统一查询DOMAIN_SIM映射。确认前不实施墙创建；本票未完成，后续仍按依赖等待。墙测量／污染分类／协议／原生正对照等尚未验证。


等待B08-Q3d确认期间，已补齐B10范围内三项评估遗漏：保持末0.5秒的估力标记按上一转移配对并在reset清零；底盘人工力槽报零输入偏置／MAE；归档显式记录sim_device、PhysX GPU与GPU pipeline。修正后49项相关回归通过，合成fixture单环境×1000步原生smoke退出0、物理失败0、策略更新0；保持末0.5秒task／estimator各50点，逐行核对上一转移标记一致。新证据为`knowledge/part_b_implementation/bt06/evaluation_followup_verification.json`，不覆盖B-T05旧哈希与归档，也不作为正式模型效果证据。R组两流独立阶段诊断仍待后续审查，不能将单外力阶段称为两流完整分段。

2026-11-07用户已确认B08-Q3d逐环境回调创建墙；按该时序继续B-T06，历史部分验证不等同于整票完成。


## B-T06完成记录（2026-11-07）

Q3d逐环境回调创建已实施。接触子类处理全局actor／body映射、完整raw张量写回、八点采样、保守污染分类、固定世界目标和两档命令；入口`legged_gym/scripts/evaluate_v3_contact.py`只加载策略，独立运行接触与0N对照，复用原step与归档／视频链。接触力不回填训练通道。已规划的trace访问职责直接复用既有`info["policy_trace"]`，不另建同义读取或二次采样接口。

73项V3测试通过；修正SDK单精度dt导致19秒边界漏点后，最终r02完成CPU Torch＋GPU PhysX的1环境×两档各20秒及0N20秒，3000条trace、0物理失败、0策略更新，3个MP4和17张图。每episode稳态／释放为100／50点，首动作前反力无效，后续逐行核验与上一末子步反力一致。4环境创建／reset隔离和主动接触探针另留原始记录。主动穿入峰值约751N只验证墙／指读数，不代表正式按压能力或训练幅值。

证据与源码哈希见`knowledge/part_b_implementation/bt06/verification.json`。合成fixture没有建立接触，不能作为正式A或B效果；正式父仍为V6 A8000。两端完整工程对照、范围冻结及A零力回归留B-T07，完整正式配额B前基准留B-T08。下一票B-T07。
