# V3桌面基准资产修复与仿真验收记录

日期：2026-10-31（Asia/Shanghai）  
项目阶段：仿真复现阶段A；资产验收，不是训练实验  
结论：桌面基准候选通过本轮资产与Gym短测，可供A阶段环境继续开发。未启动训练，也不据此声称论文效果已复现。

## 1. 实验目的

建立一份以桌面压缩包外观和零位装配为基准、可在Isaac Gym Preview4导入的V3仿真资产；核对静态结构、名义物性、轮系响应、局部自碰撞规则和当前V3环境的多环境reset。旧V3资产只作历史参考，不作为新资产源。

## 2. 资产身份与保存

桌面源压缩包为`/home/sorry/桌面/a1arm_description.zip`，SHA256：

`9638c2794cfa6874b57a8def527899b0352a29997e203b3a3b51dbc0a38346d8`

桌面URDF成员为`robot_description/a1arm_description/urdf/a1arm_rgm3.urdf`，SHA256：

`a3eacf4b991ba7bfa4be5e57e229adeb918c71a741f17be53d4f4ba799471f2c`

当前活动资产：`UniFP_v3/resources/robots/v3_a1_desktop/`。派生URDF为`urdf/a1arm_rgm3_desktop_fixed_gripper.urdf`，SHA256：

`1dc2f47cf62baf707d9f1e7595b9df6b4455193feabedbd0b46e940ddffb9c0d`

完整独立归档：`UniFP_v3/docs/experiments/archives/v3_a1_desktop_accepted_20261031.tar.gz`，SHA256：

`2eff71f2c0fd486edb82c408967319ab85de69bd3634e21c715e3eefe0a8e75f`

归档内部包含完整`v3_a1_desktop/`目录、URDF引用网格和`asset_manifest.json`；运行资产仍留在resources目录，V3配置指向该活动目录。归档可独立校验和提取，不依赖scratch诊断文件。

## 3. 旧V3资产问题记录

旧派生资产位于`UniFP_v3/resources/robots/v3_a1/`，URDF SHA256为`03db4e846dd77b51f1d7bb62a373a5fa4e3cd16535a21ae6a1068c7e24dd0239`。它从旧组合模型派生，只将两条夹爪平移关节固定，仍保留旧模型的视觉、碰撞几何和安装变换；因此其外观和轮系零位没有以桌面模型为基准，不能用作本项目的新资产基线。

旧URDF轮碰撞圆柱名义宽度为0.08 m，而桌面源轮胎视觉宽度约0.14 m；旧V3轮心／安装坐标也不能替代从桌面轮组几何恢复的中心和关节轴。旧资产在Preview4自碰撞启用时曾出现约10.59 kN的base净接触力及多处部件正重叠。将actor级collision filter设为1的诊断可消除该次base接触异常，但会关闭actor全部自碰撞，违反“保留非相邻自碰撞、只处理有证据的局部装配接触”的已确认边界，因此不是修复方案。

`UniFP_v3/resources/robots/v3_a1/asset_manifest.json`已标记为历史参考资产并注明以上问题；旧URDF和网格未删除，供之后逐项比较。运行时专用的EFFORT/stiffness修正属于V3控制初始化问题，不应误记成URDF本体参数。

## 4. 新资产处理

- 桌面模型保留外观、上装几何、非轮上装物性和零位装配；四轮从桌面视觉几何恢复轮心、转向中心及运动归属，不使用旧V3配准平移。
- 采用用户确认的估算名义物性：底盘含轮组75 kg、整机约83.317915 kg；base_link质量65.8456 kg、COM `(0,0,-0.095) m`，base惯量对角约`(1.68784221,3.1605888,4.21631325) kg·m²`。轮和转向link沿用已确认的V3估算。以上均非称重或实机惯性标定结果。
- 资产为自由root、4个steering、4个continuous drive、6个机械臂关节，共14个驱动DOF；两指改为固定关节。
- 碰撞网格先按完整CAD连通组件分组，再作为独立凸包输入。来源CAD中100个零体积连通片、共214个三角面无法形成有效三维凸包，按link记录后省略；视觉网格和有体积碰撞组件保留。URDF目前引用957个网格、Gym导入碰撞shape合计927个。
- 只对各轮固定安装组件与滚动件、闭合固定两指之间的局部装配接触应用shape过滤；没有全局关闭actor自碰撞。
- V3 `_process_dof_props`明确设`EFFORT`并清零Gym导入伺服stiffness；保留原显式PD力矩、增益、限位和裁剪。证据显示连续轮导入的stiffness近似float最大值，清零后轮关节恢复响应。该修正仅在UniFP_v3的V3实现中生效。

## 5. 验收方法与结果

运行时使用Isaac Gym Preview4专用Python 3.8/Torch 2.3.1环境、GPU PhysX、关闭GPU tensor pipeline。静态资产单测命令：

```bash
python3 -m unittest discover -s UniFP_v3/tests -p test_desktop_asset.py -v
```

5项通过：14DOF和总质量；非轮视觉／安装与上装物性；轮组零位几何双向顶点偏差不超过`1e-5 m`且面数完整；惯性正定与源哈希；URDF引用碰撞mesh至少4个唯一顶点且顶点张成三维。

Gym短测的完整JSON结果保存在`.scratch/v3-desktop-model-repair/knowledge/`，关键结果如下：

| 测试 | 结果 |
|---|---|
| 无重力悬空静置5秒 | 状态有限，base净接触力峰值0 N，DOF无漂移 |
| 接地静置5秒 | base最终高度约0.331 m，无base接触异常终止；轮地初始冲量记录在轮体净力中 |
| 接地小动作5秒 | 四驱动轮角位移约0.122、0.667、2.466、0.142 rad；base最终高度约0.330 m，base净接触力峰值0 N，状态有限 |
| 非相邻自碰撞正对照 | 折叠臂下`arm_seg6`—`base_link`发生接触，证明全身自碰撞仍开启 |
| 实际V3 4环境20策略步 | 无非超时终止，base接触峰值0 N；单环境reset未改变其余环境root、DOF或episode长度 |
| Gym物性回读 | actor总质量83.3179115 kg；base惯量对角`(1.68784225,3.16058874,4.21631336)` kg·m² |

实际V3运行的四轮驱动角均有约0.19—0.21 rad变化。此前固定base单体诊断曾见前右轮局部接触阻滞，但自由base接地动作和实际4环境运行均有四驱动轮响应；固定base诊断不等同于训练root约束。接地动作中轮体记录到较大的初始接触冲量，底盘净接触峰值仍为0；该短测仅证明模型能运行、轮系有响应，不是移动跟踪性能或力控制验收。

## 6. 清理记录

保留本实验记录、最终资产归档、活动资产及最终验收报告。已删除本轮中间Gym探针JSON、失败候选碰撞资产和V3运行生成的`__pycache__`/`.pyc`缓存；旧V3资产、官方代码副本、桌面源ZIP、最终JSON证据未删除。桌面资产修复ticket的逐项讨论及验收摘要仍在`.scratch/v3-desktop-model-repair/`。

## 7. 后续开发边界

资产验收解除A阶段环境票据02的“候选资产/局部碰撞规则”前置，但票据02还需完成目标轨迹与保守可达范围筛查、底盘速度命令生命周期、reset后TCP/FK运动学刷新及所确认的终止信息交接。奖励、随机化与episode度量属于票据04；Actor/Critic及辅助估计属于票据03。依赖未完成前不启动学习；A长训仍受票据07门控和用户已确认的启动流程约束。
