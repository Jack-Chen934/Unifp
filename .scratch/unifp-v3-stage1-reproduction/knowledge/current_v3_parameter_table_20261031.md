# 当前V3资产与控制参数事实表

核查日期：2026-10-31。本表只汇编现有文件数值，不代表已验证的硬件真值或新显式执行器的批准配置；设计效力见议题03。未导入旧运行模块、未修改资产或运行仿真。

## 文件与模型

- 当前默认资产：`legacy/v3_reference_20261030/robot_description/a1arm_description/urdf/a1arm_rgm3_original_movable.urdf`。
- SHA-256：`49b2e1b1acf728bdf1ec403126a219f7f62d7e5bbd0912c6b81af745fda498df`。
- 模型质量求和：83.317915372851kg；28个link、27个joint、16个可动关节，其中2个为夹爪平移关节。
- 逐刚体惯性、轴、限值及变体对照已保存在[关节与执行器原始记录](asset_actuator_inventory_20261030.json)；基本静态检查见[检查记录](current_v3_asset_check_20261031.json)。

## 受控旋转关节的现有URDF字段

以下顺序对应已讨论的14维策略顺序；源文件后轮名称为bl／br，对应语义rl／rr，名称转换不意味着改变原始轴向。连续轮角无位置上下限。速度字段是文件声明，不证明PhysX实际进行了硬限速。

| 策略索引 | 现有关节名 | 类型 | 局部轴 | 角度范围rad | 速度rad/s | 力矩N·m | URDF行号 |
|---|---|---|---|---|---|---|---|
| 0 | `fl_steering_joint` | revolute | 0 0 1 | [-1.5708, 1.5708] | 2.0 | 35.0 | 652 |
| 1 | `fr_steering_joint` | revolute | 0 0 1 | [-1.5708, 1.5708] | 2.0 | 35.0 | 700 |
| 2 | `bl_steering_joint` | revolute | 0 0 1 | [-1.5708, 1.5708] | 2.0 | 35.0 | 556 |
| 3 | `br_steering_joint` | revolute | 0 0 1 | [-1.5708, 1.5708] | 2.0 | 35.0 | 604 |
| 4 | `fl_wheel_drive_joint` | continuous | 0 1 0 | 连续无界 | 20.0 | 80.0 | 660 |
| 5 | `fr_wheel_drive_joint` | continuous | 0 1 0 | 连续无界 | 20.0 | 80.0 | 708 |
| 6 | `bl_wheel_drive_joint` | continuous | 0 1 0 | 连续无界 | 20.0 | 80.0 | 564 |
| 7 | `br_wheel_drive_joint` | continuous | 0 1 0 | 连续无界 | 20.0 | 80.0 | 612 |
| 8 | `arm_joint1` | revolute | 0 0 1 | [-2.8798, 2.8798] | 20.944 | 40 | 47 |
| 9 | `arm_joint2` | revolute | 0 0 1 | [0, 3.1415] | 20.944 | 40 | 76 |
| 10 | `arm_joint3` | revolute | 0 0 1 | [-3.3161, 0] | 7.5398 | 27 | 105 |
| 11 | `arm_joint4` | revolute | 0 0 1 | [-2.8798, 2.8798] | 25.133 | 7 | 134 |
| 12 | `arm_joint5` | revolute | 0 0 1 | [-1.6581, 1.6581] | 25.133 | 7 | 163 |
| 13 | `arm_joint6` | revolute | 0 0 1 | [-2.8798, 2.8798] | 25.133 | 7 | 192 |

## 底盘、安装与末端字段

| 项目 | 文件值 | 来源位置 |
|---|---|---|
| `fl_steering_joint` | parent=base_link；child=fl_steering_link；xyz=(0.247 0.185 -0.1)m；rpy=(0 0 0)rad | URDF:652 |
| `fr_steering_joint` | parent=base_link；child=fr_steering_link；xyz=(0.247 -0.185 -0.1)m；rpy=(0 0 0)rad | URDF:700 |
| `bl_steering_joint` | parent=base_link；child=bl_steering_link；xyz=(-0.247 0.185 -0.1)m；rpy=(0 0 0)rad | URDF:556 |
| `br_steering_joint` | parent=base_link；child=br_steering_link；xyz=(-0.247 -0.185 -0.1)m；rpy=(0 0 0)rad | URDF:604 |
| `aluminum_profiles_joint` | parent=base_link；child=aluminum_profiles；xyz=(0 0 -0.33)m；rpy=(0 0 0)rad | URDF:493 |
| `connect_joint` | parent=aluminum_profiles；child=arm_base；xyz=(0.174 0 0.42535)m；rpy=(0 0 0)rad | URDF:264 |
| `end_effector_joint` | parent=arm_seg6；child=end_effector_link；xyz=(0 0 0)m；rpy=(0 0 0)rad | URDF:259 |
| fl轮碰撞圆柱 | 半径0.1m；轴向宽度0.08m | `fl_Link/collision/geometry/cylinder` |
| fr轮碰撞圆柱 | 半径0.1m；轴向宽度0.08m | `fr_Link/collision/geometry/cylinder` |
| bl轮碰撞圆柱 | 半径0.1m；轴向宽度0.08m | `bl_Link/collision/geometry/cylinder` |
| br轮碰撞圆柱 | 半径0.1m；轴向宽度0.08m | `br_Link/collision/geometry/cylinder` |
| 底盘刚体 | 质量65.845600kg；局部COM=(0 0 -0.095)m | URDF:499—503 |
| 底盘碰撞盒 | 尺寸0.70×0.46×0.24m，局部中心(0,0,-0.095)m | URDF:505—509 |
| 夹爪1／2 | 平移范围[0,0.03]m，速度0.25m/s，effort=200N；源模型中仍为可动关节 | URDF:221—256 |
| 末端参考点 | end_effector_link相对arm_seg6为零变换，不能据此证明该原点就是接触表面的TCP | URDF:258—263 |

由现有四转向安装点得到：轴距0.494m、轮距0.370m；轮心相对转向关节还沿局部z偏移-0.12m。以轮碰撞半径0.1m和零转向姿态计算，理想水平地面的base_link原点高度为0.32m；这与旧ROOT_HEIGHT一致，但不代表已实测初始穿透／接地。

## 旧控制文件中的参考值

来源：`legacy/v3_reference_20261030/ranger_unifp_vec_env.py`。这些值来自旧控制配置，不属于仅确认“沿用资产”就批准的新显式控制参数。

| 参数 | 旧文件值 | 源码行号 |
|---|---|---|
| 臂默认角 | [0,1,-1.5,0,0,0]rad | 37、57—59 |
| 转向／轮初始角 | 0rad | 57—59 |
| 夹爪初始位移 | 0m | 57—59 |
| 转向／臂动作尺度 | 0.25rad／单位动作 | 100—104 |
| 轮驱动尺度 | 5rad/s／单位动作 | 105—108 |
| 转向隐式驱动 | kp=300N·m/rad、kd=20N·m·s/rad、effort=35N·m | 67—68 |
| 轮隐式驱动 | kp=0、kd=20N·m·s/rad、effort=80N·m | 69—70 |
| 臂隐式驱动 | kp=80N·m/rad、kd=4N·m·s/rad | 63—64 |
| 夹爪隐式驱动 | kp=20N/m、kd=2N·s/m、effort=10N | 65—66 |

夹爪URDF的200N与旧控制配置10N来自不同层级，不是相同数值；夹爪固定后不再为其设置驱动。旧文件109—115表明默认路径把目标交给隐式驱动，返回的显式裁剪PD值不能直接称为其实际驱动力矩。
