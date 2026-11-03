# Gym导入、物性回读与局部碰撞修复

状态：完成。候选资产在Isaac Gym中导入稳定；无重力悬空静置5秒、接地静置5秒及接地小动作5秒均状态有限，底盘接触净力峰值为0；8个轮系关节均有响应，非相邻碰撞正对照有效。碰撞网格的零体积CAD碎片按证据省略，未改变视觉几何。

Blocked by：01

## 实现范围与接口

新增UniFP_v3/tools/probe_desktop_asset.py；真实加载候选并回读body/shape/DOF/mass/COM/inertia，隔离测试接触、轮组动作；只处理有证据的局部碰撞问题，记录规则，不关闭全身自碰撞。

## 测试与验收

5s悬空静置、接地静置与接地小动作均通过；小动作覆盖8个轮系关节。非相邻 `arm_seg6`—`base_link` 接触正对照通过。局部装配过滤仍限于四轮固定安装组件与轮体、固定闭合两指；没有关闭全身自碰撞。实际actor惯量与批准值一致，无需开启额外重算修正。

最终证据：`knowledge/gym_final_air_static.json`、`gym_final_ground_static.json`、`gym_final_ground_move5s.json`、`knowledge/gym_final_nonadjacent.json`。早期固定底座探针属于临时诊断，不作为训练根约束验收；其输出已清理。自由底座接地测试及正式V3运行均观察到四驱动轮响应。

新增或修改函数完整类型标注和英文Google docstring；只改本任务相关代码。
