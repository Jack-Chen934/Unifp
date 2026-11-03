# 桌面基准资产与静态合同

状态：完成。静态回归5项通过；独立候选URDF已生成，源ZIP与旧资产保留。

Blocked by：无

## 实现范围与接口

新增UniFP_v3/tools/build_desktop_asset.py、tests/test_desktop_asset.py及resources/robots/v3_a1_desktop/。从ZIP提取引用网格，保留非轮visual/上装物性，以桌面轮几何重新恢复转向和滚动中心并拆分轮总成；采用已确认V3物性、14DOF及关节初值。旧资产和官方副本只读保留。

## 测试与验收

候选轮视觉面数完整且双向顶点误差≤1e-5m；总质量83.317915kg、14DOF、源哈希不变。额外检查所有URDF引用的碰撞网格均至少有4个唯一顶点并张成三维，避免Isaac Gym凸包烹饪崩溃。

新增或修改函数完整类型标注和英文Google docstring；只改本任务相关代码。
