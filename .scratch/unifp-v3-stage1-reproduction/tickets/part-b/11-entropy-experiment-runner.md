# B-T11：独立探索对照入口

状态：已完成。4项单元测试与函数标注／docstring AST通过；两组4096×10＋跨进程1轮原生恢复及原存档校验通过，证据见knowledge/part_b_implementation/entropy_closed_loop_20261108/native_verification.json。Blocked by：B-T09、已有只读诊断完成；用户已确认单变量草案。

范围：外部诊断脚本的快照／runtime验证、原Runner严格恢复后显式实验身份、完整状态核验、只读全子步与动作／KL观测；原业务源码不改。设计B12／B13。

测试：先写unittest再实现；恢复全部模型／双Adam／实际lr一致，新存档原validator通过。验收：函数标注／docstring、CPU单元、脚本／源码SHA可追溯。
