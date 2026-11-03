# V3接入及局部reset验收

状态：完成。

Blocked by：01、02

## 实现范围与接口

在v3_pos_force_config.py选择验收后的新资产与几何推得的初始化高度；如有确认后的物性设置改动，仅限V3路径。新增4环境step/reset短测并输出资产manifest与中文报告。

## 测试与验收

静置/轮系小动作与实际V3 actor物性回读；4环境至少20策略步、一个env局部reset，未reset环境不被误重置；不启动训练。

验收结果：`knowledge/v3_runtime_check.json`通过。4环境运行20策略步，无非超时终止，base接触峰值0 N；对env1局部reset后其他环境的root、DOF和episode长度保持不变。actor总质量83.3179115 kg，base惯量对角与批准估算相符；四个驱动轮在四环境运行中均产生约0.19—0.21 rad角位移。未启动训练。

新增或修改函数完整类型标注和英文Google docstring；只改本任务相关代码。
